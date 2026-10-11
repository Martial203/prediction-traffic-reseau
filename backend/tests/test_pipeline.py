"""Tests du pipeline de données et de l'API.  Lancement : python -m unittest discover -s tests"""
import json
import unittest

import numpy as np
import pandas as pd

from app import api
from app.config import ARTIFACTS_DIR, LOOKBACK
from app.data import (
    TABULAR_FEATURES,
    TARGET,
    DataError,
    build_features,
    cell_list,
    load_raw,
    masks,
    split_timestamps,
    validate,
)
from app.metrics import regression_metrics


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.cells = cell_list(cls.raw)
        cls.feat = build_features(cls.raw, cls.cells)

    def test_cells_sorted_naturally(self):
        df = self.raw.head(3).assign(cell_id=["Cell_10", "Cell_2", "Cell_1"])
        self.assertEqual(cell_list(df), ["Cell_1", "Cell_2", "Cell_10"])

    def test_validate_rejects_gaps_and_duplicates(self):
        g = self.raw[self.raw["cell_id"] == self.cells[0]]
        with self.assertRaises(DataError):
            validate(g.drop(g.index[10]))  # trou d'une heure
        with self.assertRaises(DataError):
            validate(pd.concat([g, g.head(1)]))  # doublon
        with self.assertRaises(DataError):
            validate(g.drop(columns=["sinr"]))  # colonne manquante

    def test_lags_use_only_the_past_of_the_same_cell(self):
        for cell, g in self.feat.groupby("cell_id"):
            g = g.sort_values("timestamp")
            np.testing.assert_allclose(g["traffic_lag1"].iloc[1:], g[TARGET].iloc[:-1].to_numpy())
            np.testing.assert_allclose(g["traffic_lag24"].iloc[24:], g[TARGET].iloc[:-24].to_numpy())
            # La moyenne glissante de l'heure t ne contient pas t.
            expected = g[TARGET].iloc[:24].mean()
            self.assertAlmostEqual(g["traffic_roll_mean24"].iloc[24], expected)

    def test_no_missing_features_after_lookback(self):
        usable = self.feat[self.feat["pos"] >= LOOKBACK]
        self.assertFalse(usable[TABULAR_FEATURES].isna().any().any())

    def test_split_is_chronological_and_disjoint(self):
        valid_start, test_start = split_timestamps(self.raw)
        train, valid, test = masks(self.feat, valid_start, test_start)
        self.assertFalse((train & valid).any() or (train & test).any() or (valid & test).any())
        self.assertLess(self.feat.loc[train, "timestamp"].max(), self.feat.loc[valid, "timestamp"].min())
        self.assertLess(self.feat.loc[valid, "timestamp"].max(), self.feat.loc[test, "timestamp"].min())


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        m = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual((m["mae"], m["rmse"], m["mape"], m["r2"]), (0, 0, 0, 1))

    def test_known_values(self):
        m = regression_metrics([10, 20], [12, 18])
        self.assertAlmostEqual(m["mae"], 2)
        self.assertAlmostEqual(m["rmse"], 2)
        self.assertAlmostEqual(m["mape"], 15)


@unittest.skipUnless((ARTIFACTS_DIR / "metrics.json").exists(), "modèles non entraînés")
class ApiTests(unittest.TestCase):
    def test_overview_lists_trained_models(self):
        o = api.overview()
        report = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual({m["id"] for m in o["models"]}, set(report["models"]))
        self.assertIn(o["best_model"], report["models"])

    def test_all_models_scored_on_same_rows(self):
        report = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        sizes = {m["metrics"]["n"] for m in report["models"].values()}
        self.assertEqual(len(sizes), 1)

    def test_series_predictions_only_on_test_period(self):
        s = api.cell_series(cell_list(api.raw())[0])
        test_idx = s["timestamps"].index(s["test_start"])
        for values in s["predictions"].values():
            self.assertTrue(all(v is None for v in values[:test_idx]))
            self.assertTrue(all(v is not None for v in values[test_idx:]))

    def test_forecast_shape(self):
        model = api.report()["best_model"]
        f = api.cell_forecast(cell_list(api.raw())[-1], model, 6)
        self.assertEqual(len(f["values"]), 6)
        self.assertTrue(all(lo <= v <= hi for lo, v, hi in zip(f["lower"], f["values"], f["upper"])))

    def test_unknown_cell_is_404(self):
        with self.assertRaises(api.HTTPException) as ctx:
            api.cell_series("cellule-inexistante")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
