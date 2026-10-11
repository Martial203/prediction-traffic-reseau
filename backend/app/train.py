"""Entraîne et évalue les 3 modèles (+ 2 baselines) sur le même CSV.

Usage (depuis backend/) :  python -m app.train [--models xgboost lstm ...]

Produit dans artifacts/ :
  - random_forest.joblib, xgboost.joblib, lstm.pt   modèles entraînés
  - predictions.csv                                 prédictions sur la période de test
  - metrics.json                                    métriques globales / par cellule / par événement
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .config import ARTIFACTS_DIR, LOOKBACK, MODELS, TEST_RATIO
from .data import TARGET, build_features, cell_list, load_raw, masks, split_timestamps
from .metrics import regression_metrics
from .models import BASELINES, LSTMModel, make_random_forest, make_xgboost


# Chaque fabrique reçoit le nombre de cellules (utile au LSTM pour son one-hot).
FACTORIES = {
    "random_forest": lambda n_cells: make_random_forest(),
    "xgboost": lambda n_cells: make_xgboost(),
    "lstm": lambda n_cells: LSTMModel(n_cells=n_cells),
    **{k: (lambda n_cells, f=f: f()) for k, f in BASELINES.items()},
}


def main(selected: list[str]):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cells = cell_list(raw)
    feat = build_features(raw, cells)
    valid_start, test_start = split_timestamps(raw)
    train_m, valid_m, test_m = masks(feat, valid_start, test_start)
    print(f"Données : {len(raw)} lignes, {len(cells)} cellules, {raw.timestamp.min()} -> {raw.timestamp.max()}")
    print(f"Train : {train_m.sum()} | Valid : {valid_m.sum()} | Test : {test_m.sum()} (à partir de {test_start})")

    candidates = {k: FACTORIES[k](len(cells)) for k in selected}

    test = feat.loc[test_m, ["timestamp", "cell_id", "event", TARGET]].rename(columns={TARGET: "actual"})
    test = test.reset_index(drop=True)
    report = {"models": {}, "feature_importance": {}}
    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(feat, train_m, valid_m)
        fit_s = time.perf_counter() - t0
        test[name] = model.predict(feat, test_m)
        m = regression_metrics(test["actual"], test[name])
        print(f"  {MODELS[name]['label']:<24} MAE={m['mae']:6.2f}  RMSE={m['rmse']:6.2f}  "
              f"MAPE={m['mape']:5.2f}%  R2={m['r2']:.3f}  ({fit_s:.1f}s)")
        if hasattr(model, "save"):
            model.save(ARTIFACTS_DIR)
        if hasattr(model, "feature_importance"):
            report["feature_importance"][name] = model.feature_importance()
        if name == "lstm":
            report["lstm_history"] = model.history
        report["models"][name] = {
            **MODELS[name],
            "fit_seconds": round(fit_s, 2),
            "metrics": m,
            "by_cell": {c: regression_metrics(g["actual"], g[name]) for c, g in test.groupby("cell_id")},
            "by_event": {e: regression_metrics(g["actual"], g[name]) for e, g in test.groupby("event")},
        }

    ml = [k for k in candidates if MODELS[k]["kind"] == "ml"] or list(candidates)
    report["best_model"] = min(ml, key=lambda k: report["models"][k]["metrics"]["rmse"])
    report["dataset"] = {
        "rows": len(raw),
        "cells": cells,
        "start": raw.timestamp.min().isoformat(),
        "end": raw.timestamp.max().isoformat(),
        "valid_start": valid_start.isoformat(),
        "test_start": test_start.isoformat(),
        "test_ratio": TEST_RATIO,
        "lookback_hours": LOOKBACK,
        "train_rows": int(train_m.sum()),
        "valid_rows": int(valid_m.sum()),
        "test_rows": int(test_m.sum()),
    }
    report["trained_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    test.to_csv(ARTIFACTS_DIR / "predictions.csv", index=False)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Meilleur modèle (RMSE) : {MODELS[report['best_model']]['label']}")
    print(f"Artefacts écrits dans {ARTIFACTS_DIR}")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None)
    selected = parser.parse_args().models
    if selected is None:
        selected = list(MODELS)
        if not _torch_available():
            print("[!] PyTorch n'est pas installé : le LSTM est ignoré (pip install torch puis relancer).")
            selected.remove("lstm")
    main(selected)
