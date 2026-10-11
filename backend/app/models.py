"""Les modèles comparés, derrière une interface commune :

    fit(feat, train_mask, valid_mask)
    predict(feat, mask) -> np.ndarray   (une prédiction par ligne de mask)
    save(dir) / load(dir)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import LOOKBACK, SEED
from .data import SEQUENCE_FEATURES, STATIC_FEATURES, TABULAR_FEATURES, TARGET


class TabularModel:
    """Random Forest / XGBoost sur les variables tabulaires (lags, calendrier…)."""

    def __init__(self, name: str, estimator=None):
        self.name = name
        self.estimator = estimator

    def fit(self, feat, train_mask, valid_mask):
        # Les arbres n'ont pas besoin de validation : on entraîne sur train + valid.
        m = train_mask | valid_mask
        self.estimator.fit(feat.loc[m, TABULAR_FEATURES], feat.loc[m, TARGET])
        return self

    def predict(self, feat, mask):
        return self.estimator.predict(feat.loc[mask, TABULAR_FEATURES]).astype(float)

    def feature_importance(self) -> dict[str, float]:
        imp = self.estimator.feature_importances_
        return {f: float(v) for f, v in sorted(zip(TABULAR_FEATURES, imp), key=lambda x: -x[1])}

    def save(self, directory: Path):
        joblib.dump(self.estimator, directory / f"{self.name}.joblib", compress=3)

    @classmethod
    def load(cls, name: str, directory: Path):
        return cls(name, joblib.load(directory / f"{name}.joblib"))


def make_random_forest():
    from sklearn.ensemble import RandomForestRegressor

    return TabularModel(
        "random_forest",
        RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    )


def make_xgboost():
    from xgboost import XGBRegressor

    return TabularModel(
        "xgboost",
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
        ),
    )


class BaselineModel:
    """Règles naïves servant de référence : un modèle ML doit les battre."""

    def __init__(self, name: str, column: str):
        self.name = name
        self.column = column

    def fit(self, feat, train_mask, valid_mask):
        return self

    def predict(self, feat, mask):
        return feat.loc[mask, self.column].to_numpy(dtype=float)


class LSTMModel:
    """LSTM multivarié (PyTorch) : fenêtre des 24 dernières heures de la cellule
    + calendrier de l'heure cible + identifiant de cellule (one-hot)."""

    name = "lstm"

    def __init__(self, n_cells: int, hidden: int = 64, epochs: int = 150, patience: int = 15):
        self.n_cells = n_cells
        self.hidden = hidden
        self.epochs = epochs
        self.patience = patience
        self.net = None
        self.mean = self.std = None
        self.history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}

    def _arrays(self, feat: pd.DataFrame, mask):
        seq = (feat[SEQUENCE_FEATURES].to_numpy(dtype=np.float32) - self.mean) / self.std
        idx = np.flatnonzero(np.asarray(mask))
        windows = idx[:, None] + np.arange(-LOOKBACK, 0)[None, :]
        x_seq = seq[windows]
        onehot = np.eye(self.n_cells, dtype=np.float32)[feat["cell_code"].to_numpy()[idx]]
        x_static = np.hstack([feat[STATIC_FEATURES].to_numpy(dtype=np.float32)[idx], onehot])
        y = seq[idx, 0]  # la cible (trafic) est la 1re variable de séquence
        return x_seq, x_static, y

    def _build_net(self, n_seq: int, n_static: int):
        import torch.nn as nn

        hidden = self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_seq, hidden, num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(
                    nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
                )

            def forward(self, x_seq, x_static):
                import torch

                out, _ = self.lstm(x_seq)
                return self.head(torch.cat([out[:, -1, :], x_static], dim=1)).squeeze(1)

        return Net()

    def fit(self, feat, train_mask, valid_mask):
        import torch

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        # Normalisation calculée uniquement sur les lignes d'entraînement.
        train_vals = feat.loc[train_mask, SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
        self.mean = train_vals.mean(axis=0)
        self.std = train_vals.std(axis=0) + 1e-6

        xs_tr, xst_tr, y_tr = (torch.from_numpy(a) for a in self._arrays(feat, train_mask))
        xs_va, xst_va, y_va = (torch.from_numpy(a) for a in self._arrays(feat, valid_mask))
        self.net = self._build_net(xs_tr.shape[2], xst_tr.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        best, best_state, wait = float("inf"), None, 0
        gen = torch.Generator().manual_seed(SEED)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(y_tr), generator=gen)
            total = 0.0
            for i in range(0, len(perm), 64):
                b = perm[i : i + 64]
                opt.zero_grad()
                loss = loss_fn(self.net(xs_tr[b], xst_tr[b]), y_tr[b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(xs_va, xst_va), y_va).item()
            self.history["train_loss"].append(total / len(y_tr))
            self.history["valid_loss"].append(val)
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, feat, mask):
        import torch

        xs, xst, _ = self._arrays(feat, mask)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(torch.from_numpy(xs), torch.from_numpy(xst)).numpy()
        return pred.astype(float) * float(self.std[0]) + float(self.mean[0])

    def save(self, directory: Path):
        import torch

        torch.save(
            {
                "state": self.net.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "n_cells": self.n_cells,
                "hidden": self.hidden,
                "history": self.history,
                "n_seq": len(SEQUENCE_FEATURES),
                "n_static": len(STATIC_FEATURES) + self.n_cells,
            },
            directory / "lstm.pt",
        )

    @classmethod
    def load(cls, directory: Path):
        import torch

        ckpt = torch.load(directory / "lstm.pt", weights_only=False)
        model = cls(ckpt["n_cells"], hidden=ckpt["hidden"])
        model.mean, model.std, model.history = ckpt["mean"], ckpt["std"], ckpt["history"]
        model.net = model._build_net(ckpt["n_seq"], ckpt["n_static"])
        model.net.load_state_dict(ckpt["state"])
        model.net.eval()
        return model


BASELINES = {
    "seasonal_naive": lambda: BaselineModel("seasonal_naive", "traffic_lag24"),
    "persistence": lambda: BaselineModel("persistence", "traffic_lag1"),
}


def load_model(name: str, directory: Path):
    if name in BASELINES:
        return BASELINES[name]()
    if name == "lstm":
        return LSTMModel.load(directory)
    return TabularModel.load(name, directory)
