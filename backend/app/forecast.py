"""Prévision récursive au-delà de la dernière heure connue.

Chaque heure prédite est réinjectée comme historique pour l'heure suivante.
Les KPI radio futurs (utilisateurs actifs, RSRP, SINR) étant inconnus, on les
remplace par le profil horaire moyen des 7 derniers jours de la cellule, et on
suppose qu'aucun événement (congestion/handover) n'aura lieu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOOKBACK
from .data import EXOG, TARGET, build_features


def recursive_forecast(model, raw: pd.DataFrame, cells: list[str], cell: str, horizon: int) -> pd.DataFrame:
    hist = raw[raw["cell_id"] == cell].sort_values("timestamp")
    profile = hist.tail(24 * 7).groupby(hist["timestamp"].dt.hour)[EXOG].mean()

    last = hist["timestamp"].max()
    future_ts = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"timestamp": future_ts, "cell_id": cell, TARGET: np.nan, "event": "normal"})
    for c in EXOG:
        future[c] = profile.loc[future_ts.hour, c].to_numpy()

    # On ne garde que l'historique nécessaire aux lags / fenêtres.
    ext = pd.concat([hist.tail(LOOKBACK * 2 + 1), future], ignore_index=True)
    for t in future_ts:
        feat = build_features(ext, cells)
        pred = float(model.predict(feat, feat["timestamp"] == t)[0])
        ext.loc[ext["timestamp"] == t, TARGET] = max(pred, 0.0)
    return ext[ext["timestamp"].isin(future_ts)][["timestamp", TARGET]].reset_index(drop=True)
