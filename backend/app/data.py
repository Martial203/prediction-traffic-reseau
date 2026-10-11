"""Chargement du CSV, feature engineering et découpage train/test.

Protocole commun à tous les modèles : on prédit le trafic d'une cellule à
l'heure t en n'utilisant que l'information disponible jusqu'à t-1 (lags du
trafic et des KPI radio) et le calendrier de t (connu à l'avance).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import DATA_PATH, LOOKBACK, TEST_RATIO, VALID_RATIO

TARGET = "traffic_volume"
EXOG = ["active_users", "rsrp", "sinr"]
EVENTS = ["congestion", "handover"]

TRAFFIC_LAGS = [1, 2, 3, 6, 12, 24]

# Variables tabulaires utilisées par Random Forest et XGBoost.
TABULAR_FEATURES = (
    ["hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "cell_code"]
    + [f"traffic_lag{k}" for k in TRAFFIC_LAGS]
    + ["traffic_roll_mean6", "traffic_roll_mean24", "traffic_roll_std24"]
    + [f"{c}_lag1" for c in EXOG]
    + [f"{e}_lag1" for e in EVENTS]
)

# Variables par pas de temps de la séquence LSTM (fenêtre t-LOOKBACK .. t-1).
SEQUENCE_FEATURES = [TARGET, *EXOG, *EVENTS, "hour_sin", "hour_cos"]
# Variables "statiques" de l'heure à prédire t, concaténées à la sortie du LSTM.
STATIC_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


REQUIRED_COLUMNS = ["timestamp", "cell_id", TARGET, *EXOG, "event"]


class DataError(ValueError):
    """Le fichier de données ne respecte pas le format attendu (voir docs/MIGRATION_DONNEES_REELLES.md)."""


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["cell_id"] = df["cell_id"].astype(str)
    validate(df)
    return df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Vérifie les hypothèses dont dépendent les features (lags = décalages de lignes)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Colonnes manquantes : {missing}. Attendu : {REQUIRED_COLUMNS}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise DataError("La colonne timestamp n'a pas pu être lue comme une date (format ISO attendu).")
    dup = df.duplicated(["cell_id", "timestamp"]).sum()
    if dup:
        raise DataError(f"{dup} lignes en double pour un même couple (cell_id, timestamp).")
    na = df[[TARGET, *EXOG]].isna().sum()
    if na.any():
        raise DataError(f"Valeurs manquantes à combler avant l'entraînement : {na[na > 0].to_dict()}")
    # Grille horaire régulière, sans trou, pour chaque cellule.
    steps = df.sort_values("timestamp").groupby("cell_id")["timestamp"].diff().dropna()
    irregular = (steps != pd.Timedelta(hours=1)).sum()
    if irregular:
        raise DataError(
            f"{irregular} écarts différents d'une heure entre mesures consécutives d'une même cellule "
            "(trous, doublons ou pas de temps différent). Ré-échantillonnez à l'heure."
        )


def cell_list(raw: pd.DataFrame) -> list[str]:
    """Identifiants triés en ordre « naturel » (Cell_2 avant Cell_10)."""
    def natural(c: str):
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", c)]

    return sorted(raw["cell_id"].unique(), key=natural)


def build_features(raw: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Ajoute calendrier, lags et rolling, calculés cellule par cellule."""
    df = raw.sort_values(["cell_id", "timestamp"]).reset_index(drop=True).copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["cell_code"] = df["cell_id"].map({c: i for i, c in enumerate(cells)})
    for e in EVENTS:
        df[e] = (df["event"] == e).astype(float)

    g = df.groupby("cell_id", sort=False)
    for k in TRAFFIC_LAGS:
        df[f"traffic_lag{k}"] = g[TARGET].shift(k)
    prev = g[TARGET].shift(1)
    pg = prev.groupby(df["cell_id"])
    df["traffic_roll_mean6"] = pg.transform(lambda s: s.rolling(6).mean())
    df["traffic_roll_mean24"] = pg.transform(lambda s: s.rolling(24).mean())
    df["traffic_roll_std24"] = pg.transform(lambda s: s.rolling(24).std())
    for c in [*EXOG, *EVENTS]:
        df[f"{c}_lag1"] = g[c].shift(1)
    # Position de la ligne dans l'historique de sa cellule (pour les fenêtres LSTM).
    df["pos"] = g.cumcount()
    return df


def split_timestamps(raw: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Retourne (début validation, début test), chronologiquement."""
    stamps = np.sort(raw["timestamp"].unique())
    usable = stamps[LOOKBACK:]
    test_start = usable[int(len(usable) * (1 - TEST_RATIO))]
    train_stamps = usable[usable < test_start]
    valid_start = train_stamps[int(len(train_stamps) * (1 - VALID_RATIO))]
    return pd.Timestamp(valid_start), pd.Timestamp(test_start)


def masks(feat: pd.DataFrame, valid_start, test_start):
    """Masques booléens train / valid / test. Les LOOKBACK premières heures
    de chaque cellule sont exclues pour que tous les modèles soient évalués
    sur exactement les mêmes lignes."""
    ok = feat["pos"] >= LOOKBACK
    ts = feat["timestamp"]
    train = ok & (ts < valid_start)
    valid = ok & (ts >= valid_start) & (ts < test_start)
    test = ok & (ts >= test_start)
    return train, valid, test
