"""API REST consommée par le frontend Angular.

Lancement (depuis backend/) :  uvicorn app.api:app --reload --port 8000
Prérequis : avoir entraîné les modèles avec `python -m app.train`.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import ARTIFACTS_DIR
from .data import TARGET, cell_list, load_raw
from .forecast import recursive_forecast
from .models import load_model

app = FastAPI(title="Prédiction du trafic réseau 4G", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _clean(values) -> list:
    """Liste JSON-compatible (NaN -> null)."""
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in values]


def _iso(ts: pd.Series) -> list[str]:
    return [t.isoformat() for t in ts]


@lru_cache
def raw() -> pd.DataFrame:
    return load_raw()


@lru_cache
def report() -> dict:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Modèles non entraînés : lancez `python -m app.train` dans backend/.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def predictions() -> pd.DataFrame:
    report()  # même message d'erreur si l'entraînement n'a pas été fait
    return pd.read_csv(ARTIFACTS_DIR / "predictions.csv", parse_dates=["timestamp"])


@lru_cache
def model(name: str):
    return load_model(name, ARTIFACTS_DIR)


def _check_cell(cell: str):
    if cell not in cell_list(raw()):
        raise HTTPException(404, f"Cellule inconnue : {cell}")


def _check_model(name: str):
    if name not in report()["models"]:
        raise HTTPException(404, f"Modèle inconnu ou non entraîné : {name}")


@app.get("/api/health")
def health():
    return {"status": "ok", "trained": (ARTIFACTS_DIR / "metrics.json").exists()}


@app.get("/api/overview")
def overview():
    rep = report()
    df = raw()
    return {
        "dataset": rep["dataset"],
        "trained_at": rep["trained_at"],
        "best_model": rep["best_model"],
        "models": [
            {"id": k, "label": v["label"], "kind": v["kind"], "fit_seconds": v["fit_seconds"], "metrics": v["metrics"]}
            for k, v in rep["models"].items()
        ],
        "kpis": {
            "mean_traffic": float(df[TARGET].mean()),
            "peak_traffic": float(df[TARGET].max()),
            "mean_users": float(df["active_users"].mean()),
            "events": df["event"].value_counts().to_dict(),
        },
    }


@app.get("/api/metrics/by-cell")
def metrics_by_cell():
    rep = report()
    return {k: v["by_cell"] for k, v in rep["models"].items()}


@app.get("/api/metrics/by-event")
def metrics_by_event():
    rep = report()
    return {k: v["by_event"] for k, v in rep["models"].items()}


@app.get("/api/feature-importance/{model_id}")
def feature_importance(model_id: str):
    imp = report()["feature_importance"].get(model_id)
    if imp is None:
        raise HTTPException(404, f"Pas d'importance de variables pour {model_id}")
    return imp


@app.get("/api/lstm/history")
def lstm_history():
    return report().get("lstm_history", {})


@app.get("/api/cells")
def cells():
    df = raw()
    out = []
    for c in cell_list(df):
        g = df[df["cell_id"] == c]
        out.append({
            "id": c,
            "mean_traffic": float(g[TARGET].mean()),
            "peak_traffic": float(g[TARGET].max()),
            "mean_users": float(g["active_users"].mean()),
            "mean_rsrp": float(g["rsrp"].mean()),
            "mean_sinr": float(g["sinr"].mean()),
            "congestion_hours": int((g["event"] == "congestion").sum()),
            "handover_hours": int((g["event"] == "handover").sum()),
        })
    return out


@app.get("/api/cells/{cell}/series")
def cell_series(cell: str):
    """Historique complet de la cellule + prédictions de chaque modèle sur la période de test."""
    _check_cell(cell)
    g = raw()[raw()["cell_id"] == cell].sort_values("timestamp")
    p = predictions()
    p = p[p["cell_id"] == cell].set_index("timestamp").reindex(g["timestamp"])
    return {
        "cell": cell,
        "test_start": report()["dataset"]["test_start"],
        "timestamps": _iso(g["timestamp"]),
        "actual": _clean(g[TARGET].tolist()),
        "active_users": g["active_users"].tolist(),
        "rsrp": _clean(g["rsrp"].tolist()),
        "sinr": _clean(g["sinr"].tolist()),
        "event": g["event"].tolist(),
        "predictions": {m: _clean(p[m].tolist()) for m in report()["models"]},
    }


@app.get("/api/cells/{cell}/forecast")
def cell_forecast(cell: str, model_id: str = Query("xgboost", alias="model"), horizon: int = Query(24, ge=1, le=72)):
    """Prévision des prochaines heures, avec un intervalle ~95 % (± 1,96 × RMSE de test de la cellule)."""
    _check_cell(cell)
    _check_model(model_id)
    fc = recursive_forecast(model(model_id), raw(), cell_list(raw()), cell, horizon)
    rmse = report()["models"][model_id]["by_cell"][cell]["rmse"]
    values = fc[TARGET].tolist()
    return {
        "cell": cell,
        "model": model_id,
        "timestamps": _iso(fc["timestamp"]),
        "values": values,
        "lower": [max(v - 1.96 * rmse, 0.0) for v in values],
        "upper": [v + 1.96 * rmse for v in values],
    }
