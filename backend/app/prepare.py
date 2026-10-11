"""Prépare un export de données réelles au format attendu par le pipeline.

Usage (depuis backend/) :
    python -m app.prepare export_brut.csv ../data/donnees_reelles.csv

Étapes :
  1. renomme les colonnes de l'export (COLUMN_MAP, à adapter) ;
  2. convertit les unités si besoin (TRAFFIC_TO_MBPS) ;
  3. agrège à l'heure et aligne chaque cellule sur une grille horaire continue ;
  4. comble les trous courts (≤ MAX_GAP_HOURS) par interpolation et écarte les cellules
     trop incomplètes ;
  5. valide le résultat (app.data.validate) et affiche un rapport.

Voir docs/MIGRATION_DONNEES_REELLES.md.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .data import EXOG, REQUIRED_COLUMNS, TARGET, validate

# À adapter à l'export réel.
# nom de colonne dans l'export -> nom attendu par le pipeline
COLUMN_MAP: dict[str, str] = {
    "timestamp": "timestamp",
    "cell_id": "cell_id",
    "traffic_volume": "traffic_volume",
    "active_users": "active_users",
    "rsrp": "rsrp",
    "sinr": "sinr",
    "event": "event",
}
# Facteur multiplicatif vers des Mbps (débit moyen sur l'heure).
# Ex. volume horaire en mégaoctets : 8 / 3600.  Déjà en Mbps : 1.
TRAFFIC_TO_MBPS = 1.0
# Fuseau horaire des timestamps de l'export ; tout est converti en UTC naïf.
SOURCE_TZ = "UTC"
# Trous plus longs que ce seuil : non interpolés (la cellule est alors écartée).
MAX_GAP_HOURS = 3
# Part maximale d'heures manquantes tolérée pour garder une cellule.
MAX_MISSING_RATIO = 0.05

# Agrégation à l'heure si l'export est plus fin (ex. 15 min).
AGG = {TARGET: "mean", "active_users": "mean", "rsrp": "mean", "sinr": "mean", "event": "first"}
# Priorité des événements quand plusieurs coexistent dans la même heure.
EVENT_PRIORITY = ["congestion", "handover", "normal"]


def _hourly_event(events: pd.Series) -> str:
    present = set(events.dropna())
    return next((e for e in EVENT_PRIORITY if e in present), "normal")


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (données propres, rapport par cellule)."""
    df = df.rename(columns=COLUMN_MAP)[REQUIRED_COLUMNS].copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(SOURCE_TZ, ambiguous="NaT", nonexistent="NaT")
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("h")
    df["cell_id"] = df["cell_id"].astype(str)
    df["event"] = df["event"].fillna("normal").astype(str).str.lower()
    df[TARGET] = df[TARGET] * TRAFFIC_TO_MBPS
    df = df.dropna(subset=["timestamp"])

    numeric = {k: v for k, v in AGG.items() if k != "event"}
    hourly = df.groupby(["cell_id", "timestamp"]).agg({**numeric, "event": _hourly_event}).reset_index()

    kept, report = [], []
    for cell, g in hourly.groupby("cell_id"):
        grid = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        g = g.set_index("timestamp").reindex(grid)
        missing = int(g[TARGET].isna().sum())
        g[[TARGET, *EXOG]] = g[[TARGET, *EXOG]].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
        still_missing = int(g[[TARGET, *EXOG]].isna().any(axis=1).sum())
        ok = still_missing == 0 and missing / len(grid) <= MAX_MISSING_RATIO
        report.append({
            "cell_id": cell,
            "debut": grid[0],
            "fin": grid[-1],
            "heures": len(grid),
            "heures_manquantes": missing,
            "non_comblees": still_missing,
            "gardee": ok,
        })
        if ok:
            g["cell_id"] = cell
            g["event"] = g["event"].fillna("normal")
            kept.append(g.rename_axis("timestamp").reset_index())

    report = pd.DataFrame(report)
    if not kept:
        raise ValueError("Aucune cellule exploitable : vérifier COLUMN_MAP et la complétude de l'export.")
    out = pd.concat(kept, ignore_index=True)[REQUIRED_COLUMNS]
    validate(out)
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="export brut (CSV)")
    parser.add_argument("destination", help="CSV au format du pipeline")
    args = parser.parse_args()

    out, report = prepare(pd.read_csv(args.source))
    out.to_csv(args.destination, index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(report.to_string(index=False))
    print(f"\n{len(out)} lignes, {out['cell_id'].nunique()} cellules gardées sur {len(report)}"
          f" -> {args.destination}")
    print("Événements :", out["event"].value_counts().to_dict())


if __name__ == "__main__":
    main()
