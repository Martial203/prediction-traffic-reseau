"""Chemins et paramètres partagés par l'entraînement et l'API."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

# Fichier de données : surchargeable sans toucher au code (ex. données réelles).
DATA_PATH = Path(os.environ.get("TRAFFIC_DATA_PATH", PROJECT_DIR / "synthetic_4g_network_data.csv"))
# Dossier des modèles entraînés et résultats : surchargeable aussi.
ARTIFACTS_DIR = Path(os.environ.get("TRAFFIC_ARTIFACTS_DIR", BACKEND_DIR / "artifacts"))

# Part finale (chronologique) des timestamps réservée au test.
TEST_RATIO = 0.2
# Part finale du train réservée à la validation (early stopping LSTM).
VALID_RATIO = 0.1
# Nombre d'heures d'historique nécessaires (plus grand lag / fenêtre LSTM).
LOOKBACK = 24
SEED = 42

# Modèles comparés. "kind" = ml (entraîné) ou baseline (règle naïve).
MODELS = {
    "random_forest": {"label": "Random Forest", "kind": "ml"},
    "xgboost": {"label": "XGBoost", "kind": "ml"},
    "lstm": {"label": "LSTM", "kind": "ml"},
    "seasonal_naive": {"label": "Naïf saisonnier (J-1)", "kind": "baseline"},
    "persistence": {"label": "Persistance (H-1)", "kind": "baseline"},
}
