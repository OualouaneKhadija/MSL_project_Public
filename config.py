import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = Path(os.environ.get("MSL_DATASET_PATH", BASE_DIR / "dataset"))

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "msl_model.keras"
LABELS_PATH = MODELS_DIR / "labels.json"

HISTORY_PATH = REPORTS_DIR / "training_history.json"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"

SEQUENCE_LENGTH = 30

N_HANDS = 2
N_LANDMARKS_PER_HAND = 21
N_COORDINATES = 3
N_LANDMARKS = N_HANDS * N_LANDMARKS_PER_HAND

PREDICTION_THRESHOLD = 0.65
SMOOTHING_WINDOW = 5
COOLDOWN_FRAMES = 8
