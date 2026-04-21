"""Project-wide settings and canonical paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_DIR = PROCESSED_DATA_DIR / "eval_reports"
BACKUP_DIR = DATA_DIR / "backups"

MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_SEQUENCE_CSV = RAW_DATA_DIR / "landmarks_sequences.csv"
DEFAULT_LABEL_MAP = RAW_DATA_DIR / "label_map.txt"

DEFAULT_MODEL_PATH = MODELS_DIR / "best_gru_hands_pose_full.pt"
DEFAULT_LAST_MODEL_PATH = MODELS_DIR / "last_gru.pt"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

DEFAULT_WINDOW_SIZE = 20
