"""Constantes compartilhadas pelo pipeline."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "label"
ATTACK_CATEGORY_COLUMN = "attack_cat"
IDENTIFIER_COLUMNS = ("id",)
DEFAULT_CATEGORICAL_COLUMNS = ("proto", "service", "state")

TRAIN_FILENAME = "UNSW_NB15_training-set.csv"
TEST_FILENAME = "UNSW_NB15_testing-set.csv"

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("outputs")
