# Authors: Nicholas Barletta and Andrea Pinnetti
"""Interactive classification experiments using the three supplied CSV files."""
# Part 01 - Imports and settings
# - Imports give us tools for files, tables, and machine learning.
# - The settings below keep the same choices across experiments.
from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# - Find data beside this file, even if the terminal starts elsewhere.
# - Reserve 20% of rows for testing; use seed 42 for a repeatable split.
DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_COLUMN = "target"
RANDOM_SEED = 42
TEST_SIZE = 0.20
NEIGHBORS = 5

# Part 02 - Menu dictionaries
# - Each string key matches what input() returns, such as "1".
# - Each dataset stores its name, CSV filename, and class meanings.
DATASETS = {
    "1": {
        "name": "Iris",
        "filename": "iris.csv",
        "labels": {0: "setosa", 1: "versicolor", 2: "virginica"},
    },
    "2": {
        "name": "Wine",
        "filename": "wine.csv",
        "labels": {0: "class_0", 1: "class_1", 2: "class_2"},
    },
    "3": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "filename": "breast_cancer.csv",
        "labels": {0: "malignant", 1: "benign"},
    },
}

