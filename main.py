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

MODELS = {
    "1": "Logistic Regression",
    "2": "Decision Tree",
    "3": "K-Nearest Neighbors",
}


# Part 03 - Choose a menu option
# - Show the menu once, then keep asking until an answer is valid.
# - Return the chosen key; return None when the user wants to exit.
def choose_option(title, options, prompt):
    """Display option names and return a valid menu key, or None to quit."""
    print(f"\n{title}")
    for key, name in options.items():
        print(f"{key}. {name}")
    print("q. Exit")
    # - Build the list of allowed keys for the error message.
    valid_choices = ", ".join(options)

    while True:
        # - Remove outer spaces and ignore capital letters.
        choice = input(prompt).strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice in options:
            return choice
        print(f"Invalid selection. Please enter {valid_choices}, or q to exit.")

# Part 04 - Repeat or stop
# - Convert accepted yes/no text into True or False for main().
# - An invalid answer stays in this loop and gets another attempt.
def ask_repeat():
    """Ask until the user enters an accepted yes/no answer."""
    while True:
        answer = input("\nRun another experiment? (yes/no): ").strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Invalid answer. Please enter yes/y or no/n.")


# Part 05 - Load the CSV
# - Join the data folder and filename, then read a pandas table.
# - Explain expected file problems and reject unusable tables.
def load_dataset(dataset, data_dir=DATA_DIR):
    """Read a supplied CSV and turn expected file problems into clear errors."""
    path = Path(data_dir) / dataset["filename"]
    try:
        dataframe = pd.read_csv(path)
    except FileNotFoundError as error:
        raise ValueError(
            f"CSV file not found: {path}. Restore it in the data folder."
        ) from error
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Cannot read CSV file {path}: {error}") from error

    # - A readable file still needs rows and a target column.
    if dataframe.empty:
        raise ValueError(f"CSV file {path.name} has no data rows.")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"CSV file {path.name} has no '{TARGET_COLUMN}' column.")
    return dataframe


# Part 06 - Describe the data
# - Print table size, class counts, and missing counts for each column.
# - This helps us inspect the data before training; it changes no values.
def show_summary(dataframe, dataset):
    """Print dimensions, class counts, and missing counts for every column."""
    print(f"\nDataset summary: {dataset['name']}")
    print(f"Rows: {len(dataframe)} | Columns: {len(dataframe.columns)}")
    print(f"Target column: {TARGET_COLUMN}")
    print("Class counts:")
    # - Include missing labels in the count so problems stay visible.
    counts = dataframe[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    for label, count in counts.items():
        meaning = dataset["labels"].get(label, "unknown or missing label")
        print(f"  {label} ({meaning}): {count}")
    print("Missing values per column:")
    for column, count in dataframe.isna().sum().items():
        print(f"  {column}: {count}")
