# Interactive Classification Application
# Authors: Nicholas Barletta and Andrea Pinnetti

# Part 01 - Tools and settings
# - Libraries handle CSV tables, missing values, and classification.
# - Find data beside this file; keep the same split and model settings.
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
DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_COLUMN = "target"
RANDOM_SEED = 42
TEST_SIZE = 0.20
NEIGHBORS = 5

# Part 02 - Menu dictionaries
# - Each menu key points to its dataset details or model name.
# - Class labels explain the numbers stored in the target column.
DATASETS = {
    "1": {"name": "Iris", "filename": "iris.csv",
          "labels": {0: "setosa", 1: "versicolor", 2: "virginica"}},
    "2": {"name": "Wine", "filename": "wine.csv",
          "labels": {0: "class_0", 1: "class_1", 2: "class_2"}},
    "3": {"name": "Breast Cancer", "filename": "breast_cancer.csv",
          "labels": {0: "malignant", 1: "benign"}},
}
MODELS = {"1": "Logistic Regression", "2": "Decision Tree",
          "3": "K-Nearest Neighbors"}

# Part 03 - Validate menu input
# - Keep asking after a wrong answer; None means the user chose to exit.
def choose_option(title, options):
    print(f"\nAvailable {title}s:")
    for key, name in options.items():
        print(f"{key}. {name}")
    print("q. Exit")
    while True:
        choice = input(f"Choose a {title}: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice in options:
            return choice
        print(f"Invalid selection. Enter {', '.join(options)}, or q.")

# Part 04 - Repeat or finish
# - Return a Boolean only after receiving an accepted yes/no answer.
def ask_repeat():
    while True:
        answer = input("\nRun another experiment? (yes/no): ").strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Invalid answer. Enter yes/y or no/n.")

# Part 05 - Load, inspect, and prepare a dataset
# - Reading errors become helpful messages in the main menu.
# - Display dimensions, class counts, and every column's missing count.
def load_and_prepare(dataset, data_dir=DATA_DIR):
    path = Path(data_dir) / dataset["filename"]
    try:
        dataframe = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Cannot read CSV {path}: {error}") from error
    if dataframe.empty or TARGET_COLUMN not in dataframe.columns:
        raise ValueError("CSV must contain data rows and a 'target' column.")
    print(f"\nRows: {len(dataframe)} | Columns: {len(dataframe.columns)}")
    print(f"Target column: {TARGET_COLUMN}\nClass counts:")
    counts = dataframe[TARGET_COLUMN].value_counts(dropna=False)
    print(counts.rename(index=dataset["labels"]).to_string())
    print("Missing values per column:\n" + dataframe.isna().sum().to_string())
    # - Remove the answer column and reject values that are not numeric.
    # - Leave missing features for training-only median filling later.
    features = dataframe.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="raise")
    labels = pd.to_numeric(dataframe[TARGET_COLUMN], errors="raise")
    if features.shape[1] == 0:
        raise ValueError("There are no input feature columns.")
    if labels.isna().any() or not labels.isin(dataset["labels"]).all():
        raise ValueError("Target labels are missing or do not match this dataset.")
    # - Both partitions need enough observations for a stratified split.
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Insufficient data: need two classes with two rows each.")
    test_count = ceil(len(labels) * TEST_SIZE)
    if min(test_count, len(labels) - test_count) < len(counts):
        raise ValueError("Insufficient data for an 80/20 stratified split.")
    return features, labels.astype(int)
