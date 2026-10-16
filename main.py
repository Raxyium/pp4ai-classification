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

# Part 06 - Create the preprocessing and classifier
# - Always fill missing features with medians.
# - Scale Logistic Regression and KNN; trees do not need scaling here.
# - These objects learn only when fit() is called on training rows.
def create_pipeline(model_key):
    if model_key not in MODELS:
        raise ValueError("Unsupported model selection.")
    classifiers = {
        "1": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "2": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "3": KNeighborsClassifier(n_neighbors=NEIGHBORS),
    }
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_key != "2":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifiers[model_key]))
    return Pipeline(steps)

# Part 07 - Train on one partition and evaluate the other
# - Stratification keeps class proportions similar; seed 42 repeats the split.
# - fit() learns medians, scaling, and the classifier from training data only.
# - predict() applies the learned steps to unseen test rows.
def train_and_evaluate(features, labels, model_key):
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels)
    if train_features.isna().all().any():
        raise ValueError("A training column is entirely missing; no median is available.")
    if model_key == "3" and len(train_labels) < NEIGHBORS:
        raise ValueError(f"KNN needs at least {NEIGHBORS} training rows.")
    pipeline = create_pipeline(model_key)
    pipeline.fit(train_features, train_labels)
    predictions = pipeline.predict(test_features)
    return {"accuracy": float(accuracy_score(test_labels, predictions)),
            "correct_count": int((predictions == test_labels.to_numpy()).sum()),
            "train_count": len(train_labels), "test_count": len(test_labels),
            "pipeline": pipeline}

# Part 08 - Run the menus, display results, and handle expected errors
# - Call the small functions in order and repeat when the user answers yes.
# - A data problem restarts the menu; q, Ctrl+C, or end of input exits cleanly.
def main(data_dir=DATA_DIR):
    print("Welcome to the Classification Application")
    dataset_names = {key: dataset["name"] for key, dataset in DATASETS.items()}
    try:
        while True:
            dataset_key = choose_option("dataset", dataset_names)
            if dataset_key is None:
                break
            dataset = DATASETS[dataset_key]
            try:
                features, labels = load_and_prepare(dataset, data_dir)
                model_key = choose_option("model", MODELS)
                if model_key is None:
                    break
                result = train_and_evaluate(features, labels, model_key)
                print(f"\nDataset: {dataset['name']} | Model: {MODELS[model_key]}")
                print(
                    f"Training instances: {result['train_count']} | Test instances: {result['test_count']}"
                )
                print(f"Test accuracy: {result['accuracy']:.2%}")
                print(f"Correct predictions: {result['correct_count']} / {result['test_count']}")
                           save_result(dataset, model_key, result)
            except ValueError as error:
                print(f"Experiment could not run: {error}\nReturning to the dataset menu.")
                continue
            if not ask_repeat():
                break
    except (EOFError, KeyboardInterrupt):
        print("\nInput ended. Exiting the application.")
    print("Goodbye!")

# Part 09 - Save each completed experiment to a CSV file
# - Create one row containing the dataset, model, accuracy, and row counts.
# - mode="a" adds a new row instead of replacing earlier results.
# - Write column headings only for a new or empty file; index=False skips row numbers.
# - If saving fails, explain the error and let the application continue.
def save_result(dataset, model_key, result):
    path = DATA_DIR.parent / "experiment_results.csv"
    row = {"dataset": dataset["name"], "model": MODELS[model_key],
           "accuracy": result["accuracy"], "train_count": result["train_count"],
           "test_count": result["test_count"], "correct_count": result["correct_count"]}
    try:
        needs_header = not path.exists() or path.stat().st_size == 0
        pd.DataFrame([row]).to_csv(path, mode="a", header=needs_header, index=False)
        print(f"Results saved to {path.name}.")
    except OSError as error:
        print(f"Could not save results to {path.name}: {error}")

# Keep this launch block last so every function above is defined before main() starts.
if __name__ == "__main__":
    main()
