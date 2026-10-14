# Interactive Classification Application

**Authors:** Nicholas Barletta and Andrea Pinnetti  
**Course:** Programming Principles for AI

A Python terminal application that loads the supplied Iris, Wine, and Breast
Cancer datasets, trains a selected classifier, and reports test accuracy.

## Setup and usage

Use Python 3.11–3.14. From the project folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, create the environment with `python -m venv .venv` and activate it
with `.venv/Scripts/Activate.ps1` in PowerShell.

Select a dataset and model using `1`, `2`, or `3`. Enter `yes` to run another
experiment, `no` to finish, or `q` at either menu to exit. Invalid selections
prompt another attempt. File and data errors produce a message and return to
the dataset menu.

## Data preparation

The supplied CSVs in `data/` are unchanged. The summary displays rows, columns,
the target, class counts, and missing values per column. Class meanings are
listed in `data/README.md`.

The `target` column is excluded from input features. Missing feature values use
column medians learned from training data only; all rows are retained. The data
is split into 80% training and 20% testing, stratified by class, with seed 42.
Preprocessing and models are fitted on training rows; accuracy uses test rows.

| Model | Settings |
| --- | --- |
| Logistic Regression | Standard scaling; `max_iter=2000`; seed 42. |
| Decision Tree | No scaling; seed 42; other settings use library defaults. |
| K-Nearest Neighbors | Standard scaling; 5 neighbors. |

Scaling is fitted on training data only for Logistic Regression and KNN.
Every model uses median imputation.

## Functions

| Function | Responsibility |
| --- | --- |
| `choose_option` | Display a menu and validate its selection. |
| `ask_repeat` | Validate a yes/no answer. |
| `load_dataset` | Load a CSV and explain file errors. |
| `show_summary` | Display dimensions, classes, and missing values. |
| `prepare_features` | Separate features and labels; validate usable data. |
| `create_pipeline` | Create preprocessing and the selected classifier. |
| `train_and_evaluate` | Split, train, predict, and calculate accuracy. |
| `show_result` | Display choices, split sizes, and results. |
| `main` | Coordinate experiments, recovery, repetition, and exit. |

## Validation

Verified on macOS with Python 3.14.8, pandas 2.3.3, and scikit-learn 1.8.0.
All nine dataset/model combinations completed without training warnings.
Checks confirmed training-only imputation and scaling, target exclusion,
disjoint training/test rows, and independently calculated accuracy.
Invalid numbers, words, blank input, repeat answers, file errors, insufficient
data, repeated experiments, and clean exits were also checked.

## AI assistance and limitations

OpenAI Codex assisted with implementation, documentation, and debugging;
generated code was reviewed and validated through execution and independent
checks. The application supports the three supplied numeric datasets and fixed
model settings. Results use one reproducible test split. Windows and Linux
execution were not tested.
