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

## Saving experiment results

Each completed experiment appends a row to `experiment_results.csv` beside
`main.py`. The file is created automatically and keeps earlier results across
experiments and application restarts. It records the dataset, model, accuracy,
training count, test count, and correct prediction count. Accuracy is stored as
a fraction: for example, 0.933333 means approximately 93.33%.

The column headings are written once for a new or empty file. Cancelled or
failed experiments do not add rows. If saving fails, the application displays
an error and continues. Results are saved locally; they are not automatically
uploaded to GitHub.

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
| `load_and_prepare` | Load and summarize a CSV, separate features and labels, and validate usable data. |
| `create_pipeline` | Create preprocessing and the selected classifier. |
| `train_and_evaluate` | Split, train, predict, and return accuracy, counts, and the fitted pipeline. |
| `main` | Coordinate menus, display and save results, handle errors, repeat, and exit. |
| `save_result` | Append a result row to the CSV and handle file-writing errors. |

## Validation

Verified on macOS with Python 3.14.8, pandas 2.3.3, and scikit-learn 1.8.0.
All nine dataset/model combinations completed without training warnings.
Checks confirmed training-only imputation and scaling, target exclusion,
disjoint training/test rows, and independently calculated accuracy.
Invalid numbers, words, blank input, repeat answers, file errors, insufficient
data, repeated experiments, and clean exits were also checked.
CSV checks confirmed correct values, one header, appending across experiments
and application restarts, and handling of write failures. The final ZIP was
extracted, its dependencies installed, and the application run successfully.

## AI assistance and limitations

OpenAI Codex helped plan and generate code; the resulting code was checked by
running the application.

The application supports the three supplied numeric datasets and fixed
model settings. Results use one reproducible test split. Windows and Linux
execution were not tested.
