# Programming Principles for AI: Project Datasets

These CSV files accompany the Interactive Classification Application project.
Extract the archive and put the data folder in your Python project.

## File format

- UTF-8 text, comma-separated fields, decimal points, and one header row.
- Each row is one instance. All feature columns are numeric.
- The final column, `target`, contains the integer class label in every file.
- Use all columns except `target` as input features. No index, identifier, or duplicate class-name column is included.
- Missing feature values are empty fields. pandas.read_csv() recognizes these as missing values by default.
- Feature names use lowercase letters and underscores. Their original order is preserved.

## Dataset summary

| File | Instances | Features | Classes | Missing feature values | Complete rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| data/iris.csv | 150 | 4 | 3 | 15 | 135 |
| data/wine.csv | 178 | 13 | 3 | 18 | 160 |
| data/breast_cancer.csv | 569 | 30 | 2 | 57 | 512 |

## Target labels

| Dataset | target | Meaning | Instances |
| --- | ---: | --- | ---: |
| iris | 0 | setosa | 50 |
| iris | 1 | versicolor | 50 |
| iris | 2 | virginica | 50 |
| wine | 0 | class_0 | 59 |
| wine | 1 | class_1 | 71 |
| wine | 2 | class_2 | 48 |
| breast_cancer | 0 | malignant | 212 |
| breast_cancer | 1 | benign | 357 |

The Wine labels class_0, class_1, and class_2 identify three wine cultivars. They are categories, not an ordered quality score.
For Breast Cancer, 0 means malignant and 1 means benign, matching the scikit-learn encoding.

## Introduced missing values

About 10% of the rows in each class were selected without replacement, and one feature value in each selected row was removed. Selection is random within each class. Missing values are distributed across feature columns. No row has more than one missing value.
All target labels, row order, and remaining feature values are unchanged. There are no originally missing values in these bundled source datasets.
This gives students practice handling missing data while retaining approximately 90% complete rows if they choose to drop incomplete rows.
The random seeds were 42 for Iris, 43 for Wine, and 44 for Breast Cancer, using NumPy default_rng. Counts per class were rounded to the nearest integer before sampling.

Follow the assignment instructions when preparing data. If you calculate imputation or scaling values, calculate them using the training set only and reuse them for the test set.

## Sources and feature names

The source data is the bundled scikit-learn 1.8.0 copies returned by load_iris(), load_wine(), and load_breast_cancer(). These are teaching copies with deliberately introduced missing values.

- Iris: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Wine: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
- Breast Cancer Wisconsin (Diagnostic): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

The table below maps the CSV headers to the feature names in the source loaders. Iris length measurements are in centimeters.

| Dataset | CSV feature | Source feature name |
| --- | --- | --- |
| iris | sepal_length_cm | sepal length (cm) |
| iris | sepal_width_cm | sepal width (cm) |
| iris | petal_length_cm | petal length (cm) |
| iris | petal_width_cm | petal width (cm) |
| wine | alcohol | alcohol |
| wine | malic_acid | malic_acid |
| wine | ash | ash |
| wine | alcalinity_of_ash | alcalinity_of_ash |
| wine | magnesium | magnesium |
| wine | total_phenols | total_phenols |
| wine | flavanoids | flavanoids |
| wine | nonflavanoid_phenols | nonflavanoid_phenols |
| wine | proanthocyanins | proanthocyanins |
| wine | color_intensity | color_intensity |
| wine | hue | hue |
| wine | od280_od315_of_diluted_wines | od280/od315_of_diluted_wines |
| wine | proline | proline |
| breast_cancer | mean_radius | mean radius |
| breast_cancer | mean_texture | mean texture |
| breast_cancer | mean_perimeter | mean perimeter |
| breast_cancer | mean_area | mean area |
| breast_cancer | mean_smoothness | mean smoothness |
| breast_cancer | mean_compactness | mean compactness |
| breast_cancer | mean_concavity | mean concavity |
| breast_cancer | mean_concave_points | mean concave points |
| breast_cancer | mean_symmetry | mean symmetry |
| breast_cancer | mean_fractal_dimension | mean fractal dimension |
| breast_cancer | radius_error | radius error |
| breast_cancer | texture_error | texture error |
| breast_cancer | perimeter_error | perimeter error |
| breast_cancer | area_error | area error |
| breast_cancer | smoothness_error | smoothness error |
| breast_cancer | compactness_error | compactness error |
| breast_cancer | concavity_error | concavity error |
| breast_cancer | concave_points_error | concave points error |
| breast_cancer | symmetry_error | symmetry error |
| breast_cancer | fractal_dimension_error | fractal dimension error |
| breast_cancer | worst_radius | worst radius |
| breast_cancer | worst_texture | worst texture |
| breast_cancer | worst_perimeter | worst perimeter |
| breast_cancer | worst_area | worst area |
| breast_cancer | worst_smoothness | worst smoothness |
| breast_cancer | worst_compactness | worst compactness |
| breast_cancer | worst_concavity | worst concavity |
| breast_cancer | worst_concave_points | worst concave points |
| breast_cancer | worst_symmetry | worst symmetry |
| breast_cancer | worst_fractal_dimension | worst fractal dimension |
