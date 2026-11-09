# Customer Churn Project Learning Timeline

This tracker describes only the files and saved work present in this repository. It does not include information from conversations. The notebook does not record when each analysis step was first completed, so earlier dates cannot be reliably assigned.

## Repository Snapshot — Saturday, 7 November 2026

### Exploratory data analysis

The saved `dataset/eda.ipynb` contains these steps:

1. Import pandas, NumPy, Matplotlib, Seaborn, and `warnings`.
2. Load the dataset through `data_loading` and create a pandas DataFrame.
3. Inspect sample rows, DataFrame structure, column names, and descriptive statistics.
4. Remove `Customer ID`, `Zip Code`, `Latitude`, `Longitude`, and `Referred a Friend`.
5. Separate numeric and categorical columns using pandas data types.
6. Inspect value counts for `Churn Category` and `Internet Type`.
7. Check for missing values and drop rows missing values in `Churn Category`, `Churn Reason`, `Internet Type`, or `Offer`.

The notebook's saved output reports 7,043 records and 50 columns before the selected columns are removed.

### Project files

- `dataset/` contains `telco.csv`, `data_loading.py`, and `eda.ipynb`.
- `src/` contains files named for preprocessing, feature engineering, training, prediction, evaluation, ensembling, cross-validation, and hyperparameter tuning. These files are currently empty.
- `requirements.txt`, the root `README.md`, and the files under `reports/` are currently empty.

## Repository Progress Added — Monday, 9 November 2026

The dated milestones below are based on the repository's Git commit history and the saved notebook. Times are recorded in IST (UTC+05:30), as shown in the commit timestamps.

### Saturday, 7 November 2026 — Project and EDA started

- The repository history records starting the project and beginning EDA.
- The EDA notebook imports the Telco customer dataset and begins inspecting its dimensions, columns, and summary statistics.

### Sunday, 8 November 2026 — Data cleaning and analysis

- The repository history records data cleaning as complete and the start of analysis.
- The notebook contains missing-value checks and imputes missing `Internet Type` and `Offer` values as `Unknown`.
- The notebook also checks duplicate rows and explores distributions, numeric correlations, and relationships between features and churn-related values.

### Monday, 9 November 2026 — EDA, model training, and evaluation in the notebook

- The latest commit records EDA completion and model training/evaluation in the notebook; it also notes that the work is not yet modularized.
- The saved notebook encodes categorical variables, removes selected columns, creates one-hot encoded features, and splits the data into stratified training and test sets.
- The notebook scales numeric features, trains a K-nearest neighbors classifier (`n_neighbors=5`) to predict `Churn Label`, and includes accuracy, classification report, and confusion matrix evaluation cells.
- The saved accuracy output is approximately **0.8836** (88.36%).
- The latest commit message lists model tuning and ensemble learning as future work.

### Current implementation status

- The saved modeling workflow is in `dataset/eda.ipynb`; the existing Python modules under `src/` remain empty.
- `requirements.txt`, the root `README.md`, and the report files remain empty in the current repository snapshot.
- Git reports an uncommitted modification to `dataset/eda.ipynb` after the latest commit.
