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

### Thursday, 12 November 2026 — Modeling workflow modularized

- Commit `3498bec` records modularizing the project and adds the root `main.py` orchestration flow and Python modules under `src/`.
- The current workflow loads the data, imputes missing `Internet Type` and `Offer` values, encodes selected categorical features, and runs three classifiers: K-nearest neighbors (`n_neighbors=5`), logistic regression, and a decision tree.
- Each trainer makes a stratified 80/20 train/test split with `random_state=42`. The KNN and logistic-regression trainers standardize the feature matrices; the decision-tree trainer uses the unscaled matrices.
- The prediction module generates test predictions. The evaluation module prints accuracy, precision, recall, F1, ROC AUC, confusion matrix, and classification report for each model.
- The current `main.py` runs all three model workflows and prints a heading before each model's metrics.
- The current evaluation function prints its metrics and returns `None`; `main.py` calls it directly rather than printing that return value.

### Current implementation status

- The notebook-based EDA and modeling work remains saved in `dataset/eda.ipynb`; the modular workflow is implemented in `main.py` and the populated modules under `src/`.
- The dataset used by the modular workflow is saved at `dataset/data/telco.csv`.
- Model tuning, ensembling, and cross-validation are not implemented yet; their corresponding files under `src/` are empty.
- The working tree contains uncommitted changes to `main.py`, `src/train.py`, and `src/predict.py` beyond the latest commit.
