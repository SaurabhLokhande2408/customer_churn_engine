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

