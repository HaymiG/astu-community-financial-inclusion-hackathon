# Bank Account Prediction — CatBoost Classifier
 
A machine learning pipeline that predicts whether an individual has a bank account, built for the Zindi financial inclusion dataset. Uses CatBoost with 10-fold stratified cross-validation and feature engineering.
 
---
 
## Requirements
 
```bash
pip install pandas numpy catboost scikit-learn
```
 
---
 
## Dataset
 
Expected at `/kaggle/input/datasets/haymig/zindi-dataset/`:
 
| File | Description |
|---|---|
| `Train.csv` | Labeled training data with `bank_account` target column |
| `Test.csv` | Unlabeled test data for submission |
 
**Key columns used:**
 
- `bank_account` — target: `Yes` / `No`
- `cellphone_access`, `gender_of_respondent` — binary features
- `country`, `location_type`, `relationship_with_head`, `marital_status`, `education_level`, `job_type` — categorical features
- `household_size`, `age_of_respondent` — numeric features
- `uniqueid` — row identifier (used in submission, dropped from features)
---
 
## Pipeline Overview
 
### 1. Encoding
- Target `bank_account`: `Yes → 1`, `No → 0`
- Binary columns (`cellphone_access`, `gender_of_respondent`): mapped to `1/0`
### 2. Feature Engineering
 
| Feature | Formula | Rationale |
|---|---|---|
| `edu_job_combo` | `education_level + "_" + job_type` | Captures interaction between education and employment |
| `household_pressure` | `household_size / (age_of_respondent + 1)` | Proxy for financial dependents relative to age |
| `avg_age_country` | Mean age per country group | Country-level demographic baseline |
| `age_gap` | `age_of_respondent − avg_age_country` | Individual deviation from country average age |
 
> `avg_age_country` is used only to derive `age_gap` and is dropped before training.
 
### 3. Model — CatBoostClassifier
 
| Parameter | Value |
|---|---|
| `iterations` | 2000 |
| `learning_rate` | 0.02 |
| `depth` | 7 |
| `l2_leaf_reg` | 12 |
| `bootstrap_type` | Bayesian |
| `early_stopping_rounds` | 100 |
 
Categorical columns are passed natively to CatBoost (no manual encoding needed):
`country`, `location_type`, `relationship_with_head`, `marital_status`, `education_level`, `job_type`, `edu_job_combo`
 
### 4. Cross-Validation
- **Strategy:** 10-fold Stratified K-Fold (`random_state=42`)
- **Metric:** Mean Absolute Error (MAE) on validation folds
- Test set probabilities are averaged across all 10 folds before final prediction
### 5. Output
- Probabilities averaged → thresholded at `0.5` → binary `0/1`
- Submission ID format: `uniqueid x country`
- Saved to `submission_6.csv`
---
 
## Running the Script
 
```bash
python solution.py
```
 
Expected console output:
```
CV MAE (10-Fold): 0.XXXX
```
 
Output file: `submission_6.csv`
 
---
 
## Output Format
 
```
unique_id,bank_account
abc123 x Kenya,1
def456 x Rwanda,0
...
```
 
---
 
## Notes
 
- The script is designed to run in a Kaggle notebook environment; adjust file paths if running locally.
- CatBoost handles `NaN` values and categorical features internally — no imputation step is needed.
- Ensemble averaging of probabilities across folds reduces variance compared to a single model run.
