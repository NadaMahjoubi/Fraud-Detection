
# FraudLens — Fraud Detection & Transaction Risk Dashboard

FraudLens is a Python project for exploring transaction data, training a Random Forest fraud classifier, and scoring transactions through an interactive Streamlit dashboard.

The project includes a data-preparation pipeline, model training and evaluation, a command-line prediction script, and a dashboard for reviewing predicted fraud risk.

> Project status: Educational / portfolio project. Predictions are model estimates and should not be treated as confirmed fraud or used as the sole basis for financial decisions.

## Features

* Data preparation: Combines transaction, card, user, merchant-category, and fraud-label data.
* Model training: Trains a Random Forest classifier with preprocessing steps.
* Model evaluation: Reports classification metrics and ROC-AUC.
* Batch predictions: Scores transactions from CSV or Excel files.
* Interactive dashboard: Displays transaction metrics, predicted-risk charts, filters, and downloadable results.
* Adjustable alert threshold: Changes the probability threshold used to flag transactions in the dashboard.

## Project structure

```
fraud-detection-project/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md
├── data/
│   ├── raw/                       # Original dataset files
│   └── processed/                 # Prepared model dataset
├── models/
│   └── fraud_random_forest.joblib # Trained model pipeline
├── outputs/
│   ├── classification_report.json # Evaluation report
│   └── predictions.csv            # Exported predictions
└── src/
    ├── __init__.py
    ├── data_preprocessing.py      # Data preparation
    ├── train.py                   # Model training and evaluation
    └── predict.py                 # Command-line predictions
```

Some files and folders are generated when you run the project. The trained model and data files may not be included in the repository or ZIP, so generate them locally if they are missing.

## Requirements

* Python 3.10 or later recommended
* pip
* The dataset files listed below
* A trained model file for dashboard scoring

## Installation (Windows / VS Code)

### 1. Open the project

Extract the project ZIP and open the `fraud-detection-project` folder in VS Code. Open Terminal → New Terminal.

### 2. Create and activate a virtual environment

In PowerShell:

PowerShell

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

bat

```
.venv\Scripts\activate.bat
```

### 3. Install dependencies

PowerShell

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset setup

Place the source files in `data/raw/`:

* `transactions_data.csv`
* `cards_data.csv`
* `users_data.csv`
* `train_fraud_labels.json`
* `mcc_codes.json`

Use the dataset version associated with your original notebook. The preprocessing script expects these filenames and locations.

Do not commit private credentials, API keys, or sensitive transaction data to Git.

## Run the pipeline

Run the commands from the project root, with the virtual environment activated.

### 1. Prepare the data

PowerShell

```
python -m src.data_preprocessing
```

This creates:

```
data/processed/fraud_model_data.csv
```

The current preprocessing workflow selects a set of model features and balances the fraud/non-fraud labels while sampling up to 5,000 rows. As a result, the processed data's fraud rate may differ substantially from the rate in real transactions.

### 2. Train and evaluate the model

PowerShell

```
python -m src.train
```

The training script evaluates the classifier and saves the fitted preprocessing/model pipeline to:

```
models/fraud_random_forest.joblib
```

The evaluation report is written to:

```
outputs/classification_report.json
```

The terminal output includes metrics such as accuracy, the classification report, and ROC-AUC.

### 3. Score a CSV or Excel file from the command line

Prepare a file containing the required feature columns, then run:

PowerShell

```
python -m src.predict "path\to\transactions.csv"
```

To choose an output path:

PowerShell

```
python -m src.predict "path\to\transactions.csv" --output "outputs\predictions.csv"
```

The output includes the model prediction and fraud probability. The input must contain all eight required features listed below.

## Launch the FraudLens dashboard

After dependencies are installed and `models/fraud_random_forest.joblib` exists, run:

PowerShell

```
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually:

```
http://localhost:8501
```

Open that address in your browser if the dashboard does not open automatically.

### Dashboard capabilities

* Upload a CSV or Excel file for scoring.
* Load `data/processed/fraud_model_data.csv` when it exists.
* Adjust the fraud-alert threshold.
* View the number of transactions scored, transactions flagged, average predicted fraud score, and high-risk queue.
* Explore prediction mix and fraud-score distribution charts.
* Filter and search scored transactions.
* Download the scored results as a CSV.

### Required input columns

The dashboard and prediction script expect these eight columns:

| Column           | Description                            |
| ---------------- | -------------------------------------- |
| `amount`       | Transaction amount                     |
| `use_chip`     | Payment method / chip usage category   |
| `card_type`    | Card category                          |
| `mcc`          | Merchant category code                 |
| `current_age`  | Customer age feature                   |
| `credit_score` | Customer credit-score feature          |
| `longitude`    | Transaction/location longitude feature |
| `latitude`     | Transaction/location latitude feature  |

Column names must match exactly. Extra columns may be present; the model uses the eight required features.

## Understanding the predictions

* `predicted_fraud = 1`: the transaction is flagged by the model at the selected threshold.
* `predicted_fraud = 0`: the transaction is not flagged at that threshold.
* `fraud_probability`: the model's estimated probability for the fraud class.
* Alert threshold: lowering it generally flags more transactions, potentially increasing false positives; raising it generally flags fewer transactions, potentially missing more fraud.

A predicted fraud label is not proof that a transaction is fraudulent. Results should be reviewed alongside other evidence.

## Evaluation and limitations

* The processed dataset is balanced by the preprocessing script. Metrics measured on this dataset may not represent performance under the naturally low fraud prevalence often seen in production.
* Accuracy alone is insufficient for fraud detection. Review fraud-class precision, recall, F1-score, the confusion matrix, and ROC-AUC; precision-recall metrics can also be useful for imbalanced data.
* Evaluate with a representative, held-out dataset and check for data leakage before drawing conclusions about real-world performance.
* The model is a demonstration and portfolio project, not a production-ready fraud prevention system.
* Keep sensitive data local and only use data you are authorized to process. Avoid deploying the dashboard publicly without appropriate authentication, access controls, and data-protection measures.

## Troubleshooting

`Trained model not found`

Train the model first:

PowerShell

```
python -m src.train
```

Confirm that `models/fraud_random_forest.joblib` exists.

`Missing required columns`

Check the CSV/Excel headers against the eight required input columns above. Rename columns if necessary.

`ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies:

PowerShell

```
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dashboard command not recognized

Run the command through the active Python environment:

PowerShell

```
python -m streamlit run app.py
```

## Tech stack

Python · pandas · scikit-learn · joblib · Streamlit · Plotly
