from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fraud_model_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

NUMERIC_FEATURES = ["amount", "mcc", "current_age", "credit_score", "longitude", "latitude"]
CATEGORICAL_FEATURES = ["use_chip", "card_type"]
TARGET = "target"


def clean_amount(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.replace("-", "", regex=False).str.strip(),
        errors="coerce"
    )


def train_model() -> Path:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}.\n"
            "Run `python -m src.data_preprocessing` first."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["amount"] = clean_amount(df["amount"])
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError("The target must contain both classes: legitimate (0) and fraud (1).")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    except ValueError:
        print("ROC-AUC could not be calculated for this test split.")

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "fraud_random_forest.joblib"
    joblib.dump(pipeline, model_path)
    with (OUTPUT_DIR / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nModel saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()
