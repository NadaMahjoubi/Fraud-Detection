from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_random_forest.joblib"
REQUIRED_COLUMNS = [
    "amount", "use_chip", "card_type", "mcc", "current_age",
    "credit_score", "longitude", "latitude"
]


def predict_file(input_path: Path, output_path: Path) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python -m src.train` first.")
    df = pd.read_csv(input_path) if input_path.suffix.lower() == ".csv" else pd.read_excel(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    model = joblib.load(MODEL_PATH)
    features = df[REQUIRED_COLUMNS].copy()
    features["amount"] = pd.to_numeric(
        features["amount"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.replace("-", "", regex=False).str.strip(),
        errors="coerce"
    )
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    result = df.copy()
    result["predicted_fraud"] = predictions
    result["fraud_probability"] = probabilities
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict potentially fraudulent transactions.")
    parser.add_argument("input", type=Path, help="Input CSV or Excel file")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predict_file(args.input, args.output)
