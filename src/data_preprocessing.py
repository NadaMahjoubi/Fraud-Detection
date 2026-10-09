from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _read_mcc_codes(path: Path) -> pd.DataFrame:
    """Read the MCC JSON file, supporting JSON Lines and regular JSON."""
    try:
        raw = pd.read_json(path, lines=True)
    except ValueError:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            raw = pd.DataFrame([obj])
        elif isinstance(obj, list):
            raw = pd.DataFrame(obj)
        else:
            raise ValueError(f"Unsupported structure in {path.name}: {type(obj).__name__}")

    # The original notebook stores MCC codes as columns in some versions.
    if raw.shape[0] == 1 and raw.shape[1] > 1:
        raw = raw.T.reset_index()
        raw.columns = ["mcc", "Name"]
    elif "mcc" not in raw.columns:
        raw = raw.T.reset_index()
        raw.columns = ["mcc", "Name"] if raw.shape[1] == 2 else list(raw.columns)

    if "mcc" not in raw.columns:
        raw = raw.reset_index()
        raw = raw.rename(columns={raw.columns[0]: "mcc"})
    if "Name" not in raw.columns:
        candidates = [c for c in raw.columns if c != "mcc"]
        if candidates:
            raw = raw.rename(columns={candidates[0]: "Name"})
    raw["mcc"] = pd.to_numeric(raw["mcc"], errors="coerce")
    return raw.dropna(subset=["mcc"])[["mcc", "Name"]].drop_duplicates("mcc")


def build_dataset(sample_size: int = 5000, random_state: int = 42) -> Path:
    """Merge source files and save the model-ready dataset used by the project."""
    required = [
        "transactions_data.csv", "cards_data.csv", "users_data.csv",
        "train_fraud_labels.json", "mcc_codes.json"
    ]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing source files in data/raw/: " + ", ".join(missing) +
            "\nDownload/extract the dataset and place those files in data/raw/."
        )

    transactions = pd.read_csv(RAW_DIR / "transactions_data.csv")
    cards = pd.read_csv(RAW_DIR / "cards_data.csv")
    users = pd.read_csv(RAW_DIR / "users_data.csv")
    labels_raw = pd.read_json(RAW_DIR / "train_fraud_labels.json")

    if labels_raw.empty or labels_raw.shape[1] == 0:
        raise ValueError("train_fraud_labels.json does not contain a label column.")
    labels = pd.DataFrame({"id": labels_raw.index, "Status": labels_raw.iloc[:, 0]})
    labels["id"] = pd.to_numeric(labels["id"], errors="coerce")
    labels = labels.dropna(subset=["id"])
    labels["id"] = labels["id"].astype(int)

    mcc_codes = _read_mcc_codes(RAW_DIR / "mcc_codes.json")

    # Clean fields used in the original notebook.
    for col in ["expires", "acct_open_date"]:
        if col in cards.columns:
            cards[col] = pd.to_datetime(cards[col], errors="coerce")
    if "credit_limit" in cards.columns:
        cards["credit_limit"] = (
            cards["credit_limit"].astype(str).str.replace("$", "", regex=False).str.strip()
        )
        cards["credit_limit"] = pd.to_numeric(cards["credit_limit"], errors="coerce")

    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        if col in users.columns:
            users[col] = pd.to_numeric(
                users[col].astype(str).str.replace("$", "", regex=False).str.strip(),
                errors="coerce"
            )

    fraudulent = labels[labels["Status"].astype(str).str.lower().isin(["yes", "true", "1"])]
    non_fraudulent = labels[~labels.index.isin(fraudulent.index)]
    if fraudulent.empty:
        raise ValueError(
            "No fraud labels were detected. Check the format of train_fraud_labels.json "
            "and the label values expected by the original notebook ('Yes'/'No')."
        )
    n = min(len(fraudulent), len(non_fraudulent))
    balanced = pd.concat([
        fraudulent.sample(n=n, random_state=random_state),
        non_fraudulent.sample(n=n, random_state=random_state)
    ]).sample(frac=1, random_state=random_state)

    transaction_fraud = transactions.merge(balanced, on="id", how="inner")
    if sample_size and len(transaction_fraud) > sample_size:
        transaction_fraud = transaction_fraud.sample(
            n=sample_size, random_state=random_state
        )
    transaction_fraud = transaction_fraud.merge(mcc_codes, on="mcc", how="inner")

    cards_drop = [c for c in [
        "card_number", "year_pin_last_changed", "acct_open_date", "expires", "cvv"
    ] if c in cards.columns]
    users_drop = [c for c in [
        "birth_year", "birth_month", "address", "per_capita_income"
    ] if c in users.columns]
    cards1 = cards.drop(columns=cards_drop).rename(columns={"id": "C_id"})
    users1 = users.drop(columns=users_drop).rename(columns={"id": "CL_id"})

    transaction_fraud = transaction_fraud.rename(columns={"id": "T_id"})
    if "client_id" not in transaction_fraud.columns or "card_id" not in transaction_fraud.columns:
        raise ValueError("Transactions file must contain client_id and card_id columns.")

    merged = transaction_fraud.set_index("client_id").join(
        users1.set_index("CL_id"), how="inner", lsuffix="_transactions", rsuffix="_users"
    )
    merged = merged.set_index("card_id").join(
        cards1.set_index("C_id"), how="inner",
        lsuffix="_transactions_users", rsuffix="_cards"
    ).reset_index(drop=True)

    drop_cols = [c for c in ["T_id", "date", "merchant_id", "client_id"] if c in merged.columns]
    data = merged.drop(columns=drop_cols)

    data["target"] = data["Status"].astype(str).str.lower().isin(
        ["yes", "true", "1"]
    ).astype(int)
    data = data.drop(columns=["Status"])

    desired = [
        "amount", "use_chip", "card_type", "mcc", "current_age",
        "credit_score", "longitude", "latitude", "target"
    ]
    absent = [c for c in desired if c not in data.columns]
    if absent:
        raise ValueError(f"Expected model columns are missing after merging: {absent}")
    data = data[desired].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "fraud_model_data.csv"
    data.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path}")
    print(f"Rows: {len(data)} | Fraud rate: {data['target'].mean():.2%}")
    return output_path


if __name__ == "__main__":
    build_dataset()
