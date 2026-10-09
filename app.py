from pathlib import Path
import io
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_random_forest.joblib"
REPORT_PATH = ROOT / "outputs" / "classification_report.json"
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "fraud_model_data.csv"
FEATURES = [
    "amount", "use_chip", "card_type", "mcc", "current_age",
    "credit_score", "longitude", "latitude"
]

st.set_page_config(
    page_title="FraudLens | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eef2ff; --muted:#9aa8c7; --panel:#111b31; --line:rgba(151,170,211,.16); }
.stApp { background: radial-gradient(ellipse at 15% 0%, #182a4b 0%, #0a1020 42%, #080d18 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#0b1324; border-right:1px solid var(--line); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
.hero { padding:1.5rem 1.7rem; border:1px solid rgba(114,143,204,.25); border-radius:24px; background:linear-gradient(115deg,rgba(28,48,82,.95),rgba(17,27,49,.82)); box-shadow:0 18px 55px rgba(0,0,0,.2); margin-bottom:1.3rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#69e0d0; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:2.45rem; line-height:1.12; font-weight:700; margin:.45rem 0 .55rem; color:#f4f7ff; }
.hero-sub { color:#aebbd5; font-size:1rem; max-width:720px; }
.metric { background:linear-gradient(145deg,rgba(22,34,59,.96),rgba(15,24,43,.94)); border:1px solid var(--line); padding:1.05rem 1.15rem; border-radius:18px; min-height:124px; }
.metric-label { color:#a7b4cf; font-size:.8rem; font-weight:600; }
.metric-value { color:#f4f7ff; font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700; margin:.28rem 0; }
.metric-note { color:#8495b8; font-size:.76rem; }
.section-head { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:600; margin:.5rem 0 .8rem; color:#f0f4ff; }
.panel { border:1px solid var(--line); background:rgba(17,27,49,.76); padding:1.1rem 1.2rem; border-radius:18px; }
div[data-testid="stFileUploader"] { border:1px dashed rgba(105,224,208,.45); border-radius:16px; padding:.5rem; background:rgba(22,37,63,.42); }
.stButton button, .stDownloadButton button { border-radius:12px; border:1px solid rgba(105,224,208,.45); font-weight:700; }
.stButton button[kind="primary"] { background:linear-gradient(90deg,#24b8ad,#4f7df0); color:white; border:0; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stAlert { border-radius:12px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    features = df[FEATURES].copy()
    features["amount"] = pd.to_numeric(
        features["amount"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.replace("-", "", regex=False).str.strip(),
        errors="coerce",
    )
    for col in ["mcc", "current_age", "credit_score", "longitude", "latitude"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    for col in ["use_chip", "card_type"]:
        features[col] = features[col].astype("string").fillna("Unknown").str.strip()
    return features


def money(value):
    return f"${value:,.2f}" if pd.notna(value) else "—"


with st.sidebar:
    st.markdown("<div style='font-size:2rem'>🛡️</div><div style='font-family:Space Grotesk;font-size:1.25rem;font-weight:700'>FraudLens</div><div style='color:#8fa0c2;font-size:.82rem;margin-bottom:1.5rem'>Transaction intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Workspace**")
    st.caption("Random Forest classifier · Local model")
    threshold = st.slider("Fraud alert threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                          help="Lower the threshold to flag more transactions; this may also increase false alerts.")
    st.markdown("---")
    st.markdown("**Required columns**")
    st.code("amount\nuse_chip\ncard_type\nmcc\ncurrent_age\ncredit_score\nlongitude\nlatitude", language="text")
    st.caption("Upload a CSV or Excel file. Data is processed by this local app; do not upload sensitive data to an untrusted deployment.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Risk analytics · Live scoring</div>
  <div class="hero-title">See the signal.<br>Stop fraud earlier.</div>
  <div class="hero-sub">Score transactions, surface high-risk activity, and export a review-ready report with your trained Random Forest model.</div>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("Trained model not found. Place `fraud_random_forest.joblib` in the project's `models/` folder.")
    st.code("python -m src.train\nstreamlit run app.py", language="powershell")
    st.stop()

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown("<div class='section-head'>01 / Import transactions</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a transaction file here", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    st.caption("CSV or Excel · One row per transaction · Must include the 8 model features")
with right:
    st.markdown("<div class='section-head'>02 / Quick start</div>", unsafe_allow_html=True)
    if DEFAULT_DATA_PATH.exists():
        st.markdown("<div class='panel'><b>Use your processed dataset</b><br><span style='color:#9aa8c7;font-size:.85rem'>Load the project's processed CSV to test the dashboard.</span></div>", unsafe_allow_html=True)
        use_default = st.button("Load processed dataset", use_container_width=True)
    else:
        st.markdown("<div class='panel'>Upload a new transaction file to start scoring.</div>", unsafe_allow_html=True)
        use_default = False

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
    st.session_state["raw_transactions"] = raw
    st.session_state["source_name"] = source_name
elif use_default and DEFAULT_DATA_PATH.exists():
    st.session_state["raw_transactions"] = pd.read_csv(DEFAULT_DATA_PATH)
    st.session_state["source_name"] = DEFAULT_DATA_PATH.name

if "raw_transactions" not in st.session_state:
    st.markdown("<div class='panel' style='text-align:center;padding:2.2rem 1rem;margin-top:1.1rem'><div style='font-size:2rem'>↥</div><b style='font-size:1.1rem'>Your transaction workspace is ready</b><div style='color:#9aa8c7;margin-top:.45rem'>Upload a file or load the processed dataset to generate fraud scores.</div></div>", unsafe_allow_html=True)
    st.stop()

raw = st.session_state["raw_transactions"]
source_name = st.session_state.get("source_name", "transactions")
missing = [col for col in FEATURES if col not in raw.columns]
if missing:
    st.error("This file is missing required columns: " + ", ".join(missing))
    st.stop()

try:
    features = clean_features(raw)
    probabilities = model.predict_proba(features)[:, 1]
except Exception as exc:
    st.error(f"Could not score these transactions: {exc}")
    st.stop()

scored = raw.copy()
scored["fraud_probability"] = probabilities
scored["predicted_fraud"] = (scored["fraud_probability"] >= threshold).astype(int)
scored["risk_level"] = pd.cut(
    scored["fraud_probability"], bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"], include_lowest=True
).astype(str)
fraud_count = int(scored["predicted_fraud"].sum())
legit_count = len(scored) - fraud_count
fraud_rate = fraud_count / len(scored) if len(scored) else 0
avg_risk = float(scored["fraud_probability"].mean()) if len(scored) else 0
high_risk_count = int((scored["fraud_probability"] >= max(threshold, 0.70)).sum())

st.markdown(f"<div class='section-head'>03 / Risk overview <span style='color:#8495b8;font-size:.82rem;font-family:DM Sans'>· {source_name}</span></div>", unsafe_allow_html=True)
metrics = [
    ("TRANSACTIONS SCORED", f"{len(scored):,}", "Rows processed"),
    ("FLAGGED AS FRAUD", f"{fraud_count:,}", f"{fraud_rate:.1%} at current threshold"),
    ("AVERAGE FRAUD SCORE", f"{avg_risk:.1%}", "Mean predicted probability"),
    ("HIGH-RISK QUEUE", f"{high_risk_count:,}", "Score ≥ max(threshold, 70%)"),
]
cols = st.columns(4, gap="medium")
for col, (label, value, note) in zip(cols, metrics):
    with col:
        st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
chart_left, chart_right = st.columns([1, 1.2], gap="large")
with chart_left:
    st.markdown("<div class='section-head'>Prediction mix</div>", unsafe_allow_html=True)
    pie_df = pd.DataFrame({"Prediction": ["Flagged fraud", "Not flagged"], "Transactions": [fraud_count, legit_count]})
    fig = px.pie(pie_df, names="Prediction", values="Transactions", hole=0.72,
                 color="Prediction", color_discrete_map={"Flagged fraud":"#ff6b82", "Not flagged":"#43d9c5"})
    fig.update_layout(margin=dict(t=8,b=8,l=8,r=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#dce5fa", legend=dict(orientation="h",y=-0.08), height=310)
    fig.add_annotation(text=f"{fraud_rate:.1%}<br><span style='font-size:11px'>flagged</span>", x=0.5,y=0.5,showarrow=False,
                       font=dict(size=23,color="#f4f7ff",family="Space Grotesk"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with chart_right:
    st.markdown("<div class='section-head'>Fraud score distribution</div>", unsafe_allow_html=True)
    hist = px.histogram(scored, x="fraud_probability", nbins=20, color_discrete_sequence=["#6488ff"])
    hist.add_vline(x=threshold, line_dash="dash", line_color="#ff7189", annotation_text=f"Threshold {threshold:.0%}", annotation_position="top right")
    hist.update_layout(margin=dict(t=15,b=15,l=5,r=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#dce5fa", xaxis_title="Predicted fraud probability", yaxis_title="Transactions", height=310,
                       xaxis=dict(gridcolor="rgba(151,170,211,.12)"), yaxis=dict(gridcolor="rgba(151,170,211,.12)"))
    st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div class='section-head'>04 / Transaction explorer</div>", unsafe_allow_html=True)
filter_col, search_col = st.columns([1, 1.5])
with filter_col:
    view = st.selectbox("Show transactions", ["All transactions", "Flagged as fraud", "Not flagged", "High risk (≥ 70%)"])
with search_col:
    search = st.text_input("Filter table", placeholder="Search visible text values…")
view_df = scored.copy()
if view == "Flagged as fraud":
    view_df = view_df[view_df["predicted_fraud"] == 1]
elif view == "Not flagged":
    view_df = view_df[view_df["predicted_fraud"] == 0]
elif view == "High risk (≥ 70%)":
    view_df = view_df[view_df["fraud_probability"] >= 0.70]
if search.strip():
    mask = view_df.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
    view_df = view_df[mask]

show_cols = [c for c in ["amount", "use_chip", "card_type", "mcc", "current_age", "credit_score", "fraud_probability", "predicted_fraud", "risk_level"] if c in view_df.columns]
st.dataframe(
    view_df[show_cols].rename(columns={"fraud_probability":"Fraud score", "predicted_fraud":"Flagged", "risk_level":"Risk level"}),
    use_container_width=True, hide_index=True, height=360,
    column_config={
        "Fraud score": st.column_config.ProgressColumn("Fraud score", min_value=0, max_value=1, format="%.1%%"),
        "Flagged": st.column_config.NumberColumn("Flagged", help="1 = flagged for review; 0 = not flagged"),
        "Risk level": st.column_config.TextColumn("Risk level"),
    },
)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button("⬇  Download all scored transactions (CSV)", data=csv_bytes,
                   file_name="fraud_predictions.csv", mime="text/csv", type="primary")
st.caption("Decision support only: flagged transactions require review. The threshold changes the number of alerts, and this demo model should not be used as the sole basis for real financial decisions.")
