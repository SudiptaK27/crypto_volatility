# src/app_streamlit.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Paths / constants
# -----------------------
THIS = Path(__file__).resolve()
# If repository layout is repo_root/src/app_streamlit.py -> ROOT = parents[1]
ROOT = THIS.parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURES_CSV = DATA_DIR / "features.csv"
RF_JOBLIB = MODELS_DIR / "rf_model.joblib"
XGB_JOBLIB = MODELS_DIR / "xgb_model.joblib"

# -----------------------
# Load data + models
# -----------------------
@st.cache_data(ttl=3600)
def load_data():
    if not FEATURES_CSV.exists():
        st.error(f"Features file not found: {FEATURES_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"], infer_datetime_format=True, low_memory=False)
    return df

@st.cache_resource
def load_models():
    rf = None
    xgb_model = None

    if RF_JOBLIB.exists():
        try:
            rf = joblib.load(RF_JOBLIB)
            st.write(f"Loaded RandomForest from: {RF_JOBLIB.name}")
        except Exception as e:
            st.error(f"Failed to load RandomForest ({RF_JOBLIB.name}): {e}")
    else:
        st.warning(f"RandomForest model file not found at: {RF_JOBLIB}")

    if XGB_JOBLIB.exists():
        try:
            xgb_model = joblib.load(XGB_JOBLIB)
            st.write(f"Loaded XGBoost from: {XGB_JOBLIB.name}")
        except Exception as e:
            st.error(f"Failed to load XGBoost ({XGB_JOBLIB.name}): {e}")
    else:
        st.warning(f"XGBoost model file not found at: {XGB_JOBLIB}")

    return rf, xgb_model

# -----------------------
# App UI
# -----------------------
st.set_page_config(page_title="Crypto Volatility Predictor", layout="wide")
st.title("📈 Crypto Volatility Prediction Demo")
st.write("Predict next-day crypto volatility using trained ML models.")

df = load_data()
rf, xgb_model = load_models()

if df.empty:
    st.stop()

# symbol selector
symbols = sorted(df["symbol"].unique().tolist())
symbol = st.selectbox("Select a cryptocurrency:", symbols)

symbol_df = df[df["symbol"] == symbol].sort_values("date")
if symbol_df.empty:
    st.warning("No data for selected symbol.")
    st.stop()

latest = symbol_df.iloc[[-1]].copy()  # keep as DataFrame (single row)

# show latest features
st.subheader("Latest Available Features")
st.dataframe(latest.reset_index(drop=True))

# -----------------------
# Prepare input & predict
# -----------------------
# Determine canonical model feature names (prefer RF's stored names)
model_feature_names = None
if rf is not None and hasattr(rf, "feature_names_in_"):
    model_feature_names = list(rf.feature_names_in_)
elif xgb_model is not None and hasattr(xgb_model, "feature_names_in_"):
    model_feature_names = list(xgb_model.feature_names_in_)
else:
    # fallback: numeric columns in the dataset excluding common non-features
    model_feature_names = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("volatility_30d", "symbol", "date", "timestamp")
    ]
    st.warning("Could not read feature_names from models; using fallback numeric columns.")

st.write(f"Model expects {len(model_feature_names)} features.")

# Build single-row input aligned to expected features
row = latest.copy()

# Drop timestamp column (models expect numeric only)
if "timestamp" in row.columns:
    row = row.drop(columns=["timestamp"])

# Drop target column if present
if "volatility_30d" in row.columns:
    row = row.drop(columns=["volatility_30d"])

# Ensure each expected column exists; create NaN where missing
for col in model_feature_names:
    if col not in row.columns:
        row[col] = np.nan

# Ensure 'close' present if model expects it — use latest or NaN fallback
if "close" in model_feature_names and "close" not in row.columns:
    try:
        last_close = symbol_df["close"].iloc[-1]
        row["close"] = last_close
    except Exception:
        row["close"] = np.nan

# Select columns in exact order
X_input = row[model_feature_names].copy()

# Coerce to numeric (non-numeric -> NaN)
for c in X_input.columns:
    X_input[c] = pd.to_numeric(X_input[c], errors="coerce")

# Fill NaNs: prefer symbol-specific median, then global median, then 0.0
for c in X_input.columns:
    if X_input[c].isna().any():
        fill_val = None
        if c in symbol_df.columns and pd.api.types.is_numeric_dtype(symbol_df[c]):
            fill_val = symbol_df[c].median()
        if fill_val is None and c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            fill_val = df[c].median()
        if fill_val is None or (pd.isna(fill_val)):
            fill_val = 0.0
        X_input[c] = X_input[c].fillna(fill_val)

# Final dtype check
non_numeric_after = [c for c in X_input.columns if not pd.api.types.is_numeric_dtype(X_input[c])]
if non_numeric_after:
    st.error("Prepared input still contains non-numeric columns: " + ", ".join(non_numeric_after))
else:
    st.info("Prepared input features (sent to models).")
    with st.expander("Show input features sent to model", expanded=False):
        # show transposed table so it's easier to read single-row
        st.dataframe(X_input.T)

# --- Predictions ---
rf_pred = None
xgb_pred = None

# RF predict
if rf is not None:
    try:
        rf_pred = rf.predict(X_input)[0]
    except Exception as e:
        st.error("RandomForest prediction failed: " + str(e))
        # helpful debugging hints
        try:
            expected = list(rf.feature_names_in_)
            provided = list(X_input.columns)
            missing = [c for c in expected if c not in provided]
            extra = [c for c in provided if c not in expected]
            if missing:
                st.write("Missing features required by RF (sample):", missing[:20])
            if extra:
                st.write("Extra features provided (sample):", extra[:20])
        except Exception:
            pass

# XGBoost predict
if xgb_model is not None:
    try:
        xgb_pred = xgb_model.predict(X_input)[0]
    except Exception as e:
        st.error("XGBoost prediction failed: " + str(e))
        # show dtypes for diagnosis
        dtypes = X_input.dtypes.astype(str).to_dict()
        st.write("Input column dtypes sent to XGBoost (sample):")
        st.write({k: dtypes[k] for k in list(dtypes)[:12]})

# Show results
st.subheader("Predicted Volatility")
if rf_pred is not None:
    st.write(f"**RandomForest:** {rf_pred:.5f}")
else:
    st.write("RandomForest: not available")

if xgb_pred is not None:
    st.write(f"**XGBoost:** {xgb_pred:.5f}")
else:
    st.write("XGBoost: not available")

st.markdown("---")
st.caption("Note: These are demo model outputs. Do not use for live trading decisions.")
