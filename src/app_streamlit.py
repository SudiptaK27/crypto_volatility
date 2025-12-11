# src/app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path.cwd()
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "xgb_model.pkl"
IMPORTANCES_PATH = ROOT / "reports" / "feature_importances.csv"

st.set_page_config(page_title="Crypto Volatility Demo", layout="wide")

@st.cache_data(show_spinner=False)
def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_importances():
    if IMPORTANCES_PATH.exists():
        return pd.read_csv(IMPORTANCES_PATH)
    return None

def predict_for_row(model, row_df, feature_cols):
    X = row_df[feature_cols]
    pred = model.predict(X)[0]
    return float(pred)

def main():
    st.title("Crypto Volatility Forecast — Demo")
    st.markdown(
        "Simple demo: predict **30-day rolling volatility** using the trained XGBoost model."
    )

    # Load assets
    df = load_features()
    model = load_model()
    imp_df = load_importances()

    if model is None:
        st.warning(f"Model not found at: {MODEL_PATH}. Train the model and place it there.")
        st.stop()

    symbols = sorted(df["symbol"].unique())
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select inputs")
        symbol = st.selectbox("Symbol", symbols, index=symbols.index("Bitcoin") if "Bitcoin" in symbols else 0)
        available_dates = df[df["symbol"] == symbol]["date"].sort_values().dt.date.unique()
        date = st.selectbox("Date", available_dates[::-1])  # latest first

        st.write("")  # spacing
        if st.button("Predict for chosen date"):
            # find the exact row (date as datetime)
            date_str = pd.to_datetime(str(date)).strftime("%Y-%m-%d")
            row = df[(df["symbol"] == symbol) & (df["date"].dt.strftime("%Y-%m-%d") == date_str)]
            if row.empty:
                st.error("No data row found for that symbol/date.")
            else:
                # determine feature columns (all numeric columns except target)
                drop_cols = {"symbol", "date", "volatility_30d"}
                feature_cols = [c for c in row.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(row[c])]
                # align single-row DF
                pred = predict_for_row(model, row, feature_cols)
                st.success(f"Predicted 30-day volatility: **{pred:.6f}**")
                # show actual if available
                actual = row["volatility_30d"].iloc[0]
                st.info(f"Actual 30-day volatility (from features file): **{actual:.6f}**")
                # show the features used
                st.subheader("Feature snapshot (selected row)")
                # show only top features to keep UI tidy
                display_cols = feature_cols[:20]
                st.dataframe(row[display_cols].T, use_container_width=True)

                # allow download
                result = row[display_cols].copy()
                result["predicted_volatility_30d"] = pred
                csv_bytes = result.to_csv(index=False).encode()
                st.download_button("Download prediction CSV", csv_bytes, file_name=f"{symbol}_{date_str}_prediction.csv")

    with col2:
        st.subheader("Recent Price & Volatility history")
        # prepare time series for symbol
        sym_df = df[df["symbol"] == symbol].sort_values("date")
        # price chart
        fig_price = px.line(sym_df, x="date", y="close", title=f"{symbol} — Close Price", labels={"close":"Close"})
        fig_price.update_layout(margin=dict(t=40,l=0,r=0,b=0))
        st.plotly_chart(fig_price, use_container_width=True)

        # volatility overlays
        vol_cols = [c for c in ["volatility_7d","volatility_14d","volatility_30d"] if c in sym_df.columns]
        if vol_cols:
            fig_vol = go.Figure()
            for c in vol_cols:
                fig_vol.add_trace(go.Scatter(x=sym_df["date"], y=sym_df[c], mode="lines", name=c))
            fig_vol.update_layout(title=f"{symbol} — Rolling Volatility", yaxis_title="Volatility", margin=dict(t=40,l=0,r=0,b=0))
            st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("Model & Feature Insights")
    st.write(f"Model file loaded from: `{MODEL_PATH}`")
    if imp_df is not None:
        st.markdown("**Top features (sample):**")
        # show top 10 by RF importance if present
        if {"feature","rf_importance"}.issubset(set(imp_df.columns)):
            st.dataframe(imp_df.sort_values("rf_importance", ascending=False).head(10).reset_index(drop=True))
        else:
            st.dataframe(imp_df.head(10))
    else:
        st.info("Feature importances not found.")

    st.markdown("### Notes")
    st.markdown(
        "- This demo uses the precomputed feature row from `data/features.csv`. The model expects the exact same preprocessing/columns as used during training.\n"
        "- If prediction fails, ensure your `models/xgb_model.pkl` and `data/features.csv` are present and consistent with the training pipeline."
    )

if __name__ == "__main__":
    main()
