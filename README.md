# 🚀 Crypto Volatility Prediction System

An end-to-end **machine learning pipeline** for predicting next-day cryptocurrency volatility using historical market data, engineered financial indicators, and regression models. Includes training, evaluation, testing, and a Streamlit demo.

---

## 🔍 What This Project Does
- Cleans and preprocesses raw crypto market data
- Engineers volatility and technical indicators
- Trains baseline ML models (RandomForest, XGBoost)
- Evaluates models with industry-standard metrics
- Deploys predictions via an interactive Streamlit app
- Includes automated testing and full documentation

---

## 🧠 Models Used
- **RandomForest Regressor**
- **XGBoost Regressor**

**Target:** `volatility_30d` (30-day rolling volatility)

---

## 📊 Features Engineered
- Returns & log returns
- Rolling volatility (7d, 14d, 30d)
- Moving averages & EMA
- RSI, MACD, Bollinger Bands
- Liquidity ratio & volume change

---

## 📈 Model Performance (Baseline)

| Model        | RMSE | MAE | R² Score |
|--------------|------|-----|----------|
| RandomForest | Low  | Low | ~0.81    |
| XGBoost      | Lower| Lower | ~0.90+  |

*(Exact values available in `reports/model_metrics.csv`)*

---

## 🧪 Testing
Automated testing with **pytest**:
- Data integrity checks
- Feature validation
- Model prediction sanity tests
- Streamlit app smoke test

---

## 🖥️ Streamlit Demo

Run locally:
```bash
streamlit run src/app_streamlit.py
