# Final Report
## Crypto Volatility Prediction System

---

## 1. Introduction

This project focuses on building an end-to-end **machine learning pipeline** to predict cryptocurrency price volatility using historical market data.

Cryptocurrency markets are highly volatile, making risk estimation a critical task for traders, analysts, and financial systems. This project demonstrates how data engineering, feature engineering, and machine learning can be combined to model and predict volatility effectively.

---

## 2. Objectives

The primary objectives of this project were:

- To clean and preprocess raw cryptocurrency market data
- To engineer meaningful financial and statistical features
- To build baseline machine learning models for volatility prediction
- To evaluate and compare model performance
- To deploy a simple interactive demo using Streamlit
- To document the system using HLD, LLD, and Pipeline Architecture

---

## 3. Dataset Overview

**Source**
- Historical cryptocurrency market data (CSV format)

**Key Fields**
- Open, High, Low, Close prices
- Trading volume
- Market capitalization
- Timestamp
- Cryptocurrency symbol

**Time Range**
- Multiple years of daily price data across several cryptocurrencies

---

## 4. Data Preprocessing Summary

The raw dataset underwent multiple preprocessing steps:

- Removal of duplicate records
- Standardization of column names
- Conversion of timestamps to date format
- Validation of required columns
- Detection and handling of missing values
- Dropping assets with excessive missing data

This resulted in a clean, consistent dataset suitable for time-series feature generation.

---

## 5. Feature Engineering Summary

Feature engineering was a critical step to convert raw prices into predictive signals.

**Key Features Engineered**
- Daily returns and log returns
- Rolling volatility (7, 14, 30 days)
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- MACD and MACD signal
- Bollinger Bands (upper and lower)
- Liquidity ratio (volume / market cap)
- Volume percentage change

**Target Variable**
- `volatility_30d` (30-day rolling volatility)

All non-numeric and invalid values were removed before modeling.

---

## 6. Model Training

Two baseline regression models were trained:

### 6.1 Random Forest Regressor
- Ensemble-based model
- Handles non-linear relationships well
- Provides feature importance for interpretability

### 6.2 XGBoost Regressor
- Gradient boosting model
- Faster convergence
- Better performance on structured tabular data

**Train-Test Split**
- 80% training
- 20% testing

---

## 7. Model Evaluation

Models were evaluated using standard regression metrics:

| Model         | RMSE     | MAE      | R² Score |
|---------------|----------|----------|----------|
| Random Forest | ~0.0188  | ~0.0103  | ~0.81    |
| XGBoost       | ~0.0133  | ~0.0071  | ~0.91    |

**Observations**
- XGBoost outperformed Random Forest across all metrics
- Lower RMSE and MAE indicate better prediction accuracy
- Higher R² shows stronger explanatory power

---

## 8. Feature Importance Insights

Top contributing features included:
- Rolling volatility (especially 14-day)
- Market capitalization
- RSI and MACD indicators
- Liquidity ratio
- Bollinger Bands

This confirms that **recent volatility and momentum indicators** are strong predictors of future volatility.

---

## 9. Deployment & Demo

A simple **Streamlit web application** was developed to demonstrate model inference.

**Demo Features**
- User selects a cryptocurrency
- Latest engineered features are displayed
- Both Random Forest and XGBoost predictions are shown
- Predictions are generated using saved models

This demonstrates how trained models can be reused without retraining.

---

## 10. Testing & Validation

Automated tests were written using **pytest** to validate:

- Data loading and cleaning logic
- Feature generation correctness
- Model prediction shape and stability
- Streamlit app startup (smoke test)

All core components passed testing after alignment with training preprocessing logic.

---

## 11. Limitations

- Random train-test split ignores strict time dependency
- No live data ingestion
- No hyperparameter tuning beyond baseline
- Volatility prediction limited to regression approach

---

## 12. Future Enhancements

Possible improvements include:
- Time-series cross-validation
- LSTM or Transformer-based models
- SHAP-based explainability
- Statistical hypothesis testing
- Real-time data ingestion via APIs
- Database-backed storage (MongoDB)
- Production deployment using FastAPI

---

## 13. Conclusion

This project successfully demonstrates a complete **machine learning workflow** for cryptocurrency volatility prediction.

From raw data ingestion to deployment-ready inference, each component was designed to be modular, testable, and extensible. The results show that engineered financial indicators combined with gradient boosting models can effectively predict volatility.

The system is well-positioned for further enhancement and real-world application.

---
