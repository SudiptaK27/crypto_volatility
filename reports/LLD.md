# Low-Level Design (LLD)
## Crypto Volatility Prediction System

---

## 1. Introduction

This document describes the **low-level design** of the Crypto Volatility Prediction System.  
It explains **how each component is implemented**, including data processing logic, feature engineering methods, model training workflow, evaluation strategy, and deployment setup.

The LLD focuses on **implementation details**, file responsibilities, and data transformations.

---

## 2. Project Directory Structure

crypto-volatility/
│
├── data/
│ ├── raw_crypto.csv
│ ├── cleaned_crypto.csv
│ └── features.csv
│
├── src/
│ ├── data_prep.py
│ ├── features.py
│ ├── model.py
│ ├── app_streamlit.py
│ └── evaluate.py
│
├── models/
│ ├── rf_model.joblib
│ ├── xgb_model.joblib
│ └── xgb_model.json
│
├── reports/
│ ├── HLD.md
│ ├── LLD.md
│ ├── model_metrics.csv
│ └── feature_importances.csv
│
├── tests/
│ ├── test_data.py
│ ├── test_features.py
│ ├── test_models.py
│ └── test_streamlit_smoke.py
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 3. Module-Level Design

### 3.1 `data_prep.py` – Data Cleaning Module

**Responsibility:**
- Load raw crypto market data
- Validate schema and required columns
- Handle missing values and duplicates
- Generate a clean dataset for downstream processing

**Key Operations:**
- Parse dates and timestamps
- Drop duplicate rows
- Remove symbols with excessive missing data
- Save cleaned dataset to `cleaned_crypto.csv`

**Output:**
- Cleaned and validated CSV file

---

### 3.2 `features.py` – Feature Engineering Module

**Responsibility:**
- Convert cleaned data into ML-ready features
- Generate volatility and technical indicators

**Features Implemented:**
- Returns and log returns
- Rolling volatility (7d, 14d, 30d)
- Moving averages (SMA, EMA)
- Liquidity ratio
- RSI
- MACD
- Bollinger Bands

**Key Design Notes:**
- Group-wise calculations by crypto symbol
- Rolling windows applied per asset
- Rows with insufficient history removed

**Output:**
- `features.csv` with numerical ML features

---

### 3.3 `model.py` – Model Training Module

**Responsibility:**
- Load engineered features
- Prepare feature matrix (X) and target variable (y)
- Train baseline ML models
- Save models and metrics

**Target Variable:**
- `volatility_30d`

**Models Used:**
- RandomForestRegressor
- XGBoostRegressor

**Processing Steps:**
1. Drop non-numeric columns
2. Replace infinite values
3. Align X and y
4. Train-test split
5. Train models
6. Evaluate using RMSE, MAE, R²
7. Save models and reports

**Output:**
- Trained models (`.joblib`, `.json`)
- Metrics report (`model_metrics.csv`)
- Feature importance report

---

### 3.4 `evaluate.py` – Model Evaluation Module

**Responsibility:**
- Load trained models
- Evaluate predictions on test data
- Generate comparison metrics
- Create optional plots (actual vs predicted)

**Evaluation Metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**Purpose:**
- Separate evaluation logic from training
- Enable independent testing and analysis

---

### 3.5 `app_streamlit.py` – Deployment & Demo Module

**Responsibility:**
- Load trained models
- Accept user input (crypto symbol)
- Generate volatility predictions
- Display results interactively

**Key Features:**
- Cached data and model loading
- Feature alignment with training
- Error handling for missing models
- Clean UI for demonstration

**Deployment Type:**
- Local Streamlit app (demo-focused)

---

## 4. Testing Layer

### Testing Framework
- **Pytest**

### Test Coverage
- Data loading and schema validation
- Feature generation correctness
- Model prediction shape and stability
- Streamlit app smoke test

### Purpose
- Ensure pipeline reliability
- Catch regression bugs early
- Validate reproducibility

---

## 5. Data Flow at Code Level

1. `data_prep.py` → produces `cleaned_crypto.csv`
2. `features.py` → produces `features.csv`
3. `model.py` → trains models, saves artifacts
4. `evaluate.py` → validates performance
5. `app_streamlit.py` → serves predictions

---

## 6. Error Handling & Validation

- Missing column checks
- NaN and infinity filtering
- Feature consistency enforcement
- Graceful model loading fallbacks

---

## 7. Design Principles Followed

- **Single Responsibility Principle**
- **Modular architecture**
- **Reproducibility**
- **Separation of training and inference**
- **Test-driven validation**

---

## 8. Conclusion

This Low-Level Design defines how each system component is implemented and interacts at the code level.  
The design ensures clarity, maintainability, and correctness while supporting future scalability and enhancements.

---
