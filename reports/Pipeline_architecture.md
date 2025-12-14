# Pipeline Architecture
## Crypto Volatility Prediction System

---

## 1. Purpose of This Document

This document explains the **end-to-end data and model pipeline** of the Crypto Volatility Prediction System.

It focuses on:
- How data flows through the system
- How each stage transforms the data
- How outputs from one stage become inputs to the next

This sits **between HLD (what the system is)** and **LLD (how each part is coded)**.

---

## 2. High-Level Pipeline Overview

Raw Market Data
↓
Data Cleaning & Validation
↓
Feature Engineering
↓
Train / Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Persistence
↓
Inference / Demo (Streamlit)


---

## 3. Stage-by-Stage Pipeline Description

---

### 3.1 Raw Data Ingestion

**Input Source**
- Historical cryptocurrency market data (CSV)

**Fields Include**
- Open, High, Low, Close prices
- Volume
- Market capitalization
- Timestamp
- Cryptocurrency name

**Output**
- Raw dataset loaded into memory

---

### 3.2 Data Cleaning & Validation

**Module**
- `data_prep.py`

**Operations**
- Parse timestamps and dates
- Remove duplicate records
- Validate required columns
- Identify missing or inconsistent records
- Drop crypto symbols with excessive missing history

**Why This Stage Exists**
- Prevents corrupted or inconsistent data from entering the ML pipeline
- Ensures uniform time-series structure per crypto asset

**Output**
- `cleaned_crypto.csv`

---

### 3.3 Feature Engineering

**Module**
- `features.py`

**Purpose**
Transform raw price data into **numerical signals** suitable for machine learning.

**Feature Categories**
- **Returns**: daily returns, log returns
- **Volatility**: rolling standard deviation (7, 14, 30 days)
- **Trend Indicators**: moving averages (SMA, EMA)
- **Momentum Indicators**: RSI, MACD
- **Liquidity Metrics**: volume-to-market-cap ratio
- **Price Bands**: Bollinger upper/lower bands

**Processing Rules**
- Calculated per cryptocurrency symbol
- Rolling windows applied independently
- Rows with insufficient lookback dropped

**Output**
- `features.csv`

---

### 3.4 Feature-Target Separation

**Module**
- `model.py`

**Target Variable**
- `volatility_30d` (30-day rolling volatility)

**Feature Matrix (X)**
- All numerical engineered features
- Excludes identifiers and timestamps

**Target Vector (y)**
- One-dimensional volatility value per row

**Data Validation**
- Remove NaN and infinite values
- Ensure X and y alignment

---

### 3.5 Train-Test Split

**Strategy**
- Random 80:20 split

**Reason**
- Baseline modeling approach
- Simple and fast evaluation
- Avoids data leakage after cleaning

---

### 3.6 Model Training

**Models Used**
- RandomForestRegressor
- XGBoostRegressor

**Training Logic**
- Fit models on training data
- Learn non-linear relationships between features and volatility

**Why Two Models**
- Random Forest: strong baseline, interpretable feature importance
- XGBoost: higher performance, gradient boosting efficiency

---

### 3.7 Model Evaluation

**Metrics Computed**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Purpose**
- Quantify prediction accuracy
- Compare baseline model performance
- Select better-performing model

**Artifacts Generated**
- `model_metrics.csv`
- `feature_importances.csv`

---

### 3.8 Model Persistence

**Storage Location**
- `/models/`

**Formats**
- Random Forest → `.joblib`
- XGBoost → `.json` (native) or `.joblib` fallback

**Why Persistence Is Important**
- Avoid retraining
- Enable deployment and reuse
- Maintain reproducibility

---

### 3.9 Inference & Deployment Pipeline

**Module**
- `app_streamlit.py`

**Flow**
1. Load saved models
2. Load feature dataset
3. User selects cryptocurrency
4. Latest feature row extracted
5. Features aligned with training schema
6. Models generate volatility predictions
7. Results displayed in UI

**Deployment Type**
- Streamlit-based interactive demo

---

## 4. Pipeline Error Handling Strategy

- Missing columns → explicit exceptions
- NaN or infinite values → filtered during preprocessing
- Feature mismatch → enforced alignment
- Missing models → fallback loading strategy

---

## 5. Pipeline Design Characteristics

- Modular and decoupled stages
- Reproducible outputs
- Easily testable components
- Extensible for future enhancements (LSTM, live data, APIs)

---

## 6. Future Pipeline Extensions

- Time-aware train-test splits
- Cross-validation
- SHAP-based explainability
- Real-time ingestion via APIs
- Database-backed storage (MongoDB)

---

## 7. Summary

The pipeline architecture ensures a **clean, logical, and reliable flow** from raw crypto market data to actionable volatility predictions.  
Each stage has a clearly defined role, enabling maintainability, scalability, and correctness.

---
