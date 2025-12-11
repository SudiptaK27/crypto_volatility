# 📘 Model Training Report

**1. Objective**

The goal was to build predictive models capable of estimating future 30-day volatility of cryptocurrencies using engineered financial and technical features. Two baseline models were trained and compared:

1. Random Forest Regressor

2. XGBoost Regressor

Both are robust, non-linear models well-suited for time-series feature sets.

**2. Dataset Used for Training**

The training data came from the output of the feature engineering pipeline (features.csv).
This dataset contained:

-Cleaned OHLCV price data

-Returns (simple and log)

-Rolling volatility windows

-Moving averages (MA)

-Exponential moving averages (EMA)

-Liquidity ratio

-RSI

-MACD & Signal line

-Bollinger Bands upper/lower

After feature generation, the dataset contained:

-61,089 rows

-23 numerical features

-Target variable: volatility_30d

Before training, the following non-numeric or invalid columns were removed:

-timestamp

-Rows containing NaN or infinite values (mostly from liquidity_ratio or early rolling windows)

**3. Train-Test Split**

A standard 80/20 split was used:

-Training set: 80% of data

-Test set: 20%

The split preserved randomness but not time order, because the models are not sequence-based (they treat each row independently).

**4. Models Trained**
**4.1 Random Forest Regressor**

A RandomForest builds an ensemble of decision trees and averages their predictions.
It is useful because:

-Handles non-linear relationships

-Does not require feature scaling

-Robust to outliers

Training configuration:

n_estimators = 300

max_depth = 12

min_samples_split = 5

random_state = 42

Training time: 27.4 seconds

**4.2 XGBoost Regressor**

XGBoost is a gradient boosting algorithm known for high accuracy and efficiency.

Reasons for using XGBoost:

Learns complex patterns

Handles uneven feature importance well

Generally outperforms bagging models like RandomForest

Training configuration:

n_estimators = 300

learning_rate = 0.05

max_depth = 6

subsample = 0.8

colsample_bytree = 0.8

Training time: 8.2 seconds

**5. Model Evaluation Metrics**

For evaluation, three standard regression metrics were used:

RMSE (Root Mean Square Error): penalizes large errors

MAE (Mean Absolute Error): average prediction error

R² Score: how much variance is explained by the model

**6. Results Summary**
Performance Comparison Table
Model        	RMSE	    MAE	       R² Score     	Train Time
Random Forest	0.018838	0.010312	0.8105	           27.4 s
XGBoost     	0.013273	0.007143	0.9059	            8.2 s

**Interpretation**

XGBoost clearly outperformed Random Forest in all metrics.

It achieved:

-Lower RMSE → fewer large prediction errors

-Lower MAE → better average prediction accuracy

-Higher R² → explains more variance in volatility

-3× faster training time

Therefore, XGBoost is the preferred model for deployment.

**7. Feature Importance Analysis**
**Top 10 features (Random Forest)**

The model considered these most important:

1. volatility_14d

2. market_cap

3. rsi

4. macd_signal

5. volume

6. liquidity_ratio

7. volatility_7d

8. macd

9. bollinger_lower

10. volume_change

**Top 10 features (XGBoost)**

**Stronger emphasis on volatility and trend signals:**

1. volatility_14d

2. volatility_7d

3. bollinger_lower

4. ema_14d

5. bollinger_upper

6. macd_signal

7. ema_30d

8. ma_7d

9. ma_14d

10. market_cap

**Insight:**

The models rely heavily on recent volatility, trend indicators, and Bollinger Bands, which aligns with how real traders assess crypto risk.

**8. Final Deliverables Generated**

The training script produced:

Trained Model Files

    models/rf_model.pkl

    models/xgb_model.pkl

Evaluation Metrics

    reports/model_metrics.csv

Feature Importance Rankings

    reports/feature_importances.csv

These are ready for downstream use in:

    Streamlit dashboards

    Prediction APIs

    Backtesting notebooks