# src/model.py
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------- Paths (robust) -----------------
THIS = Path(__file__).resolve()
# If file is inside /src, project root is parent of parent; else fallback to cwd
if THIS.parts[-2] == "src":
    ROOT = THIS.parents[1]
else:
    ROOT = Path.cwd()

FEATURES = ROOT / "data" / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- Data loading / prep -----------------
def load_data():
    df = pd.read_csv(FEATURES, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"])
    return df


def prepare_xy(df, target="volatility_30d", clip_extreme=False, clip_value=1e6):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in features file.")

    y = df[target].copy()

    # drop obvious non-features
    X = df.drop(columns=[target, "symbol"], errors="ignore")
    # drop 'date' if present
    if "date" in X.columns:
        X = X.drop(columns=["date"])

    # drop any remaining object-like columns (strings)
    non_numeric_cols = X.select_dtypes(include=["object", "category", "datetime"]).columns.tolist()
    if non_numeric_cols:
        print("Dropping non-numeric columns before training:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    # keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # Replace infinities with NaN and diagnose
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = y.replace([np.inf, -np.inf], np.nan)

    # Report NaNs per column (if any)
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print("Columns with NaN counts (before row drop):")
        print(nan_cols.sort_values(ascending=False).head(20))

    # Optionally clip extreme numeric outliers
    if clip_extreme:
        X = X.clip(lower=-clip_value, upper=clip_value)

    # Align X and y: drop rows where either X or y are NaN or non-finite
    finite_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1) & y.notna() & np.isfinite(y)

    dropped_rows = len(X) - finite_mask.sum()
    print(f"Rows before cleaning: {len(X)}. Rows to be dropped due to NaN/Inf: {dropped_rows}")

    X_clean = X[finite_mask].copy()
    y_clean = y[finite_mask].copy()

    print("After cleaning: X shape:", X_clean.shape, "y shape:", y_clean.shape)
    return X_clean, y_clean


# ----------------- Models -----------------
def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    print("Training RandomForest...")
    rf.fit(X_train, y_train)
    t1 = time.time()
    print(f"RandomForest training time: {t1 - t0:.1f}s")

    preds = rf.predict(X_test)

    return {
        "model": rf,
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "time_s": t1 - t0
    }


def run_xgboost(X_train, X_test, y_train, y_test):
    xg_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    print("Training XGBoost...")
    xg_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    t1 = time.time()
    print(f"XGBoost training time: {t1 - t0:.1f}s")

    preds = xg_model.predict(X_test)

    return {
        "model": xg_model,
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "time_s": t1 - t0
    }


# ----------------- Save artifacts -----------------
def save_models_and_metrics(rf_res, xgb_res, X_columns):
    # Save models robustly:
    rf_joblib_path = MODELS_DIR / "rf_model.joblib"
    xgb_json_path = MODELS_DIR / "xgb_model.json"
    xgb_joblib_path = MODELS_DIR / "xgb_model.joblib"  # fallback if JSON save fails

    # RandomForest -> joblib (recommended)
    joblib.dump(rf_res["model"], rf_joblib_path)
    print("Saved RF model (joblib) to:", rf_joblib_path)

    # XGBoost -> native JSON using sklearn API .save_model()
    try:
        # if it's an instance of XGBRegressor from sklearn API:
        if hasattr(xgb_res["model"], "save_model"):
            xgb_res["model"].save_model(str(xgb_json_path))
            print("Saved XGBoost model (JSON) to:", xgb_json_path)
        else:
            # fallback to joblib
            joblib.dump(xgb_res["model"], xgb_joblib_path)
            print("Saved XGBoost model (joblib fallback) to:", xgb_joblib_path)
    except Exception as e:
        print("XGBoost JSON save failed, falling back to joblib. Error:", e)
        joblib.dump(xgb_res["model"], xgb_joblib_path)
        print("Saved XGBoost model (joblib) to:", xgb_joblib_path)

    # Save metrics CSV
    metrics = pd.DataFrame([
        {
            "model": "RandomForest",
            "rmse": rf_res["rmse"],
            "mae": rf_res["mae"],
            "r2": rf_res["r2"],
            "train_time_s": rf_res.get("time_s", np.nan)
        },
        {
            "model": "XGBoost",
            "rmse": xgb_res["rmse"],
            "mae": xgb_res["mae"],
            "r2": xgb_res["r2"],
            "train_time_s": xgb_res.get("time_s", np.nan)
        }
    ])
    metrics_path = REPORTS_DIR / "model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("Saved metrics to:", metrics_path)

    # Feature importances (attempt both rf and xgb)
    try:
        feats = list(X_columns)
        rf_imp = getattr(rf_res["model"], "feature_importances_", None)
        xgb_imp = getattr(xgb_res["model"], "feature_importances_", None)

        imp_df = pd.DataFrame({"feature": feats})

        if rf_imp is not None and len(rf_imp) == len(feats):
            imp_df["rf_importance"] = rf_imp
        else:
            imp_df["rf_importance"] = np.nan
            print("Warning: RF importances missing or length mismatch; filling NaN.")

        if xgb_imp is not None and len(xgb_imp) == len(feats):
            imp_df["xgb_importance"] = xgb_imp
        else:
            imp_df["xgb_importance"] = np.nan
            print("Warning: XGB importances missing or length mismatch; filling NaN.")

        # sort by RF importance for display (fallback behavior)
        print("\nTop 10 features by RandomForest importance:")
        print(imp_df.sort_values("rf_importance", ascending=False).head(10).to_string(index=False))

        print("\nTop 10 features by XGBoost importance:")
        print(imp_df.sort_values("xgb_importance", ascending=False).head(10).to_string(index=False))

        imp_df.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)
        print("Saved feature importances to:", REPORTS_DIR / "feature_importances.csv")
    except Exception as e:
        print("Could not compute/save feature importances:", e)


# ----------------- Main -----------------
def main():
    df = load_data()
    X, y = prepare_xy(df, clip_extreme=False)

    if X.shape[0] < 10:
        raise RuntimeError("Not enough rows after cleaning to train a model.")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Train
    rf_results = run_random_forest(X_train, X_test, y_train, y_test)
    xgb_results = run_xgboost(X_train, X_test, y_train, y_test)

    # Print nice summary
    print("\n===== BASELINE MODEL RESULTS =====")
    print("\nRandom Forest:")
    print(f"  RMSE: {rf_results['rmse']:.6f}")
    print(f"  MAE : {rf_results['mae']:.6f}")
    print(f"  R2  : {rf_results['r2']:.4f}")
    print(f"  Train time (s): {rf_results.get('time_s', np.nan):.1f}")

    print("\nXGBoost:")
    print(f"  RMSE: {xgb_results['rmse']:.6f}")
    print(f"  MAE : {xgb_results['mae']:.6f}")
    print(f"  R2  : {xgb_results['r2']:.4f}")
    print(f"  Train time (s): {xgb_results.get('time_s', np.nan):.1f}")

    # Save models + metrics + importances
    save_models_and_metrics(rf_results, xgb_results, X.columns)

    print("\nAll done. Models and metrics saved in 'models/' and 'reports/' folders.")


if __name__ == "__main__":
    main()
