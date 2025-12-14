# src/evaluate.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------- Paths -----------------
THIS = Path(__file__).resolve()
if THIS.parts[-2] == "src":
    ROOT = THIS.parents[1]
else:
    ROOT = Path.cwd()

DATA = ROOT / "data" / "features.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

RF_MODEL = MODELS / "rf_model.joblib"
XGB_MODEL = MODELS / "xgb_model.joblib"

REPORTS.mkdir(exist_ok=True)


# ----------------- SAME PREP AS TRAINING -----------------
def prepare_xy(df, target="volatility_30d"):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")

    y = df[target].copy()

    X = df.drop(columns=[target, "symbol"], errors="ignore")
    if "date" in X.columns:
        X = X.drop(columns=["date"])

    # Drop non-numeric
    X = X.select_dtypes(include=[np.number]).copy()

    # Replace inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN / Inf
    mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1) & y.notna()

    return X[mask], y[mask]


# ----------------- Evaluation -----------------
def evaluate_model(name, model, X, y):
    preds = model.predict(X)

    return {
        "model": name,
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
        "preds": preds
    }


def plot_actual_vs_pred(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )
    plt.xlabel("Actual Volatility")
    plt.ylabel("Predicted Volatility")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.tight_layout()

    out = REPORTS / f"{model_name}_actual_vs_pred.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved plot: {out}")


# ----------------- Main -----------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA, parse_dates=["date"])

    X, y = prepare_xy(df)

    print("Loading models...")
    rf = joblib.load(RF_MODEL)
    xgb = joblib.load(XGB_MODEL)

    rf_res = evaluate_model("RandomForest", rf, X, y)
    xgb_res = evaluate_model("XGBoost", xgb, X, y)

    metrics = pd.DataFrame([
        {k: rf_res[k] for k in ["model", "rmse", "mae", "r2"]},
        {k: xgb_res[k] for k in ["model", "rmse", "mae", "r2"]}
    ])

    metrics_path = REPORTS / "evaluation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\nEvaluation Metrics:")
    print(metrics)

    plot_actual_vs_pred(y, rf_res["preds"], "RandomForest")
    plot_actual_vs_pred(y, xgb_res["preds"], "XGBoost")

    print("\nEvaluation complete. Results saved in reports/.")


if __name__ == "__main__":
    main()
