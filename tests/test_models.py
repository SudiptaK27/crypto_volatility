#Model loading & prediction test

import pandas as pd
import numpy as np   # ← THIS WAS MISSING
import joblib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

def test_model_prediction_shape():
    df = pd.read_csv(ROOT / "data" / "features.csv")

    X = df.select_dtypes("number").drop(columns=["volatility_30d"], errors="ignore")

    # Apply same cleaning as training
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna().iloc[:5]

    rf = joblib.load(ROOT / "models" / "rf_model.joblib")
    preds = rf.predict(X)

    assert len(preds) == len(X)

