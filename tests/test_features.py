# Feature engineering test


import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_features_file_exists():
    assert (ROOT / "data" / "features.csv").exists()

def test_required_features_present():
    df = pd.read_csv(ROOT / "data" / "features.csv")
    required = [
        "volatility_7d",
        "volatility_14d",
        "volatility_30d",
        "rsi",
        "macd",
        "liquidity_ratio"
    ]
    for col in required:
        assert col in df.columns, f"Missing feature: {col}"





