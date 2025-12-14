#Data loading test


import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_cleaned_data_exists():
    path = ROOT / "data" / "cleaned_crypto.csv"
    assert path.exists(), "cleaned_crypto.csv does not exist"

def test_cleaned_data_loads():
    df = pd.read_csv(ROOT / "data" / "cleaned_crypto.csv")
    assert not df.empty
    assert "close" in df.columns
