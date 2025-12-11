# (put in src/data_prep.py or a notebook cell)
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()            # if running inside project root
RAW = ROOT / "data" / "raw_crypto.csv"
OUT = ROOT / "data" / "cleaned_crypto.csv"

# ---------- Parameters you can tweak ----------
MAX_MISSING_DAYS_RATIO = 0.20   # if >20% days missing for a symbol, flag/drop
MAX_TOTAL_GAP_DAYS = 90         # if a symbol has large gaps (e.g., >90 days continuous), flag/drop
FILL_METHOD = "ffill"           # 'ffill' then 'bfill' for small gaps
RESAMPLE_RULE = "D"             # daily frequency
# ---------------------------------------------

def load_csv(path):
    print("Loading:", path)
    df = pd.read_csv(path, low_memory=False)
    print("Raw rows,cols:", df.shape)
    return df

def normalize_columns(df):
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))

    # common-aliases
    df = df.rename(columns={
        "marketcap": "market_cap",
        "market_capitalization": "market_cap",
        "vol": "volume"
    })

    # your dataset uses crypto_name instead of symbol
    if "crypto_name" in df.columns:
        df = df.rename(columns={"crypto_name": "symbol"})

    # drop useless unnamed index column
    if "unnamed:_0" in df.columns:
        df = df.drop(columns=["unnamed:_0"])

    return df

def basic_type_and_preview(df):
    print(df.dtypes)
    print(df.columns.tolist())
    display_sample = df.head(3)
    print("Sample:\n", display_sample)
    print("Missing per column:\n", df.isna().sum())

def require_columns_check(df, required=["date","symbol","close"]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from CSV: {missing}")

def enforce_types(df):
    # date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ensure symbol is string
    df["symbol"] = df["symbol"].astype(str)
    # numeric columns
    for col in ["open","high","low","close","volume","market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def drop_and_report_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(
        subset=["symbol","date"] + 
        ([c for c in ["open","high","low","close","volume","market_cap"] if c in df.columns]),
        keep="first"
    )
    after = len(df)
    print(f"Dropped duplicates: {before-after}")
    return df

def per_symbol_reindex_and_fill(df):
    out_frames = []
    symbol_report = {}

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").set_index("date")

        # full daily range
        full_idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq=RESAMPLE_RULE)
        g = g.reindex(full_idx)
        g["symbol"] = sym

        missing_days = g["close"].isna().sum()
        total_days = len(g)
        missing_ratio = missing_days / total_days if total_days > 0 else 1.0

        is_na = g["close"].isna().astype(int)
        max_gap = (
            is_na * (is_na.groupby((is_na != is_na.shift()).cumsum()).cumcount() + 1)
        ).max() if total_days > 0 else total_days

        symbol_report[sym] = {
            "start": full_idx.min(),
            "end": full_idx.max(),
            "total_days": total_days,
            "missing_days": int(missing_days),
            "missing_ratio": missing_ratio,
            "max_gap_days": int(max_gap)
        }

        fill_cols = [c for c in ["open","high","low","close","volume","market_cap"] if c in g.columns]
        if fill_cols:
            g[fill_cols] = g[fill_cols].ffill().bfill()

        out_frames.append(g.reset_index().rename(columns={"index":"date"}))

    combined = pd.concat(out_frames, ignore_index=True, sort=False)
    report_df = pd.DataFrame.from_dict(symbol_report, orient="index")
    return combined, report_df

def drop_problematic_symbols(df_combined, report_df):
    to_drop = report_df[
        (report_df["missing_ratio"] > MAX_MISSING_DAYS_RATIO) |
        (report_df["max_gap_days"] > MAX_TOTAL_GAP_DAYS)
    ].index.tolist()

    if to_drop:
        print("Dropping symbols due to excessive missing data:", to_drop)

    df_cleaned = df_combined[~df_combined["symbol"].isin(to_drop)].copy()
    return df_cleaned, to_drop

def final_housekeeping(df):
    df = df.dropna(subset=["date","symbol","close"])
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    return df

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Put your raw CSV at: {RAW}")

    df = load_csv(RAW)
    df = normalize_columns(df)
    basic_type_and_preview(df)
    require_columns_check(df)
    df = enforce_types(df)
    df = drop_and_report_duplicates(df)

    combined, report_df = per_symbol_reindex_and_fill(df)
    print("Per-symbol report (sample):")
    print(report_df.sort_values("missing_ratio", ascending=False).head(10))

    df_cleaned, dropped = drop_problematic_symbols(combined, report_df)
    df_cleaned = final_housekeeping(df_cleaned)

    print("After cleaning shape:", df_cleaned.shape)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(OUT, index=False)

    report_df.to_csv(ROOT / "reports" / "symbol_missing_report.csv")

    print("Saved cleaned CSV to:", OUT)
    print("Saved symbol_missing_report to: reports/symbol_missing_report.csv")

    if dropped:
        print("Dropped symbols:", dropped)

if __name__ == "__main__":
    main()
