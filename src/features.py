import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd()
CLEAN = ROOT / "data" / "cleaned_crypto.csv"
OUT = ROOT / "data" / "features.csv"

def load_data():
    df = pd.read_csv(CLEAN, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"])
    return df

def add_returns(df):
    df["return"] = df.groupby("symbol")["close"].pct_change()
    df["log_return"] = np.log1p(df["return"])
    return df

def add_rolling_volatility(df):
    for win in [7, 14, 30]:
        df[f"volatility_{win}d"] = (
            df.groupby("symbol")["return"]
              .rolling(win)
              .std()
              .reset_index(level=0, drop=True)
        )
    return df

def add_moving_averages(df):
    for win in [7, 14, 30]:
        df[f"ma_{win}d"] = (
            df.groupby("symbol")["close"]
              .rolling(win)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# ✅ FIXED: EMA uses transform() — no MultiIndex errors
def add_ema(df):
    for span in [7, 14, 30]:
        df[f"ema_{span}d"] = (
            df.groupby("symbol")["close"]
              .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )
    return df

def add_liquidity_features(df):
    df["liquidity_ratio"] = df["volume"] / df["market_cap"]
    df["volume_change"] = df.groupby("symbol")["volume"].pct_change()
    return df

def add_rsi(df, period=14):
    delta = df.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.groupby(df["symbol"])
            .rolling(period)
            .mean()
            .reset_index(level=0, drop=True)
    )
    avg_loss = (
        loss.groupby(df["symbol"])
            .rolling(period)
            .mean()
            .reset_index(level=0, drop=True)
    )

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

# ✅ FIXED MACD — uses transform() to avoid index mismatch
def add_macd(df):
    df["ema12"] = df.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df["ema26"] = df.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )

    df["macd"] = df["ema12"] - df["ema26"]

    df["macd_signal"] = df.groupby("symbol")["macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )

    # Optional: remove temporary columns
    df.drop(columns=["ema12", "ema26"], inplace=True)

    return df

def add_bollinger(df, window=20):
    rolling_mean = (
        df.groupby("symbol")["close"]
          .rolling(window)
          .mean()
          .reset_index(level=0, drop=True)
    )
    rolling_std = (
        df.groupby("symbol")["close"]
          .rolling(window)
          .std()
          .reset_index(level=0, drop=True)
    )

    df["bollinger_upper"] = rolling_mean + (rolling_std * 2)
    df["bollinger_lower"] = rolling_mean - (rolling_std * 2)
    return df

def final_cleanup(df):
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    df = load_data()
    df = add_returns(df)
    df = add_rolling_volatility(df)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_liquidity_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)

    df = final_cleanup(df)

    OUT.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT, index=False)
    print("Features saved to:", OUT)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    main()
