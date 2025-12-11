## Feature Engineering Report

**Overview**

After cleaning and exploring the cryptocurrency dataset, a complete feature engineering pipeline was developed to extract meaningful patterns related to trend, momentum, volatility, and liquidity. These features serve as inputs for the volatility prediction model and help quantify the behavior of each asset over time.

**1. Returns and Log Returns**

Daily returns measure how much the closing price changes from one day to the next, while log returns stabilize variance and are preferred in financial modeling.

Formulas (one-line):

return = (close_today - close_yesterday) / close_yesterday

log_return = ln(1 + return)

These features capture the basic movement dynamics that feed into volatility calculations.

**2. Rolling Volatility (7-day, 14-day, 30-day)**

Rolling volatility measures how much returns fluctuate over a fixed period and is one of the key indicators of market risk.

Formula:

volatility_window = standard deviation of returns over the last N days

Created features: volatility_7d, volatility_14d, volatility_30d.

These provide short-, medium-, and long-term views of asset risk.

**3. Moving Averages (MA-7, MA-14, MA-30)**

Moving averages smooth noisy price data and highlight trend direction.

Formula:

ma_window = average closing price over the last N days

These features help the model detect whether prices are trending up or down.

**4. Exponential Moving Averages (EMA-7, EMA-14, EMA-30)**

Exponential moving averages give more weight to recent prices, making them more responsive than simple moving averages.

Formula:

ema_window = exponential weighted moving average over the last N days

EMAs capture more immediate momentum shifts in price movement.

**5. Liquidity Features**

Liquidity reflects how easily an asset can be traded, which affects volatility.

(a) Liquidity Ratio

Formula :

liquidity_ratio = volume / market_cap

(b) Volume Change

Formula :

volume_change = (volume_today - volume_yesterday) / volume_yesterday

These help identify unusual trading activity and liquidity shocks.

**6. Relative Strength Index (RSI)**

RSI measures price momentum by comparing recent gains to recent losses.

Process:

delta = close_today - close_yesterday

avg_gain = average of positive deltas over 14 days

avg_loss = average of negative deltas over 14 days

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

RSI helps detect overbought or oversold conditions that often precede reversals.

**7. MACD (Moving Average Convergence Divergence)**

MACD is a widely used indicator for detecting momentum shifts.

Formulas :

macd = EMA_12day - EMA_26day

macd_signal = 9-day EMA of macd

These features capture trend acceleration or weakening.

**8. Bollinger Bands (Upper & Lower)**

Bollinger Bands quantify volatility around a moving average.

Formulas :

rolling_mean = average close price over last 20 days

rolling_std = standard deviation over last 20 days

bollinger_upper = rolling_mean + 2 * rolling_std

bollinger_lower = rolling_mean - 2 * rolling_std

These help identify price extremes and volatility breakouts.

**9. Final Cleanup**

After all features were generated:

Missing values were removed

Data was re-sorted by each symbol and date

The dataset was saved for modeling

Final feature dataset size:
62,532 rows × 27 columns

**Summary**

The feature engineering stage transformed raw crypto price and volume data into a comprehensive, structured dataset. The engineered features capture trend strength, volatility patterns, price momentum, and liquidity behavior—providing a strong foundation for training a volatility prediction model.

This set of features mirrors the indicators used by quantitative analysts and strengthens the predictive capability of the upcoming machine-learning pipeline.