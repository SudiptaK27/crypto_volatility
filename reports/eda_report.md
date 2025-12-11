# Exploratory Data Analysis (EDA) Report

**1. Introduction**

The purpose of this exploratory analysis is to understand the historical behavior of multiple cryptocurrencies, focusing on price patterns, volatility, market dynamics, and relationships between assets. This EDA helps identify structural characteristics of the data, highlight trends, detect anomalies, and provide insights that guide feature engineering and predictive modeling for volatility forecasting.

**2. Dataset Overview**

The cleaned dataset contains 72,777 rows and 9 columns, representing daily price and market data for multiple cryptocurrencies.
Core variables include:

date – trading date

symbol – cryptocurrency name

open, high, low, close – daily price levels

volume – trading activity

market_cap – estimated total valuation

The dataset spans multiple years, though coverage varies across assets. Bitcoin and Ethereum have longer historical series, while newer altcoins appear over shorter periods.

**3. Data Cleaning Summary**

Several preprocessing steps were performed before EDA:

Standardized column names and removed the unused index column.

Renamed crypto_name to symbol to ensure compatibility with analysis code.

Converted dates to a proper datetime format and ensured numerical fields were numeric.

Handled missing data by reindexing each asset to a full daily timeline and filling small gaps using forward and backward fill.

Identified and dropped cryptocurrencies with excessive missing data (>20% missing days or >90 consecutive missing days).

A total of 16 symbols were removed due to poor data quality, leaving a clean dataset suitable for analysis and modeling.

**4. Price Trend Analysis**

Visual inspection of price histories reveals distinct market cycles:

Bitcoin and Ethereum show long-term upward trends punctuated by multiple boom-and-bust cycles.

Prices exhibit clear volatility clusters, where large movements are concentrated during specific market events (e.g., 2017 boom, 2021 bull run, major crashes).

Altcoins follow similar directional trends but often display higher amplitude swings and shorter historical coverage.

These patterns confirm that cryptocurrency markets are highly dynamic and exhibit behavior typical of speculative assets.

**5. Distribution of Prices**

The distribution of closing prices is strongly right-skewed, meaning:

A few high-value assets (like Bitcoin) dominate the upper range.

Most cryptocurrencies trade at much lower prices.

The wide spread suggests significant inequality in valuation across the market.

This skewness is an important consideration for scaling and modeling.

**6. Daily Returns and Volatility**

Daily percentage returns were computed to study volatility characteristics.

**Key observations:**

The return distribution contains fat tails, indicating extreme price movements occur more frequently than in normal (Gaussian) financial series.

Most daily returns cluster near zero, but there are many large positive and negative outliers.

Rolling 30-day volatility for Bitcoin highlights periods of heightened market uncertainty, especially during major crashes and bull runs.

Overall, cryptocurrencies exhibit high and unstable volatility, consistent with their reputation as risk-heavy assets.

**7. Correlation Between Cryptocurrencies**

A correlation matrix of daily returns reveals:

Bitcoin and Ethereum have consistently strong positive correlation, reflecting shared market sentiment.

Many altcoins also correlate positively with major assets but at varying strengths.

Some smaller or newer coins display weaker relationships, suggesting more independent or speculative behavior.

These correlations imply that external market forces (news, global sentiment, regulations) often drive movements across the entire crypto sector.

**8. Key Insights from EDA**

Price Behavior: Cryptocurrencies show cyclical growth patterns with sharp rises and collapses.

Volatility: Return distributions and rolling volatility confirm that crypto markets are extremely volatile with heavy-tailed risk.

Market Structure: Bitcoin and Ethereum dominate both trend behavior and correlations.

Data Quality: Several coins exhibit incomplete histories, justifying the removal of assets with insufficient data.

**Modeling Implications:**

Volatility modeling will require transformations (returns rather than raw prices).

Feature engineering should include rolling statistics and momentum indicators.

Correlation structure suggests multi-asset forecasting could be beneficial.

**Conclusion**

The EDA provides a comprehensive understanding of cryptocurrency market behavior and prepares the foundation for feature engineering, machine learning modeling, and volatility prediction. The dataset is now well-structured, and the insights gleaned from this analysis directly guide the next stage of the project.