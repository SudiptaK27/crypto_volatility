The raw dataset contained 72,946 records across 10 features.
Column names were normalized, the unused index column removed, and ‘crypto_name’ was renamed to ‘symbol’.
Missing-date gaps were reindexed for each asset at daily frequency and forward/backward filled for small gaps.
Assets with more than 20% missing days or continuous gaps exceeding 90 days were excluded, resulting in the removal of 16 symbols.
The final cleaned dataset contains 72,777 rows across 9 features.

