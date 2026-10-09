"""
Week 4: Statistical Analysis and Hypothesis Testing
Predictive Analytics — time-series stock-price dataset

Input CSV must contain a date column and a numeric closing-price column.
Edit DATE_COL and VALUE_COL below to match your dataset.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from pathlib import Path

CSV_PATH = "stock_data.csv"   # Place your dataset in the same folder
DATE_COL = "Date"
VALUE_COL = "Close"
ALPHA = 0.05

df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
df = df[[DATE_COL, VALUE_COL]].dropna().sort_values(DATE_COL)
df = df.drop_duplicates(subset=[DATE_COL]).set_index(DATE_COL)
series = df[VALUE_COL].astype(float)

if len(series) < 20:
    raise ValueError("Use at least 20 valid observations for a more reliable analysis.")

print("Observations:", len(series))
print("Date range:", series.index.min().date(), "to", series.index.max().date())
print("Mean closing price:", round(series.mean(), 4))
print("Median closing price:", round(series.median(), 4))
print("Standard deviation:", round(series.std(), 4))

# Create daily/observation-to-observation changes and percentage returns.
change = series.diff().dropna()
returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

# Test 1: Augmented Dickey-Fuller (ADF) stationarity test.
# H0: series has a unit root (is non-stationary).
# H1: series is stationary.
adf_level = adfuller(series, autolag="AIC")
adf_diff = adfuller(change, autolag="AIC")
print("\\nADF on price level: statistic =", round(adf_level[0], 4),
      "p-value =", round(adf_level[1], 6))
print("ADF on first difference: statistic =", round(adf_diff[0], 4),
      "p-value =", round(adf_diff[1], 6))

# Test 2: One-sample t-test of mean returns against zero.
# H0: population mean return = 0.
# H1: population mean return != 0.
t_result = stats.ttest_1samp(returns, popmean=0, nan_policy="omit")
n = len(returns)
mean_r = returns.mean()
se = stats.sem(returns)
ci = stats.t.interval(0.95, df=n-1, loc=mean_r, scale=se)
print("\\nMean return:", round(mean_r, 8))
print("One-sample t-test: t =", round(t_result.statistic, 4),
      "p-value =", round(t_result.pvalue, 6))
print("95% CI for mean return:", tuple(round(x, 8) for x in ci))

# Test 3: Pearson correlation between prior price and next-period price change.
# H0: population correlation = 0.
# H1: population correlation != 0.
aligned = pd.concat([series.shift(1).rename("previous_price"),
                     change.rename("next_change")], axis=1).dropna()
corr = stats.pearsonr(aligned["previous_price"], aligned["next_change"])
print("\\nPearson correlation (previous price vs next change): r =",
      round(corr.statistic, 4), "p-value =", round(corr.pvalue, 6))

# Test 4: OLS regression of next-period change on previous price.
# H0: slope coefficient = 0.
# H1: slope coefficient != 0.
X = sm.add_constant(aligned["previous_price"])
model = sm.OLS(aligned["next_change"], X).fit()
print("\\nOLS regression summary:")
print(model.summary())

# Optional residual diagnostic if there are enough observations.
if len(model.resid) >= 10:
    lb = acorr_ljungbox(model.resid, lags=[min(10, max(1, len(model.resid)//5))],
                        return_df=True)
    print("\\nLjung-Box residual autocorrelation test:")
    print(lb)

print("\\nDecision rule: reject H0 when p-value < ", ALPHA)
print("Statistical significance does not by itself establish practical importance or causation.")
