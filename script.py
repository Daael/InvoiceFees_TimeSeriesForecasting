# Standard libraries
import os
import io
import time
import calendar

# Data handling
import pandas as pd
import numpy as np
from dateutil.parser import parse
import joblib

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from pandas.plotting import lag_plot
from statsmodels.tsa.seasonal import STL

# Statistical and time series analysis
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Set the document path to the current directory
# base_path = os.getcwd()

# Read the data file
# df = pd.read_csv(os.path.join(base_path, "data.csv"))

# Read it in Github
df = pd.read_csv("data.csv")

# Drop all variables except the date and predicted variables
df = df[['FECHA_FACTURA','FEE_FACTURAS_EUROS']]

# Translate the variables names from Spanish to English
df.rename(columns={
    'FECHA_FACTURA': 'DATE_INVOICE',
    'FEE_FACTURAS_EUROS': 'INVOICE_FEE_EUROS',
}, inplace=True)

# Convert object type variable to numeric, replacing commas.
df['INVOICE_FEE_EUROS'] = pd.to_numeric(df['INVOICE_FEE_EUROS'].str.replace(',', ''), errors='coerce')

# Convert object type date variable to date format.
df['DATE_INVOICE'] = pd.to_datetime(df['DATE_INVOICE'], format='%d/%m/%Y')

# Drop values with missing Invoice fee Euros, likely due to data entry errors.
df = df.dropna(subset=['INVOICE_FEE_EUROS'])

# Create a year index
df['DATE'] = df['DATE_INVOICE']
df.set_index('DATE', inplace=True)

# Sort values
df = df.sort_values(by=['DATE_INVOICE'], ascending=True)

# Drop columns below the year 2018 due to different operastional behavior
df = df[~(df['DATE_INVOICE'] < '2018-01-01')]

# Drop columns above 2018-05 due to lack of data.
df = df[~(df['DATE_INVOICE'] >= '2024-05-01')]

# Extract the year and month from the invoice date for time-based aggregation.
df['year'] = df['DATE_INVOICE'].dt.year
df['month'] = df['DATE_INVOICE'].dt.month

# Aggregate total invoice fees by year and month.
# This groups the data at a monthly level and sums the invoice fees for each period.
monthly_totals = (
    df.groupby(['year', 'month'], as_index=False)['INVOICE_FEE_EUROS']
      .sum()
      .sort_values(['year', 'month'])
)

# Create a new datetime column representing the first day of each month.
# This will serve as a proper time index for time series analysis.
monthly_totals['date'] = pd.to_datetime(
    monthly_totals['year'].astype(str) + '-' + monthly_totals['month'].astype(str) + '-01'
)

# let's try to make it stationary by calculating the monthly differences (month minus last month)
monthly_totals["diff"] = monthly_totals["INVOICE_FEE_EUROS"].diff(1)  # y[t] - y[t-1]

# We need to drop the first value because it will be NAN due to first month not having month to differenciate
# Store it in another Data to not loose values in our original dataset
monthly_diff = monthly_totals.copy(deep=True)
monthly_diff = monthly_diff.dropna()
monthly_diff = monthly_diff.drop(columns="INVOICE_FEE_EUROS")

# Do also seasonal difference (month minus same month of last year).
monthly_totals["diff_seasonal"] = monthly_totals["INVOICE_FEE_EUROS"].diff(12)  # y[t] - y[t-1]

# Store it in another Data to not loose values in oru original dataset
monthly_diff_seasonal = monthly_totals.copy(deep=True)
monthly_diff_seasonal = monthly_diff_seasonal.dropna()
monthly_diff_seasonal = monthly_diff_seasonal.drop(columns="INVOICE_FEE_EUROS")
monthly_diff_seasonal = monthly_diff_seasonal.drop(columns="diff")

# We apply seasonal difference and a first difference to obtain stationary data.
monthly_totals["diff_combined"] = monthly_totals["INVOICE_FEE_EUROS"].diff(1).diff(12)  # y[t] - y[t-1]

# Store it in another Data to not loose values in our original dataset
monthly_diff_combined = monthly_totals.copy(deep=True)
monthly_diff_combined = monthly_diff_combined.dropna()
monthly_diff_combined = monthly_diff_combined.drop(columns="INVOICE_FEE_EUROS")
monthly_diff_combined = monthly_diff_combined.drop(columns="diff")
monthly_diff_combined = monthly_diff_combined.drop(columns="diff_seasonal")

# Drop difference created from original data
monthly_totals = monthly_totals.drop(columns=["diff_combined", "diff", "diff_seasonal", "year", "month"])

# Forecasting with the best model
model = sm.tsa.SARIMAX(
    monthly_totals["INVOICE_FEE_EUROS"],
    order=(2,1,3),
    seasonal_order=(0,1,0,12)
)
results = model.fit()

# Forecasting the next year
n_steps = 12
forecast = results.get_forecast(steps=n_steps)

# Plot the observed data
plt.figure(figsize=(12,6))
plt.plot(monthly_totals.index, monthly_totals["INVOICE_FEE_EUROS"], label='Observed')


# Extract forecast results
forecast_df = forecast.summary_frame()
forecast_df.head()

# Forecasted values
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='orange')

# Confidence intervals
plt.fill_between(
    forecast_df.index,
    forecast_df['mean_ci_lower'],
    forecast_df['mean_ci_upper'],
    color='orange', alpha=0.2, label='95% Confidence Interval'
)

plt.title("SARIMA Forecast for Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Fees (€)")
plt.legend()
plt.show()
