# Revenue Forecasting

This project presents a **real-world machine learning application** developed for a multinational telecommunications company.
The primary goal is to **forecast the monthly invoice fees.**

The current method used is a simple linear regression model, this project explores time series forecasting techniques to achieve more accurate and reliable predictions.

**Skills used**: Statistical Modelling, EDA, Autocorrelation Analysis, SARIMA Models, Forecasting Techniques, Virtual Machines etc.

**Technologies**: Python, Azure, Matplotlib, Numpy, Pandas, Seaborn, Sheets, Jupyter, etc.

If you want to deep dive in the techniques used, you can explore the entire time series analysis process in the next Jupyter notebooks.
[Project Notebook](https://github.com/Daael/InvoiceFees_TimeSeriesForecasting/blob/main/TSforecasting.ipynb).


---

## Overview

A multinational telecommunications company currently relies on a simple linear regression model to forecast its monthly and annual revenues. To enhance accuracy, the organization seeks to implement a more advanced forecasting approach, enabling improved resource planning, budgeting, and strategic decision-making.. 

**Goal**: Build a Time Series Forecast Analysis capable of accurately predicting the monthly income of this international company. 

The dataset originates from the company’s historical invoice records. The variable INVOICE_FEE_EUROS represents the revenue associated with each generated invoice.

The dataset provided in this repository is **anonymized** and represents **5% of the original data** for efficiency.  

- **Full dataset download:** [Google Drive link](https://drive.google.com/file/d/1hlmHiU9xRZyPFD9c9qu5ZDSIZHB5Q_FX/view?usp=sharing)

---

## Summary ##

We built a forecasting model designed to match how the company’s revenue actually moves over time. When we looked at the historical data, one thing stood out immediately: revenue consistently jumps every fourth quarter. A simple linear regression misses this seasonal pattern, so it ends up giving misleading monthly predictions, especially near the end of each year.

To solve this, the model incorporates both long-term trends and seasonal behavior. It also adapts to the most recent months, which helps it stay aligned with current business conditions. **By updating the forecast each month, the company gets a much clearer picture of expected revenue and can plan with greater confidence.**

![Forecast](https://raw.githubusercontent.com/Daael/InvoiceFees_TimeSeriesForecasting/main/Images/Forecast.PNG)

---

## Insights and Recommendations ##

Some interesting Insights found were:

![Trend](https://raw.githubusercontent.com/Daael/InvoiceFees_TimeSeriesForecasting/main/Images/Trend.PNG)
- If we look at the revenue monthly for each year, we would think that there is a consistent increasing trend through the year. This is false, the significant growth trend occurs primarily in the fourth quarter, particularly in December, while the rest of the year does not exhibit a meaningful trend pattern. Is important to consider that for accurate budgeting and strategic resource planning.

![Trend2](https://raw.githubusercontent.com/Daael/InvoiceFees_TimeSeriesForecasting/main/Images/Trend2.PNG)
- After removing seasonal effects and residual noise from the historical data, a clear downward trend in revenue emerges over the past few years. This indicates a potential long-term decline in income, suggesting the company may need to reassess and adjust its strategic approach to reverse this pattern. The forecasting model is designed to adapt to changing business conditions, so if performance improves, the model will capture and reflect this recovery in future predictions.

![Table](https://raw.githubusercontent.com/Daael/InvoiceFees_TimeSeriesForecasting/main/Images/Table.PNG)
- The model provides a confidence interval for its forecasts, offering a range of values with a 95% probability of containing the actual outcome. This allows the company to manage expectations and make informed decisions based on different scenarios. For example, the upper bound could be used when projecting potential growth for investment purposes, while the lower bound might support more conservative planning. Not morally correct but they will not be fraudulent. 
