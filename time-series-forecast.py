# basic time series analysis with ARIMA, starting with Augmented Dickey-Fuller (ADF) test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Generate a synthetic dataset for demonstration
np.random.seed(42)
date_range = pd.date_range(start="2010-01", end="2020-01", freq="M")
data = 100 + 5 * np.sin(2 * np.pi * date_range.month / 12) + np.random.normal(0, 2, len(date_range))
time_series = pd.Series(data, index=date_range)

# EDA: Visualize the Data
plt.figure(figsize=(10, 6))
plt.plot(time_series, label="Time Series")
plt.title("Synthetic Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# EDA: Stationarity Check (Non-stationary data must be differenced to remove trends or seasonality)
def check_stationarity(timeseries):
    """
    Perform the Augmented Dickey-Fuller test and visualize rolling statistics.
    """
    # Rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, label="Original", color="blue")
    plt.plot(rolling_mean, label="Rolling Mean", color="orange")
    plt.plot(rolling_std, label="Rolling Std", color="green")
    plt.title("Rolling Mean & Standard Deviation")
    plt.legend()
    plt.show()
    
    # Perform the Augmented Dickey-Fuller test
    adf_test = adfuller(timeseries)
    print("ADF Statistic:", adf_test[0])
    print("p-value:", adf_test[1])
    print("Critical Values:")
    for key, value in adf_test[4].items():
        print(f"   {key}: {value}")
    if adf_test[1] > 0.05:
        print("The time series is non-stationary.")
    else:
        print("The time series is stationary.")

# Check stationarity
check_stationarity(time_series)

# If the series is non-stationary, apply differencing
if adfuller(time_series)[1] > 0.05:
    time_series_diff = time_series.diff().dropna()
    print("\nAfter Differencing:")
    check_stationarity(time_series_diff)
else:
    time_series_diff = time_series  # No differencing needed

# Split data into training and test sets
train_size = int(len(time_series_diff) * 0.8)
train, test = time_series_diff[:train_size], time_series_diff[train_size:]

# Fit the ARIMA model using the identified parameters (𝑝,𝑑,𝑞) and the historical data
p, d, q = 2, 1, 2  # ARIMA parameters
model = ARIMA(train, order=(p, d, q))
fitted_model = model.fit()

# Print model summary
print(fitted_model.summary())

# Make predictions
forecast = fitted_model.forecast(steps=len(test))

comparison_df = pd.DataFrame({
    'Actual': test,
    'Forecast': forecast
})
print(comparison_df.head(10))

# Evaluate the model
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label="Training Data")
plt.plot(test, label="Test Data")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()


# Model deployment
