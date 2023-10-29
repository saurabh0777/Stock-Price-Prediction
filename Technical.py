import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Fetch QQQ ETF historical data
ticker = "QQQ"
data = pd.read_csv("stock_data_qqq.csv")

# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data['RSI'] = calculate_rsi(data)



# Calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

data['MACD'], data['Signal'] = calculate_macd(data)

# Plotting the stock data with RSI and MACD
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='QQQ Close Price')
plt.ylabel('Price')
plt.title('QQQ ETF Technical Analysis')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data.index, data['RSI'], label='RSI', color='orange')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
plt.ylabel('RSI')
plt.grid(True)
plt.legend()

plt.show()
