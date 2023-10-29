import pandas as pd

# Load the downloaded stock data from the CSV file
data_file_path = "stock_data_qqq.csv"
stock_data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

# Calculate 7-day and 30-day moving averages
stock_data['7_Day_MA'] = stock_data['Close'].rolling(window=7).mean()
stock_data['30_Day_MA'] = stock_data['Close'].rolling(window=30).mean()

# Calculate the daily price change percentage
stock_data['Price_Change_Percentage'] = stock_data['Close'].pct_change() * 100

# Calculate the 7-day volatility (standard deviation of daily price changes)
stock_data['Volatility'] = stock_data['Price_Change_Percentage'].rolling(window=7).std()

# Calculate the Relative Strength Index (RSI) - a momentum oscillator
window_length = 14
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=window_length, min_periods=1).mean()
average_loss = loss.rolling(window=window_length, min_periods=1).mean()
relative_strength = average_gain / average_loss
rsi = 100 - (100 / (1 + relative_strength))
stock_data['RSI'] = rsi





# Print the first few rows of the updated stock data with engineered features
print(stock_data.head())

# Save the stock data with engineered features to a new CSV file
engineered_data_file_path = "engineered_stock_data_qqq.csv"
stock_data.to_csv(engineered_data_file_path)

# Print the path of the saved file with engineered features
print("Engineered stock data file path:", engineered_data_file_path)