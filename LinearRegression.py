import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def load_stock_data(file):
    # Fetch historical data
    data = pd.read_csv(file)

    # Calculate RSI Relative strength index
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate MACD (Moving average convergence divergence
    short_ema = data['Close'].ewm(span=12, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, min_periods=1, adjust=False).mean()
    data['MACD'] = macd - signal

    # Calculate Bollinger Bands
    window = 20
    rolling_mean = data['Close'].rolling(window=window).mean()
    data['Upper_band'] = rolling_mean + 2 * data['Close'].rolling(window=window).std()
    data['Lower_band'] = rolling_mean - 2 * data['Close'].rolling(window=window).std()

    # Calculate EMA_20 and EMA_50
    data['EMA_20'] = data['Close'].ewm(span=20, min_periods=1, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, min_periods=1, adjust=False).mean()

    # Select relevant features
    features = ['Close', 'RSI', 'MACD', 'Upper_band', 'Lower_band', 'EMA_20', 'EMA_50']
    data = data[features]
    qqq_data = data.fillna(data.mean())

    return qqq_data

qqq_data = load_stock_data('stock_data_qqq_latest.csv')

#qqq_data = qqq_data.fillna(qqq_data.mean())

qqq_data.to_csv('qqq_model_data.csv',mode='w',index=False)

exit(0)


X = qqq_data[['RSI', 'MACD', 'Upper_band', 'Lower_band', 'EMA_20', 'EMA_50']]
y = qqq_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)


# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Regression Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('QQQ Stock Price Prediction (Linear Regression)')
plt.legend()
plt.show()

r2_value = r2_score(y_test, predictions)

# Visualize predictions vs actual values with R-squared value

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label=f'R-squared: {r2_value:.2f}')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Regression Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('QQQ Stock Price Prediction (Linear Regression)')
plt.legend()
plt.show()

print(f'R-squared value: {r2_value:.2f}')
'''
('The R-squared value will be displayed in the legend of the graph, indicating how well the model fits the data. The closer the R-squared value is to 1, '
 'the better the model explains the variance in the target variable.')
'''