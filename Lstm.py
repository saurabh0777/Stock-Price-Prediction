import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense


# Load QQQ ETF historical data with feature engineering and technical indicators
def load_stock_data():
    # Fetch historical data
    data = pd.read_csv('stock_data_qqq.csv')

    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate MACD
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

    return data

# Load historical stock data for QQQ ETF with feature engineering and technical indicators

stock_data = load_stock_data()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data)

# Create sequences of historical data for LSTM
sequence_length = 10  # You can adjust this sequence length
sequences = []
next_values = []

for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i:i+sequence_length])
    next_values.append(scaled_data[i+sequence_length])

# Convert sequences to numpy arrays
X = np.array(sequences)
y = np.array(next_values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=y.shape[1]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Mean Squared Error on Test Data:", loss)

# Make predictions
predictions = model.predict(X_test)

# Denormalize the predictions and actual values for visualization
predicted_values = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], X.shape[2] - 1)), predictions)))
actual_values = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], X.shape[2] - 1)), y_test)))

# Visualize the predictions vs actual values
plt.figure(figsize=(14, 6))
plt.plot(actual_values[:, 0], label='Actual Close Prices', color='blue')
plt.plot(predicted_values[:, 0], label='Predicted Close Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('QQQ ETF Stock Price Prediction using LSTM')
plt.legend()
plt.show()
