import pandas as pd

# Assuming qqq_data is your pandas DataFrame with features and target variable 'Close'
# Extract features (X_test) and target variable (y_test)
qqq_data = pd.read_csv('qqq_model_data.csv')
X_test = qqq_data[['RSI', 'MACD', 'Upper_band', 'Lower_band', 'EMA_20', 'EMA_50']]
y_test = qqq_data['Close']

# Save X_test to CSV file
X_test.to_csv('X_test.csv', index=False)

# Save y_test to CSV file
y_test.to_csv('y_test.csv', index=False, header=['Close'])
