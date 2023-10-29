import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import pickle

qqq_data = pd.read_csv('qqq_model_data.csv')

X = qqq_data[['RSI', 'MACD', 'Upper_band', 'Lower_band', 'EMA_20', 'EMA_50']]
y = qqq_data['Close']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
xgb_model  = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Calculate and print training accuracy (R-squared score)
train_predictions = xgb_model.predict(X_train)
train_r2 = r2_score(y_train, train_predictions)
print(f'Training Accuracy (R-squared): {train_r2:.2f}')

# Calculate and print testing accuracy (R-squared score)
test_predictions = xgb_model.predict(X_test)
test_r2 = r2_score(y_test, test_predictions)
print(f'Testing Accuracy (R-squared): {test_r2:.2f}')


with open('xgboost_model_latest.pkl', 'wb') as model_file:
    pickle.dump(xgb_model , model_file)

print("XGBoost model saved successfully.")

# Make predictions
predictions = xgb_model.predict(X_test)

# Calculate R-squared value
r2_value = r2_score(y_test, predictions)

# Visualize predictions vs actual values with R-squared value
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label=f'R-squared: {r2_value:.2f}')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Regression Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('QQQ Stock Price Prediction (XGBoost)')
plt.legend()
plt.show()

print(f'R-squared value: {r2_value:.2f}')