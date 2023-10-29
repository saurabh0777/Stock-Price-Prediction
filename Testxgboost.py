import pickle
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the saved model from a file
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Use the loaded model to make predictions
# Assuming X_new contains the new feature data for prediction
# 365.2799988	56.57012078	1.343409969	373.7357032	350.9522973	364.8638445	365.5820766
X_new = pd.DataFrame({
    'RSI': [56],
    'MACD': [1.34],
    'Upper_band': [373],
    'Lower_band': [350],
    'EMA_20': [364],
    'EMA_50': [365]
})
print("Input Parameters:")
print(X_new.to_string(index=False))

predictions = loaded_model.predict(X_new)

print("Predictions using the loaded model:")
print(predictions)


y_valid = pd.Series([
    369
])

rmse = np.sqrt(mean_squared_error(y_valid, predictions))

accuracy = y_valid / predictions
acc= accuracy * 100
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(round(acc.iloc[0]))

# Calculate the mean absolute error (MAE).
mae = np.mean(np.abs(predictions - y_valid))

# Calculate the mean squared error (MSE).
mse = np.mean((predictions - y_valid)**2)

# Calculate the mean bias error (MBE).
mbe = np.mean(predictions - y_valid)

# Print the MAE, MSE, and MBE.
print("MAE:", mae)
print("MSE:", mse)
print("MBE:", mbe)
accuracy = np.mean((y_valid == predictions ) * 100)

# Print the accuracy.
print("Accuracy:", accuracy)
