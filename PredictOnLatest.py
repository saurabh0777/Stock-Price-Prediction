import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# Load the saved XGBoost model from a file
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Prepare input features for the specific date (20/10/2023)
# Assuming you have the feature values for this date in a pandas Series or DataFrame named 'input_features'

input_features = pd.Series({'RSI': 56, 'MACD': 1.3, 'Upper_band': 373, 'Lower_band': 350, 'EMA_20': 364, 'EMA_50': 365})

input_features = input_features.fillna(input_features.mean())

print(input_features)
# Make predictions for the specific date
predicted_close = loaded_model.predict(input_features.values.reshape(1, -1))[0]

print(f'Predicted QQQ closing price on 20/10/2023: {predicted_close:.2f}')

# Replace actual_close with the actual closing price for 20/10/2023
actual_close = 369 # Insert the actual closing price for 20/10/2023 here

