import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle


# Load the saved XGBoost model from a file
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Read X_test from CSV file
X_test = pd.read_csv('X_test.csv')
# Read y_test from CSV file
y_test = pd.read_csv('y_test.csv')

# Handle missing values in X_test (filling NaN with mean values)
X_test = X_test.fillna(X_test.mean())

# Use the loaded model to make predictions on the test data
predictions = loaded_model.predict(X_test)

# Calculate R-squared value
r2_value = r2_score(y_test, predictions)

# Convert R-squared value to percentage
r2_percentage = r2_value * 100

print(f'R-squared value: {r2_percentage:.2f}%')
