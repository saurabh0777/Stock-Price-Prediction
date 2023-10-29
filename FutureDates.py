import pickle
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime



def calculate_rsi(close_prices, window=14):

    close_diff = close_prices.diff(1).dropna()

    gain = close_diff.where(close_diff > 0, 0).fillna(0)
    loss = -close_diff.where(close_diff < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def calculate_bollinger_bands(close_prices, window=20, num_std_dev=2):

    close_series = pd.Series(close_prices)
    middle_band = close_series.rolling(window=window, min_periods=1).mean()
    std_dev = close_series.rolling(window=window, min_periods=1).std()

    upper_band = middle_band + num_std_dev * std_dev
    lower_band = middle_band - num_std_dev * std_dev


    return (upper_band.iloc[-1],lower_band.iloc[-1])

def calculate_ema(close_prices, short_window=20, long_window=50):

    close_series = pd.Series(close_prices)

    ema_short = close_series.ewm(span=short_window, min_periods=1, adjust=False).mean()
    ema_long = close_series.ewm(span=long_window, min_periods=1, adjust=False).mean()

    return (ema_short.iloc[-1],ema_long.iloc[-1])

def calculate_macd(close_prices, short_window=12, long_window=26, signal_window=9):

    close_series = pd.Series(close_prices)

    short_ema = close_series.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = close_series.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    macd_values = macd_line - signal_line

    return macd_values.iloc[-1]

'''
qqq_data = pd.read_csv('qqq_model_date.csv',index_col=None).tail(50)
closing = qqq_data['Close']
#rsi = calculate_rsi(closing)
#print("getting rsi ",rsi)
bands = calculate_bollinger_bands(closing)
print(type(bands),bands)
print("Upper_band ",bands[0])
print("Lower_band",bands[1])
ema_data = calculate_ema(closing)
print("ema_data",ema_data)
macd_value = calculate_macd(closing)
print("macd_value",macd_value)
'''


# Load the saved model from a file
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

current_date = datetime.date.today()
qqq_data = pd.read_csv('qqq_model_date.csv',index_col=None).tail(50)
closing = qqq_data['Close'].tolist()

print("columns in dataframe",qqq_data.columns.tolist())
columns = ['RSI','MACD','Upper_band','Lower_band','EMA_20','EMA_50']

new_df = pd.DataFrame(columns=['Date','Predicted','RSI','MACD','Upper_band','Lower_band','EMA_20','EMA_50'])
for i in range(14):
    next_date = current_date + datetime.timedelta(days=i)
    if next_date.weekday() not in [5, 6]:
      if new_df.empty:
          X_data = qqq_data[columns].iloc[-1:].copy()
      else:
          X_data = new_df[columns].iloc[-1:].copy()

      predictions = loaded_model.predict(X_data)

      print(f'predicted stock value for data {next_date} is {predictions}')

      closing.append(predictions[0])
      print("Closing values appended ",closing)

      s = pd.Series(closing)

      rsi = calculate_rsi(s.tail(14))

      bands = calculate_bollinger_bands(s.tail(20))
      ema_data = calculate_ema(s.tail(50))
      macd_value = calculate_macd(s.tail(26))
      print(f'New Tech Indicators rsi: {rsi},bands :{bands}, ema_data:{ema_data} ,macd_value: {macd_value}')
      new_row = {'Date': next_date ,
                 'Predicted': predictions[0],
                 'RSI':rsi,
                 'MACD':macd_value,
                 'Upper_band':bands[0],
                 'Lower_band':bands[1],
                 'EMA_20':ema_data[0],
                 'EMA_50':ema_data[1]
                 }
      print(new_row)

      new_df = new_df._append(new_row,ignore_index=True)

      # Print the DataFrame.
      print("value of output",new_df.to_string())


new_df.to_csv('pred_price.csv',index=False)
