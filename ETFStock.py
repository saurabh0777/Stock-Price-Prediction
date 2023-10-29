import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(stock_data.head())
    stock_data.to_csv('stock_data_qqq_latest.csv',mode='w')
    return stock_data

def plot_stock_data(stock_data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Close'], label=f'{ticker} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{ticker} ETF Stock Price (Last 5 Years)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Set the ticker symbol for QQQ ETF and date range
    ticker_symbol = "QQQ"
    start_date = "2018-10-28"
    end_date = "2023-10-28"

    # Fetch stock data
    stock_data = get_stock_data(ticker_symbol, start_date, end_date)
   # stock_data = pd.read_csv('stock_data_qqq.csv')

    # Plot the stock data
   # plot_stock_data(stock_data, ticker_symbol)