import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data, plot_data
import datetime as dt


def calc_bollinger(prices, symbol, window=20, num_std=2):
    """
    Calculate Bollinger Bands %B
    """
    stock_data = prices.copy()
    stock_data['EMA_12'] = stock_data[symbol].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data[symbol].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    stock_data['MACD_Hist'] = stock_data['MACD'] - stock_data['Signal']
    return stock_data['MACD_Hist']


def calc_rsi(stock_data):

    daily_rets = stock_data.diff()
    daily_rets.iloc[0, :] = np.nan
    lookback = 14
    up_rets = daily_rets.where(daily_rets > 0, 0)
    down_rets = -1 * daily_rets.where(daily_rets < 0, 0)
    up_gain = up_rets.rolling(window=lookback, min_periods=lookback).sum()
    down_loss = down_rets.rolling(window=lookback, min_periods=lookback).sum()
    rs = up_gain / down_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(down_loss != 0, 100)
    rsi[:lookback] = np.nan

    return(rsi)


def calc_macd(prices, symbol):
    """
    Calculate MACD indicator
    """
    stock_data = prices.copy()
    stock_data['EMA_12'] = stock_data[symbol].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data[symbol].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    stock_data['MACD_Hist'] = stock_data['MACD'] - stock_data['Signal']
    return stock_data['MACD_Hist']


def calc_momentum(prices, symbol, lookback=14):
    """
    Calculate momentum indicator
    """
    stock_data = prices.copy()
    momentum = (stock_data[symbol] / stock_data[symbol].shift(lookback)) - 1
    return momentum

def calc_ema(stock_data, symbol):
    """
    Calculate EMA crossover indicator
    """
    stock_data['EMA_12'] = stock_data[symbol].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data[symbol].ewm(span=26, adjust=False).mean()
    stock_data['EMA_Crossover'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Buy'] = (stock_data['EMA_12'].shift(1) < stock_data['EMA_26'].shift(1)) & (stock_data['EMA_12'] > stock_data['EMA_26'])
    stock_data['Sell'] = (stock_data['EMA_12'].shift(1) > stock_data['EMA_26'].shift(1)) & (stock_data['EMA_12'] < stock_data['EMA_26'])
    return stock_data['EMA_Crossover']

def run():
    stock_data = get_data(["JPM"], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009,12,31)))
    stock_data.drop('SPY', axis=1, inplace=True)
    calc_bollinger(stock_data, "JPM")
    calc_rsi(stock_data)
    calc_macd(stock_data, "JPM")
    calc_momentum(stock_data, "JPM")
    calc_ema(stock_data, "JPM")

if __name__ == "__main__":
    test_symbol = "JPM"
    stock_data = get_data([test_symbol], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009,12,31)))
    stock_data.drop('SPY', axis=1, inplace=True)

    # Calculate indicators
    macd = calc_macd(stock_data, test_symbol)
    momentum = calc_momentum(stock_data, test_symbol)
    bollinger = calc_bollinger(stock_data, test_symbol)

    print("MACD:", macd.head())
    print("Momentum:", momentum.head())
    print("Bollinger:", bollinger.head())
