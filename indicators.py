import datetime as dt

import pandas as pd

from util import get_data


def calc_bollinger(prices, symbol, window=20, num_std=2):
    """
    Calculate Bollinger Bands %B: where the price sits relative to the bands.
    0 is the lower band, 1 is the upper band.
    """
    price = prices[symbol]
    sma = price.rolling(window=window, min_periods=window).mean()
    std = price.rolling(window=window, min_periods=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (price - lower) / (upper - lower)


def calc_rsi(prices, symbol, lookback=14):
    """
    Calculate the Relative Strength Index (0-100) over a rolling lookback window.
    """
    daily_rets = prices[symbol].diff()
    up_gain = daily_rets.clip(lower=0).rolling(window=lookback, min_periods=lookback).sum()
    down_loss = (-daily_rets.clip(upper=0)).rolling(window=lookback, min_periods=lookback).sum()
    rs = up_gain / down_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(down_loss != 0, 100).where(up_gain.notna())


def calc_macd(prices, symbol):
    """
    Calculate the MACD histogram (MACD line minus its 9-day signal line).
    """
    ema_12 = prices[symbol].ewm(span=12, adjust=False).mean()
    ema_26 = prices[symbol].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calc_momentum(prices, symbol, lookback=14):
    """
    Calculate momentum: the fractional price change over the lookback window.
    """
    return (prices[symbol] / prices[symbol].shift(lookback)) - 1


def calc_ema(prices, symbol):
    """
    Calculate the EMA crossover spread (12-day EMA minus 26-day EMA).
    Positive values indicate the short-term trend is above the long-term trend.
    """
    ema_12 = prices[symbol].ewm(span=12, adjust=False).mean()
    ema_26 = prices[symbol].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def build_features(symbol, sd, ed, warmup_days=60):
    """
    Build the indicator feature matrix for a symbol over [sd, ed].

    Prices are loaded from warmup_days calendar days before sd so every
    indicator has a full lookback window on the first trading day.

    Returns:
        pd.DataFrame indexed by trading day with columns
        BBP, RSI, MACD, Momentum, EMA_Crossover.
    """
    dates = pd.date_range(sd - dt.timedelta(days=warmup_days), ed)
    prices = get_data([symbol], dates)[[symbol]].ffill().bfill()

    features = pd.DataFrame(index=prices.index)
    features['BBP'] = calc_bollinger(prices, symbol)
    features['RSI'] = calc_rsi(prices, symbol)
    features['MACD'] = calc_macd(prices, symbol)
    features['Momentum'] = calc_momentum(prices, symbol)
    features['EMA_Crossover'] = calc_ema(prices, symbol)
    return features.loc[sd:ed]


if __name__ == "__main__":
    test_symbol = "JPM"
    stock_data = get_data([test_symbol], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))

    print("Bollinger %B:", calc_bollinger(stock_data, test_symbol).dropna().head(), sep="\n")
    print("RSI:", calc_rsi(stock_data, test_symbol).dropna().head(), sep="\n")
    print("MACD:", calc_macd(stock_data, test_symbol).head(), sep="\n")
    print("Momentum:", calc_momentum(stock_data, test_symbol).dropna().head(), sep="\n")
    print("EMA crossover:", calc_ema(stock_data, test_symbol).head(), sep="\n")
