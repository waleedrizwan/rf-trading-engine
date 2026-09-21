import datetime as dt

import numpy as np
import pandas as pd

from indicators import build_features, calc_bollinger, calc_momentum, calc_rsi


def prices_from(values):
    return pd.DataFrame({"ABC": values}, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_rsi_is_100_when_prices_only_rise():
    rsi = calc_rsi(prices_from(np.arange(1.0, 41.0)), "ABC")
    assert rsi.isna().sum() == 14
    assert (rsi.dropna() == 100).all()


def test_rsi_is_0_when_prices_only_fall():
    rsi = calc_rsi(prices_from(np.arange(40.0, 0.0, -1.0)), "ABC")
    assert (rsi.dropna() == 0).all()


def test_bollinger_percent_b_is_half_at_the_moving_average():
    values = np.tile([10.0, 12.0], 20)
    values[-1] = values[-20:-1].sum() / 19  # makes the last price equal its own 20-day SMA
    bbp = calc_bollinger(prices_from(values), "ABC")
    assert abs(bbp.iloc[-1] - 0.5) < 1e-9


def test_momentum_matches_lookback_return():
    values = np.linspace(100, 200, 30)
    momentum = calc_momentum(prices_from(values), "ABC", lookback=14)
    assert abs(momentum.iloc[-1] - (values[-1] / values[-15] - 1)) < 1e-12


def test_build_features_has_no_gaps_from_first_day(market_data):
    features = build_features("ABC", dt.datetime(2008, 1, 1), dt.datetime(2008, 12, 31))
    assert list(features.columns) == ["BBP", "RSI", "MACD", "Momentum", "EMA_Crossover"]
    assert features.index[0] == pd.Timestamp("2008-01-01")
    assert not features.isna().any().any()
