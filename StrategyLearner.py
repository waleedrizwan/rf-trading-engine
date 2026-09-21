"""
Machine-learning trading strategy using a bagged ensemble of random trees.

Based on a template (c) 2016 Tucker Balch.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

"""

import datetime as dt
import random

import pandas as pd
import util as ut
from indicators import calc_bollinger, calc_rsi, calc_macd, calc_momentum, calc_ema
from RTLearner import RTLearner
from BagLearner import BagLearner
from marketsimcode import compute_portvals, get_stock_price
import numpy as np

class StrategyLearner(object):
    """
    Learns a trading policy from the same technical indicators used by ManualStrategy.

    Training labels each day as buy (+1), sell (-1), or hold (0) based on the
    N-day forward return, with thresholds widened to account for market impact.
    A BagLearner of RTLearners is then fit to the indicator values, and its
    predictions drive position changes at test time.

    :param verbose: If True, print debugging information.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.N = 15
        self.YBUY = 0.02
        self.YSELL = -0.02
        self.leaf_size = 5
        self.num_bags = 50


    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Train the learner over the given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        syms = [symbol]
        extended_start = sd - dt.timedelta(days=30)
        extended_end = ed + dt.timedelta(days=30)
        dates = pd.date_range(extended_start, extended_end)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]
        prices_SPY = prices_all["SPY"]
        if self.verbose:
            print(prices)

        bbp = calc_bollinger(prices, symbol)
        rsi = calc_rsi(prices)
        macd = calc_macd(prices, symbol)
        momentum = calc_momentum(prices, symbol)
        ema_crossover = calc_ema(prices.copy(), symbol)

        X = pd.DataFrame(index=prices.index)
        X['BBP'] = bbp
        X['RSI'] = rsi
        X['MACD'] = macd
        X['Momentum'] = momentum
        X['EMA_Crossover'] = ema_crossover

        X = X.loc[sd:ed]
        prices = prices.loc[sd:ed]

        X = X.dropna()
        Y = pd.Series(index=X.index, dtype=int)

        impact_multiplier = 15.0

        buy_threshold = self.YBUY + (self.impact * impact_multiplier)
        sell_threshold = self.YSELL - (self.impact * impact_multiplier)

        valid_indices = X.index[X.index < prices.index[-self.N]]

        for date in valid_indices:
            idx = prices.index.get_loc(date)
            future_idx = idx + self.N

            current_price = prices.iloc[idx].iloc[0]
            future_price = prices.iloc[future_idx].iloc[0]

            curr_return = (future_price / current_price) - 1.0

            if curr_return > buy_threshold:
                Y[date] = 1
            elif curr_return < sell_threshold:
                Y[date] = -1
            else:
                Y[date] = 0

        common_index = X.index.intersection(Y.index)
        X = X.loc[common_index]
        Y = Y.loc[common_index]

        if self.verbose:
            buy_count = (Y == 1).sum()
            sell_count = (Y == -1).sum()
            hold_count = (Y == 0).sum()
            print(f"Training samples: {len(Y)}, Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")
            print(f"X shape: {X.shape}, Y shape: {Y.shape}")

        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size, "verbose": self.verbose},
            bags=self.num_bags,
            boost=False,
            verbose=self.verbose
        )
        self.learner.add_evidence(X.values, Y.values)

    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Generate trades using the trained learner, typically on out-of-sample data.

        :param symbol: The stock symbol to trade
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: Daily trades in shares. +1000 is a buy of 1000 shares, -1000 a sell, 0 no action.
            ±2000 occurs when flipping between long and short; net holdings are always -1000, 0, or 1000.
        :rtype: pandas.DataFrame
        """

        if self.learner is None:
            return pd.DataFrame(index=pd.date_range(sd, ed), columns=['Position'], data=0.0)

        actual_trading_days = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True).index

        extended_start = sd - dt.timedelta(days=30)
        extended_end = ed
        dates = pd.date_range(extended_start, extended_end)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]]

        bbp = calc_bollinger(prices, symbol)
        rsi = calc_rsi(prices)
        macd = calc_macd(prices, symbol)
        momentum = calc_momentum(prices, symbol)
        ema_crossover = calc_ema(prices.copy(), symbol)

        X = pd.DataFrame(index=prices.index)
        X['BBP'] = bbp
        X['RSI'] = rsi
        X['MACD'] = macd
        X['Momentum'] = momentum
        X['EMA_Crossover'] = ema_crossover
        X = X.loc[sd:ed]
        X = X.dropna()

        predictions = self.learner.query(X.values)
        trades = pd.DataFrame(0.0, index=actual_trading_days, columns=['Position'])
        valid_dates = X.index.intersection(actual_trading_days)
        current_position = 0
        confidence_threshold = 0.8
        impact_penalty = 0.6 * self.impact

        buy_confidence = confidence_threshold + impact_penalty
        sell_confidence = -confidence_threshold - impact_penalty

        last_trade_day = None
        min_holding_days = 15

        for i, date in enumerate(valid_dates):
            idx = valid_dates.get_loc(date)
            pred = predictions[idx]

            if last_trade_day is not None:
                days_since_trade = i - last_trade_day
                if days_since_trade < min_holding_days:
                    continue

            if pred > buy_confidence and current_position <= 0:
                target_position = 1000
                trade_amount = target_position - current_position
                trades.loc[date, 'Position'] = trade_amount
                current_position = target_position
                last_trade_day = i
            elif pred < sell_confidence and current_position >= 0:
                target_position = -1000
                trade_amount = target_position - current_position
                trades.loc[date, 'Position'] = trade_amount
                current_position = target_position
                last_trade_day = i

        if current_position != 0 and len(valid_dates) > 0:
            last_valid_date = valid_dates[-1]
            if last_valid_date in trades.index:
                trades.loc[last_valid_date, 'Position'] -= current_position
                current_position = 0

        if self.verbose:
            total_trades = (trades['Position'] != 0).sum()
            print(f"Total trades: {total_trades}")
            print(f"Final position: {current_position}")

        return trades
