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

import pandas as pd

import util as ut
from BagLearner import BagLearner
from indicators import build_features
from RTLearner import RTLearner


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
        # Ensemble predictions are averaged votes in [-1, 1]; trade when they
        # exceed this threshold. Tuned on in-sample data only.
        self.confidence_threshold = 0.4
        self.min_holding_days = 15


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

        X = build_features(symbol, sd, ed)
        prices = ut.get_data([symbol], pd.date_range(sd, ed))[symbol].ffill().bfill()
        if self.verbose:
            print(prices)

        # Label each day by its N-day forward return. Thresholds are widened by
        # market impact so the learner only targets moves that cover trading costs.
        impact_multiplier = 15.0
        buy_threshold = self.YBUY + (self.impact * impact_multiplier)
        sell_threshold = self.YSELL - (self.impact * impact_multiplier)

        forward_return = prices.shift(-self.N) / prices - 1.0
        Y = pd.Series(0, index=prices.index, dtype=float)
        Y[forward_return > buy_threshold] = 1
        Y[forward_return < sell_threshold] = -1

        # The last N days have no forward return to label, so drop them
        labeled = forward_return.dropna().index
        X = X.loc[labeled]
        Y = Y.loc[labeled]

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
        :param sd: A datetime object that represents the start date, defaults to 1/1/2009
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2010
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: Daily trades in shares. +1000 is a buy of 1000 shares, -1000 a sell, 0 no action.
            ±2000 occurs when flipping between long and short; net holdings are always -1000, 0, or 1000.
        :rtype: pandas.DataFrame
        """

        if self.learner is None:
            raise RuntimeError("StrategyLearner must be trained with add_evidence before testPolicy")

        X = build_features(symbol, sd, ed)

        predictions = self.learner.query(X.values)
        trades = pd.DataFrame(0.0, index=X.index, columns=['Position'])
        current_position = 0
        impact_penalty = 0.6 * self.impact

        buy_confidence = self.confidence_threshold + impact_penalty
        sell_confidence = -self.confidence_threshold - impact_penalty

        last_trade_day = None

        for i, (date, pred) in enumerate(zip(X.index, predictions)):
            if last_trade_day is not None:
                days_since_trade = i - last_trade_day
                if days_since_trade < self.min_holding_days:
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

        # Close out any open position on the final day
        if current_position != 0:
            trades.iloc[-1, 0] -= current_position
            current_position = 0

        if self.verbose:
            total_trades = (trades['Position'] != 0).sum()
            print(f"Total trades: {total_trades}")
            print(f"Final position: {current_position}")

        return trades
