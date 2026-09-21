import datetime

import pandas as pd

from indicators import build_features

class ManualStrategy():
    """
    Rule-based trading strategy built on the same technical indicators used by
    StrategyLearner (Bollinger %B, RSI, MACD histogram, momentum, EMA crossover).

    Each indicator casts a weighted long/short vote; when the combined score
    crosses a threshold the strategy moves to a long or short position.

    Parameters
        verbose (bool) – If True, print debugging information.
        impact (float) – Market impact of each transaction, defaults to 0.0.
        commission (float) – Commission charged per trade, defaults to 0.0.
    """
    def __init__(self,verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol='IBM', sd=datetime.datetime(2008, 1, 1, 0, 0), ed=datetime.datetime(2009, 1, 1, 0, 0), sv=100000):
        """
        No-op. The trading rules are hand-coded, so there is nothing to train.
        Present to keep the same interface as StrategyLearner.
        """
        pass

    def testPolicy(self, symbol='IBM', sd=datetime.datetime(2009, 1, 1, 0, 0), ed=datetime.datetime(2010, 1, 1, 0, 0), sv=100000):
        """
        Generate trades for the given symbol and date range using the rule-based policy.

        Parameters
            symbol (str) – Stock symbol to trade.
            sd (datetime) – Start date.
            ed (datetime) – End date.
            sv (int) – Starting portfolio value.

        Returns
            pandas.DataFrame – Daily trades in shares. +1000 is a buy of 1000 shares,
            -1000 a sell, 0 no action. ±2000 occurs when flipping between long and
            short; net holdings are always -1000, 0, or 1000.
        """
        trading_signals = build_features(symbol, sd, ed)
        manual_trades = pd.DataFrame(index=trading_signals.index)
        manual_trades['Position'] = 0.0
        current_position = 0
        date_adjusted_signals = trading_signals.shift(1).dropna()

        for index, row in date_adjusted_signals.iterrows():
            direction_indicator = 0

            if row['BBP'] < 0.2:
                direction_indicator += 1
            elif row['BBP'] > 0.8:
                direction_indicator -= 1

            if row['RSI'] < 30:
                direction_indicator += 2
            elif row['RSI'] > 70:
                direction_indicator -= 2

            if row['MACD'] > 0:
                direction_indicator += 1
            elif row['MACD'] < 0:
                direction_indicator -= 1

            if row['Momentum'] > 0.01:
                direction_indicator += 1
            elif row['Momentum'] < -0.01:
                direction_indicator -= 1

            if row['EMA_Crossover'] > 0:
                direction_indicator += 1
            elif row['EMA_Crossover'] < 0:
                direction_indicator -= 1

            if direction_indicator >= 3:
                target_position = 1000
            elif direction_indicator <= -3:
                target_position = -1000
            else:
                target_position = current_position

            trade = target_position - current_position
            if trade != 0:
                manual_trades.loc[index, 'Position'] = trade
                current_position = target_position

        if self.verbose:
            print(manual_trades[manual_trades['Position'] != 0].head(5))

        return manual_trades
