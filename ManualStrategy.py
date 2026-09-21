from datetime import datetime
from indicators import calc_bollinger, calc_macd, calc_rsi, calc_momentum, calc_ema
from util import get_data
from marketsimcode import compute_portvals, get_stock_price
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

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
    # Class-level flag to control chart generation
    generate_charts = True

    def __init__(self,verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.comission = commission

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
        stock_data = get_data(["JPM"], pd.date_range(sd, ed))
        stock_data.drop('SPY', axis=1, inplace=True)

        first_trading_day = stock_data.index[0]
        last_trading_day  = stock_data.index[-1]

        stock_data_copy = stock_data.copy()
        bbp = calc_bollinger(stock_data_copy, symbol)
        rsi = calc_rsi(stock_data)
        macd_hist = calc_macd(stock_data, symbol)
        momentum = calc_momentum(stock_data, symbol)
        ema_crossover = calc_ema(stock_data.copy(), symbol)

        trading_signals = pd.DataFrame(index=stock_data.index)
        trading_signals['BBP'] = bbp
        trading_signals['RSI'] = rsi
        trading_signals['MACD'] = macd_hist
        trading_signals['Momentum'] = momentum
        trading_signals['EMA_Crossover'] = ema_crossover

        trading_signals = trading_signals.dropna()

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

        manual_strategy_orders = self.convert_order_format(manual_trades, symbol)
        self.add_empty_order(manual_strategy_orders, symbol, first_trading_day)
        self.add_empty_order(manual_strategy_orders, symbol, last_trading_day)

        if self.verbose:
            print(manual_strategy_orders.head(5))

        manual_strategy_daily_values = compute_portvals(manual_strategy_orders, start_val=sv, commission=9.95, impact=0.005)
        benchmark_orders = pd.DataFrame([[symbol,"BUY",1000]], columns=["Symbol","Order","Shares"], index=[first_trading_day, last_trading_day])
        daily_benchmark_performance = compute_portvals(benchmark_orders, start_val=sv, commission=9.95, impact=0.005)

        if ManualStrategy.generate_charts:
            self.handle_chart_creation(manual_strategy_daily_values, daily_benchmark_performance, sd, ed, manual_trades)

        self.calc_port_stats(daily_benchmark_performance, "Benchmark")
        self.calc_port_stats(manual_strategy_daily_values, "Manual Strategy")

        return manual_trades

    def handle_chart_creation(self, manual_strategy_values, benchmark_performance, sd, ed, manual_trades):
        if sd.year < 2010:
            chart_title = "Performance, Manual vs Benchmark, In-sample"
        else:
            chart_title = "Performance, Manual vs Benchmark, out of sample"

        self.plot_portfolio_performance(
            benchmark_performance,
            manual_strategy_values,
            chart_title,
            manual_trades
        )

    def plot_portfolio_performance(self, benchmark_return, manual_return, chart_title, manual_trades):

        combined_index = benchmark_return.index.union(manual_return.index)
        benchmark_return = benchmark_return.reindex(combined_index, method='ffill')
        manual_return = manual_return.reindex(combined_index, method='ffill')

        normalized_benchmark = benchmark_return['total_value'] / benchmark_return['total_value'].iloc[0]
        normalized_manual = manual_return['total_value'] / manual_return['total_value'].iloc[0]

        plt.figure(figsize=(16, 8))
        plt.plot(normalized_benchmark.index, normalized_benchmark,color="purple",label='Benchmark Strategy')
        plt.plot(normalized_manual.index, normalized_manual, color="red",  label='Manual Strategy')

        manual_trades = manual_trades.copy()
        manual_trades['net_position'] = manual_trades['Position'].cumsum()
        manual_trades['previous_position'] = manual_trades['net_position'].shift(1).fillna(0)

        for date, row in manual_trades.iterrows():
            new_pos = row['net_position']
            old_pos = row['previous_position']

            if (new_pos == 1000) and (old_pos != 1000):
                plt.axvline(x=date, color="blue", linestyle="--", linewidth=1)

            elif (new_pos == -1000) and (old_pos != -1000):
                plt.axvline(x=date, color="black", linestyle="--", linewidth=1)


        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.title(chart_title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"images/{chart_title}.png")


    def calc_port_stats(self, portfolio, portfolio_name):
        daily_returns = portfolio["total_value"].pct_change().dropna()
        cum_returns = portfolio["total_value"].iloc[-1] / portfolio["total_value"].iloc[0] - 1.0
        daily_std  = daily_returns.std()
        daily_mean = daily_returns.mean()

        if self.verbose:
            print(f"Name: {portfolio_name}  Cumulative Return: {cum_returns:.6f}, Daily Std: {daily_std:.6f}, Daily Mean: {daily_mean:.6f}")

    def convert_order_format(self, df_trades, symbol):
        trades = df_trades.copy()
        trades.iloc[0] = df_trades.iloc[0]
        for i in range(1, len(df_trades)):
            trades.iloc[i] = df_trades.iloc[i] - df_trades.iloc[i - 1]

        orders_list = []

        for i in range(len(trades)):
            shares_change = trades.iloc[i, 0]
            if shares_change > 0:
                orders_list.append([trades.index[i], symbol, "BUY", abs(shares_change)])
            elif shares_change < 0:
                orders_list.append([trades.index[i], symbol,"SELL", abs(shares_change)])

        df_orders = pd.DataFrame(orders_list, columns=["Date", "Symbol", "Order", "Shares"]).set_index("Date")
        return df_orders

    def add_empty_order(self, orders_df, symbol, date):
        if date not in orders_df.index:
            orders_df.loc[date] = [symbol, "BUY", 0]
            orders_df.sort_index(inplace=True)