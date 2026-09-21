import numpy as np
import pandas as pd

from util import get_data


def compute_portvals(orders_df, start_val=100000, commission=0.0, impact=0.0):
    """
    Compute daily portfolio values from an order book.

    Parameters:
    orders_df (pd.DataFrame): DataFrame with index=dates and columns=['Symbol', 'Order', 'Shares']
    start_val (float): Starting portfolio value
    commission (float): Fixed commission charged per executed order
    impact (float): Market impact per trade, as a fraction of the traded value

    Returns:
    pd.DataFrame: DataFrame with 'total_value' column representing portfolio value over time
    """
    orders_df = orders_df.sort_index()
    date_range = pd.date_range(orders_df.index.min(), orders_df.index.max())
    symbols = orders_df['Symbol'].unique().tolist()
    prices = get_data(symbols, date_range)[symbols].ffill().bfill()

    # Signed share and cash changes per trading day
    share_changes = pd.DataFrame(0.0, index=prices.index, columns=symbols)
    cash_changes = pd.Series(0.0, index=prices.index)

    for date, order in orders_df.iterrows():
        shares = order['Shares']
        if shares == 0:
            continue
        symbol = order['Symbol']
        # Orders on non-trading days execute on the next trading day
        trade_date = prices.index[prices.index.searchsorted(date)]
        price = prices.loc[trade_date, symbol]
        sign = 1 if order['Order'].upper() == 'BUY' else -1

        share_changes.loc[trade_date, symbol] += sign * shares
        cash_changes[trade_date] -= sign * price * shares + price * shares * impact + commission

    holdings = share_changes.cumsum()
    cash = start_val + cash_changes.cumsum()
    total_value = cash + (holdings * prices).sum(axis=1)
    return total_value.to_frame('total_value')


def calculate_portfolio_stats(portfolio_values, portfolio_name=None, risk_free_rate=0.0):
    daily_returns = portfolio_values["total_value"].pct_change().dropna()
    cum_returns = portfolio_values["total_value"].iloc[-1] / portfolio_values["total_value"].iloc[0] - 1.0
    daily_std = daily_returns.std()
    daily_mean = daily_returns.mean()
    sharpe_ratio = np.sqrt(252) * (daily_mean - risk_free_rate/252) / daily_std if daily_std > 0 else 0

    stats = {
        "Portfolio": portfolio_name if portfolio_name else "Portfolio",
        "Cumulative Return": cum_returns,
        "Standard Deviation": daily_std,
        "Mean Daily Return": daily_mean,
        "Sharpe Ratio": sharpe_ratio
    }

    return stats
