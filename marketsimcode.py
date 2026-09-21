import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(orders_df, start_val=100000, commission=0.0, impact=0.0):
    """
    Compute portfolio values based on orders DataFrame.

    Parameters:
    orders_df (pd.DataFrame): DataFrame with index=dates and columns=['Symbol', 'Order', 'Shares']
    start_val (float): Starting portfolio value
    commission (float): Commission cost per trade
    impact (float): Market impact per trade

    Returns:
    pd.DataFrame: DataFrame with 'total_value' column representing portfolio value over time
    """
    date_range = pd.date_range(orders_df.index.min(), orders_df.index.max())
    symbols = orders_df['Symbol'].unique().tolist()
    prices = get_data(symbols, date_range)

    portfolio = pd.DataFrame(index=prices.index)
    portfolio['cash'] = start_val
    for symbol in symbols:
        portfolio[symbol] = 0

    for date, row in orders_df.iterrows():
        if isinstance(row, pd.Series):
            symbol = row['Symbol']
            shares = row['Shares']
            action = row['Order']

            price = prices.loc[date, symbol]
            impact_cost = price * impact * shares

            if action.upper() == 'BUY':
                cost = price * shares + impact_cost + commission
                portfolio.loc[date:, symbol] += shares
                portfolio.loc[date:, 'cash'] -= cost
            else:  # 'SELL'
                proceeds = price * shares - impact_cost - commission
                portfolio.loc[date:, symbol] -= shares
                portfolio.loc[date:, 'cash'] += proceeds
        else:
            for idx, order in row.iterrows():
                symbol = order['Symbol']
                shares = order['Shares']
                action = order['Order']

                if shares == 0:
                    continue

                price = prices.loc[date, symbol]
                impact_cost = price * impact * shares

                if action.upper() == 'BUY':
                    cost = price * shares + impact_cost + commission
                    portfolio.loc[date:, symbol] += shares
                    portfolio.loc[date:, 'cash'] -= cost
                else:  # 'SELL'
                    proceeds = price * shares - impact_cost - commission
                    portfolio.loc[date:, symbol] -= shares
                    portfolio.loc[date:, 'cash'] += proceeds

    portfolio['total_value'] = portfolio['cash']
    for symbol in symbols:
        portfolio['total_value'] += portfolio[symbol] * prices[symbol]

    return portfolio[['total_value']]

def get_stock_price(pricing_df, order_date, symbol):
    order_date = pd.to_datetime(order_date)
    return float(pricing_df.loc[order_date, symbol])

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