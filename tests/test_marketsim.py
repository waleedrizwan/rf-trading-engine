import pandas as pd
import pytest

from marketsimcode import compute_portvals, compute_portvals_from_trades
from util import get_data


def orders(rows):
    df = pd.DataFrame(rows, columns=["Date", "Symbol", "Order", "Shares"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def price(date):
    return get_data(["ABC"], pd.date_range(date, date))["ABC"].iloc[0]


def test_round_trip_matches_hand_calculation(market_data):
    book = orders([["2008-01-02", "ABC", "BUY", 100], ["2008-02-01", "ABC", "SELL", 100]])
    pv = compute_portvals(book, start_val=10000, commission=9.95, impact=0.005)
    p0, p1 = price("2008-01-02"), price("2008-02-01")
    expected = 10000 - 100 * p0 * 1.005 - 9.95 + 100 * p1 * 0.995 - 9.95
    assert pv["total_value"].iloc[-1] == pytest.approx(expected)


def test_zero_share_orders_are_free(market_data):
    book = orders([["2008-01-02", "ABC", "BUY", 0], ["2008-02-01", "ABC", "BUY", 0]])
    pv = compute_portvals(book, start_val=10000, commission=9.95, impact=0.005)
    assert (pv["total_value"] == 10000).all()


def test_weekend_order_executes_next_trading_day(market_data):
    book = orders([["2008-01-05", "ABC", "BUY", 10], ["2008-01-10", "ABC", "BUY", 0]])
    pv = compute_portvals(book, start_val=10000)
    assert pv.index[0] == pd.Timestamp("2008-01-07")
    assert pv["total_value"].iloc[0] == pytest.approx(10000)


def test_trades_and_orders_simulations_agree(market_data):
    dates = pd.bdate_range("2008-01-02", "2008-03-31")
    trades = pd.DataFrame(0.0, index=dates, columns=["Position"])
    trades.loc["2008-01-15", "Position"] = 1000
    trades.loc["2008-02-15", "Position"] = -2000
    trades.loc["2008-03-31", "Position"] = 1000
    book = orders([[d, "ABC", "BUY" if s > 0 else "SELL", abs(s)] for d, s in trades["Position"].items()])

    from_trades = compute_portvals_from_trades(trades, "ABC", 100000, 9.95, 0.005)
    from_orders = compute_portvals(book, 100000, 9.95, 0.005)
    pd.testing.assert_series_equal(from_trades["total_value"], from_orders["total_value"])
