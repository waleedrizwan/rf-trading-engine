import numpy as np
import pandas as pd
import pytest


def write_prices(directory, symbol, prices):
    df = pd.DataFrame({"Adj Close": prices.values}, index=prices.index)
    df.index.name = "Date"
    df.to_csv(directory / f"{symbol}.csv")


@pytest.fixture
def market_data(tmp_path, monkeypatch):
    """Synthetic SPY and ABC price files (business days, 2007-2010) in a temp data dir."""
    dates = pd.bdate_range("2007-01-01", "2010-12-31")
    rng = np.random.RandomState(0)
    t = np.arange(len(dates))
    abc = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    write_prices(tmp_path, "SPY", pd.Series(100 + 0.01 * t, index=dates))
    write_prices(tmp_path, "ABC", pd.Series(abc, index=dates))
    monkeypatch.setenv("MARKET_DATA_DIR", str(tmp_path))
    return tmp_path
