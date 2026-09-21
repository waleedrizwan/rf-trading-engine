import datetime as dt

import pytest

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

IN_SAMPLE = (dt.datetime(2008, 1, 1), dt.datetime(2008, 12, 31))
OUT_OF_SAMPLE = (dt.datetime(2009, 1, 1), dt.datetime(2009, 12, 31))


def assert_valid_trades(trades):
    holdings = trades["Position"].cumsum()
    assert holdings.isin([-1000, 0, 1000]).all()
    assert trades["Position"].isin([-2000, -1000, 0, 1000, 2000]).all()


def test_manual_strategy_produces_valid_trades(market_data):
    trades = ManualStrategy().testPolicy("ABC", *IN_SAMPLE)
    assert (trades["Position"] != 0).any()
    assert_valid_trades(trades)


def test_strategy_learner_produces_valid_trades(market_data):
    learner = StrategyLearner()
    learner.add_evidence("ABC", *IN_SAMPLE)
    for period in (IN_SAMPLE, OUT_OF_SAMPLE):
        trades = learner.testPolicy("ABC", *period)
        assert_valid_trades(trades)
        assert trades["Position"].sum() == 0  # flat at the end


def test_strategy_learner_trades_less_with_higher_impact(market_data):
    counts = []
    for impact in (0.0, 0.02):
        learner = StrategyLearner(impact=impact)
        learner.add_evidence("ABC", *IN_SAMPLE)
        counts.append((learner.testPolicy("ABC", *IN_SAMPLE)["Position"] != 0).sum())
    assert counts[1] <= counts[0]


def test_strategy_learner_requires_training():
    with pytest.raises(RuntimeError):
        StrategyLearner().testPolicy("ABC", *OUT_OF_SAMPLE)
