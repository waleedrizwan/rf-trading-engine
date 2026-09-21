import numpy as np

from BagLearner import BagLearner
from RTLearner import RTLearner


def make_data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 5)
    y = x @ np.array([1.0, 2.0, -1.0, 0.5, 3.0])
    return x, y


def test_rtlearner_splits_on_multiple_features():
    x, y = make_data()
    np.random.seed(1)
    learner = RTLearner(leaf_size=5)
    learner.add_evidence(x, y)
    features = {int(f) for f in learner.tree[:, 0] if f != "leaf"}
    assert len(features) > 1


def test_rtlearner_leaf_size_one_memorizes_training_data():
    x, y = make_data(n=100)
    np.random.seed(1)
    learner = RTLearner(leaf_size=1)
    learner.add_evidence(x, y)
    np.testing.assert_allclose(learner.query(x), y)


def test_baglearner_generalizes_out_of_sample():
    x, y = make_data()
    bag = BagLearner(RTLearner, {"leaf_size": 5}, bags=20)
    bag.add_evidence(x[:400], y[:400])
    corr = np.corrcoef(bag.query(x[400:]), y[400:])[0, 1]
    assert corr > 0.85


def test_baglearner_is_reproducible():
    x, y = make_data()
    preds = []
    for _ in range(2):
        bag = BagLearner(RTLearner, {"leaf_size": 5}, bags=10)
        bag.add_evidence(x, y)
        preds.append(bag.query(x))
    np.testing.assert_allclose(preds[0], preds[1])
