import numpy as np

RANDOM_SEED = 42


class BagLearner:
    """
    Bootstrap aggregating (bagging) ensemble learner.

    Trains ``bags`` instances of an arbitrary base learner, each on a bootstrap
    sample of the training data, and averages their predictions.

    Parameters
        learner (class) - Base learner class to instantiate for each bag.
        kwargs (dict)   - Keyword arguments forwarded to the base learner's constructor.
        bags (int)      - Number of learners in the ensemble.
        boost (bool)    - Reserved for boosting; currently unused.
        verbose (bool)  - If True, print debugging information.
    """
    def __init__(self, learner, kwargs=None, bags=50, boost=False, verbose=False):
        kwargs = kwargs or {}
        self.learners = [learner(**kwargs) for _ in range(bags)]

        if verbose:
            print(self.learners)

    def add_evidence(self, data_x, data_y):
        """
        Train each learner on a bootstrap sample of the training data.

        Parameters:
            data_x (numpy.ndarray) - Feature set used for training.
            data_y (numpy.ndarray) - Target values corresponding to data_x.
        """
        np.random.seed(RANDOM_SEED)

        num_rows_x = data_x.shape[0]
        for learner in self.learners:
            random_records = np.random.choice(num_rows_x, num_rows_x, replace=True)
            learner.add_evidence(data_x[random_records], data_y[random_records])

    def query(self, points):
        """
        Predict target values by averaging the ensemble's predictions.

        Parameters:
            points (numpy.ndarray) - Feature values to predict on.

        Returns:
            numpy.ndarray - Predicted values.
        """
        y_predict = []
        for learner in self.learners:
            y_predict.append(learner.query(points))
        return np.mean(np.array(y_predict), axis=0)
