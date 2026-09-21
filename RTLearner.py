import numpy as np

RANDOM_SEED = 42


class RTLearner:
    def __init__(self, leaf_size, verbose=False):
        '''
        Random Tree regression learner.

        Builds a decision tree where each split uses a randomly chosen feature
        and splits on that feature's median value.

        Parameters
            leaf_size (int)  - Maximum number of samples aggregated at a leaf.
            verbose (bool)   - If True, print debugging information.
        '''
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_tree(self, data):
        data = data.astype(float)
        np.random.seed(RANDOM_SEED)
        if data.shape[0] <= self.leaf_size or np.all(data[:, -1] == data[0, -1]):
            return np.array([["leaf", np.mean(data[:, -1]), None, None]], dtype=object)

        best_feature = np.random.randint(0, data.shape[1] - 1)
        split_val = np.median(data[:, best_feature])

        if np.all(data[:, best_feature] <= split_val) or np.all(data[:, best_feature] > split_val):
            return np.array([["leaf", np.mean(data[:, -1]), None, None]])

        left_data = data[data[:, best_feature] <= split_val]
        right_data = data[data[:, best_feature] > split_val]

        left_tree = self.build_tree(left_data)
        right_tree = self.build_tree(right_data)

        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]], dtype=object)
        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        '''
        Train the learner on the given data.

        Parameters
            data_x (numpy.ndarray) – Feature values used to train the learner.
            data_y (numpy.ndarray) – Target values corresponding to data_x.
        '''
        data_y = np.array(data_y).reshape(-1)
        data = np.column_stack((data_x, data_y)).astype(float)
        self.tree = self.build_tree(data)

    def query(self, points):
        '''
        Predict target values for a set of query points.

        Parameters
            points (numpy.ndarray) – Each row is a single query.

        Returns
            numpy.ndarray – Predicted values.
        '''
        y_predict = np.empty(points.shape[0])
        for i, point in enumerate(points):
            node = 0

            while self.tree[node, 0] != "leaf":
                feature = int(self.tree[node, 0])
                split_val = self.tree[node, 1]
                if point[feature] <= split_val:
                    node += int(self.tree[node, 2])
                else:
                    node += int(self.tree[node, 3])

            y_predict[i] = self.tree[node, 1]
        return y_predict
