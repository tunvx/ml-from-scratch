import numpy as np
import copy


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


class MyKNNClassifier:
    """ K Nearest Neighbors classifier.

        Parameters:
        -----------
        k: int
            The number of closest neighbors that will determine the class of the
            sample that we wish to predict.
        """
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X):
        y_pred = np.empty(X.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X):
            # Find indices of the K-nearest neighbors for each data point in X
            k_nearest_indices = np.argsort([euclidean_distance(test_sample, x) for x in self.X_train])[:self.k]
            # Get labels of k nearest neighbors
            k_nearest_neighbors = np.array([self.y_train[idx] for idx in k_nearest_indices])
            # Predict the class with the highest count for each data point in X
            y_pred[i] = self._vote(k_nearest_neighbors)
        return y_pred