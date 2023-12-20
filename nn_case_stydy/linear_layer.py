import numpy as np


class Linear:
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros((1, n_output))

    def __call__(self, X):
        # Validate input dimensions
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(f"Input dimension ({X.shape[1]}) does not match the number of rows in weight matrix W ({self.W.shape[0]})")

        return X @ self.W + self.b