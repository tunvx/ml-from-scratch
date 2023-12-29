import numpy as np


class BaseLayer:
    def __init__(self):
        """
        Placeholder for training-related functionality.
        """
        pass

    def info(self):
        """
        Placeholder for training-related functionality.
        """
        pass

    def __call__(self, x):
        """
        Placeholder for training-related functionality.
        """
        pass

    def forward(self, x):
        """
        Placeholder for training-related functionality.
        """
        pass

    def backward(self, dout):
        """
        Placeholder for training-related functionality.
        """
        pass

    def parameters(self):
        return [np.array([0.0], dtype=np.float64)]

    def grads(self):
        return [np.array([0.0], dtype=np.float64)]

    def train(self):
        """
        Placeholder for training-related functionality.
        """
        pass

    def eval(self):
        """
        Placeholder for evaluation-related functionality.
        """
        pass
