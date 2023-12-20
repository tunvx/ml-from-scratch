import numpy as np


class Sigmoid:
    # Implement stable sigmoid function
    def __call__(self, x):
        positive_indices = x >= 0
        negative_indices = ~positive_indices
        sigmoid_x = np.empty_like(x)

        # Sigmoid for positive elements
        sigmoid_x[positive_indices] = 1.0 / (1.0 + np.exp(-x[positive_indices]))

        # Sigmoid for negative elements
        z = np.exp(x[negative_indices])
        sigmoid_x[negative_indices] = z / (1.0 + z)

        return sigmoid_x


class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_x = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return softmax_x


class Tanh:
    def __call__(self, x):
        return np.tanh(x)


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, x * self.alpha)