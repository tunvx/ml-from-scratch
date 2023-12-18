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

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_x = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return softmax_x

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class Tanh:
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        1 - self.__call__(x) ** 2
