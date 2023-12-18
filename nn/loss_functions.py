import numpy as np


class BCELoss:
    def __call__(self, y_true, probability):
        # Binary cross entropy LOSS
        epsilon = 1e-15
        # Limits predicted probability values to the interval (epsilon, 1 - epsilon)
        probability = np.clip(probability, epsilon, 1 - epsilon)

        y_zero_loss = (1 - y_true) * np.log(1 - probability)
        y_one_loss = y_true * np.log(probability)
        return - np.mean(y_zero_loss + y_one_loss)


class MSELoss:
    def __call__(self, y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2)) / y_true.shape[0]
