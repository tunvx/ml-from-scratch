import numpy as np


class BCELoss:
    def forward(self, y_true, probability):
        # Binary cross-entropy LOSS
        epsilon = 1e-15
        # Limits predicted probability values to the interval (epsilon, 1 - epsilon)
        probability = np.clip(probability, epsilon, 1 - epsilon)

        y_zero_loss = (1 - y_true) * np.log(1 - probability)
        y_one_loss = y_true * np.log(probability)
        loss = - np.mean(y_zero_loss + y_one_loss)
        self.y_true = y_true
        self.probability = probability
        return loss

    def backward(self):
        # Backward pass: Compute the gradient of the loss with respect to the input (probability)
        epsilon = 1e-15
        # Avoid division by zero
        self.probability = np.clip(self.probability, epsilon, 1 - epsilon)

        m = len(self.y_true)

        # Compute the gradient
        dprobability = - (self.y_true / self.probability - (1 - self.y_true) / (1 - self.probability)) / m

        return dprobability
