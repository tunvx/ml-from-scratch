import numpy as np
import math
import copy


class MyLinearRegression:
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.losses = []

    def initialize_weights(self, X):
        n_features = X.shape[1]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(low=-limit, high=limit, size=(n_features, 1))

    def padding_input(self, X):
        X = self._transform_x(X)
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return X

    def fit(self, X, y):
        # Prepare part
        X = self._transform_x(X)
        X = self.padding_input(X)
        y = self._transform_y(y)
        self.initialize_weights(X)
        origin_lr = self.lr

        # Calculate part
        for i in range(self.n_iter):
            # Update learning rate
            self.lr = origin_lr / math.sqrt(i + 1)
            # Calculate the output predict values
            h = X @ self.W
            # Calculate gradient, update weights
            dW = self.gradient_mes_loss(X, h, y)
            self.W -= self.lr * dW
            # Calculate metrics
            loss = self.compute_mse_loss(y, h)
            self.losses.append(loss)
            # print(f'Epoch {i}, lr = {self.lr}, mes_loss: {loss}')

    def compute_mse_loss(self, y, h):
        # Mean Squared Error LOSS
        return np.sum(np.power(y - h, 2)) / y.shape[0]

    def gradient_mes_loss(self, X, h, y):
        # Derivative of mean squared error LOSS
        # Sum of gradient of the batch computation, must be divided by y.shape[0]
        dW = X.transpose() @ (h - y) / y.shape[0]
        return dW

    def predict(self, X):
        # Prepare part
        X = self._transform_x(X)
        X = self.padding_input(X)
        # Calculate part
        return X @ self.W

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)
