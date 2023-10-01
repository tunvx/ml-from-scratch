from sklearn.metrics import accuracy_score

import numpy as np
import math
import copy

from utils.activation_functions import Sigmoid


class MyLogisticRegression:
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self._sigmoid = Sigmoid()
        self.losses = []
        self.train_accuracies = []

    def initailize_weights(self, X):
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
        self.initailize_weights(X)
        origin_lr = self.lr

        # Calculate part
        for i in range(self.n_iter):
            # Update learning rate
            self.lr = origin_lr / math.sqrt(i + 1)
            # Calculate the features
            z = X @ self.W
            # From feature, calculate the output probability
            h = self._sigmoid(z)
            # Calculate gradient, update weights
            dW = self.gradient_bce_loss(X, h, y)
            self.W -= self.lr * dW
            # Calculate metrics
            loss = self.compute_bce_loss(y, h)
            y_pred = np.where(h >= 0.5, 1, 0)
            acc_score = accuracy_score(y, y_pred)
            self.losses.append(loss)
            self.train_accuracies.append(acc_score)
            # print(f'Epoch {i}, lr = {self.lr}, bce_loss: {loss}')
            # print(f'Epoch {i}, acc: {acc_score}')

    def compute_bce_loss(self, y, h):
        # Binary cross entropy LOSS
        epsilon = 1e-15
        # Limits predicted values to the interval (epsilon, 1 - epsilon)
        h = np.clip(h, epsilon, 1 - epsilon)

        y_zero_loss = (1 - y) * np.log(1 - h)
        y_one_loss = y * np.log(h)
        return - np.mean(y_zero_loss + y_one_loss)
        # return self.binary_cross_entropy(y, h)

    def gradient_bce_loss(self, X, h, y):
        # Derivative of binary cross entropy
        # Sum of gradient of the batch computation, must be divided by y.shape[0]
        dW = X.transpose() @ (h - y) / y.shape[0]
        return dW

    def predict(self, X):
        # Prepare part
        X = self._transform_x(X)
        X = self.padding_input(X)
        # Calculate part
        h = self._sigmoid(X @ self.W)
        return np.where(h >= 0.5, 1, 0)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)





