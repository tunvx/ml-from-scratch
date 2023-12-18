from sklearn.metrics import accuracy_score

import numpy as np
import math
import copy

from nn.activation_functions import Sigmoid


class MyLogisticRegression:
    def __init__(self, lr, n_iter):
        # Set hyper-parameters
        self.lr = lr
        self.n_iter = n_iter
        self._sigmoid = Sigmoid()
        self.losses = []
        self.train_accuracies = []

    def initialize_weights(self, X):
        # Initialize weights and biases
        n_features = X.shape[1]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(low=-limit, high=limit, size=(n_features, 1))

    def padding_input(self, X):
        X = self._transform_x(X)
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return X

    def fit(self, X, y):
        # Correct input format
        X = self._transform_x(X)
        X = self.padding_input(X)
        y = self._transform_y(y)

        self.initialize_weights(X)                      # Initialize weights and biases
        origin_lr = self.lr

        # Training the model
        # Epoch loop
        for i in range(self.n_iter):
            # Update learning rate
            self.lr = origin_lr / math.sqrt(i + 1)

            # Forward pass
            z = X @ self.W                              # Calculate the features
            a = self._sigmoid(z)                        # Sigmoid Func: From feature, calculate the output probability

            # Compute the loss
            loss = self.BCELoss(y, a)                   # Binary cross entropy loss
            self.losses.append(loss)

            # Backpropagation
            dW = self.gradient_BCELoss(X, y, a)         # Compute gradient

            # Update weights
            self.W -= self.lr * dW

            # Computer other metrics
            y_pred = np.where(a >= 0.5, 1, 0)
            acc_score = accuracy_score(y, y_pred)       # Compute accuracy score
            self.train_accuracies.append(acc_score)

            # Print epoch result
            # print(f'Epoch {i}, lr = {self.lr}, bce_loss: {loss}')
            # print(f'Epoch {i}, acc: {acc_score}')

    @staticmethod
    def BCELoss(y, a):
        # Binary cross entropy LOSS
        epsilon = 1e-15
        # Limits predicted probabilities to the interval (epsilon, 1 - epsilon)
        a = np.clip(a, epsilon, 1 - epsilon)

        y_zero_loss = (1 - y) * np.log(1 - a)
        y_one_loss = y * np.log(a)
        return - np.mean(y_zero_loss + y_one_loss)

    @staticmethod
    def gradient_BCELoss(X, y, a):
        # Derivative of binary cross entropy
        # Multiply result is sum of gradient of the batch computation => Must be divided by batch_len
        dW = X.transpose() @ (a - y) / y.shape[0]
        return dW

    def predict(self, X):
        # Correct input format
        X = self._transform_x(X)
        X = self.padding_input(X)

        # Forward pass & Calculate predicted output class
        h = self._sigmoid(X @ self.W)
        return np.where(h >= 0.5, 1, 0)

    @staticmethod
    def _transform_x(x):
        x = copy.deepcopy(x)
        return x

    @staticmethod
    def _transform_y(y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)
