import numpy as np
import math
import copy


class MyLinearRegression:
    def __init__(self, lr, n_iter):
        # Set hyper-parameters
        self.lr = lr
        self.n_iter = n_iter
        self.losses = []

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

        self.initialize_weights(X)                  # Initialize weights and biases
        origin_lr = self.lr

        # Training the model
        # Epoch loop
        for i in range(self.n_iter):
            # Update learning rate
            self.lr = origin_lr / math.sqrt(i + 1)

            # Forward pass
            a = X @ self.W                          # Calculate the output predict values

            # Compute the loss
            loss = self.MSELoss(y, a)               # Mean Squared Error
            self.losses.append(loss)

            # Backpropagation
            dW = self.gradient_MSELoss(X, a, y)     # Calculate gradient

            # Update weights
            self.W -= self.lr * dW

            # Print epoch result
            # print(f'Epoch {i}, lr = {self.lr}, mes_loss: {loss}')

    def MSELoss(self, y, h):
        # Mean Squared Error LOSS
        return np.sum(np.power(y - h, 2)) / y.shape[0]

    def gradient_MSELoss(self, X, h, y):
        # Derivative of mean squared error LOSS
        # Multiply result is sum of gradient of the batch computation => Must be divided by batch_len
        dW = X.transpose() @ (h - y) / y.shape[0]
        return dW

    def predict(self, X):
        # Correct input format
        X = self._transform_x(X)
        X = self.padding_input(X)

        # Forward pass / Calculate predicted output value
        return X @ self.W

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)
