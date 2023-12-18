from sklearn.metrics import accuracy_score

import numpy as np
import math
import copy

from nn.activation_functions import Softmax


class MyMultinomialRegression:
    def __init__(self, lr, n_iter):
        # Set hyper-parameters
        self.lr = lr
        self.n_iter = n_iter
        self._softmax = Softmax()
        self.losses = []
        self.train_accuracies = []

    def initialize_weights(self, num_features, num_classes):
        # Initialize weights and biases
        limit = 1 / math.sqrt(num_features)
        self.W = np.random.uniform(low=-limit, high=limit, size=(num_features, num_classes))

    def padding_input(self, X):
        X = self._transform_x(X)
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return X

    @staticmethod
    def one_hot_encode(y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        for i in range(len(y)):
            one_hot[i, y[i]] = 1
        return one_hot

    def fit(self, X, y):
        # Correct input format
        X = self._transform_x(X)
        X = self.padding_input(X)
        y = self._transform_y(y)

        num_classes = len(np.unique(y))
        num_features = X.shape[1]
        y_encoded = self.one_hot_encode(y, num_classes)

        self.initialize_weights(num_features, num_classes)      # Initialize weights and biases
        origin_lr = self.lr

        # Training the model
        # Epoch loop
        for i in range(self.n_iter):
            # Update learning rate
            self.lr = origin_lr / math.sqrt(i + 1)

            # Forward pass
            z = X @ self.W                          # Calculate the features
            a = self._softmax(z)                    # Softmax Func: From feature, calculate the output probability

            # Compute the loss
            loss = self.CrossEntropyLoss(y_encoded, a)                  # Binary cross entropy loss
            self.losses.append(loss)

            # Backpropagation
            dW = self.gradient_CrossEntropyLoss(X, y_encoded, a)        # Compute gradient

            # Update weights
            self.W -= self.lr * dW

            # Computer other metrics
            y_pred = np.argmax(a, axis=-1, keepdims=True)
            acc_score = accuracy_score(y, y_pred)                       # Compute accuracy score
            self.train_accuracies.append(acc_score)

            # print(f'Epoch {i}, lr = {self.lr}, CrossEntropyLoss: {loss}')
            # print(f'Epoch {i}, acc: {acc_score}')

    @staticmethod
    def CrossEntropyLoss(y_encoded, a):
        # Cross entropy LOSS
        epsilon = 1e-15
        # Limits predicted probabilities to the interval (epsilon, 1 - epsilon)
        a = np.clip(a, epsilon, 1 - epsilon)

        loss = -np.sum(y_encoded * np.log(a)) / len(a)
        return loss

    @staticmethod
    def gradient_CrossEntropyLoss(X, y_encoded, a):
        # Derivative of binary cross entropy
        # Multiply result is sum of gradient of the batch computation => Must be divided by batch_len
        dW = X.transpose() @ (a - y_encoded) / len(a)
        return dW

    def predict(self, X):
        # Correct input format
        X = self._transform_x(X)
        X = self.padding_input(X)

        # Forward pass & Calculate predicted output class
        a = self._softmax(X @ self.W)
        y_pred = np.argmax(a, axis=-1, keepdims=True)
        return y_pred

    @staticmethod
    def _transform_x(x):
        x = copy.deepcopy(x)
        return x

    @staticmethod
    def _transform_y(y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)