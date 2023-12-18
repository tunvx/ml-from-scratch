import math
import copy
import numpy as np
from sklearn.metrics import accuracy_score

import nn
from nn import *


class MLPBinaryClassification:
    def __init__(self, n_input, n_hidden=4, n_output=1):
        # Initialize the neural network layers and activation functions
        self.linear1 = Linear(n_input, n_hidden)
        self.tanh1 = Tanh()
        self.linear2 = Linear(n_hidden, n_output)
        self.sigmoid = Sigmoid()

        # Define the binary cross-entropy loss function
        self.BCELoss = BCELoss()

        # Lists to store training loss and accuracy over epochs
        self.train_loss = []
        self.train_accuracy = []

    def forward_propagation(self, X):
        # Perform forward propagation through the network
        Z1 = self.linear1(X)
        A1 = self.tanh1(Z1)
        Z2 = self.linear2(A1)
        A2 = self.sigmoid(Z2)

        # Cache intermediate values for later use in backpropagation
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def backward_propagation(self, cache, X, Y):
        m = X.shape[0]

        # Retrieve cached values from forward propagation
        A1 = cache['A1']
        A2 = cache['A2']

        # Calculate gradients during backpropagation
        dZ2 = A2 - Y
        dW2 = A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.linear2.W.T

        dZ1 = dA1 * (1 - np.power(A1, 2))
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Store gradients in a dictionary
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def optimize_step(self, grads, learning_rate):
        # Update model parameters using the calculated gradients and learning rate
        self.linear1.W -= learning_rate * grads["dW1"]
        self.linear1.b -= learning_rate * grads["db1"]
        self.linear2.W -= learning_rate * grads["dW2"]
        self.linear2.b -= learning_rate * grads["db2"]

    def fit(self, X, Y, learning_rate=0.01, epochs=1000):
        # Transform input and target data
        X = self._transform_x(X)
        Y = self._transform_y(Y)

        for epoch in range(epochs):
            # Adaptive learning rate scheduling (e.g., inverse square root decay)
            lr = learning_rate / math.sqrt(epoch+1)

            # Forward propagation
            A2, cache = self.forward_propagation(X)

            # Backward propagation
            gradients = self.backward_propagation(cache, X, Y)

            # Update parameters
            self.optimize_step(gradients, lr)

            # Calculate and store training loss and accuracy
            loss = self.BCELoss(Y, A2)
            Y_pred = np.where(A2 >= 0.5, 1, 0)
            acc = accuracy_score(Y, Y_pred)
            self.train_loss.append(loss)
            self.train_accuracy.append(acc)

            # Print the loss every 50 epochs
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Perform predictions on new data
        A2, _ = self.forward_propagation(X)
        return np.where(A2 >= 0.5, 1, 0)

    def score(self, X, Y):
        X = self._transform_x(X)
        Y = self._transform_y(Y)

        A2, _ = self.forward_propagation(X)
        Y_pred = np.where(A2 >= 0.5, 1, 0)
        acc = accuracy_score(Y, Y_pred)
        return acc

    @staticmethod
    def _transform_x(x):
        # Deep copy the input data
        x = copy.deepcopy(x)
        return x

    @staticmethod
    def _transform_y(y):
        # Transform target data to a column vector
        y = copy.deepcopy(y)
        return y.reshape(-1, 1)