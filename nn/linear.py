import math
import numpy as np


class Linear:
    def __init__(self, n_input, n_output):
        # Number of input and output features
        self.n_in = n_input
        self.n_out = n_output

        # Initialize weights and biases with small random values
        self.weight = np.random.normal(scale=1.0 / math.sqrt(n_input), size=(n_input, n_output))
        self.bias = np.zeros((1, n_output))

        # Initialize gradients with zeros
        self.dW = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

    def __call__(self, x):
        # Make the instance callable, allowing it to be used as a function
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Parameters:
        - x: Input data of shape (N, n_in)

        Returns:
        - Z: Output of the linear layer, Z = XW + b
        """
        # Save input for the backward pass
        self.x = x.copy()
        # Compute the linear transformation
        self.Z = np.matmul(x, self.weight) + self.bias  # Shape: (N, n_out)
        return self.Z

    def backward(self, dZ):
        """
        Backward pass through the linear layer.

        Parameters:
        - dZ: Gradient of the loss with respect to the output of the linear layer (dL/dZ)

        Returns:
        - dx: Gradient of the loss with respect to the input (dL/dx)
        """
        m = self.x.shape[0]  # Number of samples

        # Gradient of the loss with respect to the weights and biases
        self.dW = np.matmul(self.x.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        # Gradient with respect to the input for the next layer (if needed)
        dx = np.matmul(dZ, self.weight.T)
        return dx

    def parameters(self):
        """
        Get the parameters of the linear layer.

        Returns:
        - weight: Weight matrix
        - bias: Bias vector
        """
        return self.weight, self.bias

    def grads(self):
        """
        Get the gradients of the parameters.

        Returns:
        - dW: Gradient of the loss with respect to the weights
        - db: Gradient of the loss with respect to the biases
        """
        return self.dW, self.db

    def train(self):
        """
        Placeholder for training-related functionality.
        """
        pass

    def eval(self):
        """
        Placeholder for evaluation-related functionality.
        """
        pass
