import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass of the sigmoid activation function.

        Parameters:
        - z: Input to the sigmoid function (output of the linear layer)

        Returns:
        - A / output: Output of the sigmoid function
        """
        z = z.copy()  # N x n_out
        positive_indices = z >= 0
        negative_indices = ~positive_indices
        self.output = np.empty_like(z)

        # Sigmoid for positive elements
        self.output[positive_indices] = 1.0 / (1.0 + np.exp(-z[positive_indices]))

        # Sigmoid for negative elements
        z = np.exp(z[negative_indices])
        self.output[negative_indices] = z / (1.0 + z)

        return self.output

    def backward(self, dA):
        """
        Compute the backward pass of the sigmoid activation function.

        Parameters:
        - dA: Gradient of the loss with respect to the output of the sigmoid (dL/dsigmoid)

        Returns:
        - dz: Gradient of the loss with respect to the input to the sigmoid (dL/dz)
        """
        sigmoid_derivative = self.output * (1 - self.output)
        dz = dA * sigmoid_derivative
        return dz

    def train(self):
        pass

    def eval(self):
        pass




