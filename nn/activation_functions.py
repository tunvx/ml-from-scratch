import numpy as np

from nn import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.forward(x)

    def info(self):
        print(f"(sigmoid): {self.__class__.__name__}()")

    def forward(self, z):
        """
        Compute the forward pass of the sigmoid activation function.

        Parameters:
        - z: Input to the sigmoid function (output of the linear layer)

        Returns:
        - A : Output of the sigmoid function, A = sigmoid(z)
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
        self.dz = dA * sigmoid_derivative
        return self.dz


class Softmax(BaseLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return self.forward(z)

    def info(self):
        print(f"(softmax): {self.__class__.__name__}()")

    def forward(self, z):
        """
        Compute the forward pass of the softmax activation function.

        Parameters:
        - z: Input to the softmax function (output of the linear layer)

        Returns:
        - A: Output of the softmax function
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward(self, dA):
        """
        Compute the backward pass of the softmax activation function.

        Parameters:
        - dA: Gradient of the loss with respect to the output of the softmax (dL/dsoftmax)

        Returns:
        - dz: Gradient of the loss with respect to the input to the softmax (dL/dz)
        """
        softmax_derivative = self.output * (1 - self.output)
        dz = dA * softmax_derivative
        return dz


class Tanh(BaseLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return self.forward(z)

    def info(self):
        print(f"(tanh): {self.__class__.__name__}()")

    def forward(self, z):
        """
        Compute the forward pass of the tanh activation function.

        Parameters:
        - z: Input to the tanh function (output of the linear layer)

        Returns:
        - A: Output of the tanh function, A = tanh(x)
        """
        self.input = z
        self.output = np.tanh(z)
        return self.output

    def backward(self, dA):
        """
        Compute the backward pass of the tanh activation function.

        Parameters:
        - dA: Gradient of the loss with respect to the output of the tanh (dL/dtanh)

        Returns:
        - dX: Gradient of the loss with respect to the input to the tanh (dL/dx)
        """
        tanh_derivative = 1.0 - self.output ** 2
        dX = dA * tanh_derivative
        return dX


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return self.forward(z)

    def info(self):
        print(f"(relu): {self.__class__.__name__}()")

    def forward(self, z):
        """
        Compute the forward pass of the ReLU activation function.

        Parameters:
        - z: Input to the ReLU function (output of the linear layer)

        Returns:
        - A : Output of the ReLU function, A = ReLU(z)
        """
        self.input = np.copy(z)
        self.output = np.copy(z)
        self.output[z < 0] = 0
        return self.output

    def backward(self, dA):
        """
        Compute the backward pass of the ReLU activation function.

        Parameters:
        - dA: Gradient of the loss with respect to the output of the ReLU (dL/dRelu)

        Returns:
        - dz: Gradient of the loss with respect to the input to the ReLU (dL/dz)
        """
        self.dz = dA * (self.input > 0)
        return self.dz


class LeakyReLU(BaseLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, z):
        return self.forward(z)

    def info(self):
        print(f"(leaky_relu): {self.__class__.__name__}()")

    def forward(self, z):
        """
        Compute the forward pass of the LeakyReLu activation function.

        Parameters:
        - z: Input to the LeakyReLu function (output of the linear layer)

        Returns:
        - A : Output of the LeakyReLu function, A = LeakyReLu(z)
        """
        self.input = np.copy(z)
        self.output = np.where(self.input > 0, z, self.alpha * z)
        return self.output

    def backward(self, dA):
        """
        Compute the backward pass of the LeakyReLu activation function.

        Parameters:
        - dA: Gradient of the loss with respect to the output of the LeakyReLu (dL/dLeakyReLu)

        Returns:
        - dz: Gradient of the loss with respect to the input to the LeakyReLu (dL/dz)
        """
        self.dz = dA * np.where(self.input > 0, 1, self.alpha)
        return self.dz
