import numpy as np
import unittest
import math
import nn


class TestLinearMethods(unittest.TestCase):
    def test_linear_init(self):
        linear = nn.Linear(n_input=10, n_output=5)
        self.assertEqual(linear.n_in, 10)
        self.assertEqual(linear.n_out, 5)
        self.assertEqual(linear.weight.shape, (10, 5))
        self.assertEqual(linear.bias.shape, (1, 5))
        self.assertEqual(linear.dW.shape, (10, 5))
        self.assertEqual(linear.db.shape, (1, 5))

    def test_linear_forward(self):
        linear = nn.Linear(n_input=10, n_output=5)
        x = np.zeros((3, 10), dtype=np.float32)
        y = linear.forward(x)
        error = np.sum(np.abs(y - np.zeros((3, 5))))
        self.assertEqual(y.shape, (3, 5))
        self.assertLess(error, 1e-6)

    def test_linear_backward(self):
        linear = nn.Linear(n_input=10, n_output=5)
        x = np.zeros((3, 10), dtype=np.float32)
        y = linear.forward(x)
        dy = np.ones_like(y)
        dx = linear.backward(np.ones_like(y))
        ## Exercise: should add test on error of dx at here
        # Calculate the numerical gradient using finite differences
        epsilon = 1e-6
        numerical_gradient = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_plus_eps = x.copy()
                x_plus_eps[i, j] += epsilon
                y_plus_eps = linear.forward(x_plus_eps)
                numerical_gradient[i, j] = np.sum(dy * (y_plus_eps - y)) / epsilon

        # Check the error between numerical and analytical gradients
        error = np.sum(np.abs(numerical_gradient - dx))
        self.assertLess(error, 1e-6)

        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(linear.dW.shape, linear.weight.shape)
        self.assertEqual(linear.db.shape, linear.bias.shape)


if __name__ == '__main__':
    unittest.main()

