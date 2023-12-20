import numpy as np
import unittest

from nn import Sigmoid


class TestSigmoidMethods(unittest.TestCase):
    def test_sigmoid_forward_positive_elements(self):
        sigmoid = Sigmoid()
        z = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]])
        expected_output = 1.0 / (1.0 + np.exp(-z))
        actual_output = sigmoid.forward(z)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6)

    def test_sigmoid_forward_negative_elements(self):
        sigmoid = Sigmoid()
        z = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.0, -2.0]])
        expected_output = np.exp(z) / (1.0 + np.exp(z))
        actual_output = sigmoid.forward(z)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        sigmoid.output = np.array([[0.73105858, 0.88079708, 0.95257413],
                                   [0.62245933, 0.73105858, 0.88079708]])
        dA = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        expected_dz = dA * sigmoid.output * (1 - sigmoid.output)
        actual_dz = sigmoid.backward(dA)
        np.testing.assert_allclose(actual_dz, expected_dz, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
