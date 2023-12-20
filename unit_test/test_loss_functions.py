import math

import numpy as np
import unittest

from nn import BCELoss, CrossEntropyLoss

# class TestBCELossMethods(unittest.TestCase):
#     def test_bce_loss_forward(self):
#         bce_loss = BCELoss()
#         y_true = np.array([1, 0, 1, 0])
#         probability = np.array([0.9, 0.2, 0.8, 0.1])
#
#         expected_loss = - np.mean((1 - y_true) * np.log(1 - probability) + y_true * np.log(probability))
#         actual_loss = bce_loss.forward(y_true, probability)
#
#         np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6)
#
#     def test_bce_loss_backward(self):
#         bce_loss = BCELoss()
#         y_true = np.array([1, 0, 1, 0])
#         probability = np.array([0.9, 0.2, 0.8, 0.1])
#
#         # Compute the gradient numerically for each element
#         epsilon = 1e-6
#         numerical_gradient = np.zeros_like(probability)
#         for i in range(len(probability)):
#             probability_plus_epsilon = probability.copy()
#             probability_plus_epsilon[i] += epsilon
#             loss_plus_epsilon = bce_loss.forward(y_true, probability_plus_epsilon)
#
#             probability_minus_epsilon = probability.copy()
#             probability_minus_epsilon[i] -= epsilon
#             loss_minus_epsilon = bce_loss.forward(y_true, probability_minus_epsilon)
#
#             numerical_gradient[i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
#
#         # Compute the gradient using the backward method
#         actual_gradient = bce_loss.backward()
#
#         # Compare the numerical and actual gradients
#         np.testing.assert_allclose(actual_gradient, numerical_gradient, rtol=1e-5)

class TestCEMethods(unittest.TestCase):
    def test_ce_forward(self):
      ypred = np.zeros((10, 5))
      ytrue = np.array([0,1,2,3,4,0,1,2,3,4], dtype=int)
      ce = CrossEntropyLoss()
      loss = ce.forward(ypred, ytrue)
      self.assertAlmostEqual(loss, -10*math.log(1/5))

    def test_ce_backward(self):
      ypred = np.zeros((10, 5))
      ytrue = np.array([0,1,2,3,4,0,1,2,3,4], dtype=int)
      ce = CrossEntropyLoss()
      loss = ce.forward(ypred, ytrue)
      d_ypred = ce.backward()
      desired = np.ones((10,5))*0.2
      desired[range(10), ytrue] -= 1
      error = np.sum(np.abs(d_ypred-desired))
      self.assertAlmostEqual(error, 0)

if __name__ == '__main__':
    unittest.main()
