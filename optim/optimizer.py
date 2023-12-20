import math
import numpy as np


class SGDOptimizer:
    def __init__(self, model, learning_rate, regularization=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.current_step = 0

    def parameters(self):
        return self.model.parameters()

    def grads(self):
        return self.model.grads()

    def zero_grad(self):
        for g in self.grads():
            g.fill(0)

    def step(self):
        # input is the derivative of loss function on the output of the model
        # dW has been computed by backward functions
        # perform a gradient step W = W - 1/sqrt(t) lambda dW
        # the learning rate is reduced over time for convergence
        self.current_step += 1
        for p, g in zip(self.parameters(), self.grads()):
            lr = 1.0 / math.sqrt(self.current_step) * self.learning_rate
            p -= lr * g