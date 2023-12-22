import math
import numpy as np


class SGDOptimizer:
    def __init__(self, model, learning_rate, regularization=0.0, decay_learning_rate=True):
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.decay_learning_rate = decay_learning_rate
        self.current_step = 0

    def parameters(self):
        return self.model.parameters()

    def grads(self):
        return self.model.grads()

    def zero_grad(self):
        for g in self.model.grads():
            g.fill(0)

    def step(self):
        # input is the derivative of loss function on the output of the model
        # dW has been computed by backward functions
        # perform a gradient step W = W - 1/sqrt(t) lambda dW
        # the learning rate is reduced over time for convergence
        self.current_step += 1
        for p, g in zip(self.parameters(), self.grads()):
            g = self.regularization * p + g     # apply L2 regularization
            g = np.clip(g, -1, 1)               # clip gradients to avoid exploding gradients
            
            if self.decay_learning_rate:
                # Update the model parameter using gradient descent with a decaying learning rate
                p -= 1.0 / math.sqrt(self.current_step) * self.learning_rate * g
            else:
                # Alternatively, without decaying learning rate:
                p -= self.learning_rate * g