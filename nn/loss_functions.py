import numpy as np
import nn


class BCELoss:
    def __init__(self):
        self.stable_sigmoid = nn.Sigmoid()

    def __call__(self, ypred, y_true):
        return self.forward(ypred, y_true)

    def forward(self, ypred, y_true):
        # ypred: N x n_out (logit values)
        # ytrue: N (class label: int value in {0, 1})
        self.y_true = y_true.reshape(-1, 1)
        probability = self.stable_sigmoid(ypred)
        # Binary cross-entropy LOSS
        epsilon = 1e-15
        # Limits predicted probability values to the interval (epsilon, 1 - epsilon)
        probability = np.clip(probability, epsilon, 1 - epsilon)
        self.probability = probability

        y_zero_loss = (1 - self.y_true) * np.log(1 - probability)
        y_one_loss = self.y_true * np.log(probability)
        loss = - np.mean(y_zero_loss + y_one_loss)
        
        return loss

    def backward(self):
        # Backward pass: Compute the gradient of the loss with respect to the input (logit values)
        d_ypred = (self.probability - self.y_true)
        return d_ypred


class CrossEntropyLoss:
    def __init__(self):
        self.stable_softmax = nn.Softmax()

    def __call__(self, ypred, ytrue):
        # ypred: N x n_out (logit values)
        # ytrue: N (class label: int value in 0 --> n_out-1)
        return self.forward(ypred, ytrue)

    def forward(self, ypred, ytrue):
        # ypred: N x n_out (logit values)
        # ytrue: N (class label: int yvalue in 0-->n_out-1)
        # should return - sum_i sum_c y_ic \log mu_ic
        n, n_out = ypred.shape
        self.ypred = ypred
        self.ytrue = ytrue
        self.mu = self.stable_softmax(ypred)
        # loss = np.sum(-np.log(self.mu[range(n), ytrue]))
        loss = np.sum(-np.log(self.mu[np.arange(len(ytrue)), ytrue]))
        return loss

    def backward(self):
        # should return d_ypred, derivative of loss on ypred
        # d_ypred = mu - y (one-hot encoding of ytrue)
        n, n_out = self.ypred.shape
        d_ypred = self.mu.copy()
        d_ypred[range(n), self.ytrue] -= 1
        return d_ypred
    


