import unittest
import numpy as np

from nn import CrossEntropyLoss
from optim import SGDOptimizer
from supervised_learning import MyMLPClassifier


class TestSGDMethods(unittest.TestCase):
    def test_sgd_init(self):
        model = MyMLPClassifier(n_input=10, hiddens=[5, 2], n_classes=2)
        sgd = SGDOptimizer(model, learning_rate=0.2, regularization=0.1)
        param = sgd.parameters()
        grad = sgd.grads()

        for p, g in zip(param, grad):
            self.assertEqual(p.shape, g.shape)
        self.assertEqual(sgd.learning_rate, 0.2)
        self.assertEqual(sgd.regularization, 0.1)

    def test_sgd_zero_grad(self):
        model = MyMLPClassifier(n_input=10, hiddens=[5, 2], n_classes=2)
        sgd = SGDOptimizer(model, learning_rate=0.2)
        sgd.zero_grad()

        for g in sgd.grads():
            self.assertAlmostEqual(np.sum(np.abs(g)), 0)

    def test_sgd_step(self):
        model = MyMLPClassifier(n_input=10, hiddens=[5, 2])
        sgd = SGDOptimizer(model, learning_rate=0.2)
        loss_func = CrossEntropyLoss()

        x = np.zeros((3, 10), dtype=np.float32)
        ytrue = np.array([0, 1, 0], dtype=int)

        ypred = model.forward(x)
        loss = loss_func.forward(ypred, ytrue)

        sgd.zero_grad()
        dout = loss_func.backward()
        dx = model.backward(dout)
        sgd.step()

    def test_sgd_n_step(self):
        model = MyMLPClassifier(n_input=10, hiddens=[5, 2], n_classes=2)
        sgd = SGDOptimizer(model, learning_rate=0.02)
        loss_func = CrossEntropyLoss()
        n_step = 10

        x = np.zeros((3, 10), dtype=np.float32)
        ytrue = np.array([0, 1, 0], dtype=int)

        print()
        for step in range(n_step):
            model.train()
            ypred = model.forward(x)
            loss = loss_func.forward(ypred, ytrue)

            print(f"step {step} {loss:.4f}")
            if step > 0:
                self.assertLess(loss, old_loss)  ## SGD step reduces loss function
            old_loss = loss

            sgd.zero_grad()
            dout = loss_func.backward()
            dx = model.backward(dout)
            sgd.step()


if __name__ == '__main__':
    unittest.main()
