import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def grad_W(self, X: NDArray[np.float64], y_hat: NDArray[np.float64], y: NDArray[np.float64], n: int):
        return (2/n)*((X.T)@(y_hat-y))

    def grad_b(self, y_hat: NDArray[np.float64], y: NDArray[np.float64], n:int):
        return (2/n)*np.sum(y_hat-y)

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n,m = X.shape
        w = np.zeros(m)
        b=0.0
        for _ in range(epochs):
            y_hat = np.dot(X, w) + b

            MSE = (np.sum(y_hat-y)**2)*(1/n)
            w = w-(lr*(self.grad_W(X, y_hat, y, n)))
            b = b-(lr*(self.grad_b(y_hat, y, n)))

        return (np.round(w, 5), np.round(b, 5))
