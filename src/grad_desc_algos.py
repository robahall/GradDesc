import numpy as np


def gd(X, y, theta, learning_rate):
    """The standard gradient descent algo."""
    m = X.shape[0]
    error = np.dot(X, theta) - y
    grad = np.dot(error, X)
    update = theta - learning_rate * (1/m) * grad
    MSE = np.average(error ** 2)
    return update, MSE


