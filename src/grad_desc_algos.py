import numpy as np


def gd(X, y, theta, learning_rate):
    """The standard gradient descent algo."""
    m = X.shape[0]
    error = np.dot(X, theta) - y
    grad = np.transpose((np.dot(X, theta) - y)) * X
    sum = np.sum(grad, axis = 0)
    update = theta - learning_rate * (1/m) * sum
    MSE = np.average(error ** 2)
    return update, error, MSE