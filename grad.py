import numpy as np
import random as rand

import matplotlib.pyplot as plt
plt.interactive(False)

## Create dataset


def generate_dataset(n):
    """Generates uniform random dataset for gradient descent algos."""

    x = np.ones([n, 2])
    x[:, 1] = np.random.uniform(-1,1,n)

    theta = np.array([2, 5.])
    jitter = np.random.rand(1, n)
    y = np.dot(x, theta) + jitter

    return x, y

def gd(x, y, theta, alpha):
    m = X.shape[0]
    MSE = np.dot(X, theta) - y
    grad = np.transpose((np.dot(X, theta) - y)) * X
    sum = np.sum(grad, axis = 0)
    update = theta - alpha * (1/m) * sum
    return update, MSE




if __name__ == "__main__":
    X, y = generate_dataset(100)

    plt.scatter(X[:, 1], y)
    plt.savefig('figure.png')

    #First epoch
    theta = np.array([1., 2]) ## Starting point
    learning_rate = 0.01
    update, MSE = gd(X, y, theta, learning_rate)

    #Test 10 epochs
    for i in range(1000):
        update, MSE = gd(X, y, update, learning_rate)
        print("Iteration: {}  Theta: {}".format(i, update))
