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
    print(m)
    grad = np.dot(X, theta) - y
    print(X)
    print(theta)
    print(grad)
    return theta - alpha * (1/m) * np.sum(grad)





if __name__ == "__main__":
    X, y = generate_dataset(100)

    plt.scatter(X[:, 1], y)
    plt.savefig('figure.png')

    theta = np.array([1., 2])
    learning_rate = 0.01
    update = gd(X, y, theta, learning_rate)
    print(update)
