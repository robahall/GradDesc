import numpy as np
import random as rand

import matplotlib.pyplot as plt
plt.interactive(False)

## Create dataset


def generate_dataset(n):
    """Generates uniform random dataset for gradient descent algos."""

    x = np.ones([n, 2])
    x[:, 0] = np.random.uniform(-1,1,n)

    a = np.array([5., 2])
    b = np.random.rand(1, n)
    y = np.dot(x, a) + b

    return x, y

def sgd(x, y):
    theta = np.arrary([-1., 1.])
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    cost = np.sum(loss**2)/(2*m)





if __name__ == "__main__":
    x, y = generate_dataset(100)

    plt.scatter(x[:, 0], y)
    plt.savefig('figure.png')
