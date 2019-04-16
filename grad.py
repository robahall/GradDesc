import numpy as np
import random as rand

import matplotlib.pyplot as plt
plt.interactive(False)
from matplotlib import animation, rc
rc('animation', html='jshtml')

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

    #plt.scatter(X[:, 1], y)
    #plt.savefig('data.png')

    #First epoch
    theta = np.array([1., 2]) ## Starting point
    learning_rate = 0.01
    update, MSE = gd(X, y, theta, learning_rate)

    #Test I epochs
    for i in range(9000):
        update, MSE = gd(X, y, update, learning_rate)
        theta = np.vstack((theta, update))
        print("Iteration: {}  Theta: {}".format(i, update))


    plt.scatter(theta[:, 0], theta[:, 1])
    plt.savefig('gd.png')


