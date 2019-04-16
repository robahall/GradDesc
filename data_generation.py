import numpy as np
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt
plt.interactive(False)

class Data(object):
    """Generate regression random numpy data set
    """


    def __init__(self, samples=10, features=2, informative=2, targets=1, jitter=0.):
        X, Y, coef = make_regression(n_samples=samples,
                                     n_features=features,
                                     n_informative=informative,
                                     n_targets=targets,
                                     noise=jitter,
                                     coef=True
                                     )
        self.X = X
        self.Y = Y
        self.coef = coef

    ### Redundant?
    def inspectX(self):
        return self.X

    def inspectCoef(self):
        return self.coef

    def inspectY(self):
        return self.Y

def gd(dataset, theta, alpha):
    """The standard gradient descent algo."""
    m = dataset.X.shape[0]
    error = dataset.X @ theta - dataset.Y
    grad = dataset.X.transpose() @ error ## multiplying each of the X's specific to features by the error it was off by
    theta = theta - alpha * (1/m) * grad
    MSE = np.average(error ** 2)
    return theta, MSE



if __name__ == "__main__":
    data = Data(samples=10, features=3, informative=3, targets=1, jitter=0.1)

    theta = np.array([1., 1, 1])
    learning_rate = 0.01
    update, MSE = gd(data, theta, learning_rate)

    for i in range(1000):
        update, MSE = gd(data, update, learning_rate)
        theta = np.vstack((theta, update))
        print("Iteration: {}  MSE: {}".format(i, MSE))

    print(theta[999])

    print(data.coef)
    plt.scatter(theta[:, 0], theta[:, 1], theta[:, 2])
    plt.ylim(0,10)
    plt.xlim(0, 10)
    plt.savefig('gd.png')


