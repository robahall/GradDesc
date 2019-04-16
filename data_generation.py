import numpy as np
from sklearn.datasets import make_regression

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


if __name__ == "__main__":
    data = Data(samples=10, features=3, informative=3, targets=1, jitter=0.1)
    print(data.inspectX())
    print(data.inspectY())
    print(data.inspectCoef())