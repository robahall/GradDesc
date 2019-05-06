import numpy as np
import grad_desc_algos as gda

from sklearn.datasets import make_regression


if __name__ == "__main__":
    features, target, coef = make_regression(n_samples=10000,
                                             n_features=2,
                                             n_informative=2,
                                             n_targets=1,
                                             noise=10.0,
                                             coef=True,
                                             random_state=1)

    theta = np.array([120, 40.])
    lr = 0.01
    epochs = 100
    momentum = 0



    thetas, MSE, count = gda.momentum_gd(features, target, theta, lr, epochs, momentum, batch_size=1)

    print("Determined weights: {}".format(thetas[-1,:]))
    print("Acutal Coefficients: {}".format(coef))
    print("End Mean Squared Error: {}".format(MSE[-3:,1]))
    print("Number of epochs: {}".format(count))


## See how my code stacks up to this: https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html