import numpy as np


def gd(X, y, theta, learning_rate):

    """The standard gradient descent algorithm.
    Notes:
    error => difference between predicted and actual (y-hat - y)

    grad => partial derivative of the error surface. Essentially, taking the values of X for each of the thetas
            and multiplying the errors from each of thetas (weights) for each of the linear equations and then summing
            across the respective weights. (Goal is find the average which is in next step)

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.

    MSE => provide the mean squared error of the current prediction error.
    """

    m = X.shape[0]
    error = np.dot(X, theta) - y
    grad = np.dot(X.transpose(), error)
    update = theta - learning_rate * (1/m) * grad
    MSE = np.average(error ** 2)
    return update, MSE

def batch_gradient_descent(X, y, weights, learning_rate, epochs):
    """Perform batch gradient descent.
    Batch gradient descent iterates through all sample before updating.

    Notes:
    cumulative weights => returns a numpy array that includes weights at each iteration

    results => returns iteration and mean squared error for each epoch.

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.

    """

    cumulative_weights = weights  # initialize weights
    results = np.array([0,0])  # starting point
    for i in range(epochs):
        weights, MSE = gd(X, y, weights, learning_rate)
        cumulative_weights = np.vstack([cumulative_weights, weights])   # provide a numpy stack for the weights at each
                                                                        # iteration
        results = np.vstack([results, np.array([i+1, MSE])])

    return cumulative_weights, results




