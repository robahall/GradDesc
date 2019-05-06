import numpy as np

#TODO: Unit test algorithms;
#TODO: Make sure that algorithms are generalized to different numpy array sizes
#TODO: Momentum does not work for m = 1.0. m = 1.0 should be normal SGD.
#TODO: Have early stopping when MSE is below a certain value.Something is currently getting hung up on loop.

momentum = 1


def gd(X, y, theta, learning_rate, velocity_vector = 0, momentum=0):

    """The standard gradient descent algorithm.
    Notes:
    error => difference between predicted and actual (y-hat - y)

    grad => partial derivative of the error surface. Essentially, taking the values of X for each of the thetas
            and multiplying the errors from each of thetas (weights) for each of the linear equations and then summing
            across the respective weights. (Goal is find the average which is in next step)

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.

    MSE => provide the mean squared error of the current prediction error.

    Won't work for collaborative filtering systems.
    """

    if momentum < 1 or momentum > 0:
        m = X.shape[0]
        error = np.dot(X, theta) - y
        grad = learning_rate * (1 / m) * np.dot(X.transpose(), error) # Transpose to align all the weights and the errors generate
        velocity_vector = (momentum * velocity_vector) - grad
        update = theta + velocity_vector
        MSE = np.average(error ** 2)
        return update, MSE, velocity_vector

    else:
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
    results = np.array([0,0])     # starting point

    for i in range(epochs):
        weights, MSE = gd(X, y, weights, learning_rate)
        cumulative_weights = np.vstack([cumulative_weights, weights])   # provide a numpy stack for the weights at each
                                                                        # iteration
        results = np.vstack([results, np.array([i+1, MSE])])

    return cumulative_weights, results


def stochastic_gradient_descent(X, y, weights, learning_rate, epochs):

    """Performs stochastic gradient descent (SGD)
    Stochastic gradient descent randomly shuffles the linear equations of data set and then performs gradient descent
    updating after each linear equation.

    Notes:
    cumulative weights => returns a numpy array that includes weights at each iteration

    results => returns iteration and mean squared error for each epoch.

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.
        """

    cumulative_weights = weights # initialize weights

    results = np.array([[0,0]])   # starting point

    for i in range(epochs):
        y = np.reshape(y, (y.shape[0], 1))  # Takes a single dimensional array and converts to multi-dimensional.
                                            # Need to generalize here.
        Xy = np.concatenate((X,y), axis = 1)  # combine X and y to ensure each linear equation stays the same
        np.random.shuffle(Xy)

        X = Xy[:, :X.shape[1]] # Split X  back out
        y = Xy[:, -1]  # Split y back out

        for xi, yi in zip(X,y):
            weights, MSE = gd(xi, yi, weights, learning_rate)
            cumulative_weights = np.vstack([cumulative_weights, weights])
            results = np.vstack([results, np.array([i+1, MSE])])  # Will return multiple values for each iteration

        return cumulative_weights, results


def minibatch_gradient_descent(X, y, weights, learning_rate, epochs, batch_size):
    """
    Performs minibatch gradient descent (mGD)

    mGD randomly shuffles the linear equations of data set and then performs gradient descent
    updating after each mini batch of selected linear equation.

    Notes:
    cumulative weights => returns a numpy array that includes weights at each iteration

    results => returns iteration and mean squared error for each epoch.

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.

    batch_size => when batch size = 1, mGD is stochastic gradient descent
                  when batch size = sample size, mGD is batch gradient descent

    This algorithm could be solution for previous written equations, though errors with MSE due to calculation off
    batch size.
    """

    cumulative_weights = weights  # initialize weights
    results = np.array([[0, 0]])  # starting point

    for j in range(epochs):
        y = np.reshape(y, (y.shape[0], 1))  # Takes a single dimensional array and converts to multi-dimensional.
        # Need to generalize here.
        Xy = np.concatenate((X, y), axis=1)  # combine X and y to ensure each linear equation stays the same
        np.random.shuffle(Xy)

        m = Xy.shape[0]

        if m % batch_size != 0:
            # For batch size where you have a remainder at the end of the fitting. Modulo != 0
            for i in range(m // batch_size):
                X = Xy[batch_size * (i):batch_size * (i + 1), :X.shape[1]]  # Split X  back out
                y = Xy[batch_size * (i):batch_size * (i + 1):, -1]  # Split y back out
                weights, MSE = gd(X, y, weights, learning_rate)
                cumulative_weights = np.vstack([cumulative_weights, weights])
                results = np.vstack([results, np.array([j + 1, MSE])])  # Will return multiple values for each iteration
            X = Xy[batch_size * (m // batch_size):, :X.shape[1]]  # Split X  back out
            y = Xy[batch_size * (m // batch_size):, -1]
            weights, MSE = gd(X, y, weights, learning_rate)
            cumulative_weights = np.vstack([cumulative_weights, weights])
            results = np.vstack([results, np.array([j + 1, MSE])])

        else:
            # For batch size where you do not have a remainder at the end of the fitting. Modulo = 0.
            for i in range(m // batch_size):
                X = Xy[batch_size * (i):batch_size * (i + 1), :X.shape[1]]  # Split X  back out
                y = Xy[batch_size * (i):batch_size * (i + 1):, -1]  # Split y back out
                weights, MSE = gd(X, y, weights, learning_rate)
                cumulative_weights = np.vstack([cumulative_weights, weights])
                results = np.vstack([results, np.array([j + 1, MSE])])


    return cumulative_weights, results



def momentum_gd(X, y, weights, learning_rate, epochs, momentum, batch_size = 1):

    """Performs stochastic gradient descent (SGD)
    Stochastic gradient descent randomly shuffles the linear equations of data set and then performs gradient descent
    updating after each linear equation.

    Notes:
    cumulative weights => returns a numpy array that includes weights at each iteration

    results => returns iteration and mean squared error for each epoch.

    update => taking initial inputted theta and subtracting a scaling of the average sum of squares.
        """

    cumulative_weights = weights  # initialize weights
    velocity_vector = np.ones(X.shape[1]) # Need to update
    results = np.array([[0,0]])   # starting point
    count = 0

    if momentum == 0:
        cumulative_weights, results = minibatch_gradient_descent(X, y, weights, learning_rate, epochs, batch_size)
        return cumulative_weights, results

    while count < epochs or MSE > 1:
        y = np.reshape(y, (y.shape[0], 1))  # Takes a single dimensional array and converts to multi-dimensional.
                                            # Need to generalize here.
        Xy = np.concatenate((X,y), axis = 1)  # combine X and y to ensure each linear equation stays the same
        np.random.shuffle(Xy)

        X = Xy[:, :X.shape[1]] # Split X  back out
        y = Xy[:, -1]  # Split y back out

        for c, xi, yi in enumerate(zip(X,y)):
            weights, MSE, velocity_vector = gd(xi, yi, weights, learning_rate, velocity_vector, momentum) ## Need to update
            cumulative_weights = np.vstack([cumulative_weights, weights])
            results = np.vstack([results, np.array([c, MSE])])# Will return multiple values for each iteration
        count = c

    return cumulative_weights, results, count
