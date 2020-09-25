import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logistic_regression(X, Y, alpha, max_batches):
    # initialise the gradient decent with a vector of zeros called theta
    X.astype(np.float64)
    intercept = np.ones(X.shape[0], dtype=np.float64)
    X = np.insert(X, 0, intercept, axis=1)
    xRows, xCols = X.shape
    I = list(range(xRows))
    theta = np.zeros(xCols, dtype=np.float64)

    # Gradient Ascent
    for _ in range(max_batches):
        for i in I:
            xi = X[i, :]
            yi = Y[i]
            prediction = sigmoid(np.matmul(theta.T, xi))
            gradient = yi - prediction
            theta = theta + alpha * gradient * xi

    return lambda given_feature: sigmoid(np.matmul(theta.T, np.insert(given_feature, 0, [1])))


if __name__ == "__main__":
    # Question 5 #
    xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
    ys = np.array([0, 0, 0, 1, 1, 1])
    model = logistic_regression(xs, ys, 0.05, 10000)
    test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

    for test_input in test_inputs:
        print("{:.2f}".format(np.array(model(test_input)).item()))

