import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logistic_regression(X, Y, alpha, max_batches):
    # initialise the gradient decent with a vector of zeros called theta
    xRows, xCols = X.shape
    J = list(range(xCols))
    I = list(range(xRows))
    theta = np.zeros(xCols)

    # Gradient Ascent
    for _ in range(max_batches):
        for j in J:
            for i in I:
                xi = X[i, :]
                yi = Y[i]
                xij = X[i, j]

                prediction_j = sigmoid(np.matmul(theta.T, xi))
                theta_j = theta[j] + alpha * (yi - prediction_j) * xij
                theta[j] = theta_j

    return lambda given_input: sigmoid(np.matmul(theta.T, given_input))  # Pretty sure this is correct


if __name__ == "__main__":
    # Question 5 #
    xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
    ys = np.array([0, 0, 0, 1, 1, 1])
    print(xs)
    print(ys)
    model = logistic_regression(xs, ys, 0.05, 10000)
    test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

    for test_input in test_inputs:
        val = model(test_input)
        print("{:.2f}".format(np.array(val).item()))