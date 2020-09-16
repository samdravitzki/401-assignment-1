# Assignment 2
import numpy as np


def linear_regression_2d(data):
    X = [feature_value for (feature_value, _) in data]
    Y = [response_value for (_, response_value) in data]

    # the slope of the line of least squares fit
    slope = (len(data) * np.dot(X, Y) - sum(X) * sum(Y)) / (len(data) * np.dot(X, X) - pow(sum(X), 2))
    # the intercept of the line of least squares fit
    intercept = (sum(Y) - slope * sum(X)) / len(data)

    return slope, intercept


def linear_regression(X, y, basis_functions=None):
    if basis_functions or (basis_functions is not None and len(basis_functions) == 0):
        old_X = X.copy()
        X = np.ones(old_X.shape[0]).reshape(-1, 1)
        for f in basis_functions:
            basis = [f(x) for x in old_X]
            X = np.insert(X, X.shape[1], basis, axis=1)
    else:
        intercept = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept, axis=1)

    return (np.linalg.inv(X.T @ X) @ X.T) @ y


if __name__ == "__main__":
    # Question 1 #

    # data = [(1, 4), (2, 7), (3, 10)] # (feature_value, response_value)
    # m, c = linear_regression_2d(data)
    # print(m, c)
    # print(4 * m + c)

    # Question 2 #

    # xs = np.arange(5).reshape((-1, 1))
    # ys = np.arange(1, 11, 2)
    # print(xs)
    # print(ys)
    # print(linear_regression(xs, ys))

    xs = np.array([[1, 2, 3, 4],
                   [6, 2, 9, 1]]).T
    ys = np.array([7, 5, 14, 8]).T
    print(xs)
    print(ys)
    print(linear_regression(xs, ys))

    # Question 3 #
    # xs = np.array([0, 1, 2, 3, 4]).reshape((-1, 1))
    # ys = np.array([3, 6, 11, 18, 27])
    # print(xs)
    # print(ys)
    # # Can you see y as a function of x? [hint: it's quadratic.]
    # functions = [lambda x: x[0], lambda x: x[0] ** 2]
    # print(linear_regression(xs, ys, functions))

    xs = np.array([[1, 2, 3, 4],
                   [6, 2, 9, 1]]).T
    ys = np.array([7, 5, 14, 8])
    print(linear_regression(xs, ys, []) == np.average(ys))
