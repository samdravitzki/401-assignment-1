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


def linear_regression(X, y, basis_functions=None, penalty=0.):
    if basis_functions or (basis_functions is not None and len(basis_functions) == 0):
        old_X = X.copy()
        X = np.ones(old_X.shape[0]).reshape(-1, 1)
        for f in basis_functions:
            basis = [f(x) for x in old_X]
            X = np.insert(X, X.shape[1], basis, axis=1)
    else:
        intercept = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept, axis=1)

    penalty_matrix = np.identity(X.shape[1]) * penalty
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X) + penalty_matrix), X.T), y)


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

    # xs = np.array([[1, 2, 3, 4],
    #                [6, 2, 9, 1]]).T
    # ys = np.array([7, 5, 14, 8]).T
    # print(xs)
    # print(ys)
    # print(linear_regression(xs, ys))

    # Question 3 #
    # xs = np.array([0, 1, 2, 3, 4]).reshape((-1, 1))
    # ys = np.array([3, 6, 11, 18, 27])
    # # print(xs)
    # # print(ys)
    # # Can you see y as a function of x? [hint: it's quadratic.]
    # functions = [lambda x: x[0], lambda x: x[0] ** 2]
    # print(linear_regression(xs, ys, functions))
    #
    # xs = np.array([[1, 2, 3, 4],
    #                [6, 2, 9, 1]]).T
    # ys = np.array([7, 5, 14, 8])
    # print(linear_regression(xs, ys, []) == np.average(ys))

    # Question 4 #
    # Example 1
    # xs = np.arange(5).reshape((-1, 1))
    # ys = np.arange(1, 11, 2)
    # print(linear_regression(xs, ys), end="\n\n")
    # with np.printoptions(precision=5, suppress=True):
    #     print(linear_regression(xs, ys, penalty=0.1))

    # Example 2
    # we set the seed to some number so we can replicate the computation
    np.random.seed(0)
    xs = np.arange(-1, 1, 0.1).reshape(-1, 1)
    m, n = xs.shape
    # Some true function plus some noise:
    ys = (xs ** 2 - 3 * xs + 2 + np.random.normal(0, 0.5, (m, 1))).ravel()
    functions = [lambda x: x[0], lambda x: x[0] ** 2, lambda x: x[0] ** 3, lambda x: x[0] ** 4,
                 lambda x: x[0] ** 5, lambda x: x[0] ** 6, lambda x: x[0] ** 7, lambda x: x[0] ** 8]
    for penalty in [0, 0.01, 0.1, 1, 10]:
        with np.printoptions(precision=5, suppress=True):
            print(linear_regression(xs, ys, basis_functions=functions, penalty=penalty)
                  .reshape((-1, 1)), end="\n\n")

