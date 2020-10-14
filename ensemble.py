def voting_ensemble(classifiers):
    ensemble_predictor = lambda data: [classifier(data) for classifier in classifiers]
    return lambda data: max(set(ensemble_predictor(data)), key=ensemble_predictor(data).count)


def bagging_model(learner, dataset, n_models, sample_size):
    pass


if __name__ == "__main__":
    # Question 1 #
    # Example 1
    # Modelling y > x^2
    # classifiers = [
    #     lambda p: 1 if 1.0 * p[0] < p[1] else 0,
    #     lambda p: 1 if 0.9 * p[0] < p[1] else 0,
    #     lambda p: 1 if 0.8 * p[0] < p[1] else 0,
    #     lambda p: 1 if 0.7 * p[0] < p[1] else 0,
    #     lambda p: 1 if 0.5 * p[0] < p[1] else 0,
    # ]
    # data_points = [(0.2, 0.03), (0.1, 0.12),
    #                (0.8, 0.63), (0.9, 0.82)]
    # c = voting_ensemble(classifiers)
    #
    # for v in data_points:
    #     print(c(v))

    # Example 2
    classifiers = [
        lambda p: 1,
        lambda p: 1,
        lambda p: 0,
        lambda p: 0,
    ]
    data_points = [(0.2, 0.03), (0.1, 0.12),
                   (0.8, 0.63), (0.9, 0.82)]
    c = voting_ensemble(classifiers)

    for v in data_points:
        print(c(v))


