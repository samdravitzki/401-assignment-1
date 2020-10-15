import sklearn.datasets
import sklearn.utils
import sklearn.tree
import numpy as np
from bootstrap import bootstrap
from ensemble import voting_ensemble


def bagging_model(learner, dataset, n_models, sample_size):
    bootstrap_data = bootstrap(dataset, sample_size)
    models = [learner(next(bootstrap_data)) for _ in range(n_models)]
    return voting_ensemble(models)


if "__main__" == __name__:
    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=1)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]
    dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

    def tree_learner(dataset):
        features, target = dataset[:, :-1], dataset[:, -1]
        model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
        return lambda v: model.predict(np.array([v]))[0]

    bagged = bagging_model(tree_learner, dataset, 50, len(dataset)//2)
    # Note that we get the first one wrong!
    for (v, c) in zip(test_data, test_target):
        print(int(bagged(v)), c)
