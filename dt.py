# Assignment 1

import math


class DTNode:
    def __init__(self, decision):
        self.decision = decision
        self.children = None  # cant have no children and a callable decision

    def predict(self, feature_vector):
        if self.is_decision_node():
            return self.children[self.decision(feature_vector)].predict(feature_vector)
        else:
            return self.decision

    def leaves(self):
        if self.children is None:
            return 1

        return sum([child.leaves() for child in self.children])

    def is_decision_node(self):
        return True if callable(self.decision) else False


# returns a separator function and a partitioned dataset
def partition_by_feature_value(feature_index, dataset):
    features_at_index = []
    p_dataset = {}

    for (v, c) in dataset:
        if not p_dataset.setdefault(v[feature_index]):
            p_dataset[v[feature_index]] = [(v, c)]
            features_at_index.append(v[feature_index])
        else:
            p_dataset[v[feature_index]].append((v, c))

    separator = lambda x: features_at_index.index(x[feature_index])
    return separator, list(p_dataset.values())


def find_classes_from_data(data):
    return {c for _, c in data}


def proportion_with_k(data, k):
    return sum([1 for _, yi in data if yi == k]) / len(data)


def misclassification(data):
    return 1 - max([proportion_with_k(data, k) for k in find_classes_from_data(data)])


def gini(data):
    return sum([proportion_with_k(data, k) * (1 - proportion_with_k(data, k)) for k in find_classes_from_data(data)])


# Equal true and false classifications gives an entropy of 1
def entropy(data):
    return -sum([proportion_with_k(data, k) * math.log(proportion_with_k(data, k)) for k in find_classes_from_data(data)])


def impurity_at_node_m(data, feature, criterion):
    _, feature_partitions = partition_by_feature_value(feature, data)
    impurity = sum([(len(partition) / len(data)) * criterion(partition) for partition in feature_partitions])
    return feature, impurity


def most_common_label(data):
    proportion = 0
    class_label = None
    for _, c in data:
        prop_with_c = proportion_with_k(data, c)
        if prop_with_c > proportion:
            proportion = prop_with_c
            class_label = c

    return class_label


class DTree:
    def __init__(self, dataset, criterion):
        self.dataset = dataset
        self.criterion = criterion
        self.features = set(range(len(dataset[0][0])))  # each feature is represented as an index to its data in the dataset
        self.tree = self.split(self.dataset, self.features)

    def split(self, dataset, features):
        if proportion_with_k(dataset, dataset[0][1]) == 1:  # if all examples are in one class
            return DTNode(dataset[0][1])  # return a leaf node with that class label
        elif len(features) == 0:  # if the set of features is empty
            # return a leaf node with the most common class label
            return DTNode(most_common_label(dataset))
        else:
            # pick a categorical feature F
            F, _ = min([impurity_at_node_m(dataset, f, self.criterion) for f in features], key=lambda x: x[1])
            separator, partitions = partition_by_feature_value(F, dataset)
            # create new decision node
            decision_node = DTNode(separator)
            decision_node.children = [self.split(partition, features - {F}) for partition in partitions]
            return decision_node


# Components of Iterative Dichotomiser 3
# 1. A way of partitioning the dataset by features;
# 2. A criterion for impurity of a dataset; and
# 3. An iterative loop of concurrently splitting the dataset and building the tree.
def train_tree(dataset, criterion):
    dtree = DTree(dataset, criterion)
    return dtree.tree


def main():
    # Q3 - DTNode #

    # # Example 1
    # # The following (leaf) node will always predict True
    # node = DTNode(True)
    #
    # # Prediction for the input (True, False):
    # print(node.predict((True, False)))
    #
    # # Sine it's a leaf node, the input can be anything. It's simply ignored.
    # print(node.predict(None))

    # # Example 2
    # t = DTNode(True)
    # f = DTNode(False)
    # n = DTNode(lambda v: 0 if not v else 1)
    # n.children = [t, f]
    #
    # print(n.predict(False))
    # print(n.predict(True))

    # Q4 - Partition by feature
    # # Example 1
    # from pprint import pprint
    # dataset = [
    #     ((True, True), False),
    #     ((True, False), True),
    #     ((False, True), True),
    #     ((False, False), False),
    # ]
    # f, p = partition_by_feature_value(0, dataset)
    # pprint(sorted(sorted(partition) for partition in p))
    #
    # partition_index = f((True, True))
    # # Everything in the "True" partition for feature 0 is true
    # print(all(x[0] == True for x, c in p[partition_index]))
    # partition_index = f((False, True))
    # # Everything in the "False" partition for feature 0 is false
    # print(all(x[0] == False for x, c in p[partition_index]))

    # # Example 2
    # from pprint import pprint
    # dataset = [
    #     (("a", "x", 2), False),
    #     (("b", "x", 2), False),
    #     (("a", "y", 5), True),
    # ]
    # f, p = partition_by_feature_value(1, dataset)
    # pprint(sorted(sorted(partition) for partition in p))
    # partition_index = f(("a", "y", 5))
    # # everything in the "y" partition for feature 1 has a y
    # print(all(x[1] == "y" for x, c in p[partition_index]))

    # Q5 - misclassification, gini, and entropy functions #
    # # Example 1
    # data = [
    #     ((False, False), False),
    #     ((False, True), True),
    #     ((True, False), True),
    #     ((True, True), False)
    # ]
    # print("{:.4f}".format(misclassification(data)))
    # print("{:.4f}".format(gini(data)))
    # print("{:.4f}".format(entropy(data)))

    # # Example 2
    # data = [
    #     ((0, 1, 2), 1),
    #     ((0, 2, 1), 2),
    #     ((1, 0, 2), 1),
    #     ((1, 2, 0), 3),
    #     ((2, 0, 1), 3),
    #     ((2, 1, 0), 3)
    # ]
    # print("{:.4f}".format(misclassification(data)))
    # print("{:.4f}".format(gini(data)))
    # print("{:.4f}".format(entropy(data)))

    # Q6 - DTree #
    # # Example 1
    # dataset = [
    #     ((True, True), False),
    #     ((True, False), True),
    #     ((False, True), True),
    #     ((False, False), False)
    # ]
    # t = train_tree(dataset, misclassification)
    # print(t.predict((True, False)))
    # print(t.predict((False, False)))

    # # Example 2
    # dataset = [
    #     (("Sunny", "Hot", "High", "Weak"), False),
    #     (("Sunny", "Hot", "High", "Strong"), False),
    #     (("Overcast", "Hot", "High", "Weak"), True),
    #     (("Rain", "Mild", "High", "Weak"), True),
    #     (("Rain", "Cool", "Normal", "Weak"), True),
    #     (("Rain", "Cool", "Normal", "Strong"), False),
    #     (("Overcast", "Cool", "Normal", "Strong"), True),
    #     (("Sunny", "Mild", "High", "Weak"), False),
    #     (("Sunny", "Cool", "Normal", "Weak"), True),
    #     (("Rain", "Mild", "Normal", "Weak"), True),
    #     (("Sunny", "Mild", "Normal", "Strong"), True),
    #     (("Overcast", "Mild", "High", "Strong"), True),
    #     (("Overcast", "Hot", "Normal", "Weak"), True),
    #     (("Rain", "Mild", "High", "Strong"), False),
    # ]
    # t = train_tree(dataset, misclassification)
    # print(t.predict(("Overcast", "Cool", "Normal", "Strong")), "Expected: True")
    # print(t.predict(("Sunny", "Cool", "Normal", "Strong")), "Expected: True")
    # print()

    # # Example 3
    from pprint import pprint

    dataset = []
    with open('car.data', 'r') as f:
        for line in f.readlines():
            features = line.strip().split(",")
            dataset.append((tuple(features[:-1]), features[-1]))
    pprint(dataset[:5])
    print(
        "Exprected: [(('vhigh', 'vhigh', '2', '2', 'small', 'low'), 'unacc'), \n (('vhigh', 'vhigh', '2', '2', 'small', 'med'), 'unacc'), \n (('vhigh', 'vhigh', '2', '2', 'small', 'high'), 'unacc'), \n (('vhigh', 'vhigh', '2', '2', 'med', 'low'), 'unacc'), \n (('vhigh', 'vhigh', '2', '2', 'med', 'med'), 'unacc')]")
    t = train_tree(dataset, misclassification)
    print(t.predict(("high", "vhigh", "2", "2", "med", "low")), "Expected: unacc")
    print(t.leaves(), len(dataset), len(dataset) / t.leaves())

    # # Example 4
    from pprint import pprint
    dataset = []
    with open('balance-scale.data', 'r') as f:
        for line in f.readlines():
            out, *features = line.strip().split(",")
            dataset.append((tuple(features), out))
    pprint(dataset[:5])
    print(
        "Expected: [(('1', '1', '1', '1'), 'B'), \n (('1', '1', '1', '2'), 'R'), \n (('1', '1', '1', '3'), 'R'), \n (('1', '1', '1', '4'), 'R'), \n (('1', '1', '1', '5'), 'R')]")
    t = train_tree(dataset, misclassification)
    print(t.predict(("1", "4", "3", "2")), "Expected: R")
    print(t.leaves(), len(dataset), len(dataset) / t.leaves())

    # Q7 - Leaves #
    # # Example 1
    # n = DTNode(True)
    # print(n.leaves())
    #
    # # Example 2
    # t = DTNode(True)
    # f = DTNode(False)
    # n = DTNode(lambda v: 0 if not v else 1)
    # n.children = [t, f]
    # print(n.leaves())

if __name__ == "__main__":
    main()
