class DTNode:
    def __init__(self, decision):
        self.decision = decision
        self.children = None  # cant have no children and a callable decision

    def predict(self, feature_vector):
        if self.is_decision_node():
            return self.children[self.decision(feature_vector)].predict(feature_vector)
        else:
            return self.decision

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


import math

def find_classes_from_data(data):
    return {c for _, c in data}


def proportion_with_k(data, k):
    return sum([1 for _, yi in data if yi == k]) / len(data)


def misclassification(data):
    return 1 - max([proportion_with_k(data, k) for k in find_classes_from_data(data)])


def gini(data):
    return sum([proportion_with_k(data, k) * (1 - proportion_with_k(data, k)) for k in find_classes_from_data(data)])


def entropy(data):
    return -sum([proportion_with_k(data, k) * math.log(proportion_with_k(data, k)) for k in find_classes_from_data(data)])


def train_tree(dataset, misclass):
    pass

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
    dataset = [
        ((True, True), False),
        ((True, False), True),
        ((False, True), True),
        ((False, False), False)
    ]
    t = train_tree(dataset, misclassification)
    print(t.predict((True, False)))
    print(t.predict((False, False)))



if __name__ == "__main__":
    main()
