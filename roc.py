from collections import namedtuple


class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        numbers = [self.true_positive, self.false_negative,
                   self.false_positive, self.true_negative]
        max_len = str(max(len(str(i)) for i in numbers))
        return (("{:>" + max_len + "} {:>" + max_len + "}\n" +
                 "{:>" + max_len + "} {:>" + max_len + "}").format(*numbers))


def confusion_matrix(classifier, dataset):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for classes, label in dataset:
        if classifier(classes) and label:
            tp += 1
        elif classifier(classes) and not label:
            fp += 1
        elif not classifier(classes) and label:
            fn += 1
        elif not classifier(classes) and not label:
            tn += 1

    return ConfusionMatrix(true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn)


def a_dominates_b(a, b):
    tp_a, fn_a, fp_a, tn_a = a[1]
    tp_b, fn_b, fp_b, tn_b = b[1]

    tpr_a = tp_a / (tp_a + fn_a)
    fpr_a = fp_a / (fp_a + tn_a)

    tpr_b = tp_b / (tp_b + fn_b)
    fpr_b = fp_b / (fp_b + tn_b)

    return True if tpr_a > tpr_b and fpr_a < fpr_b else False


def is_dominated(classifier, C):
    for other_c in C:
        if a_dominates_b(other_c, classifier):
            return True
    return False


def roc_non_dominated(C):
    convex_hull = []

    for c in C:
        if not is_dominated(c, C):
            convex_hull.append(c)

    return convex_hull


if __name__ == "__main__":

    # Question 6
    # data = [
    #     ((0.8, 0.2), 1),
    #     ((0.4, 0.3), 1),
    #     ((0.1, 0.35), 0),
    # ]
    # print(confusion_matrix(lambda x: 1, data)) # just tests the classifier
    # print()
    # print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, data))

    # Question 7
    # Example similar to the lecture notes

    classifiers = [
        ("Red", ConfusionMatrix(60, 40,
                                20, 80)),
        ("Green", ConfusionMatrix(40, 60,
                                  30, 70)),
        ("Blue", ConfusionMatrix(80, 20,
                                 50, 50)),
    ]
    print(sorted(label for (label, _) in roc_non_dominated(classifiers)))







