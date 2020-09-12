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