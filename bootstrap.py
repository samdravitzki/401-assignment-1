import hashlib
import numpy as np


def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
        yield r


def take(n, iterator):
    while n > 0:
        yield next(iterator)
        n -= 1


def random_feature(D, random_iterator):
    num_features = len(D)
    random = next(take(1, random_iterator))

    return D[int(num_features * random)]


def bootstrap(D, sample_size):
    random = pseudo_random()

    while True:
        new_dataset = []
        for i in range(sample_size):
            new_dataset.append(random_feature(D, random))

        yield np.array(new_dataset)


if "__main__" == __name__:
    dataset = np.array([[1, 0, 2, 3],
                        [2, 3, 0, 0],
                        [4, 1, 2, 0],
                        [3, 2, 1, 0]])

    ds_gen = bootstrap(dataset, 3)
    print(next(ds_gen))
    print(next(ds_gen))