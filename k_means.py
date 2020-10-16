import numpy as np
import math
import hashlib
from scipy.spatial import distance as dist
import sklearn.datasets
import sklearn.utils

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


def closest_centroid(p, centroids):
    distances = [(i, distance(p, c)) for i, c in enumerate(centroids)]
    return min(distances, key=lambda x: x[1])[0]


def distance(p1, p2):
    delta = p1 - p2
    return math.sqrt(sum([pow(p, 2) for p in delta]))


def points_center(ps):
    sums = []
    for i in range(len(ps[0])):
        sums.append(sum([p[i] for p in ps]) / len(ps))

    return sums


def equal(cluster_group_a, cluster_group_b):
    if any([True if len(cluster_group_a[i]) != len(cluster_group_b[i]) else False for i in range(len(cluster_group_a))]):
        return False

    return str(cluster_group_a) == str(cluster_group_b)


def k_means(dataset, centroids):
    empty_clusters = lambda: [[] for _ in range(len(centroids))]
    centroids = list(centroids)

    previous_clusters = empty_clusters()
    current_clusters = empty_clusters()
    while True:
        # Relate points to centroid
        for point in dataset:
            current_clusters[closest_centroid(point, centroids)].append(point)

        # Re-estimate centroids
        for i in range(len(centroids)):
            points = current_clusters[i]
            if len(points) > 0:
                centroids[i] = np.array(points_center(points))

        # Exit condition
        if equal(previous_clusters, current_clusters):
            return centroids

        previous_clusters = current_clusters.copy()
        current_clusters = empty_clusters()


def cluster_points(centroids, dataset):
    clusters = [[] for _ in range(len(centroids))]

    for p in dataset:
        clusters[closest_centroid(p, centroids)].append(p)

    return clusters


# Mean closest distance between every cluster
def separation(clusters):
    # min_distances = [min([distance(p1, p2) for p1 in c1 for p2 in c2])
    #                  for i, c1 in enumerate(clusters) for j, c2 in enumerate(clusters) if i != j]
    min_distances = [dist.cdist(c1, c2).min() for i, c1 in enumerate(clusters) for j, c2 in enumerate(clusters) if i != j]
    return sum(min_distances) / len(clusters)


# Diameter of a cluster
def compactness(clusters):
    clusters_distances = [[distance(p1, p2) for p1 in c for p2 in c] for c in clusters]
    diameters = [max(c) for c in clusters_distances]
    return sum(diameters) / len(clusters)


def goodness(clusters):
    return separation(clusters) / compactness(clusters)


def generate_random_vector(bounds, r):
    return np.array([(high - low) * next(r) + low for low, high in bounds])


def k_means_random_restart(dataset, k, restarts, seed=None):
    bounds = list(zip(np.min(dataset, axis=0), np.max(dataset, axis=0)))
    r = pseudo_random(seed=seed) if seed else pseudo_random()
    models = []
    for _ in range(restarts):
        random_centroids = tuple(generate_random_vector(bounds, r)
                                 for _ in range(k))
        new_centroids = k_means(dataset, random_centroids)
        clusters = cluster_points(new_centroids, dataset)
        if any(len(c) == 0 for c in clusters):
            continue
        models.append((goodness(clusters), new_centroids))
    return max(models, key=lambda x: x[0])[1]



if __name__ == "__main__":
    # Question 6
    # # Example 1
    # dataset = np.array([
    #     [0.1, 0.1],
    #     [0.2, 0.2],
    #     [0.8, 0.8],
    #     [0.9, 0.9]
    # ])
    # centroids = (np.array([0., 0.]), np.array([1., 1.]))
    # for c in k_means(dataset, centroids):
    #     print(c)
    #
    # # Example 2
    # dataset = np.array([
    #     [0.1, 0.1],
    #     [0.2, 0.2],
    #     [0.8, 0.8],
    #     [0.9, 0.9]
    # ])
    # centroids = (np.array([0., 1.]), np.array([1., 0.]))
    # for c in k_means(dataset, centroids):
    #     print(c)
    # # Example 3
    # iris = sklearn.datasets.load_iris()
    # data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    # train_data, train_target = data[:-5, :], target[:-5]
    # test_data, test_target = data[-5:, :], target[-5:]
    #
    # centroids = (
    #     np.array([5.8, 2.5, 4.5, 1.5]),
    #     np.array([6.8, 3.0, 5.7, 2.1]),
    #     np.array([5.0, 3.5, 1.5, 0.5])
    # )
    # for c in k_means(train_data, centroids):
    #     print(c)
    # dataset = np.array([
    #     [0.1, 0.1],
    #     [0.2, 0.2],
    #     [0.8, 0.8],
    #     [0.9, 0.9]
    # ])
    # for c in k_means_random_restart(dataset, k=2, restarts=5):
    #     print(c)
    #
    # import sklearn.datasets
    # import sklearn.utils
    #
    # iris = sklearn.datasets.load_iris()
    # data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    # train_data, train_target = data[:-5, :], target[:-5]
    # test_data, test_target = data[-5:, :], target[-5:]
    #
    # centroids = k_means_random_restart(train_data, k=3, restarts=10)
    # for c in centroids:
    #     print(c)

    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]

    centroids = k_means_random_restart(train_data, k=3, restarts=10)
    for c in centroids:
        print(c)