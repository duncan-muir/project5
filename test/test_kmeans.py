# Write your k-means unit tests here
import numpy as np
import pytest
from cluster import *

np.random.seed(27)

def test_2_mean_fit():

    clusters = np.array([[2, 4],
                        [2, 6],
                         [-2, 4],
                         [-2, 6],
                         [2, -4],
                         [2, -6],
                        [-2, -4],
                         [-2, -6]])

    km = KMeans(2)

    km.fit(clusters)

    assert np.array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                          km.predict(clusters))

    assert np.allclose(km.get_centroids(), np.array([[0, 5],
                                                    [0, -5]]), .001)
    assert np.isclose(km.get_error(), 5)


def test_4_mean_fit():
    clusters = np.array([[2, 4],
                         [2, 6],
                         [-2, 4],
                         [-2, 6],
                         [2, -4],
                         [2, -6],
                         [-2, -4],
                         [-2, -6]])

    km = KMeans(4)

    km.fit(clusters)

    assert np.array_equal(np.array([1, 1, 2, 2, 0, 0, 3, 3]),
                          km.predict(clusters))

    assert np.allclose(km.get_centroids(), np.array([[2, -5],
                                                     [2, 5],
                                                     [-2, 5],
                                                     [-2, -5]]), .001)
    assert np.isclose(km.get_error(), 1)
