# Write your k-means unit tests here
import numpy as np
import pytest
from cluster import *

np.random.seed(27)


@pytest.fixture
def test_clusters():
    """
    Construct test clusters for k-means testing
    """
    return np.array([[2, 4],
                    [2, 6],
                    [-2, 4],
                    [-2, 6],
                    [2, -4],
                    [2, -6],
                    [-2, -4],
                    [-2, -6]])


def test_2_mean_fit(test_clusters):
    """
    Test that clusters are properly determined, separated by x axis
    when K = 2. (np.random seeded for consistency)
    """

    km = KMeans(2)

    km.fit(test_clusters)

    # check final labels
    assert np.array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                          km.predict(test_clusters))

    # check final centroids
    assert np.allclose(km.get_centroids(), np.array([[0, 5],
                                                    [0, -5]]), .001)

    # check final error
    assert np.isclose(km.get_error(), 5)


def test_4_mean_fit(test_clusters):
    """
    Test that clusters are properly determined, separated by x and y axes
    when K = 4. (np.random seeded for consistency)
    """
    km = KMeans(4)

    km.fit(test_clusters)

    # check final labels
    assert np.array_equal(np.array([1, 1, 2, 2, 0, 0, 3, 3]),
                          km.predict(test_clusters))

    # check final centroids
    assert np.allclose(km.get_centroids(), np.array([[2, -5],
                                                     [2, 5],
                                                     [-2, 5],
                                                     [-2, -5]]), .001)
    # check final error
    assert np.isclose(km.get_error(), 1)
