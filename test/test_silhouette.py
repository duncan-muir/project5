# write your silhouette score unit tests here
import numpy as np
import pytest
from cluster import *

np.random.seed(27)


@pytest.fixture
def test_clusters():
    """
    Construct test clusters for k-means silhouette testing
    """
    return np.array([[2, 4],
                     [2, 6],
                     [-2, 4],
                     [-2, 6],
                     [2, -4],
                     [2, -6],
                     [-2, -4],
                     [-2, -6]])


def test_sillhouette_kmeans_2(test_clusters):
    """
    Test that sillhouette scores are properly computed for test clusters when k=2
    """

    k2_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    s = Silhouette()

    assert np.allclose(s.score(test_clusters, k2_labels),
                       np.tile([0.71603434, 0.76583592], 4), .001)


def test_sillhouette_kmeans_4(test_clusters):
    """
    Test that sillhouette scores are properly computed for test clusters when k=2
    """

    k4_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    s = Silhouette()

    assert np.allclose(s.score(test_clusters, k4_labels),
                       0.7574637 * np.ones(len(k4_labels)), .001)


