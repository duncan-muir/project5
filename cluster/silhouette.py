import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        self._metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        scores = []
        ks = np.max(y) + 1
        centroids = np.zeros((ks, 2))

        # get centroids based on provided labels
        for k in range(ks):
            centroids[k] = np.average(X[y == k], axis=0)

        for x, lab in zip(X, y):

            # compute average distance between point and all others in own cluster
            a = np.average(cdist(x[np.newaxis, ...], X[y == lab], metric=self._metric))

            # compute mean distance to non-self cluster
            non_self_mean_dists = [np.average(cdist(x[np.newaxis, ...], X[y == other_k], metric=self._metric))
                                   for other_k in range(ks) if other_k != lab]

            b = min(non_self_mean_dists)

            # compute final silhouette score
            score = (b - a) / max(a, b)
            scores.append(score)
        return np.array(scores)

