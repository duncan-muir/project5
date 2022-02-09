import numpy as np
from scipy.spatial.distance import cdist


class ClusteringException(Exception):
    pass


class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100,
            rand: np.random.RandomState = np.random.RandomState()):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
            rand: np.random.RandomState
                random state used in centroid initialization, and to allow consistency for testing
        """

        self._k = k
        if k <= 0:
            raise ClusteringException("K must be >= 0")
        self._metric = metric
        self._tol = tol
        self._max_iter = max_iter
        self._centroids = np.zeros((self._k, 2))
        self._error = float("inf")
        self._rand = rand
    
    def fit(self, mat: np.ndarray) -> None:
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if len(mat) < self._k:
            raise ClusteringException("Number of observations must be >= k")

        # init random centroids
        self._centroids = mat[self._rand.choice(mat.shape[0], self._k, replace=False), :]

        # init labels from random centroids
        labels = self.predict(mat)

        curr_iter = 0

        while curr_iter < self._max_iter:

            # update centroids given current labels
            for k in range(self._k):
                self._centroids[k] = np.average(mat[labels == k], axis=0)

            # compute MSE given updated centroids
            curr_error = self._compute_MSE(mat)

            # assign labels given new centroids
            labels = self.predict(mat)

            if np.isclose(self._error, curr_error, self._tol):
                # past error and current within tolerance, update final error and exit
                self._error = curr_error
                break

            curr_iter += 1
            self._error = curr_error

        return

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return np.argmin(cdist(mat, self._centroids, metric=self._metric), axis=1)

    def _compute_MSE(self, mat: np.ndarray) -> float:
        """
        computes MSE of current-fit model's centroids with a given matrix mat

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        outputs:
            float
                MSE
        """
        closest_dist_mat = np.min(cdist(mat, self._centroids, metric=self._metric), axis=1)

        return np.average(np.square(closest_dist_mat))

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return np.copy(self._centroids)
