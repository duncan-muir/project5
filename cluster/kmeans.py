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
            max_iter: int = 100):
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
        """

        self._k = k
        if k <= 0:
            raise ClusteringException("K must be >= 0")
        self._metric = metric
        self._tol = tol
        self._max_iter = max_iter
        self._centroids = np.zeros((self._k, 2))
        self._error = float("inf")
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # init random labels
        if len(mat) < self._k:
            raise ClusteringException("Number of observations must be >= k")

        # weird behavior can happen where randint doesnt fully sample k, which is necessary for algorithm
        seed_labels = np.arange(self._k)
        labels = np.concatenate((np.random.randint(0, self._k, size=len(mat) - self._k), seed_labels))
        np.random.shuffle(labels)

        curr_iter = 0

        print(labels)
        while curr_iter < self._max_iter:


            for k in range(self._k):
                self._centroids[k] = np.average(mat[labels == k], axis=0)

            curr_error = self._compute_MSE(mat)

            labels = self.predict(mat)

            if np.isclose(self._error, curr_error, self._tol):
                print(f"Converged after {curr_iter + 1} iterations")
                self._error = curr_error
                break

            curr_iter += 1
            self._error = curr_error

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
