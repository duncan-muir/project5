"""
K-Means clustering module
"""
version = "0.1.0"

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

