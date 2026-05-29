"""
OrderAnomalyModel — DBSCAN-based anomaly detector with real out-of-sample scoring.

Why DBSCAN?
- No need to specify number of clusters
- Automatically finds outliers (label = -1)
- Works well for density-based anomaly detection
- Perfect for "does this feel weird?" detection

Blue-Team Mapping:
- items_per_order      → request_size / batch_size
- time_since_last_order → request_frequency
- order_time           → time_of_day patterns
- customer_frequency   → user_activity_history
- total_cost           → resource_consumption
- prep_time_estimate   → processing_time
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

logger = logging.getLogger(__name__)

CLUSTER_NAMES: dict[int, str] = {
    0: "Teddy Regular 🔧",
    1: "Linda Chaos 🎵",
    2: "Late Night Hugo 🕐",
    -1: "Absolutely Not 🚨",
}


class OrderAnomalyModel:
    """
    Trains on order data and scores individual orders without re-fitting.

    fit()   — trains DBSCAN and builds a NearestNeighbors index over core samples
              so that score() can assign cluster membership in O(log n) without
              re-running fit_predict on the full training set every call.
    score() — classifies a single order dict; returns cluster_id, cluster_name,
              is_anomaly.
    """

    def __init__(self, eps: float = 0.8, min_samples: int = 10):
        self.scaler = StandardScaler()
        self._dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self._neighbor_index: NearestNeighbors | None = None
        self._core_labels: np.ndarray | None = None
        self._eps = eps
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train on order data.

        Returns a copy of df with 'cluster' and 'cluster_name' columns added.
        Does not mutate the input DataFrame.
        """
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        labels = self._dbscan.fit_predict(X_scaled)

        # Build NearestNeighbors index over core samples.
        # A new point is assigned to a cluster if it falls within eps of any core sample;
        # otherwise it is noise (-1). This mirrors DBSCAN's own reachability definition.
        core_indices = self._dbscan.core_sample_indices_
        if len(core_indices) == 0:
            logger.warning("DBSCAN found no core samples — all new orders will score as anomalies")
            self._neighbor_index = None
            self._core_labels = np.array([], dtype=int)
        else:
            core_samples_scaled = X_scaled[core_indices]
            self._core_labels = labels[core_indices]
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(core_samples_scaled)
            self._neighbor_index = nn

        self._is_fitted = True

        result = df.copy()
        result["cluster"] = labels
        result["cluster_name"] = result["cluster"].map(
            lambda x: CLUSTER_NAMES.get(x, f"Cluster {x}")
        )
        return result

    def score(self, order: dict) -> dict:
        """
        Score a single order without re-fitting.

        Args:
            order: dict with keys matching FEATURE_COLS.

        Returns:
            dict with cluster_id (int), cluster_name (str), is_anomaly (bool).

        Raises:
            RuntimeError: if called before fit().
            KeyError: if order is missing a required feature.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score()")

        X_new = np.array([[order[f] for f in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X_new)

        if self._neighbor_index is None:
            cluster_id = -1
        else:
            distances, indices = self._neighbor_index.kneighbors(X_scaled)
            nearest_dist = distances[0][0]
            # Reachable from a core sample → assign that core sample's cluster label
            if nearest_dist <= self._eps:
                cluster_id = int(self._core_labels[indices[0][0]])
            else:
                cluster_id = -1

        return {
            "cluster_id": cluster_id,
            "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
            "is_anomaly": cluster_id == -1,
        }
