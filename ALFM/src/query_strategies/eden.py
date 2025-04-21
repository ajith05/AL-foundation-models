'''EDEN sampling strategy'''

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from ALFM.src.query_strategies.base_query import BaseQuery


class Eden(BaseQuery):
    """Embedding DENsity-based Sampling Strategy (EDEN)."""

    def __init__(self, mode: str = 'mixed', k: int = 10, **params):
        """
        Args:
            mode (str): Sampling mode - 'low', 'high', or 'mixed'.
            k (int): Number of neighbors to consider for density estimation.
        """
        super().__init__(**params)
        assert mode in ['low', 'high', 'mixed'], "mode must be 'low', 'high', or 'mixed'"
        self.mode = mode
        self.k = k

    def _compute_density_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute 1 / (avg distance to k nearest neighbors)."""
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine').fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        avg_dist = distances[:, 1:].mean(axis=1)  # exclude self (distance 0)
        scores = 1.0 / (avg_dist + 1e-8)  # avoid divide by zero
        return scores

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        # Get features and embeddings
        features = self.features[unlabeled_indices]
        _, embeddings = self.model.get_probs_and_embedding(features)
        embeddings = embeddings.cpu().numpy()

        # Normalize for cosine distance
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute density
        density_scores = self._compute_density_scores(embeddings)

        logging.info(f"[Eden] Density score stats: min={density_scores.min()}, max={density_scores.max()}, mean={density_scores.mean()}")

        # Select indices based on mode
        if self.mode == 'low':
            selected_indices = np.argsort(density_scores)[:num_samples]
        elif self.mode == 'high':
            selected_indices = np.argsort(density_scores)[-num_samples:]
        else:  # mixed
            half = num_samples // 2
            low_idx = np.argsort(density_scores)[:half]
            high_idx = np.argsort(density_scores)[-num_samples + half:]
            selected_indices = np.concatenate([low_idx, high_idx])
            np.random.shuffle(selected_indices)
        
        logging.info(f"[Eden] Selected indices (local): {selected_indices}")

        # Convert to global mask
        mask = np.zeros(len(self.features), dtype=bool)
        selected_global_indices = unlabeled_indices[selected_indices]
        logging.info(f"[Eden] Selected indices (global): {selected_global_indices}")
        mask[selected_global_indices] = True
        return mask
