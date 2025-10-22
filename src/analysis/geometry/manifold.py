"""Analyze the geometric structure of activation manifolds."""

import torch
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ManifoldAnalyzer:
    """Analyze the geometric structure of activation spaces."""

    def __init__(self, activations: torch.Tensor):
        """
        Initialize manifold analyzer.

        Args:
            activations: Shape (num_samples, seq_len, hidden_dim) or (num_samples, hidden_dim)
        """
        self.activations = activations

        # Flatten if needed
        if len(activations.shape) == 3:
            self.flat_acts = activations.reshape(-1, activations.shape[-1])
        else:
            self.flat_acts = activations

        self.hidden_dim = self.flat_acts.shape[-1]
        self.num_samples = self.flat_acts.shape[0]

    def estimate_intrinsic_dimension(self, sample_size: int = 10000) -> Dict:
        """
        Estimate the intrinsic dimensionality of the activation manifold.

        Args:
            sample_size: Number of samples to use for estimation

        Returns:
            Dictionary with intrinsic dimension estimates
        """
        print(f"Estimating intrinsic dimension...")

        # Sample for computational efficiency
        if self.num_samples > sample_size:
            indices = np.random.choice(self.num_samples, sample_size, replace=False)
            sample = self.flat_acts[indices].cpu().numpy()
        else:
            sample = self.flat_acts.cpu().numpy()

        # Method 1: PCA explained variance
        print("  Computing PCA...")
        pca = PCA()
        pca.fit(sample)

        # Find number of components explaining different variance thresholds
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        dim_90 = int(np.argmax(cumsum >= 0.90) + 1)
        dim_95 = int(np.argmax(cumsum >= 0.95) + 1)
        dim_99 = int(np.argmax(cumsum >= 0.99) + 1)

        # Method 2: MLE estimator (Levina & Bickel, 2005)
        print("  Computing MLE estimate...")
        try:
            intrinsic_dim_mle = self._mle_intrinsic_dimension(sample[:5000])
        except Exception as e:
            print(f"  Warning: MLE estimation failed: {e}")
            intrinsic_dim_mle = -1

        results = {
            'ambient_dimension': self.hidden_dim,
            'intrinsic_dim_90pct_var': dim_90,
            'intrinsic_dim_95pct_var': dim_95,
            'intrinsic_dim_99pct_var': dim_99,
            'intrinsic_dim_mle': float(intrinsic_dim_mle),
            'explained_variance_ratio': pca.explained_variance_ratio_[:100].tolist(),
            'compression_ratio': float(self.hidden_dim / max(dim_95, 1)),
            'effective_rank': self._compute_effective_rank(pca.explained_variance_ratio_)
        }

        print(f"  Intrinsic dim (95% var): {dim_95}")
        print(f"  Compression ratio: {results['compression_ratio']:.2f}x")

        return results

    def _mle_intrinsic_dimension(self, X: np.ndarray, k: int = 20) -> float:
        """
        Maximum likelihood estimation of intrinsic dimension.

        Based on Levina & Bickel (2005).

        Args:
            X: Data matrix (n_samples, n_features)
            k: Number of nearest neighbors to use

        Returns:
            Estimated intrinsic dimension
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Use k nearest neighbors (exclude self at index 0)
        distances = distances[:, 1:]

        # MLE formula
        m = distances[:, -1:] / (distances[:, :-1] + 1e-10)
        d = -np.mean(1.0 / (np.log(m) + 1e-10))

        return max(1.0, min(d, X.shape[1]))  # Clamp to reasonable range

    def _compute_effective_rank(self, variance_ratio: np.ndarray) -> float:
        """
        Compute effective rank based on entropy of eigenvalue distribution.

        Args:
            variance_ratio: Explained variance ratios from PCA

        Returns:
            Effective rank
        """
        # Normalize to make it a probability distribution
        p = variance_ratio / variance_ratio.sum()

        # Compute entropy
        entropy = -np.sum(p * np.log(p + 1e-10))

        # Effective rank is exp(entropy)
        effective_rank = np.exp(entropy)

        return float(effective_rank)

    def compute_principal_directions(self, n_components: int = 100) -> Dict:
        """
        Compute principal directions (basis vectors for the manifold).

        Args:
            n_components: Number of principal components to compute

        Returns:
            Dictionary with principal component information
        """
        print(f"Computing {n_components} principal components...")

        # Use subset for efficiency
        sample_size = min(10000, self.num_samples)
        if self.num_samples > sample_size:
            indices = np.random.choice(self.num_samples, sample_size, replace=False)
            sample = self.flat_acts[indices].cpu().numpy()
        else:
            sample = self.flat_acts.cpu().numpy()

        pca = PCA(n_components=min(n_components, min(sample.shape)))
        pca.fit(sample)

        return {
            'n_components': len(pca.explained_variance_),
            'explained_variance': pca.explained_variance_.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'mean_activation': pca.mean_.tolist(),
            # Note: principal_components can be very large, so we don't include by default
            # 'principal_components': pca.components_.tolist(),
        }

    def compute_local_geometry(self, n_neighbors: int = 50, sample_size: int = 5000) -> Dict:
        """
        Analyze local geometric properties.

        Args:
            n_neighbors: Number of neighbors to consider
            sample_size: Number of points to sample

        Returns:
            Dictionary with local geometry statistics
        """
        print(f"Analyzing local geometry...")

        # Sample for efficiency
        if self.num_samples > sample_size:
            indices = np.random.choice(self.num_samples, sample_size, replace=False)
            sample = self.flat_acts[indices].cpu().numpy()
        else:
            sample = self.flat_acts.cpu().numpy()

        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(sample))).fit(sample)
        distances, indices = nbrs.kneighbors(sample)

        # Compute local statistics
        mean_local_distance = float(np.mean(distances))
        std_local_distance = float(np.std(distances))

        # Estimate local curvature (variance of distances to neighbors)
        local_curvature = np.var(distances, axis=1)
        mean_curvature = float(np.mean(local_curvature))
        max_curvature = float(np.max(local_curvature))

        # Determine geometry type based on curvature
        if mean_curvature < 0.1 * mean_local_distance ** 2:
            geometry_type = 'euclidean'
        elif mean_curvature < 0.5 * mean_local_distance ** 2:
            geometry_type = 'mildly_curved'
        else:
            geometry_type = 'highly_curved'

        return {
            'mean_local_distance': mean_local_distance,
            'std_local_distance': std_local_distance,
            'mean_local_curvature': mean_curvature,
            'max_local_curvature': max_curvature,
            'geometry_type': geometry_type,
            'distance_distribution': {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances)),
                '25th_percentile': float(np.percentile(distances, 25)),
                '75th_percentile': float(np.percentile(distances, 75)),
            }
        }

    def compute_activation_statistics(self) -> Dict:
        """
        Compute basic statistics about activations.

        Returns:
            Dictionary with activation statistics
        """
        print("Computing activation statistics...")

        acts = self.flat_acts.cpu().numpy()

        return {
            'shape': list(self.flat_acts.shape),
            'mean': float(np.mean(acts)),
            'std': float(np.std(acts)),
            'min': float(np.min(acts)),
            'max': float(np.max(acts)),
            'median': float(np.median(acts)),
            'sparsity': float(np.mean(acts == 0)),  # Fraction of zero activations
            'positive_fraction': float(np.mean(acts > 0)),
            'negative_fraction': float(np.mean(acts < 0)),
            'l2_norm_mean': float(np.mean(np.linalg.norm(acts, axis=1))),
            'l2_norm_std': float(np.std(np.linalg.norm(acts, axis=1))),
        }

    def analyze_all(self, n_components: int = 100) -> Dict:
        """
        Run all geometric analyses.

        Args:
            n_components: Number of principal components

        Returns:
            Complete analysis results
        """
        print("\n" + "="*60)
        print("GEOMETRIC ANALYSIS")
        print("="*60)

        results = {
            'activation_statistics': self.compute_activation_statistics(),
            'intrinsic_dimension': self.estimate_intrinsic_dimension(),
            'principal_directions': self.compute_principal_directions(n_components),
            'local_geometry': self.compute_local_geometry(),
        }

        print("="*60 + "\n")

        return results
