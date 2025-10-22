"""Clustering analysis for discovering feature groups."""

import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureClusterer:
    """Discover semantic clusters in activation space."""

    def __init__(self, activations: torch.Tensor, labels: Optional[List[str]] = None):
        """
        Initialize clusterer.

        Args:
            activations: Activation tensor (num_samples, hidden_dim) or (num_samples, seq_len, hidden_dim)
            labels: Optional labels for samples
        """
        self.activations = activations
        self.labels = labels

        # Flatten if needed
        if len(activations.shape) == 3:
            self.flat_acts = activations.reshape(-1, activations.shape[-1])
        else:
            self.flat_acts = activations

    def cluster(
        self,
        method: str = 'kmeans',
        n_clusters: int = 50,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Cluster activations to discover feature groups.

        Args:
            method: Clustering method ('kmeans' or 'minibatch_kmeans')
            n_clusters: Number of clusters
            sample_size: Number of samples to use (None = use all)
            **kwargs: Additional arguments for clustering algorithm

        Returns:
            Dictionary with clustering results
        """
        print(f"Clustering with {method}, n_clusters={n_clusters}...")

        # Convert to numpy and sample if needed
        flat_acts = self.flat_acts.cpu().numpy()

        if sample_size and len(flat_acts) > sample_size:
            indices = np.random.choice(len(flat_acts), sample_size, replace=False)
            sample = flat_acts[indices]
            print(f"  Sampled {sample_size} points from {len(flat_acts)}")
        else:
            sample = flat_acts
            indices = np.arange(len(flat_acts))

        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                **kwargs
            )
        elif method == 'minibatch_kmeans':
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1024,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        print(f"  Fitting clusterer...")
        cluster_labels = clusterer.fit_predict(sample)

        # Analyze clusters
        cluster_info = self._analyze_clusters(sample, cluster_labels, clusterer)

        # Compute clustering quality metrics
        quality_metrics = self._compute_quality_metrics(sample, cluster_labels)

        return {
            'method': method,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'sample_indices': indices.tolist() if sample_size else None,
            'cluster_info': cluster_info,
            'quality_metrics': quality_metrics,
            'inertia': float(clusterer.inertia_) if hasattr(clusterer, 'inertia_') else None,
        }

    def _analyze_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        clusterer
    ) -> List[Dict]:
        """Analyze properties of each cluster."""
        cluster_info = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:  # Noise cluster
                continue

            mask = labels == label
            cluster_acts = X[mask]

            # Compute cluster statistics
            centroid = clusterer.cluster_centers_[label] if hasattr(clusterer, 'cluster_centers_') else cluster_acts.mean(axis=0)

            # Intra-cluster variance
            inertia = float(np.sum((cluster_acts - centroid) ** 2))

            # Cluster diameter (max distance from centroid)
            distances = np.linalg.norm(cluster_acts - centroid, axis=1)
            diameter = float(np.max(distances))
            mean_distance = float(np.mean(distances))

            cluster_info.append({
                'cluster_id': int(label),
                'size': int(np.sum(mask)),
                'size_fraction': float(np.sum(mask) / len(labels)),
                'inertia': inertia,
                'diameter': diameter,
                'mean_distance_to_centroid': mean_distance,
                'std_distance_to_centroid': float(np.std(distances)),
            })

        return cluster_info

    def _compute_quality_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute clustering quality metrics."""
        print("  Computing quality metrics...")

        # Sample for efficiency if too large
        if len(X) > 10000:
            indices = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels

        metrics = {}

        # Silhouette score (how well separated are clusters)
        try:
            if len(np.unique(labels_sample)) > 1:
                metrics['silhouette_score'] = float(silhouette_score(X_sample, labels_sample))
            else:
                metrics['silhouette_score'] = 0.0
        except Exception as e:
            print(f"    Warning: Could not compute silhouette score: {e}")
            metrics['silhouette_score'] = 0.0

        # Calinski-Harabasz score (ratio of between-cluster to within-cluster variance)
        try:
            if len(np.unique(labels_sample)) > 1:
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X_sample, labels_sample))
            else:
                metrics['calinski_harabasz_score'] = 0.0
        except Exception as e:
            print(f"    Warning: Could not compute Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz_score'] = 0.0

        return metrics

    def find_optimal_k(
        self,
        k_range: range = range(5, 51, 5),
        method: str = 'elbow',
        sample_size: int = 5000
    ) -> Dict:
        """
        Find optimal number of clusters.

        Args:
            k_range: Range of k values to try
            method: Method to use ('elbow', 'silhouette')
            sample_size: Number of samples to use

        Returns:
            Dictionary with results for each k
        """
        print(f"Finding optimal k using {method} method...")

        flat_acts = self.flat_acts.cpu().numpy()

        if len(flat_acts) > sample_size:
            sample = flat_acts[np.random.choice(len(flat_acts), sample_size, replace=False)]
        else:
            sample = flat_acts

        results = []

        for k in k_range:
            print(f"  Trying k={k}...")

            clusterer = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = clusterer.fit_predict(sample)

            result = {
                'k': k,
                'inertia': float(clusterer.inertia_),
            }

            # Compute silhouette score
            if method == 'silhouette' or len(results) > 0:
                try:
                    result['silhouette_score'] = float(silhouette_score(sample, labels))
                except:
                    result['silhouette_score'] = 0.0

            results.append(result)

        # Find optimal k
        if method == 'elbow':
            # Use elbow method (look for bend in inertia curve)
            inertias = [r['inertia'] for r in results]
            optimal_k = self._find_elbow(list(k_range), inertias)
        elif method == 'silhouette':
            # Use maximum silhouette score
            optimal_k = max(results, key=lambda x: x.get('silhouette_score', 0))['k']
        else:
            optimal_k = k_range[len(k_range) // 2]

        print(f"  Optimal k: {optimal_k}")

        return {
            'optimal_k': optimal_k,
            'method': method,
            'results': results
        }

    def _find_elbow(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in inertia curve."""
        # Simple method: find maximum second derivative
        if len(inertias) < 3:
            return k_values[0]

        # Compute second derivative
        second_deriv = []
        for i in range(1, len(inertias) - 1):
            d2 = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_deriv.append(abs(d2))

        # Find maximum
        elbow_idx = np.argmax(second_deriv) + 1  # +1 because we started at index 1

        return k_values[elbow_idx]
