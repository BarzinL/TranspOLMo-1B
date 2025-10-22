"""Tests for geometric analysis."""

import pytest
import torch
import numpy as np
from src.analysis.geometry.manifold import ManifoldAnalyzer
from src.analysis.geometry.clustering import FeatureClusterer


def test_manifold_analyzer_creation():
    """Test creating a manifold analyzer."""
    # Create random activations
    activations = torch.randn(100, 50, 64)  # (samples, seq, hidden)

    analyzer = ManifoldAnalyzer(activations)

    assert analyzer.hidden_dim == 64
    assert analyzer.num_samples == 100 * 50  # Flattened


def test_intrinsic_dimension_estimation():
    """Test intrinsic dimension estimation."""
    # Create low-rank data
    U = torch.randn(500, 10)
    V = torch.randn(10, 64)
    activations = U @ V  # Rank-10 matrix

    analyzer = ManifoldAnalyzer(activations)
    results = analyzer.estimate_intrinsic_dimension(sample_size=500)

    assert 'ambient_dimension' in results
    assert 'intrinsic_dim_95pct_var' in results
    assert results['ambient_dimension'] == 64
    assert results['intrinsic_dim_95pct_var'] < 64  # Should be much less than 64


def test_activation_statistics():
    """Test computing activation statistics."""
    activations = torch.randn(100, 64)

    analyzer = ManifoldAnalyzer(activations)
    stats = analyzer.compute_activation_statistics()

    assert 'mean' in stats
    assert 'std' in stats
    assert 'sparsity' in stats
    assert isinstance(stats['mean'], float)


def test_feature_clusterer():
    """Test feature clustering."""
    activations = torch.randn(200, 32)

    clusterer = FeatureClusterer(activations)
    results = clusterer.cluster(method='kmeans', n_clusters=5)

    assert 'n_clusters' in results
    assert 'cluster_info' in results
    assert len(results['cluster_info']) <= 5


def test_clustering_quality_metrics():
    """Test clustering quality metrics computation."""
    # Create well-separated clusters
    cluster1 = torch.randn(50, 10) + torch.tensor([5.0] * 10)
    cluster2 = torch.randn(50, 10) - torch.tensor([5.0] * 10)
    activations = torch.cat([cluster1, cluster2], dim=0)

    clusterer = FeatureClusterer(activations)
    results = clusterer.cluster(method='kmeans', n_clusters=2, sample_size=100)

    assert 'quality_metrics' in results
    assert 'silhouette_score' in results['quality_metrics']
    # Well-separated clusters should have high silhouette score
    assert results['quality_metrics']['silhouette_score'] > 0.3
