"""Analyze and characterize extracted SAE features."""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter, defaultdict


class FeatureAnalyzer:
    """Analyze and characterize extracted SAE features."""

    def __init__(self, sae, tokenizer, model=None):
        """
        Initialize feature analyzer.

        Args:
            sae: Trained SparseAutoencoder
            tokenizer: Tokenizer for text processing
            model: Optional language model for getting layer activations
        """
        self.sae = sae
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def find_max_activating_examples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        texts: List[str],
        top_k: int = 100
    ) -> List[Dict]:
        """
        Find examples that maximally activate a specific feature.

        Args:
            feature_idx: Index of feature to analyze
            activations: Layer activations (num_samples, seq_len, hidden_dim)
            texts: Corresponding text samples
            top_k: Number of top examples to return

        Returns:
            List of top activating examples
        """
        self.sae.eval()

        # Pass activations through SAE encoder
        if len(activations.shape) == 3:
            batch_size, seq_len, hidden_dim = activations.shape
            flat_acts = activations.reshape(-1, hidden_dim)
        else:
            flat_acts = activations
            batch_size = len(texts)
            seq_len = flat_acts.shape[0] // batch_size

        # Encode to get feature activations
        feature_acts = self.sae.encode(flat_acts)[:, feature_idx]  # Shape: (batch*seq,)

        # Reshape to (batch, seq)
        if len(activations.shape) == 3:
            feature_acts = feature_acts.reshape(batch_size, seq_len)
        else:
            feature_acts = feature_acts.reshape(batch_size, -1)

        # Find maximum activation per sample
        max_acts, max_positions = feature_acts.max(dim=1)

        # Get top-k samples
        top_k_indices = torch.topk(max_acts, min(top_k, len(max_acts))).indices

        results = []
        for idx in top_k_indices:
            idx = idx.item()
            results.append({
                'text': texts[idx] if idx < len(texts) else '',
                'activation': float(max_acts[idx]),
                'position': int(max_positions[idx]),
            })

        return results

    def characterize_feature(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        texts: List[str],
        num_examples: int = 100
    ) -> Dict:
        """
        Comprehensive characterization of a single feature.

        Args:
            feature_idx: Index of feature
            activations: Layer activations
            texts: Text samples
            num_examples: Number of examples to analyze

        Returns:
            Feature characterization dictionary
        """
        # Get max activating examples
        top_examples = self.find_max_activating_examples(
            feature_idx,
            activations,
            texts,
            top_k=num_examples
        )

        # Compute activation statistics
        stats = self._compute_feature_statistics(feature_idx, activations)

        # Infer semantic domain
        semantic = self._infer_semantic_domain(top_examples)

        return {
            'feature_id': feature_idx,
            'top_examples': top_examples[:10],  # Store top 10
            'activation_statistics': stats,
            'semantic_domain': semantic,
            'interpretation': semantic.get('interpretation', 'Unknown')
        }

    def _compute_feature_statistics(
        self,
        feature_idx: int,
        activations: torch.Tensor
    ) -> Dict:
        """Compute statistical properties of feature activations."""
        self.sae.eval()

        # Flatten activations
        if len(activations.shape) == 3:
            flat_acts = activations.reshape(-1, activations.shape[-1])
        else:
            flat_acts = activations

        # Get feature activations
        with torch.no_grad():
            feature_acts = self.sae.encode(flat_acts)[:, feature_idx].cpu().numpy()

        # Compute statistics
        nonzero_acts = feature_acts[feature_acts > 0]

        return {
            'mean': float(np.mean(feature_acts)),
            'std': float(np.std(feature_acts)),
            'max': float(np.max(feature_acts)),
            'sparsity': float(np.mean(feature_acts > 0)),
            'nonzero_mean': float(np.mean(nonzero_acts)) if len(nonzero_acts) > 0 else 0.0,
            'percentiles': {
                '50': float(np.percentile(feature_acts, 50)),
                '90': float(np.percentile(feature_acts, 90)),
                '95': float(np.percentile(feature_acts, 95)),
                '99': float(np.percentile(feature_acts, 99)),
            }
        }

    def _infer_semantic_domain(self, examples: List[Dict]) -> Dict:
        """
        Infer semantic domain from top activating examples.

        Args:
            examples: List of top activating examples

        Returns:
            Semantic domain information
        """
        if not examples:
            return {
                'primary_domains': ['unknown'],
                'confidence': 0.0,
                'interpretation': 'No activating examples found'
            }

        # Extract text snippets
        texts = [ex.get('text', '') for ex in examples[:50]]

        # Simple pattern-based domain detection
        domains = []

        # Check for numeric/math patterns
        numeric_count = sum(1 for text in texts if any(c.isdigit() for c in text))
        if numeric_count > len(texts) * 0.3:
            domains.append('numerical')

        # Check for code patterns
        code_keywords = ['function', 'return', 'class', 'import', 'def', 'var', 'const', 'if', 'else']
        code_count = sum(1 for text in texts if any(kw in text.lower() for kw in code_keywords))
        if code_count > len(texts) * 0.3:
            domains.append('code')

        # Check for common linguistic patterns
        common_words = ['the', 'and', 'for', 'that', 'with']
        text_count = sum(1 for text in texts if any(word in text.lower() for word in common_words))
        if text_count > len(texts) * 0.5:
            domains.append('natural_language')

        if not domains:
            domains = ['unknown']

        # Generate interpretation
        if domains[0] == 'numerical':
            interpretation = "This feature appears to respond to numerical content."
        elif domains[0] == 'code':
            interpretation = "This feature appears to respond to code-related content."
        elif domains[0] == 'natural_language':
            interpretation = "This feature responds to natural language patterns."
        else:
            interpretation = "Feature domain is unclear from activating examples."

        return {
            'primary_domains': domains,
            'confidence': 0.5 if domains[0] != 'unknown' else 0.0,
            'interpretation': interpretation
        }

    def analyze_all_features(
        self,
        activations: torch.Tensor,
        texts: List[str],
        max_features: int = 100,
        min_sparsity: float = 0.001
    ) -> List[Dict]:
        """
        Analyze multiple features.

        Args:
            activations: Layer activations
            texts: Text samples
            max_features: Maximum number of features to analyze
            min_sparsity: Minimum sparsity threshold (skip features that are too sparse)

        Returns:
            List of feature characterizations
        """
        print(f"Analyzing up to {max_features} features...")

        results = []

        # Get feature statistics for all features first
        self.sae.eval()
        if len(activations.shape) == 3:
            flat_acts = activations.reshape(-1, activations.shape[-1])
        else:
            flat_acts = activations

        with torch.no_grad():
            all_feature_acts = self.sae.encode(flat_acts).cpu().numpy()

        # Compute sparsity for each feature
        feature_sparsity = (all_feature_acts > 0).mean(axis=0)

        # Select features with reasonable sparsity
        valid_features = np.where(
            (feature_sparsity > min_sparsity) & (feature_sparsity < 0.5)
        )[0]

        # Sort by sparsity (prefer moderately sparse features)
        target_sparsity = 0.05
        valid_features = sorted(
            valid_features,
            key=lambda f: abs(feature_sparsity[f] - target_sparsity)
        )

        # Analyze top features
        for i, feature_idx in enumerate(valid_features[:max_features]):
            if i % 10 == 0:
                print(f"  Analyzed {i}/{min(max_features, len(valid_features))} features...")

            try:
                char = self.characterize_feature(feature_idx, activations, texts, num_examples=50)
                results.append(char)
            except Exception as e:
                print(f"  Warning: Failed to analyze feature {feature_idx}: {e}")
                continue

        print(f"Successfully analyzed {len(results)} features")

        return results
