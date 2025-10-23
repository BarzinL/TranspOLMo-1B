#!/usr/bin/env python3
"""
Minimal test to see TranspOLMo working with very low memory usage.
Uses tiny batch size and single layer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.loader import OLMo2Loader
from src.models.hooks import ActivationCapture
from src.analysis.geometry.manifold import ManifoldAnalyzer

print("=" * 80)
print("MINIMAL TRANSPOLMO TEST")
print("=" * 80)
print()

# Load model in float16
print("Loading model in float16...")
loader = OLMo2Loader(
    model_name="allenai/OLMo-2-0425-1B",
    cache_dir="./data/models",
    device="cuda",
    dtype="float16"
)

model, tokenizer = loader.load()
arch_info = loader.get_architecture_info(model)

print(f"\nModel: {arch_info['model_name']}")
print(f"Layers: {arch_info['num_layers']}")
print(f"Hidden size: {arch_info['hidden_size']}")
print()

# Create 10 tiny samples
print("Creating 10 tiny text samples...")
texts = [
    "Hello world",
    "The cat sat",
    "Machine learning is",
    "Python programming",
    "Neural networks",
    "Data science",
    "Artificial intelligence",
    "Deep learning",
    "Natural language",
    "Computer vision"
]

# Capture activations from just ONE layer with TINY batches
print("Capturing activations from layer 0 (batch_size=1)...")
layer_name = "model.layers.0.mlp"

with ActivationCapture(model, device='cpu') as capturer:
    capturer.register_hooks([layer_name])
    
    # Process one at a time to minimize memory
    for text in texts:
        inputs = tokenizer(
            text,
            max_length=32,  # Very short!
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        # Clear GPU cache after each
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    activations = capturer.get_activations()

print(f"✓ Captured activations: {activations[layer_name].shape}")
print()

# Quick geometric analysis
print("Running geometric analysis...")
analyzer = ManifoldAnalyzer(activations[layer_name])
results = analyzer.analyze_all(n_components=10)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Intrinsic dimension (95%): {results['intrinsic_dimension']['intrinsic_dim_95pct_var']}")
print(f"Compression ratio: {results['intrinsic_dimension']['compression_ratio']:.2f}x")
print(f"Geometry type: {results['local_geometry']['geometry_type']}")
print(f"Mean activation: {results['activation_statistics']['mean']:.4f}")
print(f"Sparsity: {results['activation_statistics']['sparsity']:.2%}")
print()
print("✓ TranspOLMo is working!")
print("=" * 80)
