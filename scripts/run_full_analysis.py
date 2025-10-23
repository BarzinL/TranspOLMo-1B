#!/usr/bin/env python3
"""
Main orchestration script for full transparency analysis.
Runs all phases sequentially on OLMo2-1B.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from src.config import Config
from src.models.loader import OLMo2Loader
from src.models.hooks import ActivationCapture
from src.extraction.dataset_builder import AnalysisDatasetBuilder
from src.analysis.geometry.manifold import ManifoldAnalyzer
from src.analysis.geometry.clustering import FeatureClusterer
from src.analysis.features.sparse_autoencoder import SAETrainer
from src.analysis.features.feature_analysis import FeatureAnalyzer
from src.analysis.circuits.discovery import CircuitDiscovery
from src.documentation.generator import DocumentationGenerator
from src.verification.intervention import ModelInterventions


def main():
    """Run complete transparency analysis pipeline."""

    parser = argparse.ArgumentParser(description='Run TranspOLMo analysis pipeline')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name or path (default: from config.py)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples for analysis (default: from config.py)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu) (default: from config.py)')
    parser.add_argument('--dtype', type=str, default=None,
                        help='Data type: float32, float16, bfloat16 (default: from config.py)')
    parser.add_argument('--skip-sae', action='store_true',
                        help='Skip SAE training (faster)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer indices to analyze (e.g., "0,6,11")')

    args = parser.parse_args()

    # Initialize config from config.py (single source of truth)
    config = Config.default()

    # Only override config if arguments were explicitly provided
    if args.model is not None:
        config.model.model_name = args.model
    if args.device is not None:
        config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if args.num_samples is not None:
        config.extraction.num_samples = args.num_samples

    print("=" * 80)
    print("TRANSPOLMO: First-Principles Neural Network Interpretability")
    print("=" * 80)
    print(f"\nModel: {config.model.model_name}")
    print(f"Device: {config.model.device}")
    print(f"Samples: {config.extraction.num_samples}")
    print()

    # Parse layer selection
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(',')]
    else:
        layer_indices = None

    # PHASE 1: Load Model
    print("\n" + "=" * 80)
    print("PHASE 1: Loading Model")
    print("=" * 80)

    loader = OLMo2Loader(
        model_name=config.model.model_name,
        cache_dir=config.model.cache_dir,
        device=config.model.device,
        dtype=config.model.dtype
    )

    model, tokenizer = loader.load()
    arch_info = loader.get_architecture_info(model)

    print(f"\nArchitecture:")
    print(f"  Layers: {arch_info['num_layers']}")
    print(f"  Hidden size: {arch_info['hidden_size']}")
    print(f"  Attention heads: {arch_info['num_attention_heads']}")
    print(f"  Vocabulary: {arch_info['vocab_size']}")

    # PHASE 2: Build Dataset
    print("\n" + "=" * 80)
    print("PHASE 2: Building Analysis Dataset")
    print("=" * 80)

    dataset_builder = AnalysisDatasetBuilder(
        dataset_name=config.extraction.dataset_name,
        tokenizer=tokenizer,
        num_samples=config.extraction.num_samples,
        max_seq_length=config.extraction.max_seq_length,
        subset=config.extraction.dataset_subset
    )

    dataloader = dataset_builder.build(batch_size=config.extraction.batch_size)
    print(f"Dataset ready with {len(dataloader.dataset)} samples")

    # PHASE 3: Extract Activations
    print("\n" + "=" * 80)
    print("PHASE 3: Extracting Activations")
    print("=" * 80)

    # Determine which layers to analyze
    if layer_indices is None:
        # Analyze a subset of layers (first, middle, last)
        total_layers = arch_info['num_layers']
        layer_indices = [0, total_layers // 2, total_layers - 1]

    print(f"Analyzing layers: {layer_indices}")

    all_activations = {}
    all_texts = []

    for layer_idx in layer_indices:
        print(f"\nExtracting from layer {layer_idx}...")

        # Hook the MLP output of this layer
        layer_name = f"model.layers.{layer_idx}.mlp"

        with ActivationCapture(model, device='cpu') as capturer:
            capturer.register_hooks([layer_name])

            batch_texts = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx}")):
                    inputs = {
                        'input_ids': batch['input_ids'].to(config.model.device),
                        'attention_mask': batch['attention_mask'].to(config.model.device)
                    }

                    # Forward pass
                    _ = model(**inputs)

                    # Store texts from first layer only
                    if layer_idx == layer_indices[0]:
                        batch_texts.extend(batch.get('text', [''] * len(batch['input_ids'])))

                    # Limit batches for speed
                    if batch_idx >= 20:  # ~320 samples with batch_size=16
                        break

            # Get activations
            acts = capturer.get_activations()
            if layer_name in acts:
                all_activations[layer_idx] = acts[layer_name]
                print(f"  Captured activations: {acts[layer_name].shape}")

            # Store texts once
            if layer_idx == layer_indices[0]:
                all_texts = batch_texts

    print(f"\nCaptured activations for {len(all_activations)} layers")

    # PHASE 4: Geometric Analysis
    print("\n" + "=" * 80)
    print("PHASE 4: Geometric Analysis")
    print("=" * 80)

    geometric_results = {}

    for layer_idx, activations in all_activations.items():
        print(f"\nAnalyzing layer {layer_idx}...")

        analyzer = ManifoldAnalyzer(activations)
        results = analyzer.analyze_all(n_components=50)

        geometric_results[layer_idx] = results

        print(f"  Intrinsic dimension (95%): {results['intrinsic_dimension']['intrinsic_dim_95pct_var']}")
        print(f"  Geometry type: {results['local_geometry']['geometry_type']}")

    # Save geometric analysis
    geom_output = Path("./results/geometric_analysis.json")
    geom_output.parent.mkdir(parents=True, exist_ok=True)

    with open(geom_output, 'w') as f:
        # Remove numpy arrays for JSON serialization
        save_results = {}
        for layer_idx, results in geometric_results.items():
            save_results[f"layer_{layer_idx}"] = {
                'activation_statistics': results['activation_statistics'],
                'intrinsic_dimension': results['intrinsic_dimension'],
                'local_geometry': results['local_geometry']
            }
        json.dump(save_results, f, indent=2)

    print(f"\nGeometric analysis saved to {geom_output}")

    # PHASE 5: Feature Extraction (Optional SAE Training)
    print("\n" + "=" * 80)
    print("PHASE 5: Feature Extraction")
    print("=" * 80)

    feature_results = {}

    if not args.skip_sae:
        print("Training Sparse Autoencoders...")

        for layer_idx, activations in all_activations.items():
            print(f"\nTraining SAE for layer {layer_idx}...")

            # Create dataloader from activations
            sae_dataloader = SAETrainer.create_dataloader_from_activations(
                activations,
                batch_size=config.analysis.sae_batch_size,
                shuffle=True
            )

            # Train SAE (small version for demo)
            trainer = SAETrainer(
                input_dim=arch_info['hidden_size'],
                hidden_dim=arch_info['hidden_size'] * 4,  # 4x expansion
                l1_coefficient=config.analysis.sae_l1_coefficient,
                learning_rate=config.analysis.sae_learning_rate,
                device=config.model.device
            )

            history = trainer.train(
                sae_dataloader,
                num_epochs=3,  # Quick training for demo
            )

            # Analyze features
            print(f"  Analyzing features...")
            feature_analyzer = FeatureAnalyzer(trainer.sae, tokenizer, model)

            features = feature_analyzer.analyze_all_features(
                activations,
                all_texts,
                max_features=20,  # Analyze top 20 features
                min_sparsity=0.001
            )

            feature_results[layer_idx] = features

            # Save SAE
            sae_path = f"./results/sae_layer_{layer_idx}.pt"
            trainer.save_model(sae_path)
            print(f"  Saved SAE to {sae_path}")

    else:
        print("Skipping SAE training (--skip-sae flag set)")

    # PHASE 6: Circuit Discovery
    print("\n" + "=" * 80)
    print("PHASE 6: Circuit Discovery")
    print("=" * 80)

    circuit_discoverer = CircuitDiscovery(model, tokenizer)

    test_inputs = [
        "The capital of France is",
        "Two plus two equals",
        "The quick brown fox",
    ]

    discovered_circuits = []

    for input_text in test_inputs:
        print(f"\nAnalyzing: '{input_text}'")
        circuit = circuit_discoverer.discover_circuit_structure(input_text)
        discovered_circuits.append(circuit)

    # Summarize circuits
    circuit_summary = circuit_discoverer.summarize_circuits(discovered_circuits)
    print(f"\nDiscovered {circuit_summary['num_circuits']} circuit patterns")

    # PHASE 7: Generate Documentation
    print("\n" + "=" * 80)
    print("PHASE 7: Generating Documentation")
    print("=" * 80)

    doc_generator = DocumentationGenerator(
        output_dir=config.documentation.output_dir
    )

    # Prepare layer info
    layer_info = []
    for layer_idx in layer_indices:
        geom = geometric_results.get(layer_idx, {})
        intrinsic = geom.get('intrinsic_dimension', {})

        layer_info.append({
            'layer_id': layer_idx,
            'layer_type': 'transformer',
            'hidden_dim': arch_info['hidden_size'],
            'num_heads': arch_info['num_attention_heads'],
            'intermediate_dim': arch_info['intermediate_size'],
            'num_features_discovered': len(feature_results.get(layer_idx, [])),
            'intrinsic_dimension': intrinsic.get('intrinsic_dim_95pct_var'),
            'geometry_type': geom.get('local_geometry', {}).get('geometry_type'),
            'compression_ratio': intrinsic.get('compression_ratio')
        })

    # Generate full documentation
    full_docs = doc_generator.generate_full_documentation(
        model_name=config.model.model_name,
        architecture=arch_info,
        features_by_layer=feature_results,
        circuits=discovered_circuits,
        layer_info=layer_info
    )

    # Save documentation
    doc_generator.save_documentation(full_docs, format='json')
    doc_generator.save_documentation(full_docs, format='markdown')
    doc_generator.save_summary(full_docs)

    print(f"\nDocumentation generated!")
    print(f"Transparency Score: {full_docs.transparency_score:.2%}")

    # PHASE 8: Verification (Optional)
    print("\n" + "=" * 80)
    print("PHASE 8: Verification Tests")
    print("=" * 80)

    interventions = ModelInterventions(model, tokenizer)

    # Test layer importance
    test_layer = layer_indices[0]
    print(f"\nTesting importance of layer {test_layer}...")

    importance_results = interventions.test_feature_importance(
        test_inputs,
        test_layer
    )

    print(f"  Mean impact: {importance_results['mean_impact']:.3f}")
    print(f"  Outputs changed: {importance_results['num_changed']}/{importance_results['num_tests']}")

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Transparency Score: {full_docs.transparency_score:.2%}")
    print(f"  - Features Discovered: {full_docs.total_features_discovered}")
    print(f"  - Circuits Analyzed: {full_docs.total_circuits_discovered}")
    print(f"  - Layers Analyzed: {len(layer_indices)}")
    print(f"\nOutput files:")
    print(f"  - {geom_output}")
    print(f"  - {config.documentation.output_dir}/model_documentation.json")
    print(f"  - {config.documentation.output_dir}/model_documentation.md")
    print(f"  - {config.documentation.output_dir}/summary.json")
    print()


if __name__ == "__main__":
    main()
