"""Generate comprehensive documentation from analysis results."""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from .schema import *


class DocumentationGenerator:
    """Generate comprehensive documentation from analysis results."""

    def __init__(self, output_dir: Path):
        """
        Initialize documentation generator.

        Args:
            output_dir: Directory to save documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_feature_docs(
        self,
        features: List[Dict],
        layer: int
    ) -> List[FeatureDocumentation]:
        """
        Generate documentation for features.

        Args:
            features: List of feature analysis results
            layer: Layer index

        Returns:
            List of FeatureDocumentation objects
        """
        feature_docs = []

        for feature_data in features:
            # Extract activation statistics
            act_stats = feature_data.get('activation_statistics')
            if act_stats:
                act_stats_obj = ActivationStatistics(**act_stats)
            else:
                act_stats_obj = None

            # Extract semantic domain
            semantic = feature_data.get('semantic_domain')
            if semantic:
                semantic_obj = SemanticDomain(**semantic)
            else:
                semantic_obj = None

            doc = FeatureDocumentation(
                feature_id=feature_data['feature_id'],
                layer=layer,
                activation_statistics=act_stats_obj,
                semantic_domain=semantic_obj,
                top_examples=feature_data.get('top_examples', [])
            )

            feature_docs.append(doc)

        return feature_docs

    def generate_circuit_docs(
        self,
        circuits: List[Dict]
    ) -> List[CircuitDocumentation]:
        """
        Generate documentation for circuits.

        Args:
            circuits: List of circuit analysis results

        Returns:
            List of CircuitDocumentation objects
        """
        circuit_docs = []

        for i, circuit_data in enumerate(circuits):
            doc = CircuitDocumentation(
                circuit_id=f"circuit_{i}",
                name=circuit_data.get('name', f"Circuit {i}"),
                description=circuit_data.get('description', 'Auto-discovered circuit'),
                layers=circuit_data.get('layers', []),
                attention_heads=circuit_data.get('attention_heads', []),
                pattern_distribution=circuit_data.get('pattern_distribution', {})
            )

            circuit_docs.append(doc)

        return circuit_docs

    def generate_layer_docs(
        self,
        layer_info: List[Dict]
    ) -> List[LayerDocumentation]:
        """
        Generate documentation for layers.

        Args:
            layer_info: List of layer analysis results

        Returns:
            List of LayerDocumentation objects
        """
        layer_docs = []

        for layer_data in layer_info:
            doc = LayerDocumentation(
                layer_id=layer_data['layer_id'],
                layer_type=layer_data.get('layer_type', 'transformer'),
                hidden_dim=layer_data['hidden_dim'],
                num_heads=layer_data.get('num_heads'),
                intermediate_dim=layer_data.get('intermediate_dim'),
                num_features_discovered=layer_data.get('num_features_discovered', 0),
                intrinsic_dimension=layer_data.get('intrinsic_dimension'),
                geometry_type=layer_data.get('geometry_type'),
                compression_ratio=layer_data.get('compression_ratio')
            )

            layer_docs.append(doc)

        return layer_docs

    def generate_full_documentation(
        self,
        model_name: str,
        architecture: Dict,
        features_by_layer: Dict[int, List[Dict]],
        circuits: List[Dict],
        layer_info: List[Dict]
    ) -> ModelDocumentation:
        """
        Generate complete model documentation.

        Args:
            model_name: Name of the model
            architecture: Architecture information
            features_by_layer: Features grouped by layer
            circuits: Circuit analysis results
            layer_info: Layer analysis results

        Returns:
            Complete ModelDocumentation
        """
        # Generate feature documentation for all layers
        all_features = []
        for layer, features in features_by_layer.items():
            all_features.extend(self.generate_feature_docs(features, layer))

        # Generate circuit documentation
        circuit_docs = self.generate_circuit_docs(circuits)

        # Generate layer documentation
        layer_docs = self.generate_layer_docs(layer_info)

        # Calculate transparency score
        transparency_score = self._calculate_transparency_score(
            len(all_features),
            len(circuit_docs),
            layer_docs
        )

        # Create full documentation
        full_docs = ModelDocumentation(
            model_name=model_name,
            model_size=architecture.get('model_size', 'unknown'),
            architecture=architecture,
            features=all_features,
            circuits=circuit_docs,
            layers=layer_docs,
            total_features_discovered=len(all_features),
            total_circuits_discovered=len(circuit_docs),
            transparency_score=transparency_score
        )

        return full_docs

    def _calculate_transparency_score(
        self,
        num_features: int,
        num_circuits: int,
        layers: List[LayerDocumentation]
    ) -> float:
        """
        Calculate overall transparency score (0-1).

        Args:
            num_features: Number of features discovered
            num_circuits: Number of circuits discovered
            layers: Layer documentation

        Returns:
            Transparency score between 0 and 1
        """
        # Component scores
        feature_score = min(num_features / 1000, 1.0)  # Expect ~1000 features
        circuit_score = min(num_circuits / 10, 1.0)     # Expect ~10 circuits

        # Layer understanding score
        layers_with_analysis = sum(1 for l in layers if l.num_features_discovered > 0)
        layer_score = layers_with_analysis / max(len(layers), 1)

        # Weighted average
        transparency_score = (
            0.4 * feature_score +
            0.3 * circuit_score +
            0.3 * layer_score
        )

        return min(transparency_score, 1.0)

    def save_documentation(
        self,
        docs: ModelDocumentation,
        format: str = 'json'
    ):
        """
        Save documentation to disk.

        Args:
            docs: ModelDocumentation to save
            format: Output format ('json' or 'markdown')
        """
        if format == 'json':
            output_file = self.output_dir / 'model_documentation.json'

            # Convert to dict and handle datetime serialization
            docs_dict = docs.dict()

            with open(output_file, 'w') as f:
                json.dump(docs_dict, f, indent=2, default=str)

            print(f"Documentation saved to {output_file}")

        elif format == 'markdown':
            output_file = self.output_dir / 'model_documentation.md'
            markdown = self._generate_markdown(docs)

            with open(output_file, 'w') as f:
                f.write(markdown)

            print(f"Documentation saved to {output_file}")

    def _generate_markdown(self, docs: ModelDocumentation) -> str:
        """
        Generate human-readable markdown documentation.

        Args:
            docs: ModelDocumentation

        Returns:
            Markdown string
        """
        md = f"# {docs.model_name} - Transparency Documentation\n\n"
        md += f"**Generated:** {docs.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += f"**Transparency Score:** {docs.transparency_score:.2%}\n\n"
        md += f"---\n\n"

        # Summary
        md += "## Summary\n\n"
        md += f"- **Model Size:** {docs.model_size}\n"
        md += f"- **Features Discovered:** {docs.total_features_discovered}\n"
        md += f"- **Circuits Discovered:** {docs.total_circuits_discovered}\n"
        md += f"- **Layers Analyzed:** {len(docs.layers)}\n\n"

        # Architecture
        md += "## Model Architecture\n\n"
        md += "```json\n"
        md += json.dumps(docs.architecture, indent=2)
        md += "\n```\n\n"

        # Features by layer
        if docs.features:
            md += "## Features by Layer\n\n"

            features_by_layer = {}
            for feature in docs.features:
                layer = feature.layer
                if layer not in features_by_layer:
                    features_by_layer[layer] = []
                features_by_layer[layer].append(feature)

            for layer in sorted(features_by_layer.keys()):
                features = features_by_layer[layer]
                md += f"### Layer {layer}\n\n"
                md += f"**Total Features:** {len(features)}\n\n"

                # Show top 3 features
                for i, feature in enumerate(features[:3]):
                    md += f"#### Feature {feature.feature_id}\n"

                    if feature.semantic_domain:
                        md += f"- **Domain:** {', '.join(feature.semantic_domain.primary_domains)}\n"
                        md += f"- **Interpretation:** {feature.semantic_domain.interpretation}\n"

                    if feature.activation_statistics:
                        md += f"- **Sparsity:** {feature.activation_statistics.sparsity:.3f}\n"
                        md += f"- **Mean Activation:** {feature.activation_statistics.mean:.4f}\n"

                    md += "\n"

                if len(features) > 3:
                    md += f"*...and {len(features) - 3} more features*\n\n"

        # Circuits
        if docs.circuits:
            md += "## Discovered Circuits\n\n"

            for circuit in docs.circuits:
                md += f"### {circuit.name}\n\n"
                md += f"{circuit.description}\n\n"
                md += f"- **Layers Involved:** {len(circuit.layers)}\n"
                md += f"- **Attention Heads:** {len(circuit.attention_heads)}\n"

                if circuit.pattern_distribution:
                    md += f"- **Pattern Distribution:**\n"
                    for pattern, count in circuit.pattern_distribution.items():
                        md += f"  - {pattern}: {count}\n"

                md += "\n"

        # Layer Analysis
        if docs.layers:
            md += "## Layer Analysis\n\n"

            md += "| Layer | Type | Hidden Dim | Features | Intrinsic Dim | Geometry |\n"
            md += "|-------|------|------------|----------|---------------|----------|\n"

            for layer in docs.layers:
                intrinsic = layer.intrinsic_dimension if layer.intrinsic_dimension else 'N/A'
                geometry = layer.geometry_type if layer.geometry_type else 'N/A'

                md += f"| {layer.layer_id} | {layer.layer_type} | {layer.hidden_dim} | "
                md += f"{layer.num_features_discovered} | {intrinsic} | {geometry} |\n"

            md += "\n"

        return md

    def save_summary(self, docs: ModelDocumentation):
        """Save a brief summary in JSON format."""
        summary = {
            'model_name': docs.model_name,
            'model_size': docs.model_size,
            'transparency_score': docs.transparency_score,
            'total_features': docs.total_features_discovered,
            'total_circuits': docs.total_circuits_discovered,
            'num_layers': len(docs.layers),
            'generated_at': str(docs.created_at)
        }

        output_file = self.output_dir / 'summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {output_file}")
