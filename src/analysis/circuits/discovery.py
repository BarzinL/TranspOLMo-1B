"""Circuit discovery through attention pattern analysis and activation patching."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx


class CircuitDiscovery:
    """Discover computational circuits in the model."""

    def __init__(self, model, tokenizer):
        """
        Initialize circuit discoverer.

        Args:
            model: Language model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def trace_attention_patterns(
        self,
        input_text: str,
        layer_range: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Trace attention patterns across layers.

        Args:
            input_text: Input text to analyze
            layer_range: Optional tuple of (start_layer, end_layer)

        Returns:
            Dictionary with attention pattern analysis
        """
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Forward pass with attention outputs
        outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of (batch, heads, seq, seq) tensors

        if layer_range:
            start, end = layer_range
            attentions = attentions[start:end]
            layer_offset = start
        else:
            layer_offset = 0

        # Analyze patterns for each layer
        attention_analysis = []

        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn = layer_attn[0]  # Remove batch dimension: (heads, seq, seq)

            layer_info = {
                'layer': layer_idx + layer_offset,
                'heads': []
            }

            for head_idx in range(layer_attn.shape[0]):
                head_attn = layer_attn[head_idx].cpu().numpy()  # (seq, seq)

                # Classify attention pattern
                pattern_type = self._classify_attention_pattern(head_attn)

                layer_info['heads'].append({
                    'head': head_idx,
                    'pattern_type': pattern_type,
                    'mean_attention': float(head_attn.mean()),
                    'max_attention': float(head_attn.max()),
                })

            attention_analysis.append(layer_info)

        return {
            'input_text': input_text,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'attention_patterns': attention_analysis
        }

    def _classify_attention_pattern(self, attention_matrix: np.ndarray) -> str:
        """
        Classify attention pattern into types.

        Args:
            attention_matrix: Attention matrix (seq, seq)

        Returns:
            Pattern type string
        """
        seq_len = len(attention_matrix)

        if seq_len < 2:
            return 'too_short'

        # Check for diagonal pattern (local attention)
        diagonal_mass = np.sum(np.diag(attention_matrix))
        if seq_len > 1:
            diagonal_mass += np.sum(np.diag(attention_matrix, k=-1))
            diagonal_mass += np.sum(np.diag(attention_matrix, k=1))

        if diagonal_mass / seq_len > 0.5:
            return 'local'

        # Check for previous token pattern (causal)
        lower_tri = np.tril(attention_matrix, k=-1)
        if np.sum(lower_tri) / (seq_len * (seq_len - 1) / 2 + 0.001) > 0.8:
            return 'previous'

        # Check for first token attention
        first_token_mass = np.sum(attention_matrix[:, 0]) / seq_len
        if first_token_mass > 0.5:
            return 'first_token'

        # Check for uniform (broadcasting)
        entropy = self._compute_entropy(attention_matrix)
        max_entropy = np.log(seq_len)
        if entropy > 0.8 * max_entropy:
            return 'uniform'

        return 'mixed'

    def _compute_entropy(self, attention_matrix: np.ndarray) -> float:
        """Compute average entropy of attention distribution."""
        eps = 1e-10
        entropies = -np.sum(attention_matrix * np.log(attention_matrix + eps), axis=1)
        return float(np.mean(entropies))

    def discover_circuit_structure(
        self,
        input_text: str,
        output_token: Optional[str] = None
    ) -> Dict:
        """
        Discover circuit structure for generating a specific output.

        Args:
            input_text: Input text
            output_token: Optional target output token

        Returns:
            Circuit structure information
        """
        # Get attention patterns
        attention_info = self.trace_attention_patterns(input_text)

        # Build connectivity graph
        graph = self._build_attention_graph(attention_info)

        # Find important paths
        paths = self._find_important_paths(graph)

        return {
            'input_text': input_text,
            'output_token': output_token,
            'attention_patterns': attention_info['attention_patterns'],
            'circuit_graph': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
            },
            'important_paths': paths,
        }

    def _build_attention_graph(self, attention_info: Dict) -> nx.DiGraph:
        """
        Build directed graph representing attention flow.

        Args:
            attention_info: Attention pattern information

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        for layer_info in attention_info['attention_patterns']:
            layer_idx = layer_info['layer']

            for head_info in layer_info['heads']:
                head_idx = head_info['head']

                # Add node for this attention head
                node_id = f"L{layer_idx}H{head_idx}"
                G.add_node(node_id, layer=layer_idx, head=head_idx, type=head_info['pattern_type'])

        # Add edges between layers (simplified)
        for i in range(len(attention_info['attention_patterns']) - 1):
            layer_i = attention_info['attention_patterns'][i]
            layer_j = attention_info['attention_patterns'][i + 1]

            for head_i in layer_i['heads']:
                for head_j in layer_j['heads']:
                    node_i = f"L{layer_i['layer']}H{head_i['head']}"
                    node_j = f"L{layer_j['layer']}H{head_j['head']}"
                    G.add_edge(node_i, node_j)

        return G

    def _find_important_paths(self, graph: nx.DiGraph, max_paths: int = 10) -> List[List[str]]:
        """
        Find important paths through the network.

        Args:
            graph: Circuit graph
            max_paths: Maximum number of paths to return

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if graph.number_of_nodes() == 0:
            return []

        # Find source and sink nodes
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if not sources or not sinks:
            return []

        # Find paths from sources to sinks
        paths = []
        for source in sources[:3]:  # Limit sources
            for sink in sinks[:3]:  # Limit sinks
                try:
                    all_paths = list(nx.all_simple_paths(graph, source, sink, cutoff=10))
                    paths.extend(all_paths[:max_paths // 9])
                except nx.NetworkXNoPath:
                    continue

        return paths[:max_paths]

    def summarize_circuits(self, circuits: List[Dict]) -> Dict:
        """
        Summarize discovered circuits.

        Args:
            circuits: List of circuit dictionaries

        Returns:
            Summary statistics
        """
        if not circuits:
            return {
                'num_circuits': 0,
                'avg_path_length': 0,
                'pattern_distribution': {}
            }

        # Collect statistics
        total_paths = sum(len(c.get('important_paths', [])) for c in circuits)
        path_lengths = []

        pattern_counts = {}

        for circuit in circuits:
            for layer_info in circuit.get('attention_patterns', []):
                for head_info in layer_info.get('heads', []):
                    pattern = head_info.get('pattern_type', 'unknown')
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            for path in circuit.get('important_paths', []):
                path_lengths.append(len(path))

        return {
            'num_circuits': len(circuits),
            'total_paths': total_paths,
            'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0,
            'max_path_length': int(np.max(path_lengths)) if path_lengths else 0,
            'pattern_distribution': pattern_counts
        }
