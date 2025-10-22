"""Model interventions for verification and testing."""

import torch
from typing import Dict, Callable, Optional


class ModelInterventions:
    """Test model behavior through systematic interventions."""

    def __init__(self, model, tokenizer):
        """
        Initialize interventions system.

        Args:
            model: Language model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def test_model_output(self, input_text: str, max_new_tokens: int = 10) -> Dict:
        """
        Get baseline model output for text.

        Args:
            input_text: Input text
            max_new_tokens: Number of tokens to generate

        Returns:
            Dictionary with output information
        """
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            'input_text': input_text,
            'output_text': output_text,
            'output_tokens': outputs[0].tolist()
        }

    def ablate_layer(
        self,
        input_text: str,
        layer_idx: int,
        component: str = 'all'
    ) -> Dict:
        """
        Ablate (zero out) a layer and measure impact.

        Args:
            input_text: Input text
            layer_idx: Layer index to ablate
            component: Which component to ablate ('attention', 'mlp', or 'all')

        Returns:
            Results dictionary
        """
        # Get baseline
        baseline = self.test_model_output(input_text)

        # Define ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                # Zero out the main output tensor
                return (torch.zeros_like(output[0]),) + output[1:]
            else:
                return torch.zeros_like(output)

        # Find the layer to ablate
        layer_module = None
        for name, module in self.model.named_modules():
            if f"layers.{layer_idx}" in name:
                if component == 'all' and f"layers.{layer_idx}" == name.split('.')[-2:][0] + '.' + name.split('.')[-2:][1]:
                    layer_module = module
                    break
                elif component == 'attention' and 'self_attn' in name:
                    layer_module = module
                    break
                elif component == 'mlp' and 'mlp' in name:
                    layer_module = module
                    break

        if layer_module is None:
            return {
                'error': f'Could not find layer {layer_idx} component {component}',
                'baseline': baseline
            }

        # Apply ablation
        hook = layer_module.register_forward_hook(ablation_hook)

        try:
            ablated_output = self.test_model_output(input_text)
        finally:
            hook.remove()

        # Measure impact
        impact = self._measure_output_difference(
            baseline['output_text'],
            ablated_output['output_text']
        )

        return {
            'layer': layer_idx,
            'component': component,
            'baseline_output': baseline['output_text'],
            'ablated_output': ablated_output['output_text'],
            'impact_score': impact,
            'outputs_differ': baseline['output_text'] != ablated_output['output_text']
        }

    def _measure_output_difference(self, text1: str, text2: str) -> float:
        """
        Measure difference between two text outputs.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Difference score (0 = identical, 1 = completely different)
        """
        # Simple character-level difference
        if text1 == text2:
            return 0.0

        # Compute edit distance (normalized)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0

        # Simple difference: count differing characters
        diff_count = sum(c1 != c2 for c1, c2 in zip(text1, text2))
        diff_count += abs(len(text1) - len(text2))

        return min(diff_count / max_len, 1.0)

    def test_feature_importance(
        self,
        input_texts: list,
        layer_idx: int
    ) -> Dict:
        """
        Test importance of different components on a set of inputs.

        Args:
            input_texts: List of input texts
            layer_idx: Layer to test

        Returns:
            Summary of importance tests
        """
        results = []

        for text in input_texts[:10]:  # Limit to 10 examples
            result = self.ablate_layer(text, layer_idx)
            results.append(result)

        # Aggregate results
        impact_scores = [r.get('impact_score', 0) for r in results if 'impact_score' in r]

        return {
            'layer': layer_idx,
            'num_tests': len(results),
            'mean_impact': float(sum(impact_scores) / len(impact_scores)) if impact_scores else 0.0,
            'max_impact': float(max(impact_scores)) if impact_scores else 0.0,
            'num_changed': sum(1 for r in results if r.get('outputs_differ', False)),
            'results': results
        }
