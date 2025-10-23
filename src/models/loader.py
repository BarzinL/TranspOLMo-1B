"""Model loading and architecture inspection for OLMo2 models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple
from pathlib import Path


class OLMo2Loader:
    """Load and prepare OLMo2 models for analysis."""

    def __init__(self, model_name: str, cache_dir: Path, device: str = "cuda"):
        """
        Initialize loader.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache downloaded models
            device: Device to load model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Auto-detect if CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            self.device = "cpu"
        else:
            self.device = device

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading {self.model_name}...")
        print(f"Device: {self.device}")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True
        )

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        print(f"Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            dtype=torch.float32,  # Full precision for analysis
            trust_remote_code=True
        )

        # Move model to device
        print(f"Moving model to {self.device}...")
        model = model.to(self.device)

        model.eval()  # Always in eval mode for analysis

        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer

    def get_architecture_info(self, model: AutoModelForCausalLM) -> Dict:
        """
        Extract architectural constraints and information from model.

        Args:
            model: Loaded model

        Returns:
            Dictionary containing architecture information
        """
        config = model.config

        # Extract core architecture parameters
        arch_info = {
            'model_name': self.model_name,
            'model_size': f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
            'num_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'intermediate_size': config.intermediate_size,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads),
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.max_position_embeddings,
            'rope_theta': getattr(config, 'rope_theta', 10000.0),
            'head_dim': config.hidden_size // config.num_attention_heads,
        }

        # OLMo2-specific architecture details
        arch_info['architecture_constraints'] = {
            'activation_function': 'swiglu',  # OLMo2 uses SwiGLU
            'normalization': 'rmsnorm',  # RMSNorm instead of LayerNorm
            'attention_type': 'grouped_query',  # Grouped-query attention
            'no_biases': True,  # OLMo2 doesn't use biases
            'post_attention_norm': True,
            'qk_norm': getattr(config, 'qk_layernorm', False),
        }

        # Calculate derived quantities
        arch_info['total_parameters'] = sum(p.numel() for p in model.parameters())
        arch_info['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return arch_info

    def get_layer_names(self, model: AutoModelForCausalLM) -> Dict[str, list]:
        """
        Get names of all important layers for hooking.

        Args:
            model: Loaded model

        Returns:
            Dictionary mapping layer types to layer names
        """
        layer_names = {
            'embedding': [],
            'attention': [],
            'mlp': [],
            'attention_output': [],
            'mlp_output': [],
            'layer_norm': [],
        }

        for name, module in model.named_modules():
            # Embedding layer
            if 'embed_tokens' in name:
                layer_names['embedding'].append(name)

            # Attention layers
            elif 'self_attn' in name and not any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                layer_names['attention'].append(name)

            # MLP layers
            elif 'mlp' in name and not any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                layer_names['mlp'].append(name)

            # Layer norms
            elif 'norm' in name.lower():
                layer_names['layer_norm'].append(name)

        return layer_names

    def print_model_structure(self, model: AutoModelForCausalLM, max_depth: int = 3):
        """
        Print model structure for debugging.

        Args:
            model: Loaded model
            max_depth: Maximum depth to print
        """
        print("\n" + "="*80)
        print("MODEL STRUCTURE")
        print("="*80)

        for name, module in model.named_modules():
            depth = name.count('.')
            if depth <= max_depth:
                indent = "  " * depth
                module_type = type(module).__name__
                print(f"{indent}{name}: {module_type}")

        print("="*80 + "\n")
