"""Activation capture using forward hooks."""

import torch
from typing import Dict, List, Callable, Optional
from collections import defaultdict


class ActivationCapture:
    """Capture activations from specified layers during forward pass."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize activation capturer.

        Args:
            model: PyTorch model to hook
            device: Device to store activations on ('cpu' recommended to save GPU memory)
        """
        self.model = model
        self.device = device
        self.activations = defaultdict(list)
        self.hooks = []
        self.capture_enabled = True

    def register_hooks(self, layer_names: List[str], store_on_cpu: bool = True):
        """
        Register forward hooks to capture activations.

        Args:
            layer_names: List of layer names to hook
            store_on_cpu: If True, move activations to CPU immediately to save GPU memory
        """

        def make_hook(name: str) -> Callable:
            """Create a hook function for a specific layer."""
            def hook(module, input, output):
                if not self.capture_enabled:
                    return

                # Handle different output types
                if isinstance(output, tuple):
                    activation = output[0]  # Usually the main tensor
                else:
                    activation = output

                # Detach from computation graph
                activation = activation.detach()

                # Move to CPU if requested to save GPU memory
                if store_on_cpu and activation.device.type == 'cuda':
                    activation = activation.cpu()

                self.activations[name].append(activation)

            return hook

        # Find and hook the specified layers
        hooked_layers = set()

        for name, module in self.model.named_modules():
            # Check if this layer matches any of the requested layer names
            for pattern in layer_names:
                if pattern in name or name == pattern:
                    if name not in hooked_layers:
                        handle = module.register_forward_hook(make_hook(name))
                        self.hooks.append(handle)
                        hooked_layers.add(name)
                        break

        print(f"Registered hooks for {len(hooked_layers)} layers")

        if len(hooked_layers) < len(layer_names):
            print(f"Warning: Only found {len(hooked_layers)} layers out of {len(layer_names)} requested")

    def enable_capture(self):
        """Enable activation capture."""
        self.capture_enabled = True

    def disable_capture(self):
        """Disable activation capture (hooks remain but don't store)."""
        self.capture_enabled = False

    def clear(self):
        """Clear captured activations to free memory."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self, concatenate: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get all captured activations.

        Args:
            concatenate: If True, concatenate all batches along batch dimension

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if concatenate:
            # Concatenate all batches
            return {
                name: torch.cat(acts, dim=0)
                for name, acts in self.activations.items()
                if len(acts) > 0
            }
        else:
            # Return as lists of tensors
            return dict(self.activations)

    def get_layer_activation(self, layer_name: str, concatenate: bool = True) -> Optional[torch.Tensor]:
        """
        Get activations for a specific layer.

        Args:
            layer_name: Name of layer
            concatenate: If True, concatenate all batches

        Returns:
            Activation tensor or None if layer not found
        """
        if layer_name not in self.activations:
            return None

        acts = self.activations[layer_name]
        if len(acts) == 0:
            return None

        if concatenate:
            return torch.cat(acts, dim=0)
        else:
            return acts

    def get_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about captured activations.

        Returns:
            Dictionary with statistics for each layer
        """
        stats = {}

        for name, acts in self.activations.items():
            if len(acts) == 0:
                continue

            # Concatenate for statistics
            all_acts = torch.cat(acts, dim=0)

            stats[name] = {
                'shape': tuple(all_acts.shape),
                'num_batches': len(acts),
                'total_samples': all_acts.shape[0],
                'mean': float(all_acts.mean()),
                'std': float(all_acts.std()),
                'min': float(all_acts.min()),
                'max': float(all_acts.max()),
                'memory_mb': all_acts.element_size() * all_acts.nelement() / (1024 ** 2)
            }

        return stats

    def save_activations(self, filepath: str):
        """
        Save activations to disk.

        Args:
            filepath: Path to save file (.pt or .pth)
        """
        activations = self.get_activations(concatenate=True)
        torch.save(activations, filepath)
        print(f"Saved activations to {filepath}")

    def load_activations(self, filepath: str):
        """
        Load activations from disk.

        Args:
            filepath: Path to load file from
        """
        self.clear()
        loaded = torch.load(filepath)

        for name, acts in loaded.items():
            self.activations[name] = [acts]

        print(f"Loaded activations from {filepath}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.remove_hooks()
