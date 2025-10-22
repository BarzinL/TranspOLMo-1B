"""Sparse Autoencoder for extracting monosemantic features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting monosemantic features.

    Based on Anthropic's "Towards Monosemanticity" work.
    Uses L1 sparsity penalty to encourage sparse, interpretable features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coefficient: float = 0.001,
        tie_weights: bool = True
    ):
        """
        Initialize Sparse Autoencoder.

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of hidden layer (typically 8-32x input_dim)
            l1_coefficient: Weight for L1 sparsity penalty
            tie_weights: Whether to tie encoder and decoder weights
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.tie_weights = tie_weights

        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: hidden_dim -> input_dim
        if tie_weights:
            # Tied weights: decoder weight is transpose of encoder weight
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with normalized columns."""
        # Kaiming initialization
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')

        # Normalize encoder columns to unit norm (important for SAEs)
        with torch.no_grad():
            self.encoder.weight.data = F.normalize(self.encoder.weight.data, dim=0)

        # Initialize biases
        nn.init.zeros_(self.encoder.bias)

        if not self.tie_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature space.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Sparse features (..., hidden_dim)
        """
        return F.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode from sparse features back to input space.

        Args:
            h: Hidden features (..., hidden_dim)

        Returns:
            Reconstructed input (..., input_dim)
        """
        if self.tie_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Tuple of (reconstruction, hidden_activations)
        """
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        h: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss + L1 sparsity penalty.

        Args:
            x: Original input
            x_recon: Reconstructed input
            h: Hidden activations

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # L1 sparsity penalty
        l1_loss = self.l1_coefficient * torch.mean(torch.abs(h))

        # Total loss
        total_loss = recon_loss + l1_loss

        # Additional metrics
        mean_activation = torch.mean(h)
        fraction_active = torch.mean((h > 0).float())
        l0_norm = torch.mean((h > 0).float().sum(dim=-1))  # Average number of active features

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'mean_activation': mean_activation,
            'fraction_active': fraction_active,
            'l0_norm': l0_norm
        }

    def get_feature_statistics(self, dataloader: DataLoader, device: str = 'cuda') -> Dict:
        """
        Compute statistics about learned features.

        Args:
            dataloader: DataLoader with activation data
            device: Device to run on

        Returns:
            Dictionary with feature statistics
        """
        self.eval()
        self.to(device)

        all_activations = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # Flatten if needed
                original_shape = x.shape
                if len(x.shape) > 2:
                    x = x.reshape(-1, x.shape[-1])

                _, h = self.forward(x)
                all_activations.append(h.cpu())

        # Concatenate all activations
        all_h = torch.cat(all_activations, dim=0)

        # Per-feature statistics
        feature_activations = all_h.mean(dim=0)  # Average activation per feature
        feature_sparsity = (all_h > 0).float().mean(dim=0)  # How often each feature fires
        feature_max = all_h.max(dim=0)[0]  # Maximum activation per feature

        # Dead features (never activate)
        dead_features = (feature_sparsity < 0.0001).sum().item()

        # Frequently active features
        frequent_features = (feature_sparsity > 0.1).sum().item()

        return {
            'num_features': self.hidden_dim,
            'mean_activation': float(feature_activations.mean()),
            'median_activation': float(feature_activations.median()),
            'std_activation': float(feature_activations.std()),
            'mean_sparsity': float(feature_sparsity.mean()),
            'median_sparsity': float(feature_sparsity.median()),
            'dead_features': dead_features,
            'dead_feature_ratio': dead_features / self.hidden_dim,
            'frequent_features': frequent_features,
            'frequent_feature_ratio': frequent_features / self.hidden_dim,
            'l0_norm': float((all_h > 0).float().sum(dim=1).mean()),  # Avg # active features
            'feature_activation_distribution': {
                'min': float(feature_activations.min()),
                'max': float(feature_activations.max()),
                '10th_percentile': float(torch.quantile(feature_activations, 0.1)),
                '90th_percentile': float(torch.quantile(feature_activations, 0.9)),
            }
        }


class SAETrainer:
    """Train sparse autoencoders on activation data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coefficient: float = 0.001,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        """
        Initialize SAE trainer.

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of hidden layer
            l1_coefficient: L1 sparsity coefficient
            learning_rate: Learning rate for optimizer
            device: Device to train on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.sae = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l1_coefficient=l1_coefficient
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=learning_rate
        )

        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'l1_loss': [],
            'l0_norm': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with activation data

        Returns:
            Dictionary with epoch statistics
        """
        self.sae.train()

        epoch_stats = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'l1_loss': 0.0,
            'mean_activation': 0.0,
            'fraction_active': 0.0,
            'l0_norm': 0.0
        }

        num_batches = 0

        for batch in tqdm(dataloader, desc="Training SAE", leave=False):
            # Extract data from batch
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Flatten sequence dimension if present
            original_shape = x.shape
            if len(x.shape) > 2:
                x = x.reshape(-1, x.shape[-1])

            # Forward pass
            x_recon, h = self.sae(x)

            # Compute loss
            losses = self.sae.compute_loss(x, x_recon, h)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Renormalize encoder weights (important for SAE stability)
            with torch.no_grad():
                self.sae.encoder.weight.data = F.normalize(
                    self.sae.encoder.weight.data,
                    dim=0
                )

            # Accumulate stats
            for key in epoch_stats:
                if key in losses:
                    epoch_stats[key] += losses[key].item()

            num_batches += 1

        # Average stats
        for key in epoch_stats:
            epoch_stats[key] /= max(num_batches, 1)

        return epoch_stats

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_dataloader: Training data loader
            num_epochs: Number of epochs to train
            val_dataloader: Optional validation data loader
            save_path: Optional path to save model

        Returns:
            Training history
        """
        print(f"\nTraining Sparse Autoencoder")
        print(f"Input dim: {self.sae.input_dim}, Hidden dim: {self.sae.hidden_dim}")
        print(f"Expansion factor: {self.sae.hidden_dim / self.sae.input_dim:.1f}x")
        print(f"Device: {self.device}\n")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_stats = self.train_epoch(train_dataloader)
            self.history['train_loss'].append(train_stats['total_loss'])
            self.history['recon_loss'].append(train_stats['recon_loss'])
            self.history['l1_loss'].append(train_stats['l1_loss'])
            self.history['l0_norm'].append(train_stats['l0_norm'])

            print(f"  Train - Loss: {train_stats['total_loss']:.6f}, "
                  f"Recon: {train_stats['recon_loss']:.6f}, "
                  f"L1: {train_stats['l1_loss']:.6f}, "
                  f"L0: {train_stats['l0_norm']:.1f}, "
                  f"Active: {train_stats['fraction_active']:.2%}")

            # Validation
            if val_dataloader:
                val_stats = self.validate(val_dataloader)
                print(f"  Val   - Loss: {val_stats['total_loss']:.6f}, "
                      f"Recon: {val_stats['recon_loss']:.6f}")

                # Save best model
                if val_stats['total_loss'] < best_loss:
                    best_loss = val_stats['total_loss']
                    if save_path:
                        self.save_model(save_path)
                        print(f"  Saved best model to {save_path}")

        # Save final model if no validation
        if save_path and not val_dataloader:
            self.save_model(save_path)
            print(f"\nSaved final model to {save_path}")

        return self.history

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        """
        Validation pass.

        Args:
            dataloader: Validation data loader

        Returns:
            Validation statistics
        """
        self.sae.eval()

        val_stats = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'l1_loss': 0.0,
            'mean_activation': 0.0,
            'fraction_active': 0.0,
            'l0_norm': 0.0
        }

        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            if len(x.shape) > 2:
                x = x.reshape(-1, x.shape[-1])

            x_recon, h = self.sae(x)
            losses = self.sae.compute_loss(x, x_recon, h)

            for key in val_stats:
                if key in losses:
                    val_stats[key] += losses[key].item()

            num_batches += 1

        for key in val_stats:
            val_stats[key] /= max(num_batches, 1)

        return val_stats

    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.sae.input_dim,
                'hidden_dim': self.sae.hidden_dim,
                'l1_coefficient': self.sae.l1_coefficient,
                'tie_weights': self.sae.tie_weights,
            },
            'history': self.history
        }, filepath)

    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.sae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

    @staticmethod
    def create_dataloader_from_activations(
        activations: torch.Tensor,
        batch_size: int = 4096,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create DataLoader from activation tensor.

        Args:
            activations: Activation tensor (num_samples, seq_len, hidden_dim) or (num_samples, hidden_dim)
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader
        """
        # Flatten if needed
        if len(activations.shape) == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        dataset = TensorDataset(activations)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
