"""Configuration management for TranspOLMo analysis pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model loading and analysis."""

    model_name: str = "allenai/OLMo-2-1124-1B"
    cache_dir: Path = Path("./data/models")
    device: str = "cuda"  # or "cpu"
    dtype: str = "float32"  # or "float16", "bfloat16"


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""

    dataset_name: str = "allenai/dolma"
    dataset_subset: Optional[str] = "cc_en_head"
    num_samples: int = 10000
    max_seq_length: int = 512
    batch_size: int = 16
    layers_to_capture: Optional[List[str]] = None  # None = all layers


@dataclass
class AnalysisConfig:
    """Configuration for analysis pipelines."""

    # Sparse Autoencoder settings
    sae_hidden_size: int = 16384  # Expansion factor ~16x
    sae_l1_coefficient: float = 0.001
    sae_learning_rate: float = 1e-4
    sae_batch_size: int = 4096
    sae_num_epochs: int = 10

    # Geometric analysis
    pca_components: int = 100
    clustering_algorithm: str = "kmeans"  # or "hdbscan", "spectral"
    num_clusters: int = 50

    # Circuit discovery
    attribution_method: str = "integrated_gradients"
    circuit_threshold: float = 0.01  # Min contribution to count


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""

    output_dir: Path = Path("./docs/findings")
    database_path: Path = Path("./results/transparency.db")
    format: str = "json"  # or "yaml", "markdown"


@dataclass
class Config:
    """Master configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            extraction=ExtractionConfig(),
            analysis=AnalysisConfig(),
            documentation=DocumentationConfig()
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            extraction=ExtractionConfig(**config_dict.get("extraction", {})),
            analysis=AnalysisConfig(**config_dict.get("analysis", {})),
            documentation=DocumentationConfig(**config_dict.get("documentation", {}))
        )
