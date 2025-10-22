"""Data schemas for documentation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class ActivationStatistics(BaseModel):
    """Statistics about feature activations."""
    mean: float
    std: float
    max: float
    sparsity: float
    percentiles: Dict[str, float]


class SemanticDomain(BaseModel):
    """Semantic characterization of a feature."""
    primary_domains: List[str]
    confidence: float
    interpretation: str


class FeatureDocumentation(BaseModel):
    """Complete documentation for a single feature."""
    feature_id: int
    layer: int
    feature_type: str = "sae_feature"

    # Activation properties
    activation_statistics: Optional[ActivationStatistics] = None

    # Semantic characterization
    semantic_domain: Optional[SemanticDomain] = None

    # Top activating examples
    top_examples: List[Dict] = Field(default_factory=list)

    # Metadata
    discovered_at: datetime = Field(default_factory=datetime.now)


class CircuitDocumentation(BaseModel):
    """Documentation for a computational circuit."""
    circuit_id: str
    name: str
    description: str

    # Components
    layers: List[int] = Field(default_factory=list)
    attention_heads: List[tuple] = Field(default_factory=list)

    # Pattern information
    pattern_distribution: Dict[str, int] = Field(default_factory=dict)

    # Metadata
    discovered_at: datetime = Field(default_factory=datetime.now)


class LayerDocumentation(BaseModel):
    """Documentation for a single layer."""
    layer_id: int
    layer_type: str

    # Architecture
    hidden_dim: int
    num_heads: Optional[int] = None
    intermediate_dim: Optional[int] = None

    # Analysis results
    num_features_discovered: int = 0
    intrinsic_dimension: Optional[int] = None

    # Geometric properties
    geometry_type: Optional[str] = None
    compression_ratio: Optional[float] = None


class ModelDocumentation(BaseModel):
    """Complete documentation for the model."""
    model_name: str
    model_size: str
    architecture: Dict[str, Any]

    # Analysis results
    features: List[FeatureDocumentation] = Field(default_factory=list)
    circuits: List[CircuitDocumentation] = Field(default_factory=list)
    layers: List[LayerDocumentation] = Field(default_factory=list)

    # Summary statistics
    total_features_discovered: int = 0
    total_circuits_discovered: int = 0
    transparency_score: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "0.1.0"
