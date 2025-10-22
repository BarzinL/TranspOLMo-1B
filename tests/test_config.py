"""Tests for configuration management."""

import pytest
from pathlib import Path
from src.config import Config, ModelConfig, ExtractionConfig


def test_default_config():
    """Test default configuration creation."""
    config = Config.default()

    assert config.model.model_name == "allenai/OLMo-2-1124-1B"
    assert config.model.device in ["cuda", "cpu"]
    assert config.extraction.num_samples == 10000
    assert config.analysis.sae_hidden_size == 16384


def test_model_config():
    """Test model configuration."""
    model_config = ModelConfig(
        model_name="test-model",
        cache_dir=Path("/tmp/models"),
        device="cpu"
    )

    assert model_config.model_name == "test-model"
    assert model_config.device == "cpu"
    assert isinstance(model_config.cache_dir, Path)


def test_extraction_config():
    """Test extraction configuration."""
    extract_config = ExtractionConfig(
        num_samples=100,
        max_seq_length=256
    )

    assert extract_config.num_samples == 100
    assert extract_config.max_seq_length == 256


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "model": {"model_name": "custom-model", "device": "cpu"},
        "extraction": {"num_samples": 500}
    }

    config = Config.from_dict(config_dict)

    assert config.model.model_name == "custom-model"
    assert config.model.device == "cpu"
    assert config.extraction.num_samples == 500
