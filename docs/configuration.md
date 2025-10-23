# Configuration Guide

## Philosophy: Single Source of Truth

TranspOLMo uses `src/config.py` as the **single source of truth** for all default configuration values. Command-line arguments only **override** these defaults when explicitly provided.

## Configuration Hierarchy

1. **`src/config.py`** - Default values (edit this to change defaults)
2. **Command-line arguments** - Override defaults when provided
3. **Environment variables** - (future feature)

## How It Works

### Config File Structure

```python
# src/config.py

@dataclass
class ModelConfig:
    model_name: str = "allenai/OLMo-2-0425-1B"  # ← Default model
    cache_dir: Path = Path("./data/models")
    device: str = "cuda"
    dtype: str = "float32"

@dataclass
class ExtractionConfig:
    dataset_name: str = "allenai/dolma"
    num_samples: int = 10000                     # ← Default sample count
    max_seq_length: int = 512
    batch_size: int = 16
```

### Script Behavior

```python
# scripts/run_full_analysis.py

# 1. Load defaults from config.py
config = Config.default()

# 2. Override ONLY if argument provided
if args.model is not None:
    config.model.model_name = args.model  # ← Only overrides if --model was used
```

## Changing Defaults

### Option 1: Edit config.py (Recommended)

Edit `src/config.py` to change the default model:

```python
model_name: str = "allenai/OLMo-2-0425-1B"  # Your preferred default
```

Then run without arguments:
```bash
python scripts/run_full_analysis.py  # Uses config.py defaults
```

### Option 2: Override on Command Line

Keep `config.py` as-is, override for one run:

```bash
python scripts/run_full_analysis.py --model allenai/OLMo-7B
```

### Option 3: Create Custom Config (Future)

```python
from src.config import Config

custom_config = Config.from_dict({
    "model": {"model_name": "allenai/OLMo-7B"},
    "extraction": {"num_samples": 50000}
})
```

## Available Settings

### ModelConfig
- `model_name`: HuggingFace model identifier
- `cache_dir`: Where to cache downloaded models
- `device`: "cuda" or "cpu"
- `dtype`: "float32", "float16", or "bfloat16"

### ExtractionConfig
- `dataset_name`: Dataset to use for analysis
- `dataset_subset`: Specific subset (e.g., "cc_en_head")
- `num_samples`: Number of samples to analyze
- `max_seq_length`: Maximum sequence length
- `batch_size`: Batch size for processing
- `layers_to_capture`: Specific layers to analyze (None = all)

### AnalysisConfig
- `sae_hidden_size`: SAE expansion factor * input_dim
- `sae_l1_coefficient`: L1 sparsity penalty weight
- `sae_learning_rate`: Learning rate for SAE training
- `sae_batch_size`: Batch size for SAE training
- `sae_num_epochs`: Number of training epochs
- `pca_components`: Number of PCA components to compute
- `clustering_algorithm`: "kmeans", "hdbscan", etc.
- `num_clusters`: Number of clusters for feature grouping

### DocumentationConfig
- `output_dir`: Where to save documentation
- `database_path`: Path to SQLite database (future)
- `format`: "json" or "markdown"

## Command Line Arguments

All command-line arguments are **optional**. If not provided, config.py defaults are used.

```bash
python scripts/run_full_analysis.py \
  --model allenai/OLMo-2-0425-1B \    # Override model
  --num-samples 5000 \                 # Override sample count
  --device cuda \                      # Override device
  --skip-sae \                         # Skip SAE training
  --layers "0,6,11"                    # Analyze specific layers
```

## Best Practices

1. **For permanent changes**: Edit `src/config.py`
2. **For experiments**: Use command-line arguments
3. **For multiple configs**: Create separate config files and load with `Config.from_dict()`
4. **For automation**: Set config.py to your most common settings, override edge cases

## Example Workflows

### Daily Use (Same Model)
```python
# Edit src/config.py once:
model_name: str = "allenai/OLMo-2-0425-1B"

# Run without arguments:
python scripts/run_full_analysis.py
```

### Testing Different Models
```bash
# Keep config.py with your main model
# Override for experiments:
python scripts/run_full_analysis.py --model allenai/OLMo-7B --num-samples 1000
python scripts/run_full_analysis.py --model gpt2 --skip-sae
```

### Batch Processing
```bash
#!/bin/bash
for model in "allenai/OLMo-1B" "allenai/OLMo-7B"; do
    python scripts/run_full_analysis.py --model $model
done
```

## Why This Design?

**Benefits:**
- ✅ Single source of truth (no duplication)
- ✅ Easy to see all defaults in one place
- ✅ Command-line overrides are explicit and optional
- ✅ Version control friendly (config.py tracks changes)
- ✅ Self-documenting (config.py shows all options)

**Avoids:**
- ❌ Hardcoded values scattered across codebase
- ❌ Multiple conflicting defaults
- ❌ Confusion about which default takes precedence
