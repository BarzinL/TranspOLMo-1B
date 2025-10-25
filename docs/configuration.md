# Configuration Guide

## Philosophy: Single Source of Truth

TranspOLMo uses `src/config.py` as the **single source of truth** for all default configuration values. YAML files and command-line arguments only **override** these defaults when explicitly provided.

## Configuration Hierarchy

Configuration values are applied in order of priority (highest last):

1. **`src/config.py`** - Default values (lowest priority)
2. **YAML config file** - Optional config file via `--config`
3. **Command-line arguments** - Override everything when explicitly provided (highest priority)

**Example**: If `config.py` has `dtype: "float32"`, YAML has `dtype: "float16"`, and CLI has `--dtype bfloat16`, then **bfloat16** is used.

## Using YAML Config Files

### Creating a YAML Config

Create a YAML file to override specific settings:

```yaml
# my_config.yaml
compute:
  device: "cuda"
  dtype: "float16"
  batch_size: 32

paths:
  cache_dir: "/path/to/cache"
  output_dir: "/path/to/results"

analysis:
  num_samples: 5000
  layers: [0, 6, 11]
  skip_sae: true
```

### Using the Config File

```bash
python scripts/run_full_analysis.py --config my_config.yaml
```

### Pre-configured Configs

**`configs/kaggle.yaml`** - Optimized for Kaggle notebooks:
```yaml
compute:
  device: "cuda"
  dtype: "float16"
  batch_size: 32

paths:
  cache_dir: "/kaggle/working/cache"
  output_dir: "/kaggle/working/results"
  hf_cache: "/kaggle/working/hf_cache"

analysis:
  num_samples: 10000
  layers: [0, 6, 11]
  skip_sae: true
```

## Configuration File Structure

### Available Sections

#### `compute:` - Computation settings
```yaml
compute:
  device: "cuda"        # or "cpu"
  dtype: "float16"      # or "float32", "bfloat16"
  batch_size: 32        # Batch size for processing
```

#### `paths:` - File system paths
```yaml
paths:
  cache_dir: "./data/models"      # Model cache directory
  output_dir: "./results"          # Output directory
  hf_cache: "./hf_cache"          # HuggingFace cache (sets HF_HOME)
```

#### `analysis:` - Analysis parameters
```yaml
analysis:
  num_samples: 10000              # Number of samples to analyze
  layers: [0, 6, 11]              # Specific layers to analyze
  skip_sae: true                  # Skip SAE training
```

## Best Practices

1. **For permanent changes**: Edit `src/config.py`
2. **For environment configs**: Create YAML files (Kaggle, local, etc.)
3. **For experiments**: Use command-line arguments
4. **For automation**: Use YAML configs + CLI overrides
5. **Never duplicate**: Don't put the same setting in multiple places

## Why This Design?

**Benefits:**
- ✅ Single source of truth (`src/config.py`)
- ✅ Environment-specific configs without code changes
- ✅ Clear precedence order (config.py → YAML → CLI)
- ✅ Version control friendly

**Avoids:**
- ❌ Hardcoded values scattered across codebase
- ❌ Multiple conflicting defaults
- ❌ Editing code to change environment settings
