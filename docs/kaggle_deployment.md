# Kaggle Deployment Guide

## Overview

TranspOLMo is pre-configured for Kaggle notebooks with free GPU access. Kaggle provides:
- ✅ Free GPU T4 (16GB VRAM) - 30 hours/week
- ✅ No credit card required
- ✅ Pre-installed ML libraries
- ✅ Easy notebook sharing and version control

## Quick Start

### Method 1: Use Pre-built Notebook (Recommended)

1. **Upload Notebook**:
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Click "File" → "Upload Notebook"
   - Upload `notebooks/kaggle_starter.ipynb` from this repo

2. **Enable GPU**:
   - Click "⋮" (three dots) in top right
   - Settings → Accelerator → **GPU T4 x2**
   - Click "Save"

3. **Run All Cells**:
   - Click "Run All" or run cells one by one
   - First run will take ~5 min to clone and install
   - Analysis runs ~30-60 min depending on settings

### Method 2: Run Helper Script

In any Kaggle notebook:

```python
# Clone and run with one command
!git clone https://github.com/BarzinL/TranspOLMo2-1B.git
%cd TranspOLMo2-1B
!pip install -q -e .[all]
!bash scripts/kaggle_run.sh
```

### Method 3: Manual Setup

```python
# 1. Clone repository
!git clone https://github.com/BarzinL/TranspOLMo2-1B.git
%cd TranspOLMo2-1B

# 2. Install dependencies
!pip install -q -e .[all]

# 3. Run analysis with Kaggle config
!python scripts/run_full_analysis.py --config configs/kaggle.yaml

# 4. View results
import json
with open('/kaggle/working/results/transparency_scores.json') as f:
    scores = json.load(f)
    print(json.dumps(scores, indent=2))
```

## Kaggle-Specific Configuration

The `configs/kaggle.yaml` file is optimized for Kaggle:

```yaml
compute:
  device: "cuda"
  dtype: "float16"      # Memory efficient
  batch_size: 32        # Balanced for T4

paths:
  cache_dir: "/kaggle/working/cache"       # Kaggle working dir
  output_dir: "/kaggle/working/results"    # Results location
  hf_cache: "/kaggle/working/hf_cache"     # HuggingFace cache

analysis:
  num_samples: 10000
  layers: [0, 6, 11]    # Representative layers
  skip_sae: true        # Faster first run
```

## Saving Results

### During Notebook Session

Results are automatically saved to `/kaggle/working/results/`:
```
/kaggle/working/results/
├── transparency_scores.json      # Quick summary
├── activations_layer_*.pt        # Saved activations
├── geometric_analysis.json       # Layer geometry
├── model_documentation.json      # Full documentation
└── checkpoints/                  # Resume points
```

### Persisting Results

Kaggle's `/kaggle/working/` directory is cleared after each session. To save results:

**Option 1: Save Version (Recommended)**
- Click "Save Version" button (top right)
- Choose "Save & Run All" or "Quick Save"
- Results are preserved in notebook output
- Download from "Output" tab after run completes

**Option 2: Manual Download**
```python
# Create a ZIP of results
!cd /kaggle/working && zip -r results.zip results/
```
Then download `results.zip` from the Output tab.

**Option 3: Upload to Kaggle Dataset**
```python
# Save as Kaggle dataset for reuse
!kaggle datasets version -p /kaggle/working/results \
  -m "TranspOLMo analysis results" \
  --dir-mode zip
```

## Troubleshooting

### GPU Not Detected

**Symptoms:**
```
Warning: CUDA requested but not available. Using CPU.
```

**Fix:**
1. Click ⋮ → Settings
2. Accelerator → **GPU T4 x2** (not "None" or "TPU")
3. Click "Save"
4. Restart notebook kernel

### Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix (automatic):**
The script automatically handles OOM by reducing batch size. But you can also:

```bash
# Use smaller batch size from start
!python scripts/run_full_analysis.py \
  --config configs/kaggle.yaml \
  --num-samples 5000
```

### Session Timeout

**Symptoms:**
Notebook stops after 12 hours

**Fix:**
- The checkpoint system saves progress
- Just re-run the notebook
- It will resume from last completed layer

### Dependencies Not Found

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
```

**Fix:**
```python
# Make sure you're in the right directory
%cd /kaggle/working/TranspOLMo2-1B

# Reinstall if needed
!pip install -q -e .[all]
```

## Advanced Usage

### Custom Analysis

Override any setting:

```bash
!python scripts/run_full_analysis.py \
  --config configs/kaggle.yaml \
  --model allenai/OLMo-2-0425-1B \
  --num-samples 20000 \
  --layers "0,3,6,9,11" \
  --dtype float16
```

### Multiple Runs

Analyze different configurations:

```python
for layers in ["0,6,11", "1,7,10", "2,8,9"]:
    !python scripts/run_full_analysis.py \
      --config configs/kaggle.yaml \
      --layers {layers} \
      --output-dir /kaggle/working/results_{layers.replace(',', '_')}
```

### Memory Monitoring

Check GPU usage during run:

```python
# In a separate cell, run periodically
!nvidia-smi
```

## Performance Expectations

On Kaggle T4 GPU (16GB):

| Configuration | Time | Memory Usage |
|--------------|------|--------------|
| Quick test (1000 samples, 3 layers) | ~10 min | ~4GB |
| Default (10000 samples, 3 layers, skip SAE) | ~30-45 min | ~6-8GB |
| Full (10000 samples, 12 layers, skip SAE) | ~2-3 hours | ~6-10GB |
| With SAE (10000 samples, 3 layers) | ~4-6 hours | ~8-12GB |

**Note**: First run takes extra ~5 min to download model (~2.4GB).

## Best Practices

1. **Start small**: Run `minimal_test.py` first to verify setup
2. **Use checkpoints**: Don't worry about timeouts, checkpoints save progress
3. **Save versions**: Use Kaggle's "Save Version" to preserve outputs
4. **Monitor memory**: Check `nvidia-smi` if you modify settings
5. **Skip SAE initially**: Use `skip_sae: true` for faster first runs

## Example Workflows

### Quick Validation
```python
# Fast test to verify everything works
!python scripts/minimal_test.py
```

### Standard Analysis
```python
# Good balance of speed and completeness
!python scripts/run_full_analysis.py \
  --config configs/kaggle.yaml \
  --num-samples 10000 \
  --layers "0,6,11" \
  --skip-sae
```

### Comprehensive Analysis
```python
# Full analysis (takes several hours)
!python scripts/run_full_analysis.py \
  --config configs/kaggle.yaml \
  --num-samples 10000
```

## Kaggle Limitations

- **GPU Time**: 30 hours/week free tier (plenty for this project)
- **Session Length**: 12 hours max, but checkpointing handles this
- **Storage**: `/kaggle/working` is temporary, use "Save Version"
- **Internet**: Download limited to ~5GB/hour (model is ~2.4GB)

## Getting Help

If you encounter issues on Kaggle:

1. Check the [Memory Optimization Guide](memory_optimization.md)
2. Review error messages in notebook output
3. Open an issue at: https://github.com/BarzinL/TranspOLMo2-1B/issues
4. Include: Kaggle notebook URL, error message, configuration used

## Next Steps

After running on Kaggle:

1. **Download results**: Save Version → Download from Output tab
2. **Explore locally**: Import results into `notebooks/01_quick_start.ipynb`
3. **Visualize**: Use the documentation to understand findings
4. **Iterate**: Try different layers, samples, or models

---

**Kaggle makes it easy to run TranspOLMo without local GPU resources!**
