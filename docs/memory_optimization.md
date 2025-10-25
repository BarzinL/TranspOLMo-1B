# Memory Optimization Guide

## New: Automatic Memory Management

**TranspOLMo now includes automatic memory management!** You no longer need to manually adjust settings in most cases.

### Automatic Features (v2.0+):

- ✅ **One-layer-at-a-time processing**: Never loads all layers simultaneously
- ✅ **Automatic OOM handling**: Catches out-of-memory errors and reduces batch size  
- ✅ **Progress checkpointing**: Resume from crashes without losing work
- ✅ **Memory estimation**: Shows expected vs. available memory before starting
- ✅ **Aggressive cleanup**: `torch.cuda.empty_cache()` after each layer

These features work automatically - just run the script!

## Quick Start for Limited Memory

### 8GB GPU (RTX 3060 Ti, 3070)

```bash
python scripts/run_full_analysis.py \
  --dtype float16 \
  --layers "0,6,11" \
  --skip-sae
```

The script will:
1. Estimate memory requirements
2. Process layers one at a time
3. Automatically reduce batch size if OOM occurs
4. Save checkpoints after each layer

### 16GB GPU (Kaggle T4, RTX 4060 Ti)

```bash
python scripts/run_full_analysis.py --config configs/kaggle.yaml
```

Should work out of the box with automatic memory management.

## Understanding Memory Requirements

### Model Size Calculation

For a 1.2B parameter model:

| Precision | Bytes/Param | Total Size | GPU Needed |
|-----------|-------------|------------|------------|
| float32   | 4 bytes     | ~4.8 GB    | 8GB+ VRAM  |
| float16   | 2 bytes     | ~2.4 GB    | 4GB+ VRAM  |
| bfloat16  | 2 bytes     | ~2.4 GB    | 4GB+ VRAM  |

**Note**: Old versions processed all layers at once. **New version processes one layer at a time**, dramatically reducing memory needs.

### Memory Breakdown (New Version)

```
Total GPU Memory = Model + Single Layer Activations + Buffer

Example (float16, batch_size=32):
= 2.4GB (model) + 1-2GB (one layer) + 0.5GB (buffer)
= ~4-5GB total (fits easily in 8GB!)
```

**Old version** would need ~10-15GB for same analysis.

## How Automatic Memory Management Works

### 1. Memory Estimation

Before starting, the script estimates memory needs:

```
--- Memory Estimation ---
Model parameters: 1.17B
Estimated peak memory: 4.2 GB
Available GPU memory: 15.7 GB
Current GPU memory allocated: 2.3 GB
```

If estimated memory > 90% of available, you'll get a warning.

### 2. One-Layer-at-a-Time Processing

```
Processing Layer 0/11
============================================================
GPU memory before layer: 2.3 GB
  Attempting with batch_size=32...
  ✓ Captured activations: torch.Size([320, 512, 2048])
  💾 Saved to: /kaggle/working/results/activations_layer_0.pt
  ✓ Checkpoint saved
GPU memory after cleanup: 2.3 GB
```

Each layer:
- Is processed independently
- Saved to disk immediately
- GPU memory cleared before next layer
- Checkpoint created for resume

### 3. Automatic OOM Handling

If out-of-memory occurs:

```
Processing Layer 6/11
  Attempting with batch_size=32...
  ✗ OOM with batch_size=32
  Retrying with reduced batch_size=16...
  ✓ Captured activations: torch.Size([320, 512, 2048])
  Batch size reduced to 16, will use for remaining layers
```

The script automatically:
- Catches the OOM error
- Reduces batch size by half (32→16→8→4)
- Retries with smaller batch
- Uses reduced batch for remaining layers

### 4. Checkpointing & Resume

If the script crashes or times out:

```bash
# Just re-run the same command
python scripts/run_full_analysis.py --config configs/kaggle.yaml

# Output:
📁 Found checkpoint: 2 layers already completed
   Resuming from layer 2

✓ Layer 0 already completed, loading from disk...
  Loaded: torch.Size([320, 512, 2048])
✓ Layer 6 already completed, loading from disk...
  Loaded: torch.Size([320, 512, 2048])

Processing Layer 11/11  # Continues from where it left off
```

Checkpoints saved to: `{output_dir}/checkpoints/extraction_progress.json`

## Manual Optimizations (If Needed)

### Solution 1: Use Float16 ✅

**Recommended for all GPUs < 24GB**

```bash
python scripts/run_full_analysis.py --dtype float16
```

**Benefits:**
- 50% memory reduction (~2.4GB vs ~4.8GB)
- Faster computation on modern GPUs
- Minimal accuracy loss for inference
- **Works automatically with memory management**

### Solution 2: Reduce Samples

```bash
python scripts/run_full_analysis.py \
  --dtype float16 \
  --num-samples 5000 \
  --layers "0,6,11"
```

Fewer samples = faster processing, but still useful analysis.

### Solution 3: Use CPU (Last Resort)

```bash
python scripts/run_full_analysis.py \
  --device cpu \
  --num-samples 100 \
  --skip-sae
```

**Benefits:**
- No GPU memory limits
- Can use system RAM

**Drawbacks:**
- 10-50x slower
- Only practical for tiny analyses

## GPU Memory Recommendations

| GPU Model | VRAM | Recommended Settings | Expected Performance |
|-----------|------|---------------------|---------------------|
| RTX 3060 Ti | 8GB  | `--dtype float16` | ✅ Works well |
| RTX 3070/3080 | 8-10GB | `--dtype float16` | ✅ Works great |
| RTX 4060 Ti | 16GB | `--config configs/kaggle.yaml` | ✅ Works perfectly |
| RTX 3090/4090 | 24GB | Default settings | ✅ No issues |
| Kaggle T4 | 16GB | `--config configs/kaggle.yaml` | ✅ Optimized |
| A100 | 40-80GB | Default settings | ✅ Very fast |

**All configurations now include automatic memory management!**

## Troubleshooting

### Error: "CUDA out of memory" (Immediately)

**Cause**: Model itself doesn't fit in GPU

**Fix:**
```bash
# Use float16 to halve model size
python scripts/run_full_analysis.py --dtype float16
```

### Error: "OOM even with minimum batch size"

**Cause**: GPU is too small even with batch_size=4

**Fix:**
```bash
# Use CPU instead
python scripts/run_full_analysis.py --device cpu --num-samples 100
```

Or get access to larger GPU (Kaggle provides free 16GB T4).

### Script Crashes Mid-Analysis

**Fix:**
Just re-run the same command. Checkpointing will resume automatically:

```bash
# Re-run exact same command
python scripts/run_full_analysis.py --config configs/kaggle.yaml

# It will automatically skip completed layers and resume
```

### Want to Start Fresh

**Fix:**
Delete checkpoint file:

```bash
rm -rf /path/to/output_dir/checkpoints/
python scripts/run_full_analysis.py ...
```

## Monitoring Memory Usage

### During Script Run

The script shows memory usage automatically:

```
GPU memory before layer: 2.3 GB
...
GPU memory after cleanup: 2.3 GB
```

### Manual Monitoring

In a separate terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or in Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Example Workflows

### For 8GB GPU (Minimum)
```bash
# Quick test first
python scripts/minimal_test.py

# Then full analysis
python scripts/run_full_analysis.py \
  --dtype float16 \
  --num-samples 5000 \
  --layers "0,6,11" \
  --skip-sae
```

### For 16GB GPU (Kaggle, RTX 4060 Ti)
```bash
# Use Kaggle config (optimized)
python scripts/run_full_analysis.py --config configs/kaggle.yaml

# Or customize
python scripts/run_full_analysis.py \
  --dtype float16 \
  --num-samples 10000 \
  --layers "0,6,11"
```

### For 24GB+ GPU (RTX 3090, 4090, A100)
```bash
# Full precision, full analysis
python scripts/run_full_analysis.py

# Or faster with float16
python scripts/run_full_analysis.py --dtype float16
```

## Best Practices

1. **Start with minimal_test.py**: Verify setup works
2. **Use float16**: Always use on GPUs < 24GB
3. **Let auto-handling work**: Don't manually set batch size unless needed
4. **Trust checkpoints**: Don't worry about crashes
5. **Monitor first run**: Watch memory usage to understand your GPU limits

## Comparison: Old vs New

| Feature | Old Version | New Version |
|---------|-------------|-------------|
| Layer processing | All at once | One at a time ✅ |
| Memory usage | ~10-15GB | ~4-5GB ✅ |
| OOM handling | Crash | Auto-retry ✅ |
| Resume capability | None | Automatic ✅ |
| Memory estimation | None | Shown before start ✅ |
| Min GPU for analysis | 16GB | 8GB ✅ |

**Result**: Analysis that required 16GB GPU now works on 8GB!

## Advanced: Disable Auto Features

If you want manual control:

```python
# Edit scripts/run_full_analysis.py
# Comment out automatic memory management sections
```

But this is **not recommended** - the automatic system handles edge cases better than manual tuning.

---

**With automatic memory management, TranspOLMo works out-of-the-box on most GPUs!**
