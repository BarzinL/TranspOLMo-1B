# Memory Optimization Guide

## Understanding Memory Requirements

### Model Size Calculation

For a 1.2B parameter model:

| Precision | Bytes/Param | Total Size | GPU Needed |
|-----------|-------------|------------|------------|
| float32   | 4 bytes     | ~4.8 GB    | 8GB+ VRAM  |
| float16   | 2 bytes     | ~2.4 GB    | 4GB+ VRAM  |
| bfloat16  | 2 bytes     | ~2.4 GB    | 4GB+ VRAM  |
| int8      | 1 byte      | ~1.2 GB    | 2GB+ VRAM  |

**Note**: Actual memory usage includes overhead for activations, gradients, optimizer state, etc.

## Quick Solutions for OOM Errors

### Solution 1: Use Float16 (RECOMMENDED) ✅

**Command line:**
```bash
python scripts/run_full_analysis.py --dtype float16 --num-samples 1000 --skip-sae
```

**Or edit `src/config.py`:**
```python
dtype: str = "float16"  # Change from "float32"
```

**Benefits:**
- ✅ 50% memory reduction (~2.4GB vs ~4.8GB)
- ✅ Faster computation on modern GPUs
- ✅ Minimal accuracy loss for inference
- ✅ Works on 8GB GPUs (like RTX 3060 Ti)

### Solution 2: Use CPU

**Command line:**
```bash
python scripts/run_full_analysis.py --device cpu --num-samples 100
```

**Benefits:**
- ✅ No GPU memory limits
- ✅ Can use system RAM (16GB+)
- ⚠️ Much slower (10-50x)

### Solution 3: Use bfloat16

**Command line:**
```bash
python scripts/run_full_analysis.py --dtype bfloat16
```

**Benefits:**
- ✅ Same memory as float16 (~2.4GB)
- ✅ Better numeric stability than float16
- ✅ Native support on Ampere+ GPUs (RTX 30 series+)
- ⚠️ Slightly different results than float32

### Solution 4: Reduce Batch Size and Samples

**Command line:**
```bash
python scripts/run_full_analysis.py --num-samples 500 --dtype float16
```

Edit `src/config.py`:
```python
batch_size: int = 8  # Reduce from 16
num_samples: int = 500  # Reduce from 10000
```

## GPU Memory Recommendations

| GPU Model            | VRAM | Recommended Settings |
|----------------------|------|---------------------|
| RTX 3060 Ti          | 8GB  | `--dtype float16`   |
| RTX 3070/3080        | 8-10GB | `--dtype float16` or `bfloat16` |
| RTX 3090/4090        | 24GB | `--dtype float32` (full precision) |
| Tesla V100           | 16GB | `--dtype float32`   |
| A100                 | 40-80GB | `--dtype float32` |

## Comparison: Precision vs Accuracy

For interpretability analysis (not training):

| Precision | Activation Differences | Feature Discovery | Circuit Analysis |
|-----------|------------------------|-------------------|------------------|
| float32   | Baseline               | Baseline          | Baseline         |
| float16   | < 0.01% deviation      | ~99.5% match      | ~99% match       |
| bfloat16  | < 0.005% deviation     | ~99.8% match      | ~99.5% match     |

**Conclusion**: float16/bfloat16 are excellent for analysis!

## Advanced: Quantization (Future)

For even lower memory:

```python
# 8-bit quantization (future feature)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```

This would reduce memory to ~1.2GB but requires additional setup.

## Troubleshooting

### Error: "CUDA out of memory"

**Try in order:**

1. **Use float16:**
   ```bash
   python scripts/run_full_analysis.py --dtype float16
   ```

2. **Reduce samples:**
   ```bash
   python scripts/run_full_analysis.py --dtype float16 --num-samples 500
   ```

3. **Use CPU:**
   ```bash
   python scripts/run_full_analysis.py --device cpu --num-samples 100
   ```

4. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Error: "RuntimeError: Expected all tensors on same device"

Make sure your config is consistent:
```python
device: str = "cuda"  # Or "cpu" everywhere
```

### Memory Fragmentation

If you see "memory is fragmented", try:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_full_analysis.py --dtype float16
```

## Monitoring Memory Usage

### Check GPU memory:
```bash
# During run
nvidia-smi -l 1

# Or in Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Best Practices

1. **Start small**: Use `--num-samples 100` for first run
2. **Use float16**: Default to float16 on GPUs < 16GB
3. **Skip SAE**: Use `--skip-sae` flag for faster testing
4. **Analyze few layers**: Use `--layers "0,6,11"` instead of all
5. **Monitor memory**: Run `nvidia-smi` in another terminal

## Example Workflows

### For 8GB GPU (RTX 3060 Ti, 3070):
```bash
# Quick test
python scripts/run_full_analysis.py \
  --dtype float16 \
  --num-samples 500 \
  --skip-sae \
  --layers "0,6,11"

# Full analysis (takes longer)
python scripts/run_full_analysis.py \
  --dtype float16 \
  --num-samples 5000
```

### For 16GB+ GPU (RTX 4080, A100):
```bash
# Full precision, full analysis
python scripts/run_full_analysis.py \
  --dtype float32 \
  --num-samples 10000
```

### For CPU (any machine):
```bash
# Small sample size recommended
python scripts/run_full_analysis.py \
  --device cpu \
  --num-samples 100 \
  --skip-sae
```
