#!/bin/bash
# Kaggle execution helper
# Simplest way to clone and run this repo to Kaggle with always up-to-date code

# Setup
git clone --depth 1 https://github.com/BarzinL/TranspOLMo2-1B.git
cd TranspOLMo2-1B
pip install -q -e .[all]

# Run analysis with sensible defaults
python scripts/run_full_analysis.py \
    --num-samples 10000 \
    --dtype float16 \
    --layers "0,6,11" \
    --skip-sae \
    --output-dir /kaggle/working/results \
    "$@"  # Pass through any additional arguments
