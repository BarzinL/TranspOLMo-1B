#!/bin/bash
# Kaggle execution helper
# Simplest way to clone and run this repo to Kaggle with always up-to-date code

# Setup
git clone --depth 1 https://github.com/BarzinL/TranspOLMo2-1B.git
cd TranspOLMo2-1B
pip install -q -e .[all]

# Run analysis with Kaggle config
# You can override any setting by passing additional arguments
python scripts/run_full_analysis.py \
    --config configs/kaggle.yaml \
    "$@"  # Pass through any additional arguments to override config
