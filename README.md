# TranspOLMo: First-Principles Neural Network Interpretability

**Automated system for fully characterizing the internal workings of transformer language models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TranspOLMo is a comprehensive interpretability framework that uses first-principles analysis to systematically extract and document the geometric, algebraic, and information-theoretic structures that emerge from training transformer models. Instead of treating neural networks as black boxes, it automates the discovery and documentation of mathematical constraints that govern their behavior.

**Starting Model**: OLMo2-1B (fully open: weights, training data, code, checkpoints)
**Goal**: Complete transparency—every feature, circuit, and computation documented and verified

## Key Features

- **Geometric Analysis**: Manifold structure discovery, intrinsic dimensionality estimation, local geometry characterization
- **Feature Extraction**: Sparse Autoencoders (SAE) for discovering monosemantic features
- **Circuit Discovery**: Automated identification of computational circuits through attention pattern analysis
- **Systematic Documentation**: Structured, queryable documentation of all findings
- **Verification System**: Intervention-based validation of discovered features and circuits

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TranspOLMo-1B.git
cd TranspOLMo-1B

# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Basic Usage

Run the complete analysis pipeline:

```bash
# Uses defaults from src/config.py
python scripts/run_full_analysis.py

# Or override specific settings
python scripts/run_full_analysis.py --model allenai/OLMo-2-0425-1B --num-samples 1000
```

**Configuration**: All defaults are in `src/config.py`. Command-line args override when provided. See [Configuration Guide](docs/configuration.md) for details.

Options:
- `--model`: Model name or path (default: from config.py)
- `--num-samples`: Number of samples for analysis (default: from config.py)
- `--device`: Device to use, cuda or cpu (default: from config.py)
- `--skip-sae`: Skip SAE training for faster analysis
- `--layers`: Comma-separated layer indices to analyze (e.g., "0,6,11")

### Quick Example

```python
from src.models.loader import OLMo2Loader
from src.models.hooks import ActivationCapture
from src.analysis.geometry.manifold import ManifoldAnalyzer

# Load model
loader = OLMo2Loader("allenai/OLMo-2-0425-1B", cache_dir="./data/models")
model, tokenizer = loader.load()

# Capture activations
with ActivationCapture(model) as capturer:
    capturer.register_hooks(["model.layers.6.mlp"])

    # Run inference
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)

    # Get activations
    activations = capturer.get_activations()

# Analyze geometry
analyzer = ManifoldAnalyzer(activations["model.layers.6.mlp"])
results = analyzer.analyze_all()

print(f"Intrinsic dimension: {results['intrinsic_dimension']['intrinsic_dim_95pct_var']}")
print(f"Geometry type: {results['local_geometry']['geometry_type']}")
```

## Architecture

```
TranspOLMo/
├── src/
│   ├── models/          # Model loading and instrumentation
│   ├── extraction/      # Activation capture and dataset building
│   ├── analysis/        # Core analysis modules
│   │   ├── geometry/    # Manifold analysis, clustering
│   │   ├── features/    # Sparse Autoencoders, feature analysis
│   │   ├── circuits/    # Circuit discovery
│   │   └── constraints/ # Architectural constraints
│   ├── documentation/   # Documentation generation
│   ├── verification/    # Intervention and testing
│   └── visualization/   # Plotting and interactive tools
├── scripts/             # Execution scripts
├── notebooks/           # Jupyter notebooks for exploration
└── results/             # Analysis outputs
```

## Analysis Pipeline

### Phase 1: Model Loading
- Load OLMo2 model and tokenizer
- Extract architectural constraints
- Inspect layer structure

### Phase 2: Activation Extraction
- Build diverse analysis dataset
- Hook target layers
- Capture activations during forward passes

### Phase 3: Geometric Analysis
- Estimate intrinsic dimensionality (PCA, MLE)
- Compute principal directions
- Analyze local geometry and curvature
- Perform clustering to discover feature groups

### Phase 4: Feature Extraction
- Train Sparse Autoencoders on layer activations
- Extract monosemantic features
- Characterize features with max-activating examples
- Infer semantic domains

### Phase 5: Circuit Discovery
- Trace attention patterns across layers
- Build circuit connectivity graphs
- Identify important computational paths
- Classify attention head types

### Phase 6: Documentation
- Generate structured documentation (JSON, Markdown)
- Calculate transparency score
- Create queryable feature catalog
- Build circuit library

### Phase 7: Verification
- Test features via ablation
- Validate circuits through intervention
- Measure impact on model outputs

## Output Files

After running the full analysis:

```
results/
├── geometric_analysis.json       # Geometric properties per layer
├── sae_layer_*.pt                # Trained SAE models
└── features_catalog.json         # Feature characterizations

docs/findings/
├── model_documentation.json      # Complete structured documentation
├── model_documentation.md        # Human-readable report
└── summary.json                  # Quick summary statistics
```

## Key Concepts

### Transparency Score

A metric (0-1) measuring how well we understand the model:
- **Feature Coverage**: Fraction of neurons/heads characterized
- **Circuit Coverage**: Number of documented computational circuits
- **Layer Understanding**: How well each layer's function is understood

Target: >80% for "transparent" model

### Monosemantic Features

Features that respond to single, interpretable concepts rather than superposed mixtures. Extracted using Sparse Autoencoders with L1 regularization.

### Computational Circuits

Directed graphs showing information flow through attention heads and MLPs that implement specific algorithms (e.g., subject-verb agreement, entity tracking).

## Advanced Usage

### Analyzing Specific Layers

```bash
python scripts/run_full_analysis.py --layers "0,6,11" --num-samples 500
```

### Training Custom SAEs

```python
from src.analysis.features.sparse_autoencoder import SAETrainer

trainer = SAETrainer(
    input_dim=1024,
    hidden_dim=8192,  # 8x expansion
    l1_coefficient=0.001,
    learning_rate=1e-4
)

history = trainer.train(dataloader, num_epochs=10)
trainer.save_model("my_sae.pt")
```

### Circuit Discovery

```python
from src.analysis.circuits.discovery import CircuitDiscovery

discoverer = CircuitDiscovery(model, tokenizer)

# Analyze attention patterns
patterns = discoverer.trace_attention_patterns("The cat sat on the mat")

# Discover circuit structure
circuit = discoverer.discover_circuit_structure("The capital of France is")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/
```

## Methodology

TranspOLMo is based on the insight that **neural network behavior emerges from mathematical constraints**:

1. **Geometric constraints**: Features live in high-dimensional vector spaces
2. **Algebraic constraints**: Linear transformations and non-linear activations
3. **Information-theoretic constraints**: Optimal compression of training distribution
4. **Architectural constraints**: SwiGLU, attention, layer normalization
5. **Optimization constraints**: Gradient descent local optima

These constraints are **discoverable through systematic analysis**—we don't need human intuition, just rigorous mathematical characterization.

## Scaling to Larger Models

The methodology scales linearly with model size:

- **OLMo2-1B**: ~1-2 hours on single GPU
- **OLMo2-7B**: ~5-10 hours
- **OLMo2-13B**: ~10-15 hours
- **OLMo2-32B**: ~25-35 hours

## Citation

If you use TranspOLMo in your research, please cite:

```bibtex
@software{transpolmo2024,
  title={TranspOLMo: First-Principles Neural Network Interpretability},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TranspOLMo-1B}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- **OLMo2 Team** at Allen Institute for AI for the open model
- **Anthropic** for Sparse Autoencoder methodology
- **David Bau** and collaborators for activation capture techniques
- **Chris Olah** for mechanistic interpretability foundations

## Roadmap

- [ ] Support for more model architectures (Llama, Mistral, GPT)
- [ ] Interactive visualization dashboard
- [ ] Distributed training for larger models
- [ ] Automated hypothesis generation
- [ ] Integration with causal tracing methods
- [ ] Real-time interpretability during training

## Contact

For questions and discussions:
- GitHub Issues: [https://github.com/yourusername/TranspOLMo-1B/issues](https://github.com/yourusername/TranspOLMo-1B/issues)
- Email: your.email@example.com

---

**Built with the belief that AI systems should be fully understandable, not black boxes.**
