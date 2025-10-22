# TranspOLMo Architecture

## System Overview

TranspOLMo is organized into distinct modules that work together to provide comprehensive interpretability analysis.

## Module Hierarchy

### Core Modules

1. **src/models/** - Model loading and instrumentation
   - `loader.py`: Load OLMo2 models and extract architecture info
   - `hooks.py`: Activation capture using forward hooks

2. **src/extraction/** - Data extraction pipeline
   - `dataset_builder.py`: Build analysis datasets from various sources

3. **src/analysis/** - Core analysis algorithms
   - **geometry/**: Manifold analysis, clustering
     - `manifold.py`: Intrinsic dimensionality, PCA, local geometry
     - `clustering.py`: K-means, HDBSCAN clustering
   - **features/**: Feature extraction and analysis
     - `sparse_autoencoder.py`: SAE implementation and training
     - `feature_analysis.py`: Feature characterization
   - **circuits/**: Circuit discovery
     - `discovery.py`: Attention pattern analysis, circuit graphs

4. **src/documentation/** - Documentation generation
   - `schema.py`: Pydantic schemas for structured documentation
   - `generator.py`: Generate JSON/Markdown documentation

5. **src/verification/** - Testing and validation
   - `intervention.py`: Ablation studies, feature testing

## Data Flow

```
Input Text
    ↓
[Model Loading] → Architecture Info
    ↓
[Activation Capture] → Raw Activations
    ↓
[Geometric Analysis] → Manifold Properties
    ↓
[SAE Training] → Sparse Features
    ↓
[Feature Analysis] → Feature Characterizations
    ↓
[Circuit Discovery] → Computational Circuits
    ↓
[Documentation] → Structured Docs (JSON/MD)
```

## Key Design Decisions

### 1. Modular Architecture
Each analysis type is isolated in its own module, allowing:
- Independent testing
- Easy extension
- Parallel development

### 2. Progressive Analysis
Analysis proceeds from coarse to fine:
- Layer-level geometry
- Feature-level characteristics
- Circuit-level algorithms

### 3. Lazy Evaluation
Large intermediate results are computed on-demand and cached to disk.

### 4. Device Management
Activations are moved to CPU immediately to manage GPU memory efficiently.

## Extension Points

### Adding New Analysis Methods

1. Create new module in `src/analysis/`
2. Implement core analysis class
3. Add to orchestration pipeline in `scripts/run_full_analysis.py`
4. Update documentation schemas if needed

### Supporting New Models

1. Extend `OLMo2Loader` or create new loader
2. Update layer naming patterns in `hooks.py`
3. Adjust architecture constraints

### Custom Visualizations

1. Add visualization code to `src/visualization/`
2. Integrate with Jupyter notebooks
3. Update documentation generator to include visualizations
