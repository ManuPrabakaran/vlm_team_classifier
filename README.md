# VLM Team Classifier

A multi-stage cascade classifier for basketball team identification, replacing K-Means jersey color clustering with Vision-Language Models. Built as a prototype for Paloa Labs' sports analytics pipeline.

## Architecture

```
K-Means (< 1ms) → SigLIP (~15ms) → Qwen2-VL (~100ms) → Manual Review
```

Each stage is progressively more expensive but more accurate. The cascade short-circuits as soon as confidence exceeds the threshold for the current stage, keeping average latency low.

## Results

| Clip | K-Means Baseline | SigLIP | CLIP | Qwen2-VL 2B |
|------|-----------------|--------|------|-------------|
| clip1_easy (Celtics vs Heat) | 87.2% | **97.8%** | 84.8% | 54.3% |
| clip2_hard (Spurs vs Grizzlies) | 52.8% | **86.0%** | 70.0% | 58.0% |
| clip3_edge (Cavs vs Knicks) | 86.9% | **71.9%** | 78.1% | 56.2% |

SigLIP dominates across all clips. Full analysis in [docs/RESULTS.md](docs/RESULTS.md) and [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

## Project Structure

```
├── src/
│   ├── baseline.py          # K-Means TeamClustering (provided baseline)
│   ├── classifier.py        # VLMTeamClassifier cascade implementation
│   ├── config.py            # Thresholds, weights, model identifiers
│   └── utils.py             # Model loading, embedding extraction, helpers
├── tests/
│   ├── test_baseline.py     # Unit tests for K-Means baseline
│   ├── test_classifier.py   # Unit tests for cascade classifier
│   └── test_integration.py  # Integration tests with synthetic data
├── notebooks/
│   ├── exploration.ipynb    # Research notebook (run on Colab with GPU)
│   └── data_collection.ipynb # Ground truth labeling tool
├── docs/
│   ├── PROPOSAL.md          # System design proposal
│   ├── RESULTS.md           # Experimental results
│   └── PRODUCTION_PLAN.md   # Production deployment plan
├── data/                    # Ground truth JSON files
├── results/                 # Metrics and evaluation outputs
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU-accelerated inference (required for SigLIP and Qwen2-VL):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

```python
from src.classifier import VLMTeamClassifier

clf = VLMTeamClassifier()

# Fit on initial frame with player bounding boxes
clf.fit(frame, bboxes)

# Classify a single player
result = clf.predict(frame, bbox)
# {'team_id': 0, 'confidence': 0.92, 'method': 'kmeans', 'signals': {...}}

# Batch classify all players in a frame
results = clf.predict_batch(frame, bboxes)

# Optional: set game context for Qwen2-VL prompting
clf.set_game_context(
    team_names={0: "Celtics", 1: "Heat"},
    jersey_descriptions={0: "white jersey with green trim", 1: "black jersey with red accents"},
)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Skip integration tests that need ground truth data
pytest tests/ -v -k "not test_kmeans_on_synthetic and not test_cascade_on_synthetic"
```

## Notebook

The research notebook (`notebooks/exploration.ipynb`) is designed to run on Google Colab with a T4 GPU. It contains:
- K-Means baseline evaluation and stability analysis
- VLM model comparison (SigLIP, CLIP, Florence-2, Qwen2-VL)
- Description generation pipeline using GPT-4o
- Cascade architecture analysis and recommendations

**Note**: Florence-2 and Qwen2-VL require different `transformers` versions. The notebook uses a `RUN_MODEL` toggle in Cell 2 — set to `"florence"` or `"qwen"` and restart the runtime to switch.
