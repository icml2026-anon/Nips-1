# CAST: Causal-Aware Sparse Time-series Framework

This repository contains the official implementation of **CAST**.

CAST is a causal learning framework that reframes structural sparsity from a modeling obstacle into an inductive bias for time series classification under extreme zero-inflation. It integrates three components:

- **Stage I — Zero-Inflated Structural Equation Model (ZI-SEM):** Recovers temporal causal dependencies from both active observations and dormant states.
- **Stage II — Hyperbolic Causal Embedding:** Transforms the discovered causal graph into geometry-aware positional encodings in hyperbolic space.
- **Stage III — Causal-Gated KAN Sequence Model:** Adaptive causal gates modulate Kolmogorov–Arnold Network (KAN) basis functions, emphasizing causally active positions while suppressing redundant connections.

## Project Structure

```
CAST/
├── cast/                        # Core package
│   ├── config.py                # CASTConfig dataclass
│   ├── train.py                 # CASTTrainer (Stage I + Stage II + Stage III)
│   ├── models/
│   │   ├── cast_model.py        # CASTModel (main model)
│   │   ├── cast_kan.py          # Causal-gated KAN sequence model
│   │   ├── kan_layers.py        # KAN basis variants (Taylor, B-Spline, Fourier, etc.)
│   │   ├── causal_discovery.py  # ZI-SEM-based temporal causal graph learning
│   │   ├── hyperbolic.py        # Poincaré ball / hyperboloid embeddings
│   │   └── baselines.py         # 12 baseline models (statistical + neural)
│   ├── data/
│   │   └── dataset.py           # Dataset loading & preprocessing
│   └── utils/
│       ├── focal_loss.py        # Focal loss with label smoothing
│       ├── metrics.py           # Binary classification metrics
│       └── seed.py              # Reproducibility utilities
├── configs/                     # Per-dataset YAML hyperparameter configs
├── run.py                       # Main entry point (train / benchmark / ablation)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

CAST is evaluated on five retail and e-commerce benchmarks. Place raw data under `./data/`:

```
data/
├── tabular_datasets/
│   ├── retail.csv
│   ├── CDNOW_master.txt
│   ├── ta_feng.csv
│   └── ...
└── instacart/
    ├── orders.csv
    ├── order_products__prior.csv
    └── ...
```

| Dataset | Samples | Pos. Ratio | Sparsity | Primary Task | Auxiliary Task |
|---------|---------|------------|----------|--------------|----------------|
| Retail | 4,338 | 35.0% | 76.9% | Value Classification | Churn Prediction |
| CDNOW | 23,502 | 35.2% | 84.5% | Value Classification | Churn Prediction |
| Instacart | 102,589 | 35.0% | 47.3% | Activity Classification | Churn Prediction |
| Sales-Weekly | 811 | 35.0% | 28.2% | Risk Classification | Seasonality Detection |
| Ta-Feng | 32,266 | 35.0% | 82.8% | Risk Classification | Repurchase Detection |

All datasets follow a **70:15:15** split for training, validation, and testing.

## Usage

### Single Dataset

```bash
python run.py --dataset retail --seed 42
python run.py --dataset retail --task churn --seed 42
python run.py --dataset cdnow --data_root /path/to/data --seed 42
```

### Full Benchmark (CAST + All Baselines)

```bash
python run.py --benchmark --data_root ./data
python run.py --run_baselines --run_all --data_root ./data
```

### Ablation Study

```bash
python run.py --ablation --run_all --data_root ./data
```

### KAN Basis Function Selection

```bash
# Available: taylor (default), bspline, fourier, chebykan, jacobikan, rbfkan, waveletkan, legendrekan
python run.py --dataset retail --kan_type fourier --seed 42
```

## Hyperparameters

Default configuration is defined in `configs/default.yaml`. Dataset-specific overrides are in `configs/<dataset>.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 32 | Causal positional encoding dimension |
| kan_hidden_dim | 64 | KAN layer hidden dimension |
| kan_num_layers | 3 | Number of stacked CAST-KAN layers |
| kan_order | 8 | Basis function order |
| kan_type | taylor | KAN basis type |
| khop | 2 | k-hop neighbors for causal graph |
| pretrain_epochs | 100 | Stage I: causal discovery + hyperbolic embedding |
| finetune_epochs | 100 | Stage III: end-to-end fine-tuning (500 for sales_weekly) |
| learning_rate | 0.001 | Learning rate |
| batch_size | 128 | Batch size |
| dropout | 0.1 | Dropout rate |
| weight_decay | 0.0001 | Weight decay |
| dag_threshold | 0.1 | Edge pruning threshold for causal DAG |
| sparsity_lambda | 1.0 | Sparsity penalty for DAG learning |
| focal_gamma | 1.0 | Focal loss gamma |
| focal_alpha | 0.7 | Focal loss alpha |

## Baseline Models

12 baselines are included in `cast/models/baselines.py`:

- **Statistical:** Croston, SBA, ZIP, Hurdle
- **Neural:** DSN, SoftShape, TimeMIL, InterpGN
- **State-Space / Sequential:** UniTS, Mamba, MambaSL, GRU-D

## License

This project is released under the MIT License.
