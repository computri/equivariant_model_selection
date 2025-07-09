## Overview

This is the public code repository for our work
[On Equivariant Model Selection through the Lens of Uncertainty](https://arxiv.org/abs/2506.18629) presented at the [8th Workshop on Tractable Probabilistic Modeling at UAI 2025](https://tractable-probabilistic-modeling.github.io/tpm2025/).


#### 📝 Abstract 
---

Equivariant models leverage prior knowledge on symmetries to improve predictive performance, but misspecified architectural constraints can harm it instead. While work has explored learning or relaxing constraints, selecting among pretrained models with varying symmetry biases remains challenging. We examine this model selection task from an uncertainty-aware perspective, comparing frequentist (via Conformal Prediction), Bayesian (via the marginal likelihood), and calibration-based measures to naive error-based evaluation. We find that uncertainty metrics generally align with predictive performance, but Bayesian model evidence does so inconsistently. We attribute this to a mismatch in Bayesian and geometric notions of model complexity, and discuss possible remedies. Our findings point towards the potential of uncertainty in guiding symmetry-aware model selection.

---

##  Acknowledgments

This repository builds heavily on the following projects:

- [**Rapidash**](https://github.com/Sharvaree/EquivarianceStudy):  
  The project structure and many core components (e.g. data pipelines, training utilities) are adapted from this repository.  
  Licensed under the [MIT License](https://github.com/Sharvaree/EquivarianceStudy/blob/main/LICENSE).

- [**TorchCP**](https://github.com/ml-stat-sustech/torchcp):  
  Conformal prediction logic and CP predictors are adapted or reused from this library.  
  Licensed under the [LGPL-3.0 License](https://www.gnu.org/licenses/lgpl-3.0.html).

- [**laplace-torch**](https://github.com/aleximmer/laplace): Last-layer laplace marginal likelihood 
  approximations are acquired using this library.
  Licensed under the [MIT License](https://github.com/aleximmer/Laplace/blob/main/LICENSE.txt).


## Repo structure
This repository extends [Rapidash](https://github.com/Sharvaree/EquivarianceStudy) with conformal prediction functionality using components adapted from [TorchCP](https://github.com/ml-stat-Sustech/TorchCP). Below is an overview of the core folders and files:

<pre>
equivariant_model_selection/
├── checkpoints/                 # Pretrained model checkpoints and Laplace approximations
│   ├── modelnet40/              # ModelNet40 classification models
│   │   ├── pretrained_models/   # Base models with different symmetry constraints
│   │   └── mll_approx/          # Marginal likelihood approximations
│   └── qm9/                     # QM9 regression models
│       ├── pretrained_models/   # Base models with different symmetry constraints
│       └── mll_approx/          # Marginal likelihood approximations
│
├── configs/                     # Hydra-compatible YAML configuration files
│   ├── modelnet40/              # ModelNet40 experiment configurations
│   └── qm9/                     # QM9 experiment configurations
│
├── data/                        # Dataset storage and preprocessing
│   ├── modelnet40/              # ModelNet40 point cloud data
│   └── QM9/                     # QM9 molecular property data
│
├── common/                      # Core model implementations and utilities
│   ├── models.py                # Neural network architectures
│   ├── rapidash.py              # Main Rapidash framework components
│   └── rapidash_*.py            # Specialized modules for invariants, spherical grids, etc.
│
├── torchcp/                     # Conformal prediction framework (adapted from TorchCP)
│   ├── classification/          # Classification CP predictors
│   ├── regression/              # Regression CP predictors
│   ├── graph/                   # Graph-specific CP methods
│   └── utils/                   # CP utility functions
│
├── results/                     # Experimental results and analysis
│   ├── modelnet40/              # ModelNet40 experiment outputs
│   └── qm9/                     # QM9 experiment outputs
│
├── plots/                       # Generated figures and visualizations
│
├── train_laplace_*.py           # Training scripts for Laplace approximations
├── eval_laplace_*.py            # Evaluation scripts for Bayesian model selection
├── eval_frequentist_*.py        # Evaluation scripts for frequentist uncertainty
├── uncertainty_utils.py         # Uncertainty quantification utilities
├── laplace_utils.py             # Laplace approximation helpers
├── cp_utils.py                  # Conformal prediction utilities
└── utils.py                     # General utility functions
</pre>

##  Setup

1. Clone the repository
```
git clone https://github.com/computri/equivariant_model_selection.git
cd equivariant_model_selection
```

2. Create and activate the Conda environment
```
conda env create -f environment.yml
conda activate equivariant_model_selection
```

3. Download model checkpoints

We provide pretrained model checkpoints [**here**](https://drive.google.com/drive/folders/1hNBR3KuQmZRsyGZINonPELuASRX1YSyU?usp=sharing), and recommend placing them in the `./checkpoints` directory

⸻


## Usage

This repository provides tools for uncertainty-aware model selection across different symmetry-constrained architectures. The workflow consists of three main stages:

### 1. **Training Laplace Approximations** (Optional)
Pretrained model checkpoints and their corresponding last-layer Laplace approximations (LLLA) are provided. However, you can train new LLLA approximators on top of existing pretrained checkpoints:

```bash
# Train Laplace approximation for ModelNet40
python train_laplace_modelnet40.py model=...

# Train Laplace approximation for QM9
python train_laplace_qm9.py model=...
```

**Available model architectures:**
- `invariant` - Rotation-invariant models
- `equivariant` - Rotation-equivariant models  
- `augment` - Data augmentation-based models
- `plain` - Standard models without symmetry constraints

### 2. **Bayesian Model Selection**
Evaluate models using marginal likelihood approximations via Laplace method:

```bash
python eval_laplace_modelnet40.py model=...
python eval_laplace_qm9.py model=...
```

**Available model architectures:**
- `invariant` - Rotation-invariant models
- `equivariant` - Rotation-equivariant models  
- `augment` - Data augmentation-based models
- `plain` - Standard models without symmetry constraints

### 3. **Frequentist Uncertainty Evaluation**
Assess model performance using conformal prediction and calibration metrics:

```bash
python eval_frequentist_modelnet40.py model=...
python eval_frequentist_qm9.py model=...
```

**Available model architectures:**
- `invariant` - Rotation-invariant models
- `equivariant` - Rotation-equivariant models  
- `augment` - Data augmentation-based models
- `plain` - Standard models without symmetry constraints

### Output Structure
Results are automatically saved to:
- `results/modelnet40/` or `results/qm9/` - Numerical results and metrics
- `outputs/` - Hydra experiment logs organized by timestamp
- `plots/` - Generated visualizations and figures

### Reproducing Paper Figures
The `plots/` directory contains scripts to recreate the figures from the paper. These plotting scripts depend on the results generated by the evaluation scripts, so you must run the corresponding evaluation commands first, along with all the models:

```bash
python eval_laplace_modelnet40.py model=...
python eval_frequentist_modelnet40.py model=...
python eval_laplace_qm9.py model=...
python eval_frequentist_qm9.py model=...
```

The plotting scripts will automatically load the saved results from `results/modelnet40/` and `results/qm9/` to recreate the uncertainty comparison figures, model selection visualizations, and performance analyses presented in the paper.


⸻

#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
