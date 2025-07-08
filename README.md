## Overview

This is the public code repository for our work
[On Equivariant Model Selection through the Lens of Uncertainty](https://arxiv.org/abs/2506.18629) presented at the [8th Workshop on Tractable Probabilistic Modeling at UAI 2025](https://tractable-probabilistic-modeling.github.io/tpm2025/).


#### 📝 Abstract 
---

Equivariant models leverage prior knowledge on symmetries to improve predictive performance, but misspecified architectural constraints can harm it instead. While work has explored learning or relaxing constraints, selecting among pretrained models with varying symmetry biases remains challenging. We examine this model selection task from an uncertainty-aware perspective, comparing frequentist (via Conformal Prediction), Bayesian (via the marginal likelihood), and calibration-based measures to naive error-based evaluation. We find that uncertainty metrics generally align with predictive performance, but Bayesian model evidence does so inconsistently. We attribute this to a mismatch in Bayesian and geometric notions of model complexity, and discuss possible remedies. Our findings point towards the potential of uncertainty in guiding symmetry-aware model selection.

---

## 📌 Acknowledgments

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
will follow soon...


</pre>

## 🔧 Setup

1. Clone the repository
```
git clone https://github.com/computri/equivariant_model_selection.git
cd equivariant_model_selection
```

more will follow soon ...

4. Download model checkpoints

We provide pretrained model checkpoints, which will follow soon...

⸻


## 🚀 Usage

tba...

---


### QM9
tba...

### ModelNet40
tba...

⸻

#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
