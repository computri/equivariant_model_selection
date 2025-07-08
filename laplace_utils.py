from laplace import Laplace
from typing import Dict, Tuple, Optional
import torch
import numpy as np

def setup_laplace(model: torch.nn.Module, likelihood: str, ckpt_path: Optional[str] = None) -> Laplace:
    la = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="last_layer",
        hessian_structure="full",
        last_layer_name="net.readout",
    )

    if ckpt_path is not None:
        la.load_state_dict(torch.load(ckpt_path))
    return la


def tune_prior_precision(
    la: Laplace,
    train_data_size: int,
    log_bounds: Tuple[float, float] = (-4.0, 4.0),
    step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    lower, upper = log_bounds
    prior_precs = 10.0 ** np.arange(lower, upper + step, step=step, dtype=np.float32)
    margliks = []
    for prior_prec in prior_precs:
        marglik = la.log_marginal_likelihood(torch.tensor(prior_prec)) / train_data_size
        print(f"prior_prec {prior_prec:.1e}  marglik {marglik:.5f}")
        margliks.append(marglik.item())
    opt_prior_prec = prior_precs[np.nanargmax(margliks)]
    la.prior_precision = torch.tensor(opt_prior_prec)
    return prior_precs, np.array(margliks), opt_prior_prec


def extract_laplace_stats(la: Laplace, train_data_size: int) -> Dict[str, float]:
    return {
        "log_likelihood": la.log_likelihood,  # optionally divide by train_data_size
        "log_marglik": la.log_marginal_likelihood(),  # optionally divide
        "scatter": la.scatter,
        "log_det_ratio": la.log_det_ratio,
        "log_det_prior_precision": la.log_det_prior_precision,
        "log_det_posterior_precision": la.log_det_posterior_precision,
        "prior_prec": la.prior_precision,
        "complexity": 0.5 * (la.log_det_ratio + la.scatter),
    }


