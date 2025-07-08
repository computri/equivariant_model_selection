import json
import os

import numpy as np
import torch

import hydra
from omegaconf import DictConfig

from constants import TARGET_INDICES
from cp_utils import PygSplitPredictor
from torchcp.regression.score import ABS

from common.models import QM9Model
from utils import (
    load_data_qm9,
    scale_mae_if_energy,
    build_trainer,
    setup_wandb_logger,
    setup_environment,
)

def eval(args):
    # Set random seed
    setup_environment(args.system.seed)

    os.makedirs("./results/qm9/frequentist", exist_ok=True)   

    model_id = args.model.model_id
    
    global_results = {}
    for target in args.data.targets:
        
        target_idx = TARGET_INDICES[target]

        # Load data
        train_loader, val_loader, test_loader = load_data_qm9(args.data, target)
            
        # wandb object
        wandb_logger = setup_wandb_logger(args, "")
        
        # get trainer and model
        trainer = build_trainer(args, wandb_logger)

        model = QM9Model.load_from_checkpoint(f"checkpoints/qm9/pretrained_models/{model_id}_{target}/checkpoints/last.ckpt")
        model.set_dataset_statistics(train_loader)

        results = trainer.test(model, test_loader)
        mae = results[0]["test MAE"]

        #  Report meV instead of eV:
        mae = scale_mae_if_energy(target_idx, mae)

        if args.system.gpus > 0:
            model = model.to(torch.device("cuda"))

        model.eval()

        predictor = PygSplitPredictor(score_function=ABS(), model=model, target=target_idx)

        coverages, widths = predictor.run_trials(val_loader, test_loader, n_trials=args.cp.n_trials, alpha=args.cp.alpha, cal_ratio=0.5)
    
        global_results[target] = {
            "mae": {
                "all": None,
                "mean": mae,
                "std": None
            },
            "cov_mean": np.mean(coverages),
            "cov_std": np.std(coverages),
            "width_mean": np.mean(widths),
            "width_std": np.std(widths)
        }

    with open(f"results/qm9/frequentist/{model_id}.json", "w") as f:
        json.dump(global_results, f, indent=4)


@hydra.main(
    config_path=str("./configs/qm9/"), 
    config_name="default"
)
def main(cfg: DictConfig) -> None:
    eval(cfg)


if __name__ == "__main__":
    main()
