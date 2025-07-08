import json
import os

import numpy as np
import torch

import hydra
from omegaconf import DictConfig

from cp_utils import PygClassificationSplitPredictor
from torchcp.classification.score import THR

from common.models import ModelNet40Model
from uncertainty_utils import evaluate_model_classification
from utils import (
    load_data_modelnet40,
    build_trainer,
    setup_wandb_logger,
    setup_environment,
)

def eval(args):
    # Set random seed
    setup_environment(args.system.seed)
    
    os.makedirs("./results/modelnet40/frequentist", exist_ok=True)   
    # Load data
    train_loader, test_loader, val_loader = load_data_modelnet40(args.data)

    model_id = args.model.model_id

    # wandb object
    wandb_logger = setup_wandb_logger(args, "")

    # get trainer and model
    trainer = build_trainer(args, wandb_logger)

    model_results = {
        "model": f"{model_id}"
    }

    model = ModelNet40Model.load_from_checkpoint(f"./checkpoints/modelnet40/pretrained_models/{model_id}/checkpoints/last.ckpt")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_results["num_params"] = num_trainable
    model.hparams.test_augm = False

    # Usual test
    test_acc = trainer.test(model, test_loader)[0]["test_acc"]
    train_acc = trainer.test(model, train_loader)[0]["test_acc"]

    model.eval()
    model = model.to(torch.device("cuda"))
    

    # --- Get conformal metrics ---
    predictor = PygClassificationSplitPredictor(score_function=THR(), model=model)
    predictor.calibrate(val_loader, args.cp.alpha)
    result_dict = predictor.evaluate(test_loader)
    

    coverages, widths = predictor.run_trials(val_loader, test_loader, n_trials=args.cp.n_trials, alpha=args.cp.alpha, cal_ratio=0.5)
    model_results.update({
        "cov_mean": np.mean(coverages),
        "cov_std": np.std(coverages),
        "width_mean": np.mean(widths),
        "width_std": np.std(widths)
    })
    torch.cuda.empty_cache()

    # --- Get ll, Brier, ECE ---
    evaluation_results = evaluate_model_classification(model, test_loader, device='cuda')

    model_results.update(
        {
            "train_acc": train_acc,
            "test_acc": test_acc,
            **evaluation_results,
        }
    )

    with open(f"./results/modelnet40/frequentist/{model_id}.json", "w") as f:
        json.dump(model_results, f, indent=4)


@hydra.main(
    config_path="./configs/modelnet40/", 
    config_name="default",
    version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    eval(cfg)

if __name__ == "__main__":
    main()