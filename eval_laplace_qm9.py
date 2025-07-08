import json
import pickle

import torch
import hydra
from omegaconf import DictConfig

from common.models import QM9Model
from constants import TARGET_INDICES
from laplace_utils import (
    setup_laplace,
    tune_prior_precision,
    extract_laplace_stats,
)
from uncertainty_utils import evaluate_model_regression
from utils import (
    load_data_qm9,
    scale_mae_if_energy,
    build_trainer,
    setup_wandb_logger,
    create_output_dirs,
    setup_environment,
)

def eval(args: DictConfig) -> None:

    # Set random seed
    setup_environment(args.system.seed)

    create_output_dirs(base="results/qm9")

    model_id = args.model.model_id

    global_results = {
        "model": f"{model_id}"
    }

    global_gridsearch_results = {}

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
        
        model.incorporate_normalisation()
        
        train_mae = trainer.test(model, train_loader)[0]["test MAE"]
        test_mae = trainer.test(model, test_loader)[0]["test MAE"]
        
        #  Report meV instead of eV:
        train_mae = scale_mae_if_energy(target_idx, train_mae)
        test_mae = scale_mae_if_energy(target_idx, test_mae)


        if args.system.gpus > 0:
            model = model.to(torch.device("cuda"))

        model.eval()

        model.incorporate_normalisation()

        # --- Set up Laplace ---
        la = setup_laplace(
            model=model, 
            ckpt_path=f"checkpoints/qm9/mll_approx/{model_id}_{target}_state_dict.bin", 
            likelihood="regression"
        )

        # --- Pre-tuning stats ---
        train_data_size = len(train_loader.dataset)
        pre_stats = extract_laplace_stats(la, train_data_size)


        # --- Grid search over prior precision ---
        print("Tuning prior_prec for la")
        prior_precs, margliks, opt_prior_prec = tune_prior_precision(la, train_data_size)

        global_gridsearch_results[target] = {
            "prior_precs": prior_precs.tolist(),
            "margliks": margliks,
        }

        # --- Post-tuning stats ---
        post_stats = extract_laplace_stats(la, train_data_size)

        # --- Evaluate on test set ---
        with torch.no_grad():
            test_results = evaluate_model_regression(la, test_loader)

        # --- Save final results ---
        global_results[target] = {
            "mae": {
                "test": test_mae,
                "train": train_mae,
            },
            **{f"pre_{k}": v.item() for k, v in pre_stats.items()},
            **{f"post_{k}": v.item() for k, v in post_stats.items()},
            **test_results,
        }


    with open(f'results/qm9/hyperparam/{model_id}.pkl', 'wb') as f:
        pickle.dump(global_gridsearch_results, f)

    with open(f"results/qm9/mll/{model_id}.json", "w") as f:
        json.dump(global_results, f, indent=4)


@hydra.main(
    config_path="./configs/qm9/", 
    config_name="default",
    version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    eval(cfg)

if __name__ == "__main__":
    main()