import json
import pickle

import torch

import hydra
from omegaconf import DictConfig

from common.models import ModelNet40Model
from laplace_utils import (
    setup_laplace,
    extract_laplace_stats,
    tune_prior_precision,
)
from utils import (
    load_data_modelnet40,
    build_trainer,
    setup_wandb_logger,
    setup_environment,
    create_output_dirs,
)

def eval(args):

    # Set random seed
    setup_environment(args.system.seed)

    create_output_dirs(base="results/modelnet40")

    # Load data
    train_loader, test_loader, cal_loader = load_data_modelnet40(args.data)

    model_id = args.model.model_id

    # wandb object
    wandb_logger = setup_wandb_logger(args, "")
    
    # get trainer and model
    trainer = build_trainer(args, wandb_logger)

    model_results = {
        "model": f"{model_id}"
    }
    model = ModelNet40Model.load_from_checkpoint(f"./checkpoints/modelnet40/pretrained_models/{model_id}/checkpoints/last.ckpt")

    model.hparams.test_augm = False

    # Usual test
    test_acc = trainer.test(model, test_loader)[0]["test_acc"]
    train_acc = trainer.test(model, train_loader)[0]["test_acc"]
    
    
    if args.system.gpus > 0:
        model = model.to(torch.device("cuda"))
    
    model.eval()


    # --- Set up Laplace ---
    la = setup_laplace(
        model=model, 
        ckpt_path=f"./checkpoints/modelnet40/mll_approx/{model_id}_state_dict.bin", 
        likelihood="classification"
    )

    train_data_size = len(train_loader.dataset)

    # --- Pre-tuning stats ---
    pre_stats = extract_laplace_stats(la, train_data_size)
    
    print("Tuning prior_prec for la")

    # --- Grid search over prior precision ---
    print("Tuning prior_prec for la")
    prior_precs, margliks, opt_prior_prec = tune_prior_precision(la, train_data_size)

    gridsearch_results = {
        "prior_precs": prior_precs.tolist(),
        "margliks": margliks,
    }


    # --- Post-tuning stats ---
    post_stats = extract_laplace_stats(la, train_data_size)


    # --- Save final results ---
    model_results = {
        "acc": {
            "test": test_acc,
            "train": train_acc,
        },
        **{f"pre_{k}": v.item() for k, v in pre_stats.items()},
        **{f"post_{k}": v.item() for k, v in post_stats.items()},
    }


    with open(f'results/modelnet40/hyperparam/{model_id}.pkl', 'wb') as f:
        pickle.dump(gridsearch_results, f)

    with open(f"results/modelnet40/mll/{model_id}.json", "w") as f:
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