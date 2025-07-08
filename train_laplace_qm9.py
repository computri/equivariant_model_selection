import torch
import hydra
from omegaconf import DictConfig

from common.models import QM9Model
from constants import TARGET_INDICES
from laplace_utils import setup_laplace
from utils import (
    load_data_qm9,
    scale_mae_if_energy,
    build_trainer,
    setup_wandb_logger,
    create_output_dirs,
    setup_environment,
)


def train_laplace(args):

    setup_environment(args.system.seed)

    create_output_dirs(base="results/qm9")

    model_id = args.model.model_id
    
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

        # verify model performance
        train_mae = trainer.test(model, train_loader)[0]["test MAE"]
        test_mae = trainer.test(model, test_loader)[0]["test MAE"]

        # (Report meV instead of eV):
        train_mae = scale_mae_if_energy(train_mae, target_idx)
        test_mae = scale_mae_if_energy(test_mae, target_idx)

        # add the normalization parameters to the model
        model.incorporate_normalisation()

        if args.system.gpus > 0:
            model = model.to(torch.device("cuda"))

        model.eval()

        # --- Set up Laplace ---
        la = setup_laplace(
            model=model, likelihood="regression"
        )

        la.fit(train_loader)

        # Save model and Laplace instance
        torch.save(la.state_dict(), f"checkpoints/qm9/mll_approx/{model_id}_{target}_state_dict.bin")



@hydra.main(
    config_path=str("./configs/qm9/"), 
    config_name="default"
)
def main(cfg: DictConfig) -> None:
    train_laplace(cfg)


if __name__ == "__main__":
    main()
