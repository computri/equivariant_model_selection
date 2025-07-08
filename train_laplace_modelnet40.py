import torch
import hydra
from omegaconf import DictConfig

from common.models import ModelNet40Model
from laplace_utils import setup_laplace
from utils import (
    load_data_modelnet40,
    build_trainer,
    setup_wandb_logger,
    create_output_dirs,
    setup_environment,
)

def train_laplace(args):

    setup_environment(args.system.seed)

    create_output_dirs(base="results/modelnet40")

    model_id = args.model.model_id
    # Load data
    train_loader, test_loader, cal_loader = load_data_modelnet40(args.data)

    # wandb object
    wandb_logger = setup_wandb_logger(args, "")
        
    # get trainer and model
    trainer = build_trainer(args, wandb_logger)

    model_results = {
        "model": f"rapidash_{model_id}"
    }

    model = ModelNet40Model.load_from_checkpoint(f"./checkpoints/modelnet40/pretrained_models/{model_id}/checkpoints/last.ckpt")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_results["num_params"] = num_trainable


    model.hparams.test_augm = False

    # Usual test
    test_acc = trainer.test(model, test_loader)[0]["test_acc"]
    train_acc = trainer.test(model, train_loader)[0]["test_acc"]

    if args.system.gpus > 0:
        model = model.to(torch.device("cuda"))

    model.eval()

    # --- Set up Laplace ---
    la = setup_laplace(
        model=model, likelihood="classification"
    )

    la.fit(train_loader)

    
    torch.save(la.state_dict(), f"./checkpoints/modelnet40/mllapprox/{args.model_id}_state_dict.bin")


@hydra.main(
    config_path=str("./configs/modelnet40/"), 
    config_name="default"
)
def main(cfg: DictConfig) -> None:
    train_laplace(cfg)


if __name__ == "__main__":
    main()
