import torch
from netcal.metrics import ECE
import torch.distributions as dists
from typing import Dict, Any, Union
from torch.utils.data import DataLoader


def evaluate_model_regression(
    la: Any, 
    dataloader: DataLoader, 
    device: str = 'cuda'
) -> Dict[str, float]:
    all_mu, all_var, all_y = [], [], []

    for x in dataloader:
        x = x.to(device)
        y = x.y
        mu, var = la(x)
        
        all_mu.append(mu.detach())
        all_var.append(var.detach())
        all_y.append(y.detach())

    mu = torch.cat(all_mu)
    var = torch.cat(all_var).squeeze(-1)
    y_true = torch.cat(all_y)

    sigma = var.sqrt()

    ll = dists.Normal(mu, sigma).log_prob(y_true)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((mu - y_true) ** 2)).item()
    

    return {
        "avg_LL": ll.mean().item(),
        "sum_LL": ll.sum().item(),
        "RMSE": rmse,
        "Avg_Pred_Var": var.mean().item()
    }


def evaluate_model_classification(
    model: Any, 
    dataloader: DataLoader, 
    device: str = 'cuda'
) -> Dict[str, Union[float, int]]:
    all_pred,  all_y = [], []

    for x in dataloader:
        x = x.to(device)
        y = x.y
        pred = torch.softmax(model(x.cuda()), dim=-1)
        
        all_pred.append(pred.detach())
        all_y.append(y.detach())

    pred = torch.cat(all_pred)
    y_true = torch.cat(all_y)
    
    y_one_hot = torch.nn.functional.one_hot(y_true, num_classes=40)
    
    brier_score = torch.nn.functional.mse_loss(pred, y_one_hot)    
    
    acc_map = (pred.argmax(-1) == y_true).float().mean()
    ece_map = ECE(bins=15).measure(pred.cpu().numpy(), y_true.cpu().numpy())
    ll = dists.Categorical(pred).log_prob(y_true).sum()
    
    return {
        "predictive_ll": ll.item(),
        "ECE": ece_map,
        "brier": brier_score.item()
    }
