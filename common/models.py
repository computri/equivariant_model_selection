import os
import argparse

import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl


from common.rapidash import Rapidash

from utils import CosineWarmupScheduler, RandomSOd, TimerCallback

from torchcp.regression.score import ABS
# from torchcp.regression.predictor import SplitPredictor
# from cp_utils import PygSplitPredictor
from types import MethodType
import json
from laplace import Laplace, marglik_training
# Performance optimizations
torch.set_float32_matmul_precision('medium')


class QM9Model(pl.LightningModule):
    """Lightning module for QM9 molecular property prediction."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels = (
            11 +  # base atom features
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            1 * ("coords" in self.hparams.vector_features)    # position as vector
        )

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=1,  # Single target property prediction
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            output_dim_vec=0,
            dim=3,
            num_ori=self.hparams.orientations,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='graph',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )
        
        # Initialize normalization parameters
        self.shift = 0.
        self.scale = 1.
        
        self.normalisation = False
        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

    def incorporate_normalisation(self):
        if not self.normalisation:
            with torch.no_grad():
                self.net.readout.weight.data = self.net.readout.weight.data * self.scale
                self.net.readout.bias.data = self.net.readout.bias.data * self.scale + self.shift
            self.normalisation = True
        
    def remove_normalisation(self):
        if self.normalisation:
            with torch.no_grad():
                self.net.readout.weight.data = self.net.readout.weight.data / self.scale
                self.net.readout.bias.data = (self.net.readout.bias.data - self.shift) / self.scale
            self.normalisation = False


    def forward(self, graph):
        # Prepare input features
        x = []
        vec = []
        
        # Add base atomic features
        x.append(graph.x)
        
        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(graph.pos)
            
        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(graph.pos[:,None,:])
            
        # Combine features
        x = torch.cat(x, dim=-1) if x else torch.ones(graph.pos.size(0), 1).type_as(graph.pos)
        vec = torch.cat(vec, dim=1) if vec else None
            
        if graph.batch is None:
            graph.batch = torch.zeros(x.shape[0], device=graph.x.device).long()
        # Forward pass
        pred, _ = self.net(x, graph.pos, graph.edge_index, graph.batch, vec=vec)

        
        return pred
        
    def pre_activation_forward(self, graph):
        # Prepare input features
        x = []
        vec = []
        
        # Add base atomic features
        x.append(graph.x)
        
        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(graph.pos)
            
        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(graph.pos[:,None,:])
            
        # Combine features
        x = torch.cat(x, dim=-1) if x else torch.ones(graph.pos.size(0), 1).type_as(graph.pos)
        vec = torch.cat(vec, dim=1) if vec else None
            
        if graph.batch is None:
            graph.batch = torch.zeros(x.shape[0], device=graph.x.device).long()
        # Forward pass
        pred, _ = self.net.pre_activation_forward(x, graph.pos, graph.edge_index, graph.batch, vec=vec)

        return pred
    
    def set_dataset_statistics(self, dataloader):
        """Compute mean and standard deviation of target property."""
        print('Computing dataset statistics...')
        ys = []
        for data in dataloader:
            ys.append(data.y)
        ys = np.concatenate(ys)
        self.shift = np.mean(ys)
        self.scale = np.std(ys)
        print(f'Target statistics - Mean: {self.shift:.4f}, Std: {self.scale:.4f}')

    def training_step(self, graph, batch_idx):
        # Apply rotation augmentation if enabled
        if self.hparams.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            
        # Forward pass and loss computation
        pred = self(graph)
        loss = torch.mean(torch.abs(pred - (graph.y - self.shift) / self.scale))
        self.train_metric(pred * self.scale + self.shift, graph.y)
        return loss

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        # self.valid_metric(pred * self.scale + self.shift, graph.y)
        self.valid_metric(pred, graph.y)
    
    def on_test_epoch_start(self):
        self.incorporate_normalisation()

    def test_step(self, graph, batch_idx):
        pred = self(graph)
        # self.test_metric(pred * self.scale + self.shift, graph.y)
        self.test_metric(pred, graph.y)

    def on_train_epoch_end(self):
        self.log("train MAE", self.train_metric, prog_bar=True)

    def on_validation_epoch_start(self):
        self.incorporate_normalisation()

    def on_validation_epoch_end(self):
        self.remove_normalisation()
        self.log("valid MAE", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        self.remove_normalisation()
        self.log("test MAE", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay and learning rate schedule."""
        # Separate parameters into decay and no-decay groups
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias') or pn.endswith('layer_scale'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate parameter grouping
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not grouped!"

        # Create optimizer and scheduler
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    


class ModelNet40Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels = (
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            3 * ("normals" in self.hparams.scalar_features) +  # normal components as scalars
            1 * ("coords" in self.hparams.vector_features) +  # position as vector
            1 * ("normals" in self.hparams.vector_features) +  # normal as vector
            3 * ("pose" in self.hparams.vector_features)      # pose matrix (3 vectors)
        )

        # Ensure at least one input channel if none are specified
        if in_channels == 0:
            in_channels = 1  # will use constant ones as input

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=40,  # ModelNet40 has 40 classes
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            output_dim_vec=0,
            dim=3,
            num_ori=self.hparams.orientations,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='graph',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )
        
        # Setup metrics
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)

    def forward(self, data):
        # Apply rotation augmentation if enabled (during training)
        # print(self.training, self.hparams.train_augm, self.hparams.test_augm)
        if (self.training and self.hparams.train_augm) or (not self.training and self.hparams.test_augm):
            
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
            if hasattr(data, 'normal'):
                data.normal = torch.einsum('ij,bj->bi', rot, data.normal)
        else:
            rot = torch.eye(3, device=data.pos.device)

        # Prepare input features
        x = []  # scalar features
        vec = []  # vector features

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "normals" in self.hparams.scalar_features and hasattr(data, 'normal'):
            x.append(data.normal)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "normals" in self.hparams.vector_features and hasattr(data, 'normal'):
            vec.append(data.normal[:,None,:])
        if "pose" in self.hparams.vector_features:
            vec.append(rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1))

        # Combine features
        if not x and not vec:  # Only add constant ones if both x and vec are empty
            x = torch.ones(data.pos.size(0), 1).type_as(data.pos)
        else:
            x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None

        # Forward pass
        if data.batch is None:
            data.batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
        pred, _ = self.net(x, data.pos, data.edge_index, data.batch, vec=vec)
        return pred

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = torch.nn.functional.cross_entropy(pred, data.y)
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        self.valid_metric(pred, data.y)

    def test_step(self, data, batch_idx):
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        suffix = "_rotated" if self.hparams.test_augm else ""
        self.log(f"test_acc{suffix}", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}        
