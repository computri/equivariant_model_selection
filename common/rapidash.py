# Adapted from https://github.com/Sharvaree/EquivarianceStudy/blob/main/models/rapidash.py
#  MIT License, (c) 2024  (c) 2024 Sharvaree Vadgama

from typing import List, Optional, Union

import torch
import torch.nn as nn

import torch_geometric.nn.pool as pool
from common.rapidash_invariants import Invariants
from common.rapidash_modules import (ConvNext, PolynomialCutoff, PolynomialFeatures,
                               SeparableFiberBundleConvNext)
from common.rapidash_spherical_grids import SphericalGridGenerator
from common.rapidash_utils import (fps_edge_index, fully_connected_edge_index,
                             knn_graph, radius_graph, scatter_add)


class Rapidash(nn.Module):
    """
    SE(3) equivariant convolutional network with optional U-Net structure.
    
    Features:
    - Multi-scale architecture with FPS-based down/up-sampling
    - Optional skip connections between scales
    - Support for both scalar and vector inputs/outputs
    - Orientation-aware features through spherical signal discretization (num_ori > 0)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, List[int]],
        output_dim: int,
        num_layers: Union[int, List[int]],
        edge_types: Union[str, List[str]] = ["fc"],
        equivariance: str = "SEn",
        ratios: List[float] = [],
        output_dim_vec: int = 0,
        dim: int = 3,
        num_ori: int = 12,
        degree: int = 2,
        widening_factor: int = 4,
        layer_scale: Optional[float] = None,
        task_level: str = "node",
        last_feature_conditioning: bool = True,
        skip_connections: bool = True,
        basis_dim: Optional[int] = None,
        basis_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.last_feature_conditioning = last_feature_conditioning
        
        # Initialize invariant computation and sphere grid
        self.num_ori = num_ori
        if num_ori > 0:
            self.grid_generator = SphericalGridGenerator(dim, num_ori)
            self.register_buffer('sphere_grid', self.grid_generator())
            ConvBlock = SeparableFiberBundleConvNext
        else:
            self.register_buffer('sphere_grid', None)
            ConvBlock = ConvNext
            
        self.compute_invariants = Invariants(dim, equivariance, num_ori)
        
        # Process architecture parameters
        self.layers_per_scale = [num_layers] if isinstance(num_layers, int) else num_layers
        num_scales = len(self.layers_per_scale)
        
        self.edge_types = self._parse_edge_types(
            num_scales * [edge_types] if isinstance(edge_types, str) else edge_types
        )
        self.ratios = num_scales * [ratios] if isinstance(ratios, float) else ratios + [1.0]
        self.hidden_dims = num_scales * [hidden_dim] if isinstance(hidden_dim, int) else hidden_dim
        
        assert len(self.layers_per_scale) == len(self.edge_types) == len(self.ratios)
        
        # Configure U-Net structure
        self.up_sample = not (task_level == "graph") and num_scales > 1
        if self.up_sample:
            self.num_total_layers = (sum(self.layers_per_scale) + num_scales - 1) * 2
        else:
            self.num_total_layers = sum(self.layers_per_scale) + num_scales - 1
            
        # Network components
        self.skip_connections = skip_connections
        self.global_pooling = task_level == "graph"
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        
        # Use basis_dim if provided, otherwise use hidden_dim
        if basis_dim is None:
            basis_dim = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[0]
        if basis_hidden_dim is None:
            basis_hidden_dim = basis_dim
        
        # Single global basis function
        self.spatial_basis_fn = self._create_basis_function(degree, basis_dim, basis_hidden_dim)
        if self.sphere_grid is not None:
            self.spherical_basis_fn = self._create_basis_function(degree, basis_dim, basis_hidden_dim)
        
        # Create network layers
        self.x_embedder = nn.Linear(
            input_dim + last_feature_conditioning, self.hidden_dims[0], bias=False
        )
        
        self.layers = self._build_network(
            ConvBlock, basis_dim, widening_factor, layer_scale
        )
        
        # Single readout layer
        final_dim = self.hidden_dims[-1] if not self.up_sample else self.hidden_dims[0]
        self.readout = nn.Linear(final_dim, output_dim + output_dim_vec)

    def _create_basis_function(self, degree: int, basis_dim: int, basis_hidden_dim: int):
        """Create polynomial basis function for kernel generation."""
        return nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(basis_hidden_dim),
            nn.GELU(),
            nn.Linear(basis_hidden_dim, basis_dim),
            nn.GELU(),
        )

    def _parse_edge_types(self, edge_types: List[str]):
        """
        Parse edge type specifications into (type, params) pairs.
        Now handles radius graphs with cutoff windowing.
        """
        parsed = []
        for edge_type in edge_types:
            if edge_type.lower() == "fc":
                parsed.append(("fc", {}))
                continue
                
            type_name, params = edge_type.lower().split("-")
            if type_name == "knn":
                parsed.append(("knn", {"k": int(params)}))
            elif type_name == "r":
                r = float(params)
                parsed.append(("radius", {"r": r, "cutoff": PolynomialCutoff(r)}))
            else:
                raise ValueError(f"Unsupported edge type: {type_name}")
        return parsed

    def _build_network(self, ConvBlock, basis_dim: int, widening_factor: int, 
                      layer_scale: Optional[float]):
        """Construct the network's layers."""
        dims = []
        for scale_dims, num_layers in zip(self.hidden_dims, self.layers_per_scale):
            dims.extend([scale_dims] * (num_layers + 1))
        
        if self.up_sample:
            dims = dims + dims[:-1][::-1]
            
        layers = nn.ModuleList()
        for i in range(self.num_total_layers):
            layers.append(
                ConvBlock(
                    dims[i], basis_dim, out_channels=dims[i+1],
                    act=nn.GELU(), layer_scale=layer_scale,
                    widening_factor=widening_factor, add_attr=False
                )
            )
                
        return layers

    def _precompute_layer_data(self, pos, batch, x):
        """Precompute data for all layers including transition layers."""
        layer_data = []
        layer_data_up = []
        
        for i, ((edge_type, edge_params), ratio) in enumerate(zip(self.edge_types, self.ratios)):
            # Compute current scale edges and features
            if edge_type == "fc":
                edge_index = fully_connected_edge_index(batch)
                cutoff_fn = None
            elif edge_type == "knn":
                edge_index = knn_graph(pos, batch=batch, loop=True, **edge_params).flip(0)
                cutoff_fn = None
            elif edge_type == "radius":
                r = edge_params["r"]
                cutoff_fn = edge_params["cutoff"]
                edge_index = radius_graph(pos, batch=batch, loop=True, r=r).flip(0)
                
            spatial_inv, spherical_inv = self.compute_invariants(
                pos[edge_index[0]], pos[edge_index[1]], self.sphere_grid
            )
            
            # Apply distance windowing if using radius graph
            if cutoff_fn is not None:
                distances = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
                window = cutoff_fn(distances)
                spatial_inv = torch.einsum("b...,b->b...", spatial_inv, window)
            
            # Generate kernel basis using global basis functions
            spatial_kernel = self.spatial_basis_fn(spatial_inv)
            spherical_kernel = self.spherical_basis_fn(spherical_inv) if self.sphere_grid is not None else None
            
            # Add to layer data
            curr_data = (spatial_kernel, spherical_kernel, edge_index, batch)
            layer_data.extend([curr_data] * self.layers_per_scale[i])
            
            if self.up_sample:
                layer_data_up = [curr_data] * self.layers_per_scale[i] + layer_data_up
            
            # Handle transition layer if needed
            if ratio < 1.0 and i < len(self.ratios) - 1:
                edge_index, fps_pos, fps_batch = fps_edge_index(pos, batch, ratio=ratio)
                spatial_inv, spherical_inv = self.compute_invariants(
                    pos[edge_index[0]], fps_pos[edge_index[1]], self.sphere_grid
                )
                
                if cutoff_fn is not None:
                    distances = torch.norm(pos[edge_index[0]] - fps_pos[edge_index[1]], dim=-1)
                    window = cutoff_fn(distances)
                    spatial_inv = spatial_inv * window.unsqueeze(-1)
                    if spherical_inv is not None:
                        spherical_inv = spherical_inv * window.unsqueeze(-1)
                
                spatial_kernel = self.spatial_basis_fn(spatial_inv)
                spherical_kernel = self.spherical_basis_fn(spherical_inv) if self.sphere_grid is not None else None
                
                trans_data = (spatial_kernel, spherical_kernel, edge_index, fps_batch)
                layer_data.append(trans_data)
                
                if self.up_sample:
                    up_trans_data = (spatial_kernel, spherical_kernel, edge_index.flip(0), fps_batch)
                    layer_data_up = [up_trans_data] + layer_data_up
                
                pos, batch = fps_pos, fps_batch
                
        return layer_data + layer_data_up
    
    
    def forward(self, x, pos, edge_index, batch=None, vec=None):
        # Precompute interaction data
        spatial_cond = x[..., -1:] if (self.last_feature_conditioning and x is not None) else None
        layer_data = self._precompute_layer_data(pos, batch, spatial_cond)
        
        # Process inputs
        x = self._process_inputs(x, vec)
        x = self.x_embedder(x)
        
        # Forward pass through network
        residuals = []
        
        for i in range(self.num_total_layers):
            residual = x
            spatial_kernel, spherical_kernel, edge_index, batch = layer_data[i]
            
            x = self.layers[i](x, spatial_kernel, spherical_kernel, edge_index)
            
            # Handle residual connections
            if self.skip_connections:
                if residual.shape[0] > x.shape[0]:  # Downsampling
                    residuals.append(residual)
                elif residual.shape[0] < x.shape[0]:  # Upsampling
                    x = x + residuals.pop()
        

        if self.sphere_grid is not None:
            x = x.mean(dim=-2)
        
        output = x
        if self.global_pooling:
            # scalar_out = scatter_add(scalar_out, batch, dim_size=batch.max().item() + 1) if scalar_out is not None else None
            output = pool.global_add_pool(output, batch)
            # vector_out = scatter_add(vector_out, batch, dim_size=batch.max().item() + 1) if vector_out is not None else None

        # Final readout
        output = self.readout(output)
            
        return output, None
    
    def pre_activation_forward(self, x, pos, edge_index, batch=None, vec=None):
        # Precompute interaction data
        spatial_cond = x[..., -1:] if (self.last_feature_conditioning and x is not None) else None
        layer_data = self._precompute_layer_data(pos, batch, spatial_cond)
        
        # Process inputs
        x = self._process_inputs(x, vec)
        x = self.x_embedder(x)
        
        # Forward pass through network
        residuals = []
        
        for i in range(self.num_total_layers):
            residual = x
            spatial_kernel, spherical_kernel, edge_index, batch = layer_data[i]
            
            x = self.layers[i](x, spatial_kernel, spherical_kernel, edge_index)
            
            # Handle residual connections
            if self.skip_connections:
                if residual.shape[0] > x.shape[0]:  # Downsampling
                    residuals.append(residual)
                elif residual.shape[0] < x.shape[0]:  # Upsampling
                    x = x + residuals.pop()
        

        if self.sphere_grid is not None:
            x = x.mean(dim=-2)
        
        output = x
        if self.global_pooling:
            # scalar_out = scatter_add(scalar_out, batch, dim_size=batch.max().item() + 1) if scalar_out is not None else None
            output = pool.global_add_pool(output, batch)
            # vector_out = scatter_add(vector_out, batch, dim_size=batch.max().item() + 1) if vector_out is not None else None

        # Final readout
        # output = self.readout(output)
            
        return output, None
    
    

    def _process_inputs(self, x, vec):
        """Process scalar and vector inputs into orientation-aware features."""
        if self.sphere_grid is None:
            return x
            
        x_list = []
        if x is not None:
            x_list.append(x.unsqueeze(-2).repeat_interleave(self.num_ori, dim=-2))
        if vec is not None:
            x_list.append(torch.einsum('bcd,nd->bnc', vec, self.sphere_grid))
            
        return torch.cat(x_list, dim=-1)

    def _process_output(self, readout, batch):
        """Process network output into scalar and vector predictions."""
        scalar_out, vector_out = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)
        
        if self.sphere_grid is not None:
            scalar_out = scalar_out.mean(dim=-2)
            if vector_out.numel() > 0:
                vector_out = torch.einsum("boc,od->bcd", vector_out, self.sphere_grid) / self.num_ori
            else:
                vector_out = None
                
        if self.global_pooling:
            scalar_out = scatter_add(scalar_out, batch, dim_size=batch.max().item() + 1) if scalar_out is not None else None
            vector_out = scatter_add(vector_out, batch, dim_size=batch.max().item() + 1) if vector_out is not None else None
            
        return scalar_out, vector_out