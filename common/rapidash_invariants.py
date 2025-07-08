# MIT License, (c) 2024  (c) 2024 Sharvaree Vadgama

import torch
from torch import nn

class Invariants(nn.Module):
    """
    Computes invariant features for equivariant convolutions.
    
    Supports the following cases:
    1. Translation equivariance (Tn):
       - Returns displacement vectors for num_ori = 0
       - Returns combined displacement and angular features for num_ori > 0
       
    2. Euclidean equivariance (SEn):
       - Returns distances for num_ori = 0
       - Returns relative displacement quantities and angular distance features for num_ori > 0
    """
    
    def __init__(
        self,
        dim: int,  # Spatial dimension (2 or 3)
        equivariance: str,  # Type of equivariance ("Tn" or "SEn")
        num_ori: int = 0,  # Number of orientations (0 for no spherical features)
    ):
        super().__init__()
        
        if dim not in [2, 3]:
            raise ValueError(f"Dimension must be 2 or 3, got {dim}")
        if equivariance not in ["Tn", "SEn"]:
            raise ValueError(f"Equivariance must be 'Tn' or 'SEn', got {equivariance}")
        if num_ori < 0:
            raise ValueError(f"num_ori must be non-negative, got {num_ori}")
            
        self.dim = dim
        self.equivariance = equivariance
        self.num_ori = num_ori

    def forward(self, pos_send, pos_receive, sphere_grid=None):
        """
        Compute invariant features between pairs of points.
        
        Args:
            pos_send: Source point positions [num_edges, dim]
            pos_receive: Target point positions [num_edges, dim]
            sphere_grid: Orientation grid points [num_ori, dim] if num_ori > 0
            
        Returns:
            spatial_invariants: [num_edges, num_features] or [num_edges, num_ori, num_features]
            spherical_invariants: [num_ori, num_ori, num_features] or None
        """
        # Basic case: no orientation features
        if self.num_ori == 0:
            return self._compute_spatial_invariants(pos_send, pos_receive), None
            
        # Case with orientation features
        if sphere_grid is None:
            raise ValueError("sphere_grid required when num_ori > 0")
            
        return self._compute_oriented_invariants(pos_send, pos_receive, sphere_grid)

    def _compute_spatial_invariants(self, pos_send, pos_receive):
        """Compute basic spatial invariants without orientation features."""
        rel_pos = pos_send - pos_receive  # [num_edges, dim]
        
        if self.equivariance == "Tn":
            return rel_pos
        else:  # SEn
            return rel_pos.norm(dim=-1, keepdim=True)

    def _compute_oriented_invariants(self, pos_send, pos_receive, sphere_grid):
        """Compute invariants with orientation features."""
        rel_pos = pos_send - pos_receive  # [num_edges, dim]
        
        # Expand dimensions for broadcasting
        rel_pos = rel_pos[:, None, :]  # [num_edges, 1, dim]
        grid_send = sphere_grid[None, :, :]  # [1, num_ori, dim]
        grid_receive = sphere_grid[:, None, :]  # [num_ori, 1, dim]
        
        # Compute basic invariants
        radial_projection = (rel_pos * grid_send).sum(dim=-1, keepdim=True)  # [num_edges, num_ori, 1]
        orthogonal_distance = self._compute_orthogonal_distance(
            rel_pos, radial_projection, grid_send
        )  # [num_edges, num_ori, 1]
        angular_distance = (grid_send * grid_receive).sum(dim=-1, keepdim=True)  # [num_ori, num_ori, 1]
        
        # Combine features based on equivariance type
        if self.equivariance == "Tn":
            # Include translation invariants for Tn
            translation_invariants = rel_pos.repeat(1, self.num_ori, 1)  # [num_edges, num_ori, dim]
            spatial_invariants = torch.cat(
                [translation_invariants, radial_projection, orthogonal_distance], 
                dim=-1
            )  # [num_edges, num_ori, dim+2]
        else:  # SEn
            spatial_invariants = torch.cat(
                [radial_projection, orthogonal_distance], 
                dim=-1
            )  # [num_edges, num_ori, 2]
            
        return spatial_invariants, angular_distance

    def _compute_orthogonal_distance(self, rel_pos, radial_projection, grid):
        """Compute distance orthogonal to orientation direction."""
        projected_pos = radial_projection * grid  # [num_edges, num_ori, dim]
        orthogonal_component = rel_pos - projected_pos
        
        if self.dim == 2:
            # For 2D, we can use the signed distance
            return orthogonal_component.sum(dim=-1, keepdim=True)
        else:  # 3D
            # For 3D, we use the norm of the orthogonal component
            return orthogonal_component.norm(dim=-1, keepdim=True)