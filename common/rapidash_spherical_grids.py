# MIT License, (c) 2024  (c) 2024 Sharvaree Vadgama
import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm import trange
from torch.optim import SGD

class SphericalGridGenerator(nn.Module):
    """
    Generates uniform grids on S1 (circle) or S2 (sphere) for orientation-aware features.
    Uses uniform spacing for S1 and either Fibonacci lattice or repulsion method for S2.
    """
    
    def __init__(
        self,
        dim: int,
        num_points: int,
        use_repulsion: bool = False,
        repulsion_steps: int = 200,
        repulsion_step_size: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim: Spatial dimension (2 for S1, 3 for S2)
            num_points: Number of points to generate on the sphere
            use_repulsion: Whether to use repulsion method for S2 instead of Fibonacci lattice
            repulsion_steps: Number of optimization steps for repulsion method
            repulsion_step_size: Learning rate for repulsion optimization
            device: Torch device for the grid
        """
        super().__init__()
        
        if dim not in [2, 3]:
            raise ValueError("Dimension must be 2 (S1) or 3 (S2)")
            
        self.dim = dim
        self.num_points = num_points
        self.use_repulsion = use_repulsion
        self.repulsion_steps = repulsion_steps
        self.repulsion_step_size = repulsion_step_size
        self.device = device if device else torch.device("cpu")

    def forward(self) -> torch.Tensor:
        """Generate uniform grid on S1 or S2."""
        if self.dim == 2:
            return self._generate_s1()
        else:  # dim == 3
            if self.use_repulsion:
                return self._generate_s2_repulsion()
            return self._generate_s2_fibonacci()

    def _generate_s1(self) -> torch.Tensor:
        """Generate uniform grid on S1 (circle) using uniform angle spacing."""
        angles = torch.linspace(
            start=0,
            end=2 * math.pi * (1 - 1/self.num_points),
            steps=self.num_points,
            device=self.device
        )
        
        x = torch.cos(angles)
        y = torch.sin(angles)
        
        return torch.stack((x, y), dim=1)

    def _generate_s2_fibonacci(self) -> torch.Tensor:
        """Generate uniform grid on S2 (sphere) using Fibonacci lattice."""
        offset = 0.5  # Optimal offset for uniformity
        
        i = torch.arange(self.num_points, device=self.device)
        
        # Golden angle in radians
        theta = (math.pi * (1 + math.sqrt(5)) * i) % (2 * math.pi)
        
        # Z is evenly spaced from 1 down to -1, with offset
        phi = torch.acos(1 - 2 * (i + offset) / (self.num_points - 1 + 2 * offset))
        
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        
        return torch.stack((x, y, z), dim=1)

    def _random_s2(self) -> torch.Tensor:
        """Generate random points on S2."""
        x = torch.randn((self.num_points, 3), device=self.device)
        return x / torch.linalg.norm(x, dim=-1, keepdim=True)

    def _generate_s2_repulsion(self) -> torch.Tensor:
        """Generate uniform grid on S2 using repulsion method."""
        # Start with random points
        grid = self._random_s2()
        grid.requires_grad_(True)
        optimizer = SGD([grid], lr=self.repulsion_step_size)

        # Optimization loop with progress bar
        for _ in trange(self.repulsion_steps, desc="Optimizing grid"):
            optimizer.zero_grad()
            
            # Calculate pairwise distances and energy
            dists = torch.cdist(grid, grid, p=2)
            dists = torch.clamp(dists, min=1e-6)  # Avoid division by zero
            energy = dists.pow(-2).sum()  # Coulomb energy
            
            # Backward pass and optimization step
            energy.backward()
            optimizer.step()

            # Project points back onto sphere
            with torch.no_grad():
                grid.div_(torch.linalg.norm(grid, dim=-1, keepdim=True))

        return grid.detach()