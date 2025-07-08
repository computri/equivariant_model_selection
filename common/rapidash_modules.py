# MIT License, (c) 2024  (c) 2024 Sharvaree Vadgama

import torch
import torch.nn as nn

from .rapidash_utils import scatter_add


class SeparableFiberBundleConv(nn.Module):
    """Separable convolution that handles both spatial and spherical features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dim: int,
        bias: bool = True,
        groups: int = 1,
        edge_attr_dim: int = 0,
        add_attr: bool = False,
    ):
        super().__init__()

        # Check arguments
        if groups == 1:
            self.depthwise = False
        elif groups == in_channels and groups == out_channels:
            self.depthwise = True
        else:
            assert ValueError(
                "Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Construct kernels
        self.pre_linear = nn.Linear(in_channels + edge_attr_dim, in_channels, bias=False)
        self.kernel = nn.Linear(kernel_dim, in_channels, bias=False)
        self.fiber_kernel = nn.Linear(
            kernel_dim, int(in_channels * out_channels / groups), bias=False
        )

        # Construct bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter("bias", None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        self.add_attr = add_attr

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr=None):
        """Forward pass of the separable convolution."""
        # Process input features
        if edge_attr is not None:
            x_s = self.pre_linear(torch.cat([x[edge_index[0]], edge_attr], dim=-1))
        else:
            x_s = x[edge_index[0]]

        # 1. Spatial convolution
        if self.add_attr:
            x_s += kernel_basis
        message = x_s * self.kernel(kernel_basis)  # [num_edges, num_ori, in_channels]
        x_1 = scatter_add(
            src=message, index=edge_index[1], dim_size=edge_index[1].max().item() + 1
        )

        # 2. Spherical convolution
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise:
            x_2 = (
                torch.einsum("boc,poc->bpc", x_1, fiber_kernel) / fiber_kernel.shape[-2]
            )
        else:
            x_2 = (
                torch.einsum(
                    "boc,podc->bpd",
                    x_1,
                    fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)),
                )
                / fiber_kernel.shape[-2]
            )

        # Re-calibrate the initialization
        if self.training and not self.callibrated:
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        return x_2

    def callibrate(self, std_in, std_1, std_2):
        print("Calibrating...")
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
            self.fiber_kernel.weight.data = (
                self.fiber_kernel.weight.data * std_1 / std_2
            )
            self.callibrated = ~self.callibrated

class SeparableFiberBundleConvNext(nn.Module):
    """ConvNext block with separable convolution for both spatial and spherical features."""

    def __init__(
        self,
        in_channels: int,
        kernel_dim: int,
        out_channels: int = None,
        act: nn.Module = nn.GELU(),
        layer_scale: float = 1e-6,
        widening_factor: int = 4,
        add_attr: bool = False,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.conv = SeparableFiberBundleConv(
            in_channels,
            in_channels,
            kernel_dim,
            groups=in_channels,
            add_attr=add_attr,
        )

        self.act_fn = act
        self.linear_1 = nn.Linear(in_channels, widening_factor * in_channels)
        self.linear_2 = nn.Linear(widening_factor * in_channels, out_channels)

        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(out_channels) * layer_scale)
        else:
            self.register_buffer("layer_scale", None)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr=None):
        """Forward pass of the ConvNext block."""
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)

        if self.layer_scale is not None:
            x = self.layer_scale * x

        if x.shape == input.shape:
            x = x + input

        return x

class Conv(nn.Module):
    """Basic convolution module for non-spherical features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dim: int,
        bias: bool = True,
        groups: int = 1,
        edge_attr_dim: int = 0,
        add_attr: bool = False,
    ):
        super().__init__()

        # Check arguments
        if groups == 1:
            self.depthwise = False
        elif groups == in_channels and groups == out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError(
                "Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)"
            )

        # Construct kernels
        self.pre_linear = nn.Linear(in_channels + edge_attr_dim, in_channels, bias=False)
        self.kernel = nn.Linear(kernel_dim, int(out_channels * in_channels / groups), bias=False)

        # Construct bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter("bias", None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        self.add_attr = add_attr

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr=None):
        """Forward pass of the basic convolution."""
        if edge_attr is not None:
            x_s = self.pre_linear(torch.cat([x[edge_index[0]], edge_attr], dim=-1))
        else:
            x_s = x[edge_index[0]]

        if self.add_attr:
            x_s += kernel_basis

        # Spatial convolution
        kernel = self.kernel(kernel_basis)
        if self.depthwise:
            message = x_s * kernel
        else:
            message = torch.einsum('boi,bi->bo', kernel.unflatten(-1, (self.out_channels, self.in_channels)), x_s)

        x_1 = scatter_add(
            src=message, index=edge_index[1], dim_size=edge_index[1].max().item() + 1
        )

        # Re-calibrate the initialization
        if self.training and not self.callibrated:
            self.callibrate(x.std(), x_1.std())

        # Add bias
        if self.bias is not None:
            return x_1 + self.bias
        return x_1

    def callibrate(self, std_in, std_1):
        print("Calibrating...")
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
            self.callibrated = ~self.callibrated

class ConvNext(nn.Module):
    """ConvNext block with basic convolution for non-spherical features."""

    def __init__(
        self,
        in_channels: int,
        kernel_dim: int,
        out_channels: int = None,
        act: nn.Module = nn.GELU(),
        layer_scale: float = 1e-6,
        widening_factor: int = 4,
        add_attr: bool = False,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.conv = Conv(
            in_channels,
            in_channels,
            kernel_dim,
            groups=in_channels,
            add_attr=add_attr,
        )

        self.act_fn = act
        self.linear_1 = nn.Linear(in_channels, widening_factor * in_channels)
        self.linear_2 = nn.Linear(widening_factor * in_channels, out_channels)

        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(out_channels) * layer_scale)
        else:
            self.register_buffer("layer_scale", None)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr=None):
        """Forward pass of the ConvNext block."""
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index, edge_attr)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)

        if self.layer_scale is not None:
            x = self.layer_scale * x

        if x.shape == input.shape:
            x = x + input

        return x

class PolynomialFeatures(nn.Module):
    """Generates polynomial features up to specified degree."""
    
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate polynomial features."""
        polynomial_list = [x]
        for it in range(1, self.degree + 1):
            polynomial_list.append(
                torch.einsum("...i,...j->...ij", polynomial_list[-1], x).flatten(-2, -1)
            )
        return torch.cat(polynomial_list, -1)

class PolynomialCutoff(torch.nn.Module):
    """
    Distance windowing function from DimeNet.
    Smoothly decays to zero at r_max.
    
    Reference:
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    """
    def __init__(self, r_max, p=6):
        super().__init__()
        if r_max is not None:
            self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
            self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        else:
            self.r_max = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r_max is not None:
            envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
            )
            return envelope * (x < self.r_max)
        else:
            return torch.ones_like(x)