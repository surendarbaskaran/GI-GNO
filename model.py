import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from  config import *
############################################
# 1. GRAPH ENCODER
############################################

class GraphEncoderLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        return self.norm(out)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphEncoderLayer(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x, edge_index)
        return x  # [N, C]


############################################
# 2. LATENT GRID MAPPER (SUM / COUNT)
############################################

class LatentGridMapper(nn.Module):
    def __init__(self, hidden_dim, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, node_coords):
        C = node_features.size(1)
        H, W = self.grid_size
        device = node_features.device

        features = self.proj(node_features)

        coords = node_coords.clamp(0.0, 1.0)
        ix = (coords[:, 0] * (H - 1)).long()
        iy = (coords[:, 1] * (W - 1)).long()
        flat_idx = ix * W + iy

        grid_sum = torch.zeros((C, H * W), device=device, dtype=features.dtype)
        grid_sum.scatter_add_(
            1, flat_idx.unsqueeze(0).expand(C, -1), features.T
        )

        grid_count = torch.zeros((1, H * W), device=device, dtype=features.dtype)
        grid_count.scatter_add_(
            1, flat_idx.unsqueeze(0),
            torch.ones_like(flat_idx, dtype=features.dtype).unsqueeze(0),
        )

        grid = grid_sum / (grid_count + 1e-6)
        return grid.view(1, C, H, W)


############################################
# 3. SPECTRAL CONV (GRID-AWARE MODES)
############################################

class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        grid_size,
        modes_ratio=0.025,
        min_modes=16,
        max_modes=32,
    ):
        super().__init__()
        H, W = grid_size

        self.modes1 = max(
            min_modes, min(int(H * modes_ratio), max_modes)
        )
        self.modes2 = max(
            min_modes, min(int(W * modes_ratio), max_modes)
        )

        self.scale = 1 / (in_channels * out_channels)

        self.weights_real = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2
            )
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.float()  # FFT safety

        x_ft = torch.fft.rfft2(x)
        weights = torch.complex(self.weights_real, self.weights_imag)

        x_ft_low = x_ft[:, :, :self.modes1, :self.modes2]
        out_ft_low = torch.einsum(
            "bixy,ioxy->boxy", x_ft_low, weights
        )

        pad_right = (W // 2 + 1) - self.modes2
        pad_bottom = H - self.modes1
        out_ft = F.pad(out_ft_low, (0, pad_right, 0, pad_bottom))

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, grid_size):
        super().__init__()
        self.spectral = SpectralConv2d(
            hidden_dim, hidden_dim, grid_size
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = torch.clamp(x, -10.0, 10.0)
        x = self.spectral(x) + self.pointwise(x)
        x = F.gelu(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


############################################
# 4. MESH-AWARE DECODER (UPGRADED)
############################################

class GraphRefineLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.refine = GraphRefineLayer(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, grid, node_coords, node_latent, edge_index):
        _, C, H, W = grid.shape

        coords = node_coords.clamp(0.0, 1.0)
        ix = (coords[:, 0] * (H - 1)).long()
        iy = (coords[:, 1] * (W - 1)).long()

        grid_feat = grid[0, :, ix, iy].T

        x = grid_feat + node_latent

        x = x + self.refine(x, edge_index)

        return self.mlp(x)


############################################
# 5. FULL MODEL
############################################

class GAGNO(nn.Module):
    def __init__(
        self,
        node_in_dim,
        hidden_dim,
        out_dim,
        num_gnn_layers,
        grid_size=(1024, 512),
    ):
        super().__init__()

        self.encoder = GraphEncoder(
            node_in_dim, hidden_dim, num_gnn_layers
        )

        self.mapper = LatentGridMapper(hidden_dim, grid_size)

        self.fno = nn.Sequential(
            FNOBlock(hidden_dim, grid_size),
            FNOBlock(hidden_dim, grid_size),
            FNOBlock(hidden_dim, grid_size),
        )

        self.decoder = GraphDecoder(hidden_dim, out_dim)

    def forward(self, data):
        node_latent = self.encoder(data)
        grid = self.mapper(node_latent, data.pos)
        grid = self.fno(grid)
        grid = torch.clamp(grid, -5.0, 5.0)

        return self.decoder(
            grid, data.pos, node_latent, data.edge_index
        )
