# model.py
# GA-GNO: Graph-Aware Graph Neural Operator
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

############################################
# 1. GRAPH ENCODER (Local Geometry Encoding)
############################################

class GraphEncoderLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="mean")

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphEncoderLayer(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x, edge_index)  # residual

        return x   # [N, hidden_dim]

############################################
# 2. LATENT GRID MAPPER (Mesh → Grid)
############################################

class LatentGridMapper(nn.Module):
    """
    Projects irregular mesh node features onto a structured latent grid
    """
    def __init__(self, hidden_dim, grid_size):
        super().__init__()

        self.grid_size = grid_size
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, node_coords):
        """
        node_features : [N, C]
        node_coords   : [N, 2 or 3] (normalized to [0,1])
        """
        B = 1
        C = node_features.shape[1]
        H, W = self.grid_size

        grid = torch.zeros((B, C, H, W), device=node_features.device)

        # Map coordinates → grid indices
        coords = node_coords.clone()
        coords = coords.clamp(0, 1)

        ix = (coords[:, 0] * (H - 1)).long()
        iy = (coords[:, 1] * (W - 1)).long()

        features = self.proj(node_features)

        for i in range(features.size(0)):
            grid[0, :, ix[i], iy[i]] += features[i]

        return grid  # [1, C, H, W]

############################################
# 3. SPECTRAL OPERATOR (FNO BLOCK)
############################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        # Extract and process only the modes we care about
        x_ft_low = x_ft[:, :, :self.modes1, :self.modes2]
        
        out_ft_low = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft_low,
            self.weights
        )
        
        # Pad back to full size using torch.nn.functional.pad
        # Padding: (left, right, top, bottom, front, back)
        pad_right = (W // 2 + 1) - self.modes2
        pad_bottom = H - self.modes1
        
        out_ft = F.pad(out_ft_low, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, modes1=12, modes2=12):
        super().__init__()

        self.spectral = SpectralConv2d(hidden_dim, hidden_dim, modes1, modes2)
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))

############################################
# 4. GRAPH DECODER (Grid → Mesh)
############################################

class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, grid, node_coords):
        """
        grid        : [1, C, H, W]
        node_coords : [N, 2]
        """
        B, C, H, W = grid.shape
        coords = node_coords.clamp(0, 1)

        ix = (coords[:, 0] * (H - 1)).long()
        iy = (coords[:, 1] * (W - 1)).long()

        node_features = grid[0, :, ix, iy].transpose(0, 1)
        return self.mlp(node_features)

############################################
# 5. FULL GA-GNO MODEL
############################################

class GAGNO(nn.Module):
    def __init__(
        self,
        node_in_dim,
        hidden_dim,
        out_dim,
        num_gnn_layers,
        grid_size=(64, 64)
    ):
        super().__init__()

        self.encoder = GraphEncoder(
            node_in_dim,
            hidden_dim,
            num_gnn_layers
        )

        self.mapper = LatentGridMapper(hidden_dim, grid_size)

        self.fno = nn.Sequential(
            FNOBlock(hidden_dim),
            # FNOBlock(hidden_dim),
            # FNOBlock(hidden_dim)
        )

        self.decoder = GraphDecoder(hidden_dim, out_dim)

    def forward(self, data):
        """
        data.x          : node features
        data.edge_index : mesh connectivity
        data.pos        : node coordinates (normalized)
        """

        node_latent = self.encoder(data)
        latent_grid = self.mapper(node_latent, data.pos)
        latent_grid = self.fno(latent_grid)
        out = self.decoder(latent_grid, data.pos)

        return out   # Cp at mesh nodes
