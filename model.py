import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GENConv


############################################
# 1. GRAPH ENCODER
############################################

class FourierPositionEncoding(nn.Module):
    def __init__(self, num_bands=6):
        super().__init__()
        self.num_bands = num_bands

    @property
    def out_dim(self):
        return 3 * 2 * self.num_bands

    def forward(self, xyz):
        freqs = 2.0 ** torch.arange(
            self.num_bands,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        encoded = xyz.unsqueeze(-1) * freqs
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        return encoded.flatten(start_dim=1)


class GraphEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.2):
        super().__init__()

        self.conv = GENConv(
            hidden_dim,
            hidden_dim,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
            edge_dim=edge_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        return self.norm(self.dropout(out))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        edge_dim,
        dropout=0.2,
        fourier_bands=6,
    ):
        super().__init__()

        self.edge_dim = edge_dim
        self.pos_encoding = FourierPositionEncoding(fourier_bands)
        encoder_in_dim = in_dim - 3 + self.pos_encoding.out_dim
        self.input_proj = nn.Linear(encoder_in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphEncoderLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )

    def _edge_attr(self, data, x):
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            return data.edge_attr.to(device=x.device, dtype=x.dtype)
        return x.new_zeros((data.edge_index.size(1), self.edge_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xyz = getattr(data, "xyz", x[:, :3])
        xyz = xyz.to(device=x.device, dtype=x.dtype)
        x = torch.cat([self.pos_encoding(xyz), x[:, 3:]], dim=1)
        x = self.input_proj(x)
        edge_attr = self._edge_attr(data, x)

        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)

        return x


############################################
# 2. LEARNED ATTENTION: GRAPH -> GRID
############################################

class CoordinateEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords):
        return self.mlp(coords)


class LatentGridMapper(nn.Module):
    def __init__(
        self,
        hidden_dim,
        grid_size,
        dropout=0.2,
        num_heads=4,
        query_chunk_size=4096,
        max_attention_nodes=1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.query_chunk_size = query_chunk_size
        self.max_attention_nodes = max_attention_nodes

        self.node_pos = CoordinateEmbedding(hidden_dim)
        self.grid_pos = CoordinateEmbedding(hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.grid_query_bias = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _grid_coords(self, device, dtype):
        H, W = self.grid_size
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=-1).view(H * W, 2)

    def _select_attention_nodes(self, node_features, node_coords):
        num_nodes = node_features.size(0)
        if num_nodes <= self.max_attention_nodes:
            return node_features, node_coords

        idx = torch.linspace(
            0,
            num_nodes - 1,
            self.max_attention_nodes,
            device=node_features.device,
        ).long()
        return node_features.index_select(0, idx), node_coords.index_select(0, idx)

    def forward(self, node_features, node_coords):
        if node_features.dim() != 2:
            raise ValueError("LatentGridMapper expects a single graph with shape [N, C].")

        H, W = self.grid_size
        device = node_features.device
        dtype = node_features.dtype

        node_features, node_coords = self._select_attention_nodes(
            node_features, node_coords
        )
        kv = self.node_proj(node_features) + self.node_pos(node_coords)
        kv = kv.unsqueeze(0)

        grid_coords = self._grid_coords(device, dtype)
        grid_chunks = []

        for start in range(0, grid_coords.size(0), self.query_chunk_size):
            coords = grid_coords[start:start + self.query_chunk_size]
            query = self.grid_pos(coords)
            query = query + self.grid_query_bias.to(dtype=query.dtype)
            query = query.unsqueeze(0)
            out, _ = self.attn(query, kv, kv, need_weights=False)
            out = self.norm(out + query)
            grid_chunks.append(out.squeeze(0))

        grid_tokens = torch.cat(grid_chunks, dim=0)
        return grid_tokens.T.contiguous().view(1, self.hidden_dim, H, W)


############################################
# 3. SPECTRAL CONV (REDUCED MODES)
############################################

class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        grid_size,
        modes_ratio=0.015,
        min_modes=8,
        max_modes=20,
    ):
        super().__init__()

        H, W = grid_size

        self.modes1 = max(min_modes, min(int(H * modes_ratio), max_modes))
        self.modes2 = max(min_modes, min(int(W * modes_ratio), max_modes))

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
        x = x.float()
        x_ft = torch.fft.rfft2(x)

        weights = torch.complex(
            self.weights_real.float(),
            self.weights_imag.float(),
        )

        x_ft_low = x_ft[:, :, : self.modes1, : self.modes2]
        out_ft_low = torch.einsum("bixy,ioxy->boxy", x_ft_low, weights)

        pad_right = (x.shape[-1] // 2 + 1) - self.modes2
        pad_bottom = x.shape[-2] - self.modes1

        out_ft = F.pad(out_ft_low, (0, pad_right, 0, pad_bottom))
        x_out = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))

        return x_out


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, grid_size):
        super().__init__()

        self.spectral = SpectralConv2d(hidden_dim, hidden_dim, grid_size)
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        x = F.gelu(x)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x


############################################
# 4. GRID -> GRAPH DECODER
############################################

class GraphRefineLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.2):
        super().__init__()

        self.conv = GENConv(
            hidden_dim,
            hidden_dim,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
            edge_dim=edge_dim,
        )
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        return self.norm(self.dropout(out))


class GraphDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        edge_dim,
        dropout=0.2,
        num_heads=4,
        local_radius=1,
    ):
        super().__init__()

        self.local_radius = local_radius
        self.node_pos = CoordinateEmbedding(hidden_dim)
        self.grid_pos = CoordinateEmbedding(hidden_dim)
        self.sample_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.edge_dim = edge_dim
        self.refine = GraphRefineLayer(hidden_dim, edge_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.gnn_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _local_offsets(self, device, dtype, grid):
        H, W = grid.shape[-2:]
        offsets = []
        for dy in range(-self.local_radius, self.local_radius + 1):
            for dx in range(-self.local_radius, self.local_radius + 1):
                offsets.append((2.0 * dy / max(H - 1, 1), 2.0 * dx / max(W - 1, 1)))
        return torch.tensor(offsets, device=device, dtype=dtype)

    def _sample_local_grid(self, grid, node_coords):
        offsets = self._local_offsets(grid.device, grid.dtype, grid)
        local_coords = node_coords.unsqueeze(1) + offsets.unsqueeze(0)
        local_coords = local_coords.clamp(-1, 1)

        sample_coords = torch.stack(
            [local_coords[..., 1], local_coords[..., 0]],
            dim=-1,
        ).unsqueeze(0)

        sampled = F.grid_sample(
            grid,
            sample_coords,
            mode="bilinear",
            align_corners=True,
        )
        sampled = sampled.squeeze(0).permute(1, 2, 0).contiguous()
        return sampled, local_coords

    def _edge_attr(self, edge_index, edge_attr, node_latent):
        if edge_attr is not None:
            return edge_attr.to(device=node_latent.device, dtype=node_latent.dtype)
        return node_latent.new_zeros((edge_index.size(1), self.edge_dim))

    def forward(self, grid, node_coords, node_latent, edge_index, edge_attr=None):
        sampled_grid, sampled_coords = self._sample_local_grid(grid, node_coords)

        keys = self.sample_proj(sampled_grid) + self.grid_pos(sampled_coords)
        query = self.query_proj(node_latent) + self.node_pos(node_coords)
        query = query.unsqueeze(1)

        attn_out, _ = self.cross_attn(query, keys, keys, need_weights=False)
        node_latent = self.attn_norm(node_latent + attn_out.squeeze(1))

        edge_attr = self._edge_attr(edge_index, edge_attr, node_latent)
        refined = self.refine(node_latent, edge_index, edge_attr)
        node_latent = self.gnn_norm(node_latent + refined)

        return self.mlp(node_latent)


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
        edge_dim=8,
        dropout=0.2,
        fourier_bands=6,
        attention_heads=4,
        attention_query_chunk_size=4096,
        attention_max_nodes=1024,
        attention_local_radius=1,
    ):
        super().__init__()

        self.encoder = GraphEncoder(
            node_in_dim,
            hidden_dim,
            num_gnn_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            fourier_bands=fourier_bands,
        )
        self.mapper = LatentGridMapper(
            hidden_dim,
            grid_size,
            dropout=dropout,
            num_heads=attention_heads,
            query_chunk_size=attention_query_chunk_size,
            max_attention_nodes=attention_max_nodes,
        )

        self.fno = nn.Sequential(
            FNOBlock(hidden_dim, grid_size),
            FNOBlock(hidden_dim, grid_size),
            FNOBlock(hidden_dim, grid_size),
        )

        self.decoder = GraphDecoder(
            hidden_dim,
            out_dim,
            edge_dim=edge_dim,
            dropout=dropout,
            num_heads=attention_heads,
            local_radius=attention_local_radius,
        )

    def forward(self, data):
        node_latent = self.encoder(data)
        grid = self.mapper(node_latent, data.pos)
        grid = self.fno(grid)
        edge_attr = getattr(data, "edge_attr", None)
        return self.decoder(grid, data.pos, node_latent, data.edge_index, edge_attr)
