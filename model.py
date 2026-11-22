import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import GENConv

from config import (
    NODE_IN_DIM,
    EDGE_DIM,
    HIDDEN_DIM,
    OUT_DIM,
    NUM_ENCODER_LAYERS,
    NUM_GNO_BRANCHES,
    NUM_GNO_LAYERS,
    GNO_FUSION,
    NUM_DECODER_LAYERS,
    NUM_GNN_LAYERS,
    FOURIER_BANDS,
    DROPOUT,
    USE_GRAD_CHECKPOINT,
    GNO_EDGE_CHUNK_SIZE,
)


############################################
# 1. FOURIER POSITION ENCODING
############################################

class FourierPositionEncoding(nn.Module):
    """Multi-scale Fourier frequency embedding for coordinates starting at (0,0,0) to (1,_,_).
    Evaluated in float32 to preserve high-frequency leading-edge coordinate precision.
    """

    def __init__(self, num_bands=FOURIER_BANDS):
        super().__init__()
        self.num_bands = num_bands
        freqs = (2.0 ** torch.arange(num_bands, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self):
        return 3 * 2 * self.num_bands

    def forward(self, xyz):
        xyz_fp32 = xyz.float()
        encoded = xyz_fp32.unsqueeze(-1) * self.freqs
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        return encoded.flatten(start_dim=1).to(dtype=xyz.dtype)


############################################
# 2. GRAPH ENCODER (enc)
############################################

class GraphEncoderLayer(nn.Module):
    """Deep GENConv layer with LayerNorm, GELU, and residual connection."""

    def __init__(self, hidden_dim, edge_dim, dropout=DROPOUT):
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
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(x + out)


class GraphEncoder(nn.Module):
    """Encodes physical node features, geometry parameters, and coordinates into latent graph embeddings."""

    def __init__(
        self,
        in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_ENCODER_LAYERS,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        fourier_bands=FOURIER_BANDS,
        use_checkpoint=USE_GRAD_CHECKPOINT,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.use_checkpoint = use_checkpoint
        self.pos_encoding = FourierPositionEncoding(fourier_bands)
        encoder_in_dim = in_dim - 3 + self.pos_encoding.out_dim
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [GraphEncoderLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )

    def _edge_attr(self, data, x):
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            return data.edge_attr.to(device=x.device, dtype=x.dtype)
        return x.new_zeros((data.edge_index.size(1), self.edge_dim))

    def forward(self, data, edge_index=None, edge_attr=None, xyz=None):
        if hasattr(data, "x"):
            x = data.x
            edge_index = data.edge_index
            xyz = getattr(data, "xyz", getattr(data, "pos", x[:, :3]))
            edge_attr = self._edge_attr(data, x)
        else:
            x = data
            if xyz is None:
                xyz = x[:, :3]
            if edge_attr is None:
                edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))

        pos_feat = self.pos_encoding(xyz.to(device=x.device, dtype=torch.float32)).to(dtype=x.dtype)
        x = torch.cat([pos_feat, x[:, 3:]], dim=1)
        x = self.input_proj(x)

        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
                x = checkpoint(layer, x, edge_index, edge_attr, use_reentrant=False)
            else:
                x = layer(x, edge_index, edge_attr)

        return x


############################################
# 3. PARALLEL GRAPH NEURAL OPERATORS (GNO)
############################################

class GNOKernel(nn.Module):
    """Continuous kernel network kappa(x_i, x_j, x_j - x_i, e_ij) parameterizing integral operator."""

    def __init__(self, coord_dim=3, edge_dim=EDGE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.coord_dim = coord_dim
        self.edge_dim = edge_dim
        # Input: source coordinate x_j (3), target coordinate x_i (3), relative vector x_j - x_i (3), distance (1), edge_attr (edge_dim)
        kernel_in_dim = coord_dim * 3 + 1 + edge_dim
        self.kernel_mlp = nn.Sequential(
            nn.Linear(kernel_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pos_i, pos_j, rel_pos, edge_attr):
        # Numerically safe Euclidean norm in float32 to prevent 0/0 NaN gradients on short leading-edge edges
        rel_fp32 = rel_pos.float()
        dist = torch.sqrt(torch.sum(rel_fp32 * rel_fp32, dim=-1, keepdim=True) + 1e-12).to(dtype=pos_i.dtype)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_features = edge_attr.to(dtype=pos_i.dtype)
            if edge_features.size(-1) < self.edge_dim:
                edge_features = F.pad(edge_features, (0, self.edge_dim - edge_features.size(-1)), "constant", 0.0)
            elif edge_features.size(-1) > self.edge_dim:
                edge_features = edge_features[:, :self.edge_dim]
        else:
            edge_features = pos_i.new_zeros((pos_i.size(0), self.edge_dim))
        inp = torch.cat([pos_i, pos_j, rel_pos, dist, edge_features], dim=-1)
        return self.kernel_mlp(inp)


def _prepare_gno_edge_inputs(pos_fp32, src, dst, edge_attr, edge_dim, target_dtype):
    """Build shared GNO kernel input [pos_i, pos_j, rel_pos, dist, edge_attr] once per graph."""
    pos_i_fp32 = pos_fp32[dst]
    pos_j_fp32 = pos_fp32[src]
    rel_fp32 = pos_j_fp32 - pos_i_fp32
    dist_fp32 = torch.sqrt(torch.sum(rel_fp32 * rel_fp32, dim=-1, keepdim=True) + 1e-12)

    pos_i = pos_i_fp32.to(dtype=target_dtype)
    pos_j = pos_j_fp32.to(dtype=target_dtype)
    rel_pos = rel_fp32.to(dtype=target_dtype)
    dist = dist_fp32.to(dtype=target_dtype)

    if edge_attr is not None and edge_attr.numel() > 0:
        edge_features = edge_attr.to(dtype=target_dtype)
        if edge_features.size(-1) < edge_dim:
            edge_features = F.pad(edge_features, (0, edge_dim - edge_features.size(-1)), "constant", 0.0)
        elif edge_features.size(-1) > edge_dim:
            edge_features = edge_features[:, :edge_dim]
    else:
        edge_features = pos_i.new_zeros((pos_i.size(0), edge_dim))

    return torch.cat([pos_i, pos_j, rel_pos, dist, edge_features], dim=-1)


class GNOLayer(nn.Module):
    """Continuous Graph Neural Operator layer:
    Evaluates kernel integral (K h)(x_i) = int_{N(i)} kappa(x_i, x_j, e_ij) h(x_j) dmu(x_j)
    passing graph values along the domain from (0,0,0) to (1,_,_).
    Supports precomputed shared edge inputs & inverse degree across parallel GNO layers,
    plus optional chunked edge evaluation when VRAM is constrained.
    """

    def __init__(
        self,
        hidden_dim,
        edge_dim=EDGE_DIM,
        coord_dim=3,
        dropout=DROPOUT,
        edge_chunk_size=GNO_EDGE_CHUNK_SIZE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_chunk_size = edge_chunk_size
        self.kernel_net = GNOKernel(coord_dim=coord_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.linear_self = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def _integrate_edges(
        self,
        h,
        edge_index,
        pos_fp32,
        edge_attr=None,
        weights=None,
        precomputed_kernel_inp=None,
        inv_deg=None,
    ):
        src, dst = edge_index
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        agg_fp32 = torch.zeros((num_nodes, h.size(1)), device=h.device, dtype=torch.float32)
        if num_edges == 0:
            return agg_fp32.to(dtype=h.dtype)

        chunk_size = self.edge_chunk_size if (self.edge_chunk_size and self.edge_chunk_size > 0) else num_edges

        if precomputed_kernel_inp is not None and chunk_size >= num_edges:
            kernel_val = self.kernel_net.kernel_mlp(precomputed_kernel_inp)
            msg = h[src].to(dtype=kernel_val.dtype) * kernel_val
            if weights is not None:
                msg = msg * weights[src].unsqueeze(-1).to(dtype=msg.dtype)
            agg_fp32.index_add_(0, dst, msg.float())
        else:
            for start in range(0, num_edges, chunk_size):
                end = min(start + chunk_size, num_edges)
                src_c = src[start:end]
                dst_c = dst[start:end]

                if precomputed_kernel_inp is not None:
                    kernel_val = self.kernel_net.kernel_mlp(precomputed_kernel_inp[start:end])
                else:
                    pos_i = pos_fp32[dst_c].to(dtype=h.dtype)
                    pos_j = pos_fp32[src_c].to(dtype=h.dtype)
                    rel_pos = (pos_fp32[src_c] - pos_fp32[dst_c]).to(dtype=h.dtype)
                    ea_c = edge_attr[start:end] if (edge_attr is not None and edge_attr.numel() > 0) else None
                    kernel_val = self.kernel_net(pos_i, pos_j, rel_pos, ea_c)

                msg = h[src_c].to(dtype=kernel_val.dtype) * kernel_val
                if weights is not None:
                    msg = msg * weights[src_c].unsqueeze(-1).to(dtype=msg.dtype)

                agg_fp32.index_add_(0, dst_c, msg.float())

        if weights is None:
            if inv_deg is None:
                deg = torch.bincount(dst, minlength=num_nodes).to(dtype=torch.float32).unsqueeze(-1)
                inv_deg = deg.clamp_min(1.0).reciprocal()
            agg_fp32 = agg_fp32 * inv_deg

        return agg_fp32.to(dtype=h.dtype)

    def forward(
        self,
        h,
        edge_index,
        pos,
        edge_attr=None,
        weights=None,
        precomputed_kernel_inp=None,
        inv_deg=None,
    ):
        pos_fp32 = None
        if precomputed_kernel_inp is None:
            pos_fp32 = pos.to(dtype=torch.float32, device=h.device)
            if pos_fp32.size(-1) == 2:
                pos_fp32 = F.pad(pos_fp32, (0, 1), "constant", 0.0)
            elif pos_fp32.size(-1) > 3:
                pos_fp32 = pos_fp32[:, :3]

        agg = self._integrate_edges(
            h,
            edge_index,
            pos_fp32,
            edge_attr=edge_attr,
            weights=weights,
            precomputed_kernel_inp=precomputed_kernel_inp,
            inv_deg=inv_deg,
        )

        out = self.linear_self(h) + self.linear_out(agg)
        out = self.dropout(self.act(out))
        return self.norm(h + out)


class GraphNeuralOperator(nn.Module):
    """Single GNO branch with k stacked GNO kernel layers (configurable: k = 2, 3, 4, 5, etc.)."""

    def __init__(
        self,
        hidden_dim,
        num_layers=NUM_GNO_LAYERS,
        edge_dim=EDGE_DIM,
        coord_dim=3,
        dropout=DROPOUT,
        use_checkpoint=USE_GRAD_CHECKPOINT,
        edge_chunk_size=GNO_EDGE_CHUNK_SIZE,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            GNOLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                coord_dim=coord_dim,
                dropout=dropout,
                edge_chunk_size=edge_chunk_size,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h,
        edge_index,
        pos,
        edge_attr=None,
        weights=None,
        precomputed_kernel_inp=None,
        inv_deg=None,
    ):
        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled() and h.requires_grad:
                h = checkpoint(
                    layer,
                    h,
                    edge_index,
                    pos,
                    edge_attr,
                    weights,
                    precomputed_kernel_inp,
                    inv_deg,
                    use_reentrant=False,
                )
            else:
                h = layer(
                    h,
                    edge_index,
                    pos,
                    edge_attr=edge_attr,
                    weights=weights,
                    precomputed_kernel_inp=precomputed_kernel_inp,
                    inv_deg=inv_deg,
                )
        return h


class ParallelGraphNeuralOperator(nn.Module):
    """Parallel multi-branch Graph Neural Operator block:
                    latent (from enc)
                           |
        ---------------------------------------
        |                  |                  |
    GNO(k_1 layers)    GNO(k_2 layers)    GNO(k_3 layers) ...
        ---------------------------------------
                           |
                     fused output (to dec)

    Both `num_branches` and `num_layers` per branch (int or list like [2,3,4] or [4,5,6])
    are completely configurable from config.py.
    """

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_branches=NUM_GNO_BRANCHES,
        num_layers=NUM_GNO_LAYERS,
        edge_dim=EDGE_DIM,
        coord_dim=3,
        dropout=DROPOUT,
        fusion=GNO_FUSION,
        use_checkpoint=USE_GRAD_CHECKPOINT,
        edge_chunk_size=GNO_EDGE_CHUNK_SIZE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_branches = max(1, int(num_branches))
        self.num_layers = num_layers
        self.fusion = fusion.lower()

        # Support either uniform int k layers per branch or a list of layer counts per branch
        if isinstance(num_layers, (list, tuple)):
            branch_depths = [int(k) for k in num_layers]
            self.num_branches = len(branch_depths)
        else:
            branch_depths = [int(num_layers)] * self.num_branches
        self.branch_depths = branch_depths

        self.branches = nn.ModuleList([
            GraphNeuralOperator(
                hidden_dim=hidden_dim,
                num_layers=k,
                edge_dim=edge_dim,
                coord_dim=coord_dim,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                edge_chunk_size=edge_chunk_size,
            )
            for k in branch_depths
        ])

        if self.fusion == "concat" and self.num_branches > 1:
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * self.num_branches, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, pos, edge_attr=None, weights=None):
        # Precompute shared edge kernel inputs and inverse node degrees ONCE for all parallel GNO layers
        pos_fp32 = pos.to(dtype=torch.float32, device=h.device)
        if pos_fp32.size(-1) == 2:
            pos_fp32 = F.pad(pos_fp32, (0, 1), "constant", 0.0)
        elif pos_fp32.size(-1) > 3:
            pos_fp32 = pos_fp32[:, :3]

        src, dst = edge_index
        num_nodes = h.size(0)

        inv_deg = None
        if weights is None:
            deg = torch.bincount(dst, minlength=num_nodes).to(dtype=torch.float32).unsqueeze(-1)
            inv_deg = deg.clamp_min(1.0).reciprocal()

        if h.is_cuda and torch.is_autocast_enabled():
            target_dtype = (
                torch.get_autocast_dtype("cuda")
                if hasattr(torch, "get_autocast_dtype")
                else torch.get_autocast_gpu_dtype()
            )
        else:
            target_dtype = h.dtype
        precomputed_kernel_inp = _prepare_gno_edge_inputs(
            pos_fp32, src, dst, edge_attr, self.edge_dim, target_dtype
        )

        branch_outputs = [
            branch(
                h,
                edge_index,
                pos_fp32,
                edge_attr=edge_attr,
                weights=weights,
                precomputed_kernel_inp=precomputed_kernel_inp,
                inv_deg=inv_deg,
            )
            for branch in self.branches
        ]

        if self.num_branches == 1:
            merged = branch_outputs[0]
        elif self.fusion == "concat":
            merged = torch.cat(branch_outputs, dim=-1)
        elif self.fusion == "sum":
            merged = torch.stack(branch_outputs, dim=0).sum(dim=0)
        elif self.fusion == "mean":
            merged = torch.stack(branch_outputs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported GNO_FUSION mode: {self.fusion}")

        fused = self.fusion_proj(merged)
        return self.norm(h + fused)


############################################
# 4. GRAPH DECODER (dec)
############################################

class GraphDecoderLayer(nn.Module):
    """Decoder layer transforming fused operator features."""

    def __init__(self, hidden_dim, edge_dim, dropout=DROPOUT):
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
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(x + out)


class GraphDecoder(nn.Module):
    """Decodes parallel GNO representations into decoded graph latent representations."""

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_DECODER_LAYERS,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        use_checkpoint=USE_GRAD_CHECKPOINT,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [GraphDecoderLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _edge_attr(self, edge_index, edge_attr, x):
        if edge_attr is not None:
            return edge_attr.to(device=x.device, dtype=x.dtype)
        return x.new_zeros((edge_index.size(1), self.edge_dim))

    def forward(self, latent, edge_index, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, latent)
        x = latent
        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
                x = checkpoint(layer, x, edge_index, edge_attr, use_reentrant=False)
            else:
                x = layer(x, edge_index, edge_attr)
        return self.proj(x)


############################################
# 5. POST-DECODER GNN HEAD (gnn)
############################################

class GraphGNNLayer(nn.Module):
    """Post-decoder GNN message-passing layer for local surface field refinement."""

    def __init__(self, hidden_dim, edge_dim, dropout=DROPOUT):
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
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(x + out)


class GraphRefinementGNN(nn.Module):
    """Final GNN stage (dec -> gnn) that refines decoded features on the mesh graph and predicts Cp."""

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_GNN_LAYERS,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        use_checkpoint=USE_GRAD_CHECKPOINT,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList(
            [GraphGNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _edge_attr(self, edge_index, edge_attr, x):
        if edge_attr is not None:
            return edge_attr.to(device=x.device, dtype=x.dtype)
        return x.new_zeros((edge_index.size(1), self.edge_dim))

    def forward(self, x, edge_index, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, x)
        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
                x = checkpoint(layer, x, edge_index, edge_attr, use_reentrant=False)
            else:
                x = layer(x, edge_index, edge_attr)
        return self.readout(x)


############################################
# 6. FULL ARCHITECTURE:
#    ENC -> PARALLEL GNOs (B x K) -> DEC -> GNN
############################################

class EncoderGNODecoder(nn.Module):
    """Complete modular architecture:
                    enc
                     |
        ----------------------------
        |                          |
    GNO(k_1 layers)          GNO(k_B layers)   [Configurable B parallel branches]
        ----------------------------
                     |
                 dec -> gnn
    """

    def __init__(
        self,
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_gno_branches=NUM_GNO_BRANCHES,
        num_gno_layers=NUM_GNO_LAYERS,
        gno_fusion=GNO_FUSION,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_gnn_layers=NUM_GNN_LAYERS,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        fourier_bands=FOURIER_BANDS,
        use_checkpoint=USE_GRAD_CHECKPOINT,
        edge_chunk_size=GNO_EDGE_CHUNK_SIZE,
        **kwargs,
    ):
        super().__init__()

        # Support flexible keyword aliases
        if "num_parallel_gno" in kwargs and kwargs["num_parallel_gno"] is not None:
            num_gno_branches = kwargs["num_parallel_gno"]
        elif "gno_branches" in kwargs and kwargs["gno_branches"] is not None:
            num_gno_branches = kwargs["gno_branches"]

        if "gno_layers" in kwargs and kwargs["gno_layers"] is not None:
            num_gno_layers = kwargs["gno_layers"]

        if isinstance(num_gno_layers, (list, tuple)):
            num_gno_branches = len(num_gno_layers)

        self.num_encoder_layers = num_encoder_layers
        self.num_gno_branches = num_gno_branches
        self.num_gno_layers = num_gno_layers
        self.gno_fusion = gno_fusion
        self.num_decoder_layers = num_decoder_layers
        self.num_gnn_layers = num_gnn_layers
        self.use_checkpoint = use_checkpoint

        # 1. Encoder (enc)
        self.encoder = GraphEncoder(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            fourier_bands=fourier_bands,
            use_checkpoint=use_checkpoint,
        )

        # 2. Parallel Multi-Branch GNOs: B parallel blocks, each with k GNO layers
        self.gno = ParallelGraphNeuralOperator(
            hidden_dim=hidden_dim,
            num_branches=num_gno_branches,
            num_layers=num_gno_layers,
            edge_dim=edge_dim,
            coord_dim=3,
            dropout=dropout,
            fusion=gno_fusion,
            use_checkpoint=use_checkpoint,
            edge_chunk_size=edge_chunk_size,
        )

        # 3. Decoder (dec)
        self.decoder = GraphDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # 4. Post-Decoder GNN (gnn)
        self.gnn = GraphRefinementGNN(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_gnn_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

    def encode(self, data, edge_index=None, edge_attr=None, xyz=None):
        return self.encoder(data, edge_index=edge_index, edge_attr=edge_attr, xyz=xyz)

    def forward(self, data, edge_index=None, edge_attr=None, pos=None):
        if hasattr(data, "edge_index"):
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            if hasattr(data, "xyz") and data.xyz is not None and data.xyz.size(-1) == 3:
                pos = data.xyz
            elif hasattr(data, "pos") and data.pos is not None and data.pos.size(-1) == 3:
                pos = data.pos
            elif hasattr(data, "x") and data.x is not None and data.x.size(-1) >= 3:
                pos = data.x[:, :3]
            elif hasattr(data, "pos") and data.pos is not None:
                pos = data.pos
            else:
                pos = None
            latent = self.encoder(data)
        else:
            if pos is None:
                pos = data[:, :3] if (hasattr(data, "shape") and data.shape[-1] >= 3) else data
            latent = self.encoder(data, edge_index=edge_index, edge_attr=edge_attr)

        gno_out = self.gno(latent, edge_index, pos, edge_attr)
        dec_out = self.decoder(gno_out, edge_index, edge_attr)
        return self.gnn(dec_out, edge_index, edge_attr)


# Aliases for backward compatibility
EncoderDecoder = EncoderGNODecoder
GAGNO = EncoderGNODecoder
GIGNO = EncoderGNODecoder
