"""Single-geometry inference engine for GI-GNO."""

from __future__ import annotations

import configparser
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from config import (
    C_REF,
    CHECKPOINT,
    DEVICE,
    DROPOUT,
    EDGE_DIM,
    FOURIER_BANDS,
    GEOM_PARAM_FILE,
    GNO_FUSION,
    HIDDEN_DIM,
    KNN_K,
    NODE_IN_DIM,
    NORM_STATS_FILE,
    NUM_ENCODER_LAYERS,
    NUM_GNO_BRANCHES,
    NUM_GNO_LAYERS,
    NUM_DECODER_LAYERS,
    NUM_GNN_LAYERS,
    OUT_DIM,
    OUT_VTK_DIR,
    S_REF,
    X_REF,
    Z_REF,
    setup_run_logger,
)
from model import EncoderGNODecoder, EncoderDecoder, GAGNO

OUTPUT_DIR = OUT_VTK_DIR if OUT_VTK_DIR else "/tmp/output"
FLOW_KEYS = ("alt_kft", "Re", "M_inf", "alpha_deg", "beta_deg")
GEOM_KEYS = ("B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3")
FEATURE_LAYOUT = (
    "xyz[0:3]",
    "flow[3:8]",
    "geom[8:18]",
    "curvature[18:19]",
    "normal[19:22]",
    "local_area[22:23]",
    "edge_stats[23:27]",
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class InferenceState:
    """Cached model and static inference assets."""

    model: GAGNO
    device: torch.device
    x_mean: torch.Tensor
    x_std: torch.Tensor
    cp_mean: torch.Tensor
    cp_std: torch.Tensor
    geom_config: configparser.ConfigParser


@dataclass
class PreparedMesh:
    """Mesh tensors required by the unchanged inference pipeline."""

    mesh: pv.DataSet
    vertices: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    x_fp32: torch.Tensor
    cp_true: torch.Tensor | None
    center: np.ndarray
    scale: float
    cf_true: torch.Tensor | None = None


_STATE: InferenceState | None = None


def load_model() -> InferenceState:
    """Load checkpoint, normalization stats, geometry config, and device once."""

    global _STATE
    if _STATE is not None:
        return _STATE

    _validate_asset_paths()
    device = torch.device(DEVICE)
    LOGGER.info("Loading GI-GNO model on %s", device)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    raw_state_dict = _extract_state_dict(checkpoint)
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in raw_state_dict.items()
    }

    # Automatically detect trained parallel GNO branches & per-branch layer counts from checkpoint
    detected_gno_branches = 0
    while any(f"gno.branches.{detected_gno_branches}." in k for k in state_dict):
        detected_gno_branches += 1

    detected_gno_layers: int | list[int] = 0
    if detected_gno_branches > 0:
        branch_layer_counts = []
        for b in range(detected_gno_branches):
            b_layers = 0
            while any(f"gno.branches.{b}.layers.{b_layers}." in k for k in state_dict):
                b_layers += 1
            branch_layer_counts.append(b_layers)
        if all(k > 0 for k in branch_layer_counts):
            detected_gno_layers = branch_layer_counts
    else:
        single_layers = 0
        while any(f"gno.layers.{single_layers}." in k for k in state_dict):
            single_layers += 1
        detected_gno_layers = single_layers

    detected_enc_layers = 0
    while any(f"encoder.layers.{detected_enc_layers}." in k for k in state_dict):
        detected_enc_layers += 1

    detected_dec_layers = 0
    while any(f"decoder.layers.{detected_dec_layers}." in k for k in state_dict):
        detected_dec_layers += 1

    detected_gnn_layers = 0
    while any(f"gnn.layers.{detected_gnn_layers}." in k for k in state_dict):
        detected_gnn_layers += 1

    detected_hidden_dim = HIDDEN_DIM
    if "encoder.input_proj.0.weight" in state_dict:
        detected_hidden_dim = int(state_dict["encoder.input_proj.0.weight"].shape[0])

    gno_branches = detected_gno_branches if detected_gno_branches > 0 else NUM_GNO_BRANCHES
    gno_layers = (
        detected_gno_layers
        if (isinstance(detected_gno_layers, list) or detected_gno_layers > 0)
        else NUM_GNO_LAYERS
    )
    enc_layers = detected_enc_layers if detected_enc_layers > 0 else NUM_ENCODER_LAYERS
    dec_layers = detected_dec_layers if detected_dec_layers > 0 else NUM_DECODER_LAYERS
    gnn_layers = detected_gnn_layers if detected_gnn_layers > 0 else NUM_GNN_LAYERS

    LOGGER.info(
        "Configured model depth: hidden_dim=%d, %d encoder, %d parallel GNO branches (layers=%s), %d decoder, %d GNN layers",
        detected_hidden_dim,
        enc_layers,
        gno_branches,
        gno_layers,
        dec_layers,
        gnn_layers,
    )

    model = EncoderGNODecoder(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=detected_hidden_dim,
        out_dim=OUT_DIM,
        num_encoder_layers=enc_layers,
        num_gno_branches=gno_branches,
        num_gno_layers=gno_layers,
        gno_fusion=GNO_FUSION,
        num_decoder_layers=dec_layers,
        num_gnn_layers=gnn_layers,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        fourier_bands=FOURIER_BANDS,
        use_checkpoint=False,
    ).to(device)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        LOGGER.warning(
            "Missing keys while loading checkpoint: %s",
            incompatible.missing_keys[:10],
        )
    if incompatible.unexpected_keys:
        LOGGER.warning(
            "Unexpected keys while loading checkpoint: %s",
            incompatible.unexpected_keys[:10],
        )
    model.eval()

    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu", weights_only=False)
    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)

    _STATE = InferenceState(
        model=model,
        device=device,
        x_mean=norm_stats["x_mean"].float(),
        x_std=norm_stats["x_std"].float(),
        cp_mean=norm_stats["cp_mean"].float(),
        cp_std=norm_stats["cp_std"].float(),
        geom_config=geom_config,
    )
    LOGGER.info("Inference assets loaded")
    return _STATE


def compute_force_coefficients(
    mesh: pv.DataSet,
    cp_values: torch.Tensor,
    alpha_deg: float,
    beta_deg: float = 0.0,
    cf_values: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Compute CL, CD, and CMy from the predicted Cp (and optional Cf) distribution."""

    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    surf = surf.compute_normals(cell_normals=True, point_normals=False)
    surf = surf.compute_cell_sizes()

    raw_points = np.array(surf.points, dtype=np.float32)
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    cell_centers = raw_points[faces].mean(axis=1)

    raw_y = raw_points[:, 1]
    y_min, y_max = float(np.min(raw_y)), float(np.max(raw_y))
    symmetry_factor = 2.0 if (abs(y_min) < 1e-4 or abs(y_max) < 1e-4) else 1.0

    areas = np.array(surf.cell_data["Area"], dtype=np.float32) * symmetry_factor
    normals = np.array(surf.cell_data["Normals"], dtype=np.float32).copy()

    z_coords = cell_centers[:, 2]
    upper_mask = z_coords > np.median(z_coords)
    if np.any(upper_mask) and np.mean(normals[upper_mask, 2]) < 0:
        normals = -normals

    cp_arr = cp_values.detach().cpu().numpy()
    if cp_arr.ndim == 2 and cp_arr.shape[1] == 4:
        cp_point = cp_arr[:, 0]
        cf_point = cp_arr[:, 1:4]
    else:
        cp_point = cp_arr.reshape(-1)
        cf_point = cf_values.detach().cpu().numpy() if cf_values is not None else None

    cp_cell = cp_point[faces].mean(axis=1)
    if cf_point is not None:
        cf_cell = cf_point[faces].mean(axis=1)
    else:
        cf_cell = np.zeros((faces.shape[0], 3), dtype=np.float32)

    dF_total = (-cp_cell[:, None] * normals + cf_cell) * areas[:, None]
    F_total = dF_total.sum(axis=0)
    CX, CY, CZ = F_total[0] / S_REF, F_total[1] / S_REF, F_total[2] / S_REF

    cd = CX * np.cos(alpha) * np.cos(beta) + CY * np.sin(beta) + CZ * np.sin(alpha) * np.cos(beta)
    cl = -CX * np.sin(alpha) + CZ * np.cos(alpha)
    cmy = ((cell_centers[:, 2] - Z_REF) * dF_total[:, 0] - (cell_centers[:, 0] - X_REF) * dF_total[:, 2]).sum() / (S_REF * C_REF)
    return float(cl), float(cd), float(cmy)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply the training-time feature normalization."""

    return (x - mean) / std


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float().reshape(-1)
    if tensor.numel() == 0:
        return f"{name}: empty"
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return (
        f"{name}: min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={std.item():.6e}"
    )


def _log_vector(name: str, tensor: torch.Tensor, max_items: int = 27) -> None:
    flat = tensor.detach().float().reshape(-1)
    values = [f"{v:.6e}" for v in flat[:max_items].tolist()]
    suffix = " ..." if flat.numel() > max_items else ""
    LOGGER.info("%s shape=%s values=[%s%s]", name, tuple(tensor.shape), ", ".join(values), suffix)


def _log_first_vertices(label: str, vertices: torch.Tensor) -> None:
    count = min(3, vertices.size(0))
    LOGGER.info("%s first %d vertices: %s", label, count, vertices[:count].detach().cpu().tolist())


def _manual_centered_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denom = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
    if denom.item() == 0:
        return 0.0
    return (torch.sum(pred_centered * target_centered) / (denom + 1e-12)).item()


def _validate_asset_paths() -> None:
    LOGGER.info("Checkpoint path: %s", CHECKPOINT)
    LOGGER.info("Normalization stats path: %s", NORM_STATS_FILE)
    LOGGER.info("Geometry config path: %s", GEOM_PARAM_FILE)
    required = {
        "checkpoint": CHECKPOINT,
        "normalization stats": NORM_STATS_FILE,
        "geometry file": GEOM_PARAM_FILE,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing model asset(s): " + "; ".join(missing))


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
        if any(
            key.startswith("encoder") or key.startswith("decoder") or key.startswith("gno")
            for key in checkpoint
        ):
            return checkpoint
    return checkpoint


def _validate_request(
    vtk_path: str,
    input_json: dict[str, Any],
    geom_config: configparser.ConfigParser,
) -> dict[str, Any]:
    if not vtk_path:
        raise ValueError("Uploaded file path is required.")

    path = Path(vtk_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded VTK file does not exist: {vtk_path}")
    if path.suffix.lower() != ".vtk":
        raise ValueError("Uploaded geometry must be a .vtk file.")

    if not isinstance(input_json, dict):
        raise ValueError("Flow parameters must be provided as a JSON object.")

    required_keys = ("geom_name", *FLOW_KEYS)
    missing = [key for key in required_keys if key not in input_json]
    if missing:
        raise ValueError("Missing required JSON key(s): " + ", ".join(missing))

    geom_name = str(input_json["geom_name"])
    if geom_name not in geom_config.sections():
        raise ValueError(f"geom_name '{geom_name}' was not found in geom.ini.")

    validated = {"geom_name": geom_name}
    for key in FLOW_KEYS:
        try:
            validated[key] = float(input_json[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"JSON key '{key}' must be numeric.") from exc
    return validated


def _build_mesh_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        dim=0,
    )
    return to_undirected(edges.T)


def _build_knn_edges(vertices: torch.Tensor, k: int) -> torch.Tensor:
    num_nodes = vertices.size(0)
    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k, num_nodes - 1)
    try:
        from scipy.spatial import cKDTree
        pts = vertices.detach().cpu().numpy()
        tree = cKDTree(pts)
        _, indices = tree.query(pts, k=k + 1, workers=-1)
        nn_idx = indices[:, 1:]
        src = np.repeat(np.arange(num_nodes), k)
        dst = nn_idx.reshape(-1)
        edges = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        return to_undirected(edges)
    except Exception:
        pass

    edge_chunks = []
    chunk_size = 4096
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        dist = torch.cdist(vertices[start:end], vertices)
        local = torch.arange(end - start)
        dist[local, torch.arange(start, end)] = float("inf")
        nn_idx = dist.topk(k, largest=False).indices
        src = torch.arange(start, end).unsqueeze(1).expand(-1, k)
        edge_chunks.append(torch.stack([src.reshape(-1), nn_idx.reshape(-1)], dim=0))

    return to_undirected(torch.cat(edge_chunks, dim=1))


def _merge_edge_indices(*edge_indices: torch.Tensor) -> torch.Tensor:
    edges = [e for e in edge_indices if e.numel() > 0]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    cat_edges = torch.cat(edges, dim=1)
    max_node = cat_edges.max().item() + 1
    edge_hash = cat_edges[0] * max_node + cat_edges[1]
    unique_hash = torch.unique(edge_hash)
    edge_index = torch.stack([unique_hash // max_node, unique_hash % max_node], dim=0)
    return edge_index.contiguous()


def _compute_edge_attr(vertices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    rel_xyz = vertices[dst] - vertices[src]
    distance = torch.sqrt(torch.sum(rel_xyz * rel_xyz, dim=1, keepdim=True) + 1e-12)
    direction = rel_xyz / (distance + 1e-8)
    edge_length = distance
    return torch.cat([edge_length, rel_xyz, distance, direction], dim=1)


def _compute_vertex_normals(
    num_nodes: int,
    faces: torch.Tensor,
    cell_normals: torch.Tensor,
) -> torch.Tensor:
    vertex_normals = torch.zeros((num_nodes, 3), dtype=torch.float32)
    repeated_normals = cell_normals.repeat_interleave(3, dim=0)
    vertex_normals.index_add_(0, faces.reshape(-1), repeated_normals)
    return F.normalize(vertex_normals, p=2, dim=1, eps=1e-8)


def _compute_local_area(
    num_nodes: int,
    faces: torch.Tensor,
    cell_areas: torch.Tensor,
) -> torch.Tensor:
    local_area = torch.zeros((num_nodes, 1), dtype=torch.float32)
    area_share = (cell_areas / 3.0).repeat_interleave(3).unsqueeze(1)
    local_area.index_add_(0, faces.reshape(-1), area_share)
    return local_area


def _compute_curvature(vertices: torch.Tensor, mesh_edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = mesh_edge_index
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros((vertices.size(0), 1), dtype=vertices.dtype)
    neighbor_sum.index_add_(0, src, vertices[dst])
    neighbor_count.index_add_(0, src, torch.ones_like(neighbor_count[src]))
    neighbor_mean = neighbor_sum / (neighbor_count + 1e-8)
    return torch.linalg.norm(vertices - neighbor_mean, dim=1, keepdim=True)


def _compute_edge_length_stats(
    vertices: torch.Tensor,
    mesh_edge_index: torch.Tensor,
) -> torch.Tensor:
    src, dst = mesh_edge_index
    lengths = torch.linalg.norm(vertices[dst] - vertices[src], dim=1)
    num_nodes = vertices.size(0)

    mean = torch.zeros(num_nodes, dtype=vertices.dtype)
    count = torch.zeros(num_nodes, dtype=vertices.dtype)
    min_len = torch.full((num_nodes,), float("inf"), dtype=vertices.dtype)
    max_len = torch.zeros(num_nodes, dtype=vertices.dtype)

    mean.index_add_(0, src, lengths)
    count.index_add_(0, src, torch.ones_like(lengths))
    min_len.scatter_reduce_(0, src, lengths, reduce="amin", include_self=True)
    max_len.scatter_reduce_(0, src, lengths, reduce="amax", include_self=True)

    mean = mean / (count + 1e-8)
    sq_diff = (lengths - mean[src]) ** 2
    var = torch.zeros(num_nodes, dtype=vertices.dtype)
    var.index_add_(0, src, sq_diff)
    std = torch.sqrt(var / (count + 1e-8) + 1e-12)
    min_len = torch.where(torch.isfinite(min_len), min_len, torch.zeros_like(min_len))
    return torch.stack([mean, min_len, max_len, std], dim=1)


def _assert_preprocessing_contract(
    prepared: PreparedMesh,
    flow_params: torch.Tensor,
    geom_params: torch.Tensor,
) -> None:
    x = prepared.x_fp32
    vertices = prepared.vertices

    assert x.shape[1] == NODE_IN_DIM, (
        f"Feature width mismatch: got {x.shape[1]}, expected NODE_IN_DIM={NODE_IN_DIM}. "
        f"Expected order: {FEATURE_LAYOUT}"
    )
    assert prepared.edge_attr.shape[1] == EDGE_DIM, (
        f"Edge feature width mismatch: got {prepared.edge_attr.shape[1]}, expected EDGE_DIM={EDGE_DIM}."
    )
    assert torch.allclose(x[:, :3], vertices, atol=1e-6), "Feature order mismatch: x[:, 0:3] must be xyz."
    assert torch.allclose(x[:, 3:8], flow_params, atol=1e-6), "Feature order mismatch: x[:, 3:8] must be flow."
    assert torch.allclose(x[:, 8:18], geom_params, atol=1e-6), "Feature order mismatch: x[:, 8:18] must be geom."
    assert torch.allclose(x[:, 22:23], torch.clamp(x[:, 22:23], min=0.0), atol=1e-6), (
        "Feature order mismatch: local_area should be non-negative at x[:, 22:23]."
    )
    assert torch.allclose(prepared.edge_attr, _compute_edge_attr(vertices, prepared.edge_index), atol=1e-5), (
        "edge_attr does not match edge_index and normalized xyz."
    )
    assert torch.allclose(prepared.vertices, x[:, :3], atol=1e-6), "vertices must equal normalized xyz."
    assert torch.isfinite(x).all(), "Non-finite values found in inference x_fp32."
    assert torch.isfinite(prepared.edge_attr).all(), "Non-finite values found in inference edge_attr."
    assert vertices[:, 0].min() >= -1e-4, f"Vertices x-coordinate must start at >= 0.0, got {vertices[:, 0].min()}"
    assert vertices[:, 0].max() <= 1.0001, f"Vertices x-coordinate must end at <= 1.0, got {vertices[:, 0].max()}"


def _prepare_mesh(
    vtk_path: str,
    input_json: dict[str, Any],
    geom_config: configparser.ConfigParser,
) -> PreparedMesh:
    mesh = pv.read(vtk_path)
    mesh = mesh.connectivity(extraction_mode="largest")
    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    surf = surf.compute_normals(cell_normals=True, point_normals=False)
    surf = surf.compute_cell_sizes()

    vertices_np = surf.points.astype(np.float32)
    x_min = float(np.min(vertices_np[:, 0]))
    x_max = float(np.max(vertices_np[:, 0]))
    chord = max(x_max - x_min, 1e-8)

    nose_idx = int(np.argmin(vertices_np[:, 0]))
    nose_y = float(vertices_np[nose_idx, 1])
    nose_z = float(vertices_np[nose_idx, 2])

    vertices_np[:, 0] = (vertices_np[:, 0] - x_min) / chord
    vertices_np[:, 1] = (vertices_np[:, 1] - nose_y) / chord
    vertices_np[:, 2] = (vertices_np[:, 2] - nose_z) / chord
    vertices_np[:, 0] = np.clip(vertices_np[:, 0], 0.0, 1.0)
    vertices = torch.tensor(vertices_np, dtype=torch.float32)
    center = np.array([x_min, nose_y, nose_z], dtype=np.float32)
    scale = float(chord)

    faces_np = surf.faces.reshape(-1, 4)[:, 1:]
    faces = torch.tensor(faces_np, dtype=torch.long)
    raw_points = surf.points.astype(np.float32)
    cell_centers_np = raw_points[faces_np].mean(axis=1)
    normals_np = np.array(surf.cell_data["Normals"], dtype=np.float32).copy()
    z_coords = cell_centers_np[:, 2]
    upper_mask = z_coords > np.median(z_coords)
    if np.any(upper_mask) and np.mean(normals_np[upper_mask, 2]) < 0:
        normals_np = -normals_np
    cell_normals = torch.tensor(normals_np, dtype=torch.float32)
    normalized_cell_areas = 0.5 * torch.linalg.norm(
        torch.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
            dim=1,
        ),
        dim=1,
    )

    mesh_edge_index = _build_mesh_edges(faces)
    knn_edge_index = _build_knn_edges(vertices, KNN_K)
    edge_index = _merge_edge_indices(mesh_edge_index, knn_edge_index)
    edge_attr = _compute_edge_attr(vertices, edge_index)

    vertex_normals = _compute_vertex_normals(vertices.size(0), faces, cell_normals)
    local_area = _compute_local_area(vertices.size(0), faces, normalized_cell_areas)
    curvature = _compute_curvature(vertices, mesh_edge_index)
    edge_length_stats = _compute_edge_length_stats(vertices, mesh_edge_index)

    geom_name = input_json["geom_name"]
    geom_params = torch.tensor(
        [geom_config.getfloat(geom_name, key) for key in GEOM_KEYS],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(vertices.size(0), 1)

    flow_params = torch.tensor(
        [float(input_json[key]) for key in FLOW_KEYS],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(vertices.size(0), 1)

    x_fp32 = torch.cat(
        [
            vertices,
            flow_params,
            geom_params,
            curvature,
            vertex_normals,
            local_area,
            edge_length_stats,
        ],
        dim=1,
    )
    cp_true = None
    if "cp" in surf.point_data:
        cp_true = torch.tensor(surf.point_data["cp"], dtype=torch.float32)

    cf_true = None
    if all(k in surf.point_data for k in ("cf_x", "cf_y", "cf_z")):
        cf_true = torch.tensor(
            np.stack(
                [surf.point_data["cf_x"], surf.point_data["cf_y"], surf.point_data["cf_z"]],
                axis=1,
            ),
            dtype=torch.float32,
        )
    elif "cf" in surf.point_data and np.asarray(surf.point_data["cf"]).ndim == 2 and np.asarray(surf.point_data["cf"]).shape[1] == 3:
        cf_true = torch.tensor(surf.point_data["cf"], dtype=torch.float32)

    prepared = PreparedMesh(
        mesh=surf,
        vertices=vertices,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_fp32=x_fp32,
        cp_true=cp_true,
        center=center,
        scale=float(scale),
        cf_true=cf_true,
    )
    _assert_preprocessing_contract(prepared, flow_params, geom_params)

    LOGGER.info("Inference feature layout: %s", " | ".join(FEATURE_LAYOUT))
    LOGGER.info("Inference vertex normalization center=%s scale=%.6e", center.tolist(), float(scale))
    _log_first_vertices("Inference VTK normalized", vertices)
    LOGGER.info(
        "Inference tensor shapes: x=%s xyz=%s pos=%s edge_index=%s edge_attr=%s",
        tuple(x_fp32.shape),
        tuple(vertices.shape),
        tuple(vertices.shape),
        tuple(edge_index.shape),
        tuple(edge_attr.shape),
    )
    LOGGER.info("%s", _tensor_stats("Inference edge_attr", edge_attr))
    LOGGER.info("%s", _tensor_stats("Inference x_fp32", x_fp32))

    return prepared


def _compute_metrics(
    cp_pred: torch.Tensor,
    cp_true: torch.Tensor | None,
    cp_pred_norm: torch.Tensor | None = None,
    cp_mean: torch.Tensor | None = None,
    cp_std: torch.Tensor | None = None,
) -> dict[str, float | None]:
    if cp_true is None:
        return _empty_metrics()

    diff = cp_pred - cp_true
    mse = torch.mean(diff ** 2).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(cp_true) + 1e-8)).item()
    max_abs_error = torch.max(torch.abs(diff)).item()

    corr_after_denorm = _manual_centered_corr(cp_pred, cp_true)
    corr_before_denorm = None
    if cp_pred_norm is not None and cp_mean is not None and cp_std is not None:
        cp_true_norm = normalize(cp_true, cp_mean, cp_std)
        corr_before_denorm = _manual_centered_corr(cp_pred_norm, cp_true_norm)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "rel_l2": float(rel_l2),
        "corr": float(corr_after_denorm),
        "corr_before_denorm": float(corr_before_denorm) if corr_before_denorm is not None else None,
        "corr_after_denorm": float(corr_after_denorm),
        "max_abs_error": float(max_abs_error),
    }


def _empty_metrics() -> dict[str, None]:
    return {
        "rmse": None,
        "mae": None,
        "mse": None,
        "rel_l2": None,
        "corr": None,
        "corr_before_denorm": None,
        "corr_after_denorm": None,
        "max_abs_error": None,
    }


def _save_prediction_vtk(
    vtk_path: str,
    mesh: pv.DataSet,
    cp_pred: torch.Tensor,
    cp_true: torch.Tensor | None,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"predicted_{Path(vtk_path).name}"

    out_mesh = mesh.copy()
    out_mesh.point_data["Cp_pred"] = cp_pred.detach().cpu().numpy()
    if cp_true is not None:
        diff = cp_pred - cp_true
        out_mesh.point_data["Cp_error"] = diff.detach().cpu().numpy()

    out_mesh.save(output_path)
    return str(output_path)


def _write_report(
    output_vtk: str,
    vtk_path: str,
    input_json: dict[str, Any],
    metrics: dict[str, float | None],
    forces: dict[str, float | None],
    elapsed: float,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = Path(OUTPUT_DIR) / "simulation_report.txt"
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        "GI-GNO CFD Simulation Report",
        "=" * 30,
        f"Geometry: {Path(vtk_path).name}",
        f"Predicted VTK: {output_vtk}",
        f"Execution time: {elapsed:.6f} seconds",
        f"Timestamp: {timestamp}",
        "",
        "Flow conditions",
        "-" * 15,
        f"geom_name: {input_json['geom_name']}",
        f"alt_kft: {input_json['alt_kft']}",
        f"Re: {input_json['Re']}",
        f"M_inf: {input_json['M_inf']}",
        f"alpha_deg: {input_json['alpha_deg']}",
        f"beta_deg: {input_json['beta_deg']}",
        "",
        "Metrics",
        "-" * 7,
        f"RMSE: {_format_value(metrics['rmse'])}",
        f"MAE: {_format_value(metrics['mae'])}",
        f"MSE: {_format_value(metrics['mse'])}",
        f"Relative L2: {_format_value(metrics['rel_l2'])}",
        f"Correlation: {_format_value(metrics['corr'])}",
        f"Correlation before denorm: {_format_value(metrics['corr_before_denorm'])}",
        f"Correlation after denorm: {_format_value(metrics['corr_after_denorm'])}",
        f"Maximum Error: {_format_value(metrics['max_abs_error'])}",
        "",
        "Force coefficients",
        "-" * 18,
        f"CL_pred: {_format_value(forces['CL'])}",
        f"CD_pred: {_format_value(forces['CD'])}",
        f"CMy_pred: {_format_value(forces.get('CMy'))}",
        f"CL_true (GT Cp int): {_format_value(forces.get('CL_true'))}",
        f"CD_true (GT Cp int): {_format_value(forces.get('CD_true'))}",
        f"CMy_true (GT Cp int): {_format_value(forces.get('CMy_true'))}",
        f"CL %Error: {_format_value(forces.get('CL_pct_error'))}",
        f"CD %Error: {_format_value(forces.get('CD_pct_error'))}",
        f"CMy %Error: {_format_value(forces.get('CMy_pct_error'))}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6e}"


def run_inference(vtk_path: str, input_json: dict[str, Any]) -> dict[str, Any]:
    """Run prediction for one uploaded VTK and return structured results."""

    log_path = setup_run_logger("inference", reuse_active=True)
    start_time = time.time()
    try:
        LOGGER.info("Starting inference for %s (log_file=%s)", vtk_path, log_path)
        state = load_model()
        inputs = _validate_request(vtk_path, input_json, state.geom_config)
        prepared = _prepare_mesh(vtk_path, inputs, state.geom_config)

        _log_vector("x_mean", state.x_mean)
        _log_vector("x_std", state.x_std)
        _log_vector("cp_mean", state.cp_mean)
        _log_vector("cp_std", state.cp_std)
        assert state.x_mean.numel() == NODE_IN_DIM, (
            f"x_mean width {state.x_mean.numel()} does not match NODE_IN_DIM={NODE_IN_DIM}."
        )
        assert state.x_std.numel() == NODE_IN_DIM, (
            f"x_std width {state.x_std.numel()} does not match NODE_IN_DIM={NODE_IN_DIM}."
        )

        x = normalize(prepared.x_fp32, state.x_mean, state.x_std)
        assert torch.isfinite(x).all(), "Non-finite values found after inference feature normalization."
        LOGGER.info("%s", _tensor_stats("Inference x_norm", x))
        data = Data(
            x=x,
            xyz=prepared.vertices,
            edge_index=prepared.edge_index,
            edge_attr=prepared.edge_attr,
            pos=prepared.vertices,
        ).to(state.device)
        assert torch.allclose(data.pos.cpu(), data.xyz.cpu(), atol=1e-6), (
            "Inference Data.pos must equal Data.xyz."
        )

        with torch.no_grad():
            raw_pred = state.model(data).cpu()
            if raw_pred.dim() == 2 and raw_pred.size(-1) == 4:
                cp_pred_norm = raw_pred[:, 0]
                cf_pred = raw_pred[:, 1:4]
            else:
                cp_pred_norm = raw_pred.reshape(-1)
                cf_pred = prepared.cf_true
            cp_pred = cp_pred_norm * state.cp_std.reshape(-1) + state.cp_mean.reshape(-1)

        LOGGER.info("%s", _tensor_stats("pred_norm", cp_pred_norm))
        LOGGER.info("%s", _tensor_stats("pred", cp_pred))
        if prepared.cp_true is not None:
            cp_true_norm = normalize(prepared.cp_true, state.cp_mean, state.cp_std)
            corr_before_denorm = _manual_centered_corr(cp_pred_norm, cp_true_norm)
            corr_after_denorm = _manual_centered_corr(cp_pred, prepared.cp_true)
            LOGGER.info("corr_before_denorm=%.6f", corr_before_denorm)
            LOGGER.info("corr_after_denorm=%.6f", corr_after_denorm)
            LOGGER.info("%s", _tensor_stats("GT", prepared.cp_true))
            LOGGER.info("%s", _tensor_stats("Prediction", cp_pred))

        metrics = _compute_metrics(
            cp_pred,
            prepared.cp_true,
            cp_pred_norm=cp_pred_norm,
            cp_mean=state.cp_mean,
            cp_std=state.cp_std,
        )
        cl_pred, cd_pred, cmy_pred = compute_force_coefficients(
            prepared.mesh,
            cp_pred,
            inputs["alpha_deg"],
            beta_deg=inputs["beta_deg"],
            cf_values=cf_pred,
        )

        cl_true, cd_true, cmy_true = None, None, None
        cl_pct_err, cd_pct_err, cmy_pct_err = None, None, None
        if prepared.cp_true is not None:
            cl_true, cd_true, cmy_true = compute_force_coefficients(
                prepared.mesh,
                prepared.cp_true,
                inputs["alpha_deg"],
                beta_deg=inputs["beta_deg"],
                cf_values=prepared.cf_true,
            )
            eps = 1e-3
            cl_pct_err = abs(cl_pred - cl_true) / (abs(cl_true) + eps) * 100.0
            cd_pct_err = abs(cd_pred - cd_true) / (abs(cd_true) + eps) * 100.0
            cmy_pct_err = abs(cmy_pred - cmy_true) / (abs(cmy_true) + eps) * 100.0
            LOGGER.info(
                "Forces Pred: CL=%.6f, CD=%.6f, CMy=%.6f | GT Cp Int: CL=%.6f, CD=%.6f, CMy=%.6f | %%Err: CL=%.2f%%, CD=%.2f%%, CMy=%.2f%%",
                cl_pred, cd_pred, cmy_pred,
                cl_true, cd_true, cmy_true,
                cl_pct_err, cd_pct_err, cmy_pct_err,
            )
        else:
            LOGGER.info("Forces Pred: CL=%.6f, CD=%.6f, CMy=%.6f", cl_pred, cd_pred, cmy_pred)

        output_vtk = _save_prediction_vtk(
            vtk_path,
            prepared.mesh,
            cp_pred,
            prepared.cp_true,
        )
        elapsed = time.time() - start_time
        forces_payload = {
            "CL": cl_pred,
            "CD": cd_pred,
            "CMy": cmy_pred,
            "CL_true": cl_true,
            "CD_true": cd_true,
            "CMy_true": cmy_true,
            "CL_pct_error": cl_pct_err,
            "CD_pct_error": cd_pct_err,
            "CMy_pct_error": cmy_pct_err,
        }
        report_path = _write_report(
            output_vtk=output_vtk,
            vtk_path=vtk_path,
            input_json=inputs,
            metrics=metrics,
            forces=forces_payload,
            elapsed=elapsed,
        )
        LOGGER.info(
            "Inference completed in %.4fs | Output VTK: %s | Report: %s | Metrics: %s",
            elapsed, output_vtk, report_path, metrics,
        )

        return {
            "success": True,
            "output_vtk": output_vtk,
            "report": report_path,
            "log_file": log_path,
            "vertices": int(prepared.mesh.n_points),
            "faces": int(prepared.mesh.n_cells),
            "inference_time": elapsed,
            "metrics": metrics,
            "forces": forces_payload,
        }
    except Exception as exc:  # noqa: BLE001 - UI/API should receive readable errors.
        LOGGER.exception("Inference failed")
        return {
            "success": False,
            "error": str(exc),
            "output_vtk": None,
            "report": None,
            "log_file": log_path,
            "vertices": None,
            "faces": None,
            "inference_time": None,
            "metrics": _empty_metrics(),
            "forces": {
                "CL": None,
                "CD": None,
                "CMy": None,
            },
        }


if __name__ == "__main__":
    result = run_inference(
        "dataset/testvtk/case_0473.vtk",
        {
            "geom_name": "geom_047",
            "alt_kft": 2.541,
            "Re": 21630000,
            "M_inf": 0.4881,
            "alpha_deg": 11.05,
            "beta_deg": 0,
        },
    )
    print(result)
