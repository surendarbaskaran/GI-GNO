import os
import gc
import time
import torch
import pyvista as pv
import configparser
import pandas as pd
import numpy as np

import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from config import *


os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_LAYOUT = (
    "xyz[0:3]",
    "flow[3:8]",
    "geom[8:18]",
    "curvature[18:19]",
    "normal[19:22]",
    "local_area[22:23]",
    "edge_stats[23:27]",
)

# ============================================
# MEMORY OPTIMIZATION
# ============================================

def clear_cache():
    """Aggressively clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


############################################
# RUNNING STATS (FLOAT64 FOR 10,000+ CASES)
############################################

class RunningStats:
    """Numerically stable Welford running mean/variance in float64 for 10,000+ cases (~5e8 nodes)."""

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.n = 0
        self.mean = torch.zeros(num_features, dtype=torch.float64)
        self.m2 = torch.zeros(num_features, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return

        x64 = x.detach().to(dtype=torch.float64, device="cpu")
        batch_n = int(x64.shape[0])
        batch_mean = x64.mean(dim=0)
        batch_m2 = ((x64 - batch_mean) ** 2).sum(dim=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.n + batch_n

        self.mean += delta * (batch_n / total)
        self.m2 += batch_m2 + (delta ** 2) * (self.n * batch_n / total)
        self.n = total

    def state_dict(self):
        return {
            "num_features": self.num_features,
            "n": self.n,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
        }

    def load_state_dict(self, state):
        self.num_features = state["num_features"]
        self.n = int(state["n"])
        self.mean = state["mean"].to(dtype=torch.float64)
        self.m2 = state["m2"].to(dtype=torch.float64)

    def finalize(self):
        denom = max(self.n - 1, 1)
        var = self.m2 / denom
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        # Guard against constant features (e.g. beta_deg = 0 across symmetric dataset):
        # set std = 1.0 when variance is near zero so normalization yields 0.0 instead of blowing up by 1e6.
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return self.mean.float(), std.float()


def normalize(x, mean, std):
    return (x - mean) / std


def build_mesh_edges(faces):
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    return to_undirected(edges.T)


def build_knn_edges(vertices, k):
    """High-speed KNN edge construction.
    Uses O(N log N) cKDTree if available, with GPU and vectorized fallbacks.
    """
    num_nodes = vertices.size(0)
    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k, num_nodes - 1)

    # Strategy 1: Fast scipy cKDTree (takes ~0.03s vs 25s for brute force)
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

    # Strategy 2: GPU PyTorch if CUDA available (~5 ms)
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            v_gpu = vertices.to(device)
            edge_chunks = []
            chunk_size = 4096
            for start in range(0, num_nodes, chunk_size):
                end = min(start + chunk_size, num_nodes)
                dist = torch.cdist(v_gpu[start:end], v_gpu)
                local = torch.arange(end - start, device=device)
                dist[local, torch.arange(start, end, device=device)] = float("inf")
                nn_idx = dist.topk(k, largest=False).indices
                src = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, k)
                edge_chunks.append(torch.stack([src.reshape(-1), nn_idx.reshape(-1)], dim=0).cpu())
            return to_undirected(torch.cat(edge_chunks, dim=1))
        except Exception:
            pass

    # Strategy 3: Fast chunked CPU fallback
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


def merge_edge_indices(*edge_indices):
    """Fast linear-hash edge deduplication (10x faster than 2D unique)."""
    edges = [e.long() for e in edge_indices if e.numel() > 0]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    cat_edges = torch.cat(edges, dim=1)
    max_node = int(cat_edges.max().item()) + 1
    edge_hash = cat_edges[0] * max_node + cat_edges[1]
    unique_hash = torch.unique(edge_hash)
    edge_index = torch.stack([unique_hash // max_node, unique_hash % max_node], dim=0)
    return edge_index.contiguous()


def compute_edge_attr(vertices, edge_index):
    """Computes edge features containing directional flow displacements from (0,0,0) to (1,_,_)."""
    src, dst = edge_index.long()
    v_fp32 = vertices.float()
    rel_xyz = v_fp32[dst] - v_fp32[src]
    distance = torch.sqrt(torch.sum(rel_xyz * rel_xyz, dim=1, keepdim=True) + 1e-12)
    direction = rel_xyz / (distance + 1e-8)
    edge_length = distance
    return torch.cat([edge_length, rel_xyz, distance, direction], dim=1)


def compute_vertex_normals(num_nodes, faces, cell_normals):
    vertex_normals = torch.zeros((num_nodes, 3), dtype=torch.float32)
    repeated_normals = cell_normals.repeat_interleave(3, dim=0)
    vertex_normals.index_add_(0, faces.reshape(-1), repeated_normals)
    return F.normalize(vertex_normals, p=2, dim=1, eps=1e-8)


def compute_local_area(num_nodes, faces, cell_areas):
    local_area = torch.zeros((num_nodes, 1), dtype=torch.float32)
    area_share = (cell_areas / 3.0).repeat_interleave(3).unsqueeze(1)
    local_area.index_add_(0, faces.reshape(-1), area_share)
    return local_area


def compute_curvature(vertices, mesh_edge_index):
    src, dst = mesh_edge_index
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros((vertices.size(0), 1), dtype=vertices.dtype)
    neighbor_sum.index_add_(0, src, vertices[dst])
    neighbor_count.index_add_(0, src, torch.ones_like(neighbor_count[src]))
    neighbor_mean = neighbor_sum / (neighbor_count + 1e-8)
    return torch.linalg.norm(vertices - neighbor_mean, dim=1, keepdim=True)


def compute_edge_length_stats(vertices, mesh_edge_index):
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


def integrate_surface_forces(y_cp_fp32, y_cf_fp32, faces, cell_normals, cell_areas, cell_centers, alpha_deg, beta_deg):
    """Integrate ground-truth Cp (and optional Cf) over the mesh to obtain exact surface-integrated (CL_cp, CD_cp, CM_cp)."""
    if y_cp_fp32 is None:
        return None, None, None

    cp_node = y_cp_fp32.squeeze(-1).float()
    cp_cell = cp_node[faces].mean(dim=1)
    if y_cf_fp32 is not None:
        cf_cell = y_cf_fp32.float()[faces].mean(dim=1)
    else:
        cf_cell = torch.zeros((faces.size(0), 3), dtype=torch.float32)

    dF_total = (-cp_cell[:, None] * cell_normals.float() + cf_cell) * cell_areas.float()[:, None]
    cmy_cell = (cell_centers[:, 2] - Z_REF) * dF_total[:, 0] - (cell_centers[:, 0] - X_REF) * dF_total[:, 2]

    F_total = dF_total.sum(dim=0)
    CX, CY, CZ = F_total[0] / S_REF, F_total[1] / S_REF, F_total[2] / S_REF
    CMy = cmy_cell.sum() / (S_REF * C_REF)

    alpha = np.deg2rad(float(alpha_deg))
    beta = np.deg2rad(float(beta_deg))

    CD = float(CX * np.cos(alpha) * np.cos(beta) + CY * np.sin(beta) + CZ * np.sin(alpha) * np.cos(beta))
    CL = float(-CX * np.sin(alpha) + CZ * np.cos(alpha))
    return CL, CD, float(CMy)


def assert_feature_contract(x_fp32, vertices, flow_params, geom_params, edge_index, edge_attr):
    assert x_fp32.shape[1] == NODE_IN_DIM, (
        f"Feature width mismatch: got {x_fp32.shape[1]}, expected NODE_IN_DIM={NODE_IN_DIM}. "
        f"Expected order: {FEATURE_LAYOUT}"
    )
    assert edge_attr.shape[1] == EDGE_DIM, (
        f"Edge feature width mismatch: got {edge_attr.shape[1]}, expected EDGE_DIM={EDGE_DIM}."
    )
    assert torch.allclose(x_fp32[:, :3], vertices, atol=1e-6), "x[:, 0:3] must be xyz."
    assert torch.allclose(x_fp32[:, 3:8], flow_params, atol=1e-6), "x[:, 3:8] must be flow."
    assert torch.allclose(x_fp32[:, 8:18], geom_params, atol=1e-6), "x[:, 8:18] must be geom."
    assert torch.allclose(edge_attr, compute_edge_attr(vertices, edge_index), atol=1e-5), (
        "edge_attr does not match edge_index and normalized xyz."
    )
    assert torch.isfinite(x_fp32).all(), "x_fp32 contains non-finite values."
    assert torch.isfinite(edge_attr).all(), "edge_attr contains non-finite values."

    # Starting point must be at (0,0,0) and ending along chord at (1,_,_)
    assert vertices[:, 0].min() >= -1e-4, f"Vertices x-coordinate must start at >= 0.0, got {vertices[:, 0].min()}"
    assert vertices[:, 0].max() <= 1.0001, f"Vertices x-coordinate must end at <= 1.0, got {vertices[:, 0].max()}"
    nose_idx = torch.argmin(vertices[:, 0])
    assert torch.allclose(vertices[nose_idx], torch.zeros(3, dtype=vertices.dtype), atol=1e-3), (
        f"Starting point of geometry must be (0,0,0), got {vertices[nose_idx].tolist()}."
    )


############################################
# BUILD SINGLE CASE (RAW)
############################################

def build_case_tensors(row, geom_config, verify_contract=False):
    case_name = row[0]
    geom_name = row[1]

    vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
    if not os.path.exists(vtk_path):
        print(f"[SKIP] Missing VTK: {case_name}")
        return None

    # Read surface mesh
    mesh = pv.read(vtk_path)
    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    del mesh

    surf = surf.compute_normals(cell_normals=True, point_normals=False)
    surf = surf.compute_cell_sizes()

    ########################################################
    # GEOMETRY: STARTING POINT AT (0,0,0), ENDING AT (1,_,_)
    ########################################################
    vertices_np = surf.points.astype(np.float32)

    # Find starting point (nose / minimum x coordinate)
    x_min = float(np.min(vertices_np[:, 0]))
    x_max = float(np.max(vertices_np[:, 0]))
    chord = max(x_max - x_min, 1e-8)

    nose_idx = int(np.argmin(vertices_np[:, 0]))
    nose_y = float(vertices_np[nose_idx, 1])
    nose_z = float(vertices_np[nose_idx, 2])

    # Align starting point to strictly (0, 0, 0) and normalize chord to 1.0
    vertices_np[:, 0] = (vertices_np[:, 0] - x_min) / chord
    vertices_np[:, 1] = (vertices_np[:, 1] - nose_y) / chord
    vertices_np[:, 2] = (vertices_np[:, 2] - nose_z) / chord

    # Clamp small numerical negative zeros
    vertices_np[:, 0] = np.clip(vertices_np[:, 0], 0.0, 1.0)
    vertices = torch.tensor(vertices_np, dtype=torch.float32)

    ########################################################
    # FACES (TRIANGLES)
    ########################################################
    faces_np = surf.faces.reshape(-1, 4)[:, 1:]  # (n_cells, 3)
    faces = torch.tensor(faces_np, dtype=torch.long)

    ########################################################
    # CELL NORMALS, AREAS & CENTERS (WITH SYMMETRY & OUTWARD NORMAL CHECK)
    ########################################################
    raw_points = surf.points.astype(np.float32)
    cell_centers_np = raw_points[faces_np].mean(axis=1).astype(np.float32)
    del faces_np

    raw_y = raw_points[:, 1]
    y_min, y_max = float(np.min(raw_y)), float(np.max(raw_y))
    symmetry_factor = 2.0 if (abs(y_min) < 1e-4 or abs(y_max) < 1e-4) else 1.0

    normals_np = np.array(surf.cell_data["Normals"], dtype=np.float32).copy()
    z_coords = cell_centers_np[:, 2]
    upper_mask = z_coords > np.median(z_coords)
    if np.any(upper_mask) and np.mean(normals_np[upper_mask, 2]) < 0:
        normals_np = -normals_np

    cell_normals = torch.tensor(normals_np, dtype=torch.float32)
    cell_areas = torch.tensor(
        np.array(surf.cell_data["Area"], dtype=np.float32) * symmetry_factor,
        dtype=torch.float32,
    )
    cell_centers = torch.tensor(cell_centers_np, dtype=torch.float32)

    ########################################################
    # EDGES + EDGE FEATURES (MESH EDGES + KNN)
    ########################################################
    mesh_edge_index = build_mesh_edges(faces)
    knn_edge_index = build_knn_edges(vertices, KNN_K)
    edge_index = merge_edge_indices(mesh_edge_index, knn_edge_index)

    ########################################################
    # GEOMETRY PARAMETERS
    ########################################################
    geom_params = torch.tensor(
        [
            geom_config.getfloat(geom_name, k)
            for k in ["B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
        ],
        dtype=torch.float32,
    )
    geom_params = geom_params.unsqueeze(0).repeat(vertices.size(0), 1)

    ########################################################
    # FLOW PARAMETERS
    ########################################################
    flow_params = torch.tensor(
        [row[2], row[3], row[4], row[5], row[6]],
        dtype=torch.float32,
    )
    flow_params = flow_params.unsqueeze(0).repeat(vertices.size(0), 1)

    ########################################################
    # GEOMETRY NODE FEATURES
    ########################################################
    normalized_cell_areas = 0.5 * torch.linalg.norm(
        torch.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
            dim=1,
        ),
        dim=1,
    )
    vertex_normals = compute_vertex_normals(vertices.size(0), faces, cell_normals)
    local_area = compute_local_area(vertices.size(0), faces, normalized_cell_areas)
    curvature = compute_curvature(vertices, mesh_edge_index)
    edge_length_stats = compute_edge_length_stats(vertices, mesh_edge_index)

    ########################################################
    # NODE FEATURES (27 DIM)
    ########################################################
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

    if verify_contract or SAVE_EDGE_ATTR:
        edge_attr = compute_edge_attr(vertices, edge_index)
        assert_feature_contract(x_fp32, vertices, flow_params, geom_params, edge_index, edge_attr)
        if not SAVE_EDGE_ATTR:
            del edge_attr
            edge_attr = None
    else:
        edge_attr = None

    del geom_params, flow_params

    ########################################################
    # TARGET Cp & Cf (if available)
    ########################################################
    if "cp" in surf.point_data:
        y_cp_fp32 = torch.tensor(
            surf.point_data["cp"],
            dtype=torch.float32,
        ).unsqueeze(1)
    else:
        y_cp_fp32 = None

    if all(k in surf.point_data for k in ("cf_x", "cf_y", "cf_z")):
        y_cf_fp32 = torch.tensor(
            np.stack(
                [surf.point_data["cf_x"], surf.point_data["cf_y"], surf.point_data["cf_z"]],
                axis=1,
            ),
            dtype=torch.float32,
        )
    elif "cf" in surf.point_data and np.asarray(surf.point_data["cf"]).ndim == 2 and np.asarray(surf.point_data["cf"]).shape[1] == 3:
        y_cf_fp32 = torch.tensor(surf.point_data["cf"], dtype=torch.float32)
    else:
        y_cf_fp32 = None

    del surf

    cl_cp, cd_cp, cm_cp = integrate_surface_forces(
        y_cp_fp32=y_cp_fp32,
        y_cf_fp32=y_cf_fp32,
        faces=faces,
        cell_normals=cell_normals,
        cell_areas=cell_areas,
        cell_centers=cell_centers,
        alpha_deg=row[5],
        beta_deg=row[6],
    )

    return {
        "case_name": case_name,
        "x_fp32": x_fp32,
        "edge_index": edge_index.to(torch.int32),
        "edge_attr": edge_attr.half() if edge_attr is not None else None,
        "faces": faces.to(torch.int32),
        "cell_normals": cell_normals,
        "cell_areas": cell_areas,
        "cell_centers": cell_centers,
        "y_cp_fp32": y_cp_fp32,
        "y_cf_fp32": y_cf_fp32,
        "meta": {
            "CL": float(row[8]),
            "CD": float(row[7]),
            "CM": float(row[9]),
            "CL_cp": cl_cp if cl_cp is not None else float(row[8]),
            "CD_cp": cd_cp if cd_cp is not None else float(row[7]),
            "CM_cp": cm_cp if cm_cp is not None else float(row[9]),
            "alpha_deg": float(row[5]),
            "beta_deg": float(row[6]),
        },
    }


############################################
# SAVE NORMALIZED CASE (MEMORY EFFICIENT FOR 10,000 CASES)
############################################

def preprocess_case(case_pack, x_mean, x_std, cp_mean, cp_std):
    # Normalize node features
    x_norm = normalize(case_pack["x_fp32"], x_mean, x_std).half()

    # Keep raw chord-normalized 3D coordinates in float32 for high-precision GNO kernels & Fourier encoding
    xyz_fp32 = case_pack["x_fp32"][:, :3].float().contiguous()

    # Normalize target Cp
    if case_pack["y_cp_fp32"] is not None:
        y_cp = normalize(case_pack["y_cp_fp32"], cp_mean, cp_std).float()
    else:
        y_cp = None

    # Compact data dictionary for graph Encoder -> GNO -> Decoder -> GNN
    data = {
        "x": x_norm,
        "xyz": xyz_fp32,
        "pos": xyz_fp32,
        "edge_index": case_pack["edge_index"].to(torch.int32),
        "faces": case_pack["faces"].to(torch.int32),
        "cell_normals": case_pack["cell_normals"],
        "cell_areas": case_pack["cell_areas"],
        "cell_centers": case_pack["cell_centers"],
        "y_cp": y_cp,
        "y_cf": case_pack.get("y_cf_fp32"),
        "meta": {
            "case_name": case_pack["case_name"],
            "CL": case_pack["meta"]["CL"],
            "CD": case_pack["meta"]["CD"],
            "CM": case_pack["meta"]["CM"],
            "CL_cp": case_pack["meta"].get("CL_cp", case_pack["meta"]["CL"]),
            "CD_cp": case_pack["meta"].get("CD_cp", case_pack["meta"]["CD"]),
            "CM_cp": case_pack["meta"].get("CM_cp", case_pack["meta"]["CM"]),
            "alpha_deg": case_pack["meta"]["alpha_deg"],
            "beta_deg": case_pack["meta"]["beta_deg"],
        },
    }
    if SAVE_EDGE_ATTR and case_pack.get("edge_attr") is not None:
        data["edge_attr"] = case_pack["edge_attr"].half()

    del x_norm, y_cp
    return data


def save_chunk_cases(cases, chunk_idx):
    chunk_filename = f"chunk_{chunk_idx:04d}.pt" if chunk_idx >= 1000 else f"chunk_{chunk_idx:03d}.pt"
    chunk_path = os.path.join(OUT_DIR, chunk_filename)
    torch.save(cases, chunk_path)
    return chunk_filename


############################################
# MAIN (SCALABLE STREAMING FOR 10,000+ CASES)
############################################

def main():
    log_path = setup_run_logger("preprocessing", reuse_active=False)

    df = pd.read_csv(CASE_FILE, sep=r"\s+")
    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)

    rows = df if NO_CASES is None else df.head(NO_CASES)
    rows = rows.reset_index(drop=True)
    total_cases = len(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    temp_dir = os.path.join(OUT_DIR, "_raw_temp")
    os.makedirs(temp_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("OPTIMIZED HIGH-SPEED PREPROCESSING (10,000+ CASE SCALABLE STREAM)")
    print("=" * 70)
    print(f"Log file: {log_path}")
    print(f"Total cases: {total_cases}")
    print(f"Coordinates: starting at (0,0,0) and ending at (1,_,_) [stored in float32]")
    print(f"Edge storage: int32 edge_index | SAVE_EDGE_ATTR={SAVE_EDGE_ATTR}")
    print(f"Chunk size: {CHUNK_SIZE}")

    x_stats = RunningStats(num_features=NODE_IN_DIM)
    cp_stats = RunningStats(num_features=1)

    processed_count = 0
    raw_chunk = []
    chunk_idx = 0
    start_time = time.time()

    for idx, (_, row) in enumerate(rows.iterrows(), 1):
        try:
            verify_first = (processed_count == 0)
            pack = build_case_tensors(row, geom_config, verify_contract=verify_first)
            if pack is None:
                continue

            if verify_first:
                sample_ea = compute_edge_attr(pack["x_fp32"][:, :3], pack["edge_index"])
                print("==== PREPROCESSING RAW CHECK ====")
                print(f"Feature order: {' | '.join(FEATURE_LAYOUT)}")
                print(f"Case: {pack['case_name']}")
                nose_pt = pack["x_fp32"][torch.argmin(pack["x_fp32"][:, 0]), :3].tolist()
                print(f"Starting point coords (nose): {nose_pt}")
                print(f"x range: [{pack['x_fp32'][:, 0].min():.4f}, {pack['x_fp32'][:, 0].max():.4f}]")
                print(f"x_fp32 shape: {tuple(pack['x_fp32'].shape)}")
                print(f"edge_index shape: {tuple(pack['edge_index'].shape)} (dtype={pack['edge_index'].dtype})")
                print(f"edge_attr shape: {tuple(sample_ea.shape)} (on-the-fly={not SAVE_EDGE_ATTR})")
                print(
                    f"Table forces: CL={pack['meta']['CL']:.5f}, CD={pack['meta']['CD']:.5f}, CM={pack['meta']['CM']:.5f} | "
                    f"GT Cp-integrated forces: CL_cp={pack['meta']['CL_cp']:.5f}, CD_cp={pack['meta']['CD_cp']:.5f}, CM_cp={pack['meta']['CM_cp']:.5f}"
                )
                print("==== END PREPROCESSING RAW CHECK ====")
                del sample_ea

            # Update running statistics online in float64
            x_stats.update(pack["x_fp32"])
            if pack["y_cp_fp32"] is not None:
                cp_stats.update(pack["y_cp_fp32"])

            raw_chunk.append(pack)
            processed_count += 1

            if len(raw_chunk) >= CHUNK_SIZE:
                temp_chunk_path = os.path.join(temp_dir, f"temp_chunk_{chunk_idx:04d}.pt")
                torch.save(raw_chunk, temp_chunk_path)
                chunk_idx += 1
                raw_chunk.clear()

            if idx % 10 == 0 or idx == total_cases:
                elapsed = time.time() - start_time
                per_case = elapsed / processed_count if processed_count > 0 else 0
                eta = (total_cases - processed_count) * per_case
                mem_usage = get_memory_usage()
                print(
                    f"  [{idx:5d}/{total_cases}] {processed_count} processed | "
                    f"{per_case:.2f}s/case | ETA: {eta:.1f}s | RAM: {mem_usage:.2f} GB"
                )

            del pack
            if idx % 50 == 0:
                clear_cache()

        except Exception as e:
            print(f"[ERROR] Failed to process case {row[0]}: {str(e)}")
            continue

    if raw_chunk:
        temp_chunk_path = os.path.join(temp_dir, f"temp_chunk_{chunk_idx:04d}.pt")
        torch.save(raw_chunk, temp_chunk_path)
        chunk_idx += 1
        raw_chunk.clear()

    x_mean, x_std = x_stats.finalize()
    cp_mean, cp_std = cp_stats.finalize()

    # Save normalization statistics
    torch.save(
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "cp_mean": cp_mean,
            "cp_std": cp_std,
            "num_cases": processed_count,
            "total_nodes": x_stats.n,
        },
        NORM_STATS_FILE,
    )
    print(f"\n[OK] Computed float64 statistics from {processed_count} cases ({x_stats.n} total nodes) in {time.time() - start_time:.2f}s")
    print(f"[OK] Saved normalization stats -> {NORM_STATS_FILE}")
    print(f"     x_mean shape: {x_mean.shape}, cp_mean: {cp_mean.item():.6f}, cp_std: {cp_std.item():.6f}")

    # Normalization streaming pass + manifest construction
    print("\nWriting normalized chunks and dataset manifest...")
    manifest_chunks = []
    for c_i in range(chunk_idx):
        temp_file = os.path.join(temp_dir, f"temp_chunk_{c_i:04d}.pt")
        packs = torch.load(temp_file, map_location="cpu", weights_only=False)
        normalized_cases = [
            preprocess_case(p, x_mean, x_std, cp_mean, cp_std)
            for p in packs
        ]
        chunk_filename = save_chunk_cases(normalized_cases, c_i)
        manifest_chunks.append({
            "chunk_file": chunk_filename,
            "num_cases": len(normalized_cases),
            "case_names": [c["meta"]["case_name"] for c in normalized_cases],
        })
        del packs, normalized_cases
        os.remove(temp_file)
        if (c_i + 1) % 50 == 0:
            clear_cache()

    torch.save(
        {
            "total_cases": processed_count,
            "chunk_size": CHUNK_SIZE,
            "chunks": manifest_chunks,
        },
        MANIFEST_FILE,
    )
    print(f"[OK] Saved dataset manifest -> {MANIFEST_FILE} ({len(manifest_chunks)} chunks)")

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    clear_cache()
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PREPROCESSING COMPLETED IN {total_time:.2f} SECONDS ({total_time / 60:.2f} mins)!")
    print(f"  Total cases: {processed_count}/{total_cases}")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Chunks created: {chunk_idx}")
    print(f"  Log file: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
