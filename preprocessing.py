import os
import gc
import torch
import pyvista as pv
import configparser
import pandas as pd
import numpy as np

from torch_geometric.utils import to_undirected
from config import *


os.makedirs(OUT_DIR, exist_ok=True)

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
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


############################################
# RUNNING STATS
############################################

class RunningStats:
    def __init__(self, num_features: int):
        self.n = 0
        self.mean = torch.zeros(num_features, dtype=torch.float32)
        self.m2 = torch.zeros(num_features, dtype=torch.float32)

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return

        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)

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

    def finalize(self):
        denom = max(self.n - 1, 1)
        var = self.m2 / denom
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        return self.mean, std


def normalize(x, mean, std):
    return (x - mean) / std


############################################
# BUILD SINGLE CASE (RAW)
############################################

# def build_case_tensors(row, geom_config):

#     case_name = row[0]
#     geom_name = row[1]

#     vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
#     if not os.path.exists(vtk_path):
#         print(f"[SKIP] Missing VTK: {case_name}")
#         return None

#     mesh = pv.read(vtk_path)
#     mesh = mesh.connectivity(extraction_mode="largest")

#     ############################################
#     # GEOMETRY (CENTER + GLOBAL SCALE)
#     ############################################

#     vertices = mesh.points.astype(np.float32)

#     # Center geometry (critical for symmetry)
#     center = vertices.mean(axis=0)
#     vertices -= center

#     # Global scale (max absolute coordinate)
#     scale = np.max(np.abs(vertices))
#     vertices /= (scale + 1e-8)

#     vertices = torch.tensor(vertices, dtype=torch.float32)

#     ############################################
#     # FACES → EDGES
#     ############################################

#     faces = torch.tensor(
#         mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.long
#     )

#     edges = torch.cat(
#         [
#             faces[:, [0, 1]],
#             faces[:, [1, 2]],
#             faces[:, [2, 0]],
#         ],
#         dim=0,
#     )

#     edge_index = to_undirected(edges.T)

#     ############################################
#     # GEOMETRY PARAMETERS
#     ############################################

#     geom_params = torch.tensor(
#         [
#             geom_config.getfloat(geom_name, k)
#             for k in ["B1","B2","B3","C1","C2","C3","C4","S1","S2","S3"]
#         ],
#         dtype=torch.float32,
#     )

#     geom_params = geom_params.unsqueeze(0).repeat(vertices.size(0), 1)

#     ############################################
#     # FLOW PARAMETERS
#     ############################################

#     flow_params = torch.tensor(
#         [row[2], row[3], row[4], row[5], row[6]],
#         dtype=torch.float32,
#     )

#     flow_params = flow_params.unsqueeze(0).repeat(vertices.size(0), 1)

#     ############################################
#     # NODE FEATURES
#     ############################################

#     x_fp32 = torch.cat(
#         [vertices, flow_params, geom_params],
#         dim=1,
#     )

#     ############################################
#     # COORDS FOR FNO (USE CENTERED XY)
#     # Already in [-1,1] after scaling
#     ############################################

#     coords_2d = vertices[:, :2]  # in [-1,1]

#     ############################################
#     # TARGET Cp
#     ############################################

#     if "cp" in mesh.point_data:
#         y_cp_fp32 = torch.tensor(
#             mesh.point_data["cp"],
#             dtype=torch.float32,
#         ).unsqueeze(1)
#     else:
#         y_cp_fp32 = None

#     return {
#         "case_name": case_name,
#         "x_fp32": x_fp32,
#         "coords_2d": coords_2d,
#         "edge_index": edge_index,
#         "y_cp_fp32": y_cp_fp32,
#         "meta": {
#             "CL": row[7],
#             "CD": row[8],
#             "CM": row[9],
#         },
#     }

def build_case_tensors(row, geom_config):

    case_name = row[0]
    geom_name = row[1]

    vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
    if not os.path.exists(vtk_path):
        print(f"[SKIP] Missing VTK: {case_name}")
        return None

    # Read and process mesh
    mesh = pv.read(vtk_path)
    mesh = mesh.connectivity(extraction_mode="largest")

    ########################################################
    # EXTRACT SURFACE + TRIANGULATE (IMPORTANT)
    ########################################################

    surf = mesh.extract_surface().triangulate()
    del mesh  # Free original mesh immediately
    
    surf = surf.compute_normals(cell_normals=True, point_normals=False)
    surf = surf.compute_cell_sizes()

    ########################################################
    # GEOMETRY (CENTER + GLOBAL SCALE)
    ########################################################

    vertices = surf.points.astype(np.float32)

    center = vertices.mean(axis=0)
    vertices -= center

    scale = np.max(np.abs(vertices))
    vertices /= (scale + 1e-8)

    vertices = torch.tensor(vertices, dtype=torch.float32)

    ########################################################
    # FACES (TRIANGLES)
    ########################################################

    faces_np = surf.faces.reshape(-1, 4)[:, 1:]  # (n_cells, 3)
    faces = torch.tensor(faces_np, dtype=torch.long)
    del faces_np

    ########################################################
    # CELL NORMALS & AREAS (OPTIMIZE DTYPE)
    ########################################################

    cell_normals = torch.tensor(
        surf.cell_data["Normals"],
        dtype=torch.float32,
    )

    cell_areas = torch.tensor(
        surf.cell_data["Area"],
        dtype=torch.float32,
    )

    ########################################################
    # EDGES (FOR GNN)
    ########################################################

    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )

    edge_index = to_undirected(edges.T)
    del edges  # Free intermediate

    ########################################################
    # GEOMETRY PARAMETERS
    ########################################################

    geom_params = torch.tensor(
        [
            geom_config.getfloat(geom_name, k)
            for k in ["B1","B2","B3","C1","C2","C3","C4","S1","S2","S3"]
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
    # NODE FEATURES
    ########################################################

    x_fp32 = torch.cat(
        [vertices, flow_params, geom_params],
        dim=1,
    )
    del geom_params, flow_params  # Free components

    ########################################################
    # TARGET Cp
    ########################################################

    if "cp" in surf.point_data:
        y_cp_fp32 = torch.tensor(
            surf.point_data["cp"],
            dtype=torch.float32,
        ).unsqueeze(1)
    else:
        y_cp_fp32 = None

    del surf  # Free mesh object

    return {
        "case_name": case_name,
        "x_fp32": x_fp32,
        "coords_2d": vertices[:, :2],
        "edge_index": edge_index,
        "faces": faces,
        "cell_normals": cell_normals,
        "cell_areas": cell_areas,
        "y_cp_fp32": y_cp_fp32,
        "meta": {
            "CL": row[7],
            "CD": row[8],
            "CM": row[9],
        },
    }


############################################
# SAVE NORMALIZED CASE (MEMORY EFFICIENT)
############################################

def preprocess_case(case_pack, x_mean, x_std, cp_mean, cp_std):

    # Normalize node features
    x_norm = normalize(case_pack["x_fp32"], x_mean, x_std).half()

    # Normalize target Cp
    if case_pack["y_cp_fp32"] is not None:
        y_cp = normalize(case_pack["y_cp_fp32"], cp_mean, cp_std)
    else:
        y_cp = None

    # Build data dictionary
    data = {
        "x": x_norm,
        "edge_index": case_pack["edge_index"],
        "pos": case_pack["coords_2d"].half(),
        "faces": case_pack["faces"],
        "cell_normals": case_pack["cell_normals"],
        "cell_areas": case_pack["cell_areas"],
        "y_cp": y_cp,
        "meta": {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_cp_mean": cp_mean,
            "y_cp_std": cp_std,
            "CL": case_pack["meta"]["CL"],
            "CD": case_pack["meta"]["CD"],
            "CM": case_pack["meta"]["CM"],
        },
    }

    del x_norm, y_cp
    return data


def save_chunk_cases(cases, chunk_idx):
    chunk_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:03d}.pt")
    torch.save(cases, chunk_path)


############################################
# MAIN (MEMORY-EFFICIENT TWO-PASS STREAMING)
############################################

def main():

    df = pd.read_csv(CASE_FILE, sep=r"\s+")

    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)

    rows = df if NO_CASES is None else df.head(NO_CASES)
    rows = rows.reset_index(drop=True)
    total_cases = len(rows)

    ############################################################
    # PASS 1: COMPUTE STATISTICS (NO STORAGE)
    ############################################################
    
    print("\n" + "="*70)
    print("PASS 1: COMPUTING NORMALIZATION STATISTICS")
    print("="*70)
    print(f"Total cases to process: {total_cases}")

    x_stats = RunningStats(num_features=NODE_IN_DIM)
    cp_stats = RunningStats(num_features=1)

    processed_count = 0

    for idx, (_, row) in enumerate(rows.iterrows(), 1):
        
        try:
            pack = build_case_tensors(row, geom_config)
            if pack is None:
                continue

            x_stats.update(pack["x_fp32"])

            if pack["y_cp_fp32"] is not None:
                cp_stats.update(pack["y_cp_fp32"])

            processed_count += 1

            # Progress indicator with memory usage
            if idx % 50 == 0:
                mem_usage = get_memory_usage()
                print(f"  [{idx:5d}/{total_cases}] {processed_count} cases processed | RAM: {mem_usage:.2f} GB")
            
            # Explicitly delete to free memory
            del pack
            
            # Periodic garbage collection
            if idx % 100 == 0:
                clear_cache()
        
        except Exception as e:
            print(f"[ERROR] Failed to process case {row[0]}: {str(e)}")
            continue

    x_mean, x_std = x_stats.finalize()
    cp_mean, cp_std = cp_stats.finalize()

    # Save statistics
    torch.save(
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "cp_mean": cp_mean,
            "cp_std": cp_std,
        },
        NORM_STATS_FILE,
    )

    print(f"\n[✓] Statistics computed from {processed_count} cases")
    print(f"[✓] Saved normalization stats → {NORM_STATS_FILE}")
    print(f"    x_mean shape: {x_mean.shape}")
    print(f"    cp_mean: {cp_mean.item():.6f}, cp_std: {cp_std.item():.6f}")

    clear_cache()

    ############################################################
    # PASS 2: NORMALIZE & SAVE (STREAMING)
    ############################################################
    
    print("\n" + "="*70)
    print("PASS 2: NORMALIZING & SAVING CASES")
    print("="*70)

    saved_count = 0
    chunk_cases = []
    chunk_idx = 0

    for idx, (_, row) in enumerate(rows.iterrows(), 1):
        
        try:
            pack = build_case_tensors(row, geom_config)
            if pack is None:
                continue

            chunk_cases.append(
                preprocess_case(pack, x_mean, x_std, cp_mean, cp_std)
            )
            saved_count += 1

            if len(chunk_cases) >= CHUNK_SIZE:
                save_chunk_cases(chunk_cases, chunk_idx)
                chunk_idx += 1
                chunk_cases.clear()

            # Progress indicator with memory usage
            if idx % 50 == 0:
                mem_usage = get_memory_usage()
                print(f"  [{idx:5d}/{total_cases}] {saved_count} cases saved | RAM: {mem_usage:.2f} GB")
            
            # Explicitly delete to free memory
            del pack
            
            # Periodic garbage collection
            if idx % 100 == 0:
                clear_cache()

        except Exception as e:
            print(f"[ERROR] Failed to process case {row[0]}: {str(e)}")
            continue

    if chunk_cases:
        save_chunk_cases(chunk_cases, chunk_idx)

    clear_cache()

    print("\n" + "="*70)
    print(f"✓ PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"  Total cases processed: {saved_count}/{total_cases}")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Chunk files written: {chunk_idx + (1 if chunk_cases else 0)}")
    print("="*70)