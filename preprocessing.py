# preprocessing.py
# ---------------------------------------------
# VTK → PT preprocessing with:
# - normalization
# - FP16 tensors (inputs + targets)
# - FP32 normalization stats stored in metadata
# ---------------------------------------------

import os
import torch
import pyvista as pv
import configparser
import pandas as pd
from torch_geometric.utils import to_undirected
from  config import *

# -----------------------------
# PATH CONFIG (fixed inside file)
# -----------------------------
# RAW_VTK_DIR = r"dataset/trainvtk"
# GEOM_PARAM_FILE = r"dataset/geom_params.ini"
# CASE_FILE = r"dataset/case_data.dat"
# OUT_DIR = r"ptfiles"
# NO_CASES=9

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# NORMALIZATION UTILS
# -----------------------------
def compute_norm_stats(x: torch.Tensor):
    """
    x : [N, D] FP32
    returns mean, std in FP32
    """
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


# -----------------------------
# PREPROCESS SINGLE CASE
# -----------------------------
def preprocess_case(row, geom_config):
    case_name = row[0]
    geom_name = row[1]

    vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
    out_path = os.path.join(OUT_DIR, f"{case_name}.pt")

    if not os.path.exists(vtk_path):
        print(f"[SKIP] Missing VTK: {case_name}")
        return

    print(f"[PROCESS] {case_name}")

    mesh = pv.read(vtk_path)
    mesh = mesh.connectivity(extraction_mode="largest")

    # -----------------------------
    # GEOMETRY
    # -----------------------------
    vertices = torch.tensor(mesh.points, dtype=torch.float32)  # [N, 3]

    faces = torch.tensor(
        mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.long
    )

    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    edge_index = to_undirected(edges.T)

    # -----------------------------
    # GEOMETRY PARAMETERS
    # -----------------------------
    geom_params = torch.tensor(
        [
            geom_config.getfloat(geom_name, k)
            for k in ["B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
        ],
        dtype=torch.float32,
    )
    geom_params = geom_params.unsqueeze(0).repeat(vertices.size(0), 1)

    # -----------------------------
    # FLOW PARAMETERS
    # -----------------------------
    flow_params = torch.tensor(
        [row[2], row[3], row[4], row[5], row[6]],
        dtype=torch.float32,
    )
    flow_params = flow_params.unsqueeze(0).repeat(vertices.size(0), 1)

    # -----------------------------
    # NODE FEATURES (FP32 for stats)
    # -----------------------------
    x_fp32 = torch.cat(
        [vertices, flow_params, geom_params], dim=1
    )  # [N, 17]

    x_mean, x_std = compute_norm_stats(x_fp32)
    x_norm = normalize(x_fp32, x_mean, x_std).half()  # FP16

    # -----------------------------
    # NORMALIZED 2D COORDS (for grid)
    # -----------------------------
    xy = vertices[:, :2]
    xy_min = xy.min(0, keepdim=True)[0]
    xy_max = xy.max(0, keepdim=True)[0]
    coords_2d = ((xy - xy_min) / (xy_max - xy_min + 1e-6)).half()

    # -----------------------------
    # TARGET Cp
    # -----------------------------
    y_cp = None
    y_cp_mean = None
    y_cp_std = None

    if "cp" in mesh.point_data:
        y_cp_fp32 = torch.tensor(
            mesh.point_data["cp"], dtype=torch.float32
        ).unsqueeze(1)

        y_cp_mean, y_cp_std = compute_norm_stats(y_cp_fp32)
        y_cp = normalize(y_cp_fp32, y_cp_mean, y_cp_std)

    # -----------------------------
    # SAVE
    # -----------------------------
    data = {
        "x": x_norm,                 # FP16
        "edge_index": edge_index,    # int64
        "coords_2d": coords_2d,       # FP16
        "y_cp": y_cp,                # FP32 or None
        "meta": {
            "x_mean": x_mean,        # FP32
            "x_std": x_std,          # FP32
            "y_cp_mean": y_cp_mean,  # FP32 or None
            "y_cp_std": y_cp_std,    # FP32 or None
            "CL": row[7],
            "CD": row[8],
            "CM": row[9],
        },
    }

    torch.save(data, out_path)
    print(f"[OK] Saved: {out_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = pd.read_csv(CASE_FILE, sep=r"\s+")

    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)
    count=0

    for _, row in df.iterrows():
        preprocess_case(row, geom_config)
        if count>NO_CASES:  #       Remove these lines 
          break             #       while running full dataset if required
        count+=1            #       used these lines to limit the steps


# if __name__ == "__main__":
#     main()
