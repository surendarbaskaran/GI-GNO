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
        self.mean = self.mean + delta * (batch_n / total)
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * (self.n * batch_n / total)
        self.n = total

    def finalize(self):
        denom = max(self.n - 1, 1)
        var = self.m2 / denom
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        return self.mean, std


def normalize(x, mean, std):
    return (x - mean) / std


# -----------------------------
def build_case_tensors(row, geom_config):
    case_name = row[0]
    geom_name = row[1]

    vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
    if not os.path.exists(vtk_path):
        print(f"[SKIP] Missing VTK: {case_name}")
        return None

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

    # keep FP32 for global stats / normalization later
    x_norm = None

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
    else:
        y_cp_fp32 = None

    return {
        "case_name": case_name,
        "x_fp32": x_fp32,
        "coords_2d": coords_2d,
        "edge_index": edge_index,
        "y_cp_fp32": y_cp_fp32,
        "meta": {"CL": row[7], "CD": row[8], "CM": row[9]},
    }


# -----------------------------
# PREPROCESS SINGLE CASE
# -----------------------------
def preprocess_case(case_pack, x_mean, x_std, cp_mean, cp_std):
    case_name = case_pack["case_name"]
    out_path = os.path.join(OUT_DIR, f"{case_name}.pt")

    print(f"[PROCESS] {case_name}")
    x_norm = normalize(case_pack["x_fp32"], x_mean, x_std).half()

    y_cp = None
    if case_pack["y_cp_fp32"] is not None:
        y_cp = normalize(case_pack["y_cp_fp32"], cp_mean, cp_std)

    # -----------------------------
    # SAVE
    # -----------------------------
    data = {
        "x": x_norm,                         # FP16
        "edge_index": case_pack["edge_index"],  # int64
        "coords_2d": case_pack["coords_2d"], # FP16
        "y_cp": y_cp,                        # FP32 or None
        "meta": {
            "x_mean": x_mean,             # FP32 (global)
            "x_std": x_std,               # FP32 (global)
            "y_cp_mean": cp_mean,         # FP32 (global)
            "y_cp_std": cp_std,           # FP32 (global)
            "CL": case_pack["meta"]["CL"],
            "CD": case_pack["meta"]["CD"],
            "CM": case_pack["meta"]["CM"],
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
    count = 0
    rows = []
    for _, row in df.iterrows():
        rows.append(row)
        if count > NO_CASES:  # Remove these lines while running full dataset if required
            break
        count += 1

    # -----------------------------
    # GLOBAL STATS (train only)
    # -----------------------------
    x_stats = RunningStats(num_features=NODE_IN_DIM)
    cp_stats = RunningStats(num_features=1)
    case_packs = []

    for row in rows:
        pack = build_case_tensors(row, geom_config)
        if pack is None:
            continue
        case_packs.append(pack)
        x_stats.update(pack["x_fp32"])
        if pack["y_cp_fp32"] is not None:
            cp_stats.update(pack["y_cp_fp32"])

    x_mean, x_std = x_stats.finalize()
    cp_mean, cp_std = cp_stats.finalize()

    torch.save(
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "cp_mean": cp_mean,
            "cp_std": cp_std,
        },
        NORM_STATS_FILE,
    )
    print(f"[OK] Saved normalization stats -> {NORM_STATS_FILE}")

    # -----------------------------
    # PER-CASE SAVE
    # -----------------------------
    for pack in case_packs:
        preprocess_case(pack, x_mean, x_std, cp_mean, cp_std)


# if __name__ == "__main__":
#     main()
