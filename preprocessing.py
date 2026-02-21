import os
import torch
import pyvista as pv
import configparser
import pandas as pd
import numpy as np

from torch_geometric.utils import to_undirected
from config import *


os.makedirs(OUT_DIR, exist_ok=True)


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

def build_case_tensors(row, geom_config):

    case_name = row[0]
    geom_name = row[1]

    vtk_path = os.path.join(RAW_VTK_DIR, f"{case_name}.vtk")
    if not os.path.exists(vtk_path):
        print(f"[SKIP] Missing VTK: {case_name}")
        return None

    mesh = pv.read(vtk_path)
    mesh = mesh.connectivity(extraction_mode="largest")

    ############################################
    # GEOMETRY (CENTER + GLOBAL SCALE)
    ############################################

    vertices = mesh.points.astype(np.float32)

    # Center geometry (critical for symmetry)
    center = vertices.mean(axis=0)
    vertices -= center

    # Global scale (max absolute coordinate)
    scale = np.max(np.abs(vertices))
    vertices /= (scale + 1e-8)

    vertices = torch.tensor(vertices, dtype=torch.float32)

    ############################################
    # FACES → EDGES
    ############################################

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

    ############################################
    # GEOMETRY PARAMETERS
    ############################################

    geom_params = torch.tensor(
        [
            geom_config.getfloat(geom_name, k)
            for k in ["B1","B2","B3","C1","C2","C3","C4","S1","S2","S3"]
        ],
        dtype=torch.float32,
    )

    geom_params = geom_params.unsqueeze(0).repeat(vertices.size(0), 1)

    ############################################
    # FLOW PARAMETERS
    ############################################

    flow_params = torch.tensor(
        [row[2], row[3], row[4], row[5], row[6]],
        dtype=torch.float32,
    )

    flow_params = flow_params.unsqueeze(0).repeat(vertices.size(0), 1)

    ############################################
    # NODE FEATURES
    ############################################

    x_fp32 = torch.cat(
        [vertices, flow_params, geom_params],
        dim=1,
    )

    ############################################
    # COORDS FOR FNO (USE CENTERED XY)
    # Already in [-1,1] after scaling
    ############################################

    coords_2d = vertices[:, :2]  # in [-1,1]

    ############################################
    # TARGET Cp
    ############################################

    if "cp" in mesh.point_data:
        y_cp_fp32 = torch.tensor(
            mesh.point_data["cp"],
            dtype=torch.float32,
        ).unsqueeze(1)
    else:
        y_cp_fp32 = None

    return {
        "case_name": case_name,
        "x_fp32": x_fp32,
        "coords_2d": coords_2d,
        "edge_index": edge_index,
        "y_cp_fp32": y_cp_fp32,
        "meta": {
            "CL": row[7],
            "CD": row[8],
            "CM": row[9],
        },
    }


############################################
# SAVE NORMALIZED CASE
############################################

def preprocess_case(case_pack, x_mean, x_std, cp_mean, cp_std):

    case_name = case_pack["case_name"]
    out_path = os.path.join(OUT_DIR, f"{case_name}.pt")

    print(f"[PROCESS] {case_name}")

    x_norm = normalize(case_pack["x_fp32"], x_mean, x_std).half()

    if case_pack["y_cp_fp32"] is not None:
        y_cp = normalize(
            case_pack["y_cp_fp32"], cp_mean, cp_std
        )
    else:
        y_cp = None

    data = {
        "x": x_norm,
        "edge_index": case_pack["edge_index"],
        "pos": case_pack["coords_2d"].half(),   # renamed to match model
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

    torch.save(data, out_path)
    print(f"[OK] Saved: {out_path}")


############################################
# MAIN
############################################

def main():

    df = pd.read_csv(CASE_FILE, sep=r"\s+")

    geom_config = configparser.ConfigParser()
    geom_config.read(GEOM_PARAM_FILE)

    rows = df.head(NO_CASES)

    x_stats = RunningStats(num_features=NODE_IN_DIM)
    cp_stats = RunningStats(num_features=1)

    case_packs = []

    for _, row in rows.iterrows():

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

    print(f"[OK] Saved normalization stats → {NORM_STATS_FILE}")

    for pack in case_packs:
        preprocess_case(pack, x_mean, x_std, cp_mean, cp_std)