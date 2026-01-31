# inference.py
# -------------------------------------------------
# Raw VTK → rebuild features (same as preprocessing)
# → model inference → error metrics → predicted VTK
# -------------------------------------------------

import os
import torch
import pyvista as pv
import pandas as pd
import configparser

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from model import GAGNO
from  config import *
# -------------------------------------------------
# CONFIG (must match training)
# -------------------------------------------------
# TEST_VTK_DIR = "test"
# OUT_VTK_DIR = "predicted"

# CASE_FILE = "case_data2.dat"
# GEOM_PARAM_FILE = "geom_params.ini"

# CHECKPOINT = "output/best_model.pt"
# INFRLOG_FILE = "inference_log.txt"

# GRID_SIZE = (1024, 512)
# NODE_IN_DIM = 18
# HIDDEN_DIM = 32
# OUT_DIM = 1
# NUM_GNN_LAYERS = 5

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_VTK_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD CASE + GEOM DATA (NO HEADERS)
# -------------------------------------------------
# -------------------------------------------------
# LOAD CASE DATA (NO HEADERS IN case_data.dat)
# -------------------------------------------------
# -------------------------------------------------
# LOAD CASE DATA (WITH HEADER)
# -------------------------------------------------
# case_df = pd.read_csv(CASE_FILE, sep=r"\s+")

# # normalize column names (important)
# case_df.columns = case_df.columns.str.strip()

# # ensure numeric flow params
# flow_cols = case_df.columns[2:7]
# case_df[flow_cols] = case_df[flow_cols].astype(float)

# # index by case name
# case_df = case_df.set_index(case_df.columns[0])
case_df = pd.read_csv(CASE_FILE, sep=r"\s+", header=None)
case_df = case_df.set_index(0)


geom_config = configparser.ConfigParser()
geom_config.read(GEOM_PARAM_FILE)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = GAGNO(
    node_in_dim=NODE_IN_DIM,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    num_gnn_layers=NUM_GNN_LAYERS,
    grid_size=GRID_SIZE,
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# -------------------------------------------------
# NORMALIZATION UTILS (same as preprocessing)
# -------------------------------------------------
def compute_norm_stats(x):
    mean = x.mean(dim=0)
    std = torch.clamp(x.std(dim=0), min=1e-6)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


# -------------------------------------------------
# INFERENCE LOOP
# -------------------------------------------------
def main():
    with torch.no_grad(), open(INFRLOG_FILE, "w") as logf:
        logf.write("==== INFERENCE STARTED ====\n")

        for fname in sorted(os.listdir(TEST_VTK_DIR)):
            if not fname.endswith(".vtk"):
                continue

            case_name = fname.replace(".vtk", "")
            print(f"[INFER] {case_name}")

            if case_name not in case_df.index:
                print(f"[SKIP] {case_name} not found in case_data.dat")
                continue

            row = case_df.loc[case_name]
            geom_name = row[1]
            print(f"geom_name : {geom_name}")
            print('ROW : ',row)

            # -----------------------------
            # LOAD VTK
            # -----------------------------
            mesh = pv.read(os.path.join(TEST_VTK_DIR, fname))
            mesh = mesh.connectivity(extraction_mode="largest")

            vertices = torch.tensor(mesh.points, dtype=torch.float32)
            faces = torch.tensor(mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.long)

            edges = torch.cat(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
            )
            edge_index = to_undirected(edges.T)

            # -----------------------------
            # GEOMETRY PARAMS
            # -----------------------------
            geom_params = torch.tensor(
                [geom_config.getfloat(geom_name, k)
                for k in ["B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(vertices.size(0), 1)

            # -----------------------------
            # FLOW PARAMS
            # -----------------------------
            flow_params = torch.tensor(
                [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(vertices.size(0), 1)

            # -----------------------------
            # NODE FEATURES (same as preprocessing)
            # -----------------------------
            x_fp32 = torch.cat([vertices, flow_params, geom_params], dim=1)
            x_mean, x_std = compute_norm_stats(x_fp32)
            x = normalize(x_fp32, x_mean, x_std).float()

            # -----------------------------
            # NORMALIZED 2D COORDS
            # -----------------------------
            xy = vertices[:, :2]
            xy_min, xy_max = xy.min(0)[0], xy.max(0)[0]
            coords_2d = ((xy - xy_min) / (xy_max - xy_min + 1e-6)).float()

            # -----------------------------
            # BUILD DATA
            # -----------------------------
            data = Data(
                x=x,
                edge_index=edge_index,
                pos=coords_2d,
            ).to(DEVICE)

            # -----------------------------
            # MODEL INFERENCE (normalized Cp)
            # -----------------------------
            cp_pred_norm = model(data).float().cpu().squeeze()

            # -----------------------------
            # GROUND TRUTH Cp
            # -----------------------------
            cp_true = torch.tensor(
                mesh.point_data["cp"], dtype=torch.float32
            )

            # -----------------------------
            # DE-NORMALIZE PREDICTION
            # -----------------------------
            cp_pred = cp_pred_norm * cp_true.std() + cp_true.mean()

            # -----------------------------
            # ERROR METRICS
            # -----------------------------
            diff = cp_pred - cp_true

            mse = torch.mean(diff ** 2).item()
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            mae = torch.mean(torch.abs(diff)).item()
            rel_l2 = (torch.norm(diff) / (torch.norm(cp_true) + 1e-8)).item()
            max_err = torch.max(torch.abs(diff)).item()
            corr = torch.corrcoef(torch.stack([cp_true, cp_pred]))[0, 1].item()

            # -----------------------------
            # WRITE LOG
            # -----------------------------
            logf.write(
                f"\nCase: {case_name}\n"
                f"Nodes        : {cp_true.numel()}\n"
                f"MSE          : {mse:.6e}\n"
                f"RMSE         : {rmse:.6e}\n"
                f"MAE          : {mae:.6e}\n"
                f"Rel L2 Error : {rel_l2:.6e}\n"
                f"Max Abs Err  : {max_err:.6e}\n"
                f"Correlation  : {corr:.6f}\n"
                + "-" * 40 + "\n"
            )

            # -----------------------------
            # SAVE VTK
            # -----------------------------
            out_mesh = mesh.copy()
            out_mesh.point_data["Cp_pred"] = cp_pred.numpy()
            out_mesh.point_data["Cp_error"] = diff.numpy()
            out_mesh.point_data["Cp_abs_error"] = torch.abs(diff).numpy()

            out_mesh.save(os.path.join(OUT_VTK_DIR, fname))
            print(f"[OK] Saved → {OUT_VTK_DIR}/{fname}")

        logf.write("==== INFERENCE COMPLETED ====\n")

    print("==== INFERENCE COMPLETED ====")
