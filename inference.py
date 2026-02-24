import os
import torch
import pyvista as pv
import pandas as pd
import configparser
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from model import GAGNO
from config import *

os.makedirs(OUT_VTK_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


############################################################
# LOAD CASE DATA
############################################################

case_df = pd.read_csv(CASE_FILE, sep=r"\s+", header=None)
case_df = case_df.set_index(0)

geom_config = configparser.ConfigParser()
geom_config.read(GEOM_PARAM_FILE)


############################################################
# LOAD MODEL
############################################################

model = GAGNO(
    node_in_dim=NODE_IN_DIM,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    num_gnn_layers=NUM_GNN_LAYERS,
    grid_size=GRID_SIZE,
    dropout=DROPOUT,
).to(DEVICE)


def compute_force_coefficients(mesh, cp_values, alpha_deg):
    """
    Compute CL and CD from Cp distribution.
    """

    alpha = np.deg2rad(alpha_deg)

    # Compute surface
    surf = mesh.extract_surface()
    surf = surf.compute_normals(cell_normals=True, point_normals=False)

    areas = surf.compute_cell_sizes()["Area"].values
    normals = surf.cell_data["Normals"]

    # Average Cp per cell
    cp_point = cp_values.numpy()
    cp_cell = cp_point[surf.faces.reshape(-1,4)[:,1:]].mean(axis=1)

    # Force per cell
    Fx = -cp_cell * normals[:,0] * areas
    Fy = -cp_cell * normals[:,1] * areas
    Fz = -cp_cell * normals[:,2] * areas

    # Total force
    Fx_total = Fx.sum()
    Fy_total = Fy.sum()

    # Project to lift and drag
    CD = Fx_total * np.cos(alpha) + Fy_total * np.sin(alpha)
    CL = -Fx_total * np.sin(alpha) + Fy_total * np.cos(alpha)

    return CL, CD

def normalize(x, mean, std):
    return (x - mean) / std


############################################################
# INFERENCE
############################################################

def main():
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu")
    x_mean = norm_stats["x_mean"].float()
    x_std = norm_stats["x_std"].float()
    cp_mean = norm_stats["cp_mean"].float()
    cp_std = norm_stats["cp_std"].float()

    with torch.no_grad(), open(INFRLOG_FILE, "w") as logf:

        logf.write("==== INFERENCE STARTED ====\n")

        for fname in sorted(os.listdir(TEST_VTK_DIR)):

            if not fname.endswith(".vtk"):
                continue

            case_name = fname.replace(".vtk", "")
            print(f"[INFER] {case_name}")

            if case_name not in case_df.index:
                print(f"[SKIP] Not found in case file")
                continue

            row = case_df.loc[case_name]
            geom_name = row[1]

            ############################################################
            # LOAD + CENTER + SCALE GEOMETRY
            ############################################################

            mesh = pv.read(os.path.join(TEST_VTK_DIR, fname))
            mesh = mesh.connectivity(extraction_mode="largest")

            vertices = mesh.points.astype(np.float32)

            center = vertices.mean(axis=0)
            vertices -= center

            scale = np.max(np.abs(vertices))
            vertices /= (scale + 1e-8)

            vertices = torch.tensor(vertices, dtype=torch.float32)

            ############################################################
            # BUILD GRAPH
            ############################################################

            faces = torch.tensor(
                mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.long
            )

            edges = torch.cat(
                [faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]],
                dim=0,
            )

            edge_index = to_undirected(edges.T)

            ############################################################
            # GEOM + FLOW PARAMS
            ############################################################

            geom_params = torch.tensor(
                [geom_config.getfloat(geom_name, k)
                 for k in ["B1","B2","B3","C1","C2","C3","C4","S1","S2","S3"]],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(vertices.size(0), 1)

            flow_params = torch.tensor(
                [float(row[2]), float(row[3]), float(row[4]),
                 float(row[5]), float(row[6])],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(vertices.size(0), 1)

            ############################################################
            # FEATURES
            ############################################################

            x_fp32 = torch.cat(
                [vertices, flow_params, geom_params],
                dim=1,
            )

            x = normalize(x_fp32, x_mean, x_std)

            ############################################################
            # BUILD DATA
            ############################################################

            data = Data(
                x=x,
                edge_index=edge_index,
                pos=vertices[:, :2],   # already in [-1,1]
            ).to(DEVICE)

            ############################################################
            # PREDICT
            ############################################################

            cp_pred_norm = model(data).cpu().squeeze()
            cp_pred = cp_pred_norm * cp_std + cp_mean

            cp_true = torch.tensor(
                mesh.point_data["cp"], dtype=torch.float32
            )
            
            # ------------------------------
            # Field Errors
            # ------------------------------

            diff = cp_pred - cp_true

            mse = torch.mean(diff ** 2).item()
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            mae = torch.mean(torch.abs(diff)).item()
            rel_l2 = (torch.norm(diff) /
                    (torch.norm(cp_true) + 1e-8)).item()

            max_abs_err = torch.max(torch.abs(diff)).item()

            corr = np.corrcoef(
                cp_true.numpy().flatten(),
                cp_pred.numpy().flatten()
            )[0,1]

            # ------------------------------
            # Force Coefficients
            # ------------------------------

            alpha_deg = float(row[6])

            CL_true, CD_true = compute_force_coefficients(mesh, cp_true, alpha_deg)
            CL_pred, CD_pred = compute_force_coefficients(mesh, cp_pred, alpha_deg)

            cl_error = abs(CL_pred - CL_true) / (abs(CL_true) + 1e-8)
            cd_error = abs(CD_pred - CD_true) / (abs(CD_true) + 1e-8)


            logf.write(
                        f"\nCase: {case_name}\n"
                        f"Alpha: {alpha_deg:.2f} deg\n"
                        f"\n--- Field Errors ---\n"
                        f"MSE        : {mse:.6e}\n"
                        f"RMSE       : {rmse:.6e}\n"
                        f"MAE        : {mae:.6e}\n"
                        f"Rel L2     : {rel_l2:.6e}\n"
                        f"Max Abs Err: {max_abs_err:.6e}\n"
                        f"Correlation: {corr:.6f}\n"
                        f"\n--- Force Coefficients ---\n"
                        f"CL_true    : {CL_true:.6e}\n"
                        f"CL_pred    : {CL_pred:.6e}\n"
                        f"CL_rel_err : {cl_error:.6e}\n"
                        f"CD_true    : {CD_true:.6e}\n"
                        f"CD_pred    : {CD_pred:.6e}\n"
                        f"CD_rel_err : {cd_error:.6e}\n"
                        + "-"*50 + "\n"
                    )

            ############################################################
            # SAVE VTK
            ############################################################

            out_mesh = mesh.copy()
            out_mesh.point_data["Cp_pred"] = cp_pred.numpy()
            out_mesh.point_data["Cp_error"] = diff.numpy()

            out_mesh.save(os.path.join(OUT_VTK_DIR, fname))
            print(f"[OK] Saved → {OUT_VTK_DIR}/{fname}")

        logf.write("==== INFERENCE COMPLETED ====\n")

    print("==== INFERENCE COMPLETED ====")


if __name__ == "__main__":
    main()