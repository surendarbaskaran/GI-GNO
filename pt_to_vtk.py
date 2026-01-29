import pyvista as pv
import torch
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
VTK_IN = r"case_0009.vtk"
PT_IN  = r"case_0009_with_pred.pt"
VTK_OUT = r"case_0009_pred.vtk"

# -----------------------------
# LOAD VTK
# -----------------------------
mesh = pv.read(VTK_IN)
mesh = mesh.connectivity(extraction_mode="largest")

print("Existing point data:", mesh.point_data.keys())
print("VTK nodes:", mesh.n_points)

# -----------------------------
# LOAD INFERENCE .PT
# -----------------------------
data = torch.load(PT_IN, map_location="cpu")

# y_cp_pred: [N, 1] or [N]
y_cp_pred = data.y_cp_pred
if y_cp_pred.ndim == 2:
    y_cp_pred = y_cp_pred.squeeze(1)

y_cp_pred = y_cp_pred.numpy()

print("Pred Cp shape:", y_cp_pred.shape)

# -----------------------------
# SAFETY CHECK
# -----------------------------
assert mesh.n_points == y_cp_pred.shape[0], \
    f"Node mismatch: VTK={mesh.n_points}, PT={y_cp_pred.shape[0]}"

# -----------------------------
# ADD TO VTK
# -----------------------------
mesh.point_data["cp_pred"] = y_cp_pred

# (optional) also add error if GT exists
if hasattr(data, "y_cp"):
    y_cp_gt = data.y_cp.squeeze(1).numpy()
    mesh.point_data["cp_error"] = np.abs(y_cp_pred - y_cp_gt)

# -----------------------------
# SAVE
# -----------------------------
mesh.save(VTK_OUT)

print(f"\n✓ Saved VTK with predictions: {VTK_OUT}")
print("Final point data:", mesh.point_data.keys())
