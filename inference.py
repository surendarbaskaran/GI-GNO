import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import GAGNO

# -----------------------------
# CONFIG (MUST match training)
# -----------------------------
NODE_IN_DIM = 17
HIDDEN_DIM = 64
OUT_DIM = 1
NUM_LAYERS = 2
GRID_SIZE = (64, 64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CKPT = "best_model.pt"
INPUT_PT = "case_0009.pt"
OUTPUT_PT = "case_0009_with_pred.pt"

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = GAGNO(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_gnn_layers=NUM_LAYERS,
        grid_size=GRID_SIZE,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)

    # ---- FIX torch.compile prefix
    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("_orig_mod."):
            new_state[k.replace("_orig_mod.", "")] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model



# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - gt))
    val={
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
    }
    print(val)
    return val


# -----------------------------
# MAIN
# -----------------------------
@torch.no_grad()
def main():
    print("Loading data...")
    raw = torch.load(INPUT_PT)

    if isinstance(raw, dict):
        data = Data(
            x=raw["x"],
            edge_index=raw["edge_index"],
            pos=raw.get("coords_2d"),
            y_cp=raw.get("y_cp"),
            y_cf=raw.get("y_cf"),
            meta=raw.get("meta", {}),
        )
    else:
        data = raw
    
    print(data)
    # data: Data = torch.load(INPUT_PT)

    # Move to device
    # data = data(
    


    print("Loading model...")
    model = load_model()

    print("Running inference...")
    y_cp_pred = model(data)               # [N, 1]

    # -----------------------------
    # ACCURACY (Cp)
    # -----------------------------
    metrics = compute_metrics(
        y_cp_pred,
        data.y_cp
    )

    print("Inference metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper():5s}: {v:.6e}")

    # -----------------------------
    # SAVE BACK IN SAME FORMAT
    # -----------------------------
    out_data = Data(
        x=data.x.cpu(),
        edge_index=data.edge_index.cpu(),
        coords=data.pos.cpu(),
        y_cp=data.y_cp.cpu(),
        y_cf=data.y_cf.cpu(),
        y_cp_pred=y_cp_pred.cpu(),   # <-- added
        meta={
            **data.meta,
            "cp_mse": metrics["mse"],
            "cp_rmse": metrics["rmse"],
            "cp_mae": metrics["mae"],
        }
    )

    torch.save(out_data, OUTPUT_PT)
    print(f"\n✓ Saved inference result: {OUTPUT_PT}")


if __name__ == "__main__":
    main()
