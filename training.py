import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import GAGNO
from  config import *
# -------------------------------------------------
# CONFIG (fixed inside file)
# -------------------------------------------------
# DATA_DIR = "ptfiles"
# OUT_DIR = "output"

# NODE_IN_DIM = 18
# HIDDEN_DIM = 32
# OUT_DIM = 1
# NUM_GNN_LAYERS = 5

# GRID_SIZE = (1024, 512)
# LR = 1e-4
# WEIGHT_DECAY = 1e-5
# EPOCHS = 300
# BATCH_SIZE = 1

# USE_SMOOTHNESS_LOSS = False   # enable later if needed
# SMOOTHNESS_WEIGHT = 0.05
# TRAINING_LOG_FILE = f"log_e{EPOCHS}.txt"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_WORKERS = min(8, os.cpu_count())

os.makedirs(PT_OUT_DIR, exist_ok=True)
os.makedirs(f"{PT_OUT_DIR}/bestmodel", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
def log(msg):
    print(msg)
    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(msg + "\n")


# -------------------------------------------------
# DATASET
# -------------------------------------------------
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        data = Data(
            x=raw["x"],                          # FP16
            edge_index=raw["edge_index"],        # int64
            pos=raw["coords_2d"],                # FP16
            y_cp=raw["y_cp"].float(),            # FP32 (important)
        )
        data.meta = raw["meta"]
        return data


# -------------------------------------------------
# OPTIONAL: SMOOTHNESS LOSS (mesh gradient penalty)
# -------------------------------------------------
def smoothness_loss(pred, edge_index):
    src, dst = edge_index
    return F.mse_loss(pred[src], pred[dst])


# -------------------------------------------------
# TRAIN
# -------------------------------------------------
def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log("==== TRAINING STARTED ====")
    log(f"Device: {DEVICE}")
    log("AMP: autocast only (FFT-safe)")
    log(f"""
NODE_IN_DIM = {NODE_IN_DIM}
HIDDEN_DIM = {HIDDEN_DIM}
OUT_DIM = {OUT_DIM}
NUM_GNN_LAYERS = {NUM_GNN_LAYERS}
GRID_SIZE = {GRID_SIZE}
LR = {LR}
WEIGHT_DECAY = {WEIGHT_DECAY}
EPOCHS = {EPOCHS}
BATCH_SIZE = {BATCH_SIZE}
""")

    dataset = GraphDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = GAGNO(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        grid_size=GRID_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Huber is more stable than MSE for Cp
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        start = time.time()

        for bidx, data in enumerate(loader):
            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16):
                pred = model(data)
                loss = criterion(pred, data.y_cp)

                if USE_SMOOTHNESS_LOSS:
                    loss = loss + SMOOTHNESS_WEIGHT * smoothness_loss(
                        pred, data.edge_index
                    )

            # ---- NaN / Inf guard (critical) ----
            if not torch.isfinite(loss):
                log(f"[Epoch {epoch} | Batch {bidx}] Skipped (NaN/Inf loss)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

            log(
                f"[Epoch {epoch} | Batch {bidx}] "
                f"Loss: {loss.item():.6e}"
            )

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
        else:
            avg_loss = float("nan")

        elapsed = time.time() - start

        log(
            f"Epoch {epoch} DONE | "
            f"Avg Loss: {avg_loss:.6e} | "
            f"Valid: {valid_batches} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save epoch model
        if epoch%20==0:
            torch.save(
                model.state_dict(),
                os.path.join(PT_OUT_DIR, f"model_epoch_{epoch}.pt"),
            )

        # Save best model
        if valid_batches > 0 and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(PT_OUT_DIR, "bestmodel/best_model.pt"),
            )
            log(f"✓ New BEST model | Loss: {best_loss:.6e}")

    log("==== TRAINING COMPLETED ====")


if __name__ == "__main__":
    main()

