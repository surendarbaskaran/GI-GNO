import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import GAGNO
from  config import *

import warnings
warnings.filterwarnings("ignore") 
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


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.detach().float().view(-1)
        batch_n = x.numel()
        batch_mean = x.mean().item()
        batch_m2 = ((x - batch_mean) ** 2).sum().item()

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

    def std(self):
        if self.n < 2:
            return 0.0
        var = self.m2 / (self.n - 1)
        return float(var ** 0.5)


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
    writer = SummaryWriter(log_dir=TENSORBOARD)
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
    val_split = 0.2
    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu", weights_only=True)
    cp_mean = norm_stats["cp_mean"].float()
    cp_std = norm_stats["cp_std"].float()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        start = time.time()

        for bidx, data in enumerate(train_loader):
            global_step = (epoch - 1) * len(train_loader) + bidx
            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16):
                pred = model(data)
                shape_loss = criterion(pred, data.y_cp)
                loss = (
                    shape_loss
                    + 0.1 * (pred.mean() - data.y_cp.mean()) ** 2
                    + 0.1 * (pred.std() - data.y_cp.std()) ** 2
                )

                writer.add_scalar("debug/gt_std", data.y_cp.std(), global_step)
                writer.add_scalar("debug/pred_mean", pred.mean().item(), global_step)
                writer.add_scalar("debug/pred_std",  pred.std().item(),  global_step)
                writer.add_scalar("train/loss", loss, global_step)
                

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
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            epoch_loss += loss.item()
            valid_batches += 1
            writer.add_scalar("train/loss",loss.item(),global_step)

            log(
                f"[Epoch {epoch} | Batch {bidx}] "
                f"Loss: {loss.item():.6e}"
            )

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
        else:
            avg_loss = float("nan")

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        pred_stats = RunningStats()
        final_cp_stats = RunningStats()

        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    pred = model(data)
                    shape_loss = criterion(pred, data.y_cp)
                    vloss = (
                        shape_loss
                        + 0.1 * (pred.mean() - data.y_cp.mean()) ** 2
                        + 0.1 * (pred.std() - data.y_cp.std()) ** 2
                    )

                val_loss += vloss.item()
                val_batches += 1

                pred_stats.update(pred)
                pred_denorm = pred.detach().cpu() * cp_std + cp_mean
                final_cp_stats.update(pred_denorm)

        if val_batches > 0:
            val_avg_loss = val_loss / val_batches
        else:
            val_avg_loss = float("nan")

        val_pred_std = pred_stats.std()
        val_final_cp_std = final_cp_stats.std()

        writer.add_scalar("val/loss", val_avg_loss, epoch)
        writer.add_scalar("val/pred_std", val_pred_std, epoch)
        writer.add_scalar("val/final_cp_std", val_final_cp_std, epoch)

        writer.add_scalar("train/epoch_loss",avg_loss,epoch)
        writer.flush()
        elapsed = time.time() - start

        log(
            f"Epoch {epoch} DONE | "
            f"Avg Loss: {avg_loss:.6e} | "
            f"Valid: {valid_batches} | "
            f"Time: {elapsed:.2f}s"
        )
        log(
            f"Val | Loss: {val_avg_loss:.6e} | "
            f"Pred Std: {val_pred_std:.6e} | "
            f"Final Cp Std: {val_final_cp_std:.6e}"
        )

        if val_pred_std < 1e-4:
            log(f"[EARLY STOP] pred_std collapsed to {val_pred_std:.6e}")
            break

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

    writer.close()
    log("==== TRAINING COMPLETED ====")


if __name__ == "__main__":
    main()



