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
from config import *

import warnings
warnings.filterwarnings("ignore")


os.makedirs(PT_OUT_DIR, exist_ok=True)
os.makedirs(f"{PT_OUT_DIR}/bestmodel", exist_ok=True)
os.makedirs(TENSORBOARD, exist_ok=True)


############################################################
# DATASET
############################################################

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw = torch.load(self.files[idx], map_location="cpu")

        data = Data(
            x=raw["x"],                     # FP16
            edge_index=raw["edge_index"],   # int64
            pos=raw["pos"],                 # FP16 in [-1,1]
            y_cp=raw["y_cp"].float(),       # FP32 normalized
        )
        return data


############################################################
# OPTIONAL SMOOTHNESS LOSS
############################################################

def smoothness_loss(pred, edge_index):
    src, dst = edge_index
    return F.mse_loss(pred[src], pred[dst])


############################################################
# TRAIN
############################################################

def main():

    writer = SummaryWriter(log_dir=TENSORBOARD)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("==== TRAINING STARTED ====")
    print(f"Device: {DEVICE}")

    dataset = GraphDataset(DATA_DIR)

    # ---- Train/Val split ----
    val_split = 0.2
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    ############################################################
    # MODEL
    ############################################################

    model = GAGNO(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        grid_size=GRID_SIZE,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine LR scheduler (very good for operator learning)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.1,
    )

    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val_loss = float("inf")

    ############################################################
    # TRAIN LOOP
    ############################################################

    for epoch in range(1, EPOCHS + 1):

        start = time.time()
        model.train()
        train_loss = 0.0

        for data in train_loader:

            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16):
                pred = model(data)
                loss = criterion(pred, data.y_cp)

                if USE_SMOOTHNESS_LOSS:
                    loss = loss + SMOOTHNESS_WEIGHT * smoothness_loss(
                        pred, data.edge_index
                    )

            if not torch.isfinite(loss):
                print("Skipped batch (NaN/Inf)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ############################################################
        # VALIDATION
        ############################################################

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:

                data = data.to(DEVICE, non_blocking=True)

                with autocast("cuda", dtype=torch.float16):
                    pred = model(data)
                    loss = criterion(pred, data.y_cp)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        ############################################################
        # LOGGING
        ############################################################

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6e} | "
            f"Val Loss: {val_loss:.6e} | "
            f"Time: {elapsed:.2f}s"
        )

        ############################################################
        # SAVE BEST MODEL (based on validation)
        ############################################################

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(PT_OUT_DIR, "bestmodel/best_model.pt"),
            )
            print("✓ Saved BEST model")

        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(PT_OUT_DIR, f"model_epoch_{epoch}.pt"),
            )

    writer.close()
    print("==== TRAINING COMPLETED ====")


if __name__ == "__main__":
    main()