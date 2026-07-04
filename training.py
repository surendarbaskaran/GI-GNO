import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch_geometric.data import Data, Batch
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

def custom_collate_fn(batch):
    collated = Batch.from_data_list(batch)

    if len(batch) <= 1:
        return collated

    node_offsets = []
    offset = 0
    for item in batch:
        node_offsets.append(offset)
        offset += item.x.size(0)

    offset_faces = []
    offset_normals = []
    offset_areas = []

    for idx, item in enumerate(batch):
        faces = item.faces.clone()
        faces += node_offsets[idx]
        offset_faces.append(faces)
        offset_normals.append(item.cell_normals)
        offset_areas.append(item.cell_areas)

    collated.faces = torch.cat(offset_faces, dim=0)
    collated.cell_normals = torch.cat(offset_normals, dim=0)
    collated.cell_areas = torch.cat(offset_areas, dim=0)

    return collated


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self._cache = {}
        self.samples = []

        for filename in sorted(os.listdir(root)):
            if not filename.endswith(".pt"):
                continue
            if filename == "normalization_stats.pt":
                continue

            path = os.path.join(root, filename)
            if filename.startswith("chunk_"):
                chunk = torch.load(path, map_location="cpu")
                self._cache[path] = chunk

                if isinstance(chunk, list):
                    items = chunk
                elif isinstance(chunk, dict) and "cases" in chunk:
                    items = chunk["cases"]
                else:
                    items = [chunk]

                for item_idx in range(len(items)):
                    self.samples.append((path, item_idx))
            else:
                self.samples.append((path, 0))

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, path, item_idx):
        if path not in self._cache:
            self._cache[path] = torch.load(path, map_location="cpu")

        raw = self._cache[path]
        if isinstance(raw, list):
            return raw[item_idx]
        if isinstance(raw, dict) and "cases" in raw:
            return raw["cases"][item_idx]
        return raw

    def __getitem__(self, idx):
        path, item_idx = self.samples[idx]
        raw = self._load_sample(path, item_idx)

        data = Data(
            x=raw["x"],
            edge_index=raw["edge_index"],
            pos=raw["pos"],
            y_cp=raw["y_cp"].float(),
        )

        data.faces = raw["faces"]
        data.cell_normals = raw["cell_normals"]
        data.cell_areas = raw["cell_areas"]

        data.CL_true = torch.tensor(raw["meta"]["CL"], dtype=torch.float32)
        data.CD_true = torch.tensor(raw["meta"]["CD"], dtype=torch.float32)

        return data


############################################################
# OPTIONAL SMOOTHNESS LOSS
############################################################

def smoothness_loss(pred, edge_index):
    src, dst = edge_index
    return F.mse_loss(pred[src], pred[dst])



def compute_force_coefficients(cp_pred, data):
    """
    cp_pred: (N_nodes, 1)
    data: batch element
    """

    faces = data.faces
    normals = data.cell_normals
    areas = data.cell_areas

    cp_pred = cp_pred.squeeze(-1)

    # Convert point Cp → cell Cp
    cp_cell = cp_pred[faces].mean(dim=1)

    # Force per cell
    Fx = -cp_cell * normals[:, 0] * areas
    Fy = -cp_cell * normals[:, 1] * areas

    if hasattr(data, "batch") and data.batch is not None and data.batch.numel() > 0:
        num_graphs = int(data.batch.max().item()) + 1
        cell_graph_ids = data.batch[faces[:, 0]]

        Fx_total = torch.zeros(num_graphs, device=cp_pred.device, dtype=cp_pred.dtype)
        Fy_total = torch.zeros(num_graphs, device=cp_pred.device, dtype=cp_pred.dtype)
        Fx_total.index_add_(0, cell_graph_ids, Fx)
        Fy_total.index_add_(0, cell_graph_ids, Fy)

        alpha_deg = []
        for graph_idx in range(num_graphs):
            graph_nodes = data.x[data.batch == graph_idx]
            if graph_nodes.numel() > 0:
                alpha_deg.append(graph_nodes[0, 5])
            else:
                alpha_deg.append(torch.tensor(0.0, device=cp_pred.device))

        alpha_deg = torch.stack(alpha_deg, dim=0).to(cp_pred.device)
        alpha = torch.deg2rad(alpha_deg)

        CD = Fx_total * torch.cos(alpha) + Fy_total * torch.sin(alpha)
        CL = -Fx_total * torch.sin(alpha) + Fy_total * torch.cos(alpha)
        return CL, CD

    Fx_total = Fx.sum()
    Fy_total = Fy.sum()

    alpha_deg = data.x[0, 5]
    alpha = torch.deg2rad(alpha_deg)

    CD = Fx_total * torch.cos(alpha) + Fy_total * torch.sin(alpha)
    CL = -Fx_total * torch.sin(alpha) + Fy_total * torch.cos(alpha)

    return CL, CD

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

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training samples found in {DATA_DIR}. Run preprocessing first to create chunk files."
        )

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
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=custom_collate_fn,
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
    model = torch.compile(model)

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
                # --------------------------
                # FIELD LOSS
                # --------------------------
                L_field = criterion(pred, data.y_cp)

                # --------------------------
                # FORCE LOSS (FULL PRECISION)
                # --------------------------
                cp_pred_fp32 = pred.float()
                cp_true_fp32 = data.y_cp.float()

                CL_pred, CD_pred = compute_force_coefficients(cp_pred_fp32, data)
                CL_true = data.CL_true.to(DEVICE)
                CD_true = data.CD_true.to(DEVICE)

                L_CL = ((CL_pred - CL_true) ** 2).mean()
                L_CD = ((CD_pred - CD_true) ** 2).mean()

                # --------------------------
                # TOTAL LOSS
                # --------------------------
                LAMBDA_FIELD = 1.0
                LAMBDA_CL = 0.1
                LAMBDA_CD = 0.1

                loss = (
                    LAMBDA_FIELD * L_field
                    + LAMBDA_CL * L_CL 
                    + LAMBDA_CD * L_CD
                )

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

        val_loss=None

        if epoch % VALIDATE_EVERY == 0 or epoch == EPOCHS:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data in val_loader:

                    data = data.to(DEVICE, non_blocking=True)

                    with autocast("cuda", dtype=torch.float16):
                        pred = model(data)
                        # loss = criterion(pred, data.y_cp)
                        # --------------------------
                        # FIELD LOSS
                        # --------------------------
                        L_field = criterion(pred, data.y_cp)

                        # --------------------------
                        # FORCE LOSS (FULL PRECISION)
                        # --------------------------
                        cp_pred_fp32 = pred.float()
                        cp_true_fp32 = data.y_cp.float()

                        CL_pred, CD_pred = compute_force_coefficients(cp_pred_fp32, data)
                        CL_true = data.CL_true.to(DEVICE)
                        CD_true = data.CD_true.to(DEVICE)

                        L_CL = ((CL_pred - CL_true) ** 2).mean()
                        L_CD = ((CD_pred - CD_true) ** 2).mean()

                        # --------------------------
                        # TOTAL LOSS
                        # --------------------------
                        LAMBDA_FIELD = 1.0
                        LAMBDA_CL = 0.1
                        LAMBDA_CD = 0.1

                        loss = (
                            LAMBDA_FIELD * L_field
                            + LAMBDA_CL * L_CL
                            + LAMBDA_CD * L_CD
                        )

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            scheduler.step()

        ############################################################
        # LOGGING
        ############################################################

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss", val_loss, epoch)


        elapsed = time.time() - start

        if val_loss is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6e} | "
                f"Val Loss: {val_loss:.6e} | "
                f"Time: {elapsed:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6e} | "
                f"Time: {elapsed:.2f}s"
            )

        ############################################################
        # SAVE BEST MODEL (based on validation)
        ############################################################

        if val_loss is not None and val_loss < best_val_loss:
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