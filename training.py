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


def tensor_stats(name, tensor):
    tensor = tensor.detach().float().reshape(-1)
    if tensor.numel() == 0:
        return f"{name}: empty"
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return (
        f"{name}: min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={std.item():.6e}"
    )


def log_training_sample_contract(dataset):
    raw_path, raw_item_idx = dataset.samples[0]
    raw = dataset._load_sample(raw_path, raw_item_idx)
    case_name = raw.get("meta", {}).get("case_name", "unknown")
    sample = dataset[0]
    print("==== TRAINING PT PREPROCESSING CHECK ====")
    print(f"PT case_name: {case_name}")
    print(f"PT x shape: {tuple(sample.x.shape)}")
    print(f"PT xyz shape: {tuple(sample.xyz.shape)}")
    print(f"PT pos shape: {tuple(sample.pos.shape)}")
    print(f"PT edge_index shape: {tuple(sample.edge_index.shape)}")
    print(f"PT edge_attr shape: {tuple(sample.edge_attr.shape)}")
    print(f"PT first 3 normalized vertices: {sample.xyz[:3].float().tolist()}")
    print(f"PT feature order: xyz | flow | geom | curvature | normal | local_area | edge_stats")
    print(tensor_stats("PT x_norm", sample.x))
    print(tensor_stats("PT edge_attr", sample.edge_attr))

    assert sample.x.size(1) == NODE_IN_DIM, (
        f"PT feature width mismatch: got {sample.x.size(1)}, expected {NODE_IN_DIM}"
    )
    assert sample.edge_attr.size(1) == EDGE_DIM, (
        f"PT edge_attr width mismatch: got {sample.edge_attr.size(1)}, expected {EDGE_DIM}"
    )
    assert torch.allclose(sample.pos.float(), sample.xyz[:, :2].float(), atol=1e-3), (
        "PT pos must equal PT xyz[:, :2]."
    )
    assert torch.isfinite(sample.x.float()).all(), "PT x contains non-finite values."
    assert torch.isfinite(sample.edge_attr.float()).all(), "PT edge_attr contains non-finite values."
    print("==== END TRAINING PT PREPROCESSING CHECK ====")


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn,
    )


def build_validation_folds(dataset_size):
    if ROTATE_VALIDATION_FOLDS:
        num_folds = max(2, min(VALIDATION_FOLDS, dataset_size))
    else:
        val_size = max(1, int(dataset_size * VAL_SPLIT))
        num_folds = max(2, min(round(dataset_size / val_size), dataset_size))

    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(dataset_size, generator=generator).tolist()
    folds = [perm[i::num_folds] for i in range(num_folds)]
    folds = [fold for fold in folds if fold]
    return folds


def build_fold_datasets(dataset, folds, fold_idx):
    val_indices = folds[fold_idx]
    val_set = set(val_indices)
    train_indices = [
        idx
        for fold in folds
        for idx in fold
        if idx not in val_set
    ]
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )


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
            xyz=raw.get("xyz", raw["x"][:, :3]),
            edge_index=raw["edge_index"],
            edge_attr=raw.get(
                "edge_attr",
                torch.zeros((raw["edge_index"].size(1), EDGE_DIM), dtype=torch.float16),
            ),
            pos=raw["pos"],
            y_cp=raw["y_cp"].float(),
        )

        data.faces = raw["faces"]
        data.cell_normals = raw["cell_normals"]
        data.cell_areas = raw["cell_areas"]

        data.CL_true = torch.tensor(raw["meta"]["CL"], dtype=torch.float32)
        data.CD_true = torch.tensor(raw["meta"]["CD"], dtype=torch.float32)
        data.alpha_deg = torch.tensor(raw["meta"].get("alpha_deg", 0.0), dtype=torch.float32)

        return data


############################################################
# OPTIONAL SMOOTHNESS LOSS
############################################################

def smoothness_loss(pred, edge_index):
    src, dst = edge_index
    return F.mse_loss(pred[src], pred[dst])


def gradient_loss(pred, target, edge_index):
    src, dst = edge_index
    pred_grad = pred[src] - pred[dst]
    target_grad = target[src] - target[dst]
    return F.smooth_l1_loss(pred_grad, target_grad, beta=1.0)


def composite_loss(pred, data, criterion):
    L_field = criterion(pred, data.y_cp)
    L_gradient = gradient_loss(pred.float(), data.y_cp.float(), data.edge_index)
    L_smoothness = smoothness_loss(pred.float(), data.edge_index)

    cp_pred_fp32 = pred.float()
    CL_pred, CD_pred = compute_force_coefficients(cp_pred_fp32, data)
    CL_true = data.CL_true.to(cp_pred_fp32.device)
    CD_true = data.CD_true.to(cp_pred_fp32.device)

    L_CL = ((CL_pred - CL_true) ** 2).mean()
    L_CD = ((CD_pred - CD_true) ** 2).mean()

    loss = (
        LAMBDA_FIELD * L_field
        + LAMBDA_GRADIENT * L_gradient
        + LAMBDA_CL * L_CL
        + LAMBDA_CD * L_CD
        + LAMBDA_SMOOTHNESS * L_smoothness
    )

    return loss



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

        alpha_deg = data.alpha_deg.to(cp_pred.device)
        alpha = torch.deg2rad(alpha_deg)

        CD = Fx_total * torch.cos(alpha) + Fy_total * torch.sin(alpha)
        CL = -Fx_total * torch.sin(alpha) + Fy_total * torch.cos(alpha)
        return CL, CD

    Fx_total = Fx.sum()
    Fy_total = Fy.sum()

    alpha_deg = data.alpha_deg.to(cp_pred.device)
    alpha = torch.deg2rad(alpha_deg)

    CD = Fx_total * torch.cos(alpha) + Fy_total * torch.sin(alpha)
    CL = -Fx_total * torch.sin(alpha) + Fy_total * torch.cos(alpha)

    return CL, CD


def manual_centered_corr(pred, target):
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denom = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
    if denom.item() == 0:
        return 0.0
    return (torch.sum(pred_centered * target_centered) / (denom + 1e-12)).item()


def compute_validation_metrics(pred, data, cp_mean=None, cp_std=None):
    pred_fp32 = pred.float()
    target = data.y_cp.float()

    diff = pred_fp32.squeeze(-1) - target.squeeze(-1)
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(target.squeeze(-1)) + 1e-8)).item()

    corr_before_denorm = manual_centered_corr(pred_fp32, target)
    corr_after_denorm = corr_before_denorm
    if cp_mean is not None and cp_std is not None:
        cp_mean = cp_mean.to(pred_fp32.device, dtype=pred_fp32.dtype)
        cp_std = cp_std.to(pred_fp32.device, dtype=pred_fp32.dtype)
        corr_after_denorm = manual_centered_corr(
            pred_fp32 * cp_std + cp_mean,
            target * cp_std + cp_mean,
        )

    CL_pred, CD_pred = compute_force_coefficients(pred_fp32, data)
    CL_true = data.CL_true.to(pred_fp32.device)
    CD_true = data.CD_true.to(pred_fp32.device)

    cl_error = torch.mean(torch.abs(CL_pred - CL_true)).item()
    cd_error = torch.mean(torch.abs(CD_pred - CD_true)).item()

    return {
        "corr": corr_after_denorm,
        "corr_before_denorm": corr_before_denorm,
        "corr_after_denorm": corr_after_denorm,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "cl_error": cl_error,
        "cd_error": cd_error,
    }

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
    if len(dataset) < 2:
        raise RuntimeError("At least two samples are required for rotating validation.")
    log_training_sample_contract(dataset)

    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu")
    cp_mean = norm_stats["cp_mean"].float()
    cp_std = norm_stats["cp_std"].float()

    folds = build_validation_folds(len(dataset))
    print(
        f"Validation strategy: {'rotating' if ROTATE_VALIDATION_FOLDS else 'fixed'} "
        f"{len(folds)} folds | fold sizes: {[len(fold) for fold in folds]}"
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
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        fourier_bands=FOURIER_BANDS,
        attention_heads=ATTENTION_HEADS,
        attention_query_chunk_size=ATTENTION_QUERY_CHUNK_SIZE,
        attention_max_nodes=ATTENTION_MAX_NODES,
        attention_local_radius=ATTENTION_LOCAL_RADIUS,
    ).to(DEVICE)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
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
        active_fold = (epoch - 1) % len(folds) if ROTATE_VALIDATION_FOLDS else 0
        train_dataset, val_dataset = build_fold_datasets(dataset, folds, active_fold)
        train_loader = make_loader(train_dataset, shuffle=True)
        model.train()
        train_loss = 0.0

        for data in train_loader:

            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16):
                pred = model(data)
                loss = composite_loss(pred, data, criterion)

            if not torch.isfinite(loss):
                print("Skipped batch (NaN/Inf)")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break

            if bad_grad:
                print("NaN gradients detected")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()          # VERY IMPORTANT
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ############################################################
        # VALIDATION
        ############################################################

        val_loss=None
        val_metrics=None

        if epoch % VALIDATE_EVERY == 0 or epoch == EPOCHS:
            model.eval()
            val_loss = 0.0
            val_metrics = {
                "corr": 0.0,
                "corr_before_denorm": 0.0,
                "corr_after_denorm": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "rel_l2": 0.0,
                "cl_error": 0.0,
                "cd_error": 0.0,
            }
            val_loader = make_loader(val_dataset, shuffle=False)
            print(
                f"Validation fold {active_fold + 1}/{len(folds)} | "
                f"train cases: {len(train_dataset)} | val cases: {len(val_dataset)}"
            )

            with torch.no_grad():
                for data in val_loader:

                    data = data.to(DEVICE, non_blocking=True)

                    with autocast("cuda", dtype=torch.float16):
                        pred = model(data)
                        loss = composite_loss(pred, data, criterion)

                    val_loss += loss.item()

                    batch_metrics = compute_validation_metrics(pred, data, cp_mean, cp_std)
                    for key in val_metrics:
                        val_metrics[key] += batch_metrics[key]

            val_loss /= len(val_loader)
            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            scheduler.step()

        ############################################################
        # LOGGING
        ############################################################

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/corr", val_metrics["corr"], epoch)
            writer.add_scalar("val/corr_before_denorm", val_metrics["corr_before_denorm"], epoch)
            writer.add_scalar("val/corr_after_denorm", val_metrics["corr_after_denorm"], epoch)
            writer.add_scalar("val/rmse", val_metrics["rmse"], epoch)
            writer.add_scalar("val/mae", val_metrics["mae"], epoch)
            writer.add_scalar("val/rel_l2", val_metrics["rel_l2"], epoch)
            writer.add_scalar("val/cl_error", val_metrics["cl_error"], epoch)
            writer.add_scalar("val/cd_error", val_metrics["cd_error"], epoch)


        elapsed = time.time() - start

        if val_loss is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6e} | "
                f"Val Loss: {val_loss:.6e} | "
                f"Corr: {val_metrics['corr']:.6f} | "
                f"Corr Before Denorm: {val_metrics['corr_before_denorm']:.6f} | "
                f"Corr After Denorm: {val_metrics['corr_after_denorm']:.6f} | "
                f"RMSE: {val_metrics['rmse']:.6e} | "
                f"MAE: {val_metrics['mae']:.6e} | "
                f"Rel L2: {val_metrics['rel_l2']:.6e} | "
                f"CL Error: {val_metrics['cl_error']:.6e} | "
                f"CD Error: {val_metrics['cd_error']:.6e} | "
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

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(PT_OUT_DIR, f"model_epoch_{epoch}.pt"),
            )

    writer.close()
    print("==== TRAINING COMPLETED ====")


if __name__ == "__main__":
    main()
