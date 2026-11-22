import os
import re
import gc
import csv
import time
import glob
import random
import argparse
from collections import OrderedDict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter

from model import EncoderGNODecoder, EncoderDecoder, GAGNO
from config import *

import warnings
warnings.filterwarnings("ignore")


os.makedirs(PT_OUT_DIR, exist_ok=True)
os.makedirs(f"{PT_OUT_DIR}/bestmodel", exist_ok=True)
os.makedirs(TENSORBOARD, exist_ok=True)


############################################################
# VRAM & MEMORY MANAGEMENT (<= 80 GB CONSTRAINT)
############################################################

def configure_vram_limit(max_vram_gb=MAX_VRAM_GB):
    """Optionally enforce a GPU VRAM ceiling. When max_vram_gb is None or <= 0, use full GPU VRAM."""
    if not torch.cuda.is_available() or DEVICE != "cuda":
        return
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        if max_vram_gb is not None and max_vram_gb > 0:
            fraction = min(0.98, float(max_vram_gb) / max(total_gb, 1e-6))
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            effective_cap = total_gb * fraction
            print(
                f"[VRAM LIMIT] GPU: {props.name} | Total VRAM: {total_gb:.2f} GB | "
                f"Configured Max Cap: {max_vram_gb:.2f} GB (Effective Limit: {effective_cap:.2f} GB, fraction={fraction:.4f})"
            )
        else:
            print(f"[VRAM] GPU: {props.name} | Total VRAM: {total_gb:.2f} GB | Cap: Unlimited (full GPU potential)")
    except Exception as e:
        print(f"[VRAM LIMIT] Warning: could not configure memory fraction: {e}")


def get_vram_stats_str():
    if not torch.cuda.is_available() or DEVICE != "cuda":
        return "VRAM: CPU"
    alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
    peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    return f"VRAM Alloc: {alloc_gb:.2f}GB | Rsvd: {reserved_gb:.2f}GB | Peak: {peak_gb:.2f}GB"


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


def compute_edge_attr_on_the_fly(vertices, edge_index, out_dtype=torch.float16):
    """Fast on-the-fly edge attribute computation (runs on GPU to avoid CPU/PCIe bottleneck)."""
    src, dst = edge_index[0], edge_index[1]
    v_fp32 = vertices.float()
    rel_xyz = v_fp32[dst] - v_fp32[src]
    distance = torch.sqrt(torch.sum(rel_xyz * rel_xyz, dim=1, keepdim=True) + 1e-12)
    direction = rel_xyz / (distance + 1e-8)
    return torch.cat([distance, rel_xyz, distance, direction], dim=1).to(dtype=out_dtype)


def prepare_batch_on_device(data, device, edge_dtype=torch.float16):
    """Transfer batch asynchronously to GPU and compute edge_attr on GPU when not stored on disk."""
    data = data.to(device, non_blocking=True)
    if getattr(data, "edge_attr", None) is None:
        data.edge_attr = compute_edge_attr_on_the_fly(data.xyz, data.edge_index, out_dtype=edge_dtype)
    return data


def log_training_sample_contract(dataset):
    raw_path, raw_item_idx = dataset.samples[0]
    raw = dataset._load_sample(raw_path, raw_item_idx)
    case_name = raw.get("meta", {}).get("case_name", "unknown")
    sample = dataset[0]
    if getattr(sample, "edge_attr", None) is None:
        sample.edge_attr = compute_edge_attr_on_the_fly(sample.xyz, sample.edge_index)
    print("==== TRAINING PT PREPROCESSING CHECK ====")
    print(f"PT case_name: {case_name}")
    print(f"PT x shape: {tuple(sample.x.shape)} (dtype={sample.x.dtype})")
    print(f"PT xyz shape: {tuple(sample.xyz.shape)} (dtype={sample.xyz.dtype})")
    print(f"PT pos shape: {tuple(sample.pos.shape)} (dtype={sample.pos.dtype})")
    print(f"PT edge_index shape: {tuple(sample.edge_index.shape)} (dtype={sample.edge_index.dtype})")
    print(f"PT edge_attr shape: {tuple(sample.edge_attr.shape)} (dtype={sample.edge_attr.dtype})")
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
    pos_fp32 = sample.pos.float()
    xyz_fp32 = sample.xyz.float()
    assert (
        torch.allclose(pos_fp32, xyz_fp32, atol=1e-3)
        or torch.allclose(pos_fp32, xyz_fp32[:, :2], atol=1e-3)
    ), "PT pos must correspond to node coordinates."
    assert sample.xyz[:, 0].min() >= -1e-4, "Vertices must start at >= 0.0"
    assert sample.xyz[:, 0].max() <= 1.0001, "Vertices must end at <= 1.0"
    assert torch.isfinite(sample.x.float()).all(), "PT x contains non-finite values."
    assert torch.isfinite(sample.edge_attr.float()).all(), "PT edge_attr contains non-finite values."
    print("==== END TRAINING PT PREPROCESSING CHECK ====")


############################################################
# SCALABLE 10,000-CASE DATASET WITH BOUNDED LRU CHUNK CACHE
############################################################

class ChunkAwareSubsetSampler(torch.utils.data.Sampler):
    """Orders samples by chunk so each chunk file is loaded from disk only ONCE per epoch.
    When num_workers > 1, partitions chunks disjointly across workers and interleaves batches
    round-robin to match DataLoader worker dispatch — eliminating redundant multi-worker disk I/O.
    """

    def __init__(
        self,
        subset,
        shuffle=True,
        window_chunks=MAX_CACHED_CHUNKS,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=42,
    ):
        self.subset = subset
        self.shuffle = shuffle
        self.window_chunks = max(1, int(window_chunks))
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
        self.seed = seed
        self.epoch = 0

        # Group subset local indices by underlying chunk file path
        chunk_to_local_indices = OrderedDict()
        base_dataset = subset.dataset
        for local_idx, global_idx in enumerate(subset.indices):
            chunk_path, _ = base_dataset.samples[global_idx]
            if chunk_path not in chunk_to_local_indices:
                chunk_to_local_indices[chunk_path] = []
            chunk_to_local_indices[chunk_path].append(local_idx)

        self.chunk_groups = list(chunk_to_local_indices.values())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _flatten_groups(self, groups, rng):
        if not self.shuffle:
            return [idx for g in groups for idx in g]
        ordered = []
        for start in range(0, len(groups), self.window_chunks):
            window = [idx for g in groups[start:start + self.window_chunks] for idx in g]
            rng.shuffle(window)
            ordered.extend(window)
        return ordered

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = [list(g) for g in self.chunk_groups]
        if self.shuffle:
            rng.shuffle(groups)

        # Single-process or single-worker path
        if self.num_workers <= 1 or len(groups) < self.num_workers:
            for idx in self._flatten_groups(groups, rng):
                yield idx
            return

        # Multi-worker path: assign disjoint chunk shards to each worker and interleave
        # full batches round-robin so batch `k` lands on worker `k % num_workers`.
        w_count = self.num_workers
        b_size = self.batch_size
        worker_full_batches = []
        tail_indices = []

        for w in range(w_count):
            w_groups = groups[w::w_count]
            w_indices = self._flatten_groups(w_groups, rng)
            n_full = (len(w_indices) // b_size) * b_size
            batches = [w_indices[i:i + b_size] for i in range(0, n_full, b_size)]
            worker_full_batches.append(batches)
            if n_full < len(w_indices):
                tail_indices.extend(w_indices[n_full:])

        max_steps = max((len(b) for b in worker_full_batches), default=0)
        for step in range(max_steps):
            for w in range(w_count):
                if step < len(worker_full_batches[w]):
                    for idx in worker_full_batches[w][step]:
                        yield idx

        for idx in tail_indices:
            yield idx

    def __len__(self):
        return len(self.subset)


def make_loader(subset, shuffle, persistent=True):
    use_workers = NUM_WORKERS if len(subset) >= CHUNK_SIZE else 0
    sampler = ChunkAwareSubsetSampler(
        subset,
        shuffle=shuffle,
        window_chunks=MAX_CACHED_CHUNKS,
        batch_size=BATCH_SIZE,
        num_workers=use_workers,
    )
    prefetch = PREFETCH_FACTOR if use_workers > 0 else None
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=use_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(persistent and use_workers > 0),
        prefetch_factor=prefetch,
        collate_fn=custom_collate_fn,
    ), sampler


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


def custom_collate_fn(batch):
    collated = Batch.from_data_list(batch)

    if len(batch) <= 1:
        return collated

    offset_faces = []
    offset = 0
    for item in batch:
        if offset == 0:
            offset_faces.append(item.faces)
        else:
            offset_faces.append(item.faces + offset)
        offset += item.x.size(0)

    collated.faces = torch.cat(offset_faces, dim=0)
    return collated


class GraphDataset(torch.utils.data.Dataset):
    """Memory-bounded dataset for 10,000+ CFD cases using manifest indexing and an LRU chunk cache."""

    def __init__(self, root, max_cached_chunks=MAX_CACHED_CHUNKS):
        self.root = root
        self.max_cached_chunks = max(1, int(max_cached_chunks))
        self._cache = OrderedDict()
        self.samples = []

        manifest_path = os.path.join(root, "manifest.pt")
        if os.path.exists(manifest_path):
            try:
                manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
                for chunk_info in manifest.get("chunks", []):
                    c_path = os.path.join(root, chunk_info["chunk_file"])
                    if os.path.exists(c_path):
                        for item_idx in range(int(chunk_info["num_cases"])):
                            self.samples.append((c_path, item_idx))
            except Exception as e:
                print(f"[DATASET] Warning: could not read manifest.pt ({e}), scanning directory...")
                self.samples.clear()

        if not self.samples:
            manifest_chunks = []
            for filename in sorted(os.listdir(root)):
                if not filename.endswith(".pt"):
                    continue
                if filename in ("normalization_stats.pt", "manifest.pt"):
                    continue

                path = os.path.join(root, filename)
                if filename.startswith("chunk_"):
                    # Load one chunk at a time to count items, then IMMEDIATELY free it from RAM
                    chunk = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(chunk, list):
                        num_items = len(chunk)
                    elif isinstance(chunk, dict) and "cases" in chunk:
                        num_items = len(chunk["cases"])
                    else:
                        num_items = 1
                    del chunk

                    for item_idx in range(num_items):
                        self.samples.append((path, item_idx))
                    manifest_chunks.append({"chunk_file": filename, "num_cases": num_items})
                else:
                    self.samples.append((path, 0))

            if manifest_chunks:
                try:
                    torch.save(
                        {"total_cases": len(self.samples), "chunks": manifest_chunks},
                        manifest_path,
                    )
                except Exception:
                    pass
            gc.collect()

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, path, item_idx):
        if path in self._cache:
            self._cache.move_to_end(path)
            raw = self._cache[path]
        else:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self._cache[path] = raw
            while len(self._cache) > self.max_cached_chunks:
                _, evicted = self._cache.popitem(last=False)
                del evicted

        if isinstance(raw, list):
            return raw[item_idx]
        if isinstance(raw, dict) and "cases" in raw:
            return raw["cases"][item_idx]
        return raw

    def __getitem__(self, idx):
        path, item_idx = self.samples[idx]
        raw = self._load_sample(path, item_idx)

        xyz = raw.get("xyz", raw.get("pos", raw["x"][:, :3])).float()
        pos = raw.get("pos", xyz).float()
        edge_index = raw["edge_index"].long()

        edge_attr = raw.get("edge_attr", None)

        data = Data(
            x=raw["x"],
            xyz=xyz,
            edge_index=edge_index,
            pos=pos,
            y_cp=raw["y_cp"].float(),
        )
        if edge_attr is not None:
            data.edge_attr = edge_attr

        faces = raw["faces"].long()
        data.faces = faces
        data.cell_normals = raw["cell_normals"].float()
        data.cell_areas = raw["cell_areas"].float()
        data.cell_centers = raw.get(
            "cell_centers",
            pos[faces].mean(dim=1),
        ).float()
        if raw.get("y_cf") is not None:
            data.y_cf = raw["y_cf"].float()

        meta = raw.get("meta", {})
        data.CL_table = torch.tensor(meta.get("CL", 0.0), dtype=torch.float32)
        data.CD_table = torch.tensor(meta.get("CD", 0.0), dtype=torch.float32)
        data.CMy_table = torch.tensor(meta.get("CM", 0.0), dtype=torch.float32)

        # Store precomputed Cp-integrated targets if present in meta
        if "CL_cp" in meta and "CD_cp" in meta and "CM_cp" in meta:
            data.CL_cp_meta = torch.tensor(meta["CL_cp"], dtype=torch.float32)
            data.CD_cp_meta = torch.tensor(meta["CD_cp"], dtype=torch.float32)
            data.CMy_cp_meta = torch.tensor(meta["CM_cp"], dtype=torch.float32)

        data.CL_true = data.CL_table
        data.CD_true = data.CD_table
        data.CMy_true = data.CMy_table
        data.alpha_deg = torch.tensor(meta.get("alpha_deg", 0.0), dtype=torch.float32)
        data.beta_deg = torch.tensor(meta.get("beta_deg", 0.0), dtype=torch.float32)

        return data


############################################################
# PHYSICS-ALIGNED Cp LOSSES & SURFACE FORCE INTEGRATION
############################################################

def smoothness_loss(pred, edge_index, chunk_size=GNO_EDGE_CHUNK_SIZE):
    src, dst = edge_index
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return pred.new_tensor(0.0)
    if not chunk_size or num_edges <= chunk_size:
        return F.mse_loss(pred[src], pred[dst])

    total_sq = pred.new_tensor(0.0, dtype=torch.float32)
    for start in range(0, num_edges, chunk_size):
        end = min(start + chunk_size, num_edges)
        diff = pred[src[start:end]].float() - pred[dst[start:end]].float()
        total_sq = total_sq + torch.sum(diff * diff)
    return total_sq / float(num_edges * pred.size(-1))


def gradient_loss(pred, target, edge_index, chunk_size=GNO_EDGE_CHUNK_SIZE):
    """Chunked spatial edge-gradient loss with sharp beta=0.02 so leading-edge suction peaks
    and shock gradients receive strong supervision without allocating full-edge temporary tensors.
    """
    src, dst = edge_index
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return pred.new_tensor(0.0)
    if not chunk_size or num_edges <= chunk_size:
        pred_grad = pred[src] - pred[dst]
        target_grad = target[src] - target[dst]
        return F.smooth_l1_loss(pred_grad, target_grad, beta=0.02)

    total_loss = pred.new_tensor(0.0, dtype=torch.float32)
    for start in range(0, num_edges, chunk_size):
        end = min(start + chunk_size, num_edges)
        s_c, d_c = src[start:end], dst[start:end]
        pred_grad = pred[s_c].float() - pred[d_c].float()
        target_grad = target[s_c].float() - target[d_c].float()
        total_loss = total_loss + F.smooth_l1_loss(pred_grad, target_grad, beta=0.02, reduction="sum")
    return total_loss / float(num_edges * pred.size(-1))


def aero_weighted_cp_loss(cp_pred_norm, y_cp_norm, data):
    """Local cell-level aerodynamic normal & area-weighted Cp loss.
    Weights each surface triangle's Cp error by its physical contribution to Drag (|n_D| * Area),
    Lift (|n_L| * Area), and suction-peak magnitude (1 + |Cp_norm|).
    By the triangle inequality, minimizing this dense per-cell loss strictly upper-bounds and
    minimizes integrated CL and CD errors without the rank-1 oscillations of global scalar force loss.
    """
    faces = data.faces
    normals = data.cell_normals.to(cp_pred_norm.device, dtype=torch.float32)
    areas = data.cell_areas.to(cp_pred_norm.device, dtype=torch.float32)

    cp_pred_cell = cp_pred_norm.squeeze(-1)[faces].mean(dim=1)
    cp_true_cell = y_cp_norm.squeeze(-1)[faces].mean(dim=1)

    alpha = torch.deg2rad(data.alpha_deg.to(cp_pred_norm.device, dtype=torch.float32))
    beta_deg = getattr(data, "beta_deg", torch.zeros_like(alpha))
    beta = torch.deg2rad(beta_deg.to(cp_pred_norm.device, dtype=torch.float32))

    if hasattr(data, "batch") and data.batch is not None and data.batch.numel() > 0:
        cell_graph_ids = data.batch[faces[:, 0]]
        alpha_c = alpha.view(-1)[cell_graph_ids]
        beta_c = beta.view(-1)[cell_graph_ids]
    else:
        alpha_c = alpha.reshape(-1)[0]
        beta_c = beta.reshape(-1)[0]

    # Drag and Lift directional normal projections per cell
    n_drag = (
        normals[:, 0] * torch.cos(alpha_c) * torch.cos(beta_c)
        + normals[:, 1] * torch.sin(beta_c)
        + normals[:, 2] * torch.sin(alpha_c) * torch.cos(beta_c)
    ).abs()
    n_lift = (-normals[:, 0] * torch.sin(alpha_c) + normals[:, 2] * torch.cos(alpha_c)).abs()

    # Emphasize drag-producing (streamwise normal |n_drag|) and lift-producing cells + suction peak
    drag_weight = n_drag * areas
    drag_weight = drag_weight / (drag_weight.mean() + 1e-8)

    lift_weight = n_lift * areas
    lift_weight = lift_weight / (lift_weight.mean() + 1e-8)

    suction_weight = 1.0 + 0.5 * cp_true_cell.abs()
    cell_weights = (0.65 * drag_weight + 0.35 * lift_weight) * suction_weight
    cell_weights = cell_weights.detach()

    cell_err = F.smooth_l1_loss(cp_pred_cell, cp_true_cell, beta=0.05, reduction="none")
    return torch.mean(cell_weights * cell_err)


def composite_loss(pred, data, criterion, cp_mean, cp_std):
    pred_fp32 = pred.float()
    cp_pred_norm = pred_fp32[:, :1] if pred_fp32.size(-1) > 1 else pred_fp32
    y_cp_norm = data.y_cp.float()

    L_field = criterion(cp_pred_norm, y_cp_norm)
    L_gradient = gradient_loss(cp_pred_norm, y_cp_norm, data.edge_index) if LAMBDA_GRADIENT > 0 else pred_fp32.new_tensor(0.0)
    L_aero_cp = (
        aero_weighted_cp_loss(cp_pred_norm, y_cp_norm, data)
        if LAMBDA_AERO_WEIGHTED_CP > 0
        else pred_fp32.new_tensor(0.0)
    )
    L_smoothness = (
        smoothness_loss(cp_pred_norm, data.edge_index)
        if LAMBDA_SMOOTHNESS > 0
        else pred_fp32.new_tensor(0.0)
    )

    loss = (
        LAMBDA_FIELD * L_field
        + LAMBDA_GRADIENT * L_gradient
        + LAMBDA_AERO_WEIGHTED_CP * L_aero_cp
        + LAMBDA_SMOOTHNESS * L_smoothness
    )

    if not CP_ONLY_TRAINING and (LAMBDA_CL > 0 or LAMBDA_CD > 0 or LAMBDA_CMY > 0):
        cp_mean_dev = cp_mean.to(pred_fp32.device, dtype=pred_fp32.dtype)
        cp_std_dev = cp_std.to(pred_fp32.device, dtype=pred_fp32.dtype)
        cp_pred_denorm = cp_pred_norm * cp_std_dev + cp_mean_dev
        if pred_fp32.size(-1) == 4:
            pred_for_forces = torch.cat([cp_pred_denorm, pred_fp32[:, 1:4]], dim=-1)
        else:
            pred_for_forces = cp_pred_denorm

        CL_pred, CD_pred, CMy_pred = compute_force_coefficients(pred_for_forces, data)
        if USE_CP_INTEGRATED_FORCE_TARGETS:
            cp_true_denorm = y_cp_norm * cp_std_dev + cp_mean_dev
            CL_target, CD_target, CMy_target = compute_force_coefficients(cp_true_denorm.detach(), data)
        else:
            CL_target = data.CL_true.to(pred_fp32.device, dtype=pred_fp32.dtype)
            CD_target = data.CD_true.to(pred_fp32.device, dtype=pred_fp32.dtype)
            CMy_target = data.CMy_true.to(pred_fp32.device, dtype=pred_fp32.dtype)

        eps = 1e-3
        L_CL = (
            F.smooth_l1_loss(CL_pred, CL_target, beta=0.005, reduction="none")
            / (torch.abs(CL_target) + eps)
        ).mean()
        L_CD = (
            F.smooth_l1_loss(CD_pred, CD_target, beta=0.001, reduction="none")
            / (torch.abs(CD_target) + eps)
        ).mean()
        L_CMy = (
            F.smooth_l1_loss(CMy_pred, CMy_target, beta=0.002, reduction="none")
            / (torch.abs(CMy_target) + eps)
        ).mean()

        loss = loss + LAMBDA_CL * L_CL + LAMBDA_CD * L_CD + LAMBDA_CMY * L_CMy

    return loss


def compute_force_coefficients(cp_pred, data):
    """
    Compute aerodynamic lift (CL), drag (CD), and pitching moment (CMy) coefficients
    from physical (denormalized) Cp (and optional Cf).
    Coordinate system:
      - X (axis 0): chordwise (freestream at alpha=0)
      - Y (axis 1): spanwise (left-to-right wingtip)
      - Z (axis 2): vertical / thickness (lift direction at alpha=0)
    Surface normals point outward; pressure acts inward (-Cp * n * Area).
    """

    faces = data.faces
    normals = data.cell_normals.to(cp_pred.device, dtype=cp_pred.dtype)
    areas = data.cell_areas.to(cp_pred.device, dtype=cp_pred.dtype)
    centers = data.cell_centers.to(cp_pred.device, dtype=cp_pred.dtype)

    if cp_pred.dim() == 2 and cp_pred.size(-1) == 4:
        cp_node = cp_pred[:, 0]
        cf_node = cp_pred[:, 1:4]
        cp_cell = cp_node[faces].mean(dim=1)
        cf_cell = cf_node[faces].mean(dim=1)
    else:
        cp_node = cp_pred.squeeze(-1)
        cp_cell = cp_node[faces].mean(dim=1)
        if hasattr(data, "y_cf") and data.y_cf is not None:
            cf_node = data.y_cf.to(cp_pred.device, dtype=cp_pred.dtype)
            cf_cell = cf_node[faces].mean(dim=1)
        else:
            cf_cell = torch.zeros((faces.size(0), 3), device=cp_pred.device, dtype=cp_pred.dtype)

    # Total force vector per cell: pressure (-Cp * n) + skin friction (Cf), scaled by Area
    dF_total = (-cp_cell[:, None] * normals + cf_cell) * areas[:, None]
    cmy_cell = (centers[:, 2] - Z_REF) * dF_total[:, 0] - (centers[:, 0] - X_REF) * dF_total[:, 2]

    if hasattr(data, "batch") and data.batch is not None and data.batch.numel() > 0:
        num_graphs = int(data.batch.max().item()) + 1
        cell_graph_ids = data.batch[faces[:, 0]]

        F_total = torch.zeros((num_graphs, 3), device=cp_pred.device, dtype=cp_pred.dtype)
        F_total.index_add_(0, cell_graph_ids, dF_total)

        CMy_total = torch.zeros(num_graphs, device=cp_pred.device, dtype=cp_pred.dtype)
        CMy_total.index_add_(0, cell_graph_ids, cmy_cell)

        CX = F_total[:, 0] / S_REF
        CY = F_total[:, 1] / S_REF
        CZ = F_total[:, 2] / S_REF
        CMy = CMy_total / (S_REF * C_REF)

        alpha = torch.deg2rad(data.alpha_deg.to(cp_pred.device, dtype=cp_pred.dtype))
        beta_deg = getattr(data, "beta_deg", torch.zeros_like(alpha))
        beta = torch.deg2rad(beta_deg.to(cp_pred.device, dtype=cp_pred.dtype))

        CD = CX * torch.cos(alpha) * torch.cos(beta) + CY * torch.sin(beta) + CZ * torch.sin(alpha) * torch.cos(beta)
        CL = -CX * torch.sin(alpha) + CZ * torch.cos(alpha)
        return CL, CD, CMy

    F_total = dF_total.sum(dim=0)
    CX, CY, CZ = F_total[0] / S_REF, F_total[1] / S_REF, F_total[2] / S_REF
    CMy = cmy_cell.sum() / (S_REF * C_REF)

    alpha = torch.deg2rad(data.alpha_deg.to(cp_pred.device, dtype=cp_pred.dtype))
    beta_deg = getattr(data, "beta_deg", torch.zeros_like(alpha))
    beta = torch.deg2rad(beta_deg.to(cp_pred.device, dtype=cp_pred.dtype))

    CD = CX * torch.cos(alpha) * torch.cos(beta) + CY * torch.sin(beta) + CZ * torch.sin(alpha) * torch.cos(beta)
    CL = -CX * torch.sin(alpha) + CZ * torch.cos(alpha)

    return CL, CD, CMy


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
    cp_pred_norm = pred_fp32[:, :1] if pred_fp32.size(-1) > 1 else pred_fp32
    target = data.y_cp.float()

    diff = cp_pred_norm.squeeze(-1) - target.squeeze(-1)
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(target.squeeze(-1)) + 1e-8)).item()

    corr_before_denorm = manual_centered_corr(cp_pred_norm, target)
    corr_after_denorm = corr_before_denorm
    if cp_mean is not None and cp_std is not None:
        cp_mean_dev = cp_mean.to(pred_fp32.device, dtype=pred_fp32.dtype)
        cp_std_dev = cp_std.to(pred_fp32.device, dtype=pred_fp32.dtype)
        pred_denorm = cp_pred_norm * cp_std_dev + cp_mean_dev
        target_denorm = target * cp_std_dev + cp_mean_dev
        corr_after_denorm = manual_centered_corr(pred_denorm, target_denorm)
    else:
        pred_denorm = cp_pred_norm
        target_denorm = target

    if pred_fp32.size(-1) == 4:
        pred_for_forces = torch.cat([pred_denorm, pred_fp32[:, 1:4]], dim=-1)
    else:
        pred_for_forces = pred_denorm

    CL_pred, CD_pred, CMy_pred = compute_force_coefficients(pred_for_forces, data)

    # Consistent Cp-integrated reference forces vs solver table forces
    CL_cp_true, CD_cp_true, CMy_cp_true = compute_force_coefficients(target_denorm, data)
    CL_table = data.CL_true.to(pred_fp32.device, dtype=pred_fp32.dtype)
    CD_table = data.CD_true.to(pred_fp32.device, dtype=pred_fp32.dtype)
    CMy_table = data.CMy_true.to(pred_fp32.device, dtype=pred_fp32.dtype)

    if USE_CP_INTEGRATED_FORCE_TARGETS:
        CL_ref, CD_ref, CMy_ref = CL_cp_true, CD_cp_true, CMy_cp_true
    else:
        CL_ref, CD_ref, CMy_ref = CL_table, CD_table, CMy_table

    eps = 1e-3
    cl_error = torch.mean(torch.abs(CL_pred - CL_ref)).item()
    cd_error = torch.mean(torch.abs(CD_pred - CD_ref)).item()
    cmy_error = torch.mean(torch.abs(CMy_pred - CMy_ref)).item()

    cl_pct_error = (torch.mean(torch.abs(CL_pred - CL_ref) / (torch.abs(CL_ref) + eps)) * 100.0).item()
    cd_pct_error = (torch.mean(torch.abs(CD_pred - CD_ref) / (torch.abs(CD_ref) + eps)) * 100.0).item()
    cmy_pct_error = (torch.mean(torch.abs(CMy_pred - CMy_ref) / (torch.abs(CMy_ref) + eps)) * 100.0).item()
    aero_mape = cl_pct_error + cd_pct_error + cmy_pct_error

    cl_table_pct = (torch.mean(torch.abs(CL_pred - CL_table) / (torch.abs(CL_table) + eps)) * 100.0).item()
    cd_table_pct = (torch.mean(torch.abs(CD_pred - CD_table) / (torch.abs(CD_table) + eps)) * 100.0).item()

    return {
        "corr": corr_after_denorm,
        "corr_before_denorm": corr_before_denorm,
        "corr_after_denorm": corr_after_denorm,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "cl_error": cl_error,
        "cd_error": cd_error,
        "cmy_error": cmy_error,
        "cl_pct_error": cl_pct_error,
        "cd_pct_error": cd_pct_error,
        "cmy_pct_error": cmy_pct_error,
        "aero_mape": aero_mape,
        "cl_table_pct_error": cl_table_pct,
        "cd_table_pct_error": cd_table_pct,
    }


############################################################
# CHECKPOINT SAVE & RESUME FROM MIDDLE (E.G. EPOCH 30 -> 150)
############################################################

def resolve_resume_checkpoint(resume_arg):
    """Resolve checkpoint path from CLI or config ('auto'/'latest', explicit path, or epoch number)."""
    if not resume_arg:
        return None
    if isinstance(resume_arg, int) or (isinstance(resume_arg, str) and resume_arg.isdigit()):
        ep = int(resume_arg)
        candidate = os.path.join(PT_OUT_DIR, f"model_epoch_{ep}.pt")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Requested resume from epoch {ep}, but {candidate} does not exist.")

    resume_str = str(resume_arg).strip()
    if resume_str.lower() in ("auto", "latest"):
        latest_ckpt = os.path.join(PT_OUT_DIR, "latest_checkpoint.pt")
        if os.path.exists(latest_ckpt):
            return latest_ckpt
        epoch_files = glob.glob(os.path.join(PT_OUT_DIR, "model_epoch_*.pt"))
        if not epoch_files:
            print("[RESUME] 'auto' specified, but no existing checkpoints found. Starting from epoch 0.")
            return None

        def _extract_ep(p):
            m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        epoch_files.sort(key=_extract_ep)
        return epoch_files[-1]

    if os.path.exists(resume_str):
        return resume_str
    alt_path = os.path.join(PT_OUT_DIR, resume_str)
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"Resume checkpoint not found: {resume_str}")


def recover_best_metrics_from_csv(csv_path, up_to_epoch):
    """If resuming from a legacy weights-only checkpoint, recover best validation metrics from CSV."""
    best_val_loss = float("inf")
    best_field_rel_l2 = float("inf")
    best_aero_mape = float("inf")
    if not os.path.exists(csv_path):
        return best_val_loss, best_field_rel_l2, best_aero_mape
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row.get("epoch", 0) or 0)
                if ep <= up_to_epoch:
                    if row.get("val_loss"):
                        best_val_loss = min(best_val_loss, float(row["val_loss"]))
                    if row.get("rel_l2"):
                        best_field_rel_l2 = min(best_field_rel_l2, float(row["rel_l2"]))
                    if row.get("aero_mape"):
                        best_aero_mape = min(best_aero_mape, float(row["aero_mape"]))
    except Exception:
        pass
    return best_val_loss, best_field_rel_l2, best_aero_mape


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_loss,
    best_field_rel_l2,
    best_aero_mape,
    log_path=None,
    weights_only=False,
):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if weights_only:
        torch.save(raw_model.state_dict(), path)
        return

    payload = {
        "epoch": int(epoch),
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "best_field_rel_l2": float(best_field_rel_l2),
        "best_aero_mape": float(best_aero_mape),
        "log_path": log_path,
        "arch": {
            "node_in_dim": NODE_IN_DIM,
            "edge_dim": EDGE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "out_dim": OUT_DIM,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_gno_branches": NUM_GNO_BRANCHES,
            "num_gno_layers": NUM_GNO_LAYERS,
            "gno_fusion": GNO_FUSION,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "num_gnn_layers": NUM_GNN_LAYERS,
            "fourier_bands": FOURIER_BANDS,
        },
    }
    torch.save(payload, path)


def load_checkpoint_for_resume(
    ckpt_path,
    model,
    optimizer,
    scheduler,
    scaler,
    explicit_start_epoch,
    csv_path,
):
    """Load either a full training checkpoint or a legacy weights-only checkpoint (e.g. model_epoch_30.pt)
    and resume seamlessly from `start_epoch + 1` to `EPOCHS`.
    """
    print(f"\n[RESUME] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    is_full_ckpt = isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "model" in ckpt)

    if is_full_ckpt:
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        raw_model.load_state_dict(state_dict, strict=True)

        detected_epoch = int(ckpt.get("epoch", 0))
        start_epoch = int(explicit_start_epoch) if explicit_start_epoch is not None else detected_epoch

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[RESUME] Warning: optimizer state not loaded ({e}); using fresh optimizer.")

        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass

        if "scheduler_state_dict" in ckpt and (explicit_start_epoch is None or explicit_start_epoch == detected_epoch):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                for _ in range(start_epoch):
                    scheduler.step()
        else:
            for _ in range(start_epoch):
                scheduler.step()

        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_field_rel_l2 = float(ckpt.get("best_field_rel_l2", float("inf")))
        best_aero_mape = float(ckpt.get("best_aero_mape", float("inf")))
        print(
            f"[RESUME] Full checkpoint restored at epoch {start_epoch} -> "
            f"continuing from epoch {start_epoch + 1} to {EPOCHS} "
            f"(LR={optimizer.param_groups[0]['lr']:.6e})"
        )
        return start_epoch, best_val_loss, best_field_rel_l2, best_aero_mape

    # Legacy weights-only state_dict checkpoint (e.g. model_epoch_30.pt saved by old code)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    raw_model.load_state_dict(state_dict, strict=True)

    if explicit_start_epoch is not None:
        start_epoch = int(explicit_start_epoch)
    else:
        m = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
        if m:
            start_epoch = int(m.group(1))
        else:
            raise ValueError(
                f"Could not infer start epoch from legacy weights-only checkpoint '{ckpt_path}'. "
                f"Please set START_EPOCH in config.py (e.g. START_EPOCH = 30) or pass --start-epoch 30."
            )

    # Fast-forward learning rate scheduler to match `start_epoch`
    for _ in range(start_epoch):
        scheduler.step()

    best_val_loss, best_field_rel_l2, best_aero_mape = recover_best_metrics_from_csv(csv_path, start_epoch)
    print(
        f"[RESUME] Legacy weights-only checkpoint restored at epoch {start_epoch} -> "
        f"fast-forwarded scheduler to LR={optimizer.param_groups[0]['lr']:.6e}, "
        f"continuing from epoch {start_epoch + 1} to {EPOCHS}"
    )
    return start_epoch, best_val_loss, best_field_rel_l2, best_aero_mape


############################################################
# TRAIN
############################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train GI-GNO (Enc -> Parallel GNOs -> Dec -> GNN)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/model_epoch_30.pt or 'auto')",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Explicitly override starting epoch (e.g. 30 to train 31..EPOCHS)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total target EPOCHS (e.g. 150)",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    resume_target = args.resume if args.resume is not None else RESUME_CHECKPOINT
    explicit_start_epoch = args.start_epoch if args.start_epoch is not None else START_EPOCH
    total_epochs = int(args.epochs) if args.epochs is not None else int(EPOCHS)

    resume_path = resolve_resume_checkpoint(resume_target)
    log_path = setup_run_logger("training", reuse_active=(resume_path is not None))

    configure_vram_limit(MAX_VRAM_GB)

    writer = SummaryWriter(log_dir=TENSORBOARD)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    use_bf16 = (
        DEVICE == "cuda"
        and bool(USE_BF16)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_grad_scaler = (DEVICE == "cuda") and (not use_bf16)

    print("==== TRAINING STARTED ====")
    print(f"Device: {DEVICE} | AMP: {'bfloat16 (no GradScaler)' if use_bf16 else 'float16 (GradScaler)'} | Log file: {log_path}")
    print(
        f"Loss Mode: {'CP_ONLY_TRAINING (predict Cp + aero-weighted cell Cp -> integrate CL/CD later)' if CP_ONLY_TRAINING else 'COMPOSITE (Cp + direct global CL/CD loss)'}"
    )
    print(
        f"Performance Settings: MAX_VRAM_GB={MAX_VRAM_GB}, BATCH_SIZE={BATCH_SIZE}, "
        f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}, USE_GRAD_CHECKPOINT={USE_GRAD_CHECKPOINT}, "
        f"GNO_EDGE_CHUNK_SIZE={GNO_EDGE_CHUNK_SIZE}, NUM_WORKERS={NUM_WORKERS}, "
        f"PREFETCH_FACTOR={PREFETCH_FACTOR}, MAX_CACHED_CHUNKS={MAX_CACHED_CHUNKS}"
    )

    dataset = GraphDataset(DATA_DIR, max_cached_chunks=MAX_CACHED_CHUNKS)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training samples found in {DATA_DIR}. Run preprocessing first to create chunk files."
        )
    if len(dataset) < 2:
        raise RuntimeError("At least two samples are required for validation split.")
    log_training_sample_contract(dataset)

    norm_stats = torch.load(NORM_STATS_FILE, map_location="cpu", weights_only=False)
    cp_mean = norm_stats["cp_mean"].float().to(DEVICE)
    cp_std = norm_stats["cp_std"].float().to(DEVICE)

    folds = build_validation_folds(len(dataset))
    print(
        f"Dataset total cases: {len(dataset)} (LRU cache bounded to {MAX_CACHED_CHUNKS} chunks) | "
        f"Validation strategy: {'rotating' if ROTATE_VALIDATION_FOLDS else 'fixed'} "
        f"{len(folds)} folds | fold sizes: {[len(fold) for fold in folds]}"
    )

    ############################################################
    # MODEL
    ############################################################

    model = EncoderGNODecoder(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_gno_branches=NUM_GNO_BRANCHES,
        num_gno_layers=NUM_GNO_LAYERS,
        gno_fusion=GNO_FUSION,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_gnn_layers=NUM_GNN_LAYERS,
        edge_dim=EDGE_DIM,
        dropout=DROPOUT,
        fourier_bands=FOURIER_BANDS,
        use_checkpoint=USE_GRAD_CHECKPOINT,
        edge_chunk_size=GNO_EDGE_CHUNK_SIZE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model Architecture: Enc({NUM_ENCODER_LAYERS}) -> "
        f"{model.gno.num_branches}x Parallel GNO(layers={model.gno.branch_depths}, fusion={GNO_FUSION}) -> "
        f"Dec({NUM_DECODER_LAYERS}) -> GNN({NUM_GNN_LAYERS}) | "
        f"HIDDEN_DIM={HIDDEN_DIM} | Parameters: {total_params:,}"
    )

    if USE_TORCH_COMPILE and hasattr(torch, "compile") and os.name != "nt" and not USE_GRAD_CHECKPOINT:
        try:
            model = torch.compile(model, dynamic=True)
            print("[PERF] torch.compile(dynamic=True) enabled.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            fused=(DEVICE == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    cosine_t_max = max(1, total_epochs - WARMUP_EPOCHS)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max,
        eta_min=ETA_MIN,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    # Sharp beta=0.1 SmoothL1Loss for normalized Cp field prediction
    criterion = nn.SmoothL1Loss(beta=0.1)

    csv_path = os.path.join(PT_OUT_DIR, "training_metrics.csv")
    start_epoch = 0
    best_val_loss = float("inf")
    best_field_rel_l2 = float("inf")
    best_aero_mape = float("inf")

    if resume_path is not None:
        start_epoch, best_val_loss, best_field_rel_l2, best_aero_mape = load_checkpoint_for_resume(
            ckpt_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            explicit_start_epoch=explicit_start_epoch,
            csv_path=csv_path,
        )

    if start_epoch >= total_epochs:
        print(f"[RESUME] Checkpoint epoch ({start_epoch}) >= target EPOCHS ({total_epochs}). Nothing to train.")
        return

    csv_exists = os.path.exists(csv_path) and start_epoch > 0
    csv_file = open(csv_path, mode="a" if csv_exists else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "rmse",
            "rel_l2",
            "cl_pct_error",
            "cd_pct_error",
            "cmy_pct_error",
            "aero_mape",
            "time_s",
        ])
        csv_file.flush()

    # Build persistent loaders once outside epoch loop when using fixed validation split
    # (prevents DataLoader persistent worker & CUDA pinned memory leaks across 150 epochs!)
    fixed_train_loader = None
    fixed_train_sampler = None
    fixed_val_loader = None
    fixed_train_dataset = None
    fixed_val_dataset = None

    if not ROTATE_VALIDATION_FOLDS:
        fixed_train_dataset, fixed_val_dataset = build_fold_datasets(dataset, folds, 0)
        fixed_train_loader, fixed_train_sampler = make_loader(
            fixed_train_dataset, shuffle=True, persistent=True
        )
        fixed_val_loader, _ = make_loader(
            fixed_val_dataset, shuffle=False, persistent=True
        )

    ############################################################
    # TRAIN LOOP (START_EPOCH + 1 .. TOTAL_EPOCHS)
    ############################################################

    accum_steps = max(1, int(GRAD_ACCUM_STEPS))

    for epoch in range(start_epoch + 1, total_epochs + 1):

        start = time.time()
        if torch.cuda.is_available() and DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        if ROTATE_VALIDATION_FOLDS:
            active_fold = (epoch - 1) % len(folds)
            train_dataset, val_dataset = build_fold_datasets(dataset, folds, active_fold)
            train_loader, train_sampler = make_loader(train_dataset, shuffle=True, persistent=False)
        else:
            active_fold = 0
            train_dataset, val_dataset = fixed_train_dataset, fixed_val_dataset
            train_loader, train_sampler = fixed_train_loader, fixed_train_sampler

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_gpu = torch.zeros((), device=DEVICE, dtype=torch.float32)
        valid_batches = 0
        oom_count = 0

        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)

        for step_idx, data in enumerate(train_loader, 1):
            try:
                data = prepare_batch_on_device(data, DEVICE, edge_dtype=amp_dtype)

                with autocast("cuda", dtype=amp_dtype, enabled=(DEVICE == "cuda")):
                    pred = model(data)
                    loss = composite_loss(pred, data, criterion, cp_mean, cp_std)

                scaled_loss = loss / float(accum_steps)
                if use_grad_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Accumulate loss on GPU to avoid per-step CPU-GPU sync stalls
                train_loss_gpu = train_loss_gpu + loss.detach().float()
                valid_batches += 1
                del data, pred, loss, scaled_loss

                if step_idx % accum_steps == 0 or step_idx == num_batches:
                    if use_grad_scaler:
                        # GradScaler natively detects and skips non-finite gradients in CUDA
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(
                    f"[Epoch {epoch:03d} | Step {step_idx}] WARNING: CUDA OOM caught (#{oom_count})! "
                    f"Clearing cache and continuing... "
                    f"(If frequent, lower BATCH_SIZE or set USE_GRAD_CHECKPOINT=True in config.py)"
                )
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        train_loss = float(train_loss_gpu.item()) / max(1, valid_batches)

        ############################################################
        # VALIDATION
        ############################################################

        val_loss = None
        val_metrics = None

        if epoch % VALIDATE_EVERY == 0 or epoch == total_epochs:
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
                "cmy_error": 0.0,
                "cl_pct_error": 0.0,
                "cd_pct_error": 0.0,
                "cmy_pct_error": 0.0,
                "aero_mape": 0.0,
                "cl_table_pct_error": 0.0,
                "cd_table_pct_error": 0.0,
            }
            if ROTATE_VALIDATION_FOLDS:
                val_loader, _ = make_loader(val_dataset, shuffle=False, persistent=False)
            else:
                val_loader = fixed_val_loader

            print(
                f"Validation fold {active_fold + 1}/{len(folds)} | "
                f"train cases: {len(train_dataset)} | val cases: {len(val_dataset)}"
            )

            val_batches = 0
            with torch.no_grad():
                for data in val_loader:
                    data = prepare_batch_on_device(data, DEVICE, edge_dtype=amp_dtype)

                    with autocast("cuda", dtype=amp_dtype, enabled=(DEVICE == "cuda")):
                        pred = model(data)
                        loss = composite_loss(pred, data, criterion, cp_mean, cp_std)

                    val_loss += float(loss.detach().item())
                    batch_metrics = compute_validation_metrics(pred, data, cp_mean, cp_std)
                    for key in val_metrics:
                        val_metrics[key] += batch_metrics[key]
                    val_batches += 1
                    del data, pred, loss

            val_loss /= max(1, val_batches)
            for key in val_metrics:
                val_metrics[key] /= max(1, val_batches)

            if ROTATE_VALIDATION_FOLDS:
                del val_loader

        if ROTATE_VALIDATION_FOLDS:
            del train_loader

        scheduler.step()

        ############################################################
        # LOGGING
        ############################################################

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
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
            writer.add_scalar("val/cmy_error", val_metrics["cmy_error"], epoch)
            writer.add_scalar("val/cl_pct_error", val_metrics["cl_pct_error"], epoch)
            writer.add_scalar("val/cd_pct_error", val_metrics["cd_pct_error"], epoch)
            writer.add_scalar("val/cmy_pct_error", val_metrics["cmy_pct_error"], epoch)
            writer.add_scalar("val/aero_mape", val_metrics["aero_mape"], epoch)

        elapsed = time.time() - start
        vram_info = get_vram_stats_str()

        csv_writer.writerow([
            epoch,
            f"{current_lr:.6e}",
            f"{train_loss:.6e}",
            f"{val_loss:.6e}" if val_loss is not None else "",
            f"{val_metrics['rmse']:.6e}" if val_metrics is not None else "",
            f"{val_metrics['rel_l2']:.6e}" if val_metrics is not None else "",
            f"{val_metrics['cl_pct_error']:.4f}" if val_metrics is not None else "",
            f"{val_metrics['cd_pct_error']:.4f}" if val_metrics is not None else "",
            f"{val_metrics['cmy_pct_error']:.4f}" if val_metrics is not None else "",
            f"{val_metrics['aero_mape']:.4f}" if val_metrics is not None else "",
            f"{elapsed:.2f}",
        ])
        csv_file.flush()

        if val_loss is not None:
            print(
                f"Epoch {epoch:03d}/{total_epochs:03d} | "
                f"Train Loss: {train_loss:.6e} | "
                f"Val Loss: {val_loss:.6e} | "
                f"Corr: {val_metrics['corr']:.6f} | "
                f"RMSE: {val_metrics['rmse']:.6e} | "
                f"Rel L2: {val_metrics['rel_l2']:.6e} | "
                f"CL Err: {val_metrics['cl_error']:.4e} ({val_metrics['cl_pct_error']:.2f}%) | "
                f"CD Err: {val_metrics['cd_error']:.4e} ({val_metrics['cd_pct_error']:.2f}%) | "
                f"CMy %Err: {val_metrics['cmy_pct_error']:.2f}% | "
                f"{vram_info} | Time: {elapsed:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{total_epochs:03d} | "
                f"Train Loss: {train_loss:.6e} | "
                f"LR: {current_lr:.6e} | "
                f"{vram_info} | Time: {elapsed:.2f}s"
            )

        ############################################################
        # CHECKPOINTING (FULL RESUMABLE + BEST MODELS)
        ############################################################

        if val_metrics is not None and val_metrics["rel_l2"] < best_field_rel_l2:
            best_field_rel_l2 = val_metrics["rel_l2"]
            save_training_checkpoint(
                os.path.join(PT_OUT_DIR, "bestmodel/best_field.pt"),
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, best_field_rel_l2, best_aero_mape, log_path=log_path,
            )
            print(f"[BEST] Saved BEST field model (Rel L2={best_field_rel_l2:.6e})")

        if val_metrics is not None and val_metrics["aero_mape"] < best_aero_mape:
            best_aero_mape = val_metrics["aero_mape"]
            save_training_checkpoint(
                os.path.join(PT_OUT_DIR, "bestmodel/best_aero.pt"),
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, best_field_rel_l2, best_aero_mape, log_path=log_path,
            )
            print(f"[BEST] Saved BEST aero model (MAPE sum={best_aero_mape:.2f}%)")

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_training_checkpoint(
                os.path.join(PT_OUT_DIR, "bestmodel/best_model.pt"),
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, best_field_rel_l2, best_aero_mape, log_path=log_path,
            )

        # Always update latest_checkpoint.pt so any interrupted training can resume from the exact epoch
        save_training_checkpoint(
            os.path.join(PT_OUT_DIR, "latest_checkpoint.pt"),
            epoch, model, optimizer, scheduler, scaler,
            best_val_loss, best_field_rel_l2, best_aero_mape, log_path=log_path,
        )

        if epoch % SAVE_EVERY == 0 or epoch == total_epochs:
            save_training_checkpoint(
                os.path.join(PT_OUT_DIR, f"model_epoch_{epoch}.pt"),
                epoch, model, optimizer, scheduler, scaler,
                best_val_loss, best_field_rel_l2, best_aero_mape, log_path=log_path,
            )

    csv_file.close()
    writer.close()
    print(f"==== TRAINING COMPLETED (Log saved to {log_path}) ====")


if __name__ == "__main__":
    main()
