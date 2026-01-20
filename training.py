# train_ddp.py - Parallel GPU Training with PyTorch DDP (Multi-GPU)
import os
import time
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from dataset import GraphDataset
from model import GAGNO
from config import *

import subprocess

def usage():
  out = subprocess.check_output([
      "nvidia-smi",
      "--query-gpu=utilization.gpu,utilization.memory",
      "--format=csv,noheader,nounits"
  ]).decode().strip()

  gpu_util, mem_util = map(int, out.split(","))
  print("GPU Util %:", gpu_util)
  print("Mem Util %:", mem_util)



# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Utils
# ----------------------------
def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_distributed():
    """
    Initializes torch.distributed ONLY if launched via torchrun.
    If not launched via torchrun, returns None (single-process mode).
    """
    if "RANK" not in os.environ:
        return None  # not a distributed run

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank



def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scaler, epoch, loss, best_loss):
    # model may be wrapped by DDP -> use .module
    raw_model = model.module if hasattr(model, "module") else model
    path = f"{CHECKPOINT_DIR}/model_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": loss,
            "best_loss": best_loss,
        },
        path,
    )
    logger.info(f"✓ Saved checkpoint: {path}")


# ----------------------------
# Train
# ----------------------------
def train_ddp():
    # CUDA perf flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- init DDP
    local_rank = setup_distributed()

    if local_rank is None:
        # normal single GPU training (your original flow)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{local_rank}")

    # ---- dataset
    dataset = GraphDataset(r"data/train")

    # IMPORTANT: DistributedSampler splits dataset per GPU (no duplication)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=False,
    )

    # DataLoader: each GPU gets different batches
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,          # <-- key change for DDP
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ---- model
    model = GAGNO(
        NODE_IN_DIM,
        HIDDEN_DIM,
        OUT_DIM,
        NUM_LAYERS,
    ).to(device)

    # torch.compile is optional; in DDP sometimes works, sometimes not
    # (keeping it safe & optional)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        if is_main_process():
            logger.info("Model compiled with torch.compile")
    except Exception as e:
        if is_main_process():
            logger.warning(f"torch.compile not available / failed: {e}")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ---- optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01,
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_loss = float("inf")

    if is_main_process():
        logger.info("Starting DDP training...")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # IMPORTANT: set epoch for sampler so shuffling is different each epoch
        sampler.set_epoch(epoch)
        torch.autograd.set_detect_anomaly(True)
        model.train()
        usage()
        total_loss = 0.0

        # show progress bar only on rank0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", disable=not is_main_process())

        for batch_idx, data in enumerate(pbar):
            data = data.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # with autocast():
            print("*"*30)
            print("target y_cp :",data.y_cp)
            print("*"*30)
            pred_cp = model(data)
            loss = criterion(pred_cp, data.y_cp)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()

            # gradient clipping
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step(optimizer)
            # scaler.update()

            total_loss += loss.item()

            if is_main_process():
                pbar.set_postfix({"loss": float(loss.item())})

        scheduler.step()

        # ---- DDP: average loss across all ranks
        loss_tensor = torch.tensor(total_loss / len(loader), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_all = (loss_tensor / dist.get_world_size()).item()

        epoch_time = time.time() - epoch_start

        if is_main_process():
            logger.info(
                f"Epoch [{epoch}/{EPOCHS}] | "
                f"Loss: {avg_loss_all:.6f} | "
                f"Time: {epoch_time:.2f}s | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Save best model only from rank0
            if avg_loss_all < best_loss:
                best_loss = avg_loss_all
                raw_model = model.module
                torch.save(raw_model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")
                logger.info(f"✓ New best model saved! Loss: {best_loss:.6f}")

            if epoch % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, epoch, avg_loss_all, best_loss)

    # Save final model from rank0
    if is_main_process():
        raw_model = model.module
        torch.save(raw_model.state_dict(), f"{CHECKPOINT_DIR}/final_model.pt")
        logger.info("✅ Training complete, final model saved")

    cleanup_distributed()


if __name__ == "__main__":
    start = time.time()
    train_ddp()
    end = time.time()

    # This is printed per-process; only rank0 should show cleanly
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info(f"Total training time: {(end - start) / 60:.2f} minutes")
