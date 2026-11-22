import os
import sys
import logging
from datetime import datetime
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################################
# PREPROCESSING
############################################################

RAW_VTK_DIR = "dataset/trainvtk"
GEOM_PARAM_FILE = "dataset/geom_params.ini"
CASE_FILE = "dataset/case_data.dat"

OUT_DIR = "ptfiles"
NORM_STATS_FILE = "normalization_stats.pt"
MANIFEST_FILE = os.path.join(OUT_DIR, "manifest.pt")

NO_CASES = None          # None -> full dataset (scales to 10,000+ cases)
KNN_K = 16
PREPROCESS_WORKERS = min(8, os.cpu_count() or 1)
SAVE_EDGE_ATTR = False   # False -> compute edge_attr on-the-fly (<0.5ms) to save ~70% disk/RAM for 10,000 cases
RESUME_PREPROCESSING = True  # Resume interrupted preprocessing runs from existing temp chunks


############################################################
# MODEL ARCHITECTURE (ENCODER -> PARALLEL GNOs -> DECODER -> GNN)
############################################################

NODE_IN_DIM = 27
EDGE_DIM = 8
HIDDEN_DIM = 64
OUT_DIM = 1

# Modular stage depths & parallel GNO configuration:
NUM_ENCODER_LAYERS = 1               # Number of layers in GraphEncoder (enc)
NUM_GNO_LAYERS = [2, 3, 4]           # Layer depths per GNO branch (e.g. [2, 3, 4] or [4, 5, 6] or int k)
NUM_GNO_BRANCHES = (
    len(NUM_GNO_LAYERS)
    if isinstance(NUM_GNO_LAYERS, (list, tuple))
    else 3
)                                    # Automatically synced with NUM_GNO_LAYERS when a list is given
NUM_PARALLEL_GNO = NUM_GNO_BRANCHES  # Alias for convenience
GNO_FUSION = "concat"                # How parallel GNO branches merge: "concat", "sum", or "mean"
NUM_DECODER_LAYERS = 1               # Number of layers in GraphDecoder (dec)
NUM_GNN_LAYERS = 1                   # Number of layers in post-decoder GNN head (gnn)

FOURIER_BANDS = 6
DROPOUT = 0.0

# Legacy grid / attention parameters (retained for backward compatibility)
GRID_SIZE = (1024, 1024)
ATTENTION_HEADS = 4
ATTENTION_QUERY_CHUNK_SIZE = 4096
ATTENTION_MAX_NODES = 1024
ATTENTION_LOCAL_RADIUS = 1


############################################################
# MEMORY & SCALING OPTIMIZATION (FULL GPU POTENTIAL)
############################################################

MAX_VRAM_GB = None                   # None -> use full available GPU VRAM (no artificial ceiling)
USE_GRAD_CHECKPOINT = False          # False -> ~35-45% faster epochs (set True only if hitting CUDA OOM)
GNO_EDGE_CHUNK_SIZE = 0              # 0 -> full vectorized edge evaluation (fastest); set e.g. 500000 only if OOM
MAX_CACHED_CHUNKS = 8                # LRU chunk cache limit per DataLoader worker
GRAD_ACCUM_STEPS = 2                 # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS (2 * 2 = 4)
USE_BF16 = True                      # Auto-use native bfloat16 on Ampere/Ada/Hopper GPUs (faster, no GradScaler overhead)
USE_TORCH_COMPILE = False            # Optional: set True on Linux for extra speedup via torch.compile(dynamic=True)


############################################################
# TRAINING & CHECKPOINT RESUME
############################################################

DATA_DIR = "ptfiles"
PT_OUT_DIR = "checkpoints"

# Resume training from middle (e.g., epoch 30 to 150):
# Set RESUME_CHECKPOINT to a path (e.g. "checkpoints/model_epoch_30.pt" or "checkpoints/latest_checkpoint.pt")
# or "auto" to automatically pick the latest saved checkpoint, or None to start from epoch 0.
RESUME_CHECKPOINT = None
START_EPOCH = None                   # None -> auto-detect from checkpoint/filename; or set int (e.g. 30)
SAVE_EVERY = 10                      # Save periodic model_epoch_{epoch}.pt every N epochs

VALIDATE_EVERY = 4
ROTATE_VALIDATION_FOLDS = False
VALIDATION_FOLDS = 5
VAL_SPLIT = 0.2

LR = 1e-3
ETA_MIN = 1e-5
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4
EPOCHS = 150
BATCH_SIZE = 2                       # Increase to 3 or 4 if peak VRAM < 16 GB; set to 1 if CUDA OOM
CHUNK_SIZE = 10
NUM_WORKERS = min(6, os.cpu_count() or 1)
PREFETCH_FACTOR = 4

TENSORBOARD = f"logs/tensorboard/h{HIDDEN_DIM}_e{EPOCHS}"

### Aerodynamic Reference Constants
S_REF = 1.0
C_REF = 1.0
X_REF = 0.25
Z_REF = 0.0

### Loss Strategy: Predict Cp First & Integrate Later
# When CP_ONLY_TRAINING = True (recommended):
#   The model trains strictly on node/cell Cp prediction losses (field + spatial gradient +
#   aerodynamic normal/area-weighted cell Cp loss), avoiding global scalar CL/CD rank-1
#   gradient oscillations. CL, CD, and CMy are then computed by surface integration.
# When CP_ONLY_TRAINING = False:
#   Global integrated force losses (LAMBDA_CL, LAMBDA_CD, LAMBDA_CMY) are also added.
CP_ONLY_TRAINING = True
USE_CP_INTEGRATED_FORCE_TARGETS = True  # Compare integrated forces against GT Cp surface integral as well as table

LAMBDA_FIELD = 1.0
LAMBDA_GRADIENT = 0.5
LAMBDA_AERO_WEIGHTED_CP = 0.5        # Local drag/lift normal & area-weighted Cp loss at leading/trailing edges
LAMBDA_CL = 0.0                      # Only active when CP_ONLY_TRAINING = False (or when > 0 and enabled)
LAMBDA_CD = 0.0                      # Only active when CP_ONLY_TRAINING = False (or when > 0 and enabled)
LAMBDA_CMY = 0.0                     # Only active when CP_ONLY_TRAINING = False (or when > 0 and enabled)
LAMBDA_SMOOTHNESS = 0.0


############################################################
# INFERENCE & LOGGING
############################################################

TEST_VTK_DIR = "dataset/testvtk"
OUT_VTK_DIR = "predicted"
CHECKPOINT = "checkpoints/bestmodel/best_model.pt"

LOG_DIR = "logs"
ACTIVE_LOG_TRACKER = os.path.join(LOG_DIR, ".active_log_file.txt")
INFRLOG_FILE = os.path.join(LOG_DIR, "inference_log.txt")


############################################################
# TIMESTAMPED LOGGING UTILITY (CONFIG + TRAINING + INFERENCE)
############################################################

_CONFIG_EXPORT_KEYS = [
    "DEVICE",
    "RAW_VTK_DIR", "GEOM_PARAM_FILE", "CASE_FILE", "OUT_DIR", "NORM_STATS_FILE", "MANIFEST_FILE",
    "NO_CASES", "KNN_K", "PREPROCESS_WORKERS", "SAVE_EDGE_ATTR", "RESUME_PREPROCESSING",
    "NODE_IN_DIM", "EDGE_DIM", "HIDDEN_DIM", "OUT_DIM",
    "NUM_ENCODER_LAYERS", "NUM_GNO_BRANCHES", "NUM_PARALLEL_GNO", "NUM_GNO_LAYERS",
    "GNO_FUSION", "NUM_DECODER_LAYERS", "NUM_GNN_LAYERS", "FOURIER_BANDS", "DROPOUT",
    "MAX_VRAM_GB", "USE_GRAD_CHECKPOINT", "GNO_EDGE_CHUNK_SIZE", "MAX_CACHED_CHUNKS", "GRAD_ACCUM_STEPS",
    "USE_BF16", "USE_TORCH_COMPILE",
    "DATA_DIR", "PT_OUT_DIR", "RESUME_CHECKPOINT", "START_EPOCH", "SAVE_EVERY",
    "VALIDATE_EVERY", "ROTATE_VALIDATION_FOLDS", "VALIDATION_FOLDS", "VAL_SPLIT",
    "LR", "ETA_MIN", "WARMUP_EPOCHS", "WEIGHT_DECAY", "EPOCHS", "BATCH_SIZE", "CHUNK_SIZE", "NUM_WORKERS",
    "PREFETCH_FACTOR",
    "TENSORBOARD", "S_REF", "C_REF", "X_REF", "Z_REF",
    "CP_ONLY_TRAINING", "USE_CP_INTEGRATED_FORCE_TARGETS",
    "LAMBDA_FIELD", "LAMBDA_GRADIENT", "LAMBDA_AERO_WEIGHTED_CP",
    "LAMBDA_CL", "LAMBDA_CD", "LAMBDA_CMY", "LAMBDA_SMOOTHNESS",
    "TEST_VTK_DIR", "OUT_VTK_DIR", "CHECKPOINT", "LOG_DIR",
]


def get_config_summary() -> str:
    """Return a formatted multi-line string containing all config.py parameters."""
    g = globals()
    lines = [
        "=" * 80,
        f"COMPLETE CONFIGURATION PARAMETERS ({datetime.now().isoformat(timespec='seconds')})",
        "=" * 80,
    ]
    for key in _CONFIG_EXPORT_KEYS:
        if key in g:
            lines.append(f"  {key:<34} = {g[key]!r}")
    for key in sorted(g.keys()):
        if key.isupper() and not key.startswith("_") and key not in _CONFIG_EXPORT_KEYS:
            lines.append(f"  {key:<34} = {g[key]!r}")
    lines.append("=" * 80)
    return "\n".join(lines)


class _TeeStream:
    """Duplicates stdout/stderr writes to both the console and the timestamped .log file."""

    def __init__(self, console_stream, log_file_handle):
        self.console_stream = console_stream
        self.log_file_handle = log_file_handle

    def write(self, data):
        if self.console_stream is not None:
            self.console_stream.write(data)
            self.console_stream.flush()
        if self.log_file_handle is not None and not self.log_file_handle.closed:
            self.log_file_handle.write(data)
            self.log_file_handle.flush()

    def flush(self):
        if self.console_stream is not None:
            self.console_stream.flush()
        if self.log_file_handle is not None and not self.log_file_handle.closed:
            self.log_file_handle.flush()

    def isatty(self):
        return getattr(self.console_stream, "isatty", lambda: False)()


_ACTIVE_LOG_HANDLE = None
_ACTIVE_LOG_PATH = None


def setup_run_logger(stage_name: str, reuse_active: bool = False, log_path: str | None = None) -> str:
    """Initialize timestamped .log file capturing complete config, print(), and logging output.

    - File name is `<timestamp>.log` inside `LOG_DIR` (e.g. `logs/20261119_204500.log`).
    - When `reuse_active=True` (used by inference or resumed training), appends to the active
      timestamped `.log` file if one exists so training and inference logs reside in the same file.
    """
    global _ACTIVE_LOG_HANDLE, _ACTIVE_LOG_PATH

    os.makedirs(LOG_DIR, exist_ok=True)

    if _ACTIVE_LOG_PATH is not None and _ACTIVE_LOG_HANDLE is not None and not _ACTIVE_LOG_HANDLE.closed:
        header = f"\n\n>>> STAGE: {stage_name.upper()} ({datetime.now().isoformat(timespec='seconds')}) <<<\n"
        _ACTIVE_LOG_HANDLE.write(header + get_config_summary() + "\n\n")
        _ACTIVE_LOG_HANDLE.flush()
        return _ACTIVE_LOG_PATH

    resolved_path = log_path
    if resolved_path is None and reuse_active and os.path.exists(ACTIVE_LOG_TRACKER):
        try:
            with open(ACTIVE_LOG_TRACKER, "r", encoding="utf-8") as f:
                candidate = f.read().strip()
            if candidate and os.path.exists(candidate):
                resolved_path = candidate
        except Exception:
            resolved_path = None

    if resolved_path is None:
        timestamp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_path = os.path.join(LOG_DIR, f"{timestamp_name}.log")

    log_handle = open(resolved_path, mode="a", encoding="utf-8", buffering=1)
    _ACTIVE_LOG_HANDLE = log_handle
    _ACTIVE_LOG_PATH = resolved_path

    try:
        with open(ACTIVE_LOG_TRACKER, "w", encoding="utf-8") as f:
            f.write(resolved_path)
    except Exception:
        pass

    if not isinstance(sys.stdout, _TeeStream):
        sys.stdout = _TeeStream(sys.__stdout__, log_handle)
    else:
        sys.stdout.log_file_handle = log_handle

    if not isinstance(sys.stderr, _TeeStream):
        sys.stderr = _TeeStream(sys.__stderr__, log_handle)
    else:
        sys.stderr.log_file_handle = log_handle

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(resolved_path)
        for h in root_logger.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(fh)

    print(f"\n{'#' * 80}")
    print(f"# LOG FILE: {resolved_path} | STAGE: {stage_name.upper()}")
    print(f"{'#' * 80}")
    print(get_config_summary())

    return resolved_path
