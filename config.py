import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################################
# PREPROCESSING
############################################################

RAW_VTK_DIR = "dataset/trainvtk"
GEOM_PARAM_FILE = "dataset/geom_params.ini"
CASE_FILE = "dataset/case_data.dat"

OUT_DIR = "ptfiles"
NORM_STATS_FILE = "normalization_stats.pt"

NO_CASES = None   # None → full dataset


############################################################
# TRAINING
############################################################

DATA_DIR = "ptfiles"
PT_OUT_DIR = "checkpoints"

NODE_IN_DIM = 27
EDGE_DIM = 8
HIDDEN_DIM = 32
OUT_DIM = 1
NUM_GNN_LAYERS = 5
GRID_SIZE = (1024, 512)
KNN_K = 8
FOURIER_BANDS = 6
ATTENTION_HEADS = 4
ATTENTION_QUERY_CHUNK_SIZE = 4096
ATTENTION_MAX_NODES = 1024
ATTENTION_LOCAL_RADIUS = 1


VALIDATE_EVERY = 4
ROTATE_VALIDATION_FOLDS = True
VALIDATION_FOLDS = 5
VAL_SPLIT = 0.2


LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
BATCH_SIZE = 1
CHUNK_SIZE = 10
NUM_WORKERS = min(8, os.cpu_count())

DROPOUT = 0.2

TENSORBOARD = f"logs/tensorboard/h{HIDDEN_DIM}_e{EPOCHS}"

### Loss 
# Force loss weights
LAMBDA_FIELD=1.0
LAMBDA_GRADIENT = 0.3
LAMBDA_CL = 0.1
LAMBDA_CD = 0.1
LAMBDA_SMOOTHNESS = 0.05


############################################################
# INFERENCE
############################################################

TEST_VTK_DIR = "dataset/testvtk"
OUT_VTK_DIR = "predicted"
CHECKPOINT = "checkpoints/bestmodel/best_model.pt"
INFRLOG_FILE = "logs/inference_log.txt"
