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

NODE_IN_DIM = 18
HIDDEN_DIM = 32
OUT_DIM = 1
NUM_GNN_LAYERS = 5
GRID_SIZE = (1024, 512)

LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 300
BATCH_SIZE = 1
NUM_WORKERS = min(8, os.cpu_count())

USE_SMOOTHNESS_LOSS = False
SMOOTHNESS_WEIGHT = 0.05

DROPOUT = 0.2

TENSORBOARD = f"logs/tensorboard/h{HIDDEN_DIM}_e{EPOCHS}"


############################################################
# INFERENCE
############################################################

TEST_VTK_DIR = "dataset/testvtk"
OUT_VTK_DIR = "predicted"
CHECKPOINT = "checkpoints/bestmodel/best_model.pt"
INFRLOG_FILE = "logs/inference_log.txt"