# config.py

import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##### Preprocessing
RAW_VTK_DIR = r"dataset/trainvtk"
GEOM_PARAM_FILE = r"dataset/geom_params.ini"
CASE_FILE = r"dataset/case_data.dat"
OUT_DIR = r"ptfiles"
NO_CASES = 9
NORM_STATS_FILE = os.path.join(OUT_DIR, "normalization_stats.pt")

###### Training
DATA_DIR = "ptfiles"
PT_OUT_DIR = "checkpoints"
NODE_IN_DIM = 18
HIDDEN_DIM = 32
OUT_DIM = 1
NUM_GNN_LAYERS = 5
GRID_SIZE = (1024, 512)
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 500
BATCH_SIZE = 1
USE_SMOOTHNESS_LOSS = False   # enable later if needed
SMOOTHNESS_WEIGHT = 0.05
TRAINING_LOG_FILE = f"logs/train_log_e{EPOCHS}.txt"
NUM_WORKERS = min(8, os.cpu_count())

TENSORBOARD="logs/tensorboard/log1"
###### Model 
##SpectralConv2d
modes_ratio=0.025,
min_modes=16
max_modes=32

####### Inference 
TEST_VTK_DIR = "dataset/testvtk"
OUT_VTK_DIR = "predicted"
CHECKPOINT = "checkpoints/bestmodel/best_model.pt"

INFRLOG_FILE = "logs/inference_log.txt"

