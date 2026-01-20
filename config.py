# config.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
NODE_IN_DIM = 17        # depends on your preprocessing
EDGE_IN_DIM = 4        # optional
HIDDEN_DIM = 32
OUT_DIM = 1            # Cp / pressure scalar

NUM_LAYERS = 2

# Training
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 5#200
BATCH_SIZE = 1         # usually 1 for large meshes

SAVE_EVERY = 1
CHECKPOINT_DIR = "checkpoints/"
