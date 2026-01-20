# dataset.py

import torch
from torch.utils.data import Dataset
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os

class GraphDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw = torch.load(self.files[idx],weights_only=True)
        # print("Raw :",raw)
        data = Data(
            x=raw["x"],                       # node features
            edge_index=raw["edge_index"],     # mesh connectivity
            pos=raw["coords_2d"],             # normalized 2D coords
            y_cp=raw["y_cp"],                 # Cp target
            y_cf=raw["y_cf"],                 # Cf target
        )

        # store metadata (NOT moved to GPU)
        data.meta = raw["meta"]

        return data
