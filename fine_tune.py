# we should have the fine tuned model weights here for clip

# imports
import json
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# paths
base_dir = Path(__file__).parent
image_dir = base_dir / "dataset"
corpus_path = base_dir / "dataset_text.json"
ckpt_dir = base_dir / "checkpoints"

ckpt_dir.mkdir(exist_ok=True)





