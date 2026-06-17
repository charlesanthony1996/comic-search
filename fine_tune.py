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

# hyperparameters
batch_size = 16
epochs = 10
lr = 1e-5
min_chars = 20


# build a caption for a comic page
def build_caption(filename: str, ocr_text: str) -> str:

    if len(ocr_text.strip()) >= min_chars:

        return ocr_text.strip()[:300]
    
    # fallback to a filename 
    fname = filename.lower()

    if "punisher" in fname:
        char = "the punisher frank castle"

    elif "daredevil" in fname:
        
        char = "daredevil matt murdock"

    elif "spiderman" in fname:

        char = "spiderman peter parker"

    elif "wolverine" in fname:

        char = "wolverine logan"

    elif "venom" in fname:

        char = "venom"

    else:
        char = "marvel superhero"

    return "comic book page action scene"


# pairs each comic page with its ocr caption
# this has 3 functions
def comicdataset(Dataset):

    def __init__(self, preprocess):
        self.preprocess = preprocess
        self.pairs = []

        # load ocr corpus
        if not corpus_path.exists():
            raise FileNotFoundError("json file not found. generate it from main.py first")
        
    
        with open(corpus_path) as f:
            corpus = json.load(f)

        for filename, ocr_text in corpus.items():
            img_path = image_dir / filename
            if not img_path.exists():
                continue

            caption = build_caption(filename, ocr_text)
            self.pairs.append((img_path, caption))


        print(f"dataset: {len(self.pairs)} image caption pairs")


    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):

        img_path, caption = self.pairs[idx]

        # preprocess the image here
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        return img_tensor, caption

def contrastive_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:

    # logit_scale is clips learnable temperature parameter
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


# same golden set imported from main.py
golden_set = [
    # spiderman
    {"query": "Spider-Man in red and blue suit swinging","expected": "spiderman"},
    {"query": "close up face of masked superhero","expected": "spiderman"},

    # venom
    # {"query": "black symbiote monster with huge teeth and white eyes", "expected": "venom"},
    # {"query": "dark villain with cape and supernatural powers","expected": "venom"},

    # wolverine
    # {"query": "hero with adamantium claws outstretched","expected": "wolverine"},
    # {"query": "snowy winter forest landscape with superhero","expected": "wolverine"},
    # {"query": "fire explosion orange dramatic portal scene","expected": "wolverine"},

    # mixed appearances? need a listof comics that only share these charachters
    # or with new ones we have never trained on before?
    # how can this be a golden set query? something here

    # daredevil
    {"query": "daredevil fighting with wilson fisk", "expected": "daredevil" },
    {"query": "daredevil in red suit fighting at night", "expected": "daredevil"},
    # {"query": "", "expected": ""}

    # punisher
    {"query": "punisher fighting with bad guys", "expected": "punisher"},
    {"query": "punisher skull symbol shooting guns", "expected": "punisher"}

]




