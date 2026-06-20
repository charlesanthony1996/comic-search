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

    return f"{char} comic book page action scene"


# pairs each comic page with its ocr caption
# this has 3 functions
class comicdataset(Dataset):

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


# evaluating_model function
# running the golden set here and returning the metrics
# this also tracks improvement after every epoch
def evaluate_model(model, tokenizer, preprocess, k = 5) -> dict:

    model.eval()

    # encode all the images in the dataset folder with the current model
    paths = sorted(image_dir.glob("*.jpg"))

    if not paths:
        return {"precision": 0, "mrr": 0, "map": 0, "ndcg": 0}
    
    vecs, filenames = [], []
    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            vec = model.encode_image(tensor)

            vec = vec / vec.norm(dim = -1, keepdim = True)
            vecs.append(vec.squeeze().numpy())

            filenames.append(path.name)


    vecs = np.stack(vecs)

    # evaluate each query
    all_p = []
    all_rr = []
    all_ap = []
    all_ndcg = []

    with torch.no_grad():
        for item in golden_set:
            tokens = tokenizer([item["query"]])
            qvec = model.encode_text(tokens)
            qvec = qvec / qvec.norm(dim = -1, keepdim=True)
            qvec = qvec.squeeze().numpy()

            scores = vecs @ qvec
            top_idx = np.argsort(scores)[::-1][:k*2]
            retrieved = [filenames[i] for i in top_idx]
            expected = item["expected"]

            # precision@k
            p = sum(1 for f in retrieved[:k] if expected in f) / k
            all_p.append(p)

            # reciprocal rank
            rr = 0.0
            for rank, f in enumerate(retrieved, 1):
                if expected in f:
                    rr = 1.0 / rank
                    break

            all_rr.append(rr)

            # average precision
            hits = 0.0
            sum_p = 0.0

            for rank, f in enumerate(retrieved, 1):
                if expected in f:
                    hits += 1
                    sum_p += hits / rank
            
            all_ap.append(sum_p / hits if hits > 0 else 0.0)

            # ndcg@k
            dcg = sum(1 / np.log2(r + 1) for r, f in enumerate(retrieved[:k], 1) if expected in f)

            nr = min(sum(1 for f in retrieved if expected in f), k)

            idcg = sum(1/ np.log2(r+1) for r in range(1, nr+ 1)) if nr > 0 else 1.0

            all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision": float(np.mean(all_p)),
        "mrr": float(np.mean(all_rr)),
        "map": float(np.mean(all_ap)),
        "ndcg": float(np.mean(all_ndcg))
    }

# fine tuning function
# training here
def fine_tune():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)

    # dataset and dataloader
    dataset = comicdataset(preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    params_to_train = (
        list(model.visual.transformer.resblocks[-2:].parameters()) +
        list(model.visual.ln_post.parameters()) +
        list(model.visual.proj.parameters() if hasattr(model.visual, "proj") else []) +
        list(model.transformer.resblocks[-2:].parameters()) +
        list(model.ln_final.parameters()) +
        [model.text_projection] if hasattr(model, "text_projection") else []
    )

    # fallback
    try:
        optimizer = AdamW(params_to_train, lr=lr, weight_decay=0.01)
        print("fine tuning last 2 blocks only")
    except Exception:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print("fine tuning all parameters")


    # cosine lr decay
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # record metrics per epoch
    history = []
    best_mrr = 0.0

    # baseline before training
    baseline = evaluate_model(model, tokenizer, preprocess)

    history.append({"epoch": 0, **baseline})

    # training epochs
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_imgs, batch_captions in dataloader:
            # move images to device
            batch_imgs = batch_imgs.to(device)

            # tokenize captions
            tokens = tokenizer(list(batch_captions))
            tokens = tokens.to(device)

            # forward pass
            image_features = model.encode_image(batch_imgs)
            text_features = model.encode_text(tokens)

            # l2 normalize
            image_features = image_features / image_features.norm(dim = -1, keepdim=True)
            text_features = text_features / text_features.norm(dim = -1, keepdim=True)

            # contrastive loss
            loss = contrastive_loss(
                image_features,
                text_features,
                model.logit_scale.exp()
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # evaluate after each epoch
        metrics = evaluate_model(model, tokenizer, preprocess)
        history.append({"epoch": epoch, **metrics})

        print(f"epoch {epoch:02d}/epochs loss={avg_loss:.4f}")
        print()

        # save best checkpoint
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "loss": avg_loss
            }, ckpt_dir / "clip_finetuned_best.pt")
            print(f"new best model saved (mrr={best_mrr:.4f})")

        






    # save metrics
    with open(ckpt_dir / "metrics_per_epoch.json", "w") as f:
        json.dump(history, f, indent = 2)

    # plot metrics per epoch
    plot_training(history)
    print(f"best mrr: {best_mrr:.4f}")
    print(f"checkpoint: {ckpt_dir / 'clip_finetuned_best.pt'}")






# plot all four metrics across epochs
# one line per metric
def plot_training(history: list):

    epochs = [h["epoch"] for h in history]
    metrics = ["precision", "mrr", "map", "ndcg"]
    labels = ["precision@5", "mrr", "map", "ndcg"]
    colors = ["#2980B9", "#E67E22", "#27AE60", "#8E44AD"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("clip fine tuning - metrics per epoch", fontsize=13)

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        values = [h[metric] for h in history]

        ax.plot(epochs, values, color = color, linewidth=2, marker='o', markersize=5)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(values[0], color="gray", linestyle="--", alpha=0.5, label="baseline")

    plt.tight_layout()
    plt.savefig(base_dir / "fine_tune_metrics.png", dpi = 150, bbox_inches="tight")
    plt.show()
    print("plot saved")


def load_finetuned_clip():

    ckpt_path = ckpt_dir / "clip_fintetuned_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError("no checkpoint found")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    print("")


    return model, preprocess, tokenizer




if __name__ == "__main__":
    fine_tune()
