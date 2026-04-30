import json
import torch
import numpy as np
import os

from pathlib import Path
import zipfile
import argparse


# might not work in python3 3.12.2
import fitz
# or should we import pymupdf

import pypdf

# openai clip model
import open_clip

from PIL import Image

import subprocess

# for the final ranking plots to be shown
import matplotlib.pyplot as plt

# write the os indenpendent switch here to run locally
# not implemented yet


base_dir = Path(__file__).parent

# for all datasets
datasets = base_dir / "datasets"

# images per dataset/comic
image_dir = base_dir / "dataset"

# indexing file saved as numpy points?
index_file = base_dir / "clip_index.npz"

# in the future this has to be dynamic (need to think of a smarter capture arch). 
# since adding .cbz comics would be alot and the marvel universe itself
# has many unique keyword/names for the same characther?
charachter_map = {
    "venom": "venom",

    "wolverine":  "wolverine",
    "logan": "wolverine",

    # keywords for spiderman would be amazing, spectacular, spider
    "amazing": "spiderman",
    "spider": "spiderman",

    "punisher": "punisher",
    "frank": "punisher",
    "castle": "punisher",

    "matt": "daredevil",
    "murdock": "daredevil",
    "daredevil": "daredevil",
    "devil": "daredevil"
}

# this is a basic function to infer.
# use bm25 for query search for the input
# can we use this for title searching of a comic too? explorable
def infer_charachter(filename):
    # switch the names to lowercase so all special cases of the same name can be tracked
    name = filename.lower()
    # loop over through the items
    for keyword, char in charachter_map.items():
        # if the word is present the return the name of the superhero
        if keyword in name:
            return char
    # returns the first based result / superhero. because its zero indexed
    return Path(filename).stem.split("_")[0]


# for the time being the mini script would be function based
# class based would be the final outcome.

# dataset 1 name -> The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf
# dataset 2 name -> Ultimate Wolverine 016 (2026) (Digital) (Shan-Empire).pdf
# dataset 3 name -> Web of Venom 001 (2026) (Digital) (Shan-Empire).pdf

# log the infer charachter function
infer_charachter("The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf")
# spiderman should print out here
# do we need pytest stuff here? maybe.. idk
print(infer_charachter("The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf"))
print(infer_charachter("Ultimate Wolverine 016 (2026) (Digital) (Shan-Empire).pdf"))
print(infer_charachter("Web of Venom 001 (2026) (Digital) (Shan-Empire).pdf"))


# extract the downloadable format from getcomics.org
def extract_all():
    # parent of the directory is searchable and exists
    image_dir.mkdir(parents=True, exist_ok = True)
    
    # check for file formats that have .pdf or .cbz 
    sources = list(datasets.glob("*.pdf")) + list(datasets.glob("*.cbz"))

    # eventhough we decided to go with .cbz format. we just might need pdfs still.
    # so its just a fallback for now.
    # might be able to take it off later
    sources = list(datasets.rglob("*.cbz")) + list(datasets.rglob("*.pdf"))

    if not sources:
        print("no files found in this format")

    # counter variable
    total = 0
    # loop over each file from sources
    for src in sources:
        # earlier we used the superhero name directly but here we take the entire filename and use it inside
        # the infer function
        char = infer_charachter(src.name)
        print(f"Extracting: {src.name} -> charachter {char}")

        count = extract_cbz(src, char) if src.suffix.lower() == ".cbz" else extract_pdf(src, char)

        print("page count: ", count)

        total += count

    print("total pages in image_dir: ", total)
    for char in set(charachter_map.values()):
        n = len(list(image_dir.glob(f"{char}_*.jpg")))
        if n:
            # print(f"{char.capitalize():<12: {n} pages}")
            print(f"{char.capitalize():<12} {n} pages")

# can dpi be a hyperparameter? learnable? explore this..
def extract_pdf(pdf_path, char, dpi = 150):

    # fitz is used to extract pages from a pdf
    # based on the pymupdf library

    # takes in strings of the pdf and saves it in the doc variable
    doc = fitz.open(str(pdf_path))
    print(doc)

    # takes in the length of the strings of the doc for each pdf
    total = len(doc)

    # skip in 3 units or from a range thats less than 3
    skip = set(range(3)) | set(range(total - 3, total))

    issue = pdf_path.stem[:20].replace(" ", "_")

    saved = 0

    for i, page in enumerate(doc):
        if i in skip:
            continue
        
        # matrix
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        # get the pixelmap here? print it out
        pix = page.get_pixmap(matrix=mat)

        # save it under this format
        pix.save(str(image_dir / f"{char}_page_{issue}_{i:04d}.jpg"))

        saved += 1

    return saved


# extract a pdf here. use the above function
# this function extracts a single pdf and stores it in the dataset folder
# make sure you copy the relative path
# we will do this inside the extract_all? or are we moving it?
# extract_pdf(Path("The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf"), "s")
# extract_pdf(Path("/Users/charles/Desktop/comic-search-project/datasets/Ultimate Wolverine 016 (2026) (Digital) (Shan-Empire).pdf"), "s")


# not used yet
# switching to .cbz format till proper simple baseline model is done
def extract_cbz(cbz_path, char):
    issue = cbz_path.stem[:20].replace(" ", "_")

    saved = 0

    with zipfile.ZipFile(cbz_path, "r") as z:
        imgs = sorted([f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('__')])

        total = len(imgs)

        skip = set(range(3)) | set(range(total - 3, total))

        for i, name in enumerate(imgs):
            if i in skip:
                continue

            data = z.read(name)

            (image_dir / f"{char}_page_{issue}_{i:04d}.jpg").write_bytes(data)

            saved += 1

    return saved


# test the extract_cbz function


# better to switch to a model switching register?
# using the clip model as a function
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    # hugging face has variations here. need to check. will compare and decide on metrics
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.eval()

    return model, preprocess, tokenizer



def build_index():

    # collect all the images from the dataset directory. the ones that were already split per pdf
    # should we split per panel? not sure. explorable
    paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    # if the path is not present
    if not paths:
        print(f"no images in ' {image_dir}/' run --extract first to get it to the dataset folder")
        return
 
    # print(f"Loading CLIP ViT-B/32...")

    # load the clip model
    # load_clip is the function we wrote earlier
    model, preprocess, _ = load_clip()

    # encode the paths
    # print(f"encoding {len(paths)} images... here")

    # embedding vectors and the filenames are stored here
    vecs, names = [], []

    # loop over every page/split pdf image in the dataset folder. not datasets!
    for i, path in enumerate(paths):

        # open the image and check whether it has the three channels
        img    = Image.open(path).convert("RGB")

        # this is where clip applies its normalization/resize pipeline
        tensor = preprocess(img).unsqueeze(0)

        # speed it up. training on the cluster? need a smart switch here soon
        # for now disable gradient tracking to speed up inference and save memory
        # early stopping is needed.
        # a good model register would be needed here?
        with torch.no_grad():

            # one forward pass through the VIT and gets it to an embedding vector
            vec = model.encode_image(tensor)

            # l2 normalize the embedding so the cosine similarity gives you the dot product
            # dot product formula
            vec = vec / vec.norm(dim=-1, keepdim=True)

        # 
        vecs.append(vec.squeeze().numpy())

        # store the filename.. not the complete filepath!
        names.append(path.name)

        # just a progress update for every 50 images. 
        # dont need it in the future. just to check current length or count we have
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}...")

    # we have a 1 d vector now
    # so stack them into a 2d matrix of shape [N, embedding dimension]
    vecs = np.stack(vecs)

    # save the generated embedding matrix and the filename array into a single compressed numpy archive
    np.savez(index_file, vecs=vecs, filenames=np.array(names))

    # confirming the save with and the shape of the emedding matrix info
    print(f"\index saved -> '{index_file}'  ({vecs.shape[0]} images x {vecs.shape[1]}-d)")


# should be able to keep this in a seperate file soon
def run_search(query, top_k=5, show=False):

    # there should be an index always
    # future location of the file needs to be somewhere
    # otherwise stop the run search immediately
    if not index_file.exists():
        print("no index here... run the --index param cmd first")
        return
    
    # load the compressed numpy archive here
    data = np.load(index_file, allow_pickle=True)

    # the 2d converted array of the shape [N, embedding_dim], precomputed image embedding here
    vecs = data["vecs"]

    # so all filenames are converted to a list here.
    # just a plain python list
    filenames = data["filenames"].tolist()

    # load the clip model and the tokenizer
    # no image processing here
    # its for text queries here
    model, _, tokenizer = load_clip()

    # tokenize the query string here
    # then you wrap it in a batch
    # because the shape should match. the tokenizer expects a batch
    tokens = tokenizer([query])

    # same argument as before
    # no gradient tracking for now. skip
    with torch.no_grad():
        qvec = model.encode_text(tokens)
        qvec = qvec / qvec.norm(dim=-1, keepdim=True)

    # remove the batch dimension 
    # get it ready to calculate the dot prodfuct in the next steps
    # print shape at this point and compare. before and after. (to do!)
    qvec = qvec.squeeze().numpy()

    # compute the cosine similarity here.
    # this is between every image embedding and every query vector
    # check the output shape here
    scores  = vecs @ qvec

    # the -1 reverses the ascending to descending first and to the top k'th limit
    top_idx = np.argsort(scores)[::-1][:top_k]

    # print the ranked results here
    print(f'query: "{query}"\n')
    print(f"{'rank':<6} {'score':<8} file")
    print("-" * 60)

    # loop over from the top k ones to 1
    for rank, i in enumerate(top_idx, 1):
        # print the metrics here and the filenames for each
        print(f"{rank:<6} {scores[i]:.4f}   {filenames[i]}")

    # display the results here which are shown on the console too
    # matplotlib is used.
    # can switch this to another file at this point
    if show:
        # one subplot per plot
        fig, axes = plt.subplots(1, len(top_idx), figsize=(4 * len(top_idx), 5))

        # 
        if len(top_idx) == 1:
            axes = [axes]

        # pair each subplot with its corresponding result index
        for ax, i in zip(axes, top_idx):

            # is it okay to read it from this directory?
            # something seperate till we get the compressed numpy archive?
            # initial training time will take a bit more time then. explorable
            img = Image.open(image_dir / filenames[i])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"#{list(top_idx).index(i)+1}  {scores[i]:.3f}\n{filenames[i]}", fontsize=7)

        # suptitle will have the input query. tune fontsize.
        plt.suptitle(f'"{query}"', fontsize=10)
        plt.tight_layout()
        plt.show()
 

# the golden set that test for unique cases and also checks robustness
# seperated per superhero?
# what if two appear in the same setting? how to find that? should train on relevant comics
golden_test_set = [
    # spiderman
    {"query": "Spider-Man in red and blue suit swinging","expected": "spiderman"},
    {"query": "close up face of masked superhero","expected": "spiderman"},

    # venom
    {"query": "black symbiote monster with huge teeth and white eyes", "expected": "venom"},
    {"query": "dark villain with cape and supernatural powers","expected": "venom"},

    # wolverine
    {"query": "hero with adamantium claws outstretched","expected": "wolverine"},
    {"query": "snowy winter forest landscape with superhero","expected": "wolverine"},
    {"query": "fire explosion orange dramatic portal scene","expected": "wolverine"},

    # mixed appearances? need a listof comics that only share these charachters
    # or with new ones we have never trained on before?
    # how can this be a golden set query? something here
]
 
def evaluate():
 
    if not index_file.exists():
        print("no index yet here please run the --index first")
        return
 
    data      = np.load(index_file, allow_pickle=True)
    vecs      = data["vecs"]
    filenames = data["filenames"].tolist()
    model, _, tokenizer = load_clip()
 
    print("golden set evaluation: " + "=" * 65)
    hits = 0
    for item in golden_test_set:
        tokens = tokenizer([item["query"]])
        with torch.no_grad():
            qvec = model.encode_text(tokens)
            qvec = qvec / qvec.norm(dim=-1, keepdim=True)
        qvec = qvec.squeeze().numpy()
 
        scores  = vecs @ qvec
        top_idx = np.argsort(scores)[::-1][:3]
        top3    = [filenames[i] for i in top_idx]
        hit     = any(item["expected"] in f for f in top3)
        hits   += int(hit)
 
        print(f"[{'HIT ' if hit else 'MISS'}] {item['query']}")
        print(f"Expected: {item['expected']}")
        for rank, i in enumerate(top_idx, 1):
            marker = " <--" if item["expected"] in filenames[i] else ""
            print(f"       {rank}. [{scores[i]:.4f}] {filenames[i]}{marker}")
 
    print(f"\result: {hits}/{len(golden_test_set)} queries hit in top-3")
 

# better to have a zsh here to double click and run once
# make it for windows
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="comic similarity search")

#     parser.add_argument("--extract", action="store_true", help="extract pages from all PDFs/CBZs in datasets/")

#     parser.add_argument("--index",action="store_true",help="build CLIP image index")

#     parser.add_argument("--search",metavar="QUERY",help="search with a text query")

#     parser.add_argument("--eval",action="store_true", help="run golden set evaluation")

#     parser.add_argument("--top",type=int,default=5, help="number of results (default: 5)")

#     # display the results 
#     parser.add_argument("--show", action="store_true", help="display result images in a window")

# # and update the search call:
#     args = parser.parse_args()
 
#     if   args.extract: extract_all()
#     elif args.index:   build_index()
#     # elif args.search:  run_search(args.search, top_k=args.top)
#     elif args.eval:    evaluate()
    
#     # does order matter here?
#     elif args.search: run_search(args.search, top_k=args.top, show=args.show)

#     else:
#         parser.print_help()

extract_all()

build_index()

run_search("black symbiote with teeth", top_k=3, show=True)

evaluate()