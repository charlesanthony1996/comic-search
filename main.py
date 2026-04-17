import json
import torch
import numpy
import os

from pathlib import Path
import zipfile
import argparse

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
    # keywords for spiderman would be amazing, spectacular, spider
    "amazing": "spiderman",
    "spider": "spiderman"
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


# extract the downloadable format from getcomics.org
def extract_all():
    # parent of the directory is searchable and exists
    image_dir.mkdir(parents=True, exist_ok = True)
    
    # check for file formats that have .pdf or .cbz 
    sources = list(datasets.glob(".pdf")) + list(datasets.glob("*.cbz")) + list(datasets.glob("*.cbz"))

    if not sources:
        print("no endpoint found")

    # counter variable
    total = 0
    # loop over each file from sources
    for src in sources:
        # earlier we used the superhero name directly but here we take the entire filename and use it inside
        # the infer function
        char = infer_charachter(src.name)
        print(f"Extracting: {src.name} -> charachter {char}")
