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

def infer_charachter(filename):
    # switch the names to lowercase so all special cases of the same name can be tracked
    name = filename.lower()
    # loop over through the items
    for keyword, char in charachter_map.items():
        # if the word is present the return the name of the superhero
        if keyword in name:
            return char
    # 
    return Path(filename).stem.split("_")[0]


# for the time being the mini script would be function based
# class based would be the final outcome.

# dataset 1 name -> The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf

# log the infer charachter function
infer_charachter("The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf")
# spiderman should print out here
# do we need pytest stuff here? maybe.. idk
print(infer_charachter("The Amazing Spider-Man v03 (Pocket Book) (1979) (Edit Special) c2c.pdf"))

