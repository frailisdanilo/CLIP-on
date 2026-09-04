import torch
import pathlib
from torchvision.datasets import CelebA

DATASET_ROOT = "./Celeba"

SPLITS = ["all", "train", "valid", "test"]

def load_split(split="all"):
    if split in SPLITS:
        return CelebA(root=DATASET_ROOT, split=split, download=False)

celeba_test = load_split("test")

print(celeba_test[0][0])