import json
import os
from pathlib import Path

from torch.utils.data import Dataset

from utils.tqdmm import tqdmm


class DSTDataset(Dataset):
    def __init__(self, json_dir: Path, tokenizer=None, max_seq_length=512) -> None:
        self.root = json_dir
        self.file_names = sorted(os.listdir(json_dir))
        self.data = []
        for fname in tqdmm(self.file_names, desc="Reading training jsons"):
            d = json.load(open(self.root / fname, "r"))
            self.data += d

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


if __name__ == "__main__":
    dataset = DSTDataset(
        Path("dataset/data-0610/new-train"), Path("dataset/data/schema.json")
    )
