from torch.utils.data import Dataset
from pathlib import Path
import json
import os


class DSTDataset(Dataset):
    def __init__(self, json_dir: Path) -> None:
        self.root = json_dir
        self.file_names = os.listdir(json_dir)
        self.data = []
        for fname in self.file_names:
            d = json.load(open(self.root / fname, "r"))
            self.data.append(d)

    def __getitem__(self, index: int):
        pass

    def __len__(self) -> int:
        pass


if __name__ == "__main__":
    dataset = DSTDataset(Path("dataset/data/train"))
