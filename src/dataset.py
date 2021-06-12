from torch.utils.data import Dataset
from pathlib import Path
import json
import os


class DSTDataset(Dataset):
    def __init__(
        self, json_dir: Path, schema_file: Path, tokenizer=None, max_seq_length=512
    ) -> None:
        self.root = json_dir
        self.file_names = os.listdir(open(schema_file, "r"))
        self.data = []
        self.schema_file = json.load(schema_file)
        for fname in self.file_names:
            d = json.load(open(self.root / fname, "r"))
            self.data.append(d)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index: int):
        pass

    def __len__(self) -> int:
        pass


if __name__ == "__main__":
    dataset = DSTDataset(Path("dataset/data/train"))
