from dataset import DSTDataset
from pathlib import Path
from typing import Iterable, Tuple


def pairwise(iterable: Iterable) -> Iterable[Tuple]:
    # s -> (s0,s1), (s2,s3), ..., (sn,sn+1), ...
    return zip(*[iter(iterable)] * 2)


class DSTDatasetForNLG(DSTDataset):
    def __init__(
        self,
        json_dir: Path,
        schema_file: Path,
        tokenizer=None,
        max_seq_length=512,
        mode="test",
    ) -> None:
        super().__init__(json_dir, schema_file, tokenizer, max_seq_length)
        self.nlg_data = []

        for d in self.data:
            for idx, (user, system) in enumerate(pairwise(d["turns"])):
                assert user["speaker"] == "USER"
                assert system["speaker"] == "SYSTEM"

                tmp = {
                    "dialogue_id": d["dialogue_id"],
                    "turns_id": [idx * 2, idx * 2 + 1],
                    "user_utterance": user["utterance"],
                    "system_utterance": system["utterance"],
                }

                if mode == "train":
                    if "beginning" not in system or "end" not in system:
                        continue
                    if system["beginning"] == [] and system["end"] == []:
                        continue
                    tmp["beginning"] = system["beginning"]  # chit-chat begining
                    tmp["end"] = system["end"]  # chit-chat end

                self.nlg_data.append(tmp)

    def __getitem__(self, index):
        return self.nlg_data[index]

    def __len__(self):
        return len(self.nlg_data)


if __name__ == "__main__":
    for paths in ["train", "dev", "test_seen", "test_unseen"]:
        ds = DSTDatasetForNLG(
            Path("dataset/data") / paths,
            Path("dataset/data/schema.json"),
            mode=("test" if "test" in paths else "train"),
        )

        print(paths, len(ds))
