import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets.dataset import DSTDataset
from datasets.schema import Schema
from utils.logger import logger


class DSTDatasetForDST(DSTDataset):
    def __init__(self, *args, schema: Optional[Schema] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.schema = schema

        logger.info(f"Successfully loaded {len(self.data)} dialogues...")
        self.data = sorted(self.data, key=lambda d: d["dialogue_id"])
        self.dialogue_by_id = {d["dialogue_id"]: d for d in self.data}

        self.dialogue_user_turns = defaultdict(list)
        self.dialogue_ids = []
        self.dialogue_num_user_turns_ps = [0]

        for d in self.data:
            d_id = d["dialogue_id"]
            user_turns = [i for i, t in enumerate(d["turns"]) if t["speaker"] == "USER"]

            self.dialogue_user_turns[d_id] = user_turns
            self.dialogue_ids.append(d_id)
            self.dialogue_num_user_turns_ps.append(
                self.dialogue_num_user_turns_ps[-1] + len(user_turns)
            )

        logger.info(
            f"Finished preprocessing dialogues, there're {len(self)} user turns in total..."
        )

        logger.info("The first sample: " + json.dumps(self[0], indent=2))

    def __getitem__(self, index: int):
        # find an i s.t. a[i] <= index < a[i + 1]
        i = bisect_right(self.dialogue_num_user_turns_ps, index) - 1
        offset = index - self.dialogue_num_user_turns_ps[i]

        dialogue_id = self.dialogue_ids[i]
        dialogue = self.dialogue_by_id[dialogue_id]

        assert 0 <= offset < len(self.dialogue_user_turns[dialogue_id])

        turn_id = self.dialogue_user_turns[dialogue_id][offset]

        return {
            "dialogue_id": dialogue_id,
            "turn_id": turn_id,
            "turns": dialogue["turns"][:(turn_id + 1)],
        }

    def __len__(self):
        return self.dialogue_num_user_turns_ps[-1]


if __name__ == "__main__":
    dataset = DSTDatasetForDST(
        Path("../dataset/data-0610/new-train"), Path("../dataset/data/schema.json")
    )
