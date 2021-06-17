from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from datasets.dataset import DSTDataset
from datasets.schema import Schema, Slot
from utils.logger import logger


class DSTDatasetForDST(DSTDataset):
    def __init__(
        self,
        *args,
        schema: Schema = None,
        user_token: Optional[str] = None,
        system_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert schema is not None and self.tokenizer is not None

        self.schema = schema
        self.user_token = user_token
        self.system_token = system_token

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

    def get_utterance_tokens(self, dialogue, turn_idx: int, max_length: Optional[int] = None):
        tokens = []
        cur_len = 0

        while turn_idx >= 0:
            special_token = self.user_token if turn_idx % 2 else self.system_token
            t = self.tokenizer.tokenize(dialogue["turns"][turn_idx]["utterance"])
            if special_token is not None:
                t = [special_token] + t

            if max_length is not None and len(t) + cur_len > max_length:
                break

            tokens.append(t)
            cur_len += len(t)
            turn_idx -= 1
        return sum(tokens[::-1], [])

    def get_positive_service_slot_names(self, turn) -> List[Tuple[str, str]]:
        return [
            (frame["service"], slot)
            for frame in turn["frames"]
            for slot in frame["state"]["slot_values"]
        ]

    def get_all_service_slot_names(self, dialogue) -> List[Tuple[str, str]]:
        return [
            (service, slot)
            for service in dialogue["services"]
            for slot in self.schema.service_by_name[service].slot_by_name
        ]

    def get_positive_slots(self, turn) -> List[Slot]:
        return [
            self.schema.service_by_name[frame_name].slot_by_name[slot_name]
            for frame_name, slot_name in self.get_positive_service_slot_names(turn)
        ]

    def get_negative_slots(self, dialogue, turn):
        positive_names = set(self.get_positive_service_slot_names(turn))
        return [n for n in self.get_all_service_slot_names(dialogue) if n not in positive_names]

    def __getitem__(self, index: int):
        # find an i s.t. a[i] <= index < a[i + 1]
        i = bisect_right(self.dialogue_num_user_turns_ps, index) - 1
        offset = index - self.dialogue_num_user_turns_ps[i]

        dialogue_id = self.dialogue_ids[i]
        dialogue = self.dialogue_by_id[dialogue_id]

        assert 0 <= offset < len(self.dialogue_user_turns[dialogue_id])

        turn_idx = self.dialogue_user_turns[dialogue_id][offset]

        return (dialogue, turn_idx)

    def __len__(self):
        return self.dialogue_num_user_turns_ps[-1]


if __name__ == "__main__":
    dataset = DSTDatasetForDST(
        Path("../dataset/data-0610/new-train"), Path("../dataset/data/schema.json")
    )
