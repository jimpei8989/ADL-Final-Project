import random

from typing import Any, List

from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForSlot(DSTDatasetForDST):
    def __init__(
        self, *args, negative_ratio: float = 1.0, last_user_turn_only: bool = False, **kwargs
    ):
        self.negative_ratio = negative_ratio
        self.last_user_turn_only = last_user_turn_only

        super().__init__(*args, **kwargs)

    def expand(self, dialogue) -> List[Any]:
        ret = []

        turn_indices = (
            [len(dialogue["turns"]) - 2]
            if self.last_user_turn_only
            else range(0, len(dialogue["turns"]), 2)
        )

        for turn_idx in turn_indices:
            turn = dialogue["turns"][turn_idx]
            assert turn["speaker"] == "USER"

            all_pairs = [
                (service, slot)
                for service in dialogue["services"]
                for slot in self.schema.service_by_name[service].slot_by_name
            ]
            positive_pairs = [
                (frame["service"], slot)
                for frame in turn["frames"]
                for slot in frame["state"]["slot_values"]
            ]
            negative_pairs = list(set(all_pairs) - set(positive_pairs))

            ret.append((turn_idx, positive_pairs, negative_pairs))

        return ret

    def check_data(self, dialogue, other):
        assert all(len(obj) > 0 for obj in other[1:])
        assert all(
            service in dialogue["services"] for pairs in other[1:] for service, slot in pairs
        )

    def form_data(self, dialogue, other) -> dict:
        turn_idx, positive_pairs, negative_pairs = other

        positive = float(random.random() * (1 + self.negative_ratio) < 1.0)
        service_name, slot_name = draw_from_list(positive_pairs if positive else negative_pairs)

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        ret = {"type": 0}
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][: turn_idx + 1],
                latter=f" {self.tokenizer.sep_token} ".join(
                    [service.description, slot.description]
                ),
                max_length=self.max_seq_length,
            )
        )
        ret.update({"slot_labels": positive})
        return ret

    def form_latter(self, slot_description) -> str:
        return slot_description


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from datasets.schema import Schema

    schema = Schema.load_json(Path("../dataset/data/schema.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DSTDatasetForDSTForSlot(
        Path("../dataset/data-0614/train"),
        schema=schema,
        tokenizer=tokenizer,
        system_token=tokenizer.sep_token,
        user_token=tokenizer.sep_token,
    )

    print(len(dataset))
    print(dataset[0])
