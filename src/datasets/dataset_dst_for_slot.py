import random

from typing import Any, List

from datasets.dataset_dst import DSTDatasetForDST


def draw(a: list):
    return a[int(random.random() * len(a))]


class DSTDatasetForDSTForSlot(DSTDatasetForDST):
    def __init__(self, *args, negative_ratio: float = 1.0, **kwargs):
        self.negative_ratio = negative_ratio

        super().__init__(*args, **kwargs)

    def expand(self, dialogue) -> List[Any]:
        ret = []
        for turn_idx in range(0, len(dialogue["turns"]), 2):
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

    def check_item(self, index: int):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        assert (
            len(self.get_positive_service_slot_names(turn)) > 0
            and len(self.get_negative_slots(dialogue, turn)) > 0
        )

    def form_data(self, dialogue, other) -> dict:
        turn_idx, positive_pairs, negative_pairs = other

        positive = float(random.random() * (1 + self.negative_ratio) < 1.0)

        service, slot = draw(positive_pairs if positive else negative_pairs)

        ret = self._form_data(
            dialogue=dialogue,
            turns=dialogue["turns"][:turn_idx + 1],
            latter=self.schema.service_by_name[service].slot_by_name[slot].description,
            max_length=self.max_seq_length,
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
