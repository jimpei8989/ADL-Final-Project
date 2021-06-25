import random

from typing import Any, List


from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForCategorical(DSTDatasetForDST):
    def __init__(self, *args, negative_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.negative_ratio = negative_ratio

    def expand(self, dialogue) -> List[Any]:
        ret = []
        for turn_idx in range(0, len(dialogue["turns"]), 2):
            turn = dialogue["turns"][turn_idx]

            assert turn["speaker"] == "USER"

            categorical_pairs = [
                (frame["service"], slot)
                for frame in turn["frames"]
                for slot in frame["state"]["slot_values"]
                if self.schema.service_by_name[frame["service"]].slot_by_name[slot].is_categorical
            ]

            ret.append((turn_idx, categorical_pairs))
        return ret

    def check_data(self, dialogue, other):
        assert len(other[1]) > 0

    def form_data(self, dialogue, other) -> dict:
        turn_idx, categorical_pairs = other

        positive = float(random.random() * (1 + self.negative_ratio) < 1.0)
        service_name, slot_name = draw_from_list(categorical_pairs)

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        the_frame = next(
            filter(lambda f: f["service"] == service_name, dialogue["turns"][turn_idx]["frames"])
        )
        correct = the_frame["state"]["slot_values"][slot_name][0]

        if positive:
            value = correct
        else:
            value = draw_from_list([ans for ans in slot.possible_values if ans != correct])

        ret = {"type": 1}
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][: turn_idx + 1],
                latter=f" {self.tokenizer.sep_token} ".join(
                    [service.description, slot.description, value]
                ),
                max_length=self.max_seq_length,
            )
        )
        ret.update({"value_labels": positive})
        return ret


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from datasets.schema import Schema

    schema = Schema.load_json(Path("../dataset/data/schema.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DSTDatasetForDSTForCategorical(
        Path("../dataset/data-0614/train"),
        schema=schema,
        tokenizer=tokenizer,
        system_token=tokenizer.sep_token,
        user_token=tokenizer.sep_token,
    )

    print(len(dataset))
    print(dataset[0])
