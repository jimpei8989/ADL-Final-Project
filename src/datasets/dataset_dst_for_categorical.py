import random

from typing import Any, List


from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForCategorical(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        negative_ratio=1.0,
        **kwargs,
    ):
        self.negative_ratio = negative_ratio

        super().__init__(*args, **kwargs)

    def extract_categorical_pairs(self, turn):
        pass

    def expand1(self, dialogue) -> List[Any]:
        ret = []

        turn_indices = (
            [len(dialogue["turns"]) - 2]
            if self.last_user_turn_only
            else range(0, len(dialogue["turns"]), 2)
        )

        for turn_idx in turn_indices:
            turn = dialogue["turns"][turn_idx]
            assert turn["speaker"] == "USER"

            categorical_pairs = [
                (frame["service"], slot, frame["state"]["slot_values"][slot][0])
                for frame in turn["frames"]
                for slot in frame["state"]["slot_values"]
                if self.schema.service_by_name[frame["service"]].slot_by_name[slot].is_categorical
            ]

            ret.append((0, turn_idx, categorical_pairs))

        return ret

    def expand2(self, dialogue) -> List[Any]:
        ret = []
        turns = dialogue["turns"]
        begin_turn_idx, cursor = 0, 0

        while True:
            cur_token_cnt = 0
            categorical_pairs = {}  # (service, slot) -> correct answer

            while cursor < len(turns):
                turn = turns[cursor]
                turn_token_len = len(self.form_turn(turn)[1])

                if cur_token_cnt + turn_token_len > self.former_max_len:
                    break
                else:

                    if turn["speaker"] == "USER":
                        categorical_pairs.update(
                            {
                                (frame["service"], slot): frame["state"]["slot_values"][slot][0]
                                for frame in turn["frames"]
                                for slot in frame["state"]["slot_values"]
                                if self.schema.service_by_name[frame["service"]]
                                .slot_by_name[slot]
                                .is_categorical
                            }
                        )

                    cursor += 1
                    cur_token_cnt += turn_token_len
            else:
                # minus one when the cursor reaches the en
                cursor -= 1

            if self.ensure_user_on_both_ends:
                if cursor % 2 == 1:
                    cursor -= 1

            categorical_pairs = [(*k, v) for k, v in categorical_pairs.items()]

            ret.append((begin_turn_idx, cursor, categorical_pairs))

            if cursor >= len(dialogue["turns"]) - 2:
                break
            else:
                begin_turn_idx = cursor - self.overlap_turns
                cursor = begin_turn_idx

        return ret

    def check_data(self, dialogue, other):
        assert len(other[2]) > 0
        assert all(service in dialogue["services"] for service, _, _ in other[2])

    def form_data(self, dialogue, other) -> dict:
        begin_turn_idx, end_turn_idx, categorical_pairs = other

        positive = float(random.random() * (1 + self.negative_ratio) < 1.0)
        service_name, slot_name, correct = draw_from_list(categorical_pairs)
        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        if positive:
            value = correct
        else:
            value = draw_from_list([ans for ans in slot.possible_values if ans != correct])

        ret = {"type": 1}
        ret.update({"begin_turn_idx": begin_turn_idx, "end_turn_idx": end_turn_idx})
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][begin_turn_idx : end_turn_idx + 1],
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
        strategy="segment",
    )

    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
