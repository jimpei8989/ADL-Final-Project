import random

from typing import Any, List

from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForSlot(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        negative_ratio: float = 1.0,
        **kwargs,
    ):
        self.negative_ratio = negative_ratio

        super().__init__(*args, **kwargs)

    def expand1(self, dialogue) -> List[Any]:
        ret = []

        turn_indices = (
            [len(dialogue["turns"]) - 2]
            if self.last_user_turn_only
            else range(0, len(dialogue["turns"]), 2)
        )

        all_pairs = set(
            (service, slot)
            for service in dialogue["services"]
            for slot in self.schema.service_by_name[service].slot_by_name
        )

        for turn_idx in turn_indices:
            turn = dialogue["turns"][turn_idx]
            assert turn["speaker"] == "USER"

            positive_pairs = [
                (frame["service"], slot)
                for frame in turn["frames"]
                for slot in frame["state"]["slot_values"]
            ]
            negative_pairs = list(all_pairs - set(positive_pairs))

            ret.append((0, turn_idx, positive_pairs, negative_pairs))

        return ret

    def expand2(self, dialogue):
        ret = []
        turns = dialogue["turns"]
        begin_turn_idx, cursor = 0, 0

        all_pairs = set(
            (service, slot)
            for service in dialogue["services"]
            for slot in self.schema.service_by_name[service].slot_by_name
        )

        while True:
            cur_token_cnt = 0
            while cursor < len(turns):
                turn_token_len = len(self.form_turn(turns[cursor])[1])

                if cur_token_cnt + turn_token_len > self.former_max_len:
                    break
                else:
                    cursor += 1
                    cur_token_cnt += turn_token_len

            if self.ensure_user_on_both_ends:
                if cursor % 2 == 1:
                    cursor -= 1

            positive_pairs = [
                (frame["service"], slot)
                for turn in turns[begin_turn_idx : cursor + 1]
                if turn["speaker"] == "USER"
                for frame in turn["frames"]
                for slot in frame["state"]["slot_values"]
            ]
            negative_pairs = list(all_pairs - set(positive_pairs))

            ret.append((begin_turn_idx, cursor, positive_pairs, negative_pairs))

            if cursor >= len(dialogue["turns"]) - 2:
                break
            else:
                begin_turn_idx = cursor - self.overlap_turns
                cursor = begin_turn_idx

        return ret

    def check_data(self, dialogue, other):
        assert all(len(obj) > 0 for obj in other[2:])
        assert all(
            service in dialogue["services"] for pairs in other[2:] for service, slot in pairs
        )

    def form_data(self, dialogue, other) -> dict:
        begin_turn_idx, end_turn_idx, positive_pairs, negative_pairs = other

        positive = float(random.random() * (1 + self.negative_ratio) < 1.0)
        service_name, slot_name = draw_from_list(positive_pairs if positive else negative_pairs)

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        ret = {"type": 0}
        ret.update({"begin_turn_idx": begin_turn_idx, "end_turn_idx": end_turn_idx})
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][begin_turn_idx : end_turn_idx + 1],
                latter=f" {self.tokenizer.sep_token} ".join(
                    [service.description, slot.description]
                ),
                max_length=self.max_seq_length,
            )
        )
        ret.update({"slot_labels": positive})
        return ret


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
        strategy="segment",
        test_mode=True,
    )

    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
