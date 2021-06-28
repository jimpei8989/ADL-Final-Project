from typing import Any, List

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForPrediction(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def expand1(self, dialogue) -> List[Any]:
        ret = []

        turn_indices = (
            [len(dialogue["turns"]) - 2]
            if self.last_user_turn_only
            else range(0, len(dialogue["turns"]), 2)
        )

        for turn_idx in turn_indices:
            for service_name in dialogue["services"]:
                for slot in self.schema.service_by_name[service_name].slots:
                    ret.append((0, turn_idx, service_name, slot.name))

        return ret

    def expand2(self, dialogue) -> List[Any]:
        ret = []
        turns = dialogue["turns"]
        begin_turn_idx = 0

        while True:
            if self.ensure_user_on_both_ends and turns[begin_turn_idx]["speaker"] == "SYSTEM":
                begin_turn_idx += 1

            cursor, cur_token_cnt = begin_turn_idx, 0

            while cursor < len(turns):
                turn_token_len = len(self.form_turn(turns[cursor])[1])

                if cur_token_cnt + turn_token_len > self.former_max_len:
                    break
                else:
                    cursor += 1
                    cur_token_cnt += turn_token_len
            # Always -1 to make it right-close
            cursor -= 1

            if self.ensure_user_on_both_ends:
                if cursor % 2 == 1:
                    cursor -= 1

            if cursor >= len(dialogue["turns"]) - 2:
                break
            else:
                for service_name in dialogue["services"]:
                    for slot in self.schema.service_by_name[service_name].slots:
                        ret.append((begin_turn_idx, cursor, service_name, slot.name))

                begin_turn_idx = cursor - self.overlap_turns
        return ret

    def check_data(self, dialogue, other):
        return True

    def form_data(self, dialogue, other) -> dict:
        begin_turn_idx, end_turn_idx, service_name, slot_name = other

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        ret = self._form_data(
            dialogue=dialogue,
            turns=dialogue["turns"][begin_turn_idx : end_turn_idx + 1],
            latter=f" {self.tokenizer.sep_token} ".join([service.description, slot.description]),
            max_length=self.max_seq_length,
        )
        ret.update({"_key": (dialogue["dialogue_id"], *other)})
        return ret


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer

    from datasets.schema import Schema

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    schema = Schema.load_json(Path("../dataset/data/schema.json"))

    dataset = DSTDatasetForDSTForPrediction(
        Path("../dataset/data/test_seen"),
        schema=schema,
        tokenizer=tokenizer,
        strategy="segment",
    )

    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
