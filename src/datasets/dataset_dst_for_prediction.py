from bisect import bisect_right
from collections import defaultdict
from typing import Tuple, Optional

from transformers import BatchEncoding

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForPrediction(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        test_mode=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.test_mode = test_mode

        self.dialogue_ids_inv = {did: i for i, did in enumerate(self.dialogue_ids)}

        self.another_prefix_sum = [0]
        self.dialogue_service_slots = defaultdict(list)
        for i in range(super().__len__()):
            dialogue, turn_idx = super().__getitem__(i)
            cnt = 0
            for service in dialogue["services"]:
                for slot in self.schema.service_by_name[service].slots:
                    self.dialogue_service_slots[dialogue["dialogue_id"]].append(
                        (service, slot.name)
                    )
                    cnt += 1

            self.another_prefix_sum.append(self.another_prefix_sum[-1] + cnt)

    def __len__(self):
        if self.test_mode:
            return 10
        else:
            return self.another_prefix_sum[-1]

    def __getitem__(self, index: int):
        # Finds a super_index s.t. APS[super_index] <= index < APS[super_index + 1]
        super_index = bisect_right(self.another_prefix_sum, index) - 1

        offset = index - self.another_prefix_sum[super_index]

        dialogue, turn_idx = super().__getitem__(super_index)
        service, slot = self.dialogue_service_slots[dialogue["dialogue_id"]][offset]

        utterance, encoded = self.form_input(
            turns=dialogue["turns"][: turn_idx + 1],
            slot_description=self.schema.service_by_name[service].slot_by_name[slot].description,
            max_length=self.max_seq_length,
        )

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "service": service,
            "slot": slot,
            "turn_idx": turn_idx,
            "utterance": utterance,
            "encoded": encoded,
            "input_ids": encoded.input_ids.squeeze(0),
        }

    def check_item(self, index: int):
        return True

    def form_utterance(
        self,
        turns,
        max_length: Optional[int] = None,
    ):
        cur_length, turn_idx = 0, len(turns) - 1
        utterances = []

        while turn_idx >= 0:
            turn = turns[turn_idx]
            special_token = self.user_token if turn["speaker"] == "USER" else self.system_token
            utterance = (f"{special_token} " if special_token else "") + turn["utterance"]

            tokenized = self.tokenizer.tokenize(utterance)
            if max_length is not None and cur_length + len(tokenized) > max_length:
                break
            else:
                utterances.append(utterance)
                cur_length += len(tokenized)
                turn_idx -= 1

        return " ".join(utterances[::-1])

    def form_input(
        self,
        turns,
        slot_description: str,
        answer: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[str, BatchEncoding]:
        # [CLS] utterance [SEP] slot [SEP] answer [SEP]
        latter = (
            slot_description
            if answer is None
            else " ".join([slot_description, self.tokenizer.sep_token, answer])
        )
        latter_len = len(self.tokenizer.tokenize(latter))

        utterance = self.form_utterance(
            turns,
            max_length=(max_length - latter_len if max_length is not None else None),
        )

        return utterance, self.tokenizer(
            [utterance], [latter], padding="max_length", return_tensors="pt"
        )


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
    )
    print(len(dataset))

    for i in range(10):
        print(dataset[i])
