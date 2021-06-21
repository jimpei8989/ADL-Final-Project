import random

import torch

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForSlot(DSTDatasetForDST):
    def __init__(self, *args, negative_ratio: float = 1.0, **kwargs):
        self.negative_ratio = negative_ratio

        super().__init__(*args, **kwargs)

    def __len__(self):
        return super().__len__()

    def check_item(self, index: int):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        assert (
            len(self.get_positive_service_slot_names(turn)) > 0
            and len(self.get_negative_slots(dialogue, turn)) > 0
        )

    def __getitem__(self, index: int):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        positive = int(random.random() * (1 + self.negative_ratio) < 1.0)

        candidates = (
            self.get_positive_service_slot_names(turn)
            if positive
            else self.get_negative_slots(dialogue, turn)
        )

        slot_idx = int(random.random() * len(candidates))
        service, slot = candidates[slot_idx]
        slot_desc = self.schema.service_by_name[service].slot_by_name[slot].description

        slot_tokens = self.tokenizer.tokenize(slot_desc)
        utterance = self.get_utterance_tokens(
            dialogue, turn_idx, max_length=self.max_seq_length - len(slot_tokens) - 3
        )

        input_ids = self.tokenizer(
            self.tokenizer.convert_tokens_to_string(utterance),
            self.tokenizer.convert_tokens_to_string(slot_tokens),
            padding="max_length",
            max_length=512,
        ).input_ids
        # if (len(input_ids)) > self.max_seq_length:
        #     print(f"type: {type}")
        #     print(f"utterance: {utterance}")
        #     print(f"utterance len: {len(utterance)}")
        #     print(f"slot: {slot_tokens}")
        #     print(f"slot len: {len(slot_tokens)}")
        #     print(f"inputs: {self.tokenizer.convert_ids_to_tokens(input_ids)}")
        #     print(f"inputs len: {len(input_ids)}")
        #     print(f"turn: {turn}")
        return {
            "type": 0,
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "slot_labels": float(positive),
        }


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
