import random

import torch

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForCategorical(DSTDatasetForDST):
    def __init__(self, *args, negative_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.negative_ratio = negative_ratio

    def __len__(self):
        return super().__len__()

    def check_item(self, index):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        candidates = [
            (service, slot)
            for service, slot in self.get_positive_service_slot_names(turn)
            if self.schema.service_by_name[service].slot_by_name[slot].is_categorical
        ]

        assert len(candidates) > 0

    def __getitem__(self, index: int):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        candidates = [
            (service, slot)
            for service, slot in self.get_positive_service_slot_names(turn)
            if self.schema.service_by_name[service].slot_by_name[slot].is_categorical
        ]

        chosen_idx = int(random.random() * len(candidates))
        service_name, slot_name = candidates[chosen_idx]

        slot = self.schema.service_by_name[service_name].slot_by_name[slot_name]

        the_frame = next(filter(lambda f: f["service"] == service_name, turn["frames"]))
        correct_answer = the_frame["state"]["slot_values"][slot_name][0]

        positive = int(random.random() * (1 + self.negative_ratio) < 1.0)
        if positive:
            value = correct_answer
        else:
            incorrect_answers = [v for v in slot.possible_values if v != correct_answer]
            if len(incorrect_answers) == 0:
                print(slot.possible_values)
            value = incorrect_answers[int(random.random() * len(incorrect_answers))]

        slot_tokens = (
            self.tokenizer.tokenize(slot.description)
            + [self.tokenizer.sep_token]
            + self.tokenizer.tokenize(value)
        )

        utterance = self.get_utterance_tokens(
            dialogue, turn_idx, max_length=self.max_seq_length - len(slot_tokens) - 3
        )

        input_ids = self.tokenizer(
            utterance, slot_tokens, is_split_into_words=True, padding="max_length"
        ).input_ids

        return {
            "type": 1,
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "label": positive,
        }


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
