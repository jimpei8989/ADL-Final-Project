import random

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForSpan(DSTDatasetForDST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index: int):
        dialogue, turn_idx = super().__getitem__(index)
        turn = dialogue["turns"][turn_idx]

        candidates = [
            (service, slot)
            for service, slot in self.get_positive_service_slot_names(dialogue["turns"][turn_idx])
            if not self.schema.service_by_name[service].slot_by_name[slot].is_categorical
        ]

        chosen_idx = int(random.random() * len(candidates))
        service_name, slot_name = candidates[chosen_idx]

        slot = self.schema.service_by_name[service_name].slot_by_name[slot_name]
        the_frame = next(filter(lambda f: f["service"] == service_name, turn["frames"]))
        the_slot = next(filter(lambda s: s["slot"] == slot_name, the_frame["slots"]))

        begin_str_idx, end_str_idx = the_slot["start"], the_slot["exclusive_end"]

        slot_tokens = self.tokenizer.tokenize(slot.description)

        utterance, begin_token_idx, end_token_idx = self.get_utterance_tokens(
            dialogue,
            turn_idx,
            max_length=self.max_seq_length - len(slot_tokens) - 3,
            begin_str_idx=begin_str_idx,
            end_str_idx=end_str_idx - 1,
        )

        input_ids = self.tokenizer(
            utterance, slot_tokens, is_split_into_words=True, padding="max_length"
        ).input_ids

        return {
            "type": 2,
            "input_ids": input_ids,
            "begin_token_idx": begin_token_idx + 1,
            "end_token_idx": end_token_idx + 1,
        }


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from datasets.schema import Schema

    schema = Schema.load_json(Path("../dataset/data/schema.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DSTDatasetForDSTForSpan(
        Path("../dataset/data-0614/train"),
        schema=schema,
        tokenizer=tokenizer,
        system_token=tokenizer.sep_token,
        user_token=tokenizer.sep_token,
    )

    print(len(dataset))
    print(dataset[0])
    print(dataset[42])
