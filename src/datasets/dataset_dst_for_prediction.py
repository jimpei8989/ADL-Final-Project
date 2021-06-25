from typing import Any, List

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForPrediction(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        last_user_turn_only: bool = False,
        **kwargs,
    ):
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
            for service in dialogue["services"]:
                for slot in self.schema.service_by_name[service].slots:
                    ret.append((turn_idx, service, slot.name))

        return ret

    def check_data(self, dialogue, other):
        return True

    def form_data(self, dialogue, other) -> dict:
        turn_idx, service_name, slot_name = other

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        return self._form_data(
            dialogue=dialogue,
            turns=dialogue["turns"][: turn_idx + 1],
            latter=f" {self.tokenizer.sep_token} ".join([service.description, slot.description]),
            max_length=self.max_seq_length,
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
    print(dataset[0])
