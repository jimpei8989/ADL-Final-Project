from typing import Any, Dict, List

from datasets.dataset_dst import DSTDatasetForDST


class DSTDatasetForDSTForCategoricalPrediction(DSTDatasetForDST):
    def __init__(
        self,
        *args,
        dialogue_to_others: Dict[str, List[Any]],
        **kwargs,
    ):
        self.dialogue_to_others = dialogue_to_others

        super().__init__(*args, **kwargs)

    def expand(self, dialogue) -> List[Any]:
        if dialogue["dialogue_id"] not in self.dialogue_to_others:
            return []

        ret = []
        for turn_idx, service_name, slot_name in self.dialogue_to_others[dialogue["dialogue_id"]]:
            slot = self.schema.service_by_name[service_name].slot_by_name[slot_name]

            for answer in slot.possible_values:
                ret.append((turn_idx, service_name, slot_name, answer))

        return ret

    def check_data(self, dialogue, other):
        return True

    def form_data(self, dialogue, other) -> dict:
        turn_idx, service_name, slot_name, answer = other

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        ret = self._form_data(
            dialogue=dialogue,
            turns=dialogue["turns"][: turn_idx + 1],
            latter=f" {self.tokenizer.sep_token} ".join(
                [service.description, slot.description, answer]
            ),
            max_length=self.max_seq_length,
        )
        ret.update({"answer": answer, "_key": (dialogue["dialogue_id"], *other)})
        return ret


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer

    from datasets.schema import Schema

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    schema = Schema.load_json(Path("../dataset/data/schema.json"))

    dataset = DSTDatasetForDSTForCategoricalPrediction(
        Path("../dataset/data/test_seen"),
        schema=schema,
        tokenizer=tokenizer,
    )
    print(len(dataset))
    print(dataset[0])
