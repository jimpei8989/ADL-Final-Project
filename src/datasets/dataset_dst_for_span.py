from typing import Any, List


from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForSpan(DSTDatasetForDST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def expand(self, dialogue) -> List[Any]:
        ret = []
        for turn_idx in range(0, len(dialogue["turns"]), 2):
            turn = dialogue["turns"][turn_idx]

            assert turn["speaker"] == "USER"

            span_pairs = []

            for frame in turn["frames"]:
                service = frame["service"]

                # Currently, only new spans added in this turn will be added...
                slot_start_ends = {
                    s["slot"]: (s["start"], s["exclusive_end"])
                    for s in frame["slots"]
                    if "start" in s and "exclusive_end" in s
                }

                span_pairs.extend(
                    (service, k, *v)
                    for k, v in slot_start_ends.items()
                    if k in frame["state"]["slot_values"]
                )

            ret.append((turn_idx, span_pairs))
        return ret

    def check_data(self, dialogue, other):
        assert len(other[1]) > 0

    def form_data(self, dialogue, other) -> dict:
        turn_idx, span_pairs = other

        service, slot_name, start, end = draw_from_list(span_pairs)

        slot = self.schema.service_by_name[service].slot_by_name[slot_name]

        ret = {"type": 2}
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][: turn_idx + 1],
                latter=slot.description,
                max_length=self.max_seq_length,
                begin_str_idx=start,
                end_str_idx=end,
            )
        )
        return ret


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
