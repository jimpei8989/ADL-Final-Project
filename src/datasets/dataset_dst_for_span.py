from typing import Any, List


from datasets.dataset_dst import DSTDatasetForDST
from datasets.utils import draw_from_list


class DSTDatasetForDSTForSpan(DSTDatasetForDST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                    (service, k, turn_idx, *v)
                    for k, v in slot_start_ends.items()
                    if k in frame["state"]["slot_values"]
                )

            ret.append((0, turn_idx, span_pairs))

        return ret

    def expand2(self, dialogue):
        ret = []
        turns = dialogue["turns"]
        begin_turn_idx = 0

        while True:
            if self.ensure_user_on_both_ends and turns[begin_turn_idx]["speaker"] == "SYSTEM":
                begin_turn_idx += 1

            cursor = begin_turn_idx
            cur_token_cnt = 0
            span_pairs = {}

            while cursor < len(turns):
                turn = turns[cursor]

                turn_token_len = len(self.form_turn(turn)[1])

                if cur_token_cnt + turn_token_len > self.former_max_len:
                    break
                else:
                    if turn["speaker"] == "USER":
                        for frame in turn["frames"]:
                            service = frame["service"]

                            for s in filter(
                                lambda s: "start" in s and "exclusive_end" in s, frame["slots"]
                            ):
                                span_pairs[(service, s["slot"])] = (
                                    cursor - begin_turn_idx,
                                    s["start"],
                                    s["exclusive_end"],
                                )

                    cursor += 1
                    cur_token_cnt += turn_token_len
            # Always -1 to make it right-close
            cursor -= 1

            assert cur_token_cnt <= 512 - 48

            if self.ensure_user_on_both_ends:
                if cursor % 2 == 1:
                    cursor -= 1

            span_pairs = [
                (*k, *v) for k, v in span_pairs.items() if v[0] <= cursor - begin_turn_idx
            ]

            ret.append((begin_turn_idx, cursor, span_pairs))

            if cursor >= len(dialogue["turns"]) - 2:
                break
            else:
                begin_turn_idx = cursor - self.overlap_turns

        return ret

    def check_data(self, dialogue, other):
        assert len(other[2]) > 0
        assert all(s[0] in dialogue["services"] for s in other[2])

    def form_data(self, dialogue, other) -> dict:
        # span_pairs: list of (service, slot, relative turn_idx, start, end)
        begin_turn_idx, end_turn_idx, span_pairs = other

        service_name, slot_name, relative_turn_idx, start, end = draw_from_list(span_pairs)

        service = self.schema.service_by_name[service_name]
        slot = service.slot_by_name[slot_name]

        ret = {"type": 2}
        ret.update(
            self._form_data(
                dialogue=dialogue,
                turns=dialogue["turns"][begin_turn_idx : end_turn_idx + 1],
                latter=f" {self.tokenizer.sep_token} ".join(
                    [service.description, slot.description]
                ),
                max_length=self.max_seq_length,
                relative_turn_idx=relative_turn_idx,
                begin_str_idx=start,
                end_str_idx=end,
            )
        )
        return ret


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from datasets.schema import Schema
    from utils.tqdmm import tqdmm

    schema = Schema.load_json(Path("../dataset/data/schema.json"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DSTDatasetForDSTForSpan(
        Path("../dataset/data-0614/train"),
        schema=schema,
        tokenizer=tokenizer,
        system_token=tokenizer.sep_token,
        user_token=tokenizer.sep_token,
        strategy="segment",
    )

    print(len(dataset))

    for d in tqdmm(dataset, desc="Asserting", leave=False):
        assert not (d["begin_labels"] is None or d["end_labels"] is None)

        input_ids = d["input_ids"]
        utterance = d["utterance"]

        begin_token = tokenizer.convert_ids_to_tokens(input_ids[d["begin_labels"]].item())
        end_token = tokenizer.convert_ids_to_tokens(input_ids[d["end_labels"]].item())

        # Do not use starts with and endswith, since the dataset has wrong labels
        assert (
            utterance[d["begin_str_idx"]].lower() in begin_token
            and utterance[d["end_str_idx"] - 1].lower() in end_token
        )
