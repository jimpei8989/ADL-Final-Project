from bisect import bisect_right
from typing import Any, List, Optional, Tuple

from datasets.dataset import DSTDataset
from datasets.schema import Schema, Slot
from utils.logger import logger
from utils.tqdmm import tqdmm


class DSTDatasetForDST(DSTDataset):
    def __init__(
        self,
        *args,
        schema: Schema = None,
        user_token: Optional[str] = None,
        system_token: Optional[str] = None,
        test_mode: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert schema is not None and self.tokenizer is not None

        self.schema = schema
        self.user_token = user_token
        self.system_token = system_token
        self.test_mode = test_mode

        logger.info(f"Successfully loaded {len(self.data)} dialogues...")
        self.data = sorted(self.data, key=lambda d: d["dialogue_id"])
        self.dialogue_by_id = {d["dialogue_id"]: d for d in self.data}

        self.expanded = {}  # dialogue_id -> Any
        self.prefix_sum = [0]

        self.before_expand()

        for dialogue in self.data:
            did = dialogue["dialogue_id"]
            self.expanded[did] = self.expand(dialogue)
            self.prefix_sum.append(self.prefix_sum[-1] + len(self.expanded[did]))

        logger.info(
            f"Finished preprocessing dialogues, there're {self.prefix_sum[-1]} samples in total..."
        )

        self.valid_indices = self.filter_bad_data()

        logger.info(
            f"Finished filtering bad samples, there're {len(self.valid_indices)} samples left..."
        )

    # child classes should NOT override len and getitem if possible
    def __getitem__(self, index: int):
        index = self.valid_indices[index]
        return self.form_data(*self.get_dialogue_and_other(index))

    def __len__(self):
        return len(self.valid_indices) if not self.test_mode else 10

    # About the expanding
    def before_expand(self) -> None:
        pass

    def expand(self, dialogue) -> List[Any]:
        return [None]

    def get_dialogue_and_other(self, index):
        # find an i s.t. a[i] <= index < a[i + 1]
        i = bisect_right(self.prefix_sum, index) - 1
        offset = index - self.prefix_sum[i]

        dialogue = self.data[i]
        other = self.expanded[dialogue["dialogue_id"]][offset]

        return (dialogue, other)

    # Filter bad data away
    def filter_bad_data(self) -> List[int]:
        ret = []
        for i in tqdmm(range(self.prefix_sum[-1]), desc="Filter bad data"):
            try:
                self.check_data(*self.get_dialogue_and_other(i))
            except AssertionError:
                logger.debug(f"Sample {i} fails on sanity check")
            else:
                ret.append(i)

        return ret

    def check_data(self, dialogue, other):
        return True

    # what getitem will return
    def form_data(self, dialogue, other) -> dict:
        return self._form_data(
            dialogue,
            dialogue["turns"][:-1],  # The last one is the system's one
            "",
            max_length=self.max_seq_length,
        )

    # Form sample
    def _form_data(
        self,
        dialogue,
        turns: list,
        latter: str,
        max_length: Optional[int] = None,
        begin_str_idx: Optional[int] = None,
        end_str_idx: Optional[int] = None,
    ) -> dict:
        # [CLS] utterance [SEP] latter [SEP]
        latter_token_len = len(self.tokenizer.tokenize(latter))

        utterances = self.form_utterances(turns, max_length=max_length - latter_token_len - 3)

        if begin_str_idx is not None and end_str_idx is not None:
            offset = sum(len(u) for u in utterances[::-1])
            begin_str_idx += offset
            end_str_idx += offset

        utterance = " ".join(utterances)
        encoded = self.tokenizer([utterance], [latter], padding="max_length", return_tensors="pt")

        ret = {
            "utterance": utterance,
            "input_ids": encoded.input_ids.squeeze(0),
        }

        if begin_str_idx is not None and end_str_idx is not None:
            ret.update(
                {
                    "begin_labels": encoded.char_to_token(0, begin_str_idx),
                    "end_labels": encoded.char_to_token(0, end_str_idx),
                }
            )

        return ret

    def form_utterances(
        self,
        turns: List,
        max_length: Optional[int] = None,
    ) -> List[str]:
        cur_len, utterances = 0, []

        for turn in turns[::-1]:
            utterance = turn["utterance"]

            special_token = self.user_token if turn["speaker"] == "USER" else self.system_token
            if special_token is not None:
                utterance = special_token + " " + utterance

            utterance_tokens = self.tokenizer.tokenize(utterance)

            if cur_len + len(utterance_tokens) > max_length:
                break
            else:
                utterances.append(utterance)
                cur_len += len(utterance_tokens)

        return utterances[::-1]

    # Too lazy to refactor them QAQ, should I
    def get_positive_service_slot_names(self, turn, filter_slots=False) -> List[Tuple[str, str]]:
        ret = []

        for frame in turn["frames"]:
            # NOTE: some slots has no "start" and "exclusive_end", possibly due to its value
            # was referenced from other slots
            in_slots = {s["slot"] for s in frame["slots"] if "start" in s and "exclusive_end" in s}
            for slot in frame["state"]["slot_values"]:
                if not filter_slots or slot in in_slots:
                    ret.append((frame["service"], slot))

        return ret

    def get_all_service_slot_names(self, dialogue) -> List[Tuple[str, str]]:
        return [
            (service, slot)
            for service in dialogue["services"]
            for slot in self.schema.service_by_name[service].slot_by_name
        ]

    def get_positive_slots(self, turn) -> List[Slot]:
        return [
            self.schema.service_by_name[frame_name].slot_by_name[slot_name]
            for frame_name, slot_name in self.get_positive_service_slot_names(turn)
        ]

    def get_negative_slots(self, dialogue, turn):
        positive_names = set(self.get_positive_service_slot_names(turn))
        return [n for n in self.get_all_service_slot_names(dialogue) if n not in positive_names]


if __name__ == "__main__":
    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    schema = Path("../dataset/data/schema.json")

    dataset = DSTDatasetForDST(
        Path("../dataset/data-0610/new-train"),
        schema=schema,
        tokenizer=tokenizer,
    )

    print(len(dataset))
    print(dataset[0])
