from bisect import bisect_right
from typing import Any, List, Optional

from datasets.dataset import DSTDataset
from datasets.schema import Schema
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
        strategy: str = "turn",
        # Strategy 1
        last_user_turn_only: bool = False,
        # Strategy 2
        reserved_for_latter: int = 48,
        overlap_turns: int = 4,
        ensure_user_on_both_ends: bool = True,
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

        self.strategy = strategy

        # Strategy 1
        self.last_user_turn_only = last_user_turn_only

        # Strategy 2
        self.former_max_len = self.max_seq_length - reserved_for_latter
        self.overlap_turns = overlap_turns
        self.ensure_user_on_both_ends = ensure_user_on_both_ends

        self.expanded = {}  # dialogue_id -> Any
        self.prefix_sum = [0]

        self.before_expand()

        for dialogue in tqdmm(self.data, desc="Expanding dataset"):
            did = dialogue["dialogue_id"]
            self.expanded[did] = self.expand(dialogue)
            self.prefix_sum.append(self.prefix_sum[-1] + len(self.expanded[did]))

        self.after_expand()

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
        return len(self.valid_indices) if not self.test_mode else min(len(self.valid_indices), 64)

    # About the expanding
    def before_expand(self) -> None:
        pass

    def after_expand(self) -> None:
        pass

    def expand(self, dialogue) -> List[Any]:
        if self.strategy == "turn":
            return self.expand1(dialogue)
        elif self.strategy == "segment":
            return self.expand2(dialogue)
        else:
            raise ValueError
        return [None]

    def expand1(self, dialogue) -> List[Any]:
        return [None]

    def expand2(self, dialogue) -> List[Any]:
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
        relative_turn_idx: int = -1,
        begin_str_idx: Optional[int] = None,
        end_str_idx: Optional[int] = None,
    ) -> dict:
        # [CLS] utterance [SEP] latter [SEP]
        latter_token_len = len(self.tokenizer.tokenize(latter))

        # ???
        # if self.stategy == "turn":
        #     max_length -= 10

        utterances = self.form_utterances(turns, max_length=max_length - latter_token_len - 3)

        if begin_str_idx is not None and end_str_idx is not None:
            offset = sum(len(u) + 1 for u in utterances[:relative_turn_idx])
            if self.user_token:
                offset += len(self.user_token) + 1

            begin_str_idx += offset
            end_str_idx += offset

        utterance = " ".join(utterances)
        encoded = self.tokenizer(
            [utterance],
            [latter],
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        ret = {
            "utterance": utterance,
            "latter": latter,
            "input_ids": encoded.input_ids.squeeze(0),
            "_encoded": encoded,
        }

        if begin_str_idx is not None and end_str_idx is not None:
            ret.update(
                {
                    "begin_str_idx": begin_str_idx,
                    "end_str_idx": end_str_idx,
                    "begin_labels": encoded.char_to_token(begin_str_idx),
                    "end_labels": encoded.char_to_token(end_str_idx - 1),
                }
            )

        return ret

    def form_turn(self, turn):
        utterance = turn["utterance"]

        special_token = self.user_token if turn["speaker"] == "USER" else self.system_token
        if special_token is not None:
            utterance = special_token + " " + utterance

        utterance_tokens = self.tokenizer.tokenize(utterance)

        return utterance, utterance_tokens

    def form_utterances(
        self,
        turns: List,
        max_length: Optional[int] = None,
    ) -> List[str]:
        cur_len, utterances = 0, []

        for turn in turns[::-1]:
            utterance, utterance_tokens = self.form_turn(turn)

            if cur_len + len(utterance_tokens) > max_length:
                break
            else:
                utterances.append(utterance)
                cur_len += len(utterance_tokens)

        return utterances[::-1]


if __name__ == "__main__":
    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    schema = Path("../dataset/data/schema.json")

    dataset = DSTDatasetForDST(
        Path("../dataset/data-0614/train"),
        schema=schema,
        tokenizer=tokenizer,
    )

    print(len(dataset))
    print(dataset[0])
