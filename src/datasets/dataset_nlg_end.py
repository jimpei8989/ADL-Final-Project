from datasets.dataset_nlg import DSTDatasetForNLG
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


class DSTDatasetForNLGEnd(DSTDatasetForNLG):
    def __init__(
        self,
        json_dir: Path,
        tokenizer=None,
        max_seq_length=128,
        mode="test",
        get_full_history=False,
    ) -> None:
        super().__init__(
            json_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mode=mode,
            get_full_history=get_full_history,
        )
        self.mode = mode
        if mode == "train":
            self.good_end_nlg = []
            for d in self.nlg_data:
                for ed in d["end"]:
                    if ed["label"] == "good":
                        self.good_end_nlg.append(
                            {
                                "dialogue_id": d["dialogue_id"],
                                "turns_id": d["turns_id"],
                                "user_utterance": d["user_utterance"],
                                "system_utterance": d["system_utterance"],
                                "end": ed["candidate"],
                            }
                        )
                        # break // only use at most one good end chit-chat

    def __getitem__(self, index):
        if self.mode == "train":
            d = self.good_end_nlg[index]
            return {
                "input_ids": self.tokenizer.encode(d["system_utterance"]),
                "labels": self.tokenizer.encode(d["end"]),
            }
        else:
            return self.nlg_data[index]

    def __len__(self):
        if self.mode == "train":
            return len(self.good_end_nlg)
        else:
            return len(self.nlg_data)

    def collate_fn(self, datas):
        return {
            "dialogue_ids": [f"{d['dialogue_id']}_{d['turns_id'][0]}" for d in datas],
            "user": [d["user_utterance"] for d in datas],
            "system": [d["system_utterance"] for d in datas],
            "input_ids": pad_sequence(
                [
                    torch.LongTensor(self.tokenizer.encode(d["system_utterance"]))
                    for d in datas
                ],
                batch_first=True,
            ),
        }