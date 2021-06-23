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
        which_side="end",
    ) -> None:
        super().__init__(
            json_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            mode=mode,
            get_full_history=get_full_history,
        )
        self.mode = mode
        self.which_side = which_side
        if mode == "train":
            self.good_end_nlg = []
            for d in self.nlg_data:
                for bged in d[which_side]:
                    if bged["label"] == "good":
                        self.good_end_nlg.append(
                            {
                                "dialogue_id": d["dialogue_id"],
                                "turns_id": d["turns_id"],
                                "user_utterance": d["user_utterance"],
                                "system_utterance": d["system_utterance"],
                                which_side: bged["candidate"],
                            }
                        )
                        # break // only use at most one good end chit-chat

    def __getitem__(self, index):
        if self.mode == "train":
            inp = "system_utterance"
            # inp = (
            #     "user_utterance"
            #     if self.which_side == "beginning"
            #     else "system_utterance"
            # )
            d = self.good_end_nlg[index]
            return {
                "input_ids": self.tokenizer.encode(d[inp]),
                "labels": self.tokenizer.encode(d[self.which_side]),
            }
        else:
            return self.nlg_data[index]

    def __len__(self):
        if self.mode == "train":
            return len(self.good_end_nlg)
        else:
            return len(self.nlg_data)

    def collate_fn(self, datas):
        # inp = "user_utterance" if self.which_side == "beginning" else "system_utterance"
        inp = "system_utterance"
        return {
            "dialogue_ids": [f"{d['dialogue_id']}_{d['turns_id'][0]}" for d in datas],
            "user": [d["user_utterance"] for d in datas],
            "system": [d["system_utterance"] for d in datas],
            "input_ids": pad_sequence(
                [torch.LongTensor(self.tokenizer.encode(d[inp])) for d in datas],
                batch_first=True,
            ),
        }