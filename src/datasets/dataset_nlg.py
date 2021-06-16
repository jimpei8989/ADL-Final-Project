from datasets.dataset import DSTDataset
from pathlib import Path
from typing import Iterable, Tuple
import random
import copy

import torch
from torch.nn.utils.rnn import pad_sequence


def pairwise(iterable: Iterable) -> Iterable[Tuple]:
    # s -> (s0,s1), (s2,s3), ..., (sn,sn+1), ...
    return zip(*[iter(iterable)] * 2)


def gen_input_sentence(data, domain, max_len, tokenizer):
    cur_data = data[-1] if type(data) is list else data

    ret = []
    if domain == "begin":
        ret_before_tok = [cur_data["user_utterance"]]
    elif domain == "end":
        ret_before_tok = [cur_data["user_utterance"], cur_data["system_utterance"]]

    for d in ret_before_tok:
        ret += tokenizer.encode(d)

    # Add previous history
    if type(data) is list:
        for i in range(-2, -len(data) - 1, -1):
            tmp = tokenizer.encode(data[i]["user_utterance"]) + tokenizer.encode(
                data[i]["system_utterance"]
            )
            if len(tmp) + len(ret) < max_len:
                ret = tmp + ret
            else:
                break

    return torch.LongTensor(ret)


class DSTDatasetForNLG(DSTDataset):
    def __init__(
        self,
        json_dir: Path,
        tokenizer=None,
        max_seq_length=128,
        mode="test",
        get_full_history=False,
    ) -> None:
        super().__init__(json_dir, tokenizer, max_seq_length)
        self.nlg_data = []
        self.history = []
        self.history_map = {}
        self.get_full_history = (get_full_history,)

        for d in self.data:
            for idx, (user, system) in enumerate(pairwise(d["turns"])):
                assert user["speaker"] == "USER"
                assert system["speaker"] == "SYSTEM"

                tmp = {
                    "dialogue_id": d["dialogue_id"],
                    "turns_id": [idx * 2, idx * 2 + 1],
                    "user_utterance": user["utterance"],
                    "system_utterance": system["utterance"],
                }

                self.history.append(tmp)

                if mode == "train":
                    if "beginning" not in system or "end" not in system:
                        continue
                    if system["beginning"] == [] and system["end"] == []:
                        continue
                    tmp["beginning"] = system["beginning"]  # chit-chat begining
                    tmp["end"] = system["end"]  # chit-chat end

                self.history_map[len(self.nlg_data)] = len(self.history)
                self.nlg_data.append(tmp)

    def __getitem__(self, index):
        if self.get_full_history:
            history_idx = self.history_map[index]
            return self.history[
                history_idx - self.nlg_data[index]["turns_id"][0] // 2 : history_idx + 1
            ]

        else:
            return self.nlg_data[index]

    def __len__(self):
        return len(self.nlg_data)

    def collate_fn_gen_begin(self, datas):
        ret = pad_sequence(
            [
                gen_input_sentence(
                    d,
                    domain="begin",
                    max_len=self.max_seq_length,
                    tokenizer=self.tokenizer,
                )
                for d in datas
            ],
            batch_first=True,
        )
        if type(datas) is list:
            datas = datas[0]
        return {
            "dialogue_ids": [f"{d['dialogue_id']}_{d['turns_id'][0]}" for d in datas],
            "str": [d["user_utterance"] for d in datas],
            "input_ids": ret,
        }

    def collate_fn_gen_end(self, datas):
        ret = pad_sequence(
            [
                gen_input_sentence(
                    d,
                    domain="end",
                    max_len=self.max_seq_length,
                    tokenizer=self.tokenizer,
                )
                for d in datas
            ],
            batch_first=True,
        )
        if type(datas) is list:
            datas = datas[0]
        return {
            "dialogue_ids": [f"{d['dialogue_id']}_{d['turns_id'][1]}" for d in datas],
            "str": [d["system_utterance"] for d in datas],
            "input_ids": ret,
        }

    def classify_collate_fn(self, samples):
        input_ids = []
        label = []
        for sample in samples:
            idx = self.nlg_data.index(sample)
            system_utterance = sample["system_utterance"]
            user_utterance = sample["user_utterance"]
            chitchat_map = {
                chat["candidate"]: [chat["label"], 0] for chat in sample["beginning"]
            }
            chitchat_map.update(
                {chat["candidate"]: [chat["label"], 1] for chat in sample["end"]}
            )

            chitchat = list(chitchat_map.keys())[
                random.randint(0, len(chitchat_map) - 1)
            ]
            sentence = (
                user_utterance
                + self.tokenizer.sep_token
                + "".join(
                    [chitchat, system_utterance][:: (-1) ** chitchat_map[chitchat][1]]
                )
                + self.tokenizer.sep_token
            )

            for dialog in self.history[: self.history_map[idx]]:
                if len(sentence) > 1024:
                    break
                sentence += dialog["system_utterance"]
                sentence += dialog["user_utterance"]

            input_ids.append(
                self.tokenizer.encode(
                    sentence, truncation=True, max_length=512, padding="max_length"
                )
            )
            label.append(0 if chitchat_map[chitchat][0] == "bad" else 1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    for paths in ["train", "dev", "test_seen", "test_unseen"]:
        ds = DSTDatasetForNLG(
            Path("../dataset/data/data") / paths,
            mode=("train"),
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        )

        dataloader = DataLoader(ds, batch_size=1, collate_fn=ds.classify_collate_fn)
        for d in dataloader:
            print(d["input_ids"])
            print(d["labels"])
            break

        print(paths, len(ds))
