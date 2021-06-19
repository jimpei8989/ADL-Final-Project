from numpy import result_type
import torch
from torch.utils.data import DataLoader
from utils.tqdmm import tqdmm


def generate(model, dataset, batch_size, device):
    tokenizer = dataset.tokenizer
    loader_begin = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn_gen_begin,
    )
    loader_end = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn_gen_end,
    )
    model.to(device)
    with torch.no_grad():
        result = []
        for data_begin, data_end in tqdmm(
            zip(loader_begin, loader_end), total=len(loader_begin)
        ):
            begin = model.generate(
                data_begin["input_ids"].to(device), num_beams=10, max_length=128
            )
            begin_sentences = tokenizer.batch_decode(begin, skip_special_tokens=True)

            end = model.generate(
                data_end["input_ids"].to(device), num_beams=10, max_length=128
            )
            end_sentences = tokenizer.batch_decode(end, skip_special_tokens=True)

            if "gpt2" in model.config._name_or_path:
                begin_sentences = [s.rstrip("!") for s in begin_sentences]
                end_sentences = [s.rstrip("!") for s in end_sentences]

            result += [
                {
                    "dialogue_ids": idx,
                    "user": u,
                    "beginning": b.strip(),
                    "system": s,
                    "end": e.strip(),
                }
                for idx, u, s, b, e in zip(
                    data_begin["dialogue_ids"],
                    data_begin["str"],
                    data_end["str"],
                    begin_sentences,
                    end_sentences,
                )
            ]
            # return result

    return result


def generate_oneside(model, loader: DataLoader, tokenizer, device, which_side="end"):
    model.to(device)
    with torch.no_grad():
        result = []
        for data in tqdmm(loader):
            pred = model.generate(
                data["input_ids"].to(device), num_beams=10, max_length=128
            )
            pred_sentences = tokenizer.batch_decode(pred, skip_special_tokens=True)

            result += [
                {
                    "dialogue_ids": idx,
                    "user": u,
                    "beginning": p if which_side == "beginning" else "",
                    "system": s,
                    "end": p if which_side == "end" else "",
                }
                for idx, u, s, p in zip(
                    data["dialogue_ids"], data["user"], data["system"], pred_sentences
                )
            ]

        return result