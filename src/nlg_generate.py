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
            begin = model.generate(data_begin["input_ids"].to(device), num_beams=10)
            begin_sentences = tokenizer.batch_decode(begin, skip_special_tokens=True)

            end = model.generate(data_end["input_ids"].to(device), num_beams=10)
            end_sentences = tokenizer.batch_decode(end, skip_special_tokens=True)

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

    return result
