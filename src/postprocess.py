import json
from argparse import ArgumentParser
from pathlib import Path
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", "-i", type=Path)
    parser.add_argument("--output_file", "-o", type=Path)

    args = parser.parse_args()
    return args


def remove_banned(turn):
    banned_words = [
        "Yes",
        "No",
        "n't",
        "?",
        "not sure",
        "but",
        "no idea",
        "Oh no",
        "sorry",
    ]
    for k in ["beginning", "end"]:
        turn[k] = [s for s in turn[k] if all(w not in s for w in banned_words)]

    return turn


def sep_sentence(turn):
    turn["beginning"] = tokenize.sent_tokenize(turn["beginning"])
    turn["end"] = tokenize.sent_tokenize(turn["end"])

    turn = remove_banned(turn)
    return turn


def sentence_similarity(model, turn):
    system_vec = model.encode(turn["system"]).reshape(768, 1)  # (768, )
    for k in ["beginning", "end"]:
        if turn[k] == []:
            continue
        chit_vecs = model.encode(turn[k])
        cos_sim = chit_vecs @ system_vec
        for s, cos in zip(turn[k], cos_sim):
            if cos > 0.5:
                turn[k].remove(s)

    return turn


def filtering(res):
    model = SentenceTransformer("LaBSE")
    ret = []
    for r in tqdm(res):
        sep = sep_sentence(r)
        sentence_similarity(model, sep)
        ret.append(sep)

    return [
        {k: " ".join(v) if type(v) is list else v for k, v in r.items()} for r in ret
    ]


if __name__ == "__main__":
    args = parse_args()
    res = json.load(open(args.input_file, "r"))

    ret = filtering(res)
    json.dump(ret, open(args.output_file, "w"))
