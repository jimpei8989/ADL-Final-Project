import json
from argparse import ArgumentParser
from pathlib import Path
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict, Counter


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--begin", "-b", type=Path)
    parser.add_argument("--end", "-e", type=Path)
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


def system_similarity(
    model, turn
):  # If chit-chat is too similar as system, then remove
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


def remove_unrelated(model, turn, threshold=0, versus="system"):
    target_vec = model.encode(turn[versus]).reshape(768, 1)  # (768, )
    for k in ["beginning", "end"]:
        if turn[k] == []:
            continue
        chit_vecs = model.encode(turn[k])
        cos_sim = chit_vecs @ target_vec
        for s, cos in zip(turn[k], cos_sim):
            if cos < threshold:
                # print(versus, turn[versus], "...", s, cos)
                turn[k].remove(s)

    return turn


def filtering(res, threshold=0.1, versus="user"):
    model = SentenceTransformer("LaBSE")
    ret = []
    for r in tqdm(res):
        sep = sep_sentence(r)
        sep = system_similarity(model, sep)
        sep = remove_unrelated(model, sep, threshold, versus)
        ret.append(sep)

    return ret


def del_duplicate(res, domain="end"):
    past_rec, cur_dialogue_id = set(), "-1"
    ret = []
    for r in res:
        dialogue_id = "_".join(r["dialogue_ids"].split("_")[:-1])
        if dialogue_id != cur_dialogue_id:
            past_rec = set()
            cur_dialogue_id = dialogue_id
        for s in r[domain]:
            if s in past_rec:
                r[domain].remove(s)
            past_rec.add(s)

        ret.append(r)
    return ret


if __name__ == "__main__":
    args = parse_args()
    res_end = filtering(json.load(open(args.end, "r")), threshold=0.05, versus="system")
    res_end = del_duplicate(res_end, domain="end")
    res_begin = filtering(
        json.load(open(args.begin, "r")), threshold=0.1, versus="user"
    )

    ret = defaultdict(dict)
    for b, e in zip(res_begin, res_end):
        for key in ["dialogue_ids", "user", "system"]:
            assert b[key] == e[key]
        tmp = {
            "user": b["user"],  # need to be removed
            "start": " ".join(b["beginning"]),
            "mod": b["system"],  # need to be empty
            "end": " ".join(e["end"]),
        }
        dialogue_id, idx = (
            "_".join(b["dialogue_ids"].split("_")[:-1]),
            int(b["dialogue_ids"].split("_")[-1]) + 1,
        )
        ret[dialogue_id][idx] = tmp
    json.dump(ret, open(args.output_file, "w"), indent=2)
