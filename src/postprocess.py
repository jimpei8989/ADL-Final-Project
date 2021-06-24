import json
from argparse import ArgumentParser
from os import pardir
from pathlib import Path
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from threading import Thread
from count import count
from joblib import Parallel, delayed
from pattern.en import lemma


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--begin", "-b", type=Path)
    parser.add_argument("--end", "-e", type=Path)
    parser.add_argument("--banned", type=Path, default="banned_words.json")
    parser.add_argument("--output_file", "-o", type=Path)
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()
    return args


def sep_sentence(turn, banned_words):
    turn["beginning"] = tokenize.sent_tokenize(turn["beginning"])
    turn["end"] = tokenize.sent_tokenize(turn["end"])

    for k in ["beginning", "end"]:
        turn[k] = [s for s in turn[k] if all(w not in s for w in banned_words)]

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
    target_vec = model.encode(turn[versus]).reshape(-1, 1)  # (768, )
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


def filtering(res, threshold=0.1, versus=[], banned_words=[]):
    model = SentenceTransformer("LaBSE")
    ret = []
    for r in tqdm(res):
        sep = sep_sentence(r, banned_words=banned_words)
        sep = system_similarity(model, sep)
        for v in versus:
            sep = remove_unrelated(model, sep, threshold, v)
        ret.append(sep)

    return ret


def del_duplicate(res, domain="end"):
    model = SentenceTransformer("LaBSE")
    past_vec, cur_dialogue_id = np.array([]), "-1"
    ret = []
    for r in tqdm(res):
        dialogue_id = "_".join(r["dialogue_ids"].split("_")[:-1])
        if dialogue_id != cur_dialogue_id:
            past_vec = np.array([])
            cur_dialogue_id = dialogue_id
        for s in r[domain]:
            s_vec = model.encode(s).reshape(-1, 1)
            if past_vec.shape[0] != 0:
                if (s_vec.T @ past_vec).max() > 0.7:
                    r[domain].remove(s)
                    # print(s_vec.T @ past_vec, s_vec.T.shape, past_vec.shape, s)
                else:
                    past_vec = np.hstack([past_vec, s_vec])
            else:
                past_vec = s_vec

        ret.append(r)
    return ret


def postprocessing_single_side(ori, threshold, versus, domain, banned_words):
    res = filtering(ori, threshold=threshold, versus=versus, banned_words=banned_words)
    res = del_duplicate(res, domain=domain)
    result_global[domain] = res

    return res


def remove_single_appear(data):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    count_dictionary = count(
        data,
        ["user", "mod", "start", "end"],
    )

    for ids, dialogue in data.items():
        for turn_ids, turn in dialogue.items():
            for k in ["start", "end"]:
                sentences = tokenize.sent_tokenize(turn[k])
                for sentence in sentences:
                    if any(
                        (
                            (lemma(w) in count_dictionary)
                            for w in tokenize.word_tokenize(
                                sentence.replace(". ", ".").replace(".", ". ")
                            )
                        )
                    ):
                        data[ids][turn_ids][k] = (
                            data[ids][turn_ids][k].replace(sentence, "").strip()
                        )

    return data


result_global = {"beginning": {}, "end": {}}

if __name__ == "__main__":
    args = parse_args()
    banned_words = json.loads(args.banned.read_bytes())
    print(f"Banned List: {banned_words}")
    begin_ori = json.loads(args.begin.read_bytes())
    end_ori = json.loads(args.end.read_bytes())

    t_begin = Thread(
        target=postprocessing_single_side,
        args=(begin_ori, 0.1, ["user"], "beginning", banned_words),
    )
    t_end = Thread(
        target=postprocessing_single_side,
        args=(end_ori, 0.05, ["system"], "end", banned_words),
    )

    t_begin.start()
    t_end.start()
    t_begin.join()
    t_end.join()
    res_begin, res_end = result_global["beginning"], result_global["end"]

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
    ret = remove_single_appear(ret)

    json.dump(ret, open(args.output_file, "w"), indent=2)
