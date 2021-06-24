from collections import Counter
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
from nltk import tokenize
import itertools
from typing import List, Dict
from pattern.en import lemma


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--keys", nargs="+")
    args = parser.parse_args()

    return args


def count(
    data, keys: List[str], most: bool = False, num: int = None, threshold: int = 1
) -> Dict[str, int]:
    try:
        lemma("hi")
    except Exception as e:
        pass

    result = Counter(
        itertools.chain.from_iterable(
            [
                [
                    lemma(word)
                    for word in tokenize.word_tokenize(
                        turn[key].replace(". ", ".").replace(".", ". ")
                    )
                ]
                for dialogue in data.values()
                for turn in dialogue.values()
                for key in keys
            ]
        )
    )

    print(" and ".join(keys))
    print(f"there are {len(result)} kinds of words")
    print(f"there are {sum(result.values())} words")
    print(
        f"there are {sum([v <= threshold for v in result.values()])} words appear less equal {threshold} time"
    )

    if num is not None:
        return dict(result.most_common()[:num] if most else result.most_common()[-num:])
    if threshold is not None:
        return {k: v for k, v in result.items() if v <= threshold}


if __name__ == "__main__":
    args = get_args()
    data = json.loads(args.file.read_bytes())
    result = count(data, args.keys)
    print(result)
