import json
from pathlib import Path
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    data = json.loads(args.input.read_bytes())
    for ids, dialogue in data.items():
        for turn_ids in dialogue:
            del data[ids][turn_ids]["user"]
            data[ids][turn_ids]["mod"] = ""

    json.dump(data, args.output.open("w"), indent=2)