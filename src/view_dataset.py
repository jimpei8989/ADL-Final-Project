from argparse import ArgumentParser
from pathlib import Path
from datasets.dataset import DSTDataset

RESET = "\x1b[0m"
UTTERANCE_COLOR = "\x1b[1;38;5;87m"
STATE_COLOR = "\x1b[1;38;5;176m"


class Viewer:
    def __init__(self, data_dir):
        self.dataset = DSTDataset(data_dir)
        self.id_to_idx = {
            d["dialogue_id"].upper(): i for i, d in enumerate(self.dataset)
        }

    def print_frame(self, frame):
        print(
            STATE_COLOR,
            f"    {frame['service']:10s} - ",
            " ; ".join(
                f"{k}: {v[0]}" for k, v in frame["state"]["slot_values"].items()
            ),
            RESET,
            sep="",
        )

    def print_turn(self, turn):
        print(f"  {turn['speaker']:10s}:\t{UTTERANCE_COLOR}{turn['utterance']}{RESET}")

        if turn["speaker"] == "USER":
            print("  FRAMES:\t")
            if "frames" in turn:
                for frame in turn["frames"]:
                    self.print_frame(frame)

    def print(self, dialogue):
        print(
            "\n".join(
                [
                    "DIALOGUE_ID:\t" + dialogue["dialogue_id"],
                    "SERVICES:\t" + ", ".join(dialogue["services"]),
                ]
            )
        )

        for i, turn in enumerate(dialogue["turns"]):
            print(f"- TURN {i:2d}")
            self.print_turn(turn)

    def print_by_id(self, did):
        try:
            index = self.id_to_idx[did.upper()]
        except KeyError:
            print("No such dialogue")
        else:
            self.print(self.dataset[index])


def main(args):
    viewer = Viewer(args.data_dir)

    while True:
        did = input("Dialogue ID: ")
        print(did)
        viewer.print_by_id(did.strip())


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, default="dataset/data-0625/train")
    parser.add_argument("--schema_json", type=Path, default="dataset/data/schema.json")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
