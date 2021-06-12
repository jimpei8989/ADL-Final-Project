from argparse import ArgumentParser, Namespace
from utils.logger import logger
from pathlib import Path
from utils.utils import set_seed


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # file PATH
    parser.add_argument("--train_data", default="./dataset/data/train")
    parser.add_argument("--val_data", default="./dataset/data/dev")
    parser.add_argument("--test_data", default="./dataset/data/test_seen")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the mathing model file.",
        default="./ckpt/default",
    )
    parser.add_argument("--opt_file", default="default.json")

    # mode
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=8)

    # Misc
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=24667429)
    args = parser.parse_args()

    logger.info(args)
    set_seed(args.seed)
    return parser.parse_args()


def main(args):
    if args.train:
        pass

    if args.predict:
        pass


if __name__ == "__main__":
    main(parse_args())
