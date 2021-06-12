from argparse import ArgumentParser, Namespace
from utils.logger import logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    return parser.parse_args()


def main(args):
    logger.info(args)


if __name__ == "__main__":
    main(parse_args())
