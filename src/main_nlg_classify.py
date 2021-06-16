from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.logger import logger
from pathlib import Path
from utils.utils import set_seed
from utils.tqdmm import tqdmm
from model.NLGClassifier import NLGClassifier
from datasets.dataset_nlg import DSTDatasetForNLG

TRAIN = "train"
DEV = "val"
SPLITS = [TRAIN, DEV]


def iter_loop(dataloader, model, optimizer, accumulate_steps, device, mode):
    total_correct = 0
    total_loss = 0
    cnt = 0

    if mode == TRAIN:
        model.train()
    elif mode == DEV:
        model.eval()

    step = 0
    with torch.set_grad_enabled(mode == TRAIN):
        with tqdmm(dataloader, unit="batch") as tepoch:
            optimizer.zero_grad()
            for data in tepoch:
                tepoch.set_description(f"[{mode:>5}]")

                step += 1
                token = data["input_ids"].to(device)
                label = data["labels"].to(device)

                outputs = model(token, label)

                loss = outputs.loss
                correct = (
                    ((outputs.logits > 0.5).type(torch.float) == label)
                    .type(torch.float)
                    .mean()
                    .item()
                )
                total_correct += correct
                total_loss += loss

                if mode == TRAIN:
                    (loss / accumulate_steps).backward()
                    if step % accumulate_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                tepoch.set_postfix(loss=f"{loss.item():>.4f}", Acc=f"{correct:>.4f}")

            total_correct /= len(dataloader)
            total_loss /= len(dataloader)
            logger.info(
                f"[{mode:>5}]"
                + f" Acc: {total_correct:>.2f},"
                + f" loss: {total_loss:>.7f},"
            )

    return total_correct, total_loss


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    device = torch.device(f"cuda:{args.gpu}")
    datasets = {
        split: DSTDatasetForNLG(
            path,
            mode="train",
            tokenizer=tokenizer,
        )
        for split, path in zip(SPLITS, [args.train_data, args.val_data])
    }

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.classify_collate_fn,
        )
        for split, dataset in datasets.items()
    }

    model = NLGClassifier(model_name=args.backbone).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-7, patience=5
    )

    max_acc, min_loss = 0, 100
    early_stop = 0

    backbone = (
        args.backbone if "/" not in args.backbone else args.backbone.split("/")[-1]
    )
    ckpt_dir = args.ckpt_dir / backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch + 1}")
        iter_loop(dataloaders[TRAIN], model, optimizer, args.accumulate_steps, device, TRAIN)
        acc, loss = iter_loop(dataloaders[DEV], model, optimizer, args.accumulate_steps, device, DEV)

        scheduler.step(loss)

        if acc > max_acc:
            max_acc = acc
            torch.save(
                model.state_dict(),
                ckpt_dir / "model_best.pt",
            )
            print("model is better than before, save model to model_best.pt")

        if loss > min_loss:
            early_stop += 1
        else:
            early_stop = 0
            min_loss = loss

        if early_stop == 10:
            print("Early stop...")
            break

    print(f"Done! Best model Acc: {(max_acc):>.4f}")
    torch.save(model.state_dict(), ckpt_dir / "model.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # file PATH
    parser.add_argument(
        "--train_data", type=Path, default="./dataset/data-0610/new-train"
    )
    parser.add_argument("--val_data", type=Path, default="./dataset/data-0610/new-dev")
    parser.add_argument("--test_data", type=Path, default="./dataset/data/test_seen")
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

    # train
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--backbone", type=str, default="./models/convbert")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--accumulate_steps", type=int, default=8)

    # Misc
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=24667429)
    args = parser.parse_args()

    logger.info(args)
    set_seed(args.seed)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
