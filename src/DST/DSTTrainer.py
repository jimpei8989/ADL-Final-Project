from pathlib import Path
from typing import Callable, Dict, Optional, List
from collections import defaultdict
from dataclasses import asdict

from torch.utils.data import Dataset, DataLoader
import torch

from transformers import Trainer
from transformers import PreTrainedTokenizerBase

from datasets.dst_utils import load_dst_dataloader
from utils.logger import logger
from utils.tqdmm import tqdmm


class DSTTrainer(Trainer):
    def __init__(
        self,
        train_data_dir: Path,
        eval_data_dir: Path,
        schema_json: Path,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader_cls: Callable[[Dataset], DataLoader],
        eval_dataloader_cls: Callable[[Dataset], DataLoader],
        dataset_kwargs: Optional[Dict] = None,
        for_slot_kwargs: Optional[Dict] = None,
        for_categorical_kwargs: Optional[Dict] = None,
        for_span_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.train_dataloader = load_dst_dataloader(
            data_dir=train_data_dir,
            schema_json=schema_json,
            tokenizer=tokenizer,
            dataloader_cls=train_dataloader_cls,
            dataset_kwargs=dataset_kwargs,
            for_slot_kwargs=for_slot_kwargs,
            for_categorical_kwargs=for_categorical_kwargs,
            for_span_kwargs=for_span_kwargs,
        )
        self.eval_dataloader = load_dst_dataloader(
            data_dir=eval_data_dir,
            schema_json=schema_json,
            tokenizer=tokenizer,
            dataloader_cls=eval_dataloader_cls,
            dataset_kwargs=dataset_kwargs,
            for_slot_kwargs=for_slot_kwargs,
            for_categorical_kwargs=for_categorical_kwargs,
            for_span_kwargs=for_span_kwargs,
        )

        kwargs["tokenizer"] = tokenizer
        super().__init__(**kwargs)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        return self.eval_dataloader

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset=eval_dataset)

        ret = {label_type: [0, 0] for label_type in self.label_names}
        total_loss = 0

        for inputs in tqdmm(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            outputs = self.model(inputs["input_ids"])
            total_loss += outputs["loss"]

            for label_type in self.label_names:
                pred = outputs[f"{label_type.replace('_labels', '')}_logits"]
                if label_type in inputs:
                    labels = inputs[label_type]
                    if pred.shape[-1] > 1:
                        ret[label_type][0] += (
                            (torch.argmax(pred, dim=-1) == labels).float().mean().item()
                        )
                    else:
                        ret[label_type][0] += (
                            ((pred > 0.5) == labels).float().mean().item()
                        )
                    ret[label_type][1] += 1

        for key in ret:
            ret[key] = ret[key][0] / ret[key][1]
            ret["eval_" + key] = ret.pop(key)
        ret.update({"eval_loss": total_loss / len(eval_dataloader)})

        logger.info(ret)
        return ret
