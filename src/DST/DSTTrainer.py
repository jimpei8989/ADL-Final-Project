from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, List

from torch.utils.data import Dataset, DataLoader
import torch

from transformers import Trainer
from transformers import PreTrainedTokenizerBase

from datasets.dst_utils import load_dst_dataloader
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

        ret = defaultdict(list)

        with torch.no_grad():
            for inputs in tqdmm(eval_dataloader):
                inputs = self._prepare_inputs(inputs)
                outputs = self.model(**inputs)

                for key in outputs:
                    output = outputs[key]
                    if "loss" in key:
                        ret[key].append(output.item())
                    elif "logits" in key:
                        key = key.replace("logits", "labels")
                        if key in inputs:
                            labels = inputs[key]
                            # begin/end acc
                            if len(output.shape) > 1:
                                ret[key].append(
                                    (torch.argmax(output, dim=-1) == labels).float().mean().item()
                                )
                            # slot/value acc
                            else:
                                output = torch.nn.Sigmoid()(output)
                                ret[key].append(((output > 0.5) == labels).float().mean().item())
                    else:
                        raise KeyError

            ret = {
                f"{metric_key_prefix}_{k}": torch.Tensor(v).mean().item()
                for k, v in sorted(ret.items(), key=lambda x: x[0][::-1])
            }

            self.log(ret)
            self.control = self.callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                ret,
            )
            self._memory_tracker.stop_and_update_metrics(ret)

        return ret
