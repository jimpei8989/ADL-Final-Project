import pickle
from pathlib import Path
from typing import Dict

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Sequential, Linear, Tanh

from transformers import AutoModel, AutoConfig

from DST.DSTModelOutput import DSTModelOutput


class DSTModel(torch.nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Path,
        model_name: str = "bert-base-uncased",
        pool: bool = False,
        device=torch.device("cpu"),
    ):
        model = cls(model_name=model_name, pool=pool)
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
        return model

    def __init__(
        self,
        model_name: str = None,
        config: str = None,
        pool: bool = False,
        slot_weight: float = 1.0,
        value_weight: float = 1.0,
        span_weight: float = 1.0,
    ) -> None:
        super(DSTModel, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else AutoConfig()
            )

        self.pool = pool
        self.hidden_size = self.backbone.config.hidden_size
        self.max_position_embeddings = self.backbone.config.max_position_embeddings
        self.slot_weight = slot_weight
        self.value_weight = value_weight
        self.span_weight = span_weight

        self.cls_fc = Linear(self.hidden_size, 2)
        self.span_fc = Linear(self.hidden_size, 2)

        self.cls_criterion = BCEWithLogitsLoss()
        self.span_criterion = CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        slot_labels=None,
        value_labels=None,
        begin_labels=None,
        end_labels=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states = self.backbone(input_ids)["last_hidden_state"]

        cls_logits = (
            self.cls_fc(self.backbone.pooler(last_hidden_states))
            if self.pool
            and getattr(self.backbone, "pooler", False)
            and self.backbone.pooler is not None
            else self.cls_fc(last_hidden_states[:, 0])
        )
        slot_logits = cls_logits[:, 0]
        value_logits = cls_logits[:, 1]

        qa_logits = self.span_fc(last_hidden_states)
        begin_logits = qa_logits[:, :, 0]
        end_logits = qa_logits[:, :, 1]

        total_loss = 0
        slot_loss = None
        value_loss = None
        span_loss = None

        if slot_labels is not None:
            slot_loss = self.cls_criterion(slot_logits, slot_labels)
            total_loss += slot_loss * self.slot_weight
        if value_labels is not None:
            value_loss = self.cls_criterion(value_logits, value_labels)
            total_loss += value_loss * self.value_weight
        if begin_labels is not None and end_labels is not None:
            begin_loss = self.span_criterion(begin_logits, begin_labels)
            end_loss = self.span_criterion(end_logits, end_labels)
            span_loss = (begin_loss + end_loss) / 2
            total_loss += span_loss * self.span_weight

        return DSTModelOutput(
            loss=total_loss,
            slot_loss=slot_loss,
            value_loss=value_loss,
            span_loss=span_loss,
            slot_logits=slot_logits,
            value_logits=value_logits,
            begin_logits=begin_logits,
            end_logits=end_logits,
        )
