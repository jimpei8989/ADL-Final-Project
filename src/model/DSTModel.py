import pickle
from typing import Dict

import torch
from torch.nn import BCELoss, CrossEntropyLoss

from transformers import AutoModel, AutoConfig

from DSTModelOutput import DSTModelOutput


class DSTModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str = None,
        config: str = None,
    ) -> None:
        super(DSTModel, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else AutoConfig()
            )

        self.cls_fc = torch.nn.Linear(self.backbone.config.hidden_size, 2)
        self.span_fc = torch.nn.Linear(self.backbone.config.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()

        self.cls_criterion = BCELoss()
        self.span_criterion = CrossEntropyLoss()

    def forward(
        self, input_ids, slot_labels=None, value_labels=None, span_labels=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states = self.backbone(input_ids)["last_hidden_state"]

        cls_logits = self.fc(last_hidden_states[:, 0])
        cls_logits = self.sigmoid(cls_logits)
        slot_logits = cls_logits[:, 0]
        value_logits = cls_logits[:, 1]

        qa_logits = self.span_fc(last_hidden_states)
        start_logits = qa_logits[:, :, 0]
        end_logits = qa_logits[:, :, 1]

        total_loss = 0
        slot_loss = None
        value_loss = None
        span_loss = None

        if slot_labels is not None:
            slot_loss = self.cls_criterion(slot_logits, slot_labels)
            total_loss += slot_loss
        if value_labels is not None:
            value_loss = self.cls_criterion(value_logits, value_labels)
            total_loss += value_loss
        if span_labels is not None:
            start_loss = self.span_criterion(start_logits, span_labels[:, 0])
            end_loss = self.span_criterion(end_logits, span_labels[:, 1])
            span_loss = (start_loss + end_loss) / 2
            total_loss += span_loss

        return DSTModelOutput(
            loss=total_loss,
            slot_loss=slot_loss,
            value_loss=value_loss,
            span_loss=span_loss,
            slot_logits=slot_logits,
            value_logits=value_logits,
            start_logits=start_logits,
            end_logits=end_logits,
        )
