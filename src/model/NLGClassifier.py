import pickle
from typing import Dict, Optional
from dataclasses import dataclass

import torch

from transformers import AutoModel, AutoConfig, file_utils


@dataclass
class NLGClassifyOutput(file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class NLGClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name: str = None,
        config: str = None,
    ) -> None:
        super(NLGClassifier, self).__init__()
        if model_name is not None:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(
                pickle.load(open(config, "rb")) if config is not None else AutoConfig()
            )
        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, labels) -> Dict[str, torch.Tensor]:
        x = self.backbone(input_ids=input_ids)["last_hidden_state"][:, 0]

        x = self.fc(x)
        x = self.sigmoid(x)

        pred = x.view(-1)
        loss = torch.nn.BCELoss()(pred, labels)

        return NLGClassifyOutput(loss=loss, logits=pred)
