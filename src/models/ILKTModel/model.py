from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    BaseModelOutput,
)

from .config import ILKTConfig


class SentenceEmbeddingHead(nn.Module):

    def __init__(
        self, backbone_hidden_size: int, embedding_head_config: Dict[str, Any]
    ):
        super().__init__()

    def forward(
        self, backbone_output: BaseModelOutput, **kwargs
    ) -> BaseModelOutputWithPooling:
        return BaseModelOutputWithPooling(
            pooler_output=backbone_output.last_hidden_state[:, 0, :]  # type: ignore
        )


class MLMHead(nn.Module):

    def __init__(
        self,
        backbone_hidden_size: int,
        vocab_size: int,
        mlm_head_config: Dict[str, Any],
    ):
        super().__init__()

        self.head = nn.Linear(backbone_hidden_size, vocab_size)

    def forward(self, backbone_output: BaseModelOutput, **kwargs) -> MaskedLMOutput:
        prediction_scores = self.head(backbone_output.last_hidden_state)

        loss = None
        if "labels" in kwargs:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                kwargs["labels"].view(-1),
            )   
        return MaskedLMOutput(loss=loss)


class ILKTModel(PreTrainedModel):
    config_class = ILKTConfig

    def __init__(self, config: ILKTConfig):
        super().__init__(config)

        backbone_config = AutoConfig.from_pretrained(**config.backbone_config)
        pretrained_model_name_or_path = config.backbone_config[
            "pretrained_model_name_or_path"
        ]
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=backbone_config
        )

        backbone_hidden_size = backbone_config.hidden_size
        backbone_vocab_size = backbone_config.vocab_size
        self.embedding_head = SentenceEmbeddingHead(
            backbone_hidden_size, config.embedding_head_config
        )
        self.mlm_head = MLMHead(
            backbone_hidden_size, backbone_vocab_size, config.mlm_head_config
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        return self.get_sentence_embedding(input_ids, attention_mask, **kwargs)

    def get_sentence_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        backbone_output: BaseModelOutput = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        embedding_output = self.embedding_head(backbone_output, **kwargs)

        return embedding_output

    def get_mlm_output(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        backbone_output: BaseModelOutput = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        mlm_output = self.mlm_head(backbone_output, **kwargs)

        return mlm_output
