from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    BaseModelOutput,
    SequenceClassifierOutput,
)
from enum import Enum

from .config import ILKTConfig


def cls_pooling(last_hidden_state, attention_mask):
    return last_hidden_state[:, 0, :]


def create_head_blocks(
    hidden_size: int,
    n_dense: int,
    use_batch_norm: bool,
    use_layer_norm: bool,
    dropout: float,
    **kwargs,
) -> nn.Module:
    blocks = []
    for _ in range(n_dense):
        blocks.append(nn.Linear(hidden_size, hidden_size))
        if use_batch_norm:
            blocks.append(nn.BatchNorm1d(hidden_size))
        elif use_layer_norm:
            blocks.append(nn.LayerNorm(hidden_size))
        blocks.append(nn.ReLU())
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
    return nn.Sequential(*blocks)


class SentenceEmbeddingHead(nn.Module):
    def __init__(
        self, backbone_hidden_size: int, embedding_head_config: Dict[str, Any]
    ):
        super().__init__()
        self.config = embedding_head_config

        self.head = nn.Sequential(
            *[
                create_head_blocks(backbone_hidden_size, **embedding_head_config),
            ]
        )

    def forward(
        self, backbone_output: BaseModelOutput, attention_mask: torch.Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        if self.config["pool_type"] == "cls":
            embeddings = cls_pooling(backbone_output.last_hidden_state, attention_mask)
        else:
            raise NotImplementedError(
                f"Pooling type {self.config['pool_type']} not implemented"
            )
        if self.config["normalize_embeddings"]:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        return BaseModelOutputWithPooling(
            last_hidden_state=backbone_output.last_hidden_state,
            pooler_output=embeddings,  # type: ignore
        )


class MLMHead(nn.Module):
    def __init__(
        self,
        backbone_hidden_size: int,
        vocab_size: int,
        mlm_head_config: Dict[str, Any],
    ):
        super().__init__()
        self.config = mlm_head_config

        self.head = nn.Sequential(
            *[
                create_head_blocks(backbone_hidden_size, **mlm_head_config),
                nn.Linear(backbone_hidden_size, vocab_size),
            ]
        )

    def forward(
        self,
        backbone_output: BaseModelOutput,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        prediction_scores = self.head(backbone_output.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
            )
        return MaskedLMOutput(loss=loss)


class CLSHead(nn.Module):
    def __init__(
        self,
        backbone_hidden_size: int,
        n_classes: int,
        cls_head_config: Dict[str, Any],
    ):
        super().__init__()
        self.config = cls_head_config

        self.head = nn.Sequential(
            *[
                create_head_blocks(backbone_hidden_size, **cls_head_config),
                nn.Linear(backbone_hidden_size, n_classes),
            ]
        )

    def forward(
        self,
        backbone_output: BaseModelOutput,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if self.config["pool_type"] == "cls":
            embeddings = cls_pooling(backbone_output.last_hidden_state, attention_mask)
        else:
            raise NotImplementedError(
                f"Pooling type {self.config['pool_type']} not implemented"
            )

        prediction_scores = self.head(embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
            )
        return SequenceClassifierOutput(loss=loss)


class ForwardRouting(Enum):
    GET_SENTENCE_EMBEDDING = "get_sentence_embedding"
    GET_MLM_OUTPUT = "get_mlm_output"
    GET_CLS_OUTPUT = "get_cls_output"


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
        self.config.hidden_size = backbone_hidden_size
        backbone_vocab_size = backbone_config.vocab_size
        self.embedding_head = SentenceEmbeddingHead(
            backbone_hidden_size, config.embedding_head_config
        )
        self.mlm_head = MLMHead(
            backbone_hidden_size, backbone_vocab_size, config.mlm_head_config
        )

        self.cls_heads = nn.ModuleDict(
            dict(
                [
                    (
                        name,
                        CLSHead(
                            backbone_hidden_size, n_classes, config.cls_head_config
                        ),
                    )
                    for n_classes, name in config.cls_heads
                ]
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        forward_routing: ForwardRouting = ForwardRouting.GET_SENTENCE_EMBEDDING,
        **kwargs,
    ):
        if forward_routing == ForwardRouting.GET_SENTENCE_EMBEDDING:
            return self.get_sentence_embedding(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
        elif forward_routing == ForwardRouting.GET_MLM_OUTPUT:
            return self.get_mlm_output(
                input_ids, attention_mask, token_type_ids=token_type_ids, **kwargs
            )
        elif forward_routing == ForwardRouting.GET_CLS_OUTPUT:
            return self.get_cls_output(
                input_ids, attention_mask, token_type_ids=token_type_ids, **kwargs
            )
        else:
            raise ValueError(f"Unknown forward routing {forward_routing}")

    def get_sentence_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        backbone_output: BaseModelOutput = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        embedding_output = self.embedding_head(
            backbone_output, attention_mask, **kwargs
        )

        return embedding_output

    def get_mlm_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        backbone_output: BaseModelOutput = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        mlm_output = self.mlm_head(backbone_output, attention_mask, labels, **kwargs)

        return mlm_output

    def get_cls_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_name: str,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        backbone_output: BaseModelOutput = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        if head_name not in self.cls_heads:
            raise ValueError(f"Head {head_name} not found in model")

        cls_output = self.cls_heads[head_name](
            backbone_output, attention_mask, labels, **kwargs
        )

        return cls_output
