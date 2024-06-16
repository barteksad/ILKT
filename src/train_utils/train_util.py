from abc import abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning import Fabric

from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling

import wandb

from dataset import ContrastiveDataset, MLMDataset, SentenceClassificationDataset
from models import ILKTModel, ForwardRouting

__all__ = ["create_optimizer_v2"]

DL_TYPE = TypeVar(
    "DL_TYPE",
    DataLoader[ContrastiveDataset],
    DataLoader[MLMDataset],
    DataLoader[SentenceClassificationDataset],
)


# here we combine API for models and datasets, this is the only place where we need to know model and dataset spec
class BatchProcessor:
    def on_start(self, fabric: Fabric):
        pass

    def on_end(self, fabric: Fabric):
        pass

    @abstractmethod
    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        pass


class ModelOutputProcessor(BatchProcessor):
    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        if isinstance(dataloader.dataset, ContrastiveDataset):
            model_outputs = (
                model(
                    **inp, forward_routing=ForwardRouting.GET_SENTENCE_EMBEDDING
                ).pooler_output
                for inp in batch["model_inputs"]  # type: ignore
            )
        elif isinstance(dataloader.dataset, SentenceClassificationDataset):
            model_outputs = model(
                **batch,
                head_name=dataloader.dataset.name,
                forward_routing=ForwardRouting.GET_CLS_OUTPUT,
            )
        elif isinstance(dataloader.dataset, MLMDataset):
            model_outputs = model(
                **batch, forward_routing=ForwardRouting.GET_MLM_OUTPUT
            )
        else:
            raise ValueError(f"Unknown dataset type {type(dataloader.dataset)}")

        return {
            "model_outputs": model_outputs,
            "labels": batch["labels"] if "labels" in batch else None,  # type: ignore
        }


class LossProcessor(BatchProcessor):
    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        with fabric.autocast():
            if isinstance(dataloader.dataset, ContrastiveDataset):
                loss = dataloader.dataset.get_loss(batch)
            elif isinstance(
                dataloader.dataset, (MLMDataset, SentenceClassificationDataset)
            ):
                loss = batch["model_outputs"].loss
            else:
                raise ValueError(f"Unknown dataset type {type(dataloader.dataset)}")
        return {"loss": loss}


class MetricProcessor(BatchProcessor):
    def __init__(self, keys_to_track: List[str]):
        self.keys_to_track = keys_to_track

    def on_start(self, fabric: Fabric):
        self.values = defaultdict(int)
        self.counts = defaultdict(int)

    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        for key in self.keys_to_track:
            reduced_tensor = fabric.all_reduce(batch[key], reduce_op="mean")
            self.values[key] += reduced_tensor.item()
            self.counts[key] += 1
        return {
            **batch,
            "agg_metrics": {
                key: self.values[key] / self.counts[key] for key in self.keys_to_track
            },
        }


class WandbMetricLogger(BatchProcessor):
    def __init__(self, split: str, keys_to_track: List[str], log_per_batch: bool):
        self.split = split
        self.keys_to_track = keys_to_track
        self.log_per_batch = log_per_batch

    def log(self, name: str, key: str, value: float | torch.Tensor):
        value_to_log = value
        if isinstance(value, torch.Tensor):
            value_to_log = value.item()
        wandb.log({f"{self.split}/{name}/{key}": value_to_log})

    def on_start(self, fabric: Fabric):
        self.last_batch = {}

    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        if self.log_per_batch and fabric.is_global_zero:
            for key in self.keys_to_track:
                self.log(dataloader.dataset.name, key, batch[key])
        else:
            self.last_batch[dataloader.dataset.name] = batch["agg_metrics"]

        return batch

    def on_end(self, fabric: Fabric):
        if not self.log_per_batch and fabric.is_global_zero:
            for dataset_name, metrics in self.last_batch.items():
                for key in self.keys_to_track:
                    self.log(dataset_name, key, metrics[key])


BatchProcessOutput = NamedTuple("BatchProcessOutput", [("loss", torch.Tensor)])


class BatchProcessStrategy:
    def __init__(self, model: ILKTModel, steps: List[BatchProcessor]):
        self.model = model
        self.steps = steps

    def on_start(self, fabric: Fabric):
        for step in self.steps:
            step.on_start(fabric)

    def on_end(self, fabric: Fabric):
        for step in self.steps:
            step.on_end(fabric)

    def __call__(
        self, batch: Dict[str, Any], dataloader: DL_TYPE, fabric: Fabric
    ) -> BatchProcessOutput:
        for step in self.steps:
            batch = step.on_batch(self.model, batch, dataloader, fabric)
        return BatchProcessOutput(loss=batch["loss"])


class TrainBatchProcessStrategy(BatchProcessStrategy):
    def __init__(self, model: ILKTModel, steps=None):
        if steps is None:
            steps = [
                ModelOutputProcessor(),
                LossProcessor(),
                WandbMetricLogger("train", ["loss"], log_per_batch=True),
            ]  # W ten sposób przy każdym wykonaniu kosntruktora rtobisz nowe obiekty - znacznie mniej błędogenne
        super().__init__(model, steps)


class ValidationBatchProcessStrategy(BatchProcessStrategy):
    def __init__(
        self,
        model: ILKTModel,
        steps=None,
    ):
        if steps is None:
            steps = [
                ModelOutputProcessor(),
                LossProcessor(),
                MetricProcessor(["loss"]),
                WandbMetricLogger("val", ["loss"], log_per_batch=False),
            ]
        super().__init__(model, steps)


def custom_transformer2sentence_transformer(
    tokenizer: PreTrainedTokenizer, model: ILKTModel
):

    class DummyWrapper(nn.Module):
        def tokenize(
            self,
            texts: Union[List[str], List[Dict], List[Tuple[str, str]]],
            padding: Union[str, bool] = True,
        ):
            """Tokenizes a text and maps tokens to token-ids"""
            output = {}
            if isinstance(texts[0], str):
                to_tokenize = [texts]
            elif isinstance(texts[0], dict):
                to_tokenize = []
                output["text_keys"] = []
                for lookup in texts:
                    text_key, text = next(iter(lookup.items()))
                    to_tokenize.append(text)
                    output["text_keys"].append(text_key)
                to_tokenize = [to_tokenize]
            else:
                batch1, batch2 = [], []
                for text_tuple in texts:
                    batch1.append(text_tuple[0])
                    batch2.append(text_tuple[1])
                to_tokenize = [batch1, batch2]

            # strip
            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

            output.update(
                tokenizer(
                    *to_tokenize,
                    padding=padding,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=model.config.max_length,
                )
            )

            # data_collator = DataCollatorWithPadding(tokenizer)
            # output = data_collator(output)

            return output

        def forward(self, features):
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features[k] = v.to(model.device)

            output_states = model(
                **features,
            )
            output_tokens = output_states[0]

            features.update(
                {
                    "token_embeddings": output_tokens,
                    "attention_mask": features["attention_mask"],
                }
            )
            return features

    pooling = Pooling(
        model.config.hidden_size, model.config.embedding_head_config["pool_type"]
    )

    sentence_transformer_model = SentenceTransformer(
        modules=[DummyWrapper(), pooling, Normalize()],
    )

    return sentence_transformer_model


"""
code taken from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py#L194
"""


def param_groups_weight_decay(
    model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def create_optimizer_v2(
    model_or_params,
    opt: str = "adamw",
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    foreach: Optional[bool] = None,
    filter_bias_and_bn: bool = True,
    param_group_fn: Optional[Callable] = None,
    **kwargs,
):
    """Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        foreach: Enable / disable foreach (multi-tensor) operation if True / False. Choose safe default if None
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, "no_weight_decay"):
            no_weight_decay = model_or_params.no_weight_decay()

        if param_group_fn:
            parameters = param_group_fn(model_or_params)
        elif weight_decay and filter_bias_and_bn:
            parameters = param_groups_weight_decay(
                model_or_params, weight_decay, no_weight_decay
            )
            weight_decay = 0.0
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    opt_args = dict(weight_decay=weight_decay, lr=lr, **kwargs)

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "sgd":
        optimizer = optim.SGD(parameters, momentum=momentum, **opt_args)
    else:
        raise NotImplementedError(f"Optimizer {opt} not implemented")

    return optimizer
