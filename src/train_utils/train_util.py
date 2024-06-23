from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.optim as optim
from lightning import Fabric
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling
import bitsandbytes as bnb
from .metrics import stiffness
import wandb

from dataset import ContrastiveDataset, MLMDataset, SentenceClassificationDataset
from models import ILKTModel, ForwardRouting
from train_utils.data_iterator import FullValidIterator, DL_TYPE

__all__ = ["create_optimizer_v2"]


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


class StiffnessMonitor(BatchProcessor):
    def __init__(self, monitor_stiffness_every: int, monitor_stiffness_steps: int):
        self.monitor_stiffness_every = monitor_stiffness_every
        self.monitor_stiffness_steps = monitor_stiffness_steps
        self.steps = 0
        self.gradients = defaultdict(list)
        self.hook = None

    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:

        self.steps += 1

        if self.hook is not None:
            self.hook.remove()
            self.hook = None

        metric_dict = {"stiffness": {}}

        if self.steps % self.monitor_stiffness_every < self.monitor_stiffness_steps:
            if isinstance(dataloader.dataset, ContrastiveDataset):
                task_type = ForwardRouting.GET_SENTENCE_EMBEDDING
            elif isinstance(dataloader.dataset, SentenceClassificationDataset):
                task_type = ForwardRouting.GET_CLS_OUTPUT
            elif isinstance(dataloader.dataset, MLMDataset):
                task_type = ForwardRouting.GET_MLM_OUTPUT
            else:
                raise ValueError(f"Unknown dataset type {type(dataloader.dataset)}")

            def backward_hook(module, grad_input, grad_output):

                self.gradients[task_type].append(
                    grad_output[0][:, 0, :].detach().cpu()
                )

            self.hook = model.backbone.encoder.layer[-1].register_full_backward_hook(
                backward_hook
            )
        elif len(self.gradients.items()) > 1:
            for task, gradients in self.gradients.items():
                self.gradients[task] = torch.cat(gradients).mean(dim=0)

            for task1 in self.gradients:
                for task2 in self.gradients:
                    if str(task1) > str(task2):
                        metric_dict[f"{task1}x{task2}_cosine"] = stiffness(
                            self.gradients[task1],
                            self.gradients[task2],
                            "cosine",
                        )
            self.gradients = defaultdict(list)
            metric_dict = {"stiffness": metric_dict}
            print(metric_dict)

        return metric_dict

    def on_end(self, fabric: Fabric):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


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


class AccuracyProcessor(BatchProcessor):
    def on_batch(
        self,
        model: ILKTModel,
        batch: Dict[str, Any],
        dataloader: DL_TYPE,
        fabric: Fabric,
    ) -> Dict[str, Any]:
        with fabric.autocast():
            if isinstance(dataloader.dataset, SentenceClassificationDataset):
                logits = batch["model_outputs"].logits
                labels = batch["labels"]
                accuracy = (logits.argmax(dim=-1) == labels).float().mean()
            elif isinstance(dataloader.dataset, MLMDataset):
                logits = batch["model_outputs"].logits
                labels = batch["labels"]
                labels_mask = labels != -100
                accuracy = (logits.argmax(dim=-1) == labels)[labels_mask].float().mean()
            else:
                raise NotImplementedError(
                    f"Accuracy is not implemented for {type(dataloader.dataset)}"
                )
        return {"accuracy": accuracy}


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
                if key == "stiffness" and key in batch:
                    for k, v in batch[key].items():
                        self.log("stiffness", k, v)
                else:
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
    def __init__(self, model: ILKTModel, steps: List[List[BatchProcessor]]):
        self.model = model
        self.steps = steps

    def on_start(self, fabric: Fabric):
        for step_list in self.steps:
            for step in step_list:
                step.on_start(fabric)

    def on_end(self, fabric: Fabric):
        for step_list in self.steps:
            for step in step_list:
                step.on_end(fabric)

    def __call__(
        self, batch: Dict[str, Any], dataloader: DL_TYPE, fabric: Fabric
    ) -> BatchProcessOutput:
        for step_list in self.steps:
            step_outputs = {}
            for step in step_list:
                step_output = step.on_batch(self.model, batch, dataloader, fabric)
                step_outputs.update(step_output)
            batch = step_outputs
        return BatchProcessOutput(loss=batch["loss"])


class TrainBatchProcessStrategy(BatchProcessStrategy):

    def __init__(
        self,
        model: ILKTModel,
        monitor_stiffness_every: int,
        monitor_stiffness_steps: int,
        steps=None,
    ):
        if steps is None:
            steps = [
                [ModelOutputProcessor()],
                [
                    LossProcessor(),
                    StiffnessMonitor(monitor_stiffness_every, monitor_stiffness_steps),
                ],
                [
                    WandbMetricLogger(
                        "train",
                        [
                            "loss",
                            "stiffness",
                        ],
                        log_per_batch=True,
                    )
                ],
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
                [ModelOutputProcessor()],
                [LossProcessor(), AccuracyProcessor()],
                [MetricProcessor(["loss", "accuracy"])],
                [WandbMetricLogger("val", ["loss", "accuracy"], log_per_batch=False)],
            ]
        super().__init__(model, steps)


class TopNotchEvaluator:
    def __init__(
        self,
        model: ILKTModel,
        tokenizer: PreTrainedTokenizer,
        dataloaders: List[DL_TYPE],
        output_dir: Path,
    ):
        self.sentence_transformer_model = custom_transformer2sentence_transformer(
            tokenizer, model
        )
        self.contrastive_dataloaders = list(
            filter(lambda x: isinstance(x.dataset, ContrastiveDataset), dataloaders)
        )
        self.contrastive_evaluators = [
            dataloader.dataset.get_evaluator()
            for dataloader in self.contrastive_dataloaders
        ]
        self.other_dataloaders_iterator = FullValidIterator(
            list(
                filter(
                    lambda x: not isinstance(x.dataset, ContrastiveDataset), dataloaders
                )
            )
        )
        self.other_processor = ValidationBatchProcessStrategy(model)
        self.output_dir = output_dir

    def __call__(self, epoch: int, fabric: Fabric):
        output_path = self.output_dir / f"eval_epoch_{epoch}"
        output_path.mkdir(exist_ok=True, parents=True)
        for dataloader, evaluator in zip(
            self.contrastive_dataloaders, self.contrastive_evaluators
        ):
            results = evaluator(self.sentence_transformer_model, output_path, epoch)
            for k, v in results.items():
                wandb.log({f"{k}": v})
        self.other_processor.on_start(fabric)
        for batch, dataloader in self.other_dataloaders_iterator:
            self.other_processor(batch, dataloader, fabric)
        self.other_processor.on_end(fabric)


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
    elif opt_lower == "bnbadamw8bit":
        optimizer = bnb.optim.AdamW8bit(parameters, **opt_args)
    else:
        raise NotImplementedError(f"Optimizer {opt} not implemented")

    return optimizer
