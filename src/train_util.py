from typing import Callable, Optional

import torch.nn as nn
import torch.optim as optim

__all__ = ["create_optimizer_v2"]

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

    opt_args = dict(weight_decay=weight_decay, **kwargs)

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise NotImplementedError(f"Optimizer {opt} not implemented")

    return optimizer
