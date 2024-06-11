import logging
import os
from typing import List

import hydra
import wandb
import torch
from tqdm.auto import tqdm
from hydra.utils import instantiate
from lightning import Fabric
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from utils import (
    extract_output_dir,
    preprocess_config,
    setup_wandb
)
from train_util import (
    TrainBatchProcessStrategy,
    ValidationBatchProcessStrategy,
)

from dataset import ContrastiveDataset, MLMDataset, SentenceClassificationDataset
from train_util import create_optimizer_v2

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def get_fabric(config) -> Fabric:
    if torch.cuda.is_bf16_supported():
        fabric = instantiate(config.fabric, precision="bf16-mixed")
    else:
        fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def instantiate_datasets(
    config: DictConfig, tokenizer: PreTrainedTokenizer, dataset_type: str
) -> List[ContrastiveDataset | MLMDataset]:
    datasets = []

    if dataset_type == "train":
        if 'train_datasets' not in config:
            return []
        config_datasets = config.train_datasets
    elif dataset_type == "val":
        if 'val_datasets' not in config:
            return []
        config_datasets = config.val_datasets
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    if "contrastive" in config_datasets:
        for contrastive_dataset_config in config_datasets.contrastive.values():
            contrastive_dataset = instantiate(
                contrastive_dataset_config, tokenizer=tokenizer
            )
            datasets.append(contrastive_dataset)

    if "mlm" in config_datasets:
        for mlm_dataset_config in config_datasets.mlm.values():
            mlm_dataset = instantiate(mlm_dataset_config, tokenizer=tokenizer)
            datasets.append(mlm_dataset)

    if "sentence_classification" in config_datasets is not None:
        for cls_dataset_config in config_datasets.sentence_classification.values():
            cls_dataset = instantiate(cls_dataset_config, tokenizer=tokenizer)
            datasets.append(cls_dataset)
    return datasets


def prepare_dataloaders(
    fabric: Fabric, config: DictConfig, tokenizer: PreTrainedTokenizer
):
    train_dataloaders = []
    train_datasets = instantiate_datasets(config, tokenizer, "train")
    val_datasets = instantiate_datasets(config, tokenizer, "val")
    for dataset in train_datasets:
        train_dataloaders.append(dataset.get_dataloader())
    val_dataloaders = []
    for dataset in val_datasets:
        val_dataloaders.append(dataset.get_dataloader())

    train_dataloaders = fabric.setup_dataloaders(*train_dataloaders)
    val_dataloaders = fabric.setup_dataloaders(*val_dataloaders) if len(val_dataloaders) > 0 else []

    return train_dataloaders, val_dataloaders


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: DictConfig):
    preprocess_config(config)
    setup_wandb(config)
    output_dir = extract_output_dir(config)

    fabric = get_fabric(config)

    tokenizer = AutoTokenizer.from_pretrained(config.exp.pretrained_model_name_or_path)
    train_dataloaders, val_dataloaders = prepare_dataloaders(fabric, config, tokenizer)

    cls_heads = []
    # setting only for training datasets as this has to be trained, setting val dataset that was not present
    # in train will raise error
    for dataloader in train_dataloaders:
        if isinstance(dataloader.dataset, SentenceClassificationDataset):
            cls_heads.append((dataloader.dataset.n_classes, dataloader.dataset.name))

    config.model.config["cls_heads"] = cls_heads
    model = instantiate(config.model, _convert_="all")
    optimizer = create_optimizer_v2(model, **config.exp.optimizer)
    model, optimizer = fabric.setup(model, optimizer)

    TRAINING_STEPS = config.exp.training_steps

    current_step = 0
    iter_train_dataloaders = [iter(dataloader) for dataloader in train_dataloaders]

    train_batch_processor = TrainBatchProcessStrategy(model)
    train_batch_processor.on_start()
    val_batch_processor = ValidationBatchProcessStrategy(model)

    pbar = tqdm(total=TRAINING_STEPS,position=0, leave=True)
    while current_step < TRAINING_STEPS:
        # ----------------- training -----------------
        model.train()
        optimizer.zero_grad()
        for idx, dataloader in enumerate(train_dataloaders):
            try:
                batch = next(iter_train_dataloaders[idx])  # type: ignore
            except StopIteration:
                iter_train_dataloaders[idx] = iter(train_dataloaders[idx])  # type: ignore
                batch = next(iter_train_dataloaders[idx])  # type: ignore

            train_batch_output = train_batch_processor(batch, dataloader)
            loss = train_batch_output.loss
            fabric.backward(loss)

        optimizer.step()

        # ----------------- validation -----------------
        if (current_step + 1) % config.exp.validate_every == 0:
            train_batch_processor.on_end()
            val_batch_processor.on_start()
            model.eval()
            for idx, dataloader in enumerate(val_dataloaders):

                for batch in dataloader:
                    with torch.inference_mode():
                        _ = val_batch_processor(batch, dataloader)

            val_batch_processor.on_end()
            train_batch_processor.on_start()
            
        current_step += 1
        pbar.update(1)

    # TODO make it save properly, right now doesnt work
    model.config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    model.save_pretrained(os.path.join(output_dir, "ILKTModel"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ILKTModel"))

    # WARNING if you hange this, also hange in benchmarks where we do:
    # if name.startswith("ILKT"):
    #             group, name = name.split("/")[-1].split("_")
    # to distinguish between our models and others and save everything in wandb
    group, name = str(config.exp.log_dir).split("/")[-2:]
    model.push_to_hub(f"ILKT/{group}_{name}")
    tokenizer.push_to_hub(f"ILKT/{group}_{name}")


if __name__ == "__main__":
    main()
