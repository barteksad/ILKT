import logging
import os
from typing import List

import hydra
from hydra.utils import instantiate
from lightning import Fabric
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

import utils
from dataset import ContrastiveDataset, MLMDataset

log = logging.getLogger(__name__)


def get_fabric(config) -> Fabric:
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def instantiate_datasets(
    config: DictConfig, tokenizer: PreTrainedTokenizer
) -> List[ContrastiveDataset | MLMDataset]:
    datasets = []
    if config.datasets.contrastive is not None:
        for contrastive_dataset_config in config.datasets.contrastive.values():
            contrastive_dataset = instantiate(
                contrastive_dataset_config, tokenizer=tokenizer
            )
            datasets.append(contrastive_dataset)

    if config.datasets.mlm is not None:
        for mlm_dataset_config in config.datasets.mlm.values():
            mlm_dataset = instantiate(mlm_dataset_config, tokenizer=tokenizer)
            datasets.append(mlm_dataset)
    return datasets


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: DictConfig):

    utils.preprocess_config(config)
    utils.setup_wandb(config)
    output_dir = utils.extract_output_dir(config)

    fabric = get_fabric(config)

    tokenizer = AutoTokenizer.from_pretrained(config.exp.pretrained_model_name_or_path)
    datasets = instantiate_datasets(config, tokenizer)  # type: ignore
    dataloaders = [
        fabric.setup_dataloaders(dataset.get_dataloader()) for dataset in datasets
    ]

    model = instantiate(config.model)
    optimizer = instantiate(config.exp.optimizer, params=model.parameters())
    model, optimizer = fabric.setup(model, optimizer)

    TRAINING_STEPS = config.exp.training_steps

    current_step = 0
    iter_dataloaders = [iter(dataloader) for dataloader in dataloaders]

    while current_step < TRAINING_STEPS:
        optimizer.zero_grad()
        for idx, dataset in enumerate(datasets):
            try:
                batch = next(iter_dataloaders[idx])  # type: ignore
            except StopIteration:
                iter_dataloaders[idx] = iter(dataloaders[idx])  # type: ignore
                batch = next(iter_dataloaders[idx])  # type: ignore

            if isinstance(dataset, ContrastiveDataset):
                # TODO handle keys in batch somehow better, this is poor
                outputs = model.get_sentence_embedding(**batch["set"])
                outputs = dataset.format_for_loss_fn(
                    {"set": outputs.pooler_output, "labels": batch["labels"]}
                )
                loss = dataset.loss_fn(**outputs)
            else:
                outputs = model.get_mlm_output(**batch)
                loss = outputs.loss

            fabric.backward(loss)

        optimizer.step()
        current_step += 1

    # TODO make it save properly, right now doesnt work
    model.config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    model.save_pretrained(os.path.join(output_dir, "ILKTModel"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ILKTModel"))


if __name__ == "__main__":
    main()
