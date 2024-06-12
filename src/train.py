import logging
import os
from typing import List

import hydra
import torch
from tqdm.auto import tqdm
from hydra.utils import instantiate
from lightning import Fabric
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from train_utils.data_iterator import SingleBatchPerDatasetIterator, FullValidIterator
from train_utils.dataset_loader import DatasetLoader
from utils import (
    extract_output_dir,
    preprocess_config,
    setup_wandb
)
from train_utils.train_util import (
    TrainBatchProcessStrategy,
    ValidationBatchProcessStrategy,
)

from dataset import ContrastiveDataset, MLMDataset, SentenceClassificationDataset
from train_utils.train_util import create_optimizer_v2

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


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: DictConfig):
    preprocess_config(config)
    setup_wandb(config)
    output_dir = extract_output_dir(config)

    fabric = get_fabric(config)

    tokenizer = AutoTokenizer.from_pretrained(config.exp.pretrained_model_name_or_path)
    dataset_loader = DatasetLoader(fabric, config, tokenizer)
    cls_heads = []
    # setting only for training datasets as this has to be trained, setting val dataset that was not present
    # in train will raise error
    for dataloader in dataset_loader.train_dataloaders:
        if isinstance(dataloader.dataset, SentenceClassificationDataset):
            cls_heads.append((dataloader.dataset.n_classes, dataloader.dataset.name))

    config.model.config["cls_heads"] = cls_heads
    model = instantiate(config.model, _convert_="all")
    optimizer = create_optimizer_v2(model, **config.exp.optimizer)
    model, optimizer = fabric.setup(model, optimizer)

    TRAINING_STEPS = config.exp.training_steps

    current_step = 0

    train_batch_processor = TrainBatchProcessStrategy(model)
    train_batch_processor.on_start()
    val_batch_processor = ValidationBatchProcessStrategy(model)

    train_iterator = SingleBatchPerDatasetIterator(dataset_loader.train_dataloaders)
    valid_iterator = FullValidIterator(dataset_loader.val_dataloaders)

    pbar = tqdm(total=TRAINING_STEPS, position=0, leave=True)
    while current_step < TRAINING_STEPS:
        # ----------------- training -----------------
        model.train()
        optimizer.zero_grad()
        for batch, dataloader in train_iterator:
            train_batch_output = train_batch_processor(batch, dataloader)
            loss = train_batch_output.loss
            fabric.backward(loss)

        optimizer.step()

        # ----------------- validation -----------------
        if (current_step + 1) % config.exp.validate_every == 0:
            train_batch_processor.on_end()
            val_batch_processor.on_start()
            model.eval()
            for batch, dataloader in valid_iterator:
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
