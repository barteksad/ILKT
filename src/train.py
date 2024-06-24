import logging
import os

import hydra
import torch
from tqdm.auto import tqdm
from hydra.utils import instantiate
from lightning import Fabric
from omegaconf import DictConfig
from transformers import AutoTokenizer

from train_utils.data_iterator import SingleBatchPerDatasetIterator
from train_utils.dataset_loader import DatasetLoader
from utils import extract_output_dir, preprocess_config, setup_wandb
from train_utils.train_util import (
    TopNotchEvaluator,
    TrainBatchProcessStrategy,
)

from dataset import SentenceClassificationDataset
from train_utils.train_util import (
    create_optimizer_v2,
)

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def get_fabric(config) -> Fabric:
    if torch.cuda.is_bf16_supported():
        log.info("USING BF16-MIXED")
        fabric = instantiate(config.fabric, precision="bf16-mixed")
    else:
        log.info("USING FP32")
        fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: DictConfig):
    fabric = get_fabric(config)
    preprocess_config(config)
    setup_wandb(config, fabric)
    output_dir = extract_output_dir(config)

    tokenizer = AutoTokenizer.from_pretrained(config.exp.pretrained_model_name_or_path)
    dataset_loader = DatasetLoader(fabric, config, tokenizer)
    cls_heads = []
    # setting only for training datasets as this has to be trained, setting val dataset that was not present
    # in train will raise error
    for dataloader in dataset_loader.train_dataloaders:
        if isinstance(dataloader.dataset, SentenceClassificationDataset):
            cls_heads.append((dataloader.dataset.n_classes, dataloader.dataset.name))

    config.model.config["cls_heads"] = cls_heads
    with fabric.init_module():
        model = instantiate(config.model, _convert_="all")
    optimizer = create_optimizer_v2(model, **config.exp.optimizer)
    scheduler = instantiate(config.exp.scheduler, optimizer=optimizer)
    model, optimizer = fabric.setup(model, optimizer)

    model.config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    TRAINING_STEPS = config.exp.training_steps
    NEXT_VALIDATION_STEP = config.exp.validate_every

    current_step = 0

    train_batch_processor = TrainBatchProcessStrategy(
        model,
        monitor_stiffness_every=config.exp.monitor_stiffness_every,
        monitor_stiffness_steps=config.exp.monitor_stiffness_steps,
        beta=config.exp.beta,
    )
    train_batch_processor.on_start(fabric)

    train_iterator = SingleBatchPerDatasetIterator(dataset_loader.train_dataloaders)
    evaluator = TopNotchEvaluator(
        model, tokenizer, dataset_loader.val_dataloaders, output_dir
    )

    if fabric.is_global_zero:
        pbar = tqdm(total=TRAINING_STEPS, position=0, leave=True)
    while current_step < TRAINING_STEPS:

        # ----------------- stiffness logging -----------------
        if (current_step + 1) > NEXT_VALIDATION_STEP:
            model.log_gradients = True
        else:
            model.log_gradients = False

        # ----------------- training -----------------
        model.train()
        for batch, dataloader in train_iterator:
            optimizer.zero_grad()
            train_batch_output = train_batch_processor(batch, dataloader, fabric)
            loss = train_batch_output.loss
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=config.exp.max_grad_norm)
            optimizer.step()
            scheduler.step()

            current_step += 1
            if fabric.is_global_zero:
                pbar.update(1)

        # ----------------- validation -----------------
        if (current_step + 1) > NEXT_VALIDATION_STEP:
            train_batch_processor.on_end(fabric)
            epoch = current_step // config.exp.validate_every
            if fabric.is_global_zero:
                log.info("Validation Step")
                model.eval()
                with torch.inference_mode():
                    evaluator(epoch, fabric)

            train_batch_processor.on_start(fabric)

            NEXT_VALIDATION_STEP += config.exp.validate_every

            model.save_pretrained(os.path.join(output_dir, "ILKTModel"))
            tokenizer.save_pretrained(os.path.join(output_dir, "ILKTModel"))
            group, name = str(config.exp.log_dir).split("/")[-2:]
            model.push_to_hub(f"ILKT/{group}_{name}_epoch_{epoch}")
            tokenizer.push_to_hub(f"ILKT/{group}_{name}_epoch_{epoch}")

    model.save_pretrained(os.path.join(output_dir, "ILKTModel"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ILKTModel"))
    group, name = str(config.exp.log_dir).split("/")[-2:]
    model.push_to_hub(f"ILKT/{group}_{name}_last")
    tokenizer.push_to_hub(f"ILKT/{group}_{name}_last")


if __name__ == "__main__":
    main()
