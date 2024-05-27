from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoTokenizer

import utils
from dataset import ContrastiveDataset, MLMDataset


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: DictConfig):

    utils.preprocess_config(config)
    utils.setup_wandb(config)
    output_dir = utils.extract_output_dir(config)

    tokenizer = AutoTokenizer.from_pretrained(config.exp.pretrained_model_name_or_path)
    contrastive_datasets: List[ContrastiveDataset] = []
    mlm_datasets: List[MLMDataset] = []

    if config.datasets.contrastive is not None:
        for contrastive_dataset_config in config.datasets.contrastive.values():
            print(contrastive_dataset_config)
            contrastive_dataset = instantiate(
                contrastive_dataset_config, tokenizer=tokenizer
            )
            print(contrastive_dataset)
            contrastive_datasets.append(contrastive_dataset)

    if config.datasets.mlm is not None:
        for mlm_dataset_config in config.datasets.mlm.values():
            mlm_dataset = instantiate(mlm_dataset_config, tokenizer=tokenizer)
            mlm_datasets.append(mlm_dataset)

    model = instantiate(config.model)


if __name__ == "__main__":
    main()
