from typing import List, TypeVar

from dataset import ContrastiveDataset, MLMDataset, SentenceClassificationDataset
from hydra.utils import instantiate
from lightning import Fabric
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

DS_TYPE = TypeVar(
    "DS_TYPE",
    ContrastiveDataset,
    MLMDataset,
    SentenceClassificationDataset,
)


class DatasetLoader:
    def __init__(self, fabric: Fabric, config: DictConfig, tokenizer: PreTrainedTokenizer):
        self.fabric = fabric
        self.config = config
        self.tokenizer = tokenizer
        self.prepare_dataloaders()

    def initiate_datasets(self, dataset_type: str) -> List[DS_TYPE]:
        datasets = []
        if dataset_type == "train":
            config_datasets = self.config.train_datasets
        elif dataset_type == "val":
            config_datasets = self.config.val_datasets
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}")

        if "contrastive" in config_datasets is not None:
            for contrastive_dataset_config in config_datasets.contrastive.values():
                contrastive_dataset = instantiate(
                    contrastive_dataset_config, tokenizer=self.tokenizer
                )
                datasets.append(contrastive_dataset)

        if "mlm" in config_datasets:
            for mlm_dataset_config in config_datasets.mlm.values():
                mlm_dataset = instantiate(mlm_dataset_config, tokenizer=self.tokenizer)
                datasets.append(mlm_dataset)

        if "sentence_classification" in config_datasets is not None:
            for cls_dataset_config in config_datasets.sentence_classification.values():
                cls_dataset = instantiate(cls_dataset_config, tokenizer=self.tokenizer)
                datasets.append(cls_dataset)
        return datasets

    def prepare_dataloaders(self):
        train_dataloaders = []
        train_datasets = self.initiate_datasets("train")
        val_datasets = self.initiate_datasets("val")
        for dataset in train_datasets:
            train_dataloaders.append(dataset.get_dataloader())
        val_dataloaders = []
        for dataset in val_datasets:
            val_dataloaders.append(dataset.get_dataloader())
        self.train_dataloaders = self.fabric.setup_dataloaders(*train_dataloaders)
        self.val_dataloaders = self.fabric.setup_dataloaders(*val_dataloaders)
