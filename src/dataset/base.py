from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
)


class TextDataset(ABC, Dataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
    ):
        self.name = "--".join(name.split("/"))
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @abstractmethod
    def get_data_collator(self) -> Any:
        raise NotImplementedError

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.get_data_collator(),
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )


class ContrastiveDataset(TextDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
    ):
        super().__init__(name, tokenizer, batch_size)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_data_collator(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class MLMDataset(TextDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        mlm_probability: float,
    ):
        super().__init__(name, tokenizer, batch_size)
        self.mlm_probability = mlm_probability

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_data_collator(self) -> Any:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=16,
        )
        return data_collator


class ClassificationDataset(TextDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        n_classes: int,
        sentence_keys: List[str],
        label_key: str,
    ):
        super().__init__(name, tokenizer, batch_size)
        self.n_classes = n_classes
        self.sentence_keys = sentence_keys
        self.label_key = label_key

    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=16)
