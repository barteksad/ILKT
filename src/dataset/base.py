from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
)
from sentence_transformers.evaluation import SentenceEvaluator


class TextDataset(ABC, IterableDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        dataset: IterableDataset,
        n_examples: int,
        to_checkpoint_evaluation: bool = True,
    ):
        self.name = "--".join(name.split("/"))
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_examples = n_examples
        self.to_checkpoint_evaluation = to_checkpoint_evaluation # for now always true, but we will change this

    def reset(self):
        self.ds_iter = iter(self.dataset)

    @abstractmethod
    def _process_row(self, row: Any) -> Dict[str, Any]:
        raise NotImplementedError

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

    def __iter__(self):
        worker_info = get_worker_info()
        stride = 0
        worker_id = 0
        if (
            worker_info is not None
        ):  # single-process data loading, return the full iterator
            stride = max(stride, worker_info.num_workers - 1)
            worker_id = worker_info.id

        while True:
            self.reset()
            for initial_skip in range(0, worker_id + 1):
                _ = next(self.ds_iter)
            while True:
                try:
                    row = next(self.ds_iter)
                    yield self._process_row(row)
                    for _ in range(stride):
                        _ = next(self.ds_iter)
                except StopIteration:
                    if self.n_examples is not None:
                        return
                    else:
                        break


class ContrastiveDataset(TextDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        dataset: IterableDataset,
        n_examples: int,
    ):
        super().__init__(name, tokenizer, batch_size, dataset, n_examples)

    @abstractmethod
    def get_data_collator(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(self) -> SentenceEvaluator:
        raise NotImplementedError


class MLMDataset(TextDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        mlm_probability: float,
        dataset: IterableDataset,
        n_examples: int,
    ):
        super().__init__(name, tokenizer, batch_size, dataset, n_examples)
        self.mlm_probability = mlm_probability

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
        dataset: IterableDataset,
        n_examples: int,
    ):
        super().__init__(name, tokenizer, batch_size, dataset, n_examples)
        self.n_classes = n_classes
        self.sentence_keys = sentence_keys
        self.label_key = label_key

    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=16)
