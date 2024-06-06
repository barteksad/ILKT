import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import Any, Dict

from .base import ContrastiveDataset


class TripletDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        n_examples: int,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        **kwargs
    ):
        super().__init__(name, tokenizer, batch_size)

        self.dataset = load_dataset(name, "triplet", split=split, streaming=True)
        self.dataset = self.dataset.shuffle()
        self.n_examples = n_examples
        self.max_length = max_length
        self.loss_fn = loss_fn

    def reset(self):
        self.ds_iter = iter(self.dataset)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        if idx == 0:
            self.reset()

        row = next(self.ds_iter)

        query = self.tokenizer(
            row["query"], truncation=True, max_length=self.max_length
        )
        positive = self.tokenizer(
            row["positive"], truncation=True, max_length=self.max_length
        )
        negative = self.tokenizer(
            row["negative"], truncation=True, max_length=self.max_length
        )

        return {"query": query, "positive": positive, "negative": negative}

    def get_data_collator(self):

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def _collate_df(batch):

            query = data_collator([x["query"] for x in batch])
            positive = data_collator([x["positive"] for x in batch])
            negative = data_collator([x["negative"] for x in batch])

            return {"model_inputs": (query, positive, negative)}

        return _collate_df

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.get_data_collator(),
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )

    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        query, positive, negative = batch["model_outputs"]

        sentence_features = [
            {"sentence_embedding": query},
            {"sentence_embedding": positive},
            {"sentence_embedding": negative},
        ]

        return self.loss_fn(sentence_features, labels=None)
