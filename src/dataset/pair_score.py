import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import Any, Dict

from .base import ContrastiveDataset

class PairScoreDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        n_examples: int,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        type: str = "pair-score",
        **kwargs
    ):
        super().__init__(name, tokenizer, batch_size)

        self.dataset = load_dataset(name, type, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)  # type: ignore
        self.n_examples = n_examples
        self.max_length = max_length
        self.loss_fn = loss_fn
        
    def reset(self):
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)
        self.ds_iter = iter(self.dataset)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        if idx == 0:
            self.reset()

        row = next(self.ds_iter)

        sentence1 = self.tokenizer(
            row["sentence1"], truncation=True, max_length=self.max_length
        )
        sentence2 = self.tokenizer(
            row["sentence2"], truncation=True, max_length=self.max_length
        )

        return {"sentence1": sentence1, "sentence2": sentence2, "labels": row["score"]}

    def get_data_collator(self):

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def _collate_df(batch):

            sentence1 = data_collator([x["sentence1"] for x in batch])
            sentence2 = data_collator([x["sentence2"] for x in batch])
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.float)

            return {"model_inputs": (sentence1, sentence2), "labels": labels}

        return _collate_df

    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        sentence1, sentence2 = batch["model_outputs"]
        labels = batch["labels"]

        sentence_features = [
            {"sentence_embedding": sentence1},
            {"sentence_embedding": sentence2},
        ]

        return self.loss_fn(sentence_features, labels)
