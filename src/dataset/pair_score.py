import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import Any, Dict

from .base import ContrastiveDataset

torch.bfloat16
class PairScoreDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        use_rows: int | float,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        **kwargs
    ):
        super().__init__(name, tokenizer, batch_size)

        self.dataset = load_dataset(name, "pair-score", split=split)
        if isinstance(use_rows, float):
            use_rows = int(use_rows * len(self.dataset))  # type: ignore
        self.dataset = self.dataset.shuffle(seed=42)  # type: ignore
        self.dataset = self.dataset.select(range(use_rows))  # type: ignore
        self.max_length = max_length
        self.loss_fn = loss_fn

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int):
        row = self.dataset[idx]  # type: ignore

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
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.float32)

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
