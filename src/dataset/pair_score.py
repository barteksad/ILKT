import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from typing import Any, Dict, Optional

from .base import ContrastiveDataset


class PairScoreDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        type: str = "pair-score",
        n_examples: Optional[int] = None,
        **kwargs,
    ):
        dataset = load_dataset(name, type, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)  # type: ignore
        if n_examples is not None:
            dataset = dataset.take(n_examples)
        self.max_length = max_length
        self.loss_fn = loss_fn
        super().__init__(name, tokenizer, batch_size, dataset, n_examples)

    def _process_row(self, row: Any) -> Dict[str, Any]:

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

    def get_evaluator(self):
        return EmbeddingSimilarityEvaluator(
            sentences1=[i["sentence1"] for i in self.dataset],
            sentences2=[i["sentence2"] for i in self.dataset],
            scores=[i["score"] for i in self.dataset],
            main_similarity=SimilarityFunction.COSINE,
            name=self.name,
        )
