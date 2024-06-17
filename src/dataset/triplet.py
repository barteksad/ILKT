import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from sentence_transformers.evaluation import TripletEvaluator
from typing import Any, Dict, Optional

from .base import ContrastiveDataset


class TripletDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        query_col_name: str,
        positive_col_name: str,
        negative_col_name: str,
        subset: str = None,
        n_examples: Optional[int] = None,
        **kwargs,
    ):

        dataset = load_dataset(name, subset, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        if n_examples is not None:
            dataset = dataset.take(n_examples)
        dataset = dataset.with_format("torch")
        self.max_length = max_length
        self.loss_fn = loss_fn
        self.query_col_name = query_col_name
        self.positive_col_name = positive_col_name
        self.negative_col_name = negative_col_name
        super().__init__(name, tokenizer, batch_size, dataset, n_examples)

    def _process_row(self, row: Any) -> Dict[str, Any]:

        query = self.tokenizer(
            row[self.query_col_name], truncation=True, max_length=self.max_length
        )
        positive = self.tokenizer(
            row[self.positive_col_name], truncation=True, max_length=self.max_length
        )
        negative = self.tokenizer(
            row[self.negative_col_name], truncation=True, max_length=self.max_length
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

    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        query, positive, negative = batch["model_outputs"]

        sentence_features = [
            {"sentence_embedding": query},
            {"sentence_embedding": positive},
            {"sentence_embedding": negative},
        ]

        return self.loss_fn(sentence_features, labels=None)

    def get_evaluator(self):
        return TripletEvaluator(
            anchors=[i[self.query_col_name] for i in self.dataset],
            positives=[i[self.positive_col_name] for i in self.dataset],
            negatives=[i[self.negative_col_name] for i in self.dataset],
            name=self.name,
        )
