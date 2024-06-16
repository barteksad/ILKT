import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from typing import Any, Dict, Optional

from .base import ContrastiveDataset


class PairDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        loss_fn: torch.nn.Module,
        query_column: str,
        answer_column: str,
        subset: Optional[str] = None,
        n_examples: Optional[int] = None,
        **kwargs,
    ):
        dataset = load_dataset(name, subset, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        if n_examples is not None:
            dataset = dataset.take(n_examples)
            self.n_examples = n_examples
        self.max_length = max_length
        self.loss_fn = loss_fn
        self.query_column = query_column
        self.answer_column = answer_column
        super().__init__(name, tokenizer, batch_size, dataset)

    def _process_row(self, row: Any) -> Dict[str, Any]:

        query = self.tokenizer(
            row[self.query_column], truncation=True, max_length=self.max_length
        )
        answer = self.tokenizer(
            row[self.answer_column], truncation=True, max_length=self.max_length
        )

        return {"query": query, "answer": answer}

    def get_data_collator(self):

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def _collate_df(batch):

            query = data_collator([x["query"] for x in batch])
            answer = data_collator([x["answer"] for x in batch])

            return {"model_inputs": (query, answer)}

        return _collate_df

    def get_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        query, answer = batch["model_outputs"]

        sentence_features = [
            {"sentence_embedding": query},
            {"sentence_embedding": answer},
        ]

        return self.loss_fn(sentence_features, labels=None)

    def get_evaluator(self):
        return BinaryClassificationEvaluator(
            sentences1=[i[self.query_column] for i in self.dataset],
            sentences2=[i[self.answer_column] for i in self.dataset],
            labels=[1] * self.n_examples,  # this is only set for validation datasets
            name=self.name,
        )
