from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Optional
from .base import MLMDataset


class WikipediaDataset(MLMDataset):
    def __init__(
        self,
        name: str,
        language: str,
        date: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        split: str,
        max_length: int,
        mlm_probability: float,
        n_examples: Optional[int] = None,
    ):

        dataset = load_dataset(
            name, language=language, date=date, split=split, streaming=True
        )
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        if n_examples is not None:
            dataset = dataset.take(n_examples)
        self.max_length = max_length
        super().__init__(
            name, tokenizer, batch_size, mlm_probability, dataset, n_examples
        )

    def _process_row(self, row: Any) -> Dict[str, Any]:
        text = row["text"]

        return self.tokenizer(text, truncation=True, max_length=self.max_length)
