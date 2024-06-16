from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Optional
from .base import MLMDataset


class OscarDataset(MLMDataset):
    def __init__(
        self,
        name: str,
        language: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        split: str,
        max_length: int,
        mlm_probability: float,
        text_column: str = "text",
        n_examples: Optional[int] = None,
    ):
        if language != "":
            dataset = load_dataset(
                name,
                language,
                split=split,
                streaming=True,
                token=True,
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                name, split=split, streaming=True, token=True, trust_remote_code=True
            )
        dataset = dataset.shuffle(buffer_size=10_000, seed=42)
        if n_examples is not None:
            dataset = dataset.take(n_examples)
        self.n_examples = n_examples
        self.max_length = max_length
        self.text_column = text_column
        super().__init__(name, tokenizer, batch_size, mlm_probability, dataset)

    def _process_row(self, row: Any) -> Dict[str, Any]:
        text = row[self.text_column]

        return self.tokenizer(text, truncation=True, max_length=self.max_length)
