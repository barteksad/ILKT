from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Any, List, Optional, Dict
from .base import ClassificationDataset

class SentenceClassificationDataset(ClassificationDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        split: str,
        max_length: int,
        n_classes: int,
        sentence_keys: List[str],
        label_key: str,
        label_mapping: Optional[Dict[str, int]] = None,
        n_examples: Optional[int] = None,
    ):

        dataset = load_dataset(name, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        if n_examples is not None:
            dataset = dataset.take(n_examples)
        dataset = dataset.with_format("torch")
        self.max_length = max_length
        self.label_mapping = label_mapping
        super().__init__(
            name,
            tokenizer,
            batch_size,
            n_classes,
            sentence_keys,
            label_key,
            dataset,
        )

    def _process_row(self, row: Any) -> Dict[str, Any]:

        texts = [row[key] for key in self.sentence_keys]
        labels = row[self.label_key]
        if self.label_mapping is not None:
            labels = self.label_mapping[labels]

        result_dict = self.tokenizer(
            *texts,
            truncation="longest_first",
            max_length=self.max_length,
        )
        result_dict["labels"] = labels

        return result_dict
