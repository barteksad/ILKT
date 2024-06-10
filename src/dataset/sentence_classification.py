from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict
from .base import ClassificationDataset


class SentenceClassificationDataset(ClassificationDataset):

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        split: str,
        n_examples: int,
        max_length: int,
        n_classes: int,
        sentence_keys: List[str],
        label_key: str,
        label_mapping: Optional[Dict[str, int]] = None,
    ):

        super().__init__(
            name, tokenizer, batch_size, n_classes, sentence_keys, label_key
        )
        self.dataset = load_dataset(name, split=split, streaming=True)
        self.dataset = self.dataset.shuffle()
        self.n_examples = n_examples
        self.max_length = max_length
        self.label_mapping = label_mapping

    def reset(self):
        self.dataset = self.dataset.shuffle()
        self.ds_iter = iter(self.dataset)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        if idx == 0:
            self.reset()

        row = next(self.ds_iter)

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
