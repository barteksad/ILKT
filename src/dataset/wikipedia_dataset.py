from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

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
        n_examples: int,
        max_length: int,
        mlm_probability: float,
    ):
        super().__init__(name, tokenizer, batch_size, mlm_probability)

        self.dataset = load_dataset(
            name, language=language, date=date, split=split, streaming=True
        )
        self.dataset = self.dataset.shuffle()
        self.n_examples = n_examples
        self.max_length = max_length

    def reset(self):
        self.dataset = self.dataset.shuffle()
        self.ds_iter = iter(self.dataset)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        if idx == 0:
            self.reset()

        row = next(self.ds_iter)
        text = row["text"]

        return self.tokenizer(text, truncation=True, max_length=self.max_length)
