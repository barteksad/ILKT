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
            name, language=language, date=date, split=split
        )
        self.dataset = self.dataset.shuffle(seed=42)
        self.n_examples = n_examples
        self.max_length = max_length

    def reset(self):
        self.dataset.shuffle(seed=42)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        text = row["text"]

        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.get_data_collator(),
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        )
