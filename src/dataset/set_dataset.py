from random import sample

import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .base import ContrastiveDataset


class SetDataset(ContrastiveDataset):

    def __init__(
        self,
        name: str,
        split: str,
        use_rows: int | float,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        **kwargs
    ):

        super().__init__(tokenizer, batch_size)

        self.dataset = load_dataset(name, split)["train"]
        if isinstance(use_rows, float):
            use_rows = int(use_rows * len(self.dataset))
        self.dataset = self.dataset.shuffle(seed=42)
        self.dataset = self.dataset.select(range(use_rows))
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]

        return_set = [
            self.tokenizer(sentence, truncation=True, max_length=self.max_length)
            for sentence in row["set"]
        ]

        return {"set": return_set, "labels": [idx] * len(row["set"])}

    def get_data_collator(self):

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def _collate_df(batch):
            labels = torch.tensor(
                sum([example["labels"] for example in batch], []), dtype=torch.long
            )

            inputs = data_collator(sum([example["set"] for example in batch], []))

            return {"set": inputs, "labels": labels}

        return _collate_df
