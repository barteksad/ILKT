from random import sample

import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import Any, Dict

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
        super().__init__(name, tokenizer, batch_size)

        self.dataset = load_dataset(name, split)["train"]  # type: ignore
        if isinstance(use_rows, float):
            use_rows = int(use_rows * len(self.dataset))
        self.dataset = self.dataset.shuffle(seed=42)  # type: ignore
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

            return {"model_inputs": inputs, "labels": labels}

        return _collate_df

    def format_for_loss_fn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # TODO this set dataset has always two pairs so we are not missing anything here but this is not generic
        outputs = batch["model_outputs"]
        labels = batch["labels"]

        mask1 = labels[:-1] == labels[1:]
        mask2 = labels[1:] == labels[:-1]

        outputs1 = outputs[:-1][mask1 == 1, :]
        outputs2 = outputs[1:][mask2 == 1, :]

        return {
            "output1": outputs1,
            "output2": outputs2,
            "target": mask1[mask1 == 1].float().unsqueeze(-1),
        }
