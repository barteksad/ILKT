from .base import MLMDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class WikipediaDataset(MLMDataset):
    def __init__(
        self,
        name: str,
        language: str,
        date: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        streaming: bool,
        split: str,
        n_examples: int,
        max_length: int,
        mlm_probability: float,
    ):
        super().__init__(tokenizer, batch_size, mlm_probability)

        self.dataset = load_dataset(
            name, language=language, date=date, split=split, streaming=streaming
        )
        self.dataset = self.dataset.shuffle()
        self.n_examples = n_examples
        self.max_length = max_length

    def reset(self):
        self.ds_iter = iter(self.dataset)

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        if idx == 0:
            self.reset()

        row = next(self.ds_iter)
        text = row["text"]

        return self.tokenizer(text, truncation=True, max_length=self.max_length)
