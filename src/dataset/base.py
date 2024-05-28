from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from losses import CosineSimilarityLoss

# Pairs: ["text1", "text2"] - This is a positive pair that should be close in vector space.
# Triplets: ["anchor", "positive", "negative"] - This is a triplet: The positive text should be close to the anchor, while the negative text should be distant to the anchor.
# Sets: {"set": ["text1", "text2", ...]} A set of texts describing the same thing, e.g. different paraphrases of the same question, different captions for the same image. Any combination of the elements is considered as a positive pair.
# Query-Pairs: {"query": "text", "pos": ["text1", "text2", ...]} A query together with a set of positive texts. Can be formed to a pair ["query", "positive"] by randomly selecting a text from pos.
# Query-Triplets: {"query": "text", "pos": ["text1", "text2", ...], "neg": ["text1", "text2", ...]} A query together with a set of positive texts and negative texts. Can be formed to a triplet ["query", "positive", "negative"] by randomly selecting a text from pos and neg.


class TextDataset(ABC, Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @abstractmethod
    def get_data_collator(self) -> Any:
        raise NotImplementedError

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.get_data_collator(),
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            shuffle=True
        )


class ContrastiveDataset(TextDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        loss_fn=CosineSimilarityLoss(),
    ):
        super().__init__(tokenizer, batch_size)
        self.loss_fn = loss_fn

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_data_collator(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def format_for_loss_fn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MLMDataset(TextDataset):

    def __init__(
        self, tokenizer: PreTrainedTokenizer, batch_size: int, mlm_probability: float
    ):
        super().__init__(tokenizer, batch_size)
        self.mlm_probability = mlm_probability

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_data_collator(self) -> Any:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=16,
        )
        return data_collator
