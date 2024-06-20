from .base import ContrastiveDataset, MLMDataset
from .pair_score import PairScoreDataset
from .wikipedia_dataset import WikipediaDataset
from .triplet import TripletDataset
from .sentence_classification import SentenceClassificationDataset
from .pair import PairDataset
from .oscar_dataset import OscarDataset

__all__ = [
    "ContrastiveDataset",
    "MLMDataset",
    "PairScoreDataset",
    "WikipediaDataset",
    "TripletDataset",
    "SentenceClassificationDataset",
    "PairDataset",
    "OscarDataset",
]
