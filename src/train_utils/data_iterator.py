from typing import List, Iterable

from train_utils.train_util import DL_TYPE


class DataIterator(Iterable):

    def __iter__(self):
        raise NotImplementedError()

    def __init__(self, dataloaders: List[DL_TYPE]):
        self.dataloaders = dataloaders


class FullValidIterator(DataIterator):
    def __iter__(self):
        for idx, dataloader in enumerate(self.dataloaders):
            for batch in dataloader:
                yield batch, dataloader


class SingleBatchPerDatasetIterator(DataIterator):
    def __init__(self, dataloaders: List[DL_TYPE]):
        super().__init__(dataloaders)
        self.iter_dataloaders = [iter(dataloader) for dataloader in dataloaders]

    def __iter__(self):
        for idx, dataloader in enumerate(self.dataloaders):
            try:
                batch = next(self.iter_dataloaders[idx])  # type: ignore
            except:
                self.iter_dataloaders[idx] = iter(self.dataloaders[idx])  # type: ignore
                batch = next(self.iter_dataloaders[idx])  # type: ignore
            yield batch, dataloader
