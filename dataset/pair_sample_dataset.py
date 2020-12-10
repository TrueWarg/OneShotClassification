from torch.utils.data import Dataset
from typing import Tuple
import torch


class PairMaker:

    def __init__(self, filepath: str):
        self._filepath = filepath

    def make_pairs(self) -> Tuple:
        pass


class Transformer:
    def transform(self) -> torch.Tensor:
        pass


class PairSampleTrainDataset(Dataset):
    def __init__(self, pair_maker: PairMaker, transformer: Transformer):
        super.__init__()
        self._pairs = pair_maker.make_pairs()
        self._transformer = transformer

    def __getitem__(self, index: int) -> torch.Tensor:
        pass

    def __len__(self) -> int:
        return len(self._pairs)
