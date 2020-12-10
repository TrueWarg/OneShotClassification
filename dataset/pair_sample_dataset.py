from torch.utils.data import Dataset
from typing import Tuple
import torch
import cv2
import numpy as np


class PairMaker:
    def __init__(self, filepath: str):
        self._filepath = filepath

    def make_pairs(self) -> Tuple:
        pass


class Transformer:
    def transform(self, image: np.ndarray) -> np.ndarray:
        pass


def _read_image(filepath: str) -> np.ndarray:
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class PairSampleTrainDataset(Dataset):
    def __init__(self, pair_maker: PairMaker, transformer: Transformer = None):
        super.__init__()
        self._pairs = pair_maker.make_pairs()
        self._transformer = transformer

    def __getitem__(self, index: int) -> Tuple:
        image_path_1, class_id_1, image_path_2, class_id_2 = self._pairs[index]
        image_1 = _read_image(image_path_1)
        image_2 = _read_image(image_path_2)

        if self._transformer:
            image_1 = self._transformer.transform(image_1)
            image_2 = self._transformer.transform(image_2)

        image_1 = torch.from_numpy(image_1)
        image_2 = torch.from_numpy(image_2)

        return image_1, class_id_1, image_2, class_id_2

    def __len__(self) -> int:
        return len(self._pairs)
