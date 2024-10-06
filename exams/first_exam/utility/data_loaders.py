import pandas as pd
from typing import List, Tuple
import os

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader


class Data:
    def __init__(
            self, df: pd.DataFrame, target: str, train_size: float, test_size: float, validation_size: float = 0,
            batch_size: int = 64, path: str = None):
        self.df = df
        self.batch_size = batch_size
        self._target = target

        if abs(train_size + test_size + validation_size - 1) > 1e-5:
            raise ValueError('Sum of sizes must be equal to 1')
        self._train_size = int(train_size * len(df))
        self._test_size = int(test_size * len(df))
        self._validation_size = len(df) - self._train_size - self._test_size

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self._name = self._get_name()
        self.loaders, from_folder = self._load()
        if not from_folder:
            self._save()

    def _get_name(self) -> str:
        name = ''
        name += f'target={self._target}_'
        name += f'train={self._train_size}_'
        name += f'test={self._test_size}_'
        name += f'validation={self._validation_size}_'
        name += f'batch={self.batch_size}'

        return name

    def _get_loaders(self) -> dict[str, DataLoader]:
        features = torch.tensor(self.df.drop(columns=[self._target]).values).float()
        target = torch.tensor(self.df[self._target].values).float()
        dataset = TensorDataset(features, target)

        train_dataset, test_dataset, validation_dataset = random_split(
            dataset, [self._train_size, self._test_size, self._validation_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        return {'train': train_loader, 'test': test_loader, 'validation': validation_loader}

    def _get_loaders_from_folder(self) -> dict[str, DataLoader]:
        if self.path is None:
            raise ValueError('Path is not defined')

        loaders = {'train': None, 'test': None, 'validation': None}
        for key in loaders.keys():
            loader_name = f'{self._name}_type={key}.pt'
            loaders[key] = torch.load(os.path.join(self.path, loader_name))

        return loaders

    def _load(self) -> Tuple[dict[str, DataLoader], bool]:
        try:
            return self._get_loaders_from_folder(), True
        except (FileNotFoundError, ValueError):
            return self._get_loaders(), False

    def _save(self) -> None:
        if self.path is None:
            raise ValueError('Path is not defined')

        for key, loader in self.loaders.items():
            loader_name = f'{self._name}_type={key}.pt'
            torch.save(loader, os.path.join(self.path, loader_name))
