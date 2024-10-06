import pandas as pd
from typing import List, Tuple
import os

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader


class Data:
    """
    Class for creating DataLoader objects.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame to be used for creating DataLoader objects.
    batch_size : int
        Batch size.
    path : str
        Path to the folder where the DataLoader objects are stored.
    loaders : dict[str, DataLoader]
        DataLoader objects for training, testing, and validation.

    Public Methods
    --------------
    None
    """

    def __init__(
            self, df: pd.DataFrame, target: str, train_size: float, test_size: float, validation_size: float = 0,
            batch_size: int = 64, path: str = None):
        """
        Initialize the Data class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be used for creating DataLoader objects.
        target : str
            Target column.
        train_size : float
            Proportion of training data.
        test_size : float
            Proportion of testing data.
        validation_size : float
            Proportion of validation data.
        batch_size : int
            Batch size.
        path : str
            Path to the folder where the DataLoader objects are stored.

        Raises
        ------
        ValueError
            If sum of sizes is not equal to 1 or if sizes are negative.

        Returns
        -------
        None
        """
        self.df = df
        self.batch_size = batch_size
        self.__target = target

        if abs(train_size + test_size + validation_size - 1) > 1e-5:
            raise ValueError('Sum of sizes must be equal to 1')
        elif train_size < 0 or test_size < 0 or (validation_size is not None and validation_size < 0):
            raise ValueError('Sizes must be positive')
        self.__train_size = int(train_size * len(df))
        self.__test_size = int(test_size * len(df))
        self.__validation_size = len(df) - self.__train_size - self.__test_size

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.__name = self.__get_name()
        self.loaders, from_folder = self.__load()
        if not from_folder:
            self.__save()

    def __get_name(self) -> str:
        """
        Get the name of the DataLoader object to be saved (using all the parameters).

        Returns
        -------
        str
            Name of the DataLoader object.
        """
        name = ''
        name += f'target={self.__target}_'
        name += f'train={self.__train_size}_'
        name += f'test={self.__test_size}_'
        name += f'validation={self.__validation_size}_'
        name += f'batch={self.batch_size}'

        return name

    def __get_loaders(self) -> dict[str, DataLoader]:
        """
        Get DataLoader objects from the DataFrame.

        Returns
        -------
        dict[str, DataLoader]
            DataLoader objects of the DataFrame for training, testing, and validation.
        """
        features = torch.tensor(self.df.drop(columns=[self.__target]).values).float()
        target = torch.tensor(self.df[self.__target].values).float()
        dataset = TensorDataset(features, target)

        train_dataset, test_dataset, validation_dataset = random_split(
            dataset, [self.__train_size, self.__test_size, self.__validation_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        return {'train': train_loader, 'test': test_loader, 'validation': validation_loader}

    def __get_loaders_from_folder(self) -> dict[str, DataLoader]:
        """
        Get DataLoader objects from the folder.

        Returns
        -------
        dict[str, DataLoader]
            DataLoader objects of the DataFrame for training, testing, and validation.
        """
        if self.path is None:
            raise ValueError('Path is not defined')

        loaders = {'train': None, 'test': None, 'validation': None}
        for key in loaders.keys():
            loader_name = f'{self.__name}_type={key}.pt'
            loaders[key] = torch.load(os.path.join(self.path, loader_name))

        return loaders

    def __load(self) -> Tuple[dict[str, DataLoader], bool]:
        """
        Load DataLoader objects.

        Returns
        -------
        Tuple[dict[str, DataLoader], bool]
            Tuple of DataLoader objects and boolean indicating if DataLoader objects were loaded from folder.
        """
        try:
            return self.__get_loaders_from_folder(), True
        except (FileNotFoundError, ValueError):
            return self.__get_loaders(), False

    def __save(self) -> None:
        """
        Save DataLoader objects into the folder.

        Raises
        ------
        ValueError
            If path is not defined.

        Returns
        -------
        None
        """
        if self.path is None:
            raise ValueError('Path is not defined')

        for key, loader in self.loaders.items():
            loader_name = f'{self.__name}_type={key}.pt'
            torch.save(loader, os.path.join(self.path, loader_name))
