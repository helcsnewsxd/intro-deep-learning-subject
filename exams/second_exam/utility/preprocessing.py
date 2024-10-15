import os
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision as tv
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from io import BytesIO
from IPython.display import display


class Data:
    """
    Data class to load datasets and dataloaders.

    Attributes
    ----------
    path : str
        Path to the dataset.
    train_transform : v2.Compose
        Transform to apply to the train dataset.
    test_transform : v2.Compose
        Transform to apply to the test and valid datasets.
    batch_size : int
        Batch size.
    do_sampling : bool
        If True, do sampling. 
    random_seed : int
        Random seed.
    datasets : Dict[str, ImageFolder]
        Datasets.
    loaders : Dict[str, DataLoader]
        Dataloaders.
    """

    def __init__(
            self, path: str, train_transform: v2.Compose = None, test_transform: v2.Compose = None, batch_size: int = 64,
            do_sampling: bool = False, random_seed: int = None) -> None:
        """
        Initialize Data class.

        Parameters
        ----------
        path : str
            Path to the dataset.
        train_transform : v2.Compose, optional
            Transform to apply to the train dataset, by default None.
        test_transform : v2.Compose, optional
            Transform to apply to the test and valid datasets, by default None.
        batch_size : int, optional
            Batch size, by default 64.
        do_sampling : bool, optional
            If True, do sampling, by default False.
        random_seed : int, optional
            Random seed, by default None.

        Raises
        ------
        FileNotFoundError
            If the path doesn't exist.

        Returns
        -------
        None
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} doesn't exist.")

        self.path = path
        self.batch_size = batch_size
        self.do_sampling = do_sampling

        self.train_transform = train_transform if train_transform is not None \
            else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.test_transform = test_transform if test_transform is not None \
            else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.random_seed = random_seed
        if random_seed:
            torch.manual_seed(random_seed)

        self.datasets, self.loaders = self.__load()

    def __load(self) -> Tuple[Dict[str, ImageFolder], Dict[str, DataLoader]]:
        """
        Load datasets and dataloaders.

        Returns
        -------
        Tuple[Dict[str, ImageFolder], Dict[str, DataLoader]]
            Datasets and dataloaders.
        """
        datasets = {
            'train': ImageFolder(root=os.path.join(self.path, 'train'), transform=self.train_transform),
            'valid': ImageFolder(root=os.path.join(self.path, 'valid'), transform=self.test_transform),
            'test': ImageFolder(root=os.path.join(self.path, 'test'), transform=self.test_transform)
        }

        loaders = {
            'valid': DataLoader(datasets['valid'], batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(datasets['test'], batch_size=self.batch_size, shuffle=False)
        }

        if self.do_sampling:
            class_sample_count = [0] * len(datasets['train'].classes)
            for _, target in datasets['train']:
                class_sample_count[target] += 1

            weights = len(datasets['train']) / torch.tensor(class_sample_count, dtype=torch.float)
            sample_weights = [weights[target] for _, target in datasets['train']]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(datasets['train']), replacement=True)

            loaders['train'] = DataLoader(datasets['train'], batch_size=self.batch_size, sampler=sampler)
        else:
            loaders['train'] = DataLoader(datasets['train'], batch_size=self.batch_size, shuffle=True)

        return datasets, loaders

    def get_metrics(self, type: str) -> Tuple[int, int]:
        """
        Get mean and standard deviation of the dataset.

        Parameters
        ----------
        type : str
            Type of the dataset.

        Returns
        -------
        Tuple[int, int]
            Mean and standard deviation of the dataset.
        """
        if type not in self.loaders.keys():
            raise ValueError(f"DataLoader type {type} doesn't exist.")

        mean = 0.0
        std = 0.0
        for img, _ in self.datasets[type]:
            mean += img.mean(dim=[1, 2])
            std += img.std(dim=[1, 2])
        mean /= len(self.datasets[type])
        std /= len(self.datasets[type])

        return mean, std

    def show_distribution(self) -> None:
        """
        Show distribution of the classes in the dataloaders.

        Returns
        -------
        None
        """
        imgs = {}
        targets = set()
        for type in self.loaders.keys():
            classes = [0] * len(self.loaders[type].dataset.classes)
            for batch in self.loaders[type]:
                for target in batch[1]:
                    classes[target] += 1
                    targets.add(target)

            fig = px.bar(x=self.loaders[type].dataset.classes, y=classes, title=f'Distribution of {type} DataLoader',
                         labels={'x': 'Classes', 'y': 'Count'})
            imgs[type] = Image.open(BytesIO(fig.to_image(format='png')))
        imgs['white'] = Image.new('RGB', (imgs['train'].width, imgs['train'].height), (255, 255, 255))

        # Show images in 2 x 2 grid
        img_grid = Image.new('RGB', (2 * imgs['train'].width, 2 * imgs['train'].height))
        img_grid.paste(imgs['train'], (0, 0))
        img_grid.paste(imgs['valid'], (imgs['train'].width, 0))
        img_grid.paste(imgs['test'], (0, imgs['train'].height))
        img_grid.paste(imgs['white'], (imgs['train'].width, imgs['train'].height))

        display(img_grid)

    def show_first_batch(self, type: str) -> None:
        """
        Show first batch of the DataLoader.

        Parameters
        ----------
        type : str
            Type of the DataLoader.

        Returns
        -------
        None
        """
        if type not in self.loaders.keys():
            raise ValueError(f"DataLoader type {type} doesn't exist.")

        for batch in self.loaders[type]:
            plt.figure(figsize=(16, 16))
            plt.title(f"First batch of {type} DataLoader")
            plt.axis('off')
            plt.imshow(tv.utils.make_grid(batch[0], nrow=8).permute(1, 2, 0))
            break

        plt.show()
