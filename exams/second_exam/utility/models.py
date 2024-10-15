import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

import plotly.express as px
from PIL import Image
from io import BytesIO
from IPython.display import display

from typing import List, Tuple
from tqdm import tqdm


def compare_models_performance(model_performances: List[pd.DataFrame]):
    """
    Compare the performance of multiple models over epochs.

    Parameters
    ----------
    model_performances : List[pd.DataFrame]
        List of performance DataFrames for multiple models.

    Returns
    -------
    None
    """
    figs = {}

    for col in model_performances[0].columns:
        if col.startswith('train') or col.startswith('valid'):
            fig = px.line(title=f"{col} over Epochs")
            for i, performance in enumerate(model_performances):
                fig.add_scatter(x=performance['epoch'], y=performance[col], name=f'Model {i + 1}')

            figs[col] = Image.open(BytesIO(fig.to_image(format='png')))

    # Images in 3x2 grid
    img_grid = Image.new('RGB', (figs['train_loss'].width * 2, figs['train_loss'].height * 3))
    img_grid.paste(figs['train_loss'], (0, 0))
    img_grid.paste(figs['valid_loss'], (figs['train_loss'].width, 0))
    img_grid.paste(figs['train_accuracy'], (0, figs['train_loss'].height))
    img_grid.paste(figs['valid_accuracy'], (figs['train_loss'].width, figs['train_loss'].height))
    img_grid.paste(figs['train_f1'], (0, figs['train_loss'].height * 2))
    img_grid.paste(figs['valid_f1'], (figs['train_loss'].width, figs['train_loss'].height * 2))

    display(img_grid)


class _EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.

    Attributes
    ----------
    patience : int
        Number of epochs to wait before early stopping.
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    best_score : float
        The score of the best model.
    best_model : nn.Module
        The best model.
    counter : int
        Counter for the number of epochs with no improvement in monitored quantity.
    early_stop : bool
        Whether to stop the training.

    Methods
    -------
    __call__(loss: float, model: nn.Module)
        Update the best model and score.

    reset()
        Reset the best score, best model, counter, and early stop.
    """

    def __init__(self, patience: int = 5, delta: float = 0.0) -> None:
        """
        Initialize the early stopping class.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait before early stopping.
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement.

        Returns
        -------
        None
        """
        self.patience = patience
        self.delta = delta

        self.best_score = None
        self.best_model = None

        self.counter = 0
        self.early_stop = False

    def __call__(self, loss: float, model: nn.Module) -> None:
        """
        Update the best model and score.

        Parameters
        ----------
        loss : float
            The loss of the model.
        model : nn.Module
            The model to update.

        Returns
        -------
        None
        """
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.counter = 0

    def reset(self) -> None:
        """
        Reset the best score, best model, counter, and early stop.

        Returns
        -------
        None
        """
        self.best_score = None
        self.best_model = None
        self.counter = 0
        self.early_stop = False


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model.

    Attributes
    ----------
    layers : nn.Sequential
        The sequential model.
    performance : pd.DataFrame
        The performance of the model over epochs.

    Methods
    -------
    reset_parameters()
        Reset the parameters of the model.
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the model.
    fit(n_epochs: int, train_loader: DataLoader, valid_loader: DataLoader, loss_function: nn.Module,
        optimizer: optim.Optimizer, device: torch.device, early_stop_params: dict = None)
        Fit the model to the data, evaluate it, and save the performance.
    predict(data_loader: DataLoader, device: torch.device) -> List[int]
        Predict the data using the model.
    save_model(folder_path: str, model_name: str)
        Save the model and performance to the folder path.
    load_model(folder_path: str, model_name: str)
        Load the model and performance from the folder path.
    show_performance()
        Show the performance of the model (print and plot).
    """

    def __init__(self, sequential: nn.Sequential, random_seed: int = 0) -> None:
        """
        Initialize the MLP model.

        Parameters
        ----------
        sequential : nn.Sequential
            The sequential model.
        random_seed : int, optional
            The random seed.

        Returns
        -------
        None
        """
        super(CNN, self).__init__()

        self.layers = sequential
        self.performance = None

        torch.manual_seed(random_seed)

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.

        Returns
        -------
        None
        """
        self.performance = None
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.layers(x)

    def __train_epoch(
            self, epoch: int, train_loader: DataLoader, loss_function: nn.Module, optimizer: optim.Optimizer,
            device: torch.device, use_tqdm: bool = True) -> Tuple[float, float, float]:
        """
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        train_loader : DataLoader
            The training data loader.
        loss_function : nn.Module
            The loss function.
        optimizer : optim.Optimizer
            The optimizer.
        device : torch.device
            The device to use.
        use_tqdm : bool.
            Whether to use tqdm.

        Returns
        -------
        Tuple[float, float, float]
            The loss, accuracy, and F1 score.
        """
        self.to(device)
        self.train()

        y_true, y_pred = [], []
        total_loss = 0.0

        if use_tqdm:
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        else:
            pbar = train_loader

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, labels.long())
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        total_loss /= len(train_loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return total_loss, accuracy, f1

    def __eval_epoch(
            self, epoch: int, valid_loader: DataLoader, loss_function: nn.Module, device: torch.device,
            use_tqdm: bool = True) -> Tuple[float, float, float]:
        """
        Evaluate the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        valid_loader : DataLoader
            The valid data loader.
        loss_function : nn.Module
            The loss function.
        device : torch.device
            The device to use.
        use_tqdm : bool.
            Whether to use tqdm.

        Returns
        -------
        Tuple[float, float, float]
            The loss, accuracy, and F1 score.
        """
        self.eval()

        y_true, y_pred = [], []
        total_loss = 0.0

        if use_tqdm:
            pbar = tqdm(valid_loader, desc=f"Evaluating Epoch {epoch}")
        else:
            pbar = valid_loader

        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)

                loss = loss_function(outputs, labels.long())
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        total_loss /= len(valid_loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return total_loss, accuracy, f1

    def fit(self, n_epochs: int, train_loader: DataLoader, valid_loader: DataLoader, loss_function: nn.Module,
            optimizer: optim.Optimizer, device: torch.device, early_stop_params: dict = None) -> None:
        """
        Fit the model to the data, evaluate it, and save the performance.

        Parameters
        ----------
        n_epochs : int
            The number of epochs to train.
        train_loader : DataLoader
            The training data loader.
        valid_loader : DataLoader
            The valid data loader.
        loss_function : nn.Module
            The loss function.
        optimizer : optim.Optimizer
            The optimizer.
        device : torch.device
            The device to use.
        early_stop_params : dict.
            The parameters for early stopping.

        Returns
        -------
        None
        """
        performance = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'valid_loss': [],
            'valid_accuracy': [],
            'valid_f1': [],
        }

        if early_stop_params is not None:
            early_stopping = _EarlyStopping(**early_stop_params)

        for epoch in range(1, n_epochs + 1):
            train_loss, train_accuracy, train_f1 = self.__train_epoch(
                epoch, train_loader, loss_function, optimizer, device, True)
            valid_loss, valid_accuracy, valid_f1 = self.__eval_epoch(
                epoch, valid_loader, loss_function, device, True)

            print(f"Epoch {epoch}/{n_epochs}\n"
                  f"\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}\n"
                  f"\tValid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid F1: {valid_f1:.4f}")

            performance['epoch'].append(epoch)
            performance['train_loss'].append(train_loss)
            performance['train_accuracy'].append(train_accuracy)
            performance['train_f1'].append(train_f1)
            performance['valid_loss'].append(valid_loss)
            performance['valid_accuracy'].append(valid_accuracy)
            performance['valid_f1'].append(valid_f1)

            if early_stop_params is not None:
                early_stopping(valid_loss, self)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\nTraining finished after {epoch} epochs\n"
              f"\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}\n"
              f"\tValid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid F1: {valid_f1:.4f}")

        if early_stop_params is not None:
            self.load_state_dict(early_stopping.best_model.state_dict())

        self.performance = pd.DataFrame(performance)

    def predict(self, data_loader: DataLoader, device: torch.device) -> List[int]:
        """
        Predict the data using the model.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader to predict.
        device : torch.device
            The device to use.

        Returns
        -------
        List[int]
            The predicted labels.
        """
        self.eval()

        y_pred = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)

                outputs = self(inputs)

                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())

        return y_pred

    def validate(self, validation_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
        """
        Validate the model using the validation data.

        Parameters
        ----------
        validation_loader : DataLoader
            The validation data loader.
        device : torch.device
            The device to use.

        Returns
        -------
        Tuple[float, float]
            The accuracy and F1 score.
        """
        self.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return accuracy, f1

    def save(self, folder_path: str, model_name: str) -> None:
        """
        Save the model and performance to the folder path.

        Parameters
        ----------
        folder_path : str
            The folder path to save the model and performance.
        model_name : str
            The name of the model.

        Returns
        -------
        None
        """
        if self.performance is None:
            raise ValueError("Model has not been trained yet.")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.state_dict(), os.path.join(folder_path, model_name + '_model.pt'))
        self.performance.to_csv(os.path.join(folder_path, model_name + '_performance.csv'))

    def load(self, folder_path: str, model_name: str):
        """
        Load the model and performance from the folder path.

        Parameters
        ----------
        folder_path : str
            The folder path to load the model and performance.
        model_name : str
            The name of the model.

        Raises
        ------
        FileNotFoundError
            If the folder path, model file, or performance file does not exist.

        Returns
        -------
        None
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError("Folder path does not exist.")
        if not os.path.exists(os.path.join(folder_path, model_name + '_model.pt')):
            raise FileNotFoundError("Model file does not exist.")
        if not os.path.exists(os.path.join(folder_path, model_name + '_performance.csv')):
            raise FileNotFoundError("Performance file does not exist.")

        self.load_state_dict(torch.load(os.path.join(folder_path, model_name + '_model.pt')))
        self.performance = pd.read_csv(os.path.join(folder_path, model_name + '_performance.csv'), index_col=0)

    def show_performance(self) -> None:
        """
        Show the performance of the model (print and plot).

        Raises
        ------
        ValueError
            If the model has not been trained yet.

        Returns
        -------
        None
        """
        if self.performance is None:
            raise ValueError("Model has not been trained yet.")

        print("=" * 50 + " PERFORMANCE " + "=" * 50)

        # Print best performance
        best_epoch = self.performance['valid_loss'].idxmin()
        best_performance = self.performance.iloc[best_epoch]
        print(f"Best Performance at Epoch {best_performance['epoch']}\n"
              f"\tTrain Loss: {best_performance['train_loss']:.4f}, Train Accuracy: {best_performance['train_accuracy']:.4f}, Train F1: {best_performance['train_f1']:.4f}\n"
              f"\tValid Loss: {best_performance['valid_loss']:.4f}, Valid Accuracy: {best_performance['valid_accuracy']:.4f}, Valid F1: {best_performance['valid_f1']:.4f}")

        # Plot performance
        fig_loss = px.line(
            self.performance,
            x='epoch',
            y=['train_loss', 'valid_loss'],
            labels={'epoch': 'Epoch', 'value': 'Loss', 'variable': 'Data'},
            title='Loss over Epochs',
        )

        fig_acc = px.line(
            self.performance,
            x='epoch',
            y=['train_accuracy', 'valid_accuracy'],
            labels={'epoch': 'Epoch', 'value': 'Accuracy', 'variable': 'Data'},
            title='Accuracy over Epochs',
        )

        fig_f1 = px.line(
            self.performance,
            x='epoch',
            y=['train_f1', 'valid_f1'],
            labels={'epoch': 'Epoch', 'value': 'F1 Score', 'variable': 'Data'},
            title='F1 Score over Epochs',
        )

        # Convert figures to images
        img_loss = Image.open(BytesIO(fig_loss.to_image(format='png')))
        img_acc = Image.open(BytesIO(fig_acc.to_image(format='png')))
        img_f1 = Image.open(BytesIO(fig_f1.to_image(format='png')))
        img_white = Image.new('RGB', (img_loss.width, img_loss.height), (255, 255, 255))

        # Images in 2x2 grid
        img_grid = Image.new('RGB', (img_loss.width * 2, img_loss.height * 2))
        img_grid.paste(img_loss, (0, 0))
        img_grid.paste(img_acc, (img_loss.width, 0))
        img_grid.paste(img_f1, (0, img_loss.height))
        img_grid.paste(img_white, (img_loss.width, img_loss.height))

        display(img_grid)
