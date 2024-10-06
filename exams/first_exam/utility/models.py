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


class MLP(nn.Module):
    """
    Multi-layer perceptron model.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the model.
    activation_function : nn.Module
        The activation function to use.
    dropout : nn.Dropout
        The dropout layer.
    performance : pd.DataFrame
        The performance of the model over epochs.

    Methods
    -------
    reset_parameters()
        Reset the parameters of the model.
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the model.
    fit(n_epochs: int, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module,
        optimizer: optim.Optimizer, device: torch.device, early_stop_params: dict = None)
        Fit the model to the data, evaluate it, and save the performance.
    save_model(folder_path: str, model_name: str)
        Save the model and performance to the folder path.
    load_model(folder_path: str, model_name: str)
        Load the model and performance from the folder path.
    show_performance()
        Show the performance of the model (print and plot).
    """

    def __init__(
            self, input_size: int, hidden_size: List[int],
            output_size: int, activation_function: nn.Module = nn.ReLU(),
            dropout: float = 0.0) -> None:
        """
        Initialize the MLP model.

        Parameters
        ----------
        input_size : int
            The size of the input.
        hidden_size : List[int]
            The size of the hidden layers.
        output_size : int
            The size of the output.
        activation_function : nn.Module.
            The activation function to use.
        dropout : float, optional
            The dropout rate.

        Returns
        -------
        None
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.layers.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

        self.activation_function = activation_function
        self.dropout = nn.Dropout(dropout)

        self.performance = None

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
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

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
            outputs = self(inputs.view(inputs.shape[0], -1))

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
            self, epoch: int, test_loader: DataLoader, loss_function: nn.Module, device: torch.device,
            use_tqdm: bool = True) -> Tuple[float, float, float]:
        """
        Evaluate the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        test_loader : DataLoader
            The test data loader.
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
            pbar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}")
        else:
            pbar = test_loader

        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs.view(inputs.shape[0], -1))

                loss = loss_function(outputs, labels.long())
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        total_loss /= len(test_loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return total_loss, accuracy, f1

    def fit(self, n_epochs: int, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module,
            optimizer: optim.Optimizer, device: torch.device, early_stop_params: dict = None) -> None:
        """
        Fit the model to the data, evaluate it, and save the performance.

        Parameters
        ----------
        n_epochs : int
            The number of epochs to train.
        train_loader : DataLoader
            The training data loader.
        test_loader : DataLoader
            The test data loader.
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
        self.reset_parameters()

        performance = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_f1': [],
        }

        if early_stop_params is not None:
            early_stopping = _EarlyStopping(**early_stop_params)

        for epoch in range(1, n_epochs + 1):
            print_data = (epoch % 10 == 1)

            train_loss, train_accuracy, train_f1 = self.__train_epoch(
                epoch, train_loader, loss_function, optimizer, device, print_data)
            test_loss, test_accuracy, test_f1 = self.__eval_epoch(epoch, test_loader, loss_function, device, print_data)

            if print_data:
                print(f"Epoch {epoch}/{n_epochs}\n"
                      f"\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}\n"
                      f"\tTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

            performance['epoch'].append(epoch)
            performance['train_loss'].append(train_loss)
            performance['train_accuracy'].append(train_accuracy)
            performance['train_f1'].append(train_f1)
            performance['test_loss'].append(test_loss)
            performance['test_accuracy'].append(test_accuracy)
            performance['test_f1'].append(test_f1)

            if early_stop_params is not None:
                early_stopping(test_loss, self)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\nTraining finished after {epoch} epochs\n"
              f"\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}\n"
              f"\tTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

        if early_stop_params is not None:
            self.load_state_dict(early_stopping.best_model.state_dict())

        self.performance = pd.DataFrame(performance)

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
        self.performance = pd.read_csv(os.path.join(folder_path, model_name + '_performance.csv'))

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
        best_epoch = self.performance['test_loss'].idxmin()
        best_performance = self.performance.iloc[best_epoch]
        print(f"Best Performance at Epoch {best_performance['epoch']}\n"
              f"\tTrain Loss: {best_performance['train_loss']:.4f}, Train Accuracy: {best_performance['train_accuracy']:.4f}, Train F1: {best_performance['train_f1']:.4f}\n"
              f"\tTest Loss: {best_performance['test_loss']:.4f}, Test Accuracy: {best_performance['test_accuracy']:.4f}, Test F1: {best_performance['test_f1']:.4f}")

        # Plot performance
        fig_loss = px.line(
            self.performance,
            x='epoch',
            y=['train_loss', 'test_loss'],
            labels={'epoch': 'Epoch', 'value': 'Loss', 'variable': 'Data'},
            title='Loss over Epochs',
        )

        fig_acc = px.line(
            self.performance,
            x='epoch',
            y=['train_accuracy', 'test_accuracy'],
            labels={'epoch': 'Epoch', 'value': 'Accuracy', 'variable': 'Data'},
            title='Accuracy over Epochs',
        )

        fig_f1 = px.line(
            self.performance,
            x='epoch',
            y=['train_f1', 'test_f1'],
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
