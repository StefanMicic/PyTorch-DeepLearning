from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import calculate_metrics, print_metrics


class Classifier(nn.Module):
    """ Artificial neural network that classify images. """

    def __init__(self, D_in: int, H1: int, H2: int, D_out: int) -> None:
        """Init method.

        Args:
            D_in (int): Input dimension
            H1 (int): Number of nodes in first hidden layer
            H2 (int) : Number of nodes in second hidden layer
            D_out (int) : Output dimension
        """
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        self.criterion = nn.CrossEntropyLoss()
        (
            self.training_loader,
            self.validation_loader,
        ) = self.create_dataset_loaders()
        self.running_loss = 0.0
        self.running_corrects = 0.0
        self.val_running_loss = 0.0
        self.val_running_corrects = 0.0

    def create_dataset_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates data loaders for training and validation.

        Returns:
            Tuple[DataLoader, DataLoader]:
                    Training and validation dataset loaders
        """
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        training_dataset = datasets.MNIST(
            root="../data", train=True, download=True, transform=transform
        )
        validation_dataset = datasets.MNIST(
            root="../data", train=False, download=True, transform=transform
        )

        training_loader = DataLoader(
            training_dataset, batch_size=100, shuffle=True
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=100, shuffle=False
        )
        return training_loader, validation_loader

    def forward(self, x: float) -> float:
        """Forward pass in neural network.
        Args:
            x (float): Input feature for forward pass
        Returns:
            float: Result of forward pass
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def batch_iteration(self, optimizer: Adam) -> None:
        for inputs, labels in self.training_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            self.running_loss += loss.item()
            self.running_corrects += torch.sum(preds == labels.data)

    def validation_iteration(self) -> None:
        """Goes through validation set and predicts."""
        for val_inputs, val_labels in self.validation_loader:
            val_inputs = val_inputs.view(val_inputs.shape[0], -1)
            val_outputs = self(val_inputs)
            val_loss = self.criterion(val_outputs, val_labels)

            _, val_preds = torch.max(val_outputs, 1)
            self.val_running_loss += val_loss.item()
            self.val_running_corrects += torch.sum(
                val_preds == val_labels.data
            )

    def train(self) -> None:
        """Trains neural network."""

        optimizer = Adam(self.parameters(), lr=0.0001)

        epochs = 15

        for epoch_num in range(epochs):

            self.batch_iteration(optimizer)

            with torch.no_grad():
                self.validation_iteration()
                (epoch_loss, epoch_acc) = calculate_metrics(
                    self.running_loss,
                    self.running_corrects,
                    self.training_loader,
                )
                (val_epoch_loss, val_epoch_acc) = calculate_metrics(
                    self.val_running_loss,
                    self.val_running_corrects,
                    self.validation_loader,
                )
                print_metrics(
                    epoch_num,
                    epoch_loss,
                    epoch_acc,
                    val_epoch_loss,
                    val_epoch_acc,
                )


def main():
    model = Classifier(784, 125, 65, 10)
    model.train()


if __name__ == "__main__":
    main()
