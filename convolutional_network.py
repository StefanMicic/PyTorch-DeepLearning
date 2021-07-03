from typing import Tuple

import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch._C import device
from torch.functional import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import calculate_metrics, print_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class LeNet(nn.Module):
    """Convolutional neural network with LeNet architecture."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.training_loader, self.validation_loader = self.create_dataset()
        self.running_loss = 0.0
        self.running_corrects = 0.0
        self.val_running_loss = 0.0
        self.val_running_corrects = 0.0

    def create_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Creates data loaders for training and validation.

        Returns:
            Tuple[DataLoader, DataLoader]:
                    Training and validation dataset loaders
        """
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        training_dataset = datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
            transform=transform_train,
        )
        validation_dataset = datasets.CIFAR10(
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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def calculate_metrics(
        self,
        loader: DataLoader,
    ) -> Tuple[float, float]:
        """Calculates loss and accuracy.
        Args:
            loader (DataLoader)
        Returns:
            Tuple[float, float]:
                loss value and current accuracy
        """
        epoch_loss = self.running_loss / len(loader)
        epoch_acc = self.running_corrects.float() / len(loader)
        return (epoch_loss, epoch_acc)

    def batch_iteration(self, optimizer: Adam) -> None:
        """Goes through training set and does forward and backward pass.
        Args:
            optimizer(Adam): Optimizer for optimizing weights
        """
        for inputs, labels in self.training_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
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
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = self(val_inputs)
            val_loss = self.criterion(val_outputs, val_labels)

            _, val_preds = torch.max(val_outputs, 1)
            self.val_running_loss += val_loss.item()
            self.val_running_corrects += torch.sum(
                val_preds == val_labels.data
            )

    def train(self):
        """Trains neural network"""
        epochs = 15
        optimizer = Adam(self.parameters(), lr=0.001)

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
                epoch_num, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc
            )


def main():
    model = LeNet().to(device)
    model.train()


if __name__ == "__main__":
    main()
