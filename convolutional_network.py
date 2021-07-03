from typing import Tuple

import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch._C import device
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)
        self.training_loader, self.validation_loader = self.create_dataset()
        self.running_loss_history: list
        self.running_corrects_history: list
        self.val_running_loss_history: list
        self.val_running_corrects_history: list

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
            root="./data", train=True, download=True, transform=transform_train
        )
        validation_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
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

    def print_metrics(
        self,
        epoch_num: int,
        epoch_loss: float,
        epoch_acc: Tensor,
        val_epoch_loss: float,
        val_epoch_acc: Tensor,
    ) -> None:
        """Prints all metrics.

        Args:
            epoch_num (int),
            epoch_loss (float),
            epoch_acc (Tensor),
            val_epoch_loss (float),
            val_epoch_acc (Tensor),
        """
        log.info(f"epoch :{(epoch_num + 1)}")
        log.info(f"training loss: {epoch_loss}, acc {epoch_acc.item()}")
        log.info(
            f"validation loss: {val_epoch_loss}, validation acc {val_epoch_acc.item()} "  # noqa E501
        )

    def calculate_metrics(
        self,
        running_loss: float,
        running_corrects: float,
        running_loss_history: list,
        running_corrects_history: list,
        loader: DataLoader,
    ) -> Tuple[float, float, list, list]:
        """Forward pass in neural network.
        Args:
            running_loss (float): current value of loss function
            running_corrects (list): number of correct predictions
            running_loss_history (float) : list of losses
            running_corrects_history (list): list of accuracies
            loader (DataLoader)
        Returns:
            Tuple[float, float, list, list]:
                loss value, current accuracy, list of losses
                and list of accuracies
        """
        epoch_loss = running_loss / len(loader)
        epoch_acc = running_corrects.float() / len(loader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)
        return (
            epoch_loss,
            epoch_acc,
            running_loss_history,
            running_corrects_history,
        )

    def train(self, epochs: int = 15, learning_rate: float = 0.001):
        """Trains neural network.
        Args:
            epochs (int): Number of epochs for model training
            lr (float): Speed of learning process
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch_num in range(epochs):

            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0

            for inputs, labels in self.training_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            else:
                with torch.no_grad():
                    for val_inputs, val_labels in self.validation_loader:
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = self(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                        _, val_preds = torch.max(val_outputs, 1)
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(
                            val_preds == val_labels.data
                        )

                (
                    epoch_loss,
                    epoch_acc,
                    self.running_loss_history,
                    self.running_corrects_history,
                ) = self.calculate_metrics(
                    running_loss,
                    running_corrects,
                    self.running_loss_history,
                    self.running_corrects_history,
                    self.training_loader,
                )
                (
                    val_epoch_loss,
                    val_epoch_acc,
                    self.val_running_loss_history,
                    self.val_running_corrects_history,
                ) = self.calculate_metrics(
                    val_running_loss,
                    val_running_corrects,
                    self.val_running_loss_history,
                    self.val_running_corrects_history,
                    self.validation_loader,
                )

            self.print_metrics(
                epoch_num, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc
            )


def main():
    model = LeNet().to(device)
    model.train()


if __name__ == "__main__":
    main()
