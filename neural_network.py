from typing import Tuple

import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
        (
            self.training_loader,
            self.validation_loader,
        ) = self.create_dataset_loaders()

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
            root="./data", train=True, download=True, transform=transform
        )
        validation_dataset = datasets.MNIST(
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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def train(self):
        """Trains neural network."""
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        epochs = 15
        running_loss_history = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []

        for e in range(epochs):

            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0

            for inputs, labels in self.training_loader:
                inputs = inputs.view(inputs.shape[0], -1)
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
                        val_inputs = val_inputs.view(val_inputs.shape[0], -1)
                        val_outputs = self(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                        _, val_preds = torch.max(val_outputs, 1)
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(
                            val_preds == val_labels.data
                        )

                epoch_loss = running_loss / len(self.training_loader)
                epoch_acc = running_corrects.float() / len(
                    self.training_loader
                )
                running_loss_history.append(epoch_loss)
                running_corrects_history.append(epoch_acc)

                val_epoch_loss = val_running_loss / len(self.validation_loader)
                val_epoch_acc = val_running_corrects.float() / len(
                    self.validation_loader
                )
                val_running_loss_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_acc)
                log.info(f"epoch :{(e + 1)}")
                log.info(
                    f"training loss: {epoch_loss}, acc {epoch_acc.item()}"
                )
                log.info(
                    f"validation loss: {val_epoch_loss}, validation acc {val_epoch_acc.item()} "  # noqa E501
                )


def main():
    model = Classifier(784, 125, 65, 10)
    model.train()


if __name__ == "__main__":
    main()
