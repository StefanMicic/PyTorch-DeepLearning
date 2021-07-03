from typing import Tuple

import torch
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.vgg import VGG

from common import calculate_metrics, print_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransferLearner:
    """Class for training VGG with transfer learning."""

    def __init__(self) -> None:
        """Init method."""
        self.model = models.vgg16(pretrained=True)
        self.training_loader, self.validation_loader = self.create_dataset()
        self.criterion = nn.CrossEntropyLoss()

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
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        training_dataset = datasets.ImageFolder(
            "train", transform=transform_train
        )
        validation_dataset = datasets.ImageFolder("val", transform=transform)

        training_loader = DataLoader(
            training_dataset, batch_size=20, shuffle=True
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=20, shuffle=False
        )
        return training_loader, validation_loader

    def adapt_model(self):
        """Adapt VGG model to have 10 possible output classes."""
        for param in self.model.features.parameters():
            param.requires_grad = False

        n_inputs = self.model.classifier[6].in_features
        last_layer = nn.Linear(n_inputs, len(classes))
        self.model.classifier[6] = last_layer
        self.model.to(device)

    def batch_iteration(self, optimizer) -> None:
        """Goes through training set and does forward and backward pass.
        Args:
            optimizer(Adam): Optimizer for optimizing weights
        """
        for inputs, labels in self.training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.model(inputs)
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
            val_outputs = self.model(val_inputs)
            val_loss = self.criterion(val_outputs, val_labels)

            _, val_preds = torch.max(val_outputs, 1)
            self.val_running_loss += val_loss.item()
            self.val_running_corrects += torch.sum(
                val_preds == val_labels.data
            )

    def train(self) -> None:
        """Trains neural network using transfer learning.
        Args:
            model (VFF): Pretrained VGG network
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        epochs = 5
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


classes = ("ant", "bee")


def main():
    model = TransferLearner()
    model.train()


if __name__ == "__main__":
    main()
