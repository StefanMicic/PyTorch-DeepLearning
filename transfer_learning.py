from typing import Tuple

import numpy as np
import torch
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.vgg import VGG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataset() -> Tuple[DataLoader, DataLoader]:
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

    training_dataset = datasets.ImageFolder("train", transform=transform_train)
    validation_dataset = datasets.ImageFolder("val", transform=transform)

    training_loader = DataLoader(training_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=20, shuffle=False
    )
    return training_loader, validation_loader


def train(model: VGG) -> None:
    """Trains neural network using transfer learning.
    Args:
        model (VFF): Pretrained VGG network
    """
    training_loader, validation_loader = create_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 5
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):

        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for inputs, labels in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        else:
            with torch.no_grad():
                for val_inputs, val_labels in validation_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(
                        val_preds == val_labels.data
                    )

            epoch_loss = running_loss / len(training_loader.dataset)
            epoch_acc = running_corrects.float() / len(training_loader.dataset)
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)

            val_epoch_loss = val_running_loss / len(validation_loader.dataset)
            val_epoch_acc = val_running_corrects.float() / len(
                validation_loader.dataset
            )
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            log.info(f"epoch :{(e + 1)}")
            log.info(f"training loss: {epoch_loss}, acc {epoch_acc.item()}")
            log.info(
                f"validation loss: {val_epoch_loss}, validation acc {val_epoch_acc.item()} "
            )


classes = ("ant", "bee")


def main():

    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(classes))
    model.classifier[6] = last_layer
    model.to(device)


if __name__ == "__main__":
    main()
