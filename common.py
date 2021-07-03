from typing import Tuple

from loguru import logger as log
from torch.functional import Tensor
from torch.utils.data import DataLoader


def print_metrics(
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
    running_loss: float,
    running_corrects: float,
    loader: DataLoader,
) -> Tuple[float, float]:
    """Calculates loss and accuracy.
    Args:
        running_loss (float): Current loss value
        running_corrects (float): Number of correct predictions
        loader (DataLoader)
    Returns:
        Tuple[float, float]:
            loss value and current accuracy
    """
    epoch_loss = running_loss / len(loader)
    epoch_acc = running_corrects.float() / len(loader)
    return (epoch_loss, epoch_acc)
