# stopping_loss_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from exceptions import InitializationError, TypeError, ValueError, ModelSavingError
from typing import Union, Dict

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Implements early stopping to prevent overfitting during training.

    Monitors a validation loss and stops training if it doesn't improve for a
    specified patience. Saves the model state corresponding to the best
    validation loss encountered.

    Args:
        patience (int, optional): Number of epochs with no improvement after
            which training will be stopped. Defaults to 7.
        verbose (bool, optional): If True, prints a message for each validation
            loss improvement. Defaults to False.
        delta (float, optional): Minimum change in the monitored quantity to
            qualify as an improvement. Defaults to 0.0.
        path (str, optional): Path to save the checkpoint of the best model.
            Defaults to 'chk_learn.pt'.

    Attributes:
        patience (int): The patience value set during initialization.
        verbose (bool): The verbosity flag set during initialization.
        delta (float): The delta value set during initialization.
        path (str): The path to save the best model.
        counter (int): Counter that tracks the number of epochs with no improvement.
        best_score (float or None): The best validation loss encountered so far.
        early_stop (bool): Flag indicating if early stopping should be triggered.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, path: str = 'chk_learn.pt') -> None:
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError(f"Expected patience to be a positive integer, but got {patience}")
        if not isinstance(verbose, bool):
            raise TypeError(f"Expected verbose to be a boolean, but got {type(verbose).__name__}")
        if not isinstance(delta, (int, float)):
            raise TypeError(f"Expected delta to be a number, but got {type(delta).__name__}")
        if not isinstance(path, str):
            raise TypeError(f"Expected path to be a string, but got {type(path).__name__}")
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score: Union[float, None] = None
        self.early_stop = False

    def __call__(self, valid_loss: Union[float, torch.Tensor], model: nn.Module) -> None:
        """
        Evaluates the current validation loss and potentially triggers early stopping.

        Args:
            valid_loss (float or torch.Tensor): The current validation loss.
            model (torch.nn.Module): The neural network model being trained.
        """
        if not isinstance(valid_loss, (int, float, torch.Tensor)):
            raise TypeError(f"Expected valid_loss to be a number or torch.Tensor, but got {type(valid_loss).__name__}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected model to be a torch.nn.Module, but got {type(model).__name__}")

        current_loss = valid_loss.item() if isinstance(valid_loss, torch.Tensor) else valid_loss

        if self.best_score is None:
            self.best_score = current_loss
            self.save_model_state(model)
        elif current_loss < self.best_score - self.delta:
            if self.verbose:
                logger.info(f'Validation Loss improves {self.best_score: .4f}->{current_loss: .4f}=>$ave model')
            self.best_score = current_loss
            self.save_model_state(model)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'EarlyStopping patience ticks {self.counter} from {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model_state(self, model: nn.Module) -> None:
        """
        Saves the state dictionary of the provided model.

        Args:
            model (torch.nn.Module): The neural network model whose state dictionary
                will be saved.

        Raises:
            ModelSavingError: If there is an error during the saving process.
        """
        try:
            torch.save({'model': model.state_dict(), 'state_dict': model.state_dict()}, self.path)
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            raise ModelSavingError(f"Error saving model state to {self.path}: {e}")

def nequip_loss(energy_pred: torch.Tensor, force_pred: torch.Tensor, energy_target: torch.Tensor, force_target: torch.Tensor, energy_weight: float = 1.0, force_weight: float = 0.1, delta: float = 1.0) -> torch.Tensor:
    """
    Computes a combined Huber loss for energy and force predictions, commonly
    used in NequIP models.

    This loss function calculates the Huber loss separately for the energy and
    force predictions and then combines them using the provided weights.

    Args:
        energy_pred (torch.Tensor): Predicted energy values. The tensor should
            have a shape compatible with the energy target.
        force_pred (torch.Tensor): Predicted force values. The tensor should have
            the same shape as the force target.
        energy_target (torch.Tensor): Target energy values.
        force_target (torch.Tensor): Target force values.
        energy_weight (float, optional): Weighting factor for the energy loss.
            Defaults to 1.0.
        force_weight (float, optional): Weighting factor for the force loss.
            Defaults to 0.1.
        delta (float, optional): Threshold (delta) for the Huber loss. Values
            where the absolute error is less than delta are squared, otherwise
            the loss becomes linear. Defaults to 1.0.

    Returns:
        torch.Tensor: The combined Huber loss, a scalar tensor.

    Raises:
        TypeError: If any of the input arguments have an incorrect type.
    """
    try:
        if not isinstance(energy_pred, torch.Tensor):
            raise TypeError(f"Expected energy_pred to be a torch.Tensor, but got {type(energy_pred).__name__}")
        if not isinstance(force_pred, torch.Tensor):
            raise TypeError(f"Expected force_pred to be a torch.Tensor, but got {type(force_pred).__name__}")
        if not isinstance(energy_target, torch.Tensor):
            raise TypeError(f"Expected energy_target to be a torch.Tensor, but got {type(energy_target).__name__}")
        if not isinstance(force_target, torch.Tensor):
            raise TypeError(f"Expected force_target to be a torch.Tensor, but got {type(force_target).__name__}")
        if not isinstance(energy_weight, (int, float)):
            raise TypeError(f"Expected energy_weight to be a number, but got {type(energy_weight).__name__}")
        if not isinstance(force_weight, (int, float)):
            raise TypeError(f"Expected force_weight to be a number, but got {type(force_weight).__name__}")
        if not isinstance(delta, (int, float)):
            raise TypeError(f"Expected delta to be a number, but got {type(delta).__name__}")

        energy_loss = F.huber_loss(energy_pred.squeeze(-1), energy_target, delta=delta)
        force_loss = F.huber_loss(force_pred, force_target, delta=delta)

        total_loss = (energy_weight * energy_loss) + (force_weight * force_loss)
        return total_loss
    except TypeError as e:
        logger.error(f"TypeError in nequip_loss: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in nequip_loss: {e}")
        raise
