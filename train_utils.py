# train_utils.py
import torch
import numpy as np
import torch.nn as nn
import logging
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as PyTorchDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from stopping_loss_utils import EarlyStopping, nequip_loss
from graph_collate_utils import generate_graph, collate_graph_data
from model import NequIP
from exceptions import ParameterError, DataLoaderCreationError, BatchTypeError
from typing import Tuple, Union

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
    step_lr: Union[StepLR, ReduceLROnPlateau],
    train_loader: PyTorchDataLoader,
    epoch: int,
    energy_mean: Union[int, float, torch.Tensor],
    energy_std: Union[int, float, torch.Tensor],
    force_mean: Union[int, float, torch.Tensor],
    force_std: Union[int, float, torch.Tensor],
) -> float:
    """
    Performs one epoch of training on the provided data.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (callable): The loss function used for training. It should accept
            predicted and target energies and forces.
        optimizer (torch.optim.Optimizer): The optimizer responsible for updating model weights.
        step_lr (Union[StepLR, ReduceLROnPlateau]): The learning rate scheduler.
        train_loader (PyTorchDataLoader): DataLoader providing batches of training graphs.
        epoch (int): The current training epoch number (0-indexed).
        energy_mean (Union[int, float, torch.Tensor]): Mean of the energy values in the training set.
        energy_std (Union[int, float, torch.Tensor]): Standard deviation of the energy values in the training set.
        force_mean (Union[int, float, torch.Tensor]): Mean of the force values in the training set.
        force_std (Union[int, float, torch.Tensor]): Standard deviation of the force values in the training set.

    Returns:
        float: The average training loss for the epoch.

    Raises:
        ParameterError: If any of the input arguments are of incorrect type.
        DataLoaderCreationError: If the `train_loader` is not a `torch.utils.data.DataLoader`.
        BatchTypeError: If a batch from the `train_loader` is not a `torch_geometric.data.Batch`.
        RuntimeError: If a runtime error occurs during the training process.
        Exception: For any other unexpected errors during training.
    """
    model.train()
    total_loss = 0
    num_graphs = 0

    try:
        if not isinstance(model, nn.Module):
            raise ParameterError(f"Expected model to be a torch.nn.Module, but got {type(model).__name__}")
        if not callable(criterion):
            raise ParameterError(f"Expected criterion to be a callable, but got {type(criterion).__name__}")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ParameterError(f"Expected optimizer to be a torch.optim.Optimizer, but got {type(optimizer).__name__}")
        if not isinstance(step_lr, (StepLR, ReduceLROnPlateau)):
            raise ParameterError(f"Expected step_lr to be a StepLR or ReduceLROnPlateau object, but got {type(step_lr).__name__}")
        if not isinstance(train_loader, PyTorchDataLoader):
            raise DataLoaderCreationError(f"Expected train_loader to be a torch.utils.data.DataLoader, but got {type(train_loader).__name__}")
        if not isinstance(epoch, int):
            raise ParameterError(f"Expected epoch to be an integer, but got {type(epoch).__name__}")
        if not isinstance(energy_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_mean to be a number or torch.Tensor, but got {type(energy_mean).__name__}")
        if not isinstance(energy_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_std to be a number or torch.Tensor, but got {type(energy_std).__name__}")
        if not isinstance(force_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_mean to be a number or torch.Tensor, but got {type(force_mean).__name__}")
        if not isinstance(force_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_std to be a number or torch.Tensor, but got {type(force_std).__name__}")

        for batch in train_loader:
            if not isinstance(batch, Batch):
                raise BatchTypeError(f"Expected batch to be a torch_geometric.data.Batch, but got {type(batch).__name__}")
            optimizer.zero_grad()
            graph_energy_pred, force_pred, _ = model(batch)

            energy_target_orig = batch.energy * energy_std + energy_mean
            force_target_orig = batch.force * force_std + force_mean
            energy_pred_orig = graph_energy_pred * energy_std + energy_mean
            force_pred_orig = force_pred * force_std + force_mean

            loss = criterion(energy_pred_orig, force_pred_orig, energy_target_orig, force_target_orig)
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step_lr.step()

        average_loss = total_loss / num_graphs if num_graphs > 0 else 0
        logger.info(f'Epoch {epoch + 1}, Training Loss {average_loss: .4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        return average_loss

    except (ParameterError, DataLoaderCreationError, BatchTypeError) as e:
        logger.error(f"Error in train: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"RuntimeError in train: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in train: {e}")
        raise

@torch.no_grad()
def valid(
    model: nn.Module,
    criterion: callable,
    valid_loader: PyTorchDataLoader,
    epoch: int,
    early_stopping: EarlyStopping = None,
    energy_mean: Union[int, float, torch.Tensor] = 0,
    energy_std: Union[int, float, torch.Tensor] = 1,
    force_mean: Union[int, float, torch.Tensor] = 0,
    force_std: Union[int, float, torch.Tensor] = 1,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        criterion (callable): The loss function used for evaluation. It should accept
            predicted and target energies and forces.
        valid_loader (PyTorchDataLoader): DataLoader providing batches of validation graphs.
        epoch (int): The current training epoch number (0-indexed).
        early_stopping (EarlyStopping, optional): An `EarlyStopping` object to track
            validation loss and potentially stop training early. Defaults to None.
        energy_mean (Union[int, float, torch.Tensor], optional): Mean of the energy
            values in the training set (used for de-normalization). Defaults to 0.
        energy_std (Union[int, float, torch.Tensor], optional): Standard deviation
            of the energy values in the training set (used for de-normalization). Defaults to 1.
        force_mean (Union[int, float, torch.Tensor], optional): Mean of the force
            values in the training set (used for de-normalization). Defaults to 0.
        force_std (Union[int, float, torch.Tensor], optional): Standard deviation
            of the force values in the training set (used for de-normalization). Defaults to 1.

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the average
        validation loss, energy Mean Absolute Error (MAE), energy Root Mean Squared
        Error (RMSE), energy R-squared (R2) score, and force Mean Absolute Error (MAE).

    Raises:
        ParameterError: If any of the input arguments are of incorrect type.
        DataLoaderCreationError: If the `valid_loader` is not a `torch.utils.data.DataLoader`.
        BatchTypeError: If a batch from the `valid_loader` is not a `torch_geometric.data.Batch`.
        RuntimeError: If a runtime error occurs during the validation process.
        Exception: For any other unexpected errors during validation.
    """
    model.eval()
    total_loss = 0
    num_graphs = 0

    all_energy_targets = []
    all_energy_preds = []
    all_force_targets = []
    all_force_preds = []

    try:
        if not isinstance(model, nn.Module):
            raise ParameterError(f"Expected model to be a torch.nn.Module, but got {type(model).__name__}")
        if not callable(criterion):
            raise ParameterError(f"Expected criterion to be a callable, but got {type(criterion).__name__}")
        if not isinstance(valid_loader, PyTorchDataLoader):
            raise DataLoaderCreationError(f"Expected valid_loader to be a torch.utils.data.DataLoader, but got {type(valid_loader).__name__}")
        if not isinstance(epoch, int):
            raise ParameterError(f"Expected epoch to be an integer, but got {type(epoch).__name__}")
        if early_stopping is not None and not isinstance(early_stopping, EarlyStopping):
            raise ParameterError(f"Expected early_stopping to be an EarlyStopping object or None, but got {type(early_stopping).__name__}")
        if not isinstance(energy_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_mean to be a number or torch.Tensor, but got {type(energy_mean).__name__}")
        if not isinstance(energy_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_std to be a number or torch.Tensor, but got {type(energy_std).__name__}")
        if not isinstance(force_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_mean to be a number or torch.Tensor, but got {type(force_mean).__name__}")
        if not isinstance(force_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_std to be a number or torch.Tensor, but got {type(force_std).__name__}")

        for batch in valid_loader:
            if not isinstance(batch, Batch):
                raise BatchTypeError(f"Expected batch to be a torch_geometric.data.Batch, but got {type(batch).__name__}")
            graph_energy_pred, force_pred, _ = model(batch)

            energy_target_orig = batch.energy * energy_std + energy_mean
            force_target_orig = batch.force * force_std + force_mean
            energy_pred_orig = graph_energy_pred * energy_std + energy_mean
            force_pred_orig = force_pred * force_std + force_mean

            loss = criterion(energy_pred_orig, force_pred_orig, energy_target_orig, force_target_orig)
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

            all_energy_targets.extend(energy_target_orig.cpu().numpy())
            all_energy_preds.extend(energy_pred_orig.squeeze(-1).cpu().numpy())
            all_force_targets.extend(force_target_orig.cpu().numpy().flatten())
            all_force_preds.extend(force_pred_orig.cpu().numpy().flatten())

        average_loss = total_loss / num_graphs if num_graphs > 0 else 0
        energy_mae = mean_absolute_error(all_energy_targets, all_energy_preds)
        energy_rmse = np.sqrt(mean_squared_error(all_energy_targets, all_energy_preds))
        energy_r2 = r2_score(all_energy_targets, all_energy_preds)
        force_mae = mean_absolute_error(all_force_targets, all_force_preds)

        logger.info(f'Epoch {epoch + 1}, Validation Loss {average_loss: .4f}, Energy MAE: {energy_mae: .4f}, RMSE: {energy_rmse: .4f}, R2: {energy_r2: .4f}, Force MAE: {force_mae: .4f}')

        if early_stopping is not None:
            early_stopping(average_loss, model)

        return average_loss, energy_mae, energy_rmse, energy_r2, force_mae

    except (ParameterError, DataLoaderCreationError, BatchTypeError) as e:
        logger.error(f"Error in valid: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"RuntimeError in valid: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in valid: {e}")
        raise

@torch.no_grad()
def test(
    model: nn.Module,
    criterion: callable,
    test_loader: PyTorchDataLoader,
    energy_mean: Union[int, float, torch.Tensor] = 0,
    energy_std: Union[int, float, torch.Tensor] = 1,
    force_mean: Union[int, float, torch.Tensor] = 0,
    force_std: Union[int, float, torch.Tensor] = 1,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the trained model on the test dataset.

    Args:
        model (nn.Module): The trained neural network model.
        criterion (callable): The loss function used for evaluation. It should accept
            predicted and target energies and forces.
        test_loader (PyTorchDataLoader): DataLoader providing batches of test graphs.
        energy_mean (Union[int, float, torch.Tensor], optional): Mean of the energy
            values in the training set (used for de-normalization). Defaults to 0.
        energy_std (Union[int, float, torch.Tensor], optional): Standard deviation
            of the energy values in the training set (used for de-normalization). Defaults to 1.
        force_mean (Union[int, float, torch.Tensor], optional): Mean of the force
            values in the training set (used for de-normalization). Defaults to 0.
        force_std (Union[int, float, torch.Tensor], optional): Standard deviation
            of the force values in the training set (used for de-normalization). Defaults to 1.

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the average
        test loss, energy Mean Absolute Error (MAE), energy Root Mean Squared Error
        (RMSE), energy R-squared (R2) score, and force Mean Absolute Error (MAE).

    Raises:
        ParameterError: If any of the input arguments are of incorrect type.
        DataLoaderCreationError: If the `test_loader` is not a `torch.utils.data.DataLoader`.
        BatchTypeError: If a batch from the `test_loader` is not a `torch_geometric.data.Batch`.
        RuntimeError: If a runtime error occurs during the testing process.
        Exception: For any other unexpected errors during testing.
    """
    model.eval()
    total_loss = 0
    num_graphs = 0

    all_energy_targets = []
    all_energy_preds = []
    all_force_targets = []
    all_force_preds = []

    try:
        if not isinstance(model, nn.Module):
            raise ParameterError(f"Expected model to be a torch.nn.Module, but got {type(model).__name__}")
        if not callable(criterion):
            raise ParameterError(f"Expected criterion to be a callable, but got {type(criterion).__name__}")
        if not isinstance(test_loader, PyTorchDataLoader):
            raise DataLoaderCreationError(f"Expected test_loader to be a torch.utils.data.DataLoader, but got {type(test_loader).__name__}")
        if not isinstance(energy_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_mean to be a number or torch.Tensor, but got {type(energy_mean).__name__}")
        if not isinstance(energy_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected energy_std to be a number or torch.Tensor, but got {type(energy_std).__name__}")
        if not isinstance(force_mean, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_mean to be a number or torch.Tensor, but got {type(force_mean).__name__}")
        if not isinstance(force_std, (int, float, torch.Tensor)):
            raise ParameterError(f"Expected force_std to be a number or torch.Tensor, but got {type(force_std).__name__}")

        for batch in test_loader:
            if not isinstance(batch, Batch):
                raise BatchTypeError(f"Expected batch to be a torch_geometric.data.Batch, but got {type(batch).__name__}")
            graph_energy_pred, force_pred, _ = model(batch)

            energy_target_orig = batch.energy * energy_std + energy_mean
            force_target_orig = batch.force * force_std + force_mean
            energy_pred_orig = graph_energy_pred * energy_std + energy_mean
            force_pred_orig = force_pred * force_std + force_mean

            loss = criterion(energy_pred_orig, force_pred_orig, energy_target_orig, force_target_orig)
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

            all_energy_targets.extend(energy_target_orig.cpu().numpy())
            all_energy_preds.extend(energy_pred_orig.squeeze(-1).cpu().numpy())
            all_force_targets.extend(force_target_orig.cpu().numpy().flatten())
            all_force_preds.extend(force_pred_orig.cpu().numpy().flatten())

        average_loss = total_loss / num_graphs if num_graphs > 0 else 0
        energy_mae = mean_absolute_error(all_energy_targets, all_energy_preds)
        energy_rmse = np.sqrt(mean_squared_error(all_energy_targets, all_energy_preds))
        energy_r2 = r2_score(all_energy_targets, all_energy_preds)
        force_mae = mean_absolute_error(all_force_targets, all_force_preds)

        logger.info(f'Testing Loss {average_loss: .4f}, Energy MAE: {energy_mae: .4f}, RMSE: {energy_rmse: .4f}, R2: {energy_r2: .4f}, Force MAE: {force_mae: .4f}')

        return average_loss, energy_mae, energy_rmse, energy_r2, force_mae

    except (ParameterError, DataLoaderCreationError, BatchTypeError) as e:
        logger.error(f"Error in test: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"RuntimeError in test: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in test: {e}")
        raise
