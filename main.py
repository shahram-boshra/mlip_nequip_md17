# main.py
import os
import yaml
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import MD17
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader as PyTorchDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from config import config
from stopping_loss_utils import EarlyStopping, nequip_loss
from graph_collate_utils import generate_graph, collate_graph_data
from model import NequIP
from train_utils import train, valid, test
from exceptions import DataLoadingError, GraphGenerationError, DataLoaderCreationError, ModelInstantiationError, CheckpointLoadingError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """
    Main script to train and evaluate a NequIP model (Neural Equivariant Interatomic Potentials) on the MD17 dataset
    for molecular property prediction (energy and forces).

    This script performs the following steps:
    1. Loads configuration from `config.py`.
    2. Downloads and preprocesses the specified molecules from the MD17 dataset, generating graph representations.
    3. Splits the data into training, validation, and testing sets.
    4. Creates PyTorch DataLoaders for efficient batch processing.
    5. Instantiates the NequIP model.
    6. Defines the optimizer, learning rate schedulers, and loss function.
    7. Implements an early stopping mechanism based on validation loss.
    8. Trains the model on the training data, evaluating performance on the validation set at each epoch.
    9. Loads the best performing model from the checkpoint.
    10. Evaluates the loaded model on the test set and reports the final performance metrics.
    11. Visualizes the training and validation losses, as well as the validation energy and force MAE, RMSE, and R2 score over epochs.

    Raises:
        DataLoadingError: If there is an issue loading the MD17 dataset.
        GraphGenerationError: If graph generation fails for any data sample.
        DataLoaderCreationError: If there is an error creating the PyTorch DataLoaders.
        ModelInstantiationError: If there is an error instantiating the NequIP model.
        CheckpointLoadingError: If loading the best model checkpoint fails.
    """
    cutoff_distance = config.data.cutoff_distance
    all_molecules = config.data.molecules
    root_dir = config.data.root_dir
    subset_size = config.data.subset_size
    random_seed = config.data.random_seed

    batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    step_lr_step_size = config.training.step_lr_step_size
    step_lr_gamma = config.training.step_lr_gamma
    reduce_lr_factor = config.training.reduce_lr_factor
    reduce_lr_patience = config.training.reduce_lr_patience
    reduce_lr_min_lr = config.training.reduce_lr_min_lr
    energy_weight = config.training.energy_weight
    force_weight = config.training.force_weight
    huber_delta = config.training.huber_delta
    patience = config.training.patience
    checkpoint_path = config.training.checkpoint_path

    hidden_channels = config.model.hidden_channels
    num_radial = config.model.num_radial
    num_layers = config.model.num_layers
    graph_out_channels = config.model.graph_out_channels
    force_out_channels = config.model.force_out_channels

    os.makedirs(root_dir, exist_ok=True)

    all_molecule_data_with_graphs = []
    rng = np.random.default_rng(random_seed)

    all_energies = []
    all_forces = []
    skipped_molecules = []

    logger.info(f"Attempting to load and generate graph for molecules of MD17 dataset...")
    for molecule_name in all_molecules:
        try:
            dataset = MD17(root=root_dir, name=molecule_name)
            num_samples = min(subset_size, len(dataset))
            indices = rng.choice(len(dataset), size=num_samples, replace=False)
            molecule_subset = [dataset[i].clone() for i in sorted(indices)]
            molecule_graphs = []
            for data in molecule_subset:
                try:
                    data = generate_graph(data, cutoff_distance)
                    molecule_graphs.append(data)
                    if hasattr(data, 'energy') and data.energy is not None and hasattr(data, 'force') and data.force is not None:
                        all_energies.append(data.energy.item())
                        all_forces.extend(data.force.flatten().tolist())
                    else:
                        logger.warning(f"Skipping sample from {molecule_name} due to None energy or force.")
                except Exception as e:
                    raise GraphGenerationError(f"Failed to generate graph for a sample in {molecule_name}: {e}")
            all_molecule_data_with_graphs.extend(molecule_graphs)
            logger.info(f"Generated graphs for {len(molecule_graphs)} snapshots from {molecule_name}.")
        except Exception as e:
            if isinstance(e, GraphGenerationError):
                logger.error(str(e))
            else:
                logger.error(f"Error loading {molecule_name}: {e}")
            skipped_molecules.append(molecule_name)

    loaded_snapshots_count = len(all_molecule_data_with_graphs)
    final_log_message = f"Successfully processed {loaded_snapshots_count} snapshots from all molecules"
    if skipped_molecules:
        skipped_log = ", ".join(skipped_molecules)
        final_log_message += f" (Skipped loading for: {skipped_log})"
    logger.info(final_log_message)

    if all_energies and all_forces:
        all_energies = torch.tensor(all_energies)
        all_forces = torch.tensor(all_forces)

        energy_mean = all_energies.mean()
        energy_std = all_energies.std()
        force_mean = all_forces.mean()
        force_std = all_forces.std()

        logger.info(f"Energy Mean: {energy_mean:.4f}, Energy Std: {energy_std:.4f}")
        logger.info(f"Force Mean: {force_mean:.4f}, Force Std: {force_std:.4f}")

        for data in all_molecule_data_with_graphs:
            if hasattr(data, 'energy') and data.energy is not None and hasattr(data, 'force') and data.force is not None:
                data.energy = (data.energy - energy_mean) / energy_std
                data.force = (data.force - force_mean) / force_std
            elif hasattr(data, 'energy') is not None or hasattr(data, 'force') is not None:
                logger.warning("Found a data point with either energy or force as None after normalization attempt.")
    else:
        logger.error("No energy or force data collected. Cannot proceed with normalization.")
        exit()

    rng.shuffle(all_molecule_data_with_graphs)
    filtered_data_with_graphs = [data for data in all_molecule_data_with_graphs if hasattr(data, 'energy') and data.energy is not None and hasattr(data, 'force') and data.force is not None]
    if len(filtered_data_with_graphs) < len(all_molecule_data_with_graphs):
        logger.warning(f"Filtered out {len(all_molecule_data_with_graphs) - len(filtered_data_with_graphs)} samples with None energy or force values after shuffling.")
    all_molecule_data_with_graphs = filtered_data_with_graphs

    train_size = int(0.65 * len(all_molecule_data_with_graphs))
    valid_size = int(0.25 * len(all_molecule_data_with_graphs))
    train_dataset_with_graphs = all_molecule_data_with_graphs[:train_size]
    valid_dataset_with_graphs = all_molecule_data_with_graphs[train_size:train_size + valid_size]
    test_dataset_with_graphs = all_molecule_data_with_graphs[train_size + valid_size:]

    logger.info(f"\nCombined dataset size (with generated graphs): {len(all_molecule_data_with_graphs)}")
    logger.info(f"Training dataset size (with generated graphs): {len(train_dataset_with_graphs)}")
    logger.info(f"Validation dataset size (with generated graphs): {len(valid_dataset_with_graphs)}")
    logger.info(f"Testing dataset size (with generated graphs): {len(test_dataset_with_graphs)}")

    try:
        train_loader = PyTorchDataLoader(train_dataset_with_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_data)
        valid_loader = PyTorchDataLoader(valid_dataset_with_graphs, batch_size=batch_size, collate_fn=collate_graph_data)
        test_loader = PyTorchDataLoader(test_dataset_with_graphs, batch_size=batch_size, collate_fn=collate_graph_data)
    except Exception as e:
        raise DataLoaderCreationError(f"Error creating DataLoaders: {e}")

    max_atomic_num = 0
    if all_molecule_data_with_graphs:
        for data in all_molecule_data_with_graphs:
            if hasattr(data, 'z') and isinstance(data.z, torch.Tensor) and data.z.numel() > 0:
                max_atomic_num = max(max_atomic_num, data.z.max().item() + 1)
    num_atom_types = max_atomic_num if max_atomic_num > 0 else 100
    logger.info(f"Number of atom types inferred: {num_atom_types}")

    try:
        model = NequIP(num_atom_types, hidden_channels, num_radial, num_layers, cutoff_distance, graph_out_channels, force_out_channels)
    except (ValueError, TypeError) as e:
        raise ModelInstantiationError(f"Error during model instantiation: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_lr = StepLR(optimizer, step_size=step_lr_step_size, gamma=step_lr_gamma)
    red_lr = ReduceLROnPlateau(optimizer, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
    criterion = lambda p_e, p_f, t_e, t_f: nequip_loss(p_e, p_f, t_e, t_f, energy_weight, force_weight, huber_delta)

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    train_losses = []
    valid_losses = []
    valid_energy_maes = []
    valid_energy_rmses = []
    valid_energy_r2s = []
    valid_force_maes = []
    epoch = 0
    while not early_stopping.early_stop:
        train_loss = train(model, criterion, optimizer, step_lr, train_loader, epoch, energy_mean, energy_std, force_mean, force_std)
        valid_loss, energy_mae, energy_rmse, energy_r2, force_mae = valid(model, criterion, valid_loader, epoch, early_stopping, energy_mean, energy_std, force_mean, force_std)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_energy_maes.append(energy_mae)
        valid_energy_rmses.append(energy_rmse)
        valid_energy_r2s.append(energy_r2)
        valid_force_maes.append(force_mae)
        red_lr.step(valid_loss)
        epoch += 1

    if early_stopping.early_stop:
        try:
            chk_lrn = torch.load(early_stopping.path, map_location=torch.device('cpu'))
            model.load_state_dict(chk_lrn['model'])
            logger.info(f"Loaded best model state from checkpoint: {early_stopping.path}")
        except Exception as e:
            raise CheckpointLoadingError(f'Trouble loading model checkpoint from {early_stopping.path}: {e}')

    test_loss, test_energy_mae, test_energy_rmse, test_energy_r2, test_force_mae = test(model, criterion, test_loader, energy_mean, energy_std, force_mean, force_std)
    logger.info(f"Final Test Loss: {test_loss:.4f}, Energy MAE: {test_energy_mae:.4f}, RMSE: {test_energy_rmse:.4f}, R2: {test_energy_r2:.4f}, Force MAE: {test_force_mae:.4f}")

    # Plotting losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting metrics
    epochs_range = range(len(valid_losses))

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, valid_energy_maes, label='Validation Energy MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation Energy MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, valid_energy_rmses, label='Validation Energy RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation Energy RMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, valid_energy_r2s, label='Validation Energy R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.title('Validation Energy R2')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, valid_force_maes, label='Validation Force MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation Force MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
