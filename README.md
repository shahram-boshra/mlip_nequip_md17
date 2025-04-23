# NequIP Training Toolkit for MD17 Property Prediction   

**Modular & Robust Utilities for Neural Equivariant Interatomic Potentials on the MD17 Dataset**

[![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%E2%82%82.0+-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric Version](https://img.shields.io/badge/PyTorch--Geometric-2.0+-green.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a focused and efficient set of Python utilities (`model.py`, `stopping_loss_utils.py`, `train_utils.py`, `graph_collate_utils.py`, and `exceptions.py`) specifically tailored for training NequIP (Neural Equivariant Interatomic Potentials) models for accurate molecular property prediction, with a particular emphasis on applications involving the MD17 dataset.

These modules offer a streamlined and well-structured framework for:

* **`model.py`**: Defining and implementing the core NequIP neural network architecture, including equivariant layers and feature processing optimized for molecular systems.
* **`stopping_loss_utils.py`**: Enhancing the training process with intelligent early stopping to prevent overfitting and a specialized loss function that effectively combines energy and force learning signals, crucial for NequIP models applied to datasets like MD17.
* **`train_utils.py`**: Providing robust and error-handled training, validation, and testing loops. These utilities manage optimization, learning rate scheduling, and comprehensive performance evaluation using relevant metrics commonly used in MD17 benchmarks.
* **`graph_collate_utils.py`**: Efficiently handling the conversion of MD17 molecular data into graph representations compatible with PyTorch Geometric and implementing custom batching strategies for optimized training on this dataset.
* **`exceptions.py`**: Defining a clear hierarchy of custom exception classes to ensure more informative and manageable error handling throughout the training pipeline for MD17-related tasks.

**Key Features:**

* **MD17 Focused (Adaptable):** While optimized with considerations for MD17, the modular design allows for adaptation to other molecular datasets.
* **Modular Design:** Each module promotes clarity and reusability, facilitating easy integration into MD17 training workflows.
* **Error Handling:** Comprehensive error handling ensures a more stable and debuggable training process.
* **Best Practices:** Implements state-of-the-art training techniques relevant to MD17 benchmarks, such as early stopping and gradient clipping.
* **NequIP Optimized:** Specifically tailored to the unique requirements of training Neural Equivariant Interatomic Potentials for molecular dynamics trajectory data.
* **Clear Documentation:** Each module and function is well-documented for easy understanding and use within the context of MD17 property prediction.

This repository aims to empower researchers and practitioners to efficiently and reliably train high-performing NequIP models for molecular property prediction on the MD17 dataset and similar molecular dynamics trajectory datasets.

## Repository Structure

.
├── exceptions.py
├── graph_collate_utils.py
├── model.py
├── README.md
├── stopping_loss_utils.py
└── train_utils.py


* **`exceptions.py`**: Defines custom exception classes for robust error management during various stages of MD17 model training.
* **`graph_collate_utils.py`**: Provides utilities for generating graph representations from MD17 molecular data and implementing custom collation for efficient batch processing.
* **`model.py`**: Contains the implementation of the NequIP model architecture, including crucial components like Gaussian smearing and tensor product layers, suitable for MD17 tasks.
* **`README.md`**: The current comprehensive guide to the repository and its contents, with a focus on MD17 applications.
* **`stopping_loss_utils.py`**: Implements the `EarlyStopping` callback to prevent overfitting on MD17 data and the `nequip_loss` function, a specialized loss combining energy and force contributions relevant to MD17 targets.
* **`train_utils.py`**: Offers well-structured functions for executing the training, validation, and testing phases on MD17 data, including optimization, learning rate scheduling, and performance metric calculation commonly used in MD17 evaluations.

## Getting Started

### Prerequisites

* Python 3.6+
* PyTorch (version compatible with PyTorch Geometric)
* PyTorch Geometric (`torch-geometric`)
* NumPy
* scikit-learn (`sklearn`)
* Logging module (standard Python library)
* **MD17 Dataset:** You will need to download and preprocess the MD17 dataset according to your specific needs.

Install the necessary dependencies using pip:

```bash
pip install torch torch_geometric numpy scikit-learn
Installation
Clone this repository to your local machine:

Bash

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
Usage
Python

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader  # Or your custom DataLoader using graph_collate_utils
from model import NequIP
from stopping_loss_utils import nequip_loss, EarlyStopping
from train_utils import train, valid

# Define your NequIP model
model = NequIP(...)

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define your learning rate scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Define your loss function
criterion = nequip_loss

# Create your DataLoaders (adapt for MD17 data and graph creation)
# Example: Assuming you have a way to load MD17 data into a list of graph-like objects
train_dataset = [...] # Your MD17 training dataset processed for graphs
valid_dataset = [...] # Your MD17 validation dataset processed for graphs
from graph_collate_utils import collate_graph_data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_graph_data)
valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_graph_data)

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, verbose=True, path='checkpoint.pt')

# Training loop
num_epochs = 100
energy_mean = 0.0 # Replace with your MD17 training set's energy mean
energy_std = 1.0  # Replace with your MD17 training set's energy std
force_mean = 0.0  # Replace with your MD17 training set's force mean
force_std = 1.0   # Replace with your MD17 training set's force std

for epoch in range(num_epochs):
    train_loss = train(model, criterion, optimizer, scheduler, train_loader, epoch, energy_mean, energy_std, force_mean, force_std)
    valid_loss, energy_mae, energy_rmse, energy_r2, force_mae = valid(
        model, criterion, valid_loader, epoch, early_stopping, energy_mean, energy_std, force_mean, force_std
    )

    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load the best model
# checkpoint = torch.load('checkpoint.pt')
# model.load_state_dict(checkpoint['model'])

# Perform testing using the 'test' function from train_utils.py
# test_dataset = [...] # Your MD17 test dataset processed for graphs
# test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_graph_data)
# test_loss, test_energy_mae, test_energy_rmse, test_energy_r2, test_force_mae = test(
#     model, criterion, test_loader, energy_mean, energy_std, force_mean, force_std
# )
# print(f"Test Loss: {test_loss:.4f}, Energy MAE: {test_energy_mae:.4f}, RMSE: {test_energy_rmse:.4f}, R2: {test_energy_r2:.4f}, Force MAE: {test_force_mae:.4f}")
Note: You will need to adapt the data loading and preprocessing steps to specifically handle the MD17 dataset format and how you generate molecular graphs from it. Ensure that your data can be processed by the generate_graph function (if you use the provided graph_collate_utils) after being loaded from the MD17 files. You might need to implement a custom dataset class or modify the generate_graph function to suit the MD17 data structure.

Contributing
Contributions to this repository are highly encouraged, especially those that improve the performance or usability of NequIP for the MD17 dataset.

License
This project is licensed under the Specify your license here, e.g., MIT License.

Acknowledgements
This work leverages the following open-source resources:

MD17 Dataset: We gratefully acknowledge the creators and maintainers of the MD17 dataset, a collection of molecular dynamics trajectories and corresponding energies and forces used for benchmarking machine learning potentials. The dataset is described in:
Christoph Baldauf, Ansgar Schütt, Franz Xaver Gigler, Patrick разъяснения, Robert J. Maurer, Mikhail Gastegger, Alexandre Tkatchenko, Klaus-Robert Müller, and Stefan Chmiela. High-dimensional neural network potentials for accurate predictions of molecular vibrations and thermodynamic properties. Scientific Data 4, 170193 (2017). https://doi.org/10.1038/sdata.2017.193

PyTorch: The fundamental machine learning framework used in this project. We acknowledge the PyTorch development team for creating and maintaining this powerful library. (https://pytorch.org/)
PyTorch Geometric: The library used for handling graph-structured data and implementing graph neural networks, including components relevant to NequIP. We thank the PyTorch Geometric team for their excellent work. (https://pytorch-geometric.readthedocs.io/en/latest/)
NumPy: A fundamental package for numerical computation in Python. (https://numpy.org/)
scikit-learn: A comprehensive library for machine learning in Python, used here for evaluating model performance with metrics like MAE, RMSE, and R2 score. (https://scikit-learn.org/)

