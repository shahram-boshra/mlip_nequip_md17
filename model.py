# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch, Data
import logging
from torch_geometric.utils import scatter
from typing import Any, Dict, Optional, Union
from exceptions import InitializationError, ForwardError, TensorShapeError, TensorTypeError, ParameterError


logger = logging.getLogger(__name__)


class GaussianSmearing(nn.Module):
    """
    Applies Gaussian smearing to a tensor of distances.

    This layer expands scalar distances into a set of Gaussian basis functions.
    This representation can then be used as input features in subsequent
    layers of a neural network.

    Args:
        start (float, optional): The center of the first Gaussian. Defaults to 0.0.
        stop (float, optional): The center of the last Gaussian. Defaults to 5.0.
        num_gaussians (int, optional): The number of Gaussian basis functions. Defaults to 50.

    Raises:
        InitializationError: If input arguments are invalid (e.g., non-numeric start/stop,
            non-positive integer num_gaussians, or start >= stop).

    Attributes:
        centers (torch.Tensor): The centers of the Gaussian basis functions.
        width (float): The width of the Gaussian basis functions.
        scaling (float): The scaling factor for the Gaussian functions.
    """
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 100) -> None:
        super().__init__()
        if not isinstance(start, (int, float)):
            raise InitializationError(f"Expected start to be a number, but got {type(start).__name__}")
        if not isinstance(stop, (int, float)):
            raise InitializationError(f"Expected stop to be a number, but got {type(stop).__name__}")
        if not isinstance(num_gaussians, int) or num_gaussians <= 0:
            raise InitializationError(f"Expected num_gaussians to be a positive integer, but got {num_gaussians}")
        if start >= stop:
            raise InitializationError("Start value must be less than stop value.")

        try:
            self.centers = torch.linspace(start, stop, num_gaussians)
            if num_gaussians > 1:
                self.width = (stop - start) / (num_gaussians - 1)
            else:
                self.width = 1.0
                logger.warning("Number of Gaussians is 1, setting width to 1.0.")
            self.scaling = 1.0 / torch.sqrt(torch.tensor(2.0) * torch.pi)
        except RuntimeError as e:
            logger.error(f"RuntimeError during GaussianSmearing initialization: {e}")
            raise InitializationError(f"Error during initialization: {e}") from e

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gaussian smearing of the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances. Can be a scalar, a 1D vector, or a 2D matrix.

        Returns:
            torch.Tensor: A tensor where each distance has been expanded into a set of Gaussian values.
                The output shape will be `distances.shape + (num_gaussians,)`.

        Raises:
            TensorTypeError: If the input `distances` is not a torch.Tensor.
            TensorShapeError: If the input `distances` has more than 2 dimensions.
            ForwardError: If a RuntimeError or ArithmeticError occurs during the forward pass.
        """
        if not isinstance(distances, torch.Tensor):
            raise TensorTypeError(f"Expected distances to be a torch.Tensor, but got {type(distances).__name__}")
        if distances.ndim == 0:
            distances = distances.unsqueeze(0)
        if distances.ndim == 1:
            distances = distances.unsqueeze(-1)
        if distances.ndim > 2:
            raise TensorShapeError(f"Expected distances to have at most 2 dimensions, but got {distances.ndim}")

        try:
            distances = distances - self.centers
            return self.scaling * torch.exp(-0.5 * (distances / self.width) ** 2)
        except RuntimeError as e:
            logger.error(f"RuntimeError in GaussianSmearing forward: {e}, input shape: {distances.shape}")
            raise ForwardError(f"Runtime error during forward pass: {e}") from e
        except ArithmeticError as e:
            logger.error(f"ArithmeticError in GaussianSmearing forward: {e}, width: {self.width}")
            raise ForwardError(f"Arithmetic error during forward pass: {e}") from e


class TensorProduct(nn.Module):
    """
    Performs a tensor product operation between node features and radial basis functions.

    This layer takes in features from neighboring nodes (`x_j`), radial basis function
    values (`rbf`) corresponding to the edges, and edge vectors (`edge_vector`). It
    computes a weighted combination of the neighbor features based on the radial basis
    functions.

    Args:
        in_channels (int): The number of input channels for the node features.
        out_channels (int): The number of output channels for the resulting features.
        num_radial (int): The number of radial basis functions used.

    Raises:
        InitializationError: If input channel numbers or `num_radial` are not positive integers.

    Attributes:
        weight (nn.Parameter): Learnable weight tensor of shape `(in_channels, num_radial, out_channels)`.
    """
    def __init__(self, in_channels: int, out_channels: int, num_radial: int) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise InitializationError(f"Expected in_channels to be a positive integer, but got {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise InitializationError(f"Expected out_channels to be a positive integer, but got {out_channels}")
        if not isinstance(num_radial, int) or num_radial <= 0:
            raise InitializationError(f"Expected num_radial to be a positive integer, but got {num_radial}")
        self.weight = nn.Parameter(torch.Tensor(in_channels, num_radial, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes the weight parameters using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, rbf: torch.Tensor, edge_vector: torch.Tensor) -> torch.Tensor:
        """
        Performs the tensor product operation.

        Args:
            x_i (torch.Tensor): Features of the central nodes, shape `(num_edges, in_channels)`.
            x_j (torch.Tensor): Features of the neighboring nodes, shape `(num_edges, in_channels)`.
            rbf (torch.Tensor): Radial basis function values for each edge, shape `(num_edges, num_radial)`.
            edge_vector (torch.Tensor): The vector pointing from neighbor nodes to central nodes, shape `(num_edges, 3)`.
                While the edge vector is not directly used in the tensor product itself in this implementation,
                it is typically an input for subsequent operations in a geometric neural network.

        Returns:
            torch.Tensor: The result of the tensor product, shape `(num_edges, num_radial, out_channels)`.

        Raises:
            TensorShapeError: If the input tensor shapes are inconsistent.
            ForwardError: If a RuntimeError, ValueError, or other unexpected error occurs during the forward pass.
        """
        try:
            num_edges = rbf.size(0)
            in_channels_weight = self.weight.size(0)
            out_channels_weight = self.weight.size(2)
            num_radial_weight = self.weight.size(1)

            if x_j.size(0) != num_edges:
                raise TensorShapeError(f"Number of nodes in x_j ({x_j.size(0)}) must match the number of edges ({num_edges}).")
            if rbf.size(1) != num_radial_weight:
                raise TensorShapeError(f"Number of radial basis functions ({rbf.size(1)}) must match num_radial ({num_radial_weight}).")
            if x_j.size(1) != in_channels_weight:
                raise TensorShapeError(f"Input channel dimension of x_j ({x_j.size(1)}) must match in_channels ({in_channels_weight}).")
            if edge_vector.size(0) != num_edges:
                raise TensorShapeError(f"Number of edge vectors ({edge_vector.size(0)}) must match the number of edges ({num_edges}).")

            weighted_x = torch.einsum('ec,cro->ero', x_j, self.weight)
            output = weighted_x * rbf.unsqueeze(-1)
            return output
        except RuntimeError as e:
            logger.error(f"RuntimeError in TensorProduct forward: {e}, x_j shape: {x_j.shape}, rbf shape: {rbf.shape}, edge_vector shape: {edge_vector.shape}, weight shape: {self.weight.shape}")
            raise ForwardError(f"Runtime error during forward pass: {e}") from e
        except ValueError as e:
            logger.error(f"ValueError in TensorProduct forward: {e}")
            raise ForwardError(f"Value error during forward pass: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in TensorProduct forward: {e}")
            raise ForwardError(f"Unexpected error during forward pass: {e}") from e


class NequIPConv(nn.Module):
    """
    A single convolutional layer in the NequIP architecture.

    This layer performs message passing between neighboring atoms, updating the
    feature vectors of each atom based on its neighbors and their relative positions.
    It utilizes Gaussian smearing to encode distances and a tensor product for
    combining features.

    Args:
        in_channels (int): The number of input channels for the atomic features.
        num_radial (int): The number of radial basis functions to use.
        cutoff (float): The cutoff distance for considering neighboring atoms.

    Raises:
        InitializationError: If input channel number, `num_radial`, or `cutoff` are not positive.

    Attributes:
        cutoff (float): The cutoff distance.
        radial_basis (GaussianSmearing): The Gaussian smearing layer for encoding distances.
        tensor_product (TensorProduct): The tensor product layer for combining features.
        mlp (nn.Sequential): A multi-layer perceptron for further processing of the aggregated features.
        in_channels (int): Stores the input channel dimension.
        num_radial (int): Stores the number of radial basis functions.
    """
    def __init__(self, in_channels: int, num_radial: int, cutoff: float) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise InitializationError(f"Expected in_channels to be a positive integer, but got {in_channels}")
        if not isinstance(num_radial, int) or num_radial <= 0:
            raise InitializationError(f"Expected num_radial to be a positive integer, but got {num_radial}")
        if not isinstance(cutoff, (int, float)) or cutoff <= 0:
            raise InitializationError(f"Expected cutoff to be a positive number, but got {cutoff}")

        self.cutoff = cutoff
        try:
            self.radial_basis = GaussianSmearing(stop=cutoff, num_gaussians=num_radial)
            self.tensor_product = TensorProduct(in_channels, in_channels * 3, num_radial)
            self.mlp = nn.Sequential(
                nn.Linear(in_channels * num_radial + in_channels, in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels * 3)
            )
            self.in_channels = in_channels
            self.num_radial = num_radial
        except Exception as e:
            logger.error(f"Error during NequIPConv initialization: {e}")
            raise InitializationError(f"Error during initialization: {e}") from e

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the NequIP convolution layer.

        Args:
            x (torch.Tensor): Atomic features of shape `(num_nodes, in_channels)`.
            pos (torch.Tensor): Atomic positions of shape `(num_nodes, 3)`.
            edge_index (torch.Tensor): Graph connectivity in COO format of shape `(2, num_edges)`.

        Returns:
            torch.Tensor: Updated atomic features of shape `(num_nodes, 3 * in_channels)`.

        Raises:
            TensorTypeError: If input tensors are not of the expected type or dimensionality.
            TensorShapeError: If the shapes of the input tensors are inconsistent.
            ForwardError: If a RuntimeError occurs during the forward pass.
        """
        try:
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                raise TensorTypeError(f"Expected x to be a 2D torch.Tensor, but got {type(x).__name__} with shape {x.shape if isinstance(x, torch.Tensor) else None}")
            if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.size(1) != 3:
                raise TensorTypeError(f"Expected pos to be a 2D torch.Tensor with 3 columns, but got {type(pos).__name__} with shape {pos.shape if isinstance(pos, torch.Tensor) else None}")
            if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise TensorTypeError(f"Expected edge_index to be a 2D torch.Tensor with 2 rows, but got {type(edge_index).__name__} with shape {edge_index.shape if isinstance(edge_index, torch.Tensor) else None}")

            row, col = edge_index
            edge_vector = pos[row] - pos[col]
            distances = torch.norm(edge_vector, dim=-1)
            radial_basis = self.radial_basis(distances)

            mask = distances < self.cutoff
            edge_index = edge_index[:, mask]
            edge_vector = edge_vector[mask]
            radial_basis = radial_basis[mask]
            row, col = edge_index

            tp_out = self.message(x[col], x[row], radial_basis, edge_vector)
            aggregated_tp = self.aggregate(tp_out, row, dim_size=x.size(0))

            weighted_xj = x[col].unsqueeze(-1) * radial_basis.unsqueeze(1)
            aggregated_weighted_xj = scatter(weighted_xj, row, dim=0, dim_size=x.size(0), reduce='sum')
            aggregated_weighted_xj = aggregated_weighted_xj.view(x.size(0), -1)

            mlp_input = torch.cat([aggregated_weighted_xj, x], dim=-1)
            updated_x = self.mlp(mlp_input)
            return updated_x

        except RuntimeError as e:
            logger.error(f"RuntimeError in NequIPConv forward: {e}, input x shape: {x.shape}, pos shape: {pos.shape}, edge_index shape: {edge_index.shape}")
            raise ForwardError(f"Runtime error during forward pass: {e}") from e
        except TensorTypeError as e:
            logger.error(f"TypeError in NequIPConv forward: {e}")
            raise
        except TensorShapeError as e:
            logger.error(f"ValueError in NequIPConv forward: {e}")
            raise

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, rbf: torch.Tensor, edge_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes the messages passed between neighboring atoms.

        Args:
            x_j (torch.Tensor): Features of the neighboring atoms, shape `(num_edges, in_channels)`.
            x_i (torch.Tensor): Features of the central atoms, shape `(num_edges, in_channels)`.
            rbf (torch.Tensor): Radial basis function values for each edge, shape `(num_edges, num_radial)`.
            edge_vector (torch.Tensor): The vector pointing from neighbor nodes to central nodes, shape `(num_edges, 3)`.

        Returns:
            torch.Tensor: The resulting messages after the tensor product, shape `(num_edges, num_radial, 3 * in_channels)`.

        Raises:
            TensorTypeError: If input tensors are not of the expected type or dimensionality.
            ForwardError: If an error occurs during the message passing.
        """
        try:
            if not isinstance(x_j, torch.Tensor) or x_j.ndim != 2:
                raise TensorTypeError(f"Expected x_j to be a 2D torch.Tensor, but got {type(x_j).__name__} with shape {x_j.shape if isinstance(x_j, torch.Tensor) else None}")
            if not isinstance(x_i, torch.Tensor) or x_i.ndim != 2:
                raise TensorTypeError(f"Expected x_i to be a 2D torch.Tensor, but got {type(x_i).__name__} with shape {x_i.shape if isinstance(x_i, torch.Tensor) else None}")
            if not isinstance(rbf, torch.Tensor) or rbf.ndim != 2:
                raise TensorTypeError(f"Expected rbf to be a 2D torch.Tensor, but got {type(rbf).__name__} with shape {rbf.shape if isinstance(rbf, torch.Tensor) else None}")
            if not isinstance(edge_vector, torch.Tensor) or edge_vector.ndim != 2:
                raise TensorTypeError(f"Expected edge_vector to be a 2D torch.Tensor, but got {type(edge_vector).__name__} with shape {edge_vector.shape if isinstance(edge_vector, torch.Tensor) else None}")

            weighted_xj = x_j.unsqueeze(1) * rbf.unsqueeze(-1)
            tp_out = self.tensor_product(x_i, x_j, rbf, edge_vector)
            return tp_out

        except Exception as e:
            logger.error(f"Error in NequIPConv message: {e}")
            raise ForwardError(f"Error during message passing: {e}") from e

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr: Union[torch.Tensor, None] = None, dim_size: Union[int, None] = None) -> torch.Tensor:
        """
        Aggregates the incoming messages based on the central atom index.

        Args:
            inputs (torch.Tensor): The messages to aggregate, shape `(num_edges, num_radial, 3 * in_channels)`.
            index (torch.Tensor): The indices of the central atoms for each edge, shape `(num_edges,)`.
            ptr (torch.Tensor, optional): If `index` is not sorted, can be used to indicate the start
                and end indices of each graph in the batch. Defaults to None.
            dim_size (int, optional): The number of nodes in the graph(s). If None, it is inferred
                from the maximum value in `index`. Defaults to None.

        Returns:
            torch.Tensor: The aggregated messages for each atom, shape `(num_nodes, num_radial, 3 * in_channels)`.

        Raises:
            TensorTypeError: If `inputs` or `index` are not of the expected type or dimensionality.
            ForwardError: If a RuntimeError occurs during the aggregation.
        """
        try:
            if not isinstance(inputs, torch.Tensor):
                raise TensorTypeError(f"Expected inputs to be a torch.Tensor, but got {type(inputs).__name__}")
            if not isinstance(index, torch.Tensor) or index.dtype != torch.long or index.ndim != 1:
                raise TensorTypeError(f"Expected index to be a 1D torch.LongTensor, but got {type(index).__name__} with dtype {index.dtype} and shape {index.shape if isinstance(index, torch.Tensor) else None}")
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
        except RuntimeError as e:
            logger.error("--- RuntimeError in aggregate (scatter) ---")
            logger.error(f"Shape of inputs: {inputs.shape}")
            logger.error(f"Shape of index: {index.shape}")
            logger.error(f"dim_size: {dim_size}")
            logger.error(f"Error: {e}")
            raise ForwardError(f"Runtime error during aggregation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in aggregate: {e}")
            raise ForwardError(f"Unexpected error during aggregation: {e}") from e

    def message_and_aggregate(self, edge_index: torch.Tensor, x: torch.Tensor, rbf: torch.Tensor, edge_vector: torch.Tensor) -> torch.Tensor:
        """
        Combines the message passing and aggregation steps.

        Args:
            edge_index (torch.Tensor): Graph connectivity in COO format of shape `(2, num_edges)`.
            x (torch.Tensor): Atomic features of shape `(num_nodes, in_channels)`.
            rbf (torch.Tensor): Radial basis function values for each edge, shape `(num_edges, num_radial)`.
            edge_vector (torch.Tensor): The vector pointing from neighbor nodes to central nodes, shape `(num_edges, 3)`.

        Returns:
            torch.Tensor: The aggregated messages for each atom, shape `(num_nodes, num_radial, 3 * in_channels)`.

        Raises:
            TensorTypeError: If input tensors are not of the expected type or dimensionality.
            ForwardError: If a RuntimeError occurs during the message and aggregation.
        """
        try:
            if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise TensorTypeError(f"Expected edge_index to be a 2D torch.Tensor with 2 rows, but got {type(edge_index).__name__} with shape {edge_index.shape if isinstance(edge_index, torch.Tensor) else None}")
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                raise TensorTypeError(f"Expected x to be a 2D torch.Tensor, but got {type(x).__name__} with shape {x.shape if isinstance(x, torch.Tensor) else None}")
            if not isinstance(rbf, torch.Tensor) or rbf.ndim != 2:
                raise TensorTypeError(f"Expected rbf to be a 2D torch.Tensor, but got {type(rbf).__name__} with shape {rbf.shape if isinstance(rbf, torch.Tensor) else None}")
            if not isinstance(edge_vector, torch.Tensor) or edge_vector.ndim != 2:
                raise TensorTypeError(f"Expected edge_vector to be a 2D torch.Tensor, but got {type(edge_vector).__name__} with shape {edge_vector.shape if isinstance(edge_vector, torch.Tensor) else None}")

            row, col = edge_index
            tp_out = self.message(x[col], x[row], rbf, edge_vector)
            aggregated_output = self.aggregate(tp_out, row, dim_size=x.size(0))
            return aggregated_output
        except RuntimeError as e:
            logger.error("--- RuntimeError in message_and_aggregate (aggregate call) ---")
            logger.error(f"Shape of tp_out: {tp_out.shape}")
            logger.error(f"Shape of row (index): {row.shape}")
            logger.error(f"dim_size: {x.size(0)}")
            logger.error(f"Error: {e}")
            raise ForwardError(f"Runtime error during message and aggregation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in message_and_aggregate: {e}")
            raise ForwardError(f"Unexpected error during message and aggregation: {e}") from e

class NequIP(nn.Module):
    """
    The main NequIP (Network for Equivariant Interatomic Potentials) model.

    This model predicts graph-level properties (e.g., energy) and node-level
    properties (e.g., forces) of atomic systems. It consists of an embedding layer
    followed by multiple NequIP convolutional layers and final linear layers
    for property prediction.

    Args:
        num_atom_types (int): The number of different atom types in the system.
        hidden_channels (int): The number of hidden channels in the convolutional layers.
        num_radial (int): The number of radial basis functions used in the convolutional layers.
        num_layers (int): The number of NequIP convolutional layers.
        cutoff (float): The cutoff distance for neighbor interactions.
        graph_out_channels (int, optional): The number of output channels for graph-level predictions (e.g., energy). Defaults to 1.
        force_out_channels (int, optional): The number of output channels for node-level predictions (e.g., forces). Defaults to 3.

    Raises:
        InitializationError: If any of the input arguments are invalid (e.g., non-positive integers/floats).

    Attributes:
        embedding (nn.Embedding): The embedding layer for atomic numbers.
        convs (nn.ModuleList): A list of NequIP convolutional layers.
        energy_lin1 (nn.Linear): The first linear layer for energy prediction.
        energy_lin2 (nn.Linear): The second linear layer for energy prediction.
        force_lin (nn.Linear): The linear layer for force prediction.
        num_layers (int): Stores the number of convolutional layers.
        hidden_channels (int): Stores the number of hidden channels.
        num_radial (int): Stores the number of radial basis functions.
        cutoff (float): Stores the cutoff distance.
        graph_out_channels (int): Stores the number of graph output channels.
        force_out_channels (int): Stores the number of force output channels.
    """
    def __init__(self, num_atom_types: int, hidden_channels: int, num_radial: int, num_layers: int, cutoff: float, graph_out_channels: int = 1, force_out_channels: int = 3) -> None:
        super().__init__()
        if not isinstance(num_atom_types, int) or num_atom_types <= 0:
            raise InitializationError(f"Expected num_atom_types to be a positive integer, but got {num_atom_types}")
        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise InitializationError(f"Expected hidden_channels to be a positive integer, but got {hidden_channels}")
        if not isinstance(num_radial, int) or num_radial <= 0:
            raise InitializationError(f"Expected num_radial to be a positive integer, but got {num_radial}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise InitializationError(f"Expected num_layers to be a positive integer, but got {num_layers}")
        if not isinstance(cutoff, (int, float)) or cutoff <= 0:
            raise InitializationError(f"Expected cutoff to be a positive number, but got {cutoff}")
        if not isinstance(graph_out_channels, int) or graph_out_channels <= 0:
            raise InitializationError(f"Expected graph_out_channels to be a positive integer, but got {graph_out_channels}")
        if not isinstance(force_out_channels, int) or force_out_channels <= 0:
            raise InitializationError(f"Expected force_out_channels to be a positive integer, but got {force_out_channels}")

        try:
            self.embedding = nn.Embedding(num_atom_types, hidden_channels)
            conv_layers = nn.ModuleList()
            in_channels = hidden_channels
            for i in range(num_layers):
                conv = NequIPConv(in_channels, num_radial, cutoff)
                conv_layers.append(conv)
                in_channels = conv.mlp[-1].out_features
            self.convs = conv_layers
            self.energy_lin1 = nn.Linear(in_channels, hidden_channels)
            self.energy_lin2 = nn.Linear(hidden_channels, graph_out_channels)
            self.force_lin = nn.Linear(in_channels, force_out_channels)

            self.num_layers = num_layers
            self.hidden_channels = hidden_channels
            self.num_radial = num_radial
            self.cutoff = cutoff
            self.graph_out_channels = graph_out_channels
            self.force_out_channels = force_out_channels

        except ValueError as e:
            logger.error(f"ValueError during NequIP initialization: {e}")
            raise InitializationError(f"Value error during initialization: {e}") from e
        except TypeError as e:
            logger.error(f"TypeError during NequIP initialization: {e}")
            raise InitializationError(f"Type error during initialization: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during NequIP initialization: {e}")
            raise InitializationError(f"Unexpected error during initialization: {e}") from e

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the NequIP model.

        Args:
            data (torch_geometric.data.Batch): A PyTorch Geometric Batch object containing the graph data,
                including atomic numbers (`z`), atomic positions (`pos`), edge indices (`edge_index`),
                and batch assignments (`batch`).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - graph_out (torch.Tensor): Graph-level predictions (e.g., energy) of shape `(batch_size, graph_out_channels)`.
                - force_out (torch.Tensor): Node-level predictions (e.g., forces) of shape `(num_nodes, force_out_channels)`.
                - pos (torch.Tensor): The original atomic positions of shape `(num_nodes, 3)`.

        Raises:
            AttributeError: If the input `data` object is missing required attributes.
            TensorTypeError: If the attributes of the input `data` object are not of the expected type.
            ForwardError: If a RuntimeError occurs during the forward pass.
        """
        try:
            if not isinstance(data, Batch):
                raise TensorTypeError(f"Expected input to be a torch_geometric.data.Batch object, but got {type(data).__name__}")
            if not hasattr(data, 'z') or not isinstance(data.z, torch.Tensor):
                raise AttributeError("Input Batch object must have 'z' attribute (atomic numbers).")
            if not hasattr(data, 'pos') or not isinstance(data.pos, torch.Tensor):
                raise AttributeError("Input Batch object must have 'pos' attribute (atomic positions).")
            if not hasattr(data, 'edge_index') or not isinstance(data.edge_index, torch.Tensor):
                raise AttributeError("Input Batch object must have 'edge_index' attribute.")
            if not hasattr(data, 'batch') or not isinstance(data.batch, torch.Tensor):
                raise AttributeError("Input Batch object must have 'batch' attribute.")

            z, pos, edge_index, batch = data.z, data.pos, data.edge_index, data.batch
            x = self.embedding(z)

            for i, conv in enumerate(self.convs):
                x = conv(x, pos, edge_index)

            graph_out = global_mean_pool(x, batch)
            graph_out = self.energy_lin1(graph_out).relu()
            graph_out = self.energy_lin2(graph_out)

            force_out = self.force_lin(x)

            return graph_out, force_out, pos

        except AttributeError as e:
            logger.error(f"AttributeError in NequIP forward: {e}, likely missing attributes in the input data object.")
            raise ForwardError(f"Missing data attribute during forward pass: {e}") from e
        except TensorTypeError as e:
            logger.error(f"TypeError in NequIP forward: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"RuntimeError in NequIP forward: {e}")
            raise ForwardError(f"Runtime error during forward pass: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in NequIP forward: {e}")
            raise ForwardError(f"Unexpected error during forward pass: {e}") from e
