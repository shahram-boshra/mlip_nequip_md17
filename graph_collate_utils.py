# graph_collate_utils.py
import torch
from torch_geometric.data import Batch, Data
import logging
from exceptions import GraphGenerationError, GraphCollationError
from config import config
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


def generate_graph(data: Data, cutoff: Optional[Union[int, float]] = None) -> Data:
    """Generates an edge-attributed graph from node positions based on a distance cutoff.

    Given a PyTorch Geometric Data object containing node positions, this function constructs
    the graph's edge index and edge attributes. Edges are formed between pairs of nodes
    whose Euclidean distance is less than the specified cutoff. The edge attribute is the
    Euclidean distance between the connected nodes.

    Args:
        data (Data): A PyTorch Geometric Data object containing node positions in `data.pos`
                      (a 2D tensor of shape [num_nodes, 3]).
        cutoff (Optional[Union[int, float]], optional): The distance threshold for creating
            edges. If None, the default cutoff distance from the global configuration is used.
            Defaults to None.

    Returns:
        Data: The input Data object with added `edge_index` (a 2D tensor of shape [2, num_edges])
              and `edge_attr` (a 2D tensor of shape [num_edges, 1]) attributes. If no edges
              are formed, `edge_attr` will be an empty tensor.

    Raises:
        GraphGenerationError: If the input `data` is not a `torch_geometric.data.Data` object,
            if `cutoff` is not a number, or if `data.pos` is not a 2D tensor with 3 columns.
        TypeError: If there are type-related issues during tensor operations.
        RuntimeError: If there are runtime errors during tensor operations.
        Exception: For any other unexpected errors during graph generation.
    """
    if cutoff is None:
        cutoff = config.data.cutoff_distance
    try:
        if not isinstance(data, Data):
            raise GraphGenerationError(f"Expected data to be a torch_geometric.data.Data object, but got {type(data).__name__}")
        if not isinstance(cutoff, (int, float)):
            raise GraphGenerationError(f"Expected cutoff to be a number, but got {type(cutoff).__name__}")
        pos = data.pos
        if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.size(1) != 3:
            raise GraphGenerationError(f"Expected data.pos to be a 2D torch.Tensor with 3 columns, but got {type(pos).__name__} with shape {pos.shape if isinstance(pos, torch.Tensor) else None}")
        num_nodes = pos.size(0)
        row, col = [], []
        edge_attr_list = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist_vec = pos[i] - pos[j]
                dist = torch.sqrt(torch.sum(dist_vec ** 2))
                if dist < cutoff:
                    row.append(i)
                    col.append(j)
                    edge_attr_list.append(dist.unsqueeze(-1))

                    row.append(j)
                    col.append(i)
                    edge_attr_list.append(dist.unsqueeze(-1))

        edge_index = torch.tensor([row, col], dtype=torch.long)
        data.edge_index = edge_index
        if edge_attr_list:
            data.edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            data.edge_attr = torch.empty(0, 1)
        return data
    except GraphGenerationError as e:
        logger.error(f"GraphGenerationError: {e}")
        raise
    except TypeError as e:
        logger.error(f"TypeError in generate_graph: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"RuntimeError in generate_graph: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_graph: {e}")
        raise

def collate_graph_data(batch: List[Data]) -> Batch:
    """Collates a list of PyTorch Geometric Data objects into a single Batch object.

    This function takes a list of `torch_geometric.data.Data` objects and uses
    `torch_geometric.data.Batch.from_data_list` to combine them into a batched
    graph representation suitable for graph neural network training or inference.

    Args:
        batch (List[Data]): A list of PyTorch Geometric Data objects to be collated.

    Returns:
        Batch: A PyTorch Geometric Batch object containing the combined graph data.

    Raises:
        GraphCollationError: If any item in the input `batch` is not a
            `torch_geometric.data.Data` object.
        TypeError: If there are type-related issues during batch creation.
        Exception: For any other unexpected errors during collation.
    """
    try:
        if not all(isinstance(item, Data) for item in batch):
            raise GraphCollationError("Expected each item in the batch to be a torch_geometric.data.Data object.")
        return Batch.from_data_list(batch)
    except GraphCollationError as e:
        logger.error(f"GraphCollationError: {e}")
        raise
    except TypeError as e:
        logger.error(f"TypeError in collate_graph_data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in collate_graph_data: {e}")
        raise
