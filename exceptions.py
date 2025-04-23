# exceptions.py
import yaml

"""
This module defines custom exception classes for the project, providing more specific
error handling and clearer debugging information. These exceptions are organized
by the area of the codebase where they might occur.
"""

class InitializationError(Exception):
    """
    Raised when an error occurs during the initialization phase of a module or component.
    This could include issues with setting up internal state or dependencies.
    """
    pass

class ForwardError(Exception):
    """
    Raised when an error occurs during the forward pass of a neural network module.
    This typically indicates a problem with the data flow or operations within the module.
    """
    pass

class TensorShapeError(ValueError):
    """
    Raised when a tensor has an unexpected or invalid shape.
    Inherits from ValueError to align with standard Python shape-related errors.
    """
    pass

class TensorTypeError(TypeError):
    """
    Raised when a variable is expected to be a torch.Tensor but is of a different type.
    Inherits from TypeError to align with standard Python type-related errors.
    """
    pass

class ParameterError(TypeError):
    """
    Raised when a parameter provided to a function or module has an invalid type.
    Inherits from TypeError to align with standard Python type-related errors.
    """
    pass

class DataLoadingError(Exception):
    """
    Raised when an error occurs during the process of loading the dataset.
    This could involve issues with file access, data format, or data preprocessing.
    """
    pass

class GraphGenerationError(Exception):
    """
    Raised when an error occurs during the generation or creation of a graph structure.
    This is relevant for graph neural networks and related tasks.
    """
    pass

class GraphCollationError(Exception):
    """
    Raised when an error occurs while collating a batch of graph data.
    This is specific to how individual graph samples are combined into a single batch.
    """
    pass

class DataLoaderCreationError(Exception):
    """
    Raised when an error occurs during the creation or initialization of a DataLoader.
    This could involve issues with the dataset, batch size, or sampler.
    """
    pass

class ModelInstantiationError(Exception):
    """
    Raised when an error occurs during the instantiation or creation of a model.
    This could involve issues with model architecture or provided arguments.
    """
    pass

class CheckpointLoadingError(Exception):
    """
    Raised when an error occurs while loading a model checkpoint from a file.
    This could involve issues with file access, format, or compatibility.
    """
    pass

class ModelSavingError(Exception):
    """
    Raised when an error occurs while saving a model's state to a file.
    This could involve issues with file access or serialization.
    """
    pass

class ValueError(ValueError):
    """
    Custom ValueError specific to the project for general value-related errors
    that don't fit into more specific custom exceptions.
    """
    pass

class TypeError(TypeError):
    """
    Custom TypeError specific to the project for general type-related errors
    that don't fit into more specific custom exceptions.
    """
    pass

class BatchTypeError(TypeError):
    """
    Raised when a batch of data is not of the expected type.
    This helps ensure data integrity during training and evaluation.
    """
    pass

class ConfigError(Exception):
    """
    Base class for all configuration-related exceptions in the project.
    Provides a common ancestor for easier handling of configuration issues.
    """
    pass

class ConfigFileNotFoundError(ConfigError, FileNotFoundError):
    """
    Raised when the specified configuration file cannot be found.
    Inherits from both ConfigError and FileNotFoundError for clearer error identification.
    """
    pass

class ConfigFileParseError(ConfigError, yaml.YAMLError):
    """
    Raised when there is an error parsing the configuration file (e.g., invalid YAML syntax).
    Inherits from both ConfigError and yaml.YAMLError to provide context about the parsing issue.
    """
    pass

class ConfigAttributeError(ConfigError, AttributeError):
    """
    Raised when an expected attribute is not found within the loaded configuration.
    Inherits from both ConfigError and AttributeError for clearer error identification.
    """
    pass
