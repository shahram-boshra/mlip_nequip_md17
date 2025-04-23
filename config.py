# config.py
import yaml
import os
from typing import Any, Dict, Optional
from exceptions import ConfigError, ConfigFileNotFoundError, ConfigFileParseError, ConfigAttributeError


class Config:
    """
    A hierarchical configuration manager that loads settings from a YAML file or dictionary.

    Provides attribute-based access to configuration values and handles nested configurations.
    Custom exceptions are raised for file not found, parsing errors, and attribute access issues.
    """
    def __init__(self, config_path: str = 'config.yaml', data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Config object.

        Loads configuration from the specified YAML file or uses the provided dictionary.
        Nested dictionaries within the configuration are automatically converted into nested Config objects
        for convenient hierarchical access.

        Args:
            config_path: Path to the YAML configuration file (default: 'config.yaml').
            data: Optional dictionary containing configuration data. If provided, the file is ignored.

        Raises:
            ConfigFileNotFoundError: If the specified configuration file does not exist.
            ConfigFileParseError: If there is an error parsing the YAML configuration file.
        """
        if data is None:
            try:
                with open(config_path, 'r') as f:
                    self._config: Dict[str, Any] = yaml.safe_load(f)
            except FileNotFoundError:
                raise ConfigFileNotFoundError(f"Configuration file '{config_path}' not found.")
            except yaml.YAMLError as e:
                raise ConfigFileParseError(f"Error parsing '{config_path}': {e}")
        else:
            self._config = data
        self._convert_nested_dicts_to_configs()

    def _convert_nested_dicts_to_configs(self) -> None:
        """Recursively converts nested dictionaries within the configuration into Config objects."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                self._config[key] = Config(data=value)

    def __getattr__(self, name: str) -> Any:
        """
        Provides attribute-based access to configuration values.

        Accessing a configuration key as an attribute (e.g., `config.database_url`).

        Args:
            name: The name of the configuration key to access.

        Returns:
            The value associated with the given key. If the value is a dictionary, it will be a nested Config object.

        Raises:
            ConfigAttributeError: If the requested attribute (configuration key) does not exist.
        """
        if name in self._config:
            return self._config[name]
        raise ConfigAttributeError(f"'Config' object has no attribute '{name}'")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Safely get a configuration value, with an optional default.

        Args:
            key: The configuration key to retrieve.
            default: An optional default value to return if the key is not found. Defaults to None.

        Returns:
            The value associated with the key, or the default value if the key is not found.
        """
        return self._config.get(key, default)


config = Config()
