# This code is a Qiskit project.
#
# (C) Copyright IBM and Cleveland Clinic Foundation 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Configuration management for quantum fragment methods.

This module provides utilities for loading and managing QPU configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_qpu_config(
    config_name: str = "ibm_quantum", config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load QPU configuration from YAML file.

    Args:
        config_name: Name of the configuration section to load (e.g., 'ibm_quantum')
        config_path: Path to the config file. If None, searches in standard locations:
                    1. Current directory
                    2. quantum_fragment_methods/config/
                    3. User's home directory (~/.qfm/)

    Returns:
        Dictionary containing the QPU configuration

    Raises:
        FileNotFoundError: If config file is not found
        ValueError: If config_name is not found in the config file

    Example:
        >>> config = load_qpu_config('ibm_quantum')
        >>> backend = IBMQuantumBackend(config)
    """
    # Search for config file in standard locations
    if config_path is None:
        search_paths = [
            Path.cwd() / "qpu_config.yaml",
            Path(__file__).parent / "qpu_config.yaml",
            Path.home() / ".qfm" / "qpu_config.yaml",
        ]

        config_path = None
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

        if config_path is None:
            raise FileNotFoundError(
                "qpu_config.yaml not found. Please create it from qpu_config.yaml.example\n"
                f"Searched in: {[str(p) for p in search_paths]}"
            )

    # Load YAML file
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    # Extract requested configuration
    if config_name not in all_configs:
        available = list(all_configs.keys())
        raise ValueError(
            f"Configuration '{config_name}' not found in {config_path}. "
            f"Available configurations: {available}"
        )

    return all_configs[config_name]


def create_default_config(output_path: Optional[str] = None) -> str:
    """Create a default qpu_config.yaml from the example template.

    Args:
        output_path: Where to create the config file. If None, creates in current directory.

    Returns:
        Path to the created config file

    Raises:
        FileExistsError: If config file already exists
    """
    if output_path is None:
        output_path = "qpu_config.yaml"

    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(
            f"Config file already exists at {output_path}. "
            "Please edit it manually or delete it first."
        )

    # Copy from example template
    example_path = Path(__file__).parent / "qpu_config.yaml.example"

    if not example_path.exists():
        raise FileNotFoundError(f"Example template not found at {example_path}")

    with open(example_path) as src:
        content = src.read()

    with open(output_file, "w") as dst:
        dst.write(content)

    print(f"Created config file at: {output_file}")
    print("Please edit it with your credentials before using.")

    return str(output_file)


__all__ = ["load_qpu_config", "create_default_config"]
