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

"""Base abstract interface for quantum processing unit (QPU) backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class QPUBackend(ABC):
    """Abstract base class for quantum hardware backends.

    This class defines the interface that all QPU backend implementations
    must follow.

    Attributes:
        config: Configuration dictionary for the backend
        backend: The underlying backend instance
        service: The quantum service instance
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the QPU backend.

        Args:
            config: Configuration dictionary containing backend settings.
                   Expected keys depend on the specific implementation.
        """
        self.config = config
        self.backend: Optional[Any] = None
        self.service: Optional[Any] = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize connection to the QPU service.

        This method should establish the connection to the quantum service
        and authenticate using credentials from the config.

        Raises:
            ConnectionError: If connection to the service fails
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    def get_backend(self, backend_name: Optional[str] = None) -> Any:
        """Get a specific backend instance.

        Args:
            backend_name: Name of the backend to retrieve. If None, uses
                         the backend specified in config.

        Returns:
            Backend instance for the specified quantum hardware

        Raises:
            ValueError: If backend_name is invalid or not available
        """
        pass

    @abstractmethod
    def create_sampler(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create a sampler primitive for the backend.

        Args:
            backend: Backend instance to use. If None, uses self.backend
            options: Dictionary of sampler configuration options such as:
                    - shots: Number of measurement shots
                    - dynamical_decoupling: DD configuration
                    - twirling: Twirling configuration
                    - optimization_level: Transpilation optimization level

        Returns:
            Configured sampler primitive instance

        Raises:
            ValueError: If backend is not set and not provided
        """
        pass

    @abstractmethod
    def create_estimator(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create an estimator primitive for the backend.

        Args:
            backend: Backend instance to use. If None, uses self.backend
            options: Dictionary of estimator configuration options

        Returns:
            Configured estimator primitive instance

        Raises:
            ValueError: If backend is not set and not provided
        """
        pass

    @abstractmethod
    def submit_job(self, circuits: List[Any], primitive: Any, **kwargs) -> str:
        """Submit quantum circuits for execution.

        Args:
            circuits: List of quantum circuits to execute
            primitive: Sampler or Estimator primitive to use
            **kwargs: Additional arguments for job submission

        Returns:
            Job ID string for tracking the submitted job

        Raises:
            RuntimeError: If job submission fails
        """
        pass

    @abstractmethod
    def retrieve_job(self, job_id: str) -> Any:
        """Retrieve a job by its ID.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Job object

        Raises:
            ValueError: If job_id is invalid
            RuntimeError: If job retrieval fails
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """Get the status of a job.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Status string (e.g., 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')

        Raises:
            ValueError: If job_id is invalid
        """
        pass

    @abstractmethod
    def get_job_result(self, job_id: str) -> Any:
        """Get the result of a completed job.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Result object containing measurement outcomes or expectation values

        Raises:
            ValueError: If job_id is invalid
            RuntimeError: If job is not completed or failed
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued job.

        Args:
            job_id: Unique identifier for the job

        Returns:
            True if cancellation was successful, False otherwise

        Raises:
            ValueError: If job_id is invalid
        """
        pass

    @abstractmethod
    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties of the current backend.

        Returns:
            Dictionary containing backend properties such as:
            - num_qubits: Number of qubits
            - basis_gates: List of supported gates
            - coupling_map: Qubit connectivity
            - gate_errors: Error rates for gates
            - readout_errors: Measurement error rates
        """
        pass

    @abstractmethod
    def get_backend_configuration(self) -> Dict[str, Any]:
        """Get configuration of the current backend.

        Returns:
            Dictionary containing backend configuration details
        """
        pass

    def validate_config(self) -> bool:
        """Validate the configuration dictionary.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration keys are missing
        """
        required_keys = ["provider", "backend_name"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        return True

    def __repr__(self) -> str:
        """String representation of the QPU backend."""
        backend_name = self.config.get("backend_name", "unknown")
        provider = self.config.get("provider", "unknown")
        return f"{self.__class__.__name__}(provider={provider}, backend={backend_name})"
