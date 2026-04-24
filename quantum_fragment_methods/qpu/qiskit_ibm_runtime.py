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

"""IBM Quantum backend implementation using Qiskit Runtime.

This module implements the QPUBackend interface
for IBM Quantum hardware using the Qiskit Runtime service.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from quantum_fragment_methods.qpu.base import QPUBackend

logger = logging.getLogger(__name__)


class IBMQuantumBackend(QPUBackend):
    """IBM Quantum backend implementation.

    This class wraps the Qiskit Runtime service to provide access to
    IBM Quantum hardware through a standardized interface.

    Configuration example:
        {
            'provider': 'ibm_quantum',
            'backend_name': 'ibm_fez',
            'credentials': {
                'channel': 'ibm_cloud',
                'instance': 'crn:v1:bluemix:...',
                'token': 'your_token_here'
            },
            'sampler_options': {
                'default_shots': 1000000,
                'dynamical_decoupling': {
                    'enable': True,
                    'sequence_type': 'XY4'
                },
                'twirling': {
                    'enable_gates': False,
                    'enable_measure': False
                }
            }
        }
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize IBM Quantum backend.

        Args:
            config: Configuration dictionary with IBM Quantum settings
        """
        super().__init__(config)
        self._validate_ibm_config()

    def _validate_ibm_config(self) -> None:
        """Validate IBM-specific configuration."""
        self.validate_config()

        if "credentials" not in self.config:
            raise ValueError("Missing 'credentials' in configuration")

        creds = self.config["credentials"]
        required_creds = ["channel", "token"]
        for key in required_creds:
            if key not in creds:
                raise ValueError(f"Missing required credential: {key}")

    def initialize(self) -> None:
        """Initialize connection to IBM Quantum service.

        Raises:
            ImportError: If qiskit_ibm_runtime is not installed
            ConnectionError: If connection to service fails
            AuthenticationError: If authentication fails
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as e:
            raise ImportError(
                "qiskit_ibm_runtime is required for IBM Quantum backend. "
                "Install it with: pip install qiskit-ibm-runtime"
            ) from e

        creds = self.config["credentials"]

        try:
            logger.info("Initializing IBM Quantum Runtime service...")
            self.service = QiskitRuntimeService(
                channel=creds["channel"], token=creds["token"], instance=creds.get("instance")
            )
            logger.info("Successfully connected to IBM Quantum service")
        except Exception as e:
            import pdb

            pdb.set_trace()
            raise ConnectionError(f"Failed to connect to IBM Quantum service: {e}") from e

    def get_backend(self, backend_name: Optional[str] = None) -> Any:
        """Get IBM Quantum backend instance.

        Args:
            backend_name: Name of the backend (e.g., 'ibm_fez', 'ibm_kyoto').
                         If None, uses backend_name from config.

        Returns:
            IBM Quantum backend instance

        Raises:
            ValueError: If backend_name is invalid or not available
            RuntimeError: If service is not initialized
        """
        if self.service is None:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        name = backend_name or self.config.get("backend_name")
        if not name:
            raise ValueError("No backend_name specified")

        try:
            logger.info(f"Retrieving backend: {name}")
            self.backend = self.service.backend(name)
            logger.info(f"Successfully retrieved backend: {name}")
            return self.backend
        except Exception as e:
            raise ValueError(f"Failed to get backend '{name}': {e}") from e

    def create_sampler(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create a SamplerV2 primitive for IBM Quantum.

        Args:
            backend: Backend instance. If None, uses self.backend
            options: Sampler options. If None, uses config['sampler_options']

        Returns:
            Configured SamplerV2 instance

        Raises:
            ImportError: If qiskit_ibm_runtime is not installed
            ValueError: If backend is not set
        """
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
        except ImportError as e:
            raise ImportError(
                "qiskit_ibm_runtime is required. " "Install it with: pip install qiskit-ibm-runtime"
            ) from e

        backend_to_use = backend or self.backend
        if backend_to_use is None:
            raise ValueError("No backend available. Call get_backend() first.")

        logger.info("Creating Sampler primitive...")
        sampler = Sampler(mode=backend_to_use)

        # Apply options
        opts = options or self.config.get("sampler_options", {})

        # Set default shots
        if "default_shots" in opts:
            sampler.options.default_shots = opts["default_shots"]
            logger.info(f"Set default_shots: {opts['default_shots']}")

        # Configure dynamical decoupling
        if "dynamical_decoupling" in opts:
            dd_opts = opts["dynamical_decoupling"]
            sampler.options.dynamical_decoupling.enable = dd_opts.get("enable", False)
            if dd_opts.get("enable"):
                seq_type = dd_opts.get("sequence_type", "XY4")
                sampler.options.dynamical_decoupling.sequence_type = seq_type
                logger.info(f"Enabled dynamical decoupling: {seq_type}")

        # Configure twirling
        if "twirling" in opts:
            tw_opts = opts["twirling"]
            sampler.options.twirling.enable_gates = tw_opts.get("enable_gates", False)
            sampler.options.twirling.enable_measure = tw_opts.get("enable_measure", False)
            logger.info(
                f"Twirling - gates: {tw_opts.get('enable_gates')}, "
                f"measure: {tw_opts.get('enable_measure')}"
            )

        logger.info("Sampler created successfully")
        return sampler

    def create_estimator(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create an EstimatorV2 primitive for IBM Quantum.

        Args:
            backend: Backend instance. If None, uses self.backend
            options: Estimator options

        Returns:
            Configured EstimatorV2 instance

        Raises:
            ImportError: If qiskit_ibm_runtime is not installed
            ValueError: If backend is not set
        """
        try:
            from qiskit_ibm_runtime import EstimatorV2 as Estimator
        except ImportError as e:
            raise ImportError(
                "qiskit_ibm_runtime is required. " "Install it with: pip install qiskit-ibm-runtime"
            ) from e

        backend_to_use = backend or self.backend
        if backend_to_use is None:
            raise ValueError("No backend available. Call get_backend() first.")

        logger.info("Creating Estimator primitive...")
        estimator = Estimator(mode=backend_to_use)

        # Apply options if provided
        if options:
            for key, value in options.items():
                if hasattr(estimator.options, key):
                    setattr(estimator.options, key, value)

        logger.info("Estimator created successfully")
        return estimator

    def submit_job(self, circuits: List[Any], primitive: Any, **kwargs) -> str:
        """Submit circuits to IBM Quantum hardware.

        Args:
            circuits: List of quantum circuits to execute
            primitive: Sampler or Estimator primitive
            **kwargs: Additional arguments passed to primitive.run()

        Returns:
            Job ID string

        Raises:
            RuntimeError: If job submission fails
        """
        try:
            logger.info(f"Submitting {len(circuits)} circuit(s) to IBM Quantum...")
            job = primitive.run(circuits, **kwargs)
            job_id = job.job_id()
            logger.info(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        except Exception as e:
            raise RuntimeError(f"Failed to submit job: {e}") from e

    def retrieve_job(self, job_id: str) -> Any:
        """Retrieve a job by ID from IBM Quantum.

        Args:
            job_id: Job identifier

        Returns:
            Job object

        Raises:
            ValueError: If job_id is invalid
            RuntimeError: If service is not initialized or job retrieval fails
        """
        if self.service is None:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            logger.info(f"Retrieving job: {job_id}")
            job = self.service.job(job_id)
            return job
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve job '{job_id}': {e}") from e

    def get_job_status(self, job_id: str) -> str:
        """Get status of an IBM Quantum job.

        Args:
            job_id: Job identifier

        Returns:
            Status string (e.g., 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')
        """
        job = self.retrieve_job(job_id)
        status = job.status()
        logger.info(f"Job {job_id} status: {status}")
        return str(status)

    def get_job_result(self, job_id: str) -> Any:
        """Get result of a completed IBM Quantum job.

        Args:
            job_id: Job identifier

        Returns:
            Result object with measurement data or expectation values

        Raises:
            RuntimeError: If job is not completed or failed
        """
        job = self.retrieve_job(job_id)

        try:
            logger.info(f"Retrieving result for job: {job_id}")
            result = job.result()
            logger.info(f"Successfully retrieved result for job: {job_id}")
            return result
        except Exception as e:
            status = job.status()
            raise RuntimeError(
                f"Failed to get result for job '{job_id}' (status: {status}): {e}"
            ) from e

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an IBM Quantum job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancellation was successful
        """
        job = self.retrieve_job(job_id)

        try:
            logger.info(f"Cancelling job: {job_id}")
            job.cancel()
            logger.info(f"Job {job_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job '{job_id}': {e}")
            return False

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties of the IBM Quantum backend.

        Returns:
            Dictionary with backend properties including:
            - num_qubits: Number of qubits
            - basis_gates: Supported gate set
            - coupling_map: Qubit connectivity
            - backend_name: Name of the backend
            - backend_version: Version string
        """
        if self.backend is None:
            raise RuntimeError("No backend set. Call get_backend() first.")

        config = self.backend.configuration()

        properties = {
            "num_qubits": config.n_qubits,
            "basis_gates": config.basis_gates,
            "coupling_map": config.coupling_map,
            "backend_name": config.backend_name,
            "backend_version": config.backend_version,
            "max_shots": getattr(config, "max_shots", None),
            "max_experiments": getattr(config, "max_experiments", None),
        }

        return properties

    def get_backend_configuration(self) -> Dict[str, Any]:
        """Get full configuration of the IBM Quantum backend.

        Returns:
            Dictionary with complete backend configuration
        """
        if self.backend is None:
            raise RuntimeError("No backend set. Call get_backend() first.")

        config = self.backend.configuration()
        return config.to_dict()

    def list_available_backends(self) -> List[str]:
        """List all available IBM Quantum backends.

        Returns:
            List of backend names

        Raises:
            RuntimeError: If service is not initialized
        """
        if self.service is None:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        backends = self.service.backends()
        return [backend.name for backend in backends]
