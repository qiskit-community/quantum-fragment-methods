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

"""IBM Quantum backend implementation using QRMI.

This module implements the QPUBackend interface using QRMI — the vendor-agnostic
HPC middleware for quantum resources. Unlike IBMQuantumBackend, credentials are
NOT passed in config. They are read from environment variables, which on SCC are
set automatically by the Slurm SPANK plugin when a QPU resource is allocated.

Environment variables required (set by Slurm SPANK plugin on SCC, or manually):
    QRMI_JOB_QPU_RESOURCES   Comma-separated resource names, e.g. ibm_torino
    QRMI_JOB_QPU_TYPES        Comma-separated types, e.g. qiskit-runtime-service
    {resource}_QRMI_IBM_QRS_ENDPOINT      Qiskit Runtime endpoint URL
    {resource}_QRMI_IBM_QRS_IAM_ENDPOINT  IBM Cloud IAM endpoint URL
    {resource}_QRMI_IBM_QRS_IAM_APIKEY    IBM Cloud IAM API key
    {resource}_QRMI_IBM_QRS_SERVICE_CRN   Cloud Resource Name (CRN)

See: https://github.com/qiskit-community/qrmi
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from quantum_fragment_methods.qpu.base import QPUBackend

logger = logging.getLogger(__name__)


class QRMIBackend(QPUBackend):
    """IBM Quantum backend via QRMI.

    Credentials are read from environment variables — no secrets in config files.
    On SCC, env vars are injected automatically by the Slurm SPANK plugin when
    a QPU resource is allocated via --gres=qpu:ibm_torino or similar.

    Configuration example:
        {
            'provider': 'qrmi',
            'backend_name': 'ibm_torino',   # must match QRMI_JOB_QPU_RESOURCES
        }

    Usage:
        config = {'provider': 'qrmi', 'backend_name': 'ibm_torino'}
        backend = QRMIBackend(config)
        backend.initialize()   # selects resource matching backend_name
        backend.get_backend()  # builds transpiler target from QRMI resource
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._qrmi = None   # raw QRMI resource object
        self._target = None  # Qiskit transpiler Target built from QRMI resource

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Select the QRMI resource matching config['backend_name'].

        Reads environment variables set by the Slurm SPANK plugin (or manually).

        Raises:
            ImportError: If qrmi[ibm] is not installed
            RuntimeError: If no matching resource is found
        """
        try:
            from qrmi.primitives import QRMIService
        except ImportError as e:
            raise ImportError(
                "qrmi[ibm] is required for QRMIBackend. "
                "Install with: pip install 'qrmi[ibm]'"
            ) from e

        service = QRMIService()
        resources = service.resources()

        if not resources:
            raise RuntimeError(
                "No QRMI quantum resources available. "
                "Ensure QRMI_JOB_QPU_RESOURCES and credentials env vars are set."
            )

        target_name = self.config.get("backend_name")
        if target_name:
            for res in resources:
                if res.resource_id() == target_name:
                    self._qrmi = res
                    logger.info(f"Selected QRMI resource: {target_name}")
                    break
            if self._qrmi is None:
                available = [r.resource_id() for r in resources]
                raise RuntimeError(
                    f"QRMI resource '{target_name}' not found. "
                    f"Available: {available}"
                )
        else:
            self._qrmi = resources[0]
            logger.info(f"Selected first available QRMI resource: {self._qrmi.resource_id()}")

        logger.info(f"QRMI initialized: {self._qrmi.metadata()}")

    def get_backend(self, backend_name: Optional[str] = None) -> Any:
        """Build and return a Qiskit transpiler Target from the QRMI resource.

        The returned Target is set as self.backend and used by SQDSolver's
        _transpile_circuit() to generate a pass manager.

        Returns:
            Qiskit transpiler Target built from QRMI backend configuration
        """
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")

        try:
            from qrmi.primitives.ibm import get_target
        except ImportError as e:
            raise ImportError(
                "qrmi[ibm] is required. Install: pip install 'qrmi[ibm]'"
            ) from e

        self._target = get_target(self._qrmi)
        self.backend = self._target
        logger.info(f"Built transpiler target from QRMI resource: {self._qrmi.resource_id()}")
        return self.backend

    def create_sampler(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create a QRMI SamplerV2 primitive.

        Args:
            backend: Unused — QRMI sampler is bound to the resource, not a backend object.
            options: Sampler options dict. If None, uses config['sampler_options'].
                     Supported keys: default_shots

        Returns:
            Configured QRMI SamplerV2 instance
        """
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")

        try:
            from qrmi.primitives.ibm import SamplerV2
        except ImportError as e:
            raise ImportError(
                "qrmi[ibm] is required. Install: pip install 'qrmi[ibm]'"
            ) from e

        opts = options or self.config.get("sampler_options", {})
        sampler_options = {}
        if "default_shots" in opts:
            sampler_options["default_shots"] = opts["default_shots"]

        sampler = SamplerV2(self._qrmi, options=sampler_options if sampler_options else None)
        logger.info(f"Created QRMI SamplerV2 with options: {sampler_options}")
        return sampler

    def create_estimator(
        self, backend: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create a QRMI EstimatorV2 primitive.

        Returns:
            Configured QRMI EstimatorV2 instance
        """
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")

        try:
            from qrmi.primitives.ibm import EstimatorV2
        except ImportError as e:
            raise ImportError(
                "qrmi[ibm] is required. Install: pip install 'qrmi[ibm]'"
            ) from e

        opts = options or {}
        estimator = EstimatorV2(self._qrmi, options=opts if opts else None)
        logger.info("Created QRMI EstimatorV2")
        return estimator

    def submit_job(self, circuits: List[Any], primitive: Any, **kwargs) -> str:
        """Submit circuits via QRMI primitive.

        Args:
            circuits: List of ISA circuits (wrapped as PUB tuples for sampler.run)
            primitive: QRMI SamplerV2 instance
            **kwargs: Additional arguments passed to primitive.run()

        Returns:
            Job ID string
        """
        try:
            job = primitive.run([(c,) for c in circuits], **kwargs)
            job_id = job.job_id()
            logger.info(f"QRMI job submitted: {job_id}")
            return job_id
        except Exception as e:
            raise RuntimeError(f"Failed to submit QRMI job: {e}") from e

    def retrieve_job(self, job_id: str) -> Any:
        """Retrieve a QRMI job handle for status polling.

        Returns:
            QRMI job object
        """
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")

        try:
            from qrmi.primitives.ibm import SamplerV2
        except ImportError as e:
            raise ImportError("qrmi[ibm] is required.") from e

        sampler = SamplerV2(self._qrmi)
        return sampler.job(job_id)

    def get_job_status(self, job_id: str) -> str:
        """Get the status of a QRMI job.

        Returns:
            Uppercase status string (QUEUED, RUNNING, COMPLETED, FAILED, etc.)
        """
        job = self.retrieve_job(job_id)
        status = str(job.status()).upper()
        logger.info(f"QRMI job {job_id} status: {status}")
        return status

    def get_job_result(self, job_id: str) -> Any:
        """Get the result of a completed QRMI job.

        Returns:
            PrimitiveResult compatible with result[0].data.meas.get_counts()

        Raises:
            RuntimeError: If job errored (includes task logs in message)
        """
        job = self.retrieve_job(job_id)
        try:
            result = job.result()
            logger.info(f"Retrieved result for QRMI job: {job_id}")
            return result
        except RuntimeError as e:
            if job.errored():
                logs = self._qrmi.task_logs(job_id)
                logger.error(f"QRMI job logs:\n{logs}")
            raise RuntimeError(f"Failed to get result for job '{job_id}': {e}") from e

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a QRMI job."""
        try:
            job = self.retrieve_job(job_id)
            job.cancel()
            logger.info(f"Cancelled QRMI job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel QRMI job '{job_id}': {e}")
            return False

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties from the QRMI resource metadata."""
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")

        metadata = self._qrmi.metadata()
        return {
            "backend_name": self._qrmi.resource_id(),
            "resource_type": str(self._qrmi.resource_type()),
            "metadata": metadata,
        }

    def get_backend_configuration(self) -> Dict[str, Any]:
        """Get configuration from the QRMI resource."""
        if self._qrmi is None:
            raise RuntimeError("QRMI not initialized. Call initialize() first.")
        return {"resource_id": self._qrmi.resource_id()}
