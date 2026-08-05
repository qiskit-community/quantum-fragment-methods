# HPC Slurm Workflow Example

This directory contains an example workflow for running quantum fragment methods on HPC systems using Slurm batch scheduling.

## Overview

This example demonstrates the complete quantum fragment workflow for an alanine molecule using:
- **System**: Alanine (13 atoms) from `system.xyz`
- **Method**: Embedded Wave Function (EWF) with SQD solver
- **Execution**: Slurm batch jobs with containerized environment
- **Backend**: IBM Quantum hardware for quantum sampling

## Workflow Steps

1. **Mean-Field Calculation** (`01_meanfield.slurm`) - Hartree-Fock calculation
2. **Fragment Construction** (`02_fragments.slurm`) - EWF embedding and fragmentation
3. **Fragment Solving** (`03_solve.slurm`) - SQD solver with quantum backend
4. **Energy Reconstruction** (`04_reconstruct.slurm`) - Total energy calculation

## Prerequisites

- Access to an HPC cluster with Slurm
- Container image built and transferred (see `docs/HPC-BUILD-GUIDE.md`)
- IBM Quantum account and credentials
- Configuration file with your credentials

## Quick Start

```bash
# TODO: Add quick start instructions
```

## Configuration

```bash
# TODO: Add configuration instructions
```

## Running the Workflow

```bash
# TODO: Add workflow execution instructions
```

## Expected Outputs

```bash
# TODO: Add expected outputs description
```

## Troubleshooting

```bash
# TODO: Add troubleshooting guide
```

## References

This example is based on the workflow demonstrated in `examples/local_ewf_sqd_demo/sqd_N2_sto-3g_demo.ipynb`.

For detailed HPC setup instructions, see `docs/IBM-SCC-BUILD-GUIDE.md`.