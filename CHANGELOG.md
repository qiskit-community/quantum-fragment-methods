# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Embedding Methods**: EWF
- **Classical Solvers**: FCI, CCSD
- **Quantum Solvers**: SQD with LUCJ ansatz and SBD post-processing
- **QPU Integration**: IBM Quantum backend with checkpoint-based workflows
- **Workflow System**: Adaptive solver selection and fragment management
- **Examples**: Demonstration notebooks for standalone and workflow usage
- **HPC Support**: Containerfile 

### Technical Details
- SQD solver with custom `diagonalize_fermionic_hamiltonian` for HPC
- SBDInterface for Selected Basis Diagonalization
- LUCJ circuit building with zigzag layout for IBM heavy-hex topology
- Fragment-level CCSD amplitude computation
- Vayesta DUMP solver compatibility

[Unreleased]: https://ibm.github.com/thaddeus-pellegrini/quantum-fragment-methods/compare/v0.1.0...HEAD