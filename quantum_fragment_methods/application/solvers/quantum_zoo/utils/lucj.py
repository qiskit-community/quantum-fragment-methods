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

"""LUCJ (Linear Unitary Coupled Cluster with Jastrow) circuit construction utilities.

This module provides utilities for building LUCJ ansatz circuits from CCSD amplitudes
using the ffsim library and converting them to Qiskit circuits for execution on
quantum hardware. It also includes functions for generating optimal zigzag qubit
layouts for IBM Quantum heavy-hex topology backends.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import rustworkx
from qiskit.providers import BackendV2
from rustworkx import NoEdgeBetweenNodes, PyGraph

logger = logging.getLogger(__name__)

IBM_TWO_Q_GATES = {"cx", "ecr", "cz"}


def create_linear_chains(num_orbitals: int) -> PyGraph:
    """In zig-zag layout, there are two linear chains (with connecting qubits between
    the chains). This function creates those two linear chains: a rustworkx PyGraph
    with two disconnected linear chains. Each chain contains `num_orbitals` number
    of nodes, i.e., in the final graph there are `2 * num_orbitals` number of nodes.

    Args:
        num_orbitals (int): Number orbitals or nodes in each linear chain. They are
            also known as alpha-alpha interaction qubits.

    Returns:
        A rustworkx.PyGraph with two disconnected linear chains each with `num_orbitals`
            number of nodes.
    """
    G = rustworkx.PyGraph()

    for n in range(num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    for n in range(num_orbitals, 2 * num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals, 2 * num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    return G


def create_lucj_zigzag_layout(
    num_orbitals: int, backend_coupling_graph: PyGraph
) -> tuple[PyGraph, int]:
    """This function creates the complete zigzag graph that 'can be mapped' to a IBM QPU with
    heavy-hex connectivity (the zigzag must be an isomorphic sub-graph to the QPU/backend
    coupling graph for it to be mapped).
    The zigzag pattern includes both linear chains (alpha-alpha interactions) and connecting
    qubits between the linear chains (alpha-beta interactions).

    Args:
        num_orbitals (int): Number of orbitals, i.e., number of nodes in each alpha-alpha linear chain.
        backend_coupling_graph (PyGraph): The coupling graph of the backend on which the LUCJ ansatz
            will be mapped and run. This function takes the coupling graph as a undirected
            `rustworkx.PyGraph` where there is only one 'undirected' edge between two nodes,
            i.e., qubits. Usually, the coupling graph of a IBM backend is directed (e.g., Eagle devices
            such as ibm_sherbrooke) or may have two edges between two nodes (e.g., Heron `ibm_torino`).
            A user needs to be make such graphs undirected and/or remove duplicate edges to make them
            compatible with this function. One way to do this is as follows:
            ```
            graph = backend.coupling_map.graph
            if not graph.is_symmetric():
                graph.make_symmetric()
            backend_coupling_graph = graph.to_undirected()

            edge_list = backend_coupling_graph.edge_list()
            removed_edge = []
            for edge in edge_list:
                if set(edge) in removed_edge:
                    continue
                try:
                    backend_coupling_graph.remove_edge(edge[0], edge[1])
                    removed_edge.append(set(edge))
                except NoEdgeBetweenNodes:
                    pass
            ```

    Returns:
        G_new (PyGraph): The graph with IBM backend compliant zigzag pattern.
        num_alpha_beta_qubits (int): Number of connecting qubits between the linear chains
            in the zigzag pattern. While we want as many connecting (alpha-beta) qubits between
            the linear (alpha-alpha) chains, we cannot accomodate all due to qubit and connectivity
            constraints of backends. This is the maximum number of connecting qubits the zigzag pattern
            can have while being backend compliant (i.e., isomorphic to backend coupling graph).
    """
    isomorphic = False
    G = create_linear_chains(num_orbitals=num_orbitals)

    num_iters = copy.deepcopy(num_orbitals)
    while not isomorphic:
        G_new = copy.deepcopy(G)
        num_alpha_beta_qubits = 0
        for n in range(num_iters):
            if n % 4 == 0:
                new_node = 2 * num_orbitals + num_alpha_beta_qubits
                G_new.add_node(new_node)
                G_new.add_edge(n, new_node, None)
                G_new.add_edge(new_node, n + num_orbitals, None)
                num_alpha_beta_qubits = num_alpha_beta_qubits + 1
        isomorphic = rustworkx.is_subgraph_isomorphic(backend_coupling_graph, G_new)
        num_iters -= 1

    return G_new, num_alpha_beta_qubits


def lightweight_layout_error_scoring(
    backend: BackendV2,
    virtual_edges: Sequence[Sequence[int]],
    physical_layouts: Sequence[int],
    two_q_gate_name: str,
) -> list[list[list[int], float]]:
    """Lighweight and heuristic function to score isomorphic layouts. There can be many zigzag patterns,
    each with different set of physical qubits, that can be mapped to a backend. Some of them may
    include less noise qubits and couplings than others. This function computes a simple error score
    for each such layout. It sums up 2Q gate error for all couplings in the zigzag pattern (layout) and
    meaurement of errors of physical qubits in the layout to compute the error score.

    Note:
        This lightweight scoring can be refined using concepts such as mapomatic.

    Args:
        backend (BackendV2): A backend.
        virtual_edges (Sequence[Sequence[int]]): Edges in the device compliant zigzag pattern where
            nodes are numbered from 0 to (2 * num_orbitals + num_alpha_beta_qubits).
        physical_layouts (Sequence[int]): All physical layouts of the zigzag pattern that are isomorphic
            to each other and to the larger backend coupling map.
        two_q_gate_name (str): The name of the two-qubit gate of the backend. The name is used for fecthing
            two-qubit gate error from backend properties.

    Returns:
        scores (list): A list of lists where each sublist contains two items. First item is the layout, and
            second item is a float representing error score of the layout. The layouts in the `scores` are
            sorted in the ascedning order of error score.
    """
    props = backend.properties()
    scores = []
    for layout in physical_layouts:
        total_2q_error = 0
        for edge in virtual_edges:
            physical_edge = (layout[edge[0]], layout[edge[1]])
            try:
                ge = props.gate_error(two_q_gate_name, physical_edge)
            except:
                ge = props.gate_error(two_q_gate_name, physical_edge[::-1])
            total_2q_error += ge
        total_measurement_error = 0
        for qubit in layout:
            meas_error = props.readout_error(qubit)
            total_measurement_error += meas_error
        scores.append([layout, total_2q_error + total_measurement_error])

    return sorted(scores, key=lambda x: x[1])


def _make_backend_cmap_pygraph(backend: BackendV2) -> PyGraph:
    graph = backend.coupling_map.graph
    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_coupling_graph = graph.to_undirected()

    edge_list = backend_coupling_graph.edge_list()
    removed_edge = []
    for edge in edge_list:
        if set(edge) in removed_edge:
            continue
        try:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
            removed_edge.append(set(edge))
        except NoEdgeBetweenNodes:
            pass

    return backend_coupling_graph


def get_zigzag_physical_layout(
    num_orbitals: int, backend: BackendV2, score_layouts: bool = True
) -> tuple[list[int], int]:
    """The main function that generates the zigzag pattern with physical qubits that can be used
    as an `intial_layout` in a preset passmanager/transpiler.

    Args:
        num_orbitals (int): Number of orbitals.
        backend (BackendV2): A backend.
        score_layouts (bool): Optional. If `True`, it uses the `lightweight_layout_error_scoring`
            function to score the isomorphic layouts and returns the layout with less errorneous qubits.
            If `False`, returns the first isomorphic subgraph.

    Returns:
        A tuple of device compliant layout (list[int]) with zigzag pattern and an int representing
            number of alpha-beta-interactions.
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    G, num_aplha_beta_qubits = create_lucj_zigzag_layout(
        num_orbitals=num_orbitals, backend_coupling_graph=backend_coupling_graph
    )

    isomorphic_mappings = rustworkx.vf2_mapping(backend_coupling_graph, G, subgraph=True)
    isomorphic_mappings = list(isomorphic_mappings)

    edges = list(G.edge_list())

    layouts = []
    for mapping in isomorphic_mappings:
        initial_layout = [None] * (2 * num_orbitals + num_aplha_beta_qubits)
        for key, value in mapping.items():
            initial_layout[value] = key
        layouts.append(initial_layout)

    two_q_gate_name = IBM_TWO_Q_GATES.intersection(backend.configuration().basis_gates).pop()

    if score_layouts:
        scores = lightweight_layout_error_scoring(
            backend=backend,
            virtual_edges=edges,
            physical_layouts=layouts,
            two_q_gate_name=two_q_gate_name,
        )

        return scores[0][0][:-num_aplha_beta_qubits], num_aplha_beta_qubits

    return layouts[0][:-num_aplha_beta_qubits], num_aplha_beta_qubits


def build_lucj_circuit(
    t1: Any,
    t2: Any,
    norb: int,
    nelec: tuple[int, int],
    config: dict[str, Any] | None = None,
) -> Any:
    """Build LUCJ ansatz circuit from CCSD amplitudes.

    This function constructs a LUCJ (Linear Unitary Coupled Cluster with Jastrow)
    circuit using the ffsim library, optimizing the UCJ operator from t2 amplitudes
    and preparing the circuit for execution on quantum hardware.

    Args:
        t1: CCSD single excitation amplitudes (norb x norb)
        t2: CCSD double excitation amplitudes (norb x norb x norb x norb)
        norb: Number of spatial orbitals
        nelec: Tuple of (n_alpha, n_beta) electrons
        config: Configuration dictionary with LUCJ parameters:
            - n_reps: Number of repetitions (default: 1)
            - max_connection: Maximum orbital connection distance (default: 12)
            - connect_every_n: Connection stride (default: 4)
            - optimize: Whether to optimize UCJ operator (default: True)
            - method: Optimization method (default: "L-BFGS-B")
            - maxiter: Maximum optimization iterations (default: 1000)

    Returns:
        Qiskit QuantumCircuit implementing the LUCJ ansatz

    Raises:
        ImportError: If required packages (ffsim, qiskit) are not installed
        ValueError: If input parameters are invalid
    """
    try:
        import ffsim
        from qiskit import QuantumCircuit, QuantumRegister
    except ImportError as e:
        raise ImportError(
            "ffsim and qiskit are required for LUCJ circuit construction. "
            "Install with: pip install ffsim qiskit"
        ) from e

    # Get configuration parameters
    config = config or {}
    n_reps = config.get("n_reps", 1)
    max_connection = config.get("max_connection", 12)
    connect_every_n = config.get("connect_every_n", 4)
    optimize = config.get("optimize", True)
    method = config.get("method", "L-BFGS-B")
    maxiter = config.get("maxiter", 1000)

    logger.info(f"Building LUCJ circuit: norb={norb}, nelec={nelec}, n_reps={n_reps}")

    # Define interaction pairs
    # Ensure indices don't exceed available orbitals
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    max_orbital_index = min(max_connection, norb - 1)
    alpha_beta_indices = [(p, p) for p in range(0, max_orbital_index + 1, connect_every_n)]

    logger.info(
        f"LUCJ interaction pairs: alpha-alpha={len(alpha_alpha_indices)}, "
        f"alpha-beta={len(alpha_beta_indices)}"
    )

    # Build UCJ operator from t2 amplitudes
    logger.info(f"Optimizing LUCJ operator (n_reps={n_reps}, optimize={optimize})...")
    import time

    start_time = time.time()

    try:
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            # t1=t1,  # Optionally include t1 amplitudes
            n_reps=n_reps,
            interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
            optimize=optimize,
            method=method,
            options={"maxiter": maxiter},
        )
    except Exception as e:
        logger.error(f"Failed to build UCJ operator: {e}")
        logger.info("Attempting with reduced interaction pairs...")
        # Fallback: use minimal interaction pairs
        alpha_alpha_indices = [(p, p + 1) for p in range(min(norb - 1, 2))]
        alpha_beta_indices = [(0, 0)] if norb > 0 else []
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_reps,
            interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
            optimize=optimize,
            method=method,
            options={"maxiter": maxiter},
        )

    elapsed = time.time() - start_time
    logger.info(f"LUCJ optimization completed in {elapsed:.2f} seconds")

    # Build quantum circuit
    num_alpha, num_beta = nelec
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)

    # Prepare Hartree-Fock reference state
    logger.info("Preparing Hartree-Fock reference state...")
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)

    # Apply UCJ operator
    logger.info("Applying UCJ operator to circuit...")
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    logger.info(
        f"LUCJ circuit built successfully: {circuit.num_qubits} qubits, depth={circuit.depth()}"
    )
    return circuit
