"""
JEFF format export for stimdx circuits.

This module provides functionality to export stimdx Circuit AST to the JEFF
(JSON Exchange Format For circuits) format for interoperability with HUGR and Guppy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import stim

from jeff import (
    FunctionDef,
    IntType,
    JeffModule,
    JeffOp,
    JeffRegion,
    JeffValue,
    QubitType,
    SwitchSCF,
    WhileSCF,
    DoWhileSCF,
    quantum_gate,
    qubit_alloc,
    qubit_free,
)

from ._cond import Cond, LastMeas, MeasParity
from ._core import Circuit, Node, StimBlock, IfNode, WhileNode, DoWhileNode


# Gate mapping from Stim to JEFF WellKnownGate names
# Format: stim_name -> (jeff_name, num_qubits, num_params, num_controls, adjoint)
STIM_TO_JEFF_GATE: dict[str, tuple[str, int, int, int, bool]] = {
    # Single-qubit Pauli gates
    "I": ("i", 1, 0, 0, False),
    "X": ("x", 1, 0, 0, False),
    "Y": ("y", 1, 0, 0, False),
    "Z": ("z", 1, 0, 0, False),
    # Clifford gates
    "H": ("h", 1, 0, 0, False),
    "S": ("s", 1, 0, 0, False),
    "S_DAG": ("s", 1, 0, 0, True),  # S† = adjoint of S
    "SQRT_Z": ("s", 1, 0, 0, False),  # Same as S
    "SQRT_Z_DAG": ("s", 1, 0, 0, True),
    # T gates
    "T": ("t", 1, 0, 0, False),
    "T_DAG": ("t", 1, 0, 0, True),
    # Two-qubit gates
    "CNOT": ("x", 1, 0, 1, False),  # Controlled-X
    "CX": ("x", 1, 0, 1, False),
    "CY": ("y", 1, 0, 1, False),
    "CZ": ("z", 1, 0, 1, False),
    "SWAP": ("swap", 2, 0, 0, False),
    "ISWAP": ("swap", 2, 0, 0, False),  # Note: ISWAP != SWAP, but closest approx
    # Controlled gates
    "XCX": ("x", 1, 0, 1, False),
    "XCY": ("y", 1, 0, 1, False),
    "XCZ": ("z", 1, 0, 1, False),
    "YCX": ("x", 1, 0, 1, False),
    "YCY": ("y", 1, 0, 1, False),
    "YCZ": ("z", 1, 0, 1, False),
    "ZCX": ("x", 1, 0, 1, False),
    "ZCY": ("y", 1, 0, 1, False),
    "ZCZ": ("z", 1, 0, 1, False),
}

# Parametric gates (with rotation angles)
STIM_PARAMETRIC_GATES: dict[str, tuple[str, int]] = {
    # Note: Stim doesn't have RX/RY/RZ directly, but we support them if present
}


@dataclass
class ExportState:
    """Tracks state during JEFF export."""

    # Map from stim qubit index to current JeffValue representing that qubit
    qubit_values: dict[int, JeffValue] = field(default_factory=dict)
    # List of measurement result values in order
    measurement_results: list[JeffValue] = field(default_factory=list)
    # Operations accumulated in current region
    operations: list[JeffOp] = field(default_factory=list)
    # Track which measurement indices belong to the "last block"
    last_block_meas_start: int = 0


class JeffExporter:
    """Exports a stimdx Circuit to JEFF format."""

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self._validate_circuit()

    def _validate_circuit(self) -> None:
        """Validate that the circuit can be exported to JEFF."""

        def check_node(node: Node) -> None:
            if isinstance(node, (IfNode, WhileNode, DoWhileNode)):
                if not isinstance(node.cond, Cond):
                    raise ValueError(
                        f"Cannot export circuit with lambda conditionals to JEFF. "
                        f"Use Cond subclasses (LastMeas, MeasParity) instead. "
                        f"Found: {type(node.cond)}"
                    )
                # Recursively check body
                for child in node.body.nodes:
                    check_node(child)

        for node in self.circuit.nodes:
            check_node(node)

    def export(self, name: str = "main") -> JeffModule:
        """Export the circuit to a JeffModule."""
        # First pass: determine all qubits used
        all_qubits = self._collect_qubits()

        # Create initial state
        state = ExportState()

        # Allocate qubits
        for q in sorted(all_qubits):
            alloc_op = qubit_alloc()
            state.operations.append(alloc_op)
            state.qubit_values[q] = alloc_op.outputs[0]

        # Process all nodes
        self._process_nodes(self.circuit.nodes, state)

        # Free qubits at the end
        for q in sorted(all_qubits):
            if q in state.qubit_values:
                free_op = qubit_free(state.qubit_values[q])
                state.operations.append(free_op)

        # Build region from operations
        region = self._build_region(state.operations, [], [])

        # Create function and module
        func = FunctionDef(name, region)
        module = JeffModule([func], entrypoint=0)

        return module

    def _collect_qubits(self) -> set[int]:
        """Collect all qubit indices used in the circuit."""
        qubits: set[int] = set()

        def collect_from_stim(circuit: stim.Circuit) -> None:
            for inst in circuit:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    collect_from_stim(inst.body_copy())
                else:
                    for target in inst.targets_copy():
                        if target.is_qubit_target:
                            qubits.add(target.value)

        def collect_from_node(node: Node) -> None:
            if isinstance(node, StimBlock):
                collect_from_stim(node.circuit)
            elif isinstance(node, (IfNode, WhileNode, DoWhileNode)):
                for child in node.body.nodes:
                    collect_from_node(child)

        for node in self.circuit.nodes:
            collect_from_node(node)

        return qubits

    def _process_nodes(self, nodes: list[Node], state: ExportState) -> None:
        """Process a list of AST nodes."""

        for node in nodes:
            if isinstance(node, StimBlock):
                self._process_stim_block(node, state)
            elif isinstance(node, IfNode):
                self._process_if_node(node, state)
            elif isinstance(node, WhileNode):
                self._process_while_node(node, state)
            elif isinstance(node, DoWhileNode):
                self._process_do_while_node(node, state)

    def _process_stim_block(self, block: StimBlock, state: ExportState) -> None:
        """Process a StimBlock, converting instructions to JEFF operations."""

        if block.capture_as_last:
            state.last_block_meas_start = len(state.measurement_results)

        for inst in block.circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                # Handle repeat blocks by unrolling (simple approach)
                for _ in range(inst.repeat_count):
                    inner_block = StimBlock(inst.body_copy(), capture_as_last=False)
                    self._process_stim_block(inner_block, state)
            else:
                self._process_instruction(inst, state)

    def _process_instruction(
        self, inst: stim.CircuitInstruction, state: ExportState
    ) -> None:
        """Process a single Stim instruction."""
        name = inst.name
        targets = inst.targets_copy()

        if name in ("M", "MZ"):
            # Measurement in Z basis
            for target in targets:
                if target.is_qubit_target:
                    q = target.value
                    qubit_val = state.qubit_values[q]
                    # Non-destructive measurement: qubit -> (qubit, int(1))
                    meas_result = JeffValue(IntType(1))
                    new_qubit = JeffValue(QubitType())
                    meas_op = JeffOp(
                        "qubit", "measureNd", [qubit_val], [new_qubit, meas_result]
                    )
                    state.operations.append(meas_op)
                    state.qubit_values[q] = new_qubit
                    state.measurement_results.append(meas_result)

        elif name in ("MX", "MY"):
            # Measurement in X/Y basis - apply basis change, measure, undo
            for target in targets:
                if target.is_qubit_target:
                    q = target.value
                    qubit_val = state.qubit_values[q]

                    # Basis change
                    if name == "MX":
                        gate_op = quantum_gate("h", qubit_val)
                        state.operations.append(gate_op)
                        qubit_val = gate_op.outputs[0]
                        state.qubit_values[q] = qubit_val

                    # Measure
                    meas_result = JeffValue(IntType(1))
                    new_qubit = JeffValue(QubitType())
                    meas_op = JeffOp(
                        "qubit", "measureNd", [qubit_val], [new_qubit, meas_result]
                    )
                    state.operations.append(meas_op)
                    state.qubit_values[q] = new_qubit
                    state.measurement_results.append(meas_result)

                    # Undo basis change
                    if name == "MX":
                        gate_op = quantum_gate("h", state.qubit_values[q])
                        state.operations.append(gate_op)
                        state.qubit_values[q] = gate_op.outputs[0]

        elif name in ("R", "RZ"):
            # Reset to |0>
            for target in targets:
                if target.is_qubit_target:
                    q = target.value
                    qubit_val = state.qubit_values[q]
                    new_qubit = JeffValue(QubitType())
                    reset_op = JeffOp("qubit", "reset", [qubit_val], [new_qubit])
                    state.operations.append(reset_op)
                    state.qubit_values[q] = new_qubit

        elif name in ("RX", "RY"):
            # Reset in X/Y basis
            for target in targets:
                if target.is_qubit_target:
                    q = target.value
                    qubit_val = state.qubit_values[q]

                    # Reset to |0>
                    new_qubit = JeffValue(QubitType())
                    reset_op = JeffOp("qubit", "reset", [qubit_val], [new_qubit])
                    state.operations.append(reset_op)
                    state.qubit_values[q] = new_qubit

                    # Apply basis change
                    if name == "RX":
                        gate_op = quantum_gate("h", state.qubit_values[q])
                        state.operations.append(gate_op)
                        state.qubit_values[q] = gate_op.outputs[0]

        elif name == "MR":
            # Measure and reset
            for target in targets:
                if target.is_qubit_target:
                    q = target.value
                    qubit_val = state.qubit_values[q]

                    # Non-destructive measurement
                    meas_result = JeffValue(IntType(1))
                    new_qubit = JeffValue(QubitType())
                    meas_op = JeffOp(
                        "qubit", "measureNd", [qubit_val], [new_qubit, meas_result]
                    )
                    state.operations.append(meas_op)
                    state.measurement_results.append(meas_result)

                    # Reset
                    reset_qubit = JeffValue(QubitType())
                    reset_op = JeffOp("qubit", "reset", [new_qubit], [reset_qubit])
                    state.operations.append(reset_op)
                    state.qubit_values[q] = reset_qubit

        elif name in STIM_TO_JEFF_GATE:
            self._process_gate(name, targets, state)

        elif name in (
            "TICK",
            "QUBIT_COORDS",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS",
        ):
            # Annotations - skip in JEFF export
            pass

        else:
            raise ValueError(f"Unsupported Stim instruction: {name}")

    def _process_gate(
        self,
        name: str,
        targets: list[stim.GateTarget],
        state: ExportState,
    ) -> None:
        """Process a gate instruction."""
        jeff_name, num_qubits, num_params, num_controls, adjoint = STIM_TO_JEFF_GATE[
            name
        ]

        # Collect qubit targets
        qubit_targets = [t for t in targets if t.is_qubit_target]

        if num_controls > 0:
            # Controlled gate: first qubit is control, rest are targets
            # For two-qubit controlled gates like CNOT
            for i in range(0, len(qubit_targets), 2):
                if i + 1 >= len(qubit_targets):
                    break
                control_q = qubit_targets[i].value
                target_q = qubit_targets[i + 1].value

                control_val = state.qubit_values[control_q]
                target_val = state.qubit_values[target_q]

                gate_op = quantum_gate(
                    jeff_name,
                    target_val,
                    control_qubits=[control_val],
                    adjoint=adjoint,
                )
                state.operations.append(gate_op)

                # Update qubit values (gate outputs: target, control)
                state.qubit_values[target_q] = gate_op.outputs[0]
                state.qubit_values[control_q] = gate_op.outputs[1]

        elif num_qubits == 2:
            # Two-qubit gate like SWAP
            for i in range(0, len(qubit_targets), 2):
                if i + 1 >= len(qubit_targets):
                    break
                q1 = qubit_targets[i].value
                q2 = qubit_targets[i + 1].value

                val1 = state.qubit_values[q1]
                val2 = state.qubit_values[q2]

                gate_op = quantum_gate(jeff_name, [val1, val2], adjoint=adjoint)
                state.operations.append(gate_op)

                state.qubit_values[q1] = gate_op.outputs[0]
                state.qubit_values[q2] = gate_op.outputs[1]

        else:
            # Single-qubit gate
            for target in qubit_targets:
                q = target.value
                qubit_val = state.qubit_values[q]

                gate_op = quantum_gate(jeff_name, qubit_val, adjoint=adjoint)
                state.operations.append(gate_op)

                state.qubit_values[q] = gate_op.outputs[0]

    def _process_if_node(self, node: IfNode, state: ExportState) -> None:
        """Process an IfNode using SwitchSCF."""

        # Build condition value
        # After validation, we know cond is a Cond instance, not a lambda
        assert isinstance(node.cond, Cond), "Condition should be validated as Cond"
        cond_val = self._build_condition_value(node.cond, state)

        # Build the two branches: false (index 0) and true (index 1)
        # False branch: do nothing, just pass through state
        false_state = self._clone_state_for_branch(state)
        false_region = self._build_passthrough_region(false_state)

        # True branch: execute body
        true_state = self._clone_state_for_branch(state)
        self._process_nodes(node.body.nodes, true_state)
        true_region = self._build_region_from_state(true_state, state)

        # Create switch operation
        # Inputs: condition, qubit values, measurement values
        region_inputs = self._get_state_values(state)
        switch_scf = SwitchSCF([false_region, true_region])
        switch_op = JeffOp(
            "scf",
            "switch",
            [cond_val] + region_inputs,
            self._create_output_values(region_inputs),
            instruction_data=switch_scf,
        )
        state.operations.append(switch_op)

        # Update state with switch outputs
        self._update_state_from_outputs(state, switch_op.outputs)

    def _process_while_node(self, node: WhileNode, state: ExportState) -> None:
        """Process a WhileNode using WhileSCF."""

        # Build condition region
        # After validation, we know cond is a Cond instance, not a lambda
        assert isinstance(node.cond, Cond), "Condition should be validated as Cond"
        cond_state = self._clone_state_for_branch(state)
        cond_val = self._build_condition_value(node.cond, cond_state)
        cond_region = self._build_condition_region(cond_state, cond_val, state)

        # Build body region
        body_state = self._clone_state_for_branch(state)
        self._process_nodes(node.body.nodes, body_state)
        body_region = self._build_region_from_state(body_state, state)

        # Create while operation
        region_inputs = self._get_state_values(state)
        while_scf = WhileSCF(cond_region, body_region)
        while_op = JeffOp(
            "scf",
            "while",
            region_inputs,
            self._create_output_values(region_inputs),
            instruction_data=while_scf,
        )
        state.operations.append(while_op)

        # Update state
        self._update_state_from_outputs(state, while_op.outputs)

    def _process_do_while_node(self, node: DoWhileNode, state: ExportState) -> None:
        """Process a DoWhileNode using DoWhileSCF."""

        # Build body region first
        body_state = self._clone_state_for_branch(state)
        self._process_nodes(node.body.nodes, body_state)
        body_region = self._build_region_from_state(body_state, state)

        # Build condition region - uses body_state since condition references
        # measurements from the body (do-while executes body first, then checks)
        # After validation, we know cond is a Cond instance, not a lambda
        assert isinstance(node.cond, Cond), "Condition should be validated as Cond"
        cond_state = self._clone_state_for_branch(body_state)
        cond_val = self._build_condition_value(node.cond, cond_state)
        # Condition region takes body's output state as input
        cond_region = self._build_condition_region(cond_state, cond_val, body_state)

        # Create do-while operation
        region_inputs = self._get_state_values(state)
        do_while_scf = DoWhileSCF(body_region, cond_region)
        do_while_op = JeffOp(
            "scf",
            "doWhile",
            region_inputs,
            self._create_output_values(region_inputs),
            instruction_data=do_while_scf,
        )
        state.operations.append(do_while_op)

        # Update state with body's final state structure
        self._update_state_from_outputs(state, do_while_op.outputs)

    def _build_condition_value(self, cond: Cond, state: ExportState) -> JeffValue:
        """Build a JeffValue representing the condition result."""
        if isinstance(cond, LastMeas):
            # Reference the measurement from the last block
            meas_idx = state.last_block_meas_start + cond.index
            if meas_idx >= len(state.measurement_results):
                raise IndexError(
                    f"LastMeas index {cond.index} out of range for last block"
                )
            return state.measurement_results[meas_idx]

        elif isinstance(cond, MeasParity):
            # XOR of multiple measurement results
            if not cond.indices:
                raise ValueError("MeasParity requires at least one index")

            # Get first measurement value
            first_idx = cond.indices[0]
            if first_idx < 0:
                first_idx = len(state.measurement_results) + first_idx
            result = state.measurement_results[first_idx]

            # XOR with remaining measurements
            for idx in cond.indices[1:]:
                if idx < 0:
                    idx = len(state.measurement_results) + idx
                other = state.measurement_results[idx]

                # Create XOR operation
                xor_result = JeffValue(IntType(1))
                xor_op = JeffOp("int", "xor", [result, other], [xor_result])
                state.operations.append(xor_op)
                result = xor_result

            return result

        else:
            raise ValueError(f"Unsupported condition type: {type(cond)}")

    def _clone_state_for_branch(self, state: ExportState) -> ExportState:
        """Create a copy of state for processing a branch."""
        return ExportState(
            qubit_values=dict(state.qubit_values),
            measurement_results=list(state.measurement_results),
            operations=[],
            last_block_meas_start=state.last_block_meas_start,
        )

    def _get_state_values(self, state: ExportState) -> list[JeffValue]:
        """Get all values that represent the current state."""
        values = []
        for q in sorted(state.qubit_values.keys()):
            values.append(state.qubit_values[q])
        values.extend(state.measurement_results)
        return values

    def _create_output_values(self, inputs: list[JeffValue]) -> list[JeffValue]:
        """Create output values matching the input types."""
        return [JeffValue(v.type) for v in inputs]

    def _update_state_from_outputs(
        self, state: ExportState, outputs: list[JeffValue]
    ) -> None:
        """Update state with new values from operation outputs."""
        sorted_qubits = sorted(state.qubit_values.keys())
        idx = 0
        for q in sorted_qubits:
            state.qubit_values[q] = outputs[idx]
            idx += 1
        for i in range(len(state.measurement_results)):
            state.measurement_results[i] = outputs[idx]
            idx += 1

    def _build_region(
        self,
        operations: list[JeffOp],
        sources: list[JeffValue],
        targets: list[JeffValue],
    ) -> JeffRegion:
        """Build a JeffRegion from operations."""
        region = JeffRegion.__new__(JeffRegion)
        region._sources = sources
        region._targets = targets
        region._operations = operations
        region._is_dirty = True
        return region

    def _build_region_from_state(
        self, branch_state: ExportState, parent_state: ExportState
    ) -> JeffRegion:
        """Build a region from a branch state."""
        sources = self._get_state_values(parent_state)
        targets = self._get_state_values(branch_state)
        return self._build_region(branch_state.operations, sources, targets)

    def _build_passthrough_region(self, state: ExportState) -> JeffRegion:
        """Build a region that passes values through unchanged."""
        values = self._get_state_values(state)
        return self._build_region([], values, values)

    def _build_condition_region(
        self, cond_state: ExportState, cond_val: JeffValue, parent_state: ExportState
    ) -> JeffRegion:
        """Build a condition region that outputs a boolean."""
        sources = self._get_state_values(parent_state)
        return self._build_region(cond_state.operations, sources, [cond_val])


def to_jeff(circuit: Circuit, name: str = "main") -> JeffModule:
    """
    Export a Circuit to JEFF format.

    Args:
        circuit: The Circuit to export.
        name: Name for the main function in the module.

    Returns:
        A JeffModule representing the circuit.

    Raises:
        ValueError: If the circuit contains lambda conditionals.
    """
    exporter = JeffExporter(circuit)
    return exporter.export(name)

