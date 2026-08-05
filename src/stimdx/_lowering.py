from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import stim

from ._cond import LastMeas, MeasParity
from ._core import (
    Circuit,
    DetectorNode,
    DoWhileNode,
    EmitNode,
    IfNode,
    LetNode,
    ObservableIncludeNode,
    RepeatNode,
    StimBlock,
    WhileNode,
)
from ._expr import (
    AddExpr,
    Expr,
    InvertExpr,
    LiteralExpr,
    ModExpr,
    RecExpr,
    VarExpr,
    XorExpr,
)


class LoweringError(NotImplementedError):
    """Raised when a stimdx circuit cannot be exactly lowered to static Stim."""


@dataclass
class _LoweringState:
    meas_count: int = 0
    last_block_meas_start: int | None = None
    last_block_meas_len: int = 0
    vars: dict[str, int | bool] | None = None


@dataclass(frozen=True)
class _AffineCondition:
    constant: bool
    rec_controls: frozenset[int]


def lower_to_stim(program: Circuit) -> stim.Circuit:
    out = stim.Circuit()
    state = _LoweringState()
    _lower_program(program, out, state)
    return out


def _lower_program(program: Circuit, out: stim.Circuit, state: _LoweringState) -> None:
    for node in program.nodes:
        if isinstance(node, StimBlock):
            before = state.meas_count
            out += node.circuit
            added = _num_measurements(node.circuit)
            state.meas_count += added
            if node.capture_as_last:
                state.last_block_meas_start = before
                state.last_block_meas_len = added
            continue

        if isinstance(node, DetectorNode):
            line = _detector_line(node.indices, state.meas_count, node.coords)
            out += stim.Circuit(line + "\n")
            continue

        if isinstance(node, ObservableIncludeNode):
            line = _observable_line(
                node.observable_index, node.indices, state.meas_count
            )
            out += stim.Circuit(line + "\n")
            continue

        if isinstance(node, IfNode):
            _lower_if(node, out, state)
            continue

        if isinstance(node, RepeatNode):
            _lower_repeat(node, out, state)
            continue

        if isinstance(node, (WhileNode, DoWhileNode)):
            _lower_dynamic_loop_if_compile_time_deterministic(node, out, state)
            continue

        if isinstance(node, LetNode):
            _lower_let(node, state)
            continue

        if isinstance(node, EmitNode):
            raise LoweringError(
                "EmitNode is not supported by exact Stim lowering"
            )

        raise LoweringError(f"Unsupported node type during lowering: {type(node).__name__}")


def _lower_if(node: IfNode, out: stim.Circuit, state: _LoweringState) -> None:
    cond = _extract_affine_condition(node.cond, state)
    ops = _extract_pauli_ops(node.body)

    if cond.constant:
        for gate, target in ops:
            out += stim.Circuit(f"{gate} {target}\n")

    if not cond.rec_controls:
        return

    for ctrl_abs in sorted(cond.rec_controls):
        rel = ctrl_abs - state.meas_count
        for gate, target in ops:
            out += stim.Circuit(f"C{gate} rec[{rel}] {target}\n")


def _extract_affine_condition(cond, state: _LoweringState) -> _AffineCondition:
    if isinstance(cond, LastMeas):
        if state.last_block_meas_start is None:
            raise LoweringError("LastMeas condition used before any capture_as_last block")
        if cond.index < 0:
            raise LoweringError("LastMeas indices must be non-negative for lowering")
        if cond.index >= state.last_block_meas_len:
            raise LoweringError(
                f"LastMeas({cond.index}) out of range for last block size {state.last_block_meas_len}"
            )
        return _AffineCondition(
            constant=False,
            rec_controls=frozenset({state.last_block_meas_start + cond.index}),
        )

    if isinstance(cond, MeasParity):
        return _AffineCondition(
            constant=False,
            rec_controls=frozenset(
                _resolve_measurement_index(i, state.meas_count) for i in cond.indices
            ),
        )

    if isinstance(cond, Expr):
        return _expr_to_affine_condition(cond, state)

    raise LoweringError(
        "Only LastMeas, MeasParity, and affine Expr conditions are supported by exact lowering"
    )


def _expr_to_affine_condition(expr: Expr, state: _LoweringState) -> _AffineCondition:
    constant, rec_controls, _has_var = _affine_expr_terms(expr, state)
    return _AffineCondition(constant=constant, rec_controls=frozenset(rec_controls))


def _affine_expr_terms(
    expr: Expr, state: _LoweringState
) -> tuple[bool, set[int], bool]:
    """
    Returns (constant, rec_controls, has_var_dependency).
    Supports affine boolean expressions over rec[...] and literals.
    """
    if isinstance(expr, LiteralExpr):
        return (bool(expr.value) & 1 == 1, set(), False)

    if isinstance(expr, RecExpr):
        return (False, {_resolve_measurement_index(expr.index, state.meas_count)}, False)

    if isinstance(expr, InvertExpr):
        c, recs, has_var = _affine_expr_terms(expr.expr, state)
        return (not c, recs, has_var)

    if isinstance(expr, XorExpr):
        c1, r1, v1 = _affine_expr_terms(expr.left, state)
        c2, r2, v2 = _affine_expr_terms(expr.right, state)
        out = set(r1)
        out.symmetric_difference_update(r2)
        return (bool(c1) ^ bool(c2), out, v1 or v2)

    if isinstance(expr, VarExpr):
        if state.vars is None or expr.name not in state.vars:
            raise LoweringError(
                f"Variable {expr.name!r} is not defined for compile-time lowering"
            )
        value = state.vars[expr.name]
        return (bool(int(value) & 1), set(), True)

    if isinstance(expr, AddExpr):
        c1, r1, v1 = _affine_expr_terms(expr.left, state)
        c2, r2, v2 = _affine_expr_terms(expr.right, state)
        out = set(r1)
        out.symmetric_difference_update(r2)
        return (bool(c1) ^ bool(c2), out, v1 or v2)

    if isinstance(expr, ModExpr):
        if not isinstance(expr.right, LiteralExpr) or int(expr.right.value) != 2:
            raise LoweringError("Only modulo-2 affine expressions are supported in lowering")
        return _affine_expr_terms(expr.left, state)

    raise LoweringError(
        "Expr lowering supports only affine boolean forms using rec/vars, literals, ^, ~, +, and % 2"
    )


def _extract_pauli_ops(body: Circuit) -> List[tuple[str, int]]:
    ops: List[tuple[str, int]] = []
    for child in body.nodes:
        if not isinstance(child, StimBlock):
            raise LoweringError(
                "Conditional lowering only supports StimBlock bodies containing Pauli corrections"
            )
        if _num_measurements(child.circuit) != 0:
            raise LoweringError(
                "Conditional lowering does not support measurements inside IfNode bodies"
            )
        ops.extend(_parse_pauli_ops_from_stim_block(child.circuit))
    return ops


def _lower_repeat(node: RepeatNode, out: stim.Circuit, state: _LoweringState) -> None:
    if node.repetitions == 0:
        return
    if node.repetitions == 1:
        _lower_program(node.body, out, state)
        return

    for _ in range(node.repetitions):
        _lower_program(node.body, out, state)


def _lower_dynamic_loop_if_compile_time_deterministic(
    node: WhileNode | DoWhileNode, out: stim.Circuit, state: _LoweringState
) -> None:
    """
    Supports loops whose termination is compile-time deterministic via vars/literals only.
    Any dependency on measurement record bits is rejected.
    """
    iterations = 0
    seen_states: set[tuple[tuple[str, int], ...]] = set()

    if isinstance(node, WhileNode):
        while _compile_time_loop_cond(node.cond, state):
            key = _loop_state_key(state)
            if key in seen_states:
                raise LoweringError(
                    "Loop entered a compile-time periodic cycle and does not terminate"
                )
            seen_states.add(key)
            iterations += 1
            if iterations > node.max_iter:
                raise LoweringError(
                    f"Loop exceeds max_iter={node.max_iter} during compile-time lowering"
                )
            _lower_program(node.body, out, state)
        return

    while True:
        key = _loop_state_key(state)
        if key in seen_states:
            raise LoweringError(
                "Do-while loop entered a compile-time periodic cycle and does not terminate"
            )
        seen_states.add(key)
        iterations += 1
        if iterations > node.max_iter:
            raise LoweringError(
                f"Do-while loop exceeds max_iter={node.max_iter} during compile-time lowering"
            )
        _lower_program(node.body, out, state)
        if not _compile_time_loop_cond(node.cond, state):
            return


def _compile_time_loop_cond(cond, state: _LoweringState) -> bool:
    if isinstance(cond, (LastMeas, MeasParity)):
        raise LoweringError(
            "Loop lowering only supports compile-time deterministic conditions (no rec dependencies)"
        )
    if isinstance(cond, Expr):
        return bool(_evaluate_compile_time_expr(cond, state))
    raise LoweringError(
        "Loop lowering only supports compile-time deterministic Expr conditions"
    )


def _lower_let(node: LetNode, state: _LoweringState) -> None:
    expr = node.expr
    if isinstance(expr, Expr):
        value = _evaluate_compile_time_expr(expr, state)
    else:
        raise LoweringError(
            "LetNode lowering only supports stimdx Expr instances for compile-time variables"
        )
    if state.vars is None:
        state.vars = {}
    state.vars[node.name] = value


def _evaluate_compile_time_expr(expr: Expr, state: _LoweringState) -> int | bool:
    if isinstance(expr, LiteralExpr):
        return expr.value
    if isinstance(expr, VarExpr):
        if state.vars is None or expr.name not in state.vars:
            raise LoweringError(
                f"Variable {expr.name!r} is not defined for compile-time lowering"
            )
        return state.vars[expr.name]
    if isinstance(expr, InvertExpr):
        return 1 - int(bool(_evaluate_compile_time_expr(expr.expr, state)))
    if isinstance(expr, XorExpr):
        return int(_evaluate_compile_time_expr(expr.left, state)) ^ int(
            _evaluate_compile_time_expr(expr.right, state)
        )
    if isinstance(expr, AddExpr):
        return int(_evaluate_compile_time_expr(expr.left, state)) + int(
            _evaluate_compile_time_expr(expr.right, state)
        )
    if isinstance(expr, ModExpr):
        return int(_evaluate_compile_time_expr(expr.left, state)) % int(
            _evaluate_compile_time_expr(expr.right, state)
        )
    if isinstance(expr, RecExpr):
        raise LoweringError(
            "Compile-time variable expressions cannot depend on measurement record bits"
        )
    raise LoweringError(
        f"Compile-time Let expression not supported: {type(expr).__name__}"
    )


def _loop_state_key(state: _LoweringState) -> tuple[tuple[str, int], ...]:
    vars_items: tuple[tuple[str, int], ...]
    if not state.vars:
        vars_items = ()
    else:
        vars_items = tuple(
            sorted((name, int(value)) for name, value in state.vars.items())
        )
    return vars_items


def _parse_pauli_ops_from_stim_block(block: stim.Circuit) -> List[tuple[str, int]]:
    ops: List[tuple[str, int]] = []
    text = str(block)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "{" in line or "}" in line:
            raise LoweringError("REPEAT blocks are not supported inside lowered IfNode bodies")

        parts = line.split()
        op_name = parts[0]
        if "(" in op_name or ")" in op_name:
            raise LoweringError(
                f"Parameterized operation '{op_name}' is not supported in lowered IfNode bodies"
            )
        if op_name not in {"X", "Y", "Z"}:
            raise LoweringError(
                f"Only X/Y/Z are supported in lowered IfNode bodies (got '{op_name}')"
            )
        for token in parts[1:]:
            try:
                q = int(token)
            except ValueError as ex:
                raise LoweringError(
                    f"Unsupported target '{token}' in lowered IfNode body"
                ) from ex
            if q < 0:
                raise LoweringError("Negative qubit targets are not supported")
            ops.append((op_name, q))
    return ops


def _detector_line(
    indices: Sequence[int], meas_count: int, coords: Sequence[float] | None
) -> str:
    rec_terms = " ".join(_rec_token(i, meas_count) for i in indices)
    if coords:
        coords_text = ",".join(_format_coord(v) for v in coords)
        return f"DETECTOR({coords_text}) {rec_terms}"
    return f"DETECTOR {rec_terms}"


def _observable_line(observable_index: int, indices: Sequence[int], meas_count: int) -> str:
    rec_terms = " ".join(_rec_token(i, meas_count) for i in indices)
    return f"OBSERVABLE_INCLUDE({observable_index}) {rec_terms}"


def _rec_token(index: int, meas_count: int) -> str:
    abs_index = _resolve_measurement_index(index, meas_count)
    rel = abs_index - meas_count
    return f"rec[{rel}]"


def _resolve_measurement_index(index: int, meas_count: int) -> int:
    abs_index = meas_count + index if index < 0 else index
    if abs_index < 0 or abs_index >= meas_count:
        raise LoweringError(
            f"Measurement index {index} out of range at lowering point (size {meas_count})"
        )
    return abs_index


def _num_measurements(circuit: stim.Circuit) -> int:
    count = getattr(circuit, "num_measurements", None)
    if count is None:
        raise LoweringError(
            "Stim version does not expose circuit.num_measurements; upgrade Stim to use lowering"
        )
    return int(count)


def _format_coord(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))
