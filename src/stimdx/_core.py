from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional
import stim

from ._cond import Cond, LastMeas, MeasParity


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._execution import DynamicSampler

# ---- AST Nodes ----


class Node:
    """Base class for AST nodes."""

    pass


@dataclass
class StimBlock(Node):
    """Leaf node representing a static Stim circuit block."""

    circuit: stim.Circuit
    capture_as_last: bool = True


@dataclass
class IfNode(Node):
    """Node representing a conditional branch."""

    cond: Cond
    body: Circuit


@dataclass
class WhileNode(Node):
    """Node representing a while-loop (check condition before execution)."""

    cond: Cond
    body: Circuit
    max_iter: int = 10_000


@dataclass
class DoWhileNode(Node):
    """Node representing a do-while loop (execute body once, then check condition)."""

    cond: Cond
    body: Circuit
    max_iter: int = 10_000


# ---- Circuit Builder ----


class Circuit:
    """
    A builder for dynamic circuits.
    accumulates an AST (Abstract Syntax Tree) of nodes.
    """

    def __init__(self, block: Optional[str | stim.Circuit] = None):
        self.nodes: List[Node] = []
        if block:
            self.block(block)

    def __str__(self) -> str:
        return self._str_recursive(indent=0)

    def _str_recursive(self, indent: int) -> str:
        lines = []
        prefix = "  " * indent

        for node in self.nodes:
            if isinstance(node, StimBlock):
                lines.append(f"{prefix}StimBlock:")
                for line in str(node.circuit).split("\n"):
                    if line.strip():
                        lines.append(f"{prefix}  {line}")

            elif isinstance(node, IfNode):
                lines.append(f"{prefix}If {node.cond}:")
                lines.append(node.body._str_recursive(indent + 1))

            elif isinstance(node, WhileNode):
                lines.append(f"{prefix}While {node.cond}:")
                lines.append(node.body._str_recursive(indent + 1))

            elif isinstance(node, DoWhileNode):
                lines.append(f"{prefix}Do:")
                lines.append(node.body._str_recursive(indent + 1))
                lines.append(f"{prefix}While {node.cond}")

            else:
                lines.append(f"{prefix}{node}")

        return "\n".join(lines)

    def block(self, c: stim.Circuit | str, *, capture_as_last: bool = True) -> Circuit:
        """
        Appends a static block of Stim instructions.
        Args:
            c: The stim.Circuit or string representation.
            capture_as_last: If True, measurements in this block are captured for LastMeas conditions.
        """
        if isinstance(c, str):
            c = stim.Circuit(c)
        self.nodes.append(StimBlock(c, capture_as_last=capture_as_last))
        return self

    def conditional(
        self,
        body: Union[stim.Circuit, Circuit, str],
        cond: Cond,
    ) -> Circuit:
        """
        Appends a conditional block (If statement).
        Args:
            body: The circuit to execute if `cond` evaluates to True.
            cond: The condition to evaluate (must be a Cond object, not a lambda).
        """
        if isinstance(body, stim.Circuit) or (is_str := isinstance(body, str)):
            if is_str:
                body = stim.Circuit(body)
            c = Circuit()
            c.block(body)
            body = c
        self.nodes.append(IfNode(cond=cond, body=body))
        return self

    def while_loop(
        self,
        body: Circuit,
        cond: Cond,
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Appends a while-loop.
        Args:
            body: The loop body.
            cond: Loop continues while this evaluates to True (must be a Cond object).
        """
        self.nodes.append(WhileNode(cond=cond, body=body, max_iter=max_iter))
        return self

    def do_while(
        self,
        body: Circuit,
        cond: Cond,
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Executes 'body' at least once, then repeats while 'cond' is true.
        Useful for Repeat-Until-Success patterns (run -> measure -> check).
        Args:
            cond: Must be a Cond object (not a lambda) for C++ execution.
        """
        self.nodes.append(DoWhileNode(cond=cond, body=body, max_iter=max_iter))
        return self

    def to_proto(self) -> bytes:
        """
        Serialize the circuit AST to protobuf bytes for C++ execution.
        """
        from . import stimdx_pb2 as pb

        def serialize_condition(cond: Cond) -> pb.Condition:
            pb_cond = pb.Condition()
            if isinstance(cond, LastMeas):
                pb_cond.last_meas.index = cond.index
            elif isinstance(cond, MeasParity):
                pb_cond.meas_parity.indices.extend(cond.indices)
            else:
                raise TypeError(f"Unsupported condition type: {type(cond)}")
            return pb_cond

        def serialize_circuit(circuit: Circuit) -> pb.Circuit:
            pb_circuit = pb.Circuit()
            for node in circuit.nodes:
                pb_node = pb.Node()

                if isinstance(node, StimBlock):
                    pb_node.stim_block.stim_circuit_text = str(node.circuit)
                    pb_node.stim_block.capture_as_last = node.capture_as_last

                elif isinstance(node, IfNode):
                    pb_node.if_node.condition.CopyFrom(serialize_condition(node.cond))
                    pb_node.if_node.body.CopyFrom(serialize_circuit(node.body))

                elif isinstance(node, WhileNode):
                    pb_node.while_node.condition.CopyFrom(
                        serialize_condition(node.cond)
                    )
                    pb_node.while_node.body.CopyFrom(serialize_circuit(node.body))
                    pb_node.while_node.max_iter = node.max_iter

                elif isinstance(node, DoWhileNode):
                    pb_node.do_while_node.condition.CopyFrom(
                        serialize_condition(node.cond)
                    )
                    pb_node.do_while_node.body.CopyFrom(serialize_circuit(node.body))
                    pb_node.do_while_node.max_iter = node.max_iter

                else:
                    raise TypeError(f"Unknown node type: {type(node)}")

                pb_circuit.nodes.append(pb_node)

            return pb_circuit

        return serialize_circuit(self).SerializeToString()

    def compile_sampler(self, seed: Optional[int] = None) -> DynamicSampler:
        """Creates a sampler for the circuit."""
        from ._execution import DynamicSampler

        return DynamicSampler(self, seed=seed)
