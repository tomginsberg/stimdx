from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Callable, Optional
import stim

from ._context import ExecContext
from ._cond import Cond
from ._expr import Expr


# Forward declaration handled by annotations,
# but we need to deal with DynamicSampler import cycle?
# DynamicSampler needs Circuit. Circuit needs DynamicSampler (compile_sampler).
# Solution: Import DynamicSampler inside the method or use TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._execution import DynamicSampler
    from ._static_detectors import StaticDetectorSampler

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

    # Condition can be a specialized Cond object or a raw Python lambda
    cond: Union[Cond, Expr, Callable[[ExecContext], bool]]
    body: Circuit


@dataclass
class WhileNode(Node):
    """Node representing a while-loop (check condition before execution)."""

    cond: Union[Cond, Expr, Callable[[ExecContext], bool]]
    body: Circuit
    max_iter: int = 10_000


@dataclass
class DoWhileNode(Node):
    """Node representing a do-while loop (execute body once, then check condition)."""

    cond: Union[Cond, Expr, Callable[[ExecContext], bool]]
    body: Circuit
    max_iter: int = 10_000


@dataclass
class LetNode(Node):
    """Node representing a variable assignment."""

    name: str
    expr: Callable[[ExecContext], Union[int, bool]]


@dataclass
class EmitNode(Node):
    """Node representing an output emission."""

    expr: Callable[[ExecContext], bool]
    name: Optional[str] = None


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

    def __iadd__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        self.nodes.extend(other.nodes)
        return self

    def __add__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        combined = Circuit()
        combined.nodes.extend(self.nodes)
        combined.nodes.extend(other.nodes)
        return combined

    def _str_recursive(self, indent: int) -> str:
        lines = []
        prefix = "  " * indent

        for node in self.nodes:
            if isinstance(node, StimBlock):
                # lines.append(f"{prefix}StimBlock:")
                for line in str(node.circuit).split("\n"):
                    if line.strip():
                        lines.append(f"{prefix}{line}")

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

            elif isinstance(node, LetNode):
                lines.append(f"{prefix}Let {node.name} = <expr>")

            elif isinstance(node, EmitNode):
                name_str = f" ({node.name})" if node.name else ""
                lines.append(f"{prefix}Emit <expr>{name_str}")

            else:
                lines.append(f"{prefix}{node}")

        return "\n".join(lines)

    def block(
        self,
        c: stim.Circuit | str | Circuit | Node,
        *,
        capture_as_last: bool = True,
    ) -> Circuit:
        """
        Appends a static block of Stim instructions or an existing stimdx Circuit/Node.
        Args:
            c: The stim.Circuit, string representation, or a stimdx Circuit/Node.
            capture_as_last: If True, measurements in this block are captured for LastMeas conditions.
        """
        if isinstance(c, Circuit):
            self.nodes.extend(c.nodes)
            return self
        if isinstance(c, Node):
            self.nodes.append(c)
            return self
        if isinstance(c, str):
            c = stim.Circuit(c)
        self.nodes.append(StimBlock(c, capture_as_last=capture_as_last))
        return self

    def _wrap_body(self, body: Union[stim.Circuit, Circuit, str, Node]) -> Circuit:
        if isinstance(body, Circuit):
            return body
        if isinstance(body, Node):
            wrapper = Circuit()
            wrapper.nodes.append(body)
            return wrapper
        if isinstance(body, str):
            body = stim.Circuit(body)
        wrapper = Circuit()
        wrapper.block(body)
        return wrapper

    def node(self, node: Node) -> Circuit:
        """
        Appends a raw AST node to the circuit.
        """
        self.nodes.append(node)
        return self

    def conditional(
        self,
        body: Union[stim.Circuit, Circuit, str, Node],
        cond: Union[Cond, Expr, Callable[[ExecContext], bool]],
    ) -> Circuit:
        """
        Appends a conditional block (If statement).
        Args:
            body: The circuit to execute if `cond` evaluates to True.
            cond: The condition to evaluate.
        """
        wrapped_body = self._wrap_body(body)
        self.nodes.append(IfNode(cond=cond, body=wrapped_body))
        return self

    def while_loop(
        self,
        body: Union[stim.Circuit, Circuit, str, Node],
        cond: Union[Cond, Expr, Callable[[ExecContext], bool]],
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Appends a while-loop.
        Args:
            body: The loop body.
            cond: Loop continues while this evaluates to True.
        """
        wrapped_body = self._wrap_body(body)
        self.nodes.append(WhileNode(cond=cond, body=wrapped_body, max_iter=max_iter))
        return self

    def do_while(
        self,
        body: Union[stim.Circuit, Circuit, str, Node],
        cond: Union[Cond, Expr, Callable[[ExecContext], bool]],
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Executes 'body' at least once, then repeats while 'cond' is true.
        Useful for Repeat-Until-Success patterns (run -> measure -> check).
        """
        wrapped_body = self._wrap_body(body)
        self.nodes.append(DoWhileNode(cond=cond, body=wrapped_body, max_iter=max_iter))
        return self

    def let(self, name: str, expr: Callable[[ExecContext], int | bool]) -> Circuit:
        """
        Appends a LetNode to store a variable.
        """
        self.nodes.append(LetNode(name=name, expr=expr))
        return self

    def emit(
        self, expr: Callable[[ExecContext], bool], *, name: Optional[str] = None
    ) -> Circuit:
        """
        Appends an EmitNode to emit a classical output bit.
        """
        self.nodes.append(EmitNode(expr=expr, name=name))
        return self

    def compile_sampler(self, seed: Optional[int] = None) -> DynamicSampler:
        """Creates a sampler for the circuit."""
        from ._execution import DynamicSampler

        return DynamicSampler(self, seed=seed)

    def compile_detector_sampler(
        self, seed: Optional[int] = None
    ) -> StaticDetectorSampler:
        """
        Creates a detector sampler for the circuit.
        Only supported for static circuits (containing only StimBlocks).
        """
        if not self.is_static():
            raise NotImplementedError(
                "Detector sampling is not supported for dynamic circuits (must contain only StimBlocks)."
            )

        from ._static_detectors import StaticDetectorSampler

        return StaticDetectorSampler(self.to_stim(), seed=seed)

    @staticmethod
    def from_stim(c: stim.Circuit) -> Circuit:
        """
        Wraps an existing stim.Circuit in a stimdx Circuit without parsing text.
        """
        wrapper = Circuit()
        wrapper.nodes.append(StimBlock(c, capture_as_last=True))
        return wrapper

    def is_static(self) -> bool:
        """
        Returns True if the circuit contains only StimBlock nodes (no control flow).
        """
        return all(isinstance(node, StimBlock) for node in self.nodes)

    def to_stim(self) -> stim.Circuit:
        """
        Converts a static stimdx Circuit back to a stim.Circuit.
        Raises NotImplementedError if the circuit is not static.
        """
        if not self.is_static():
            raise NotImplementedError(
                "to_stim only supports static circuits (StimBlocks only)."
            )

        out = stim.Circuit()
        for node in self.nodes:
            # We know it's a StimBlock because of is_static check
            if isinstance(node, StimBlock):
                out += node.circuit
        return out
