from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Callable, Optional
import stim

from ._context import ExecContext
from ._cond import Cond


# Forward declaration handled by annotations,
# but we need to deal with DynamicSampler import cycle?
# DynamicSampler needs Circuit. Circuit needs DynamicSampler (compile_sampler).
# Solution: Import DynamicSampler inside the method or use TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._execution import DynamicSampler

    import jeff

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
    cond: Union[Cond, Callable[[ExecContext], bool]]
    body: Circuit


@dataclass
class WhileNode(Node):
    """Node representing a while-loop (check condition before execution)."""

    cond: Union[Cond, Callable[[ExecContext], bool]]
    body: Circuit
    max_iter: int = 10_000


@dataclass
class DoWhileNode(Node):
    """Node representing a do-while loop (execute body once, then check condition)."""

    cond: Union[Cond, Callable[[ExecContext], bool]]
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
        body: Union[stim.Circuit, Circuit | str],
        cond: Union[Cond, Callable[[ExecContext], bool]],
    ) -> Circuit:
        """
        Appends a conditional block (If statement).
        Args:
            body: The circuit to execute if `cond` evaluates to True.
            cond: The condition to evaluate.
        """
        if (is_str := isinstance(body, str)) or isinstance(body, stim.Circuit):
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
        cond: Union[Cond, Callable[[ExecContext], bool]],
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Appends a while-loop.
        Args:
            body: The loop body.
            cond: Loop continues while this evaluates to True.
        """
        self.nodes.append(WhileNode(cond=cond, body=body, max_iter=max_iter))
        return self

    def do_while(
        self,
        body: Circuit,
        cond: Union[Cond, Callable[[ExecContext], bool]],
        max_iter: int = 10000,
    ) -> Circuit:
        """
        Executes 'body' at least once, then repeats while 'cond' is true.
        Useful for Repeat-Until-Success patterns (run -> measure -> check).
        """
        self.nodes.append(DoWhileNode(cond=cond, body=body, max_iter=max_iter))
        return self

    def compile_sampler(self, seed: Optional[int] = None) -> DynamicSampler:
        """Creates a sampler for the circuit."""
        from ._execution import DynamicSampler

        return DynamicSampler(self, seed=seed)

    def to_jeff(self, name: str = "main") -> jeff.JeffModule:
        """
        Export the circuit to JEFF format.

        JEFF (JSON Exchange Format For circuits) is an intermediate representation
        for quantum circuits that can be converted to HUGR and executed with Guppy.

        Args:
            name: Name for the main function in the module.

        Returns:
            A JeffModule representing the circuit.

        Raises:
            ValueError: If the circuit contains lambda conditionals, which cannot
                be serialized to JEFF's declarative format. Use Cond subclasses
                (LastMeas, MeasParity) instead.

        Example:
            >>> circuit = Circuit()
            >>> circuit.block("H 0")
            >>> circuit.block("M 0")
            >>> module = circuit.to_jeff()
        """
        from ._jeff import to_jeff

        return to_jeff(self, name)

    def write_jeff(self, path: str, name: str = "main") -> None:
        """
        Export the circuit to a .jeff file.

        Args:
            path: Path to the output file.
            name: Name for the main function in the module.

        Raises:
            ImportError: If pycapnp is not installed or schema cannot be loaded.
            ValueError: If the circuit contains lambda conditionals.
        """
        from ._jeff import write_jeff

        write_jeff(self, path, name)
