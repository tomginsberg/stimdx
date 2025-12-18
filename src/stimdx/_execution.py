from __future__ import annotations
from typing import List, Optional, Union, Callable
import stim

from ._context import ExecContext
from ._core import Circuit, StimBlock, IfNode, WhileNode, DoWhileNode
from ._cond import Cond


class DynamicSampler:
    """
    Executes the Circuit by running it shot-by-shot in a Python loop.
    """

    def __init__(self, program: Circuit, *, seed: Optional[int] = None):
        self.program = program
        self.seed = seed

    def sample(self, *, shots: int) -> List[List[bool]]:
        """
        Samples 'shots' times. Returns a list of measurement lists (one per shot).
        Note: The length of each measurement list might vary if the circuit is dynamic!
        """
        all_samples = []
        for s in range(shots):
            # Create a fresh simulator for each shot
            # We seed it deterministically if a master seed is provided
            current_seed = None if self.seed is None else (self.seed + s)
            sim = stim.TableauSimulator(seed=current_seed)

            ctx = ExecContext(sim=sim, meas_record=[], last_block_meas=[])

            execute(self.program, ctx)

            all_samples.append(list(ctx.meas_record))

        return all_samples


def execute(program: Circuit, ctx: ExecContext):
    """Recursively executes the program AST against the context."""
    for node in program.nodes:
        if isinstance(node, StimBlock):
            before_len = len(ctx.sim.current_measurement_record())
            ctx.sim.do(node.circuit)
            full_record = ctx.sim.current_measurement_record()
            new_meas = full_record[before_len:]
            ctx.meas_record.extend(new_meas)

            if node.capture_as_last:
                ctx.last_block_meas = list(new_meas)

        elif isinstance(node, IfNode):
            if _eval_cond(node.cond, ctx):
                execute(node.body, ctx)

        elif isinstance(node, WhileNode):
            iterations = 0
            while _eval_cond(node.cond, ctx):
                iterations += 1
                if iterations > node.max_iter:
                    raise RuntimeError(f"While-loop exceeded max_iter={node.max_iter}")
                execute(node.body, ctx)
        elif isinstance(node, DoWhileNode):
            iterations = 0
            while True:
                iterations += 1
                if iterations > node.max_iter:
                    raise RuntimeError(
                        f"Do-While loop exceeded max_iter={node.max_iter}"
                    )
                execute(node.body, ctx)
                if not _eval_cond(node.cond, ctx):
                    break
        else:
            raise TypeError(f"Unknown node type: {type(node)}")


def _eval_cond(
    cond: Union[Cond, Callable[[ExecContext], bool]], ctx: ExecContext
) -> bool:
    if isinstance(cond, Cond):
        return cond.eval(ctx)
    return cond(ctx)
