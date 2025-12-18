from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ._context import ExecContext


class Cond:
    """Base class for structured conditions."""

    def eval(self, ctx: ExecContext) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class LastMeas(Cond):
    """
    Helper to access the i-th measurement of the *immediately preceding* block.
    This is safer than raw indexing if you just finished a block and want to branch.
    """

    index: int

    def eval(self, ctx: ExecContext) -> bool:
        if self.index >= len(ctx.last_block_meas):
            raise IndexError(
                f"LastMeas index {self.index} out of range for last block of size {len(ctx.last_block_meas)}"
            )
        return bool(ctx.last_block_meas[self.index])


@dataclass(frozen=True)
class MeasParity(Cond):
    """
    Checks the parity of a set of measurements (specified by indices).
    Indices are relative to the *global* measurement record of the current shot.
    Supports negative indexing (e.g. -1 for the last measurement).
    """

    indices: List[int]

    def eval(self, ctx: ExecContext) -> bool:
        parity = 0
        total_len = len(ctx.meas_record)
        for i in self.indices:
            try:
                val = ctx.meas_record[i]
            except IndexError:
                raise IndexError(
                    f"MeasParity index {i} out of range for record of size {total_len}"
                )

            if val:
                parity ^= 1
        return parity == 1
