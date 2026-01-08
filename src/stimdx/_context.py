from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Union
import stim


@dataclass
class ExecContext:
    """
    Holds the runtime state of a dynamic circuit execution.
    """

    sim: stim.TableauSimulator
    meas_record: List[bool]
    last_block_meas: List[bool]
    vars: Dict[str, Union[int, bool]] = field(default_factory=dict)
    outputs: List[bool] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    def rec(self, i: int) -> bool:
        """Helper to access measurement record with bounds checking and negative indexing."""
        try:
            return self.meas_record[i]
        except IndexError:
            raise IndexError(
                f"Measurement record index {i} out of bounds (size {len(self.meas_record)})"
            )
