from __future__ import annotations
from dataclasses import dataclass
from typing import List
import stim


@dataclass
class ExecContext:
    """
    Holds the runtime state of a dynamic circuit execution.
    """

    sim: stim.TableauSimulator
    meas_record: List[bool]
    last_block_meas: List[bool]
