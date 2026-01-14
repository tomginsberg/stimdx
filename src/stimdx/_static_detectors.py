import stim
import numpy as np
from typing import Optional
import math


class StaticDetectorSampler:
    """
    A sampler that delegates to Stim's compiled detector sampler.
    Efficient for static circuits with DETECTOR/OBSERVABLE_INCLUDE instructions.
    """

    def __init__(self, stim_circuit: stim.Circuit, seed: Optional[int] = None):
        self._stim_circuit = stim_circuit
        self._seed = seed
        self._compiled_sampler = stim_circuit.compile_detector_sampler(seed=seed)

    def sample(
        self,
        shots: int,
        *,
        prepend_observables: bool = False,
        append_observables: bool = True,
        bit_packed: bool = False,
    ) -> np.ndarray:
        """
        Samples detection events (and optionally observables) from the circuit.
        Delegates directly to stim.CompiledDetectorSampler.sample().

        Args:
            shots: Number of shots to sample.
            prepend_observables: If True, observable bits are prepended to the output.
            append_observables: If True, observable bits are appended to the output.
            bit_packed: If True, returns packed bits (uint8). If False, returns 0/1 (uint8/bool).

        Returns:
            A numpy array of sample data.
        """
        return self._compiled_sampler.sample(
            shots=shots,
            prepend_observables=prepend_observables,
            append_observables=append_observables,
            bit_packed=bit_packed,
        )
