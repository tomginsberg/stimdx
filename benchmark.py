#!/usr/bin/env python3
"""Benchmark comparing Python vs C++ execution backends for stimdx."""

import time
from stimdx import Circuit, LastMeas
import stimdx


def create_benchmark_circuit(num_blocks: int = 100) -> Circuit:
    """Create a benchmark circuit with gates, measurements, and conditionals."""
    c = Circuit()
    for i in range(num_blocks):
        # Block with Hadamards, CNOTs, and measurements
        c.block(f"H 0 1 2 3\nCNOT 0 1\nCNOT 2 3\nH 4 5\nCNOT 4 5\nM 0 1 2")
        # Conditional correction
        c.conditional(body="X 3 4", cond=LastMeas(0))
    return c


def benchmark_shots(circuit: Circuit, shots_list: list[int]):
    """Benchmark varying shot counts."""
    sampler = circuit.compile_sampler(seed=42)

    # Warmup
    sampler.sample(shots=10, use_cpp=False)
    if stimdx.__cpp_available__:
        sampler.sample(shots=10, use_cpp=True)

    print(f"\n{'Shots':>6} | {'Python':>10} | {'C++':>10} | {'Speedup':>8}")
    print("-" * 45)

    for shots in shots_list:
        # Python
        start = time.perf_counter()
        sampler.sample(shots=shots, use_cpp=False)
        py_time = time.perf_counter() - start

        if stimdx.__cpp_available__:
            # C++
            start = time.perf_counter()
            sampler.sample(shots=shots, use_cpp=True)
            cpp_time = time.perf_counter() - start
            speedup = py_time / cpp_time
            print(
                f"{shots:>6} | {py_time * 1000:>8.1f}ms | {cpp_time * 1000:>8.1f}ms | {speedup:>7.2f}x"
            )
        else:
            print(f"{shots:>6} | {py_time * 1000:>8.1f}ms |        N/A |      N/A")


def main():
    print("=" * 50)
    print("stimdx Python vs C++ Benchmark")
    print("=" * 50)
    print(f"\nC++ backend available: {stimdx.__cpp_available__}")
    if stimdx.__cpp_available__:
        print(f"C++ version: {stimdx.__cpp_version__}")

    # Create benchmark circuit
    circuit = create_benchmark_circuit(num_blocks=100)
    print(f"\nCircuit: 100 blocks with conditionals")
    print(f"  Total nodes: {len(circuit.nodes)}")

    # Run benchmarks
    benchmark_shots(circuit, [100, 250, 500, 1000, 2000])

    print("\n" + "=" * 50)
    print("Benchmark complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
