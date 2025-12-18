# stimdx

stimdx is a simple Python extension for [Stim](https://github.com/quantumlib/Stim) that adds support for dynamic circuits (branching, loops, and conditional logic) based on real-time measurement results. 

## Quick Start

Here is a "Repeat Until Success" (RUS) loop that creates outcomes until it measures a `0`.

```python
from stimdx import Circuit, LastMeas

c = Circuit("H 0\nM 0")
fix = Circuit("H 0\nM 0")
c.while_loop(body=fix, cond=LastMeas(0))
sampler = c.compile_sampler(seed=123)
samples = sampler.sample(shots=3)

# [[True, False], [True, True, False], [False]]
```

## Philosophy

stimdx is inspired by modern QC languages/compilers like Catalyst and Guppylang, while keeping the "minimalism" of stim. It doesn't require LLVM, complex IRs, or a new embeddeed language (not that these ideas are bad).

## How It Works

stimdx works by building a high-level abstract syntax tree (AST) that sits *above* Stim.

### 1. The Data Structure
A `stimdx.Circuit` is a tree of AST Nodes.
- StimBlock A chunk of standard, static `stim.Circuit`. These are the leaves of the tree.
- `IfNode`: Conditional branching. Contains a child `Circuit` that only runs if a condition is met.
- `WhileNode`: Standard `while` loop. Repeats a child `Circuit`.
- `DoWhileNode`: Executes a child `Circuit` once, then repeats based on a condition.

Because the nodes are recursive (e.g., an `IfNode` contains a `Circuit`, which can contain a `WhileNode`), you can nest control flow arbitrarily deep.

### 2. The Execution Flow
When you run `sampler.sample()`, stimdx runs a hybrid circuit simulation directly in python that involves:

1.  State Tracking: A single `stim.TableauSimulator` is kept alive for the duration of a shot.
2.  AST Traversal: The Python interpreter walks the AST.
3.  Just-In-Time Execution:
    - When it hits a `StimBlock`, it hands that chunk of gates to the simulator (`sim.do(block)`).
    - It immediately retrieves the *new* measurement results from the simulator.
    - Branching decisions (`If` / `While`) are made in Python using these real, live measurement results.
4.  Result Stitching: The measurement bits from all executed blocks are stitched together into a single history for that shot.

### 3. Simulator State Persistence

The key to this hybrid execution is that the `stim.TableauSimulator` object acts as the persistent state for the full stabilizer tableau between execution of conditional logic.

Suppose we have a conditional correction on qubit 0 while qubit 1 stays in memory.

```python
c.block("H 0 1\nM 0")
c.conditional(body="X 0", cond=LastMeas(0))
c.block("M 0 1")
```

- Block A Runs: `sim.do("H 0 1\nM 0")`
   - The simulator evolves the tableau. Qubits 0 and 1 are in superposition.
   - Qubit 0 is measured. The result (e.g., `True`) is returned to Python.
   - Crucial Qubit 1 is still in the `|+>` state inside the simulator object.
- Python Logic:
   - Python sees `LastMeas(0)` is `True`. It decides to enter the `If` block.
- Block B Runs: `sim.do("X 0")`
   - The simulator applies X to qubit 0.
   - Crucial: The state of Qubit 1 is *untouched* and persists perfectly.
- Block C Runs: `sim.do("M 0 1")`
   - We measure both. Qubit 1 is finally collapsed.
