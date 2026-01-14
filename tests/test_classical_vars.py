import stim
import stimdx
import pytest


def test_variable_assignment_and_use():
    c = stimdx.Circuit()
    c.block("H 0\nM 0")
    c.let("m", lambda ctx: ctx.meas_record[-1])
    c.emit(lambda ctx: bool(ctx.vars["m"]), name="m_out")

    sampler = c.compile_sampler(seed=123)
    results = sampler.sample_with_classical(shots=100)

    for res in results:
        meas = res["measurements"]
        outputs = res["outputs"]
        assert len(outputs) == 1
        assert outputs[0] == meas[-1]
        assert res["output_names"] == ["m_out"]
        assert res["vars"]["m"] == meas[-1]


def test_use_classical_var_in_control_flow():
    # Measure two qubits, store parity, apply conditional quantum op, then measure again.
    c = stimdx.Circuit()
    c.block("R 0 1\nH 0 1\nCX 0 1\nM 0 1")
    # Store parity of first two measurements
    c.let("p", lambda ctx: ctx.meas_record[-1] ^ ctx.meas_record[-2])

    # Conditional logic: if parity is 1, apply X on 0
    c.conditional("X 0", lambda ctx: bool(ctx.vars["p"]))

    c.block("M 0")

    sampler = c.compile_sampler(seed=123)
    results = sampler.sample_with_classical(shots=100)

    for res in results:
        m0 = res["measurements"][0]
        m1 = res["measurements"][1]
        parity = m0 ^ m1
        assert res["vars"]["p"] == parity
        # We verify that 'p' was correctly calculated and stored.


def test_multiple_outputs_stable_order():
    c = stimdx.Circuit()
    c.emit(lambda ctx: True, name="first")
    c.emit(lambda ctx: False, name="second")

    sampler = c.compile_sampler()
    results = sampler.sample_with_classical(shots=10)

    for res in results:
        assert res["outputs"] == [True, False]
        assert res["output_names"] == ["first", "second"]


def test_negative_indexing_helper():
    c = stimdx.Circuit()
    c.block("M 0 1")

    # Check ctx.rec(-1) and ctx.rec(-2)
    # We can use let to capture them
    c.let("last", lambda ctx: ctx.rec(-1))
    c.let("second_last", lambda ctx: ctx.rec(-2))

    sampler = c.compile_sampler(seed=123)
    results = sampler.sample_with_classical(shots=10)

    for res in results:
        assert res["vars"]["last"] == res["measurements"][-1]
        assert res["vars"]["second_last"] == res["measurements"][-2]


def test_rec_out_of_bounds():
    c = stimdx.Circuit()
    # No measurements yet
    c.let("fail", lambda ctx: ctx.rec(0))

    sampler = c.compile_sampler()

    with pytest.raises(IndexError, match="out of bounds"):
        sampler.sample_with_classical(shots=1)
