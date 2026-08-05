import pytest
import stim
import stimdx
from stimdx import context as ctx
from stimdx._expr import LiteralExpr


def test_dynamic_detector_and_observable_runtime_outputs():
    c = stimdx.Circuit()
    c.block("X 0\nM 0")
    c.detector(-1)
    c.observable_include(0, -1)

    res = c.compile_sampler(seed=1).sample_with_classical(shots=1)[0]
    assert res["measurements"] == [True]
    assert res["detectors"] == [True]
    assert res["observables"] == {0: True}


def test_lower_if_lastmeas_to_rec_controlled_pauli():
    c = stimdx.Circuit()
    c.block("M 0")
    c.conditional("X 1\nZ 2", stimdx.LastMeas(0))
    c.block("M 1 2", capture_as_last=False)
    c.detector(-1, -2)
    c.observable_include(0, 0)

    lowered = c.to_stim_lowered()
    lowered_text = str(lowered)

    assert "CX rec[-1] 1" in lowered_text
    assert "CZ rec[-1] 2" in lowered_text
    assert "DETECTOR rec[-1] rec[-2]" in lowered_text
    assert "OBSERVABLE_INCLUDE(0) rec[-3]" in lowered_text


def test_lower_meas_parity_conditional():
    c = stimdx.Circuit()
    c.block("M 0 1")
    c.conditional("X 2", stimdx.MeasParity([0, 1]))

    lowered_text = str(c.to_stim_lowered())
    assert "CX rec[-2] 2" in lowered_text
    assert "CX rec[-1] 2" in lowered_text


def test_lower_affine_expr_conditional_with_invert_and_xor():
    c = stimdx.Circuit()
    c.block("M 0 1")
    # ~(rec[-1] ^ rec[-2]) == 1 ^ rec[-1] ^ rec[-2]
    c.conditional("X 2", ~(ctx.rec(-1) ^ ctx.rec(-2)))

    lowered_text = str(c.to_stim_lowered())
    assert "X 2" in lowered_text
    assert "CX rec[-1] 2" in lowered_text
    assert "CX rec[-2] 2" in lowered_text


def test_lower_affine_expr_conditional_mod2_add():
    c = stimdx.Circuit()
    c.block("M 0 1")
    c.conditional("Z 2", (ctx.rec(-1) + ctx.rec(-2)) % 2)

    lowered_text = str(c.to_stim_lowered())
    assert "CZ rec[-1] 2" in lowered_text
    assert "CZ rec[-2] 2" in lowered_text


def test_repeat_node_executes_and_lowers_by_unrolling():
    c = stimdx.Circuit()
    c.repeat(3, "M 0")
    c.detector(-1)

    samples = c.compile_sampler(seed=1).sample(shots=2)
    assert all(len(s) == 3 for s in samples)

    lowered_text = str(c.to_stim_lowered())
    assert lowered_text.count("M 0") == 3
    assert "DETECTOR rec[-1]" in lowered_text


def test_compile_time_deterministic_while_expr_lowering():
    c = stimdx.Circuit()
    c.while_loop("X 0", LiteralExpr(False))
    c.block("M 0")

    lowered_text = str(c.to_stim_lowered())
    assert "X 0" not in lowered_text
    assert "M 0" in lowered_text


def test_compile_time_var_controlled_while_lowers():
    c = stimdx.Circuit()
    c.let("go", LiteralExpr(True))
    body = stimdx.Circuit()
    body.block("X 0")
    body.let("go", ~ctx.vars["go"])
    c.while_loop(body, ctx.vars["go"], max_iter=5)
    c.block("M 0")

    lowered_text = str(c.to_stim_lowered())
    assert lowered_text.count("X 0") == 1
    assert "M 0" in lowered_text


def test_compile_time_infinite_while_is_rejected():
    c = stimdx.Circuit()
    c.while_loop("X 0", LiteralExpr(True), max_iter=3)

    with pytest.raises(stimdx.LoweringError, match="periodic cycle|max_iter"):
        c.to_stim_lowered()


def test_lowering_rejects_loops():
    c = stimdx.Circuit()
    c.block("M 0")
    c.while_loop("X 0", stimdx.LastMeas(0))

    with pytest.raises(stimdx.LoweringError, match="loops"):
        c.to_stim_lowered()


def test_lowering_rejects_measurement_in_conditional_body():
    c = stimdx.Circuit()
    c.block("M 0")
    c.conditional("M 1", stimdx.LastMeas(0))

    with pytest.raises(stimdx.LoweringError, match="measurements"):
        c.to_stim_lowered()


def test_to_sinter_task_import_error_when_missing():
    c = stimdx.Circuit.from_stim(stim.Circuit("M 0\nDETECTOR rec[-1]"))

    with pytest.raises(ImportError, match="sinter is required"):
        c.to_sinter_task()
