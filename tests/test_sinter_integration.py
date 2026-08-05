import pytest

sinter = pytest.importorskip("sinter")
stim = pytest.importorskip("stim")

import stimdx
from stimdx import context as ctx


def test_to_sinter_task_static_smoke():
    c = stimdx.Circuit.from_stim(stim.Circuit("M 0\nDETECTOR rec[-1]"))
    task = c.to_sinter_task(json_metadata={"kind": "static"})

    assert isinstance(task, sinter.Task)
    assert getattr(task, "circuit", None) is not None
    assert getattr(task, "detector_error_model", None) is not None
    assert task.json_metadata["kind"] == "static"


def test_to_sinter_task_lowerable_dynamic_smoke():
    c = stimdx.Circuit()
    c.block("M 0")
    c.conditional("X 1", ctx.rec(-1))
    c.block("M 1")
    c.detector(-1)

    task = c.to_sinter_task(json_metadata={"kind": "dynamic_lowered"})

    assert isinstance(task, sinter.Task)
    assert getattr(task, "circuit", None) is not None
    assert getattr(task, "detector_error_model", None) is not None
    assert task.json_metadata["kind"] == "dynamic_lowered"


def test_sinter_collect_smoke_with_pymatching():
    pytest.importorskip("pymatching")

    c = stimdx.Circuit.from_stim(
        stim.Circuit(
            """
            X_ERROR(0.01) 0
            M 0
            DETECTOR rec[-1]
            """
        )
    )
    task = c.to_sinter_task(json_metadata={"name": "smoke"})

    stats = sinter.collect(
        num_workers=1,
        tasks=[task],
        decoders=["pymatching"],
        max_shots=100,
        max_errors=5,
        print_progress=False,
    )

    assert len(stats) == 1
    assert stats[0].json_metadata["name"] == "smoke"
