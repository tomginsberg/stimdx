"""Tests for stimdx circuit execution and protobuf serialization."""

import pytest
from stimdx import Circuit, LastMeas, MeasParity


class TestCircuitBasics:
    """Test basic circuit construction."""

    def test_empty_circuit(self):
        c = Circuit()
        assert len(c.nodes) == 0

    def test_single_block(self):
        c = Circuit("H 0")
        assert len(c.nodes) == 1

    def test_chained_blocks(self):
        c = Circuit()
        c.block("H 0").block("M 0")
        assert len(c.nodes) == 2


class TestCondition:
    """Test condition construction and serialization."""

    def test_last_meas_condition(self):
        cond = LastMeas(0)
        assert cond.index == 0

    def test_meas_parity_condition(self):
        cond = MeasParity([0, 1, 2])
        assert cond.indices == [0, 1, 2]


class TestProtoSerialization:
    """Test protobuf serialization of circuits."""

    def test_simple_circuit_serializes(self):
        c = Circuit("H 0\nM 0")
        proto_bytes = c.to_proto()
        assert isinstance(proto_bytes, bytes)
        assert len(proto_bytes) > 0

    def test_conditional_circuit_serializes(self):
        c = Circuit("H 0\nM 0")
        c.conditional(body="X 0", cond=LastMeas(0))
        proto_bytes = c.to_proto()
        assert isinstance(proto_bytes, bytes)

    def test_while_loop_serializes(self):
        c = Circuit("H 0\nM 0")
        fix = Circuit("X 0\nM 0")
        c.while_loop(body=fix, cond=LastMeas(0))
        proto_bytes = c.to_proto()
        assert isinstance(proto_bytes, bytes)

    def test_do_while_serializes(self):
        c = Circuit()
        body = Circuit("H 0\nM 0")
        c.do_while(body=body, cond=LastMeas(0))
        proto_bytes = c.to_proto()
        assert isinstance(proto_bytes, bytes)


class TestSampling:
    """Test circuit sampling."""

    def test_simple_measurement(self):
        c = Circuit("M 0")  # Measure |0> state
        sampler = c.compile_sampler(seed=42)
        samples = sampler.sample(shots=10, use_cpp=False)
        assert len(samples) == 10
        # |0> always measures False
        for sample in samples:
            assert sample == [False]

    def test_hadamard_randomness(self):
        c = Circuit("H 0\nM 0")
        sampler = c.compile_sampler(seed=42)
        samples = sampler.sample(shots=100, use_cpp=False)
        # Should have roughly 50/50 True/False
        true_count = sum(1 for s in samples if s[0])
        assert 20 < true_count < 80

    def test_conditional_execution(self):
        c = Circuit("H 0\nM 0")
        c.conditional(body="X 1", cond=LastMeas(0))
        c.block("M 1")
        sampler = c.compile_sampler(seed=42)
        samples = sampler.sample(shots=100, use_cpp=False)

        # When first measurement is True, X is applied, so second should be True
        # When first measurement is False, X is not applied, so second should be False
        for sample in samples:
            assert sample[0] == sample[1]

    def test_repeat_until_success(self):
        # Create a repeat-until-success pattern
        c = Circuit()
        body = Circuit("H 0\nM 0")
        c.do_while(body=body, cond=LastMeas(0), max_iter=100)
        sampler = c.compile_sampler(seed=42)
        samples = sampler.sample(shots=10, use_cpp=False)

        # Each shot should end with a False measurement (success condition)
        for sample in samples:
            assert sample[-1] == False

    def test_deterministic_seeding(self):
        c = Circuit("H 0\nM 0")
        sampler1 = c.compile_sampler(seed=12345)
        sampler2 = c.compile_sampler(seed=12345)

        samples1 = sampler1.sample(shots=50, use_cpp=False)
        samples2 = sampler2.sample(shots=50, use_cpp=False)

        assert samples1 == samples2


class TestCppBackend:
    """Test C++ backend if available."""

    def test_cpp_availability(self):
        import stimdx

        # Just check we can access the flag
        assert isinstance(stimdx.__cpp_available__, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("stimdx").__cpp_available__,
        reason="C++ extension not available",
    )
    def test_cpp_python_equivalence(self):
        """Test that C++ backend produces statistically correct results.

        Note: We can't test exact equivalence because Python and C++ use
        different RNG implementations. Instead we test behavioral correctness.
        """
        # Test 1: Conditional execution - both backends should have M0 == M1
        c = Circuit("H 0\nM 0")
        c.conditional(body="X 1", cond=LastMeas(0))
        c.block("M 1")

        sampler = c.compile_sampler()
        cpp_samples = sampler.sample(shots=100, use_cpp=True)

        # Verify conditional logic: when first is True, X applied, so second is True
        for sample in cpp_samples:
            assert sample[0] == sample[1], "C++ conditional logic failed"

        # Test 2: RUS should always end with False
        c2 = Circuit()
        body = Circuit("H 0\nM 0")
        c2.do_while(body=body, cond=LastMeas(0), max_iter=100)

        sampler2 = c2.compile_sampler()
        cpp_samples2 = sampler2.sample(shots=50, use_cpp=True)

        for sample in cpp_samples2:
            assert sample[-1] == False, "C++ RUS did not terminate correctly"

    @pytest.mark.skipif(
        not pytest.importorskip("stimdx").__cpp_available__,
        reason="C++ extension not available",
    )
    def test_cpp_deterministic_seeding(self):
        """Test that C++ backend is deterministic with same seed."""
        c = Circuit("H 0\nM 0")

        sampler1 = c.compile_sampler(seed=12345)
        sampler2 = c.compile_sampler(seed=12345)

        samples1 = sampler1.sample(shots=50, use_cpp=True)
        samples2 = sampler2.sample(shots=50, use_cpp=True)

        assert samples1 == samples2, "C++ backend is not deterministic"
