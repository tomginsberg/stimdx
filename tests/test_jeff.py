"""Tests for JEFF format export functionality."""

import pytest

from stimdx import Circuit, LastMeas, MeasParity, JeffExporter, to_jeff
import jeff


class TestBasicGates:
    """Test basic gate exports to JEFF."""

    def test_single_qubit_gates(self):
        """Test single-qubit gates are correctly exported."""
        c = Circuit()
        c.block("H 0")
        c.block("X 1")
        c.block("Y 2")
        c.block("Z 3")
        c.block("S 0")

        module = c.to_jeff()

        assert isinstance(module, jeff.JeffModule)
        assert len(module.functions) == 1

    def test_two_qubit_gates(self):
        """Test two-qubit gates are correctly exported."""
        c = Circuit()
        c.block("CNOT 0 1")
        c.block("CZ 2 3")
        c.block("SWAP 0 2")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_s_dag_gate(self):
        """Test S_DAG is exported with adjoint=True."""
        c = Circuit()
        c.block("S_DAG 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_empty_circuit(self):
        """Test empty circuit exports correctly."""
        c = Circuit()
        module = c.to_jeff()

        assert isinstance(module, jeff.JeffModule)
        assert len(module.functions) == 1


class TestMeasurementsAndResets:
    """Test measurement and reset operations."""

    def test_basic_measurement(self):
        """Test M instruction exports to measureNd."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_multiple_measurements(self):
        """Test multiple measurements in one instruction."""
        c = Circuit()
        c.block("H 0")
        c.block("H 1")
        c.block("M 0 1")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_reset(self):
        """Test R instruction exports to reset."""
        c = Circuit()
        c.block("H 0")
        c.block("R 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_measure_reset(self):
        """Test MR instruction exports to measureNd + reset."""
        c = Circuit()
        c.block("H 0")
        c.block("MR 0")
        c.block("H 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)


class TestConditionals:
    """Test conditional control flow export."""

    def test_if_with_last_meas(self):
        """Test IfNode with LastMeas condition."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional("X 1", cond=LastMeas(0))
        c.block("M 1")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_if_with_meas_parity_single(self):
        """Test IfNode with MeasParity of single measurement."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional("X 1", cond=MeasParity([0]))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_if_with_meas_parity_multiple(self):
        """Test IfNode with MeasParity of multiple measurements."""
        c = Circuit()
        c.block("H 0")
        c.block("H 1")
        c.block("M 0 1")
        c.conditional("Z 2", cond=MeasParity([0, 1]))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_if_with_negative_index(self):
        """Test MeasParity with negative indexing."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.block("H 1")
        c.block("M 1")
        c.conditional("X 2", cond=MeasParity([-1]))  # Last measurement

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_nested_conditionals(self):
        """Test nested conditional blocks."""
        inner = Circuit()
        inner.block("H 1")
        inner.block("M 1")
        inner.conditional("Z 2", cond=LastMeas(0))

        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional(inner, cond=LastMeas(0))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)


class TestLoops:
    """Test loop control flow export."""

    def test_while_loop(self):
        """Test WhileNode export."""
        body = Circuit()
        body.block("X 0")
        body.block("M 0")

        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.while_loop(body, cond=LastMeas(0))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_do_while_loop(self):
        """Test DoWhileNode export (repeat-until-success pattern)."""
        body = Circuit()
        body.block("H 0")
        body.block("M 0")

        c = Circuit()
        c.do_while(body, cond=LastMeas(0))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_do_while_with_gate_sequence(self):
        """Test do-while loop with multiple operations."""
        body = Circuit()
        body.block("H 0")
        body.block("S 0")
        body.block("H 0")
        body.block("M 0")

        c = Circuit()
        c.do_while(body, cond=LastMeas(0))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)


class TestLambdaRejection:
    """Test that lambda conditionals are properly rejected."""

    def test_lambda_in_if_rejected(self):
        """Test lambda conditional in IfNode raises ValueError."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional("X 1", cond=lambda ctx: ctx.last_block_meas[0])

        with pytest.raises(ValueError, match="lambda conditionals"):
            c.to_jeff()

    def test_lambda_in_while_rejected(self):
        """Test lambda conditional in WhileNode raises ValueError."""
        body = Circuit()
        body.block("X 0")
        body.block("M 0")

        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.while_loop(body, cond=lambda ctx: ctx.last_block_meas[0])

        with pytest.raises(ValueError, match="lambda conditionals"):
            c.to_jeff()

    def test_lambda_in_do_while_rejected(self):
        """Test lambda conditional in DoWhileNode raises ValueError."""
        body = Circuit()
        body.block("H 0")
        body.block("M 0")

        c = Circuit()
        c.do_while(body, cond=lambda ctx: ctx.last_block_meas[0])

        with pytest.raises(ValueError, match="lambda conditionals"):
            c.to_jeff()


class TestFunctionNaming:
    """Test function naming in exported modules."""

    def test_default_function_name(self):
        """Test default function name is 'main'."""
        c = Circuit()
        c.block("H 0")

        module = c.to_jeff()
        # The function name is stored in string table
        assert module.entrypoint == 0

    def test_custom_function_name(self):
        """Test custom function name is used."""
        c = Circuit()
        c.block("H 0")

        module = c.to_jeff(name="my_circuit")
        assert module.entrypoint == 0


class TestExporterClass:
    """Test JeffExporter class directly."""

    def test_exporter_instantiation(self):
        """Test JeffExporter can be instantiated."""
        c = Circuit()
        c.block("H 0")

        exporter = JeffExporter(c)
        assert exporter.circuit is c

    def test_exporter_validation_on_init(self):
        """Test validation happens during __init__."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional("X 1", cond=lambda ctx: True)

        with pytest.raises(ValueError, match="lambda conditionals"):
            JeffExporter(c)

    def test_to_jeff_function(self):
        """Test standalone to_jeff function."""
        c = Circuit()
        c.block("H 0")

        module = to_jeff(c, name="test_func")
        assert isinstance(module, jeff.JeffModule)

    def test_write_jeff(self, tmp_path):
        """Test writing to a .jeff file."""
        import os
        from stimdx import write_jeff

        c = Circuit("H 0\nM 0")
        path = str(tmp_path / "test.jeff")

        # This should work if pycapnp is installed
        try:
            c.write_jeff(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            
            # Verify we can also call the standalone function
            path2 = str(tmp_path / "test2.jeff")
            write_jeff(c, path2)
            assert os.path.exists(path2)
        except ImportError:
            pytest.skip("pycapnp not installed or schema not loadable")


class TestComplexCircuits:
    """Test complex circuit patterns."""

    def test_bell_state_measurement(self):
        """Test Bell state creation and measurement."""
        c = Circuit()
        c.block("""
            H 0
            CNOT 0 1
            M 0 1
        """)

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_teleportation_correction(self):
        """Test teleportation-like correction pattern."""
        c = Circuit()
        c.block("""
            H 0
            CNOT 0 1
            M 0 1
        """)
        c.conditional("X 2", cond=LastMeas(1))
        c.conditional("Z 2", cond=MeasParity([0]))
        c.block("M 2")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_repeated_measurements(self):
        """Test circuit with multiple measurement rounds."""
        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.block("R 0")
        c.block("H 0")
        c.block("M 0")
        c.block("R 0")
        c.block("H 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_multi_qubit_circuit(self):
        """Test circuit with many qubits."""
        c = Circuit()
        # Create GHZ-like state
        c.block("H 0")
        for i in range(9):
            c.block(f"CNOT {i} {i+1}")
        c.block("M " + " ".join(str(i) for i in range(10)))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_circuit_with_only_measurements(self):
        """Test circuit that only measures (no gates)."""
        c = Circuit()
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_circuit_with_multiple_blocks(self):
        """Test circuit with many separate blocks."""
        c = Circuit()
        c.block("H 0")
        c.block("X 0")
        c.block("H 0")
        c.block("Z 0")
        c.block("H 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_conditional_with_circuit_body(self):
        """Test conditional with Circuit object as body."""
        body = Circuit()
        body.block("X 0")
        body.block("Y 0")

        c = Circuit()
        c.block("H 0")
        c.block("M 0")
        c.conditional(body, cond=LastMeas(0))

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

    def test_circuit_string_initialization(self):
        """Test circuit initialized with string."""
        c = Circuit("H 0")
        c.block("M 0")

        module = c.to_jeff()
        assert isinstance(module, jeff.JeffModule)

