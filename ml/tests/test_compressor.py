from __future__ import annotations

import pytest

from ml import StreamState, build_pr_flasher, compress_stream


COMPOUNDS = ["methane", "ethane", "nitrogen"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 100000.0,
        "molar_flow_mols": 1.0,
        "composition": {"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
    }
    values.update(overrides)
    return StreamState(**values)


def test_compress_stream_with_pressure_ratio_increases_pressure_temperature_and_power():
    provider = build_pr_flasher(COMPOUNDS)

    result = compress_stream(
        _feed(),
        provider,
        pressure_ratio=2.0,
        isentropic_efficiency=0.75,
    )

    assert result.success
    assert result.outlet_stream is not None
    assert result.outlet_stream.pressure_Pa == pytest.approx(200000.0)
    assert result.outlet_stream.temperature_K > _feed().temperature_K
    assert result.outlet_stream.molar_flow_mols == pytest.approx(_feed().molar_flow_mols)
    assert result.outlet_stream.composition == pytest.approx(_feed().composition)
    assert result.outlet_stream.history[-1] == "compressor"
    assert result.fluid_power_W > 0.0
    assert result.shaft_power_W == pytest.approx(result.fluid_power_W)
    assert result.ideal_outlet_enthalpy_Jmol > result.inlet_enthalpy_Jmol
    assert result.actual_outlet_enthalpy_Jmol > result.ideal_outlet_enthalpy_Jmol


def test_compress_stream_with_outlet_pressure_and_mechanical_efficiency():
    provider = build_pr_flasher(COMPOUNDS)

    result = compress_stream(
        _feed(),
        provider,
        outlet_pressure_Pa=300000.0,
        isentropic_efficiency=0.8,
        mechanical_efficiency=0.9,
        outlet_stream_id="Compressed",
    )

    assert result.success
    assert result.outlet_stream.id == "Compressed"
    assert result.outlet_pressure_Pa == pytest.approx(300000.0)
    assert result.pressure_ratio == pytest.approx(3.0)
    assert result.shaft_power_W == pytest.approx(result.fluid_power_W / 0.9)


def test_compress_stream_with_delta_pressure():
    provider = build_pr_flasher(COMPOUNDS)

    result = compress_stream(
        _feed(),
        provider,
        delta_P_Pa=100000.0,
        isentropic_efficiency=0.75,
    )

    assert result.success
    assert result.outlet_stream.pressure_Pa == pytest.approx(200000.0)
    assert result.delta_P_Pa == pytest.approx(100000.0)
    assert result.pressure_ratio == pytest.approx(2.0)
    assert result.shaft_power_W > 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"outlet_pressure_Pa": 200000.0, "pressure_ratio": 2.0}, "exactly one"),
        ({"pressure_ratio": 2.0, "delta_P_Pa": 100000.0}, "exactly one"),
        ({"pressure_ratio": 1.0}, "greater than 1"),
        ({"outlet_pressure_Pa": -1.0}, "positive"),
        ({"delta_P_Pa": 0.0}, "delta_P_Pa"),
    ],
)
def test_invalid_pressure_spec_returns_failed_result(kwargs, message):
    provider = build_pr_flasher(COMPOUNDS)

    result = compress_stream(_feed(), provider, **kwargs)

    assert not result.success
    assert message in result.error_message


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"isentropic_efficiency": 0.0}, "isentropic_efficiency"),
        ({"isentropic_efficiency": 1.1}, "isentropic_efficiency"),
        ({"mechanical_efficiency": 0.0}, "mechanical_efficiency"),
        ({"mechanical_efficiency": 1.1}, "mechanical_efficiency"),
    ],
)
def test_invalid_efficiency_returns_failed_result(kwargs, message):
    provider = build_pr_flasher(COMPOUNDS)

    result = compress_stream(_feed(), provider, pressure_ratio=2.0, **kwargs)

    assert not result.success
    assert message in result.error_message


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(composition={"methane": 0.9, "argon": 0.1})

    result = compress_stream(stream, provider, pressure_ratio=2.0)

    assert not result.success
    assert "components not in the flasher" in result.error_message
