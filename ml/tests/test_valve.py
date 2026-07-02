from __future__ import annotations

import pytest

from ml import StreamState, build_pr_flasher, valve_stream


COMPOUNDS = ["methane", "ethane", "nitrogen"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 200000.0,
        "molar_flow_mols": 1.0,
        "composition": {"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
    }
    values.update(overrides)
    return StreamState(**values)


def test_valve_stream_with_pressure_ratio_reduces_pressure_isenthalpically():
    provider = build_pr_flasher(COMPOUNDS)

    result = valve_stream(_feed(), provider, pressure_ratio=0.5)

    assert result.success
    assert result.outlet_stream is not None
    assert result.outlet_stream.pressure_Pa == pytest.approx(100000.0)
    assert result.delta_P_Pa == pytest.approx(100000.0)
    assert result.pressure_ratio == pytest.approx(0.5)
    assert result.outlet_stream.molar_flow_mols == pytest.approx(_feed().molar_flow_mols)
    assert result.outlet_stream.composition == pytest.approx(_feed().composition)
    assert result.outlet_stream.history[-1] == "valve"
    assert result.outlet_enthalpy_Jmol == pytest.approx(result.inlet_enthalpy_Jmol)


def test_valve_stream_with_outlet_pressure():
    provider = build_pr_flasher(COMPOUNDS)

    result = valve_stream(_feed(), provider, outlet_pressure_Pa=150000.0)

    assert result.success
    assert result.outlet_pressure_Pa == pytest.approx(150000.0)
    assert result.delta_P_Pa == pytest.approx(50000.0)
    assert result.pressure_ratio == pytest.approx(0.75)


def test_valve_stream_with_delta_pressure():
    provider = build_pr_flasher(COMPOUNDS)

    result = valve_stream(_feed(), provider, delta_P_Pa=50000.0)

    assert result.success
    assert result.outlet_pressure_Pa == pytest.approx(150000.0)
    assert result.delta_P_Pa == pytest.approx(50000.0)
    assert result.pressure_ratio == pytest.approx(0.75)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"outlet_pressure_Pa": 100000.0, "pressure_ratio": 0.5}, "exactly one"),
        ({"pressure_ratio": 0.5, "delta_P_Pa": 100000.0}, "exactly one"),
        ({"pressure_ratio": 1.0}, "0 < ratio < 1"),
        ({"pressure_ratio": 0.0}, "0 < ratio < 1"),
        ({"outlet_pressure_Pa": -1.0}, "positive"),
        ({"delta_P_Pa": 0.0}, "positive pressure drop"),
        ({"delta_P_Pa": 250000.0}, "outlet pressure must be positive"),
        ({"outlet_pressure_Pa": 250000.0}, "lower than inlet"),
    ],
)
def test_invalid_pressure_spec_returns_failed_result(kwargs, message):
    provider = build_pr_flasher(COMPOUNDS)

    result = valve_stream(_feed(), provider, **kwargs)

    assert not result.success
    assert message in result.error_message


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(composition={"methane": 0.9, "argon": 0.1})

    result = valve_stream(stream, provider, pressure_ratio=0.5)

    assert not result.success
    assert "components not in the flasher" in result.error_message
