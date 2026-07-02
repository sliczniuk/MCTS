from __future__ import annotations

import pytest

from ml import StreamState, build_pr_flasher, flash_split, pump_stream


COMPOUNDS = ["methane", "ethane", "nitrogen"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 110.0,
        "pressure_Pa": 100000.0,
        "molar_flow_mols": 1.0,
        "composition": {"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
    }
    values.update(overrides)
    return StreamState(**values)


def _liquid_stream(provider) -> StreamState:
    result = flash_split(_feed(), provider)
    assert result.success
    assert result.liquid_stream is not None
    return result.liquid_stream


def test_pump_stream_with_pressure_ratio_increases_pressure_and_power():
    provider = build_pr_flasher(COMPOUNDS)
    liquid = _liquid_stream(provider)

    result = pump_stream(
        liquid,
        provider,
        pressure_ratio=2.0,
        isentropic_efficiency=0.75,
    )

    assert result.success
    assert result.outlet_stream is not None
    assert result.inlet_vapor_fraction <= 1e-6
    assert result.outlet_stream.pressure_Pa == pytest.approx(200000.0)
    assert result.outlet_stream.temperature_K > liquid.temperature_K
    assert result.outlet_stream.molar_flow_mols == pytest.approx(liquid.molar_flow_mols)
    assert result.outlet_stream.composition == pytest.approx(liquid.composition)
    assert result.outlet_stream.history[-1] == "pump"
    assert result.fluid_power_W > 0.0
    assert result.shaft_power_W == pytest.approx(result.fluid_power_W)
    assert result.ideal_outlet_enthalpy_Jmol > result.inlet_enthalpy_Jmol
    assert result.actual_outlet_enthalpy_Jmol > result.ideal_outlet_enthalpy_Jmol


def test_pump_stream_with_outlet_pressure_and_mechanical_efficiency():
    provider = build_pr_flasher(COMPOUNDS)
    liquid = _liquid_stream(provider)

    result = pump_stream(
        liquid,
        provider,
        outlet_pressure_Pa=300000.0,
        isentropic_efficiency=0.8,
        mechanical_efficiency=0.9,
        outlet_stream_id="Pumped",
    )

    assert result.success
    assert result.outlet_stream.id == "Pumped"
    assert result.outlet_pressure_Pa == pytest.approx(300000.0)
    assert result.pressure_ratio == pytest.approx(3.0)
    assert result.shaft_power_W == pytest.approx(result.fluid_power_W / 0.9)


def test_pump_stream_with_delta_pressure():
    provider = build_pr_flasher(COMPOUNDS)
    liquid = _liquid_stream(provider)

    result = pump_stream(
        liquid,
        provider,
        delta_P_Pa=100000.0,
        isentropic_efficiency=0.75,
    )

    assert result.success
    assert result.outlet_stream.pressure_Pa == pytest.approx(200000.0)
    assert result.delta_P_Pa == pytest.approx(100000.0)
    assert result.pressure_ratio == pytest.approx(2.0)
    assert result.shaft_power_W > 0.0


def test_pump_rejects_vapor_inlet():
    provider = build_pr_flasher(COMPOUNDS)
    vapor = _feed(temperature_K=300.0)

    result = pump_stream(vapor, provider, pressure_ratio=2.0)

    assert not result.success
    assert "requires a liquid or near-liquid inlet" in result.error_message
    assert "Use a compressor" in result.error_message


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
    liquid = _liquid_stream(provider)

    result = pump_stream(liquid, provider, **kwargs)

    assert not result.success
    assert message in result.error_message


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"isentropic_efficiency": 0.0}, "isentropic_efficiency"),
        ({"isentropic_efficiency": 1.1}, "isentropic_efficiency"),
        ({"mechanical_efficiency": 0.0}, "mechanical_efficiency"),
        ({"mechanical_efficiency": 1.1}, "mechanical_efficiency"),
        ({"max_inlet_vapor_fraction": -0.1}, "max_inlet_vapor_fraction"),
        ({"max_inlet_vapor_fraction": 1.1}, "max_inlet_vapor_fraction"),
    ],
)
def test_invalid_efficiency_or_vapor_limit_returns_failed_result(kwargs, message):
    provider = build_pr_flasher(COMPOUNDS)
    liquid = _liquid_stream(provider)

    result = pump_stream(liquid, provider, pressure_ratio=2.0, **kwargs)

    assert not result.success
    assert message in result.error_message


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(composition={"methane": 0.9, "argon": 0.1})

    result = pump_stream(stream, provider, pressure_ratio=2.0)

    assert not result.success
    assert "components not in the flasher" in result.error_message
