from __future__ import annotations

import math

import pytest

from ml import StreamState, build_pr_flasher, heat_stream


COMPOUNDS = ["methane", "ethane", "nitrogen"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 110.0,
        "pressure_Pa": 100000.0,
        "molar_flow_mols": 2.0,
        "composition": {"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
    }
    values.update(overrides)
    return StreamState(**values)


def test_heat_stream_to_outlet_temperature_computes_positive_duty():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed()

    result = heat_stream(stream, provider, outlet_temperature_K=120.0)

    assert result.success, result.error_message
    assert result.outlet_stream is not None
    assert result.outlet_stream.temperature_K == pytest.approx(120.0)
    assert result.outlet_stream.pressure_Pa == stream.pressure_Pa
    assert result.outlet_stream.molar_flow_mols == stream.molar_flow_mols
    assert result.outlet_stream.composition == pytest.approx(stream.composition)
    assert result.duty_W is not None
    assert result.duty_W > 0.0
    assert result.vapor_fraction is not None
    assert result.phase_state in {"two_phase", "vapor", "liquid", "unknown"}
    assert result.phase_compositions


def test_heat_stream_with_delta_temperature_matches_absolute_temperature():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed()

    by_delta = heat_stream(stream, provider, delta_T_K=10.0)
    by_absolute = heat_stream(stream, provider, outlet_temperature_K=120.0)

    assert by_delta.success, by_delta.error_message
    assert by_absolute.success, by_absolute.error_message
    assert by_delta.outlet_stream.temperature_K == pytest.approx(
        by_absolute.outlet_stream.temperature_K
    )
    assert by_delta.duty_W == pytest.approx(by_absolute.duty_W)
    assert by_delta.vapor_fraction == pytest.approx(by_absolute.vapor_fraction)


def test_cooling_returns_negative_duty():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(temperature_K=120.0)

    result = heat_stream(stream, provider, outlet_temperature_K=110.0)

    assert result.success, result.error_message
    assert result.duty_W is not None
    assert result.duty_W < 0.0


def test_phase_compositions_are_normalized_when_present():
    provider = build_pr_flasher(COMPOUNDS)
    result = heat_stream(_feed(), provider, outlet_temperature_K=120.0)

    assert result.success, result.error_message
    for composition in result.phase_compositions.values():
        assert math.isclose(sum(composition.values()), 1.0, rel_tol=1e-10)
        assert list(composition) == COMPOUNDS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"outlet_temperature_K": 120.0, "delta_T_K": 10.0}, "only one"),
        ({"outlet_temperature_K": 0.0}, "positive"),
    ],
)
def test_invalid_temperature_spec_returns_failed_result(kwargs, message):
    provider = build_pr_flasher(COMPOUNDS)

    result = heat_stream(_feed(), provider, **kwargs)

    assert not result.success
    assert message in result.error_message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature_K", 0.0, "non-positive temperature"),
        ("pressure_Pa", -1.0, "non-positive pressure"),
        ("molar_flow_mols", 0.0, "non-positive molar_flow"),
    ],
)
def test_invalid_stream_conditions_return_failed_result(field, value, message):
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(**{field: value})

    result = heat_stream(stream, provider, outlet_temperature_K=120.0)

    assert not result.success
    assert message in result.error_message


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(composition={"methane": 0.9, "argon": 0.1})

    result = heat_stream(stream, provider, outlet_temperature_K=120.0)

    assert not result.success
    assert "components not in the flasher" in result.error_message
