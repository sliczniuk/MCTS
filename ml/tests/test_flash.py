from __future__ import annotations

import math

import pytest

from ml import StreamState, build_pr_flasher, flash_split


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


def test_build_pr_flasher_creates_provider():
    provider = build_pr_flasher(COMPOUNDS)

    assert provider.compounds == tuple(COMPOUNDS)
    assert provider.flasher is not None


def test_two_phase_flash_returns_child_streams_and_molar_balance():
    provider = build_pr_flasher(COMPOUNDS)

    result = flash_split(_feed(), provider)

    assert result.success, result.error_message
    assert result.phase_state == "two_phase"
    assert result.vapor_stream is not None
    assert result.liquid_stream is not None
    assert result.vapor_fraction is not None
    assert result.liquid_fraction is not None
    assert 0.0 < result.vapor_fraction < 1.0

    vapor = result.vapor_stream
    liquid = result.liquid_stream
    assert math.isclose(
        vapor.molar_flow_mols + liquid.molar_flow_mols,
        1.0,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(sum(vapor.composition.values()), 1.0, rel_tol=1e-10)
    assert math.isclose(sum(liquid.composition.values()), 1.0, rel_tol=1e-10)

    for compound, z in _feed().composition.items():
        recovered = (
            vapor.molar_flow_mols * vapor.composition[compound]
            + liquid.molar_flow_mols * liquid.composition[compound]
        )
        assert math.isclose(recovered, z, rel_tol=1e-7, abs_tol=1e-9)


def test_single_phase_flash_returns_no_split():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(temperature_K=300.0, pressure_Pa=100000.0)

    result = flash_split(stream, provider)

    assert result.success, result.error_message
    assert result.phase_state in {"vapor", "liquid", "unknown"}
    assert result.phase_state == "vapor"
    assert result.vapor_stream is None
    assert result.liquid_stream is None
    assert result.vapor_fraction == pytest.approx(1.0)
    assert result.liquid_fraction == pytest.approx(0.0)


def test_invalid_compound_names_raise_clear_error():
    with pytest.raises(ValueError, match="Could not create thermo constants"):
        build_pr_flasher(["not-a-real-thermo-compound"])


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

    result = flash_split(stream, provider)

    assert not result.success
    assert message in result.error_message
    assert result.phase_state == "unknown"


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(composition={"methane": 0.9, "argon": 0.1})

    result = flash_split(stream, provider)

    assert not result.success
    assert "components not in the flasher" in result.error_message


def test_two_phase_flash_reports_positive_duty_W():
    provider = build_pr_flasher(COMPOUNDS)

    result = flash_split(_feed(), provider)

    assert result.phase_state == "two_phase"
    assert result.duty_W is not None
    assert result.duty_W > 0.0


def test_composition_is_normalized_to_provider_order():
    provider = build_pr_flasher(COMPOUNDS)
    stream = _feed(
        composition={
            "nitrogen": 17.0,
            "methane": 965.0,
            "ethane": 18.0,
        }
    )

    result = flash_split(stream, provider)

    assert result.success, result.error_message
    assert result.phase_state == "two_phase"
    assert result.vapor_stream is not None
    assert list(result.vapor_stream.composition) == COMPOUNDS
