from __future__ import annotations

import math

import pytest

from ml import (
    StreamState,
    build_pr_flasher,
    estimate_relative_volatilities,
    shortcut_distillation_fug,
)


COMPOUNDS = ["nitrogen", "propane", "n-butane"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 500000.0,
        "molar_flow_mols": 2.0,
        "composition": {"nitrogen": 0.1, "propane": 0.45, "n-butane": 0.45},
    }
    values.update(overrides)
    return StreamState(**values)


def test_estimate_relative_volatilities_uses_thermo_flash():
    provider = build_pr_flasher(COMPOUNDS)

    alphas, k_values, warnings = estimate_relative_volatilities(
        _feed(),
        provider,
        heavy_key="n-butane",
        vapor_fraction=0.5,
    )

    assert warnings == []
    assert set(alphas) == set(COMPOUNDS)
    assert set(k_values) == set(COMPOUNDS)
    assert alphas["n-butane"] == pytest.approx(1.0)
    assert alphas["propane"] > 1.0
    assert alphas["nitrogen"] > alphas["propane"]
    assert all(value > 0.0 for value in k_values.values())


def test_shortcut_distillation_fug_with_explicit_alphas_closes_balance():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()

    result = shortcut_distillation_fug(
        feed,
        provider,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        relative_volatilities={"nitrogen": 100.0, "propane": 4.0, "n-butane": 1.0},
        column_id="C1",
    )

    assert result.success, result.error_message
    assert result.distillate_stream is not None
    assert result.bottoms_stream is not None
    assert result.distillate_stream.id == "C1_distillate"
    assert result.bottoms_stream.id == "C1_bottoms"
    assert result.pressure_Pa == pytest.approx(feed.pressure_Pa)
    assert result.distillate_stream.pressure_Pa == pytest.approx(feed.pressure_Pa)
    assert result.component_recoveries["propane"] == pytest.approx(0.95)
    assert result.component_recoveries["n-butane"] == pytest.approx(0.05)
    assert result.minimum_stages > 0.0
    assert result.minimum_reflux_ratio > 0.0
    assert result.reflux_ratio > result.minimum_reflux_ratio
    assert result.theoretical_stages > result.minimum_stages
    assert result.underwood_theta > result.relative_volatilities["n-butane"]
    assert result.underwood_theta < result.relative_volatilities["propane"]
    assert result.distillate_stream.history[-1] == (
        "shortcut_distillation:total_condenser_distillate"
    )
    assert math.isclose(
        result.distillate_stream.molar_flow_mols + result.bottoms_stream.molar_flow_mols,
        feed.molar_flow_mols,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(sum(result.distillate_stream.composition.values()), 1.0)
    assert math.isclose(sum(result.bottoms_stream.composition.values()), 1.0)

    for compound in COMPOUNDS:
        recovered = (
            result.distillate_stream.molar_flow_mols
            * result.distillate_stream.composition[compound]
            + result.bottoms_stream.molar_flow_mols
            * result.bottoms_stream.composition[compound]
        )
        expected = feed.molar_flow_mols * feed.composition[compound]
        assert recovered == pytest.approx(expected, rel=1e-10, abs=1e-12)

    feed_zs = [feed.composition[compound] for compound in provider.compounds]
    feed_flash = provider.flasher.flash(
        T=feed.temperature_K,
        P=feed.pressure_Pa,
        zs=feed_zs,
    )
    assert result.feed_quality == pytest.approx(1.0 - feed_flash.VF)

    distillate_zs = [
        result.distillate_stream.composition[compound]
        for compound in provider.compounds
    ]
    condenser_flash = provider.flasher.flash(
        P=feed.pressure_Pa,
        VF=0.0,
        zs=distillate_zs,
    )
    assert result.distillate_stream.temperature_K == pytest.approx(condenser_flash.T)


def test_shortcut_distillation_fug_can_estimate_alphas_from_thermo():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
    )

    assert result.success, result.error_message
    assert result.k_values
    assert result.relative_volatilities["n-butane"] == pytest.approx(1.0)
    assert result.relative_volatilities["propane"] > 1.0
    assert result.component_recoveries["propane"] == pytest.approx(0.95)
    assert result.component_recoveries["n-butane"] == pytest.approx(0.05)


def test_invalid_key_selection_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="argon",
    )

    assert not result.success
    assert "heavy_key 'argon' is not in provider compounds" in result.error_message


def test_same_light_and_heavy_key_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="propane",
    )

    assert not result.success
    assert "light_key and heavy_key must be different" in result.error_message


def test_invalid_key_recoveries_return_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.05,
        heavy_key_recovery=0.95,
    )

    assert not result.success
    assert "light_key_recovery must be greater" in result.error_message


def test_invalid_relative_volatility_order_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="n-butane",
        relative_volatilities={"nitrogen": 100.0, "propane": 0.8, "n-butane": 1.0},
    )

    assert not result.success
    assert "must be more volatile" in result.error_message


def test_unknown_stream_component_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed(composition={"nitrogen": 0.1, "propane": 0.45, "argon": 0.45})

    result = shortcut_distillation_fug(
        feed,
        provider,
        light_key="propane",
        heavy_key="n-butane",
    )

    assert not result.success
    assert "components not in the flasher" in result.error_message


def test_pressure_override_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="n-butane",
        pressure_Pa=100000.0,
    )

    assert not result.success
    assert "column pressure is taken from stream.pressure_Pa" in result.error_message


def test_feed_quality_override_returns_failed_result():
    provider = build_pr_flasher(COMPOUNDS)

    result = shortcut_distillation_fug(
        _feed(),
        provider,
        light_key="propane",
        heavy_key="n-butane",
        feed_quality=1.0,
    )

    assert not result.success
    assert "feed_quality is not an MCTS decision variable" in result.error_message
