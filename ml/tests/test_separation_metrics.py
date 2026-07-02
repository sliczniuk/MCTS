from __future__ import annotations

import pytest

from ml import StreamState, mutual_information_separation, separation_indicator


def _feed() -> StreamState:
    return StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=100000.0,
        molar_flow_mols=5.0,
        composition={
            "A": 0.2,
            "B": 0.2,
            "C": 0.2,
            "D": 0.2,
            "E": 0.2,
        },
    )


def test_perfect_five_component_separation_reaches_target():
    feed = _feed()
    products = [
        StreamState(
            id=f"{component}_product",
            temperature_K=300.0,
            pressure_Pa=100000.0,
            molar_flow_mols=1.0,
            composition={component: 1.0},
        )
        for component in feed.composition
    ]

    metric = separation_indicator(feed, products)

    assert metric["score"] == pytest.approx(5.0)
    assert metric["target"] == 5
    assert metric["fraction_of_target"] == pytest.approx(1.0)
    assert all(score == pytest.approx(1.0) for score in metric["component_scores"].values())
    assert all(purity == pytest.approx(1.0) for purity in metric["purities"].values())
    assert all(recovery == pytest.approx(1.0) for recovery in metric["recoveries"].values())


def test_mixed_streams_score_below_target():
    feed = _feed()
    outlets = [
        StreamState(
            id="AB",
            temperature_K=300.0,
            pressure_Pa=100000.0,
            molar_flow_mols=2.0,
            composition={"A": 0.5, "B": 0.5},
        ),
        StreamState(
            id="CDE",
            temperature_K=300.0,
            pressure_Pa=100000.0,
            molar_flow_mols=3.0,
            composition={"C": 1 / 3, "D": 1 / 3, "E": 1 / 3},
        ),
    ]

    metric = separation_indicator(feed, outlets)

    assert metric["target"] == 5
    assert metric["score"] < 5.0
    assert metric["score"] == pytest.approx(2.0)
    assert metric["component_scores"]["A"] == pytest.approx(0.5)
    assert metric["component_scores"]["C"] == pytest.approx(1 / 3)


def test_lost_material_lowers_recovery_and_score():
    feed = _feed()
    outlets = [
        StreamState(
            id="A_half",
            temperature_K=300.0,
            pressure_Pa=100000.0,
            molar_flow_mols=0.5,
            composition={"A": 1.0},
        )
    ]

    metric = separation_indicator(feed, outlets, components=["A"])

    assert metric["target"] == 1
    assert metric["purities"]["A"] == pytest.approx(1.0)
    assert metric["recoveries"]["A"] == pytest.approx(0.5)
    assert metric["component_scores"]["A"] == pytest.approx(0.5)
    assert metric["score"] == pytest.approx(0.5)


def test_high_purity_low_recovery_scores_below_one():
    feed = _feed()
    outlets = [
        StreamState(
            id="A_pure_trace",
            temperature_K=300.0,
            pressure_Pa=100000.0,
            molar_flow_mols=0.1,
            composition={"A": 1.0},
        )
    ]

    metric = separation_indicator(feed, outlets, components=["A"])

    assert metric["purities"]["A"] == pytest.approx(1.0)
    assert metric["recoveries"]["A"] == pytest.approx(0.1)
    assert metric["score"] == pytest.approx(0.1)


def test_components_are_inferred_from_meaningful_feed_fractions():
    feed = StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=100000.0,
        molar_flow_mols=1.0,
        composition={"A": 0.999999, "trace": 1e-9},
    )
    product = StreamState(
        id="A_product",
        temperature_K=300.0,
        pressure_Pa=100000.0,
        molar_flow_mols=0.999999,
        composition={"A": 1.0},
    )

    metric = separation_indicator(feed, [product], min_component_fraction=1e-8)

    assert metric["target"] == 1
    assert set(metric["component_scores"]) == {"A"}


def test_invalid_basis_raises_value_error():
    with pytest.raises(ValueError, match="basis='molar'"):
        separation_indicator(_feed(), [], basis="mass")  # type: ignore[arg-type]


def _two_comp_feed() -> StreamState:
    return StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=100_000.0,
        molar_flow_mols=1.0,
        composition={"A": 0.5, "B": 0.5},
    )


def _recycle_streams() -> tuple[StreamState, StreamState]:
    """A-enriched product (0.7 mol/s) + recycle-mixed feed (1.3 mol/s).

    Total outlet flow = 2.0 mol/s > F_0 = 1.0 mol/s, simulating the one-pass
    recycle approximation where F_mix = F_feed + F_recycle.
    """
    a_rich = StreamState(
        id="a_rich",
        temperature_K=300.0,
        pressure_Pa=100_000.0,
        molar_flow_mols=0.7,
        composition={"A": 0.7, "B": 0.3},
    )
    recycle_mix = StreamState(
        id="recycle_mix",
        temperature_K=300.0,
        pressure_Pa=100_000.0,
        molar_flow_mols=1.3,
        composition={"A": 0.408, "B": 0.592},
    )
    return a_rich, recycle_mix


def test_mi_feed_fraction_recycle_excess_mi_positive():
    """feed_fraction: recycle excess (F_total > F_0) must not clamp MI to 0."""
    feed = _two_comp_feed()
    a_rich, recycle_mix = _recycle_streams()
    m = mutual_information_separation(feed, [a_rich, recycle_mix], weight_mode="feed_fraction")
    assert m["mi_nats"] > 0.0, "MI was clamped to 0 despite genuine A-enrichment"
    assert m["score"] > 0.0


def test_mi_equal_weight_recycle_excess_mi_positive():
    """equal_weight: recycle excess (F_total > F_0) must not suppress MI."""
    feed = _two_comp_feed()
    a_rich, recycle_mix = _recycle_streams()
    m = mutual_information_separation(feed, [a_rich, recycle_mix], weight_mode="equal_weight")
    assert m["mi_nats"] > 0.0, "MI was zero despite genuine A-enrichment"
    assert m["score"] > 0.0
