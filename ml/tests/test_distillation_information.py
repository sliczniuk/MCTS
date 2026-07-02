from __future__ import annotations

import pytest

from ml import (
    StreamState,
    distillation_boundary_information,
    distillation_feed_priority,
    distillation_score_curve,
    rank_distillation_boundaries,
    stream_priority,
)


COMPONENTS = ["ethane", "propane", "n-butane", "n-pentane", "n-hexane"]
ALPHAS = {
    "ethane": 16.0,
    "propane": 8.0,
    "n-butane": 4.0,
    "n-pentane": 2.0,
    "n-hexane": 1.0,
}


def _stream(stream_id: str, flow: float, composition: dict[str, float]) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=350.0,
        pressure_Pa=600000.0,
        molar_flow_mols=flow,
        composition=composition,
    )


def test_distillation_feed_priority_matches_flow_weighted_entropy():
    feed = _stream("Feed", 100.0, {"ethane": 0.5, "propane": 0.5})
    stream = _stream("Cut", 25.0, {"ethane": 0.5, "propane": 0.5})

    assert distillation_feed_priority(stream, feed) == pytest.approx(
        stream_priority(stream, feed)
    )
    assert distillation_feed_priority(stream, feed) == pytest.approx(0.25)


def test_boundary_information_ranks_binary_cut_boundary_highest():
    feed = _stream(
        "Feed",
        100.0,
        {
            "ethane": 0.1,
            "propane": 0.15,
            "n-butane": 0.25,
            "n-pentane": 0.25,
            "n-hexane": 0.25,
        },
    )
    binary_cut = _stream(
        "C4C5Cut",
        25.0,
        {
            "ethane": 0.01,
            "propane": 0.03,
            "n-butane": 0.47,
            "n-pentane": 0.46,
            "n-hexane": 0.03,
        },
    )

    rows = rank_distillation_boundaries(
        binary_cut,
        ALPHAS,
        feed_stream=feed,
        components=COMPONENTS,
    )

    assert rows[0]["key_pair"] == "n-butane/n-pentane"
    assert rows[0]["adjacent"] is True
    assert rows[0]["p_light"] == pytest.approx(0.51)
    assert rows[0]["p_heavy"] == pytest.approx(0.49)
    assert rows[0]["boundary_information"] > rows[1]["boundary_information"]


def test_feed_priority_and_boundary_information_suppress_pure_and_low_flow_streams():
    feed = _stream("Feed", 100.0, {"ethane": 0.2, "propane": 0.2, "n-butane": 0.2, "n-pentane": 0.2, "n-hexane": 0.2})
    broad = _stream("Broad", 50.0, {"ethane": 0.2, "propane": 0.2, "n-butane": 0.2, "n-pentane": 0.2, "n-hexane": 0.2})
    nearly_pure = _stream("Pure", 12.0, {"ethane": 0.97, "propane": 0.02, "n-butane": 0.005, "n-pentane": 0.003, "n-hexane": 0.002})
    low_flow = _stream("Slip", 2.0, {"ethane": 0.2, "propane": 0.2, "n-butane": 0.2, "n-pentane": 0.2, "n-hexane": 0.2})

    broad_rows = rank_distillation_boundaries(broad, ALPHAS, feed, COMPONENTS)
    pure_rows = rank_distillation_boundaries(nearly_pure, ALPHAS, feed, COMPONENTS)
    low_flow_rows = rank_distillation_boundaries(low_flow, ALPHAS, feed, COMPONENTS)

    assert distillation_feed_priority(broad, feed, COMPONENTS) > 0.4
    assert distillation_feed_priority(nearly_pure, feed, COMPONENTS) < 0.02
    assert distillation_feed_priority(low_flow, feed, COMPONENTS) < 0.03
    assert broad_rows[0]["boundary_information"] > pure_rows[0]["boundary_information"]
    assert broad_rows[0]["boundary_information"] > low_flow_rows[0]["boundary_information"]


def test_non_adjacent_boundary_reports_middle_group_and_coverage():
    stream = _stream(
        "Broad",
        10.0,
        {"ethane": 0.2, "propane": 0.2, "n-butane": 0.2, "n-pentane": 0.2, "n-hexane": 0.2},
    )

    row = distillation_boundary_information(
        stream,
        ALPHAS,
        "ethane",
        "n-pentane",
        components=COMPONENTS,
    )

    assert row["adjacent"] is False
    assert row["middle_group"] == ("propane", "n-butane")
    assert row["coverage"] == pytest.approx(0.6)
    assert row["p_middle"] == pytest.approx(0.4)


def test_score_curve_adds_drop_and_relative_score_diagnostics():
    rows = [
        {"key_pair": "a/b", "boundary_information": 1.0},
        {"key_pair": "b/c", "boundary_information": 0.8},
        {"key_pair": "c/d", "boundary_information": 0.1},
    ]

    curve = distillation_score_curve(rows)

    assert [row["rank"] for row in curve] == [1, 2, 3]
    assert curve[0]["relative_to_best"] == pytest.approx(1.0)
    assert curve[1]["next_drop_fraction"] == pytest.approx(0.875)
    assert curve[-1]["next_drop_fraction"] is None


def test_invalid_key_pair_mode_is_actionable():
    stream = _stream("S", 1.0, {"ethane": 0.5, "propane": 0.5})

    with pytest.raises(ValueError, match="key_pair_mode must be 'adjacent' or 'all'"):
        rank_distillation_boundaries(
            stream,
            {"ethane": 2.0, "propane": 1.0},
            key_pair_mode="wide",
        )
