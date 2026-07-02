from __future__ import annotations

import pytest

from ml import (
    StreamState,
    rank_streams_by_priority,
    stream_composition_entropy,
    stream_priority,
)


def _stream(stream_id: str, flow: float, composition: dict[str, float]) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=flow,
        composition=composition,
    )


def test_stream_composition_entropy_scores_pure_and_binary_streams():
    pure = _stream("Pure", 1.0, {"a": 1.0, "b": 0.0})
    binary = _stream("Binary", 1.0, {"a": 0.5, "b": 0.5})

    assert stream_composition_entropy(pure, ["a", "b"]) == 0.0
    assert stream_composition_entropy(binary, ["a", "b"]) == pytest.approx(1.0)


def test_stream_priority_is_flow_weighted_entropy():
    feed = _stream("Feed", 100.0, {"a": 0.5, "b": 0.5})
    high_flow_mixed = _stream("HighFlowMixed", 50.0, {"a": 0.5, "b": 0.5})
    low_flow_mixed = _stream("LowFlowMixed", 5.0, {"a": 0.5, "b": 0.5})
    high_flow_pure = _stream("HighFlowPure", 50.0, {"a": 0.99, "b": 0.01})

    assert stream_priority(high_flow_mixed, feed) == pytest.approx(0.5)
    assert stream_priority(low_flow_mixed, feed) == pytest.approx(0.05)
    assert stream_priority(high_flow_pure, feed) < stream_priority(high_flow_mixed, feed)


def test_rank_streams_by_priority_orders_mixed_high_flow_streams_first():
    feed = _stream("Feed", 100.0, {"a": 0.5, "b": 0.5})
    low_flow_mixed = _stream("LowFlowMixed", 5.0, {"a": 0.5, "b": 0.5})
    high_flow_pure = _stream("HighFlowPure", 80.0, {"a": 0.99, "b": 0.01})
    high_flow_mixed = _stream("HighFlowMixed", 50.0, {"a": 0.5, "b": 0.5})

    ranked = rank_streams_by_priority(
        (low_flow_mixed, high_flow_pure, high_flow_mixed),
        feed,
    )

    assert [stream.id for stream, _ in ranked] == [
        "HighFlowMixed",
        "HighFlowPure",
        "LowFlowMixed",
    ]


def test_stream_entropy_rejects_duplicate_component_order():
    stream = _stream("S", 1.0, {"a": 0.5, "b": 0.5})

    with pytest.raises(ValueError, match="components must be unique"):
        stream_composition_entropy(stream, ["a", "a"])
