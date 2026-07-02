"""Stream-priority diagnostics for MCTS action generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .types import StreamState


def stream_composition_entropy(
    stream: StreamState,
    components: Sequence[str] | None = None,
    *,
    normalise: bool = True,
    min_component_fraction: float = 0.0,
) -> float:
    """Return Shannon entropy of a stream composition vector.

    Args:
        stream: Stream to score.
        components: Optional component order. Defaults to stream composition
            keys with mole fraction above ``min_component_fraction``.
        normalise: If True, divide by ``log(n_components)`` so the score is
            in ``[0, 1]`` for non-negative mole fractions.
        min_component_fraction: Components below this mole fraction are ignored
            when ``components`` is omitted.

    Returns:
        Composition entropy. Pure or single-component streams return ``0.0``.

    Example:
        entropy = stream_composition_entropy(stream, ["propane", "n-butane"])
    """
    component_order = _component_order(
        stream.composition,
        components,
        min_component_fraction,
    )
    if len(component_order) <= 1:
        return 0.0

    fractions = [
        max(0.0, float(stream.composition.get(component, 0.0)))
        for component in component_order
    ]
    total = sum(fractions)
    if total <= 0.0:
        return 0.0

    entropy = 0.0
    for fraction in fractions:
        if fraction <= 0.0:
            continue
        probability = fraction / total
        entropy -= probability * math.log(probability)

    if not normalise:
        return entropy
    return entropy / math.log(len(component_order))


def stream_priority(
    stream: StreamState,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    min_component_fraction: float = 1e-8,
) -> float:
    """Return flow-weighted composition-mixing priority for a stream.

    The score is:

    ``priority = (F_stream / F_reference) * H_norm(z_stream)``

    where ``H_norm`` is normalized Shannon entropy. If ``feed_stream`` is not
    supplied, the stream's own flow is used as the reference flow, so the score
    reduces to normalized composition entropy.

    Args:
        stream: Stream to prioritize.
        feed_stream: Optional original feed used for flow scaling and default
            component selection.
        components: Optional component order. Defaults to meaningful feed
            components when ``feed_stream`` is supplied, otherwise stream
            components.
        min_component_fraction: Minimum mole fraction used for inferred
            components.

    Returns:
        Non-negative priority. Higher values indicate more useful separation
        work remains.

    Example:
        score = stream_priority(stream, feed_stream=feed)
    """
    component_order = components
    if component_order is None and feed_stream is not None:
        component_order = _component_order(
            feed_stream.composition,
            None,
            min_component_fraction,
        )
    entropy = stream_composition_entropy(
        stream,
        component_order,
        normalise=True,
        min_component_fraction=min_component_fraction,
    )
    reference_flow = (
        float(feed_stream.molar_flow_mols)
        if feed_stream is not None and feed_stream.molar_flow_mols > 0.0
        else max(float(stream.molar_flow_mols), 1e-12)
    )
    return max(0.0, float(stream.molar_flow_mols)) / reference_flow * entropy


def rank_streams_by_priority(
    streams: Sequence[StreamState],
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    min_component_fraction: float = 1e-8,
) -> tuple[tuple[StreamState, float], ...]:
    """Rank streams by decreasing stream-priority score.

    Args:
        streams: Streams to rank.
        feed_stream: Optional original feed used for flow scaling and default
            component selection.
        components: Optional component order.
        min_component_fraction: Minimum mole fraction used for inferred
            components.

    Returns:
        Tuple of ``(stream, priority)`` pairs sorted by decreasing priority and
        then stream id for deterministic ties.

    Example:
        ranked = rank_streams_by_priority(state.open_streams, feed)
    """
    scored = [
        (
            stream,
            stream_priority(
                stream,
                feed_stream=feed_stream,
                components=components,
                min_component_fraction=min_component_fraction,
            ),
        )
        for stream in streams
    ]
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0].id)))


def _component_order(
    composition: Mapping[str, float],
    components: Sequence[str] | None,
    min_component_fraction: float,
) -> tuple[str, ...]:
    if min_component_fraction < 0.0:
        raise ValueError("min_component_fraction must be non-negative.")
    if components is not None:
        selected = tuple(str(component) for component in components)
        if len(set(selected)) != len(selected):
            raise ValueError("components must be unique.")
        return selected
    return tuple(
        component
        for component, fraction in composition.items()
        if float(fraction) >= min_component_fraction
    )
