"""Information diagnostics for distillation action generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .distillation import estimate_relative_volatilities
from .flash import ThermoFlashProvider
from .stream_priority import stream_composition_entropy, stream_priority
from .types import StreamState


def distillation_feed_priority(
    stream: StreamState,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    min_component_fraction: float = 1e-8,
) -> float:
    """Return normalized mixing-entropy flow for a distillation feed stream.

    The score is ``(F_stream / F_reference) * H_norm(z_stream)``. It is high
    only when the stream is both compositionally mixed and materially
    significant relative to the reference feed.

    Args:
        stream: Candidate distillation feed stream.
        feed_stream: Optional reference feed for flow scaling.
        components: Optional component order.
        min_component_fraction: Minimum mole fraction used when inferring
            component order.

    Returns:
        Non-negative stream-level distillation priority.

    Example:
        priority = distillation_feed_priority(stream, feed_stream=feed)
    """
    return stream_priority(
        stream,
        feed_stream=feed_stream,
        components=components,
        min_component_fraction=min_component_fraction,
    )


def distillation_boundary_information(
    stream: StreamState,
    relative_volatilities: Mapping[str, float],
    light_key: str,
    heavy_key: str,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    min_component_fraction: float = 1e-8,
) -> dict[str, Any]:
    """Score one candidate distillation boundary before solving the column.

    The information score is:

    ``flow_fraction * coverage * H_binary(p_light, p_heavy) * log(alpha_LK/HK)``

    where ``p_light`` and ``p_heavy`` are the composition fractions on the
    light and heavy sides of the proposed volatility boundary. For adjacent
    boundaries, coverage is one. For non-adjacent key pairs, coverage excludes
    the middle components so broad key-pair jumps do not receive free credit
    for unresolved material.

    Args:
        stream: Candidate distillation feed stream.
        relative_volatilities: Positive relative volatility mapping by
            component.
        light_key: Candidate light key.
        heavy_key: Candidate heavy key.
        feed_stream: Optional reference feed for flow scaling.
        components: Optional component order. Defaults to components present in
            ``relative_volatilities`` and meaningful in the stream.
        min_component_fraction: Minimum mole fraction used when inferring
            component order.

    Returns:
        Plain dictionary containing score components and the final
        ``boundary_information`` value.

    Raises:
        ValueError: If keys, component order, or relative volatilities are
            invalid.

    Example:
        row = distillation_boundary_information(
            stream,
            {"propane": 2.2, "n-butane": 1.0},
            "propane",
            "n-butane",
            feed_stream=feed,
        )
    """
    component_order = _component_order(
        stream,
        relative_volatilities,
        components,
        min_component_fraction,
    )
    alphas = _validated_alphas(relative_volatilities, component_order)
    if light_key not in alphas:
        raise ValueError(
            f"light_key {light_key!r} is not in relative_volatilities/components."
        )
    if heavy_key not in alphas:
        raise ValueError(
            f"heavy_key {heavy_key!r} is not in relative_volatilities/components."
        )

    ordered = _volatility_order(component_order, alphas)
    light_index = ordered.index(light_key)
    heavy_index = ordered.index(heavy_key)
    if light_index >= heavy_index:
        raise ValueError(
            "light_key must be more volatile than heavy_key in the supplied "
            "relative_volatilities."
        )

    alpha_light = alphas[light_key]
    alpha_heavy = alphas[heavy_key]
    alpha_ratio = alpha_light / alpha_heavy
    if alpha_ratio <= 1.0:
        raise ValueError(
            f"alpha_LK/HK must be greater than one, got {alpha_ratio}."
        )

    light_group = ordered[: light_index + 1]
    middle_group = ordered[light_index + 1 : heavy_index]
    heavy_group = ordered[heavy_index:]
    p_light = _composition_sum(stream, light_group)
    p_middle = _composition_sum(stream, middle_group)
    p_heavy = _composition_sum(stream, heavy_group)
    coverage = p_light + p_heavy
    boundary_entropy = _binary_entropy(p_light, p_heavy)
    reference_flow = _reference_flow(stream, feed_stream)
    flow_fraction = max(0.0, float(stream.molar_flow_mols)) / reference_flow
    log_alpha_ratio = math.log(alpha_ratio)
    boundary_information = (
        flow_fraction * coverage * boundary_entropy * log_alpha_ratio
    )
    entropy = stream_composition_entropy(
        stream,
        component_order,
        normalise=True,
        min_component_fraction=min_component_fraction,
    )
    feed_priority = flow_fraction * entropy

    return {
        "stream_id": stream.id,
        "stream_flow_mols": float(stream.molar_flow_mols),
        "reference_flow_mols": reference_flow,
        "flow_fraction": flow_fraction,
        "composition_entropy": entropy,
        "distillation_feed_priority": feed_priority,
        "light_key": light_key,
        "heavy_key": heavy_key,
        "key_pair": f"{light_key}/{heavy_key}",
        "light_index": light_index,
        "heavy_index": heavy_index,
        "adjacent": heavy_index == light_index + 1,
        "light_group": tuple(light_group),
        "middle_group": tuple(middle_group),
        "heavy_group": tuple(heavy_group),
        "p_light": p_light,
        "p_middle": p_middle,
        "p_heavy": p_heavy,
        "coverage": coverage,
        "boundary_entropy": boundary_entropy,
        "alpha_light": alpha_light,
        "alpha_heavy": alpha_heavy,
        "alpha_ratio": alpha_ratio,
        "log_alpha_ratio": log_alpha_ratio,
        "boundary_information": boundary_information,
    }


def rank_distillation_boundaries(
    stream: StreamState,
    relative_volatilities: Mapping[str, float],
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    key_pair_mode: str = "adjacent",
    min_component_fraction: float = 1e-8,
) -> tuple[dict[str, Any], ...]:
    """Rank candidate LK/HK boundaries by information score.

    Args:
        stream: Candidate distillation feed stream.
        relative_volatilities: Positive relative volatility mapping by
            component.
        feed_stream: Optional reference feed for flow scaling.
        components: Optional component order.
        key_pair_mode: ``"adjacent"`` for neighboring volatility boundaries or
            ``"all"`` for every lighter/heavier pair.
        min_component_fraction: Minimum mole fraction used when inferring
            component order.

    Returns:
        Tuple of boundary rows sorted by decreasing ``boundary_information``.

    Example:
        rows = rank_distillation_boundaries(stream, alphas, feed_stream=feed)
    """
    component_order = _component_order(
        stream,
        relative_volatilities,
        components,
        min_component_fraction,
    )
    alphas = _validated_alphas(relative_volatilities, component_order)
    ordered = _volatility_order(component_order, alphas)
    pairs = _key_pairs(ordered, key_pair_mode)
    rows = [
        distillation_boundary_information(
            stream,
            alphas,
            light_key,
            heavy_key,
            feed_stream=feed_stream,
            components=component_order,
            min_component_fraction=min_component_fraction,
        )
        for light_key, heavy_key in pairs
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -float(row["boundary_information"]),
                str(row["key_pair"]),
            ),
        )
    )


def distillation_information_table(
    streams: Sequence[StreamState],
    provider: ThermoFlashProvider,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    *,
    key_pair_mode: str = "adjacent",
    min_component_fraction: float = 1e-8,
) -> list[dict[str, Any]]:
    """Create information-score diagnostics for candidate distillation feeds.

    Args:
        streams: Candidate open streams.
        provider: Thermo provider used to estimate relative volatilities.
        feed_stream: Optional reference feed for flow scaling.
        components: Optional component order.
        key_pair_mode: ``"adjacent"`` or ``"all"`` key-pair generation.
        min_component_fraction: Minimum mole fraction used when inferring
            component order.

    Returns:
        List of boundary diagnostic dictionaries sorted by stream priority and
        boundary information. No MCTS behavior is changed.

    Example:
        rows = distillation_information_table(state.open_streams, provider, feed)
    """
    rows: list[dict[str, Any]] = []
    component_order = tuple(components) if components is not None else provider.compounds
    for stream in streams:
        alphas, _, warnings = estimate_relative_volatilities(stream, provider)
        ranked = rank_distillation_boundaries(
            stream,
            alphas,
            feed_stream=feed_stream,
            components=component_order,
            key_pair_mode=key_pair_mode,
            min_component_fraction=min_component_fraction,
        )
        for row in ranked:
            row = dict(row)
            row["alpha_warnings"] = tuple(warnings)
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row["distillation_feed_priority"]),
            str(row["stream_id"]),
            -float(row["boundary_information"]),
            str(row["key_pair"]),
        ),
    )


def distillation_score_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "boundary_information",
) -> list[dict[str, Any]]:
    """Return sorted score-curve diagnostics for elbow/drop inspection.

    Args:
        rows: Boundary diagnostic rows.
        score_key: Numeric score column to sort and analyse.

    Returns:
        Rows with rank, relative score, cumulative score fraction, and drop to
        the next score. These values are useful for plotting sharp changes or
        flat score regions.

    Example:
        curve = distillation_score_curve(boundary_rows)
    """
    sorted_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (-float(row.get(score_key, 0.0)), str(row.get("key_pair", ""))),
    )
    if not sorted_rows:
        return []
    scores = [max(0.0, float(row.get(score_key, 0.0))) for row in sorted_rows]
    best = scores[0]
    total = sum(scores)
    cumulative = 0.0
    curve: list[dict[str, Any]] = []
    for index, (row, score) in enumerate(zip(sorted_rows, scores), start=1):
        next_score = scores[index] if index < len(scores) else None
        cumulative += score
        drop_abs = None if next_score is None else score - next_score
        drop_fraction = (
            None
            if next_score is None or score <= 0.0
            else (score - next_score) / score
        )
        row.update(
            {
                "rank": index,
                "score": score,
                "relative_to_best": 0.0 if best <= 0.0 else score / best,
                "cumulative_score_fraction": (
                    0.0 if total <= 0.0 else cumulative / total
                ),
                "next_score": next_score,
                "next_drop_abs": drop_abs,
                "next_drop_fraction": drop_fraction,
            }
        )
        curve.append(row)
    return curve


def _component_order(
    stream: StreamState,
    relative_volatilities: Mapping[str, float],
    components: Sequence[str] | None,
    min_component_fraction: float,
) -> tuple[str, ...]:
    if min_component_fraction < 0.0:
        raise ValueError("min_component_fraction must be non-negative.")
    if components is None:
        selected = tuple(
            component
            for component in relative_volatilities
            if float(stream.composition.get(component, 0.0)) >= min_component_fraction
        )
    else:
        selected = tuple(str(component) for component in components)
    if len(set(selected)) != len(selected):
        raise ValueError("components must be unique.")
    missing = [component for component in selected if component not in relative_volatilities]
    if missing:
        raise ValueError(
            "relative_volatilities must include every selected component; "
            f"missing {missing}."
        )
    if len(selected) < 2:
        raise ValueError(
            "At least two components are required for distillation information "
            "diagnostics."
        )
    return selected


def _validated_alphas(
    relative_volatilities: Mapping[str, float],
    components: Sequence[str],
) -> dict[str, float]:
    alphas: dict[str, float] = {}
    for component in components:
        try:
            value = float(relative_volatilities[component])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"relative_volatilities[{component!r}] must be a positive number."
            ) from exc
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(
                f"relative_volatilities[{component!r}] must be positive and finite, "
                f"got {value}."
            )
        alphas[component] = value
    return alphas


def _volatility_order(
    components: Sequence[str],
    relative_volatilities: Mapping[str, float],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            components,
            key=lambda component: (-float(relative_volatilities[component]), component),
        )
    )


def _key_pairs(ordered_components: Sequence[str], key_pair_mode: str) -> tuple[tuple[str, str], ...]:
    if key_pair_mode == "adjacent":
        return tuple(zip(ordered_components, ordered_components[1:]))
    if key_pair_mode == "all":
        return tuple(
            (light_key, heavy_key)
            for light_index, light_key in enumerate(ordered_components)
            for heavy_key in ordered_components[light_index + 1 :]
        )
    raise ValueError("key_pair_mode must be 'adjacent' or 'all'.")


def _composition_sum(stream: StreamState, components: Sequence[str]) -> float:
    return sum(max(0.0, float(stream.composition.get(component, 0.0))) for component in components)


def _binary_entropy(left: float, right: float) -> float:
    total = left + right
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in (left / total, right / total):
        if value > 0.0:
            entropy -= value * math.log(value)
    return entropy


def _reference_flow(stream: StreamState, feed_stream: StreamState | None) -> float:
    if feed_stream is not None and feed_stream.molar_flow_mols > 0.0:
        return float(feed_stream.molar_flow_mols)
    return max(float(stream.molar_flow_mols), 1e-12)
