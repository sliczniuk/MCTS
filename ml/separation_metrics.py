"""Separation quality metrics for flowsheet search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .types import StreamState


_COMPOSITION_TOL = 1e-12


def separation_indicator(
    feed_stream: StreamState,
    outlet_streams: Sequence[StreamState],
    components: Sequence[str] | None = None,
    basis: Literal["molar"] = "molar",
    min_component_fraction: float = 1e-8,
) -> dict:
    """Compute a continuous component-separation indicator.

    For each feed component, the metric finds the outlet stream maximizing
    purity * recovery. The ideal total score equals the number of meaningful
    feed components.

    Args:
        feed_stream: Original inlet stream.
        outlet_streams: Candidate outlet/product streams.
        components: Optional components to score. Defaults to feed components
            with mole fraction >= min_component_fraction.
        basis: Flow basis. Only "molar" is supported in v1.
        min_component_fraction: Minimum feed mole fraction for inferred
            components.

    Returns:
        Plain dict with score, target, fraction_of_target, component scores,
        best streams, purities, and recoveries.

    Raises:
        ValueError: if inputs are invalid or an unsupported basis is requested.

    Example:
        metric = separation_indicator(feed, [distillate, bottoms])
        print(metric["score"], metric["target"])
    """
    if basis != "molar":
        raise ValueError("Only basis='molar' is supported in v1.")
    if feed_stream.molar_flow_mols <= 0:
        raise ValueError(
            f"feed_stream '{feed_stream.id}' must have positive molar_flow_mols."
        )
    if min_component_fraction < 0:
        raise ValueError("min_component_fraction must be non-negative.")

    scored_components = _components(feed_stream, components, min_component_fraction)
    if not scored_components:
        raise ValueError("No feed components were selected for separation scoring.")

    feed_component_flows = {
        component: feed_stream.molar_flow_mols
        * float(feed_stream.composition.get(component, 0.0))
        for component in scored_components
    }
    for component, flow in feed_component_flows.items():
        if flow <= _COMPOSITION_TOL:
            raise ValueError(
                f"Component '{component}' has zero feed flow and cannot be scored."
            )

    component_scores: dict[str, float] = {}
    best_stream_by_component: dict[str, str | None] = {}
    purities: dict[str, float] = {}
    recoveries: dict[str, float] = {}

    for component in scored_components:
        best_score = 0.0
        best_stream_id: str | None = None
        best_purity = 0.0
        best_recovery = 0.0

        for stream in outlet_streams:
            if stream.molar_flow_mols <= 0:
                continue
            purity = max(0.0, float(stream.composition.get(component, 0.0)))
            component_flow = stream.molar_flow_mols * purity
            recovery = component_flow / feed_component_flows[component]
            recovery = max(0.0, min(1.0, recovery))
            score = purity * recovery
            if score > best_score:
                best_score = score
                best_stream_id = stream.id
                best_purity = purity
                best_recovery = recovery

        component_scores[component] = best_score
        best_stream_by_component[component] = best_stream_id
        purities[component] = best_purity
        recoveries[component] = best_recovery

    target = len(scored_components)
    score = sum(component_scores.values())
    return {
        "score": score,
        "target": target,
        "fraction_of_target": score / target if target else 0.0,
        "component_scores": component_scores,
        "best_stream_by_component": best_stream_by_component,
        "purities": purities,
        "recoveries": recoveries,
    }


def mutual_information_separation(
    feed_stream: StreamState,
    outlet_streams: Sequence[StreamState],
    components: Sequence[str] | None = None,
    min_component_fraction: float = 1e-8,
    weight_mode: Literal["feed_fraction", "equal_weight"] = "feed_fraction",
) -> dict:
    """Mutual-information separation score.

    Models separation as a joint distribution over (component, stream):

        P(C=i, K=k) = F_k * x_{i,k} / F_0

    The mutual information I(C; K) = H(C) - H(C|K) measures how much knowing
    the stream identity reduces uncertainty about the component identity.
    Perfect separation achieves I(C; K) = H(C); no separation gives I = 0.

    The score is scaled to [0, N_C] for drop-in compatibility with
    ``separation_indicator``:

        score = N_C * I(C; K) / H(C)

    This metric is weight-free: no tunable bonus or penalty parameters. It
    naturally handles degeneracy — a stream that carries all components at
    near-feed composition contributes almost zero to I(C; K) regardless of
    its flow.

    Args:
        feed_stream: Original inlet stream.
        outlet_streams: All outlet / product streams (products + open streams).
            Together they should account for all feed flow by material balance.
        components: Components to score. Defaults to those with
            z_i >= min_component_fraction.
        min_component_fraction: Minimum feed mole fraction for component
            selection.
        weight_mode: Component weighting scheme for the MI calculation.
            ``"feed_fraction"`` (default) uses the feed mole fraction z_i as
            the component prior — standard MI under the true feed distribution.
            ``"equal_weight"`` uses a uniform prior 1/N_C for every component,
            so minority components (e.g. n-heptane at 5 %) receive the same
            importance as majority components (e.g. propane at 30 %). The
            score range [0, N_C] and all returned keys are identical in both
            modes; the reference entropy changes from H(z_feed) to log(N_C).

    Returns:
        Plain dict with keys:
            score           — N_C * I(C;K) / H_ref, range [0, N_C]
            target          — N_C (number of scored components)
            fraction_of_target — I(C;K) / H_ref, range [0, 1]
            mi_nats         — raw mutual information I(C;K) in nats
            feed_entropy_nats — reference entropy H_ref in nats

    Example:
        m = mutual_information_separation(feed, [distillate, bottoms])
        print(m["score"], "/", m["target"])
    """
    if feed_stream.molar_flow_mols <= 0:
        raise ValueError(
            f"feed_stream '{feed_stream.id}' must have positive molar_flow_mols."
        )
    if min_component_fraction < 0:
        raise ValueError("min_component_fraction must be non-negative.")

    scored_components = _components(feed_stream, components, min_component_fraction)
    if not scored_components:
        raise ValueError("No feed components were selected for separation scoring.")

    n_c = len(scored_components)
    F_0 = feed_stream.molar_flow_mols
    z = {c: float(feed_stream.composition.get(c, 0.0)) for c in scored_components}

    if weight_mode == "equal_weight":
        return _mutual_information_equal_weight(
            scored_components, n_c, F_0, z, outlet_streams
        )

    # ── feed_fraction (default) ───────────────────────────────────────────────
    # Feed composition entropy H(C) = -Σ_i z_i ln(z_i)
    h_feed = -sum(zi * math.log(zi) for zi in z.values() if zi > _COMPOSITION_TOL)

    if h_feed < _COMPOSITION_TOL:
        # Single-component feed — trivially separated; avoid division by zero.
        return {
            "score": float(n_c),
            "target": n_c,
            "fraction_of_target": 1.0,
            "mi_nats": 0.0,
            "feed_entropy_nats": 0.0,
        }

    # Build component-flow matrix; derive normalisation constant from actual outlet flows.
    # F_norm = max(F_0, F_total_out): for dropped streams (F_total < F_0) we keep F_0
    # so the slight sub-normalisation is harmless; for recycle excess (F_total > F_0)
    # we use F_total so Σ w_k = 1 and H(C|K) cannot exceed H(C).
    A = _component_flow_matrix(scored_components, list(outlet_streams))
    F_total_out = float(A.sum())
    F_norm = max(F_0, F_total_out)

    # Conditional entropy H(C|K) = Σ_k (F_k/F_norm) × (-Σ_i x_{i,k} ln(x_{i,k}))
    # I(C;K) = H(C) - H(C|K); computed per-stream for numerical stability.
    col_sums = A.sum(axis=0)                           # F_k per stream
    h_conditional = 0.0
    for k, stream in enumerate(outlet_streams):
        w_k = float(col_sums[k]) / F_norm
        if w_k <= 0:
            continue
        h_k = 0.0
        for c in scored_components:
            x_ik = max(0.0, float(stream.composition.get(c, 0.0)))
            if x_ik > _COMPOSITION_TOL:
                h_k -= x_ik * math.log(x_ik)
        h_conditional += w_k * h_k

    mi = max(0.0, h_feed - h_conditional)
    fraction = min(1.0, mi / h_feed)

    return {
        "score": n_c * fraction,
        "target": n_c,
        "fraction_of_target": fraction,
        "mi_nats": mi,
        "feed_entropy_nats": h_feed,
    }


def _mutual_information_equal_weight(
    scored_components: tuple[str, ...],
    n_c: int,
    F_0: float,
    z: dict[str, float],
    outlet_streams: Sequence[StreamState],
) -> dict:
    """Equal-weight MI: uniform component prior P(C=i) = 1/N_C.

    Under the uniform prior the reference entropy is H_ref = log(N_C).
    Each component contributes equally to the score regardless of feed
    fraction.

    Uses a component-flow matrix A[i,k] = F_k * x_{i,k}.  Row i is
    normalised to give P(K=k | C=i), then each component row is weighted
    equally (1/N_C valid components).  This formulation is independent of
    F_0 and z_i, and self-consistent under any material-balance error.
    """
    if n_c <= 1:
        return {
            "score": float(n_c),
            "target": n_c,
            "fraction_of_target": 1.0,
            "mi_nats": 0.0,
            "feed_entropy_nats": 0.0,
        }

    h_ref = math.log(n_c)  # entropy under uniform prior

    # Build the component-flow matrix and form B[i,k] = P(C=i) * P(K=k | C=i)
    # with uniform P(C=i) = 1 / (number of components with any outlet flow).
    A = _component_flow_matrix(scored_components, list(outlet_streams))
    row_sums = A.sum(axis=1)                              # total flow per component
    valid = row_sums > _COMPOSITION_TOL

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "score": 0.0,
            "target": n_c,
            "fraction_of_target": 0.0,
            "mi_nats": 0.0,
            "feed_entropy_nats": h_ref,
        }

    B = np.zeros_like(A)
    B[valid, :] = A[valid, :] / row_sums[valid, np.newaxis]  # P(K=k | C=i)
    B[valid, :] /= float(n_valid)                             # × uniform P(C=i)

    total_B = float(B.sum())
    if total_B <= _COMPOSITION_TOL:
        return {
            "score": 0.0,
            "target": n_c,
            "fraction_of_target": 0.0,
            "mi_nats": 0.0,
            "feed_entropy_nats": h_ref,
        }

    p_i = B.sum(axis=1, keepdims=True)                       # marginal P(C=i)
    p_k = B.sum(axis=0, keepdims=True)                       # marginal P(K=k)
    expected = (p_i * p_k) / total_B                         # independence baseline

    mask = (B > _COMPOSITION_TOL) & (expected > _COMPOSITION_TOL)
    mi_eq = float(np.sum(B[mask] * np.log(B[mask] / expected[mask])))
    mi_eq = max(0.0, mi_eq)
    fraction = min(1.0, mi_eq / h_ref)

    return {
        "score": n_c * fraction,
        "target": n_c,
        "fraction_of_target": fraction,
        "mi_nats": mi_eq,
        "feed_entropy_nats": h_ref,
    }


def _component_flow_matrix(
    scored_components: tuple[str, ...],
    outlet_streams: Sequence[StreamState],
) -> np.ndarray:
    """A[i, k] = molar flow of component i in stream k  (shape: n_c × n_k)."""
    A = np.zeros((len(scored_components), len(outlet_streams)), dtype=float)
    for k, stream in enumerate(outlet_streams):
        if stream.molar_flow_mols <= 0:
            continue
        for i, c in enumerate(scored_components):
            A[i, k] = max(0.0, stream.molar_flow_mols * float(stream.composition.get(c, 0.0)))
    return A


def _components(
    feed_stream: StreamState,
    components: Sequence[str] | None,
    min_component_fraction: float,
) -> tuple[str, ...]:
    if components is not None:
        selected = tuple(str(component) for component in components)
        if len(set(selected)) != len(selected):
            raise ValueError(f"components must be unique, got {list(selected)}.")
        missing = [
            component
            for component in selected
            if component not in feed_stream.composition
        ]
        if missing:
            raise ValueError(
                f"components {missing} are not present in feed_stream composition."
            )
        return selected

    return tuple(
        component
        for component, fraction in feed_stream.composition.items()
        if float(fraction) >= min_component_fraction
    )
