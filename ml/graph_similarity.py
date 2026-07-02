"""Similarity diagnostics for explicit process graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .graph_identity import state_identity_hash, state_topology_hash
from .process_graph import ProcessEdge, ProcessGraph, ProcessNode
from .separation_metrics import separation_indicator
from .types import StreamState


@dataclass(frozen=True)
class ProcessGraphSimilarity:
    """Similarity scores between two process graphs.

    Args:
        overall: Weighted score in [0, 1].
        topology_similarity: Weighted topology feature-bag score.
        terminal_stream_similarity: Similarity of terminal open/product stream
            states when available.
        objective_similarity: Similarity of objective-quality indicators when
            available.
        branch_similarity: Multiset Jaccard score for root-to-leaf branch
            fingerprints.
        unit_similarity: Multiset Jaccard score for unit action signatures.
        edge_role_similarity: Multiset Jaccard score for edge roles.
        product_role_similarity: Multiset Jaccard score for product roles.
        limitations: Reasons why one or more diagnostic layers were neutral.

    Returns:
        Immutable similarity score record.
    """

    overall: float
    topology_similarity: float
    terminal_stream_similarity: float
    objective_similarity: float
    branch_similarity: float
    unit_similarity: float
    edge_role_similarity: float
    product_role_similarity: float
    limitations: tuple[str, ...] = ()


def process_graph_fingerprints(graph_or_state: Any) -> dict[str, tuple[tuple[Any, ...], ...]]:
    """Return canonical feature bags used for process-graph similarity.

    Args:
        graph_or_state: ProcessGraph or SearchState-like object with a
            ``process_graph`` attribute.

    Returns:
        Dictionary of sorted tuple feature bags. Features ignore generated
        stream labels and graph node ids.

    Example:
        features = process_graph_fingerprints(result.best_state)
    """
    graph = _as_process_graph(graph_or_state)
    if graph is None or not graph.nodes:
        return {
            "branches": (),
            "units": (),
            "edge_roles": (),
            "product_roles": (),
        }

    nodes = {node.id: node for node in graph.nodes}
    outgoing: dict[str, list[ProcessEdge]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge)

    unit_features = tuple(
        sorted(_unit_feature(node) for node in graph.nodes if node.kind == "unit")
    )
    edge_role_features = tuple(sorted((edge.role,) for edge in graph.edges))
    product_role_features = tuple(
        sorted(
            ("product", _node_data(node, "role") or node.label)
            for node in graph.nodes
            if node.kind == "product"
        )
    )
    branch_features = tuple(
        sorted(
            branch
            for root_id in graph.root_node_ids
            for branch in _branch_fingerprints(root_id, nodes, outgoing)
        )
    )

    return {
        "branches": branch_features,
        "units": unit_features,
        "edge_roles": edge_role_features,
        "product_roles": product_role_features,
    }


def process_graph_similarity(
    left: Any,
    right: Any,
    weights: Mapping[str, float] | None = None,
    *,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    include_terminal_streams: bool = True,
    include_objective: bool = True,
) -> ProcessGraphSimilarity:
    """Compare two process graphs/states using layered diagnostics.

    Args:
        left: ProcessGraph or SearchState-like object.
        right: ProcessGraph or SearchState-like object.
        weights: Optional weights. Topology feature keys are ``branches``,
            ``units``, ``edge_roles``, and ``product_roles``. Overall layer keys
            are ``topology``, ``terminal_streams``, and ``objective``.
        feed_stream: Optional feed stream used for separation objective
            similarity.
        components: Optional component order for separation objective scoring.
        include_terminal_streams: Include terminal stream-state similarity when
            state data is available.
        include_objective: Include reward/duty/separation objective similarity
            when data is available.

    Returns:
        Similarity score record with component scores and weighted overall
        score in [0, 1].

    Example:
        score = process_graph_similarity(result_a.best_state, result_b.best_state)
    """
    left_features = process_graph_fingerprints(left)
    right_features = process_graph_fingerprints(right)
    branch_similarity = _multiset_jaccard(
        left_features["branches"], right_features["branches"]
    )
    unit_similarity = _multiset_jaccard(left_features["units"], right_features["units"])
    edge_role_similarity = _multiset_jaccard(
        left_features["edge_roles"], right_features["edge_roles"]
    )
    product_role_similarity = _multiset_jaccard(
        left_features["product_roles"], right_features["product_roles"]
    )

    topology_weights = {
        "branches": 0.4,
        "units": 0.3,
        "edge_roles": 0.2,
        "product_roles": 0.1,
    }
    layer_weights = {
        "topology": 0.6,
        "terminal_streams": 0.25,
        "objective": 0.15,
    }
    if weights is not None:
        topology_weights.update(
            {
                key: float(value)
                for key, value in weights.items()
                if key in topology_weights
            }
        )
        layer_weights.update(
            {
                key: float(value)
                for key, value in weights.items()
                if key in layer_weights
            }
        )
    topology_similarity = _weighted_average(
        {
            "branches": branch_similarity,
            "units": unit_similarity,
            "edge_roles": edge_role_similarity,
            "product_roles": product_role_similarity,
        },
        topology_weights,
    )

    limitations: list[str] = []
    if include_terminal_streams:
        terminal_stream_similarity = _terminal_stream_similarity(left, right, limitations)
    else:
        terminal_stream_similarity = 1.0
        limitations.append("terminal stream similarity disabled")

    if include_objective:
        objective_similarity = _objective_similarity(
            left,
            right,
            feed_stream,
            components,
            limitations,
        )
    else:
        objective_similarity = 1.0
        limitations.append("objective similarity disabled")

    overall = _weighted_average(
        {
            "topology": topology_similarity,
            "terminal_streams": terminal_stream_similarity,
            "objective": objective_similarity,
        },
        layer_weights,
    )

    return ProcessGraphSimilarity(
        overall=_normalise_score(overall),
        topology_similarity=topology_similarity,
        terminal_stream_similarity=terminal_stream_similarity,
        objective_similarity=objective_similarity,
        branch_similarity=branch_similarity,
        unit_similarity=unit_similarity,
        edge_role_similarity=edge_role_similarity,
        product_role_similarity=product_role_similarity,
        limitations=tuple(limitations),
    )


def rank_similar_process_graphs(
    reference: Any,
    candidates: Mapping[str, Any] | Sequence[Any],
    *,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    include_terminal_streams: bool = True,
    include_objective: bool = True,
) -> list[dict[str, Any]]:
    """Rank candidate graphs by similarity to a reference graph.

    Args:
        reference: Reference ProcessGraph or SearchState-like object.
        candidates: Mapping of label to graph/state, or an unlabeled sequence.
        feed_stream: Optional feed stream used for separation objective
            similarity.
        components: Optional component order for separation objective scoring.
        include_terminal_streams: Include terminal stream-state similarity.
        include_objective: Include objective-quality similarity.

    Returns:
        Rows sorted by decreasing overall similarity.

    Example:
        rows = rank_similar_process_graphs(result.best_state, other_states)
    """
    if isinstance(candidates, Mapping):
        labelled = [(str(label), candidate) for label, candidate in candidates.items()]
    else:
        labelled = [(f"candidate_{index}", candidate) for index, candidate in enumerate(candidates)]

    rows = []
    for label, candidate in labelled:
        score = process_graph_similarity(
            reference,
            candidate,
            feed_stream=feed_stream,
            components=components,
            include_terminal_streams=include_terminal_streams,
            include_objective=include_objective,
        )
        rows.append(
            {
                "label": label,
                "overall": score.overall,
                "topology_similarity": score.topology_similarity,
                "terminal_stream_similarity": score.terminal_stream_similarity,
                "objective_similarity": score.objective_similarity,
                "branch_similarity": score.branch_similarity,
                "unit_similarity": score.unit_similarity,
                "edge_role_similarity": score.edge_role_similarity,
                "product_role_similarity": score.product_role_similarity,
                "limitations": score.limitations,
            }
        )
    return sorted(rows, key=lambda row: (-row["overall"], row["label"]))


def stream_composition_cosine(
    left: StreamState,
    right: StreamState,
    components: Sequence[str] | None = None,
) -> float:
    """Return cosine similarity between stream composition vectors.

    Args:
        left: First stream.
        right: Second stream.
        components: Optional component order. Defaults to the union of both
            stream composition keys.

    Returns:
        Cosine similarity in [0, 1] for non-negative mole-fraction vectors.

    Example:
        score = stream_composition_cosine(stream_a, stream_b, components)
    """
    component_order = tuple(components or sorted(set(left.composition) | set(right.composition)))
    return _cosine(
        tuple(float(left.composition.get(component, 0.0)) for component in component_order),
        tuple(float(right.composition.get(component, 0.0)) for component in component_order),
    )


def stream_condition_vector(
    stream: StreamState,
    config: Any | None = None,
    feed_stream: StreamState | None = None,
) -> tuple[float, float, float]:
    """Return a dimensionless T/P/F condition vector for a stream.

    Args:
        stream: Stream to vectorize.
        config: Optional MCTSConfig-like object with temperature/pressure
            bounds.
        feed_stream: Optional feed stream used to scale molar flow.

    Returns:
        Tuple ``(T_scaled, logP_scaled, F_scaled)``.

    Example:
        vector = stream_condition_vector(stream, config, feed)
    """
    min_temperature = float(getattr(config, "min_temperature_K", 50.0))
    max_temperature = float(getattr(config, "max_temperature_K", 500.0))
    min_pressure = max(float(getattr(config, "min_pressure_Pa", 1.0)), 1e-12)
    max_pressure = max(float(getattr(config, "max_pressure_Pa", 1.0e7)), min_pressure * (1.0 + 1e-12))
    feed_flow = (
        float(feed_stream.molar_flow_mols)
        if feed_stream is not None and feed_stream.molar_flow_mols > 0.0
        else max(float(stream.molar_flow_mols), 1e-12)
    )
    temperature_span = max(max_temperature - min_temperature, 1e-12)
    pressure_span = max(math.log(max_pressure) - math.log(min_pressure), 1e-12)
    return (
        (float(stream.temperature_K) - min_temperature) / temperature_span,
        (math.log(max(float(stream.pressure_Pa), 1e-12)) - math.log(min_pressure))
        / pressure_span,
        float(stream.molar_flow_mols) / feed_flow,
    )


def stream_condition_cosine(
    left: StreamState,
    right: StreamState,
    config: Any | None = None,
    feed_stream: StreamState | None = None,
) -> float:
    """Return cosine similarity between scaled stream condition vectors.

    Args:
        left: First stream.
        right: Second stream.
        config: Optional MCTSConfig-like object with temperature/pressure
            bounds.
        feed_stream: Optional feed stream used to scale molar flow.

    Returns:
        Cosine similarity of ``(T_scaled, logP_scaled, F_scaled)``.

    Example:
        score = stream_condition_cosine(stream_a, stream_b, config, feed)
    """
    return _cosine(
        stream_condition_vector(left, config=config, feed_stream=feed_stream),
        stream_condition_vector(right, config=config, feed_stream=feed_stream),
    )


def terminal_stream_cosine_profile(
    left: Any,
    right: Any,
    components: Sequence[str] | None = None,
    config: Any | None = None,
    feed_stream: StreamState | None = None,
) -> dict[str, Any]:
    """Return terminal-stream cosine diagnostics for two states/results.

    Args:
        left: SearchState/MCTS result-like object.
        right: SearchState/MCTS result-like object.
        components: Optional composition vector order.
        config: Optional MCTSConfig-like object for condition scaling.
        feed_stream: Optional feed stream used to scale molar flow.

    Returns:
        Plain dict with matched-pair composition and condition cosine scores.

    Example:
        profile = terminal_stream_cosine_profile(state_a, state_b, components, config, feed)
    """
    left_streams = terminal_streams_for_similarity(left)
    right_streams = terminal_streams_for_similarity(right)
    if not left_streams or not right_streams:
        return {
            "average_composition_cosine": None,
            "minimum_composition_cosine": None,
            "average_condition_cosine": None,
            "minimum_condition_cosine": None,
            "matched_pair_count": 0,
            "unmatched_stream_count": len(left_streams) + len(right_streams),
            "matched_pairs": (),
            "limitations": ("terminal stream state unavailable",),
        }

    pair_scores = sorted(
        (
            (
                min(composition_score, condition_score),
                composition_score,
                condition_score,
                left_index,
                right_index,
            )
            for left_index, left_stream in enumerate(left_streams)
            for right_index, right_stream in enumerate(right_streams)
            for composition_score in (
                stream_composition_cosine(left_stream, right_stream, components),
            )
            for condition_score in (
                stream_condition_cosine(left_stream, right_stream, config, feed_stream),
            )
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matched_pairs = []
    for _, composition_score, condition_score, left_index, right_index in pair_scores:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        left_stream = left_streams[left_index]
        right_stream = right_streams[right_index]
        matched_pairs.append(
            {
                "left_stream_id": left_stream.id,
                "right_stream_id": right_stream.id,
                "composition_cosine": composition_score,
                "condition_cosine": condition_score,
            }
        )
    denominator = max(len(left_streams), len(right_streams))
    unmatched_count = denominator - len(matched_pairs)
    composition_scores = [pair["composition_cosine"] for pair in matched_pairs]
    condition_scores = [pair["condition_cosine"] for pair in matched_pairs]
    average_composition = sum(composition_scores) / denominator if denominator else None
    average_condition = sum(condition_scores) / denominator if denominator else None
    return {
        "average_composition_cosine": _normalise_score(average_composition or 0.0),
        "minimum_composition_cosine": min(composition_scores) if composition_scores and unmatched_count == 0 else 0.0,
        "average_condition_cosine": _normalise_score(average_condition or 0.0),
        "minimum_condition_cosine": min(condition_scores) if condition_scores and unmatched_count == 0 else 0.0,
        "matched_pair_count": len(matched_pairs),
        "unmatched_stream_count": unmatched_count,
        "matched_pairs": tuple(matched_pairs),
        "limitations": (),
    }


def topology_feature_cosine(left: Any, right: Any) -> float:
    """Return cosine similarity between cheap topology feature-count vectors.

    Args:
        left: ProcessGraph or SearchState-like object.
        right: ProcessGraph or SearchState-like object.

    Returns:
        Cosine similarity over unit, edge-role, product-role, and node-count
        features. This is a prefilter diagnostic, not exact identity.

    Example:
        score = topology_feature_cosine(state_a, state_b)
    """
    return _sparse_cosine(
        _topology_feature_counter(left),
        _topology_feature_counter(right),
    )


def suspicious_similarity_report(
    left: Any,
    right: Any,
    components: Sequence[str] | None = None,
    config: Any | None = None,
    feed_stream: StreamState | None = None,
    composition_threshold: float = 0.995,
    condition_threshold: float = 0.98,
    topology_threshold: float = 0.95,
) -> dict[str, Any]:
    """Classify whether two states/results are suspiciously similar.

    Args:
        left: ProcessGraph/SearchState/MCTS result-like object.
        right: ProcessGraph/SearchState/MCTS result-like object.
        components: Optional composition vector order.
        config: Optional MCTSConfig-like object for condition scaling.
        feed_stream: Optional feed stream used to scale molar flow.
        composition_threshold: Minimum terminal composition cosine.
        condition_threshold: Minimum terminal condition cosine.
        topology_threshold: Minimum cheap topology feature cosine.

    Returns:
        Plain dict with exact hash checks, cosine diagnostics, and a
        conservative classification label.

    Example:
        report = suspicious_similarity_report(result_a, result_b, components, config, feed)
    """
    exact_duplicate = _state_identity(left) is not None and _state_identity(left) == _state_identity(right)
    same_topology = _topology_identity(left) is not None and _topology_identity(left) == _topology_identity(right)
    stream_profile = terminal_stream_cosine_profile(
        left,
        right,
        components=components,
        config=config,
        feed_stream=feed_stream,
    )
    topology_cosine = topology_feature_cosine(left, right)
    composition_cosine = stream_profile["minimum_composition_cosine"]
    condition_cosine = stream_profile["minimum_condition_cosine"]
    similar_streams = (
        composition_cosine is not None
        and condition_cosine is not None
        and composition_cosine >= composition_threshold
        and condition_cosine >= condition_threshold
    )
    similar_topology = topology_cosine >= topology_threshold
    if exact_duplicate:
        classification = "exact_duplicate"
    elif same_topology and similar_streams:
        classification = "same_topology_similar_streams"
    elif similar_streams and not same_topology:
        classification = "similar_streams_different_topology"
    elif similar_topology and not similar_streams:
        classification = "similar_topology_different_streams"
    else:
        classification = "different"
    return {
        "classification": classification,
        "exact_duplicate": exact_duplicate,
        "same_topology_hash": same_topology,
        "similar_streams": similar_streams,
        "similar_topology_features": similar_topology,
        "minimum_composition_cosine": composition_cosine,
        "minimum_condition_cosine": condition_cosine,
        "topology_feature_cosine": topology_cosine,
        "stream_profile": stream_profile,
    }


def terminal_streams_for_similarity(item: Any) -> tuple[StreamState, ...]:
    """Return terminal streams used for stream-state similarity.

    Args:
        item: MCTS result, SearchState-like object, or other object with a
            ``best_state``/``open_streams``/``products`` shape.

    Returns:
        Tuple of open streams followed by accepted product streams. Returns an
        empty tuple when no terminal stream state is available.

    Example:
        terminal_streams = terminal_streams_for_similarity(result.best_state)
    """
    state = _as_state(item)
    if state is None:
        return ()
    streams = list(getattr(state, "open_streams", ()))
    streams.extend(getattr(product, "stream") for product in getattr(state, "products", ()))
    return tuple(streams)


def _as_process_graph(graph_or_state: Any) -> ProcessGraph | None:
    if isinstance(graph_or_state, ProcessGraph):
        return graph_or_state
    state = _as_state(graph_or_state)
    if state is not None:
        graph = getattr(state, "process_graph", None)
        if isinstance(graph, ProcessGraph):
            return graph
    graph = getattr(graph_or_state, "process_graph", None)
    if isinstance(graph, ProcessGraph):
        return graph
    return None


def _as_state(item: Any) -> Any | None:
    best_state = getattr(item, "best_state", None)
    if best_state is not None:
        return best_state
    if hasattr(item, "open_streams") and hasattr(item, "products"):
        return item
    return None


def _state_identity(item: Any) -> str | None:
    state = _as_state(item)
    if state is None:
        return None
    return state_identity_hash(state)


def _topology_identity(item: Any) -> str | None:
    state = _as_state(item)
    if state is None:
        return None
    return state_topology_hash(state)


def _topology_feature_counter(item: Any) -> Counter[tuple[Any, ...]]:
    graph = _as_process_graph(item)
    if graph is None or not graph.nodes:
        return Counter()
    counter: Counter[tuple[Any, ...]] = Counter()
    for node in graph.nodes:
        counter[(f"node_kind:{node.kind}",)] += 1
        if node.kind == "unit":
            counter[_unit_feature(node)] += 1
        elif node.kind == "product":
            counter[("product", _node_data(node, "role") or node.label)] += 1
    for edge in graph.edges:
        counter[("edge", edge.role)] += 1
    return counter


def _terminal_stream_similarity(
    left: Any,
    right: Any,
    limitations: list[str],
) -> float:
    left_streams = terminal_streams_for_similarity(left)
    right_streams = terminal_streams_for_similarity(right)
    if not left_streams or not right_streams:
        limitations.append("terminal stream state unavailable")
        return 1.0

    pair_scores = sorted(
        (
            (_stream_state_similarity(left_stream, right_stream), left_index, right_index)
            for left_index, left_stream in enumerate(left_streams)
            for right_index, right_stream in enumerate(right_streams)
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matched_score = 0.0
    for score, left_index, right_index in pair_scores:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matched_score += score
    return _normalise_score(matched_score / max(len(left_streams), len(right_streams)))


def _stream_state_similarity(left: StreamState, right: StreamState) -> float:
    composition_similarity = _composition_similarity(left.composition, right.composition)
    flow_similarity = _numeric_similarity(left.molar_flow_mols, right.molar_flow_mols)
    temperature_similarity = _numeric_similarity(left.temperature_K, right.temperature_K)
    pressure_similarity = _numeric_similarity(left.pressure_Pa, right.pressure_Pa)
    return _weighted_average(
        {
            "composition": composition_similarity,
            "flow": flow_similarity,
            "temperature": temperature_similarity,
            "pressure": pressure_similarity,
        },
        {
            "composition": 0.45,
            "flow": 0.25,
            "temperature": 0.15,
            "pressure": 0.15,
        },
    )


def _composition_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    components = set(left) | set(right)
    if not components:
        return 1.0
    distance = sum(abs(float(left.get(component, 0.0)) - float(right.get(component, 0.0))) for component in components)
    return _normalise_score(1.0 - 0.5 * distance)


def _objective_similarity(
    left: Any,
    right: Any,
    feed_stream: StreamState | None,
    components: Sequence[str] | None,
    limitations: list[str],
) -> float:
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}

    left_reward = getattr(left, "best_reward", None)
    right_reward = getattr(right, "best_reward", None)
    if left_reward is not None and right_reward is not None:
        scores["reward"] = _numeric_similarity(left_reward, right_reward)
        weights["reward"] = 0.25

    left_state = _as_state(left)
    right_state = _as_state(right)
    if left_state is not None and right_state is not None:
        scores["duty"] = _numeric_similarity(
            getattr(left_state, "total_abs_duty_W", 0.0),
            getattr(right_state, "total_abs_duty_W", 0.0),
        )
        weights["duty"] = 0.2

    left_streams = terminal_streams_for_similarity(left)
    right_streams = terminal_streams_for_similarity(right)
    if feed_stream is not None and left_streams and right_streams:
        try:
            left_metric = separation_indicator(feed_stream, left_streams, components=components)
            right_metric = separation_indicator(feed_stream, right_streams, components=components)
        except ValueError as exc:
            limitations.append(f"separation objective unavailable: {exc}")
        else:
            scores["separation_fraction"] = _numeric_similarity(
                left_metric["fraction_of_target"],
                right_metric["fraction_of_target"],
            )
            scores["component_scores"] = _dict_numeric_similarity(
                left_metric["component_scores"],
                right_metric["component_scores"],
            )
            scores["purities"] = _dict_numeric_similarity(
                left_metric["purities"],
                right_metric["purities"],
            )
            scores["recoveries"] = _dict_numeric_similarity(
                left_metric["recoveries"],
                right_metric["recoveries"],
            )
            weights.update(
                {
                    "separation_fraction": 0.25,
                    "component_scores": 0.15,
                    "purities": 0.075,
                    "recoveries": 0.075,
                }
            )
    else:
        limitations.append("separation objective unavailable without feed stream and terminal streams")

    if not scores:
        limitations.append("objective quality data unavailable")
        return 1.0
    return _weighted_average(scores, weights)


def _branch_fingerprints(
    node_id: str,
    nodes: dict[str, ProcessNode],
    outgoing: dict[str, list[ProcessEdge]],
) -> tuple[tuple[Any, ...], ...]:
    node = nodes.get(node_id)
    if node is None:
        return ()
    node_token = _node_token(node)
    child_edges = outgoing.get(node_id, ())
    if not child_edges:
        return ((node_token,),)
    branches = []
    for edge in child_edges:
        for child_branch in _branch_fingerprints(edge.target, nodes, outgoing):
            branches.append((node_token, ("edge", edge.role)) + child_branch)
    return tuple(branches)


def _node_token(node: ProcessNode) -> tuple[Any, ...]:
    if node.kind == "unit":
        return _unit_feature(node)
    if node.kind == "product":
        return ("product", _node_data(node, "role") or node.label)
    if node.kind == "stream":
        return ("stream", _node_data(node, "role"))
    return (node.kind, tuple(sorted(node.data)))


def _unit_feature(node: ProcessNode) -> tuple[Any, ...]:
    signature = _node_data(node, "action_signature")
    if signature is not None:
        return ("unit", signature)
    return ("unit", _node_data(node, "action_kind") or node.label)


def _node_data(node: ProcessNode, key: str) -> Any:
    for item_key, value in node.data:
        if item_key == key:
            return value
    return None


def _multiset_jaccard(left: tuple[tuple[Any, ...], ...], right: tuple[tuple[Any, ...], ...]) -> float:
    left_counter = Counter(left)
    right_counter = Counter(right)
    if not left_counter and not right_counter:
        return 1.0
    keys = set(left_counter) | set(right_counter)
    intersection = sum(min(left_counter[key], right_counter[key]) for key in keys)
    union = sum(max(left_counter[key], right_counter[key]) for key in keys)
    if union == 0:
        return 1.0
    return _normalise_score(intersection / union)


def _dict_numeric_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    return _normalise_score(
        sum(_numeric_similarity(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
        / len(keys)
    )


def _sparse_cosine(left: Counter[tuple[Any, ...]], right: Counter[tuple[Any, ...]]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    dot = sum(float(left[key]) * float(right[key]) for key in keys)
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left.values()))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return _normalise_score(dot / (left_norm * right_norm))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cosine vectors must have the same length.")
    if not left:
        return 1.0
    dot = sum(float(left_value) * float(right_value) for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return _normalise_score(dot / (left_norm * right_norm))


def _numeric_similarity(left: Any, right: Any) -> float:
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return 0.0
    if left_value == right_value:
        return 1.0
    denominator = max(abs(left_value), abs(right_value), 1e-12)
    return _normalise_score(1.0 / (1.0 + abs(left_value - right_value) / denominator))


def _weighted_average(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = sum(max(float(weights.get(key, 0.0)), 0.0) for key in scores)
    if total_weight <= 0.0:
        return 0.0
    return _normalise_score(
        sum(
            max(float(weights.get(key, 0.0)), 0.0) * _normalise_score(float(score))
            for key, score in scores.items()
        )
        / total_weight
    )


def _normalise_score(value: float) -> float:
    if abs(value - 1.0) < 1e-12:
        return 1.0
    if abs(value) < 1e-12:
        return 0.0
    return max(0.0, min(1.0, value))
