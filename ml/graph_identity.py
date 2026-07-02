"""Canonical identity helpers for search-side flowsheet states."""

from __future__ import annotations

import hashlib
from typing import Any

from .process_graph import ProcessGraph, ProcessNode, get_stream_node_id
from .types import StreamState


_DEFAULT_FLOAT_DIGITS = 8


def action_signature(action: Any, float_digits: int = _DEFAULT_FLOAT_DIGITS) -> tuple:
    """Return a stream-id-independent signature for a unit action.

    Args:
        action: UnitAction-like object with MCTS action attributes.
        float_digits: Decimal places used for floating action parameters.

    Returns:
        Tuple suitable for canonical state signatures and hashing.

    Example:
        sig = action_signature(UnitAction(kind="hx", stream_id="S1", delta_T_K=10.0))
    """
    kind = getattr(action, "kind", None)
    if kind == "hx":
        return ("hx", _round_value(getattr(action, "delta_T_K", None), float_digits))
    if kind in {"compressor", "pump", "valve"}:
        return (
            kind,
            _round_value(getattr(action, "pressure_ratio", None), float_digits),
            _round_value(getattr(action, "delta_P_Pa", None), float_digits),
        )
    if kind == "distillation":
        return (
            "distillation",
            getattr(action, "light_key", None),
            getattr(action, "heavy_key", None),
            _round_value(getattr(action, "light_key_recovery", None), float_digits),
            _round_value(getattr(action, "heavy_key_recovery", None), float_digits),
            _round_value(
                getattr(action, "reflux_ratio_multiplier", None),
                float_digits,
            ),
        )
    if kind == "flash":
        return ("flash",)
    if kind == "accept":
        return ("accept", getattr(action, "role", None))
    return (kind,)


def stream_signature(
    stream: StreamState,
    float_digits: int = _DEFAULT_FLOAT_DIGITS,
) -> tuple:
    """Return a stream-id-independent rounded stream state signature.

    Args:
        stream: Search-side stream state.
        float_digits: Decimal places used for stream state and composition.

    Returns:
        Tuple containing rounded T, P, flow, composition, and history labels.

    Example:
        sig = stream_signature(feed)
    """
    composition = tuple(
        sorted(
            (
                component,
                _round_value(value, float_digits),
            )
            for component, value in stream.composition.items()
        )
    )
    return (
        _round_value(stream.temperature_K, float_digits),
        _round_value(stream.pressure_Pa, float_digits),
        _round_value(stream.molar_flow_mols, float_digits),
        composition,
        tuple(stream.history),
    )


def state_topology_signature(
    state: Any,
    float_digits: int = _DEFAULT_FLOAT_DIGITS,
) -> tuple:
    """Return a canonical topology signature for a SearchState-like object.

    Args:
        state: SearchState-like object. Graph-backed states use
            ``state.process_graph``; legacy states use a minimal state-only
            fallback that does not infer generated stream names.
        float_digits: Decimal places used for floating action parameters.

    Returns:
        Hashable tuple representing explicit topology, product roles, edge
        roles, and error count.

    Example:
        sig = state_topology_signature(result.best_state)
    """
    graph = getattr(state, "process_graph", None)
    errors = ("errors", len(getattr(state, "errors", ())))
    if isinstance(graph, ProcessGraph) and graph.nodes:
        return ("topology", _canonical_graph_signature(graph, None), errors)
    return ("topology", _minimal_topology_signature(state), errors)


def state_identity_signature(
    state: Any,
    float_digits: int = _DEFAULT_FLOAT_DIGITS,
) -> tuple:
    """Return a canonical topology plus terminal-stream state signature.

    Args:
        state: SearchState-like object.
        float_digits: Decimal places used for terminal stream state fields.

    Returns:
        Hashable tuple suitable for exact duplicate state detection.

    Example:
        sig = state_identity_signature(result.best_state)
    """
    graph = getattr(state, "process_graph", None)
    errors = ("errors", len(getattr(state, "errors", ())))
    if isinstance(graph, ProcessGraph) and graph.nodes:
        terminal_streams = _terminal_stream_signatures(state, graph, float_digits)
        return ("state", _canonical_graph_signature(graph, terminal_streams), errors)
    return ("state", _minimal_state_signature(state, float_digits), errors)


def state_topology_hash(
    state: Any,
    float_digits: int = _DEFAULT_FLOAT_DIGITS,
) -> str:
    """Return a stable hash for a canonical topology signature.

    Args:
        state: SearchState-like object.
        float_digits: Decimal places used for floating action parameters.

    Returns:
        Hex digest for the canonical topology signature.

    Example:
        hash_value = state_topology_hash(result.best_state)
    """
    return _hash_signature(state_topology_signature(state, float_digits))


def state_identity_hash(
    state: Any,
    float_digits: int = _DEFAULT_FLOAT_DIGITS,
) -> str:
    """Return a stable hash for exact duplicate state detection.

    Args:
        state: SearchState-like object.
        float_digits: Decimal places used for floating fields.

    Returns:
        Hex digest for exact duplicate state detection.

    Example:
        hash_value = state_identity_hash(result.best_state)
    """
    return _hash_signature(state_identity_signature(state, float_digits))


def _canonical_graph_signature(
    graph: ProcessGraph,
    terminal_streams: dict[str, tuple] | None,
) -> tuple:
    nodes = {node.id: node for node in graph.nodes}
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append((edge.role, edge.target))

    def node_signature(node_id: str) -> tuple:
        node = nodes[node_id]
        children = tuple(
            sorted(
                (role, node_signature(child_id))
                for role, child_id in outgoing.get(node_id, [])
            )
        )
        if node.kind == "unit":
            return (
                "unit",
                _node_data(node, "action_signature"),
                children,
            )
        if node.kind == "product":
            return ("product", _node_data(node, "role"), children)
        if node.kind == "stream":
            terminal = None if terminal_streams is None else terminal_streams.get(node.id)
            # Use (False, ()) / (True, sig) so non-terminal and terminal streams
            # are always sortable even when mixed as siblings (e.g. two recycle
            # mixer units both connected to the feed node).
            terminal_key = (False, ()) if terminal is None else (True, terminal)
            return ("stream", terminal_key, children)
        return (node.kind, _normalise_node_data(node.data), children)

    roots = tuple(sorted(node_signature(node_id) for node_id in graph.root_node_ids))
    return roots


def _terminal_stream_signatures(
    state: Any,
    graph: ProcessGraph,
    float_digits: int,
) -> dict[str, tuple]:
    terminal_streams: dict[str, tuple] = {}
    for stream in getattr(state, "open_streams", ()):
        node_id = get_stream_node_id(graph, stream.id)
        if node_id is not None:
            terminal_streams[node_id] = stream_signature(stream, float_digits)
    for product in getattr(state, "products", ()):
        stream = getattr(product, "stream")
        node_id = get_stream_node_id(graph, stream.id)
        if node_id is not None:
            terminal_streams[node_id] = stream_signature(stream, float_digits)
    return terminal_streams


def _minimal_topology_signature(state: Any) -> tuple:
    open_items = tuple(
        sorted(("open", tuple(stream.history)) for stream in getattr(state, "open_streams", ()))
    )
    product_items = tuple(
        sorted(
            (
                "product",
                getattr(product, "role", ""),
                tuple(getattr(product, "stream").history),
            )
            for product in getattr(state, "products", ())
        )
    )
    return (open_items, product_items)


def _minimal_state_signature(state: Any, float_digits: int) -> tuple:
    open_items = tuple(
        sorted(stream_signature(stream, float_digits) for stream in getattr(state, "open_streams", ()))
    )
    product_items = tuple(
        sorted(
            (
                getattr(product, "role", ""),
                stream_signature(getattr(product, "stream"), float_digits),
            )
            for product in getattr(state, "products", ())
        )
    )
    return (open_items, product_items)


def _node_data(node: ProcessNode, key: str) -> Any:
    for item_key, value in node.data:
        if item_key == key:
            return value
    return None


def _normalise_node_data(data: tuple[tuple[str, Any], ...]) -> tuple:
    return tuple(sorted(data))


def _round_value(value: Any, digits: int) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _hash_signature(signature: tuple) -> str:
    payload = repr(signature).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
