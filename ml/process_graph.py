"""Explicit process graph records for search-side flowsheet states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import StreamState


@dataclass(frozen=True)
class ProcessNode:
    """Node in a search-side process graph."""

    id: str
    kind: str
    label: str
    data: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ProcessEdge:
    """Directed, role-labelled edge in a search-side process graph."""

    source: str
    target: str
    role: str


@dataclass(frozen=True)
class ProcessGraph:
    """Immutable process graph used for topology/state identity."""

    nodes: tuple[ProcessNode, ...] = ()
    edges: tuple[ProcessEdge, ...] = ()
    root_node_ids: tuple[str, ...] = ()
    next_node_index: int = 0
    stream_node_ids: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def empty(cls) -> "ProcessGraph":
        """Return an empty process graph."""
        return cls()


def process_graph_from_feed(feed_stream: StreamState) -> ProcessGraph:
    """Create a process graph with one feed stream root.

    Args:
        feed_stream: Feed stream represented as the root material stream.

    Returns:
        Process graph containing one stream node with internal id ``S0``.

    Example:
        graph = process_graph_from_feed(feed)
    """
    return append_stream_root(ProcessGraph.empty(), feed_stream, role="feed")


def append_stream_root(
    graph: ProcessGraph,
    stream: StreamState,
    role: str = "open",
) -> ProcessGraph:
    """Append a root stream node unless its display label already exists.

    Args:
        graph: Existing process graph.
        stream: Stream to add as a root node.
        role: Semantic role stored on the stream node data.

    Returns:
        Updated process graph. If the stream label already exists, the input
        graph is returned unchanged.

    Example:
        graph = append_stream_root(graph, side_feed, role="feed")
    """
    if get_stream_node_id(graph, stream.id) is not None:
        return graph
    node_id = _next_node_id("S", graph.next_node_index)
    node = ProcessNode(
        id=node_id,
        kind="stream",
        label=stream.id,
        data=(("role", role),),
    )
    return ProcessGraph(
        nodes=graph.nodes + (node,),
        edges=graph.edges,
        root_node_ids=graph.root_node_ids + (node_id,),
        next_node_index=graph.next_node_index + 1,
        stream_node_ids=graph.stream_node_ids + ((stream.id, node_id),),
    )


def append_unit_operation(
    graph: ProcessGraph,
    input_stream_id: str,
    action: Any,
    output_streams_with_roles: tuple[tuple[StreamState, str], ...],
    action_signature_value: tuple | None = None,
) -> ProcessGraph:
    """Append a unit node and its output stream nodes.

    Args:
        graph: Existing process graph.
        input_stream_id: Current display label of the consumed stream.
        action: UnitAction-like object that produced the outputs.
        output_streams_with_roles: Output streams paired with semantic edge
            roles such as ``out``, ``vapor``, ``liquid``, ``distillate``, or
            ``bottoms``.
        action_signature_value: Optional canonical action signature. Passing
            this avoids importing graph identity here.

    Returns:
        Updated immutable process graph.

    Example:
        graph = append_unit_operation(graph, "Feed", action, ((outlet, "out"),))
    """
    input_node_id = get_stream_node_id(graph, input_stream_id)
    if input_node_id is None:
        raise ValueError(f"Input stream '{input_stream_id}' is not present in graph.")

    unit_index = graph.next_node_index
    unit_node_id = _next_node_id("U", unit_index)
    unit_node = ProcessNode(
        id=unit_node_id,
        kind="unit",
        label=_action_label(action),
        data=(
            ("action_kind", getattr(action, "kind", "")),
            ("action_signature", action_signature_value or _raw_action_signature(action)),
        ),
    )
    edges = graph.edges + (
        ProcessEdge(source=input_node_id, target=unit_node_id, role="feed"),
    )
    nodes = graph.nodes + (unit_node,)
    stream_node_ids = graph.stream_node_ids
    next_index = unit_index + 1

    for output_stream, role in output_streams_with_roles:
        stream_node_id = _next_node_id("S", next_index)
        stream_node = ProcessNode(
            id=stream_node_id,
            kind="stream",
            label=output_stream.id,
            data=(("role", role),),
        )
        nodes += (stream_node,)
        edges += (ProcessEdge(source=unit_node_id, target=stream_node_id, role=role),)
        stream_node_ids += ((output_stream.id, stream_node_id),)
        next_index += 1

    return ProcessGraph(
        nodes=nodes,
        edges=edges,
        root_node_ids=graph.root_node_ids,
        next_node_index=next_index,
        stream_node_ids=stream_node_ids,
    )


def append_mixer_unit(
    graph: ProcessGraph,
    recycle_stream_id: str,
    feed_stream_id: str,
    action: Any,
    output_streams_with_roles: tuple[tuple[StreamState, str], ...],
    action_signature_value: tuple | None = None,
) -> ProcessGraph:
    """Append a two-inlet mixer unit node (recycle + feed) and its output stream nodes.

    Args:
        graph: Existing process graph.
        recycle_stream_id: Display label of the stream being recycled (consumed).
        feed_stream_id: Display label of the original feed stream (second inlet).
        action: UnitAction-like object that produced the outputs.
        output_streams_with_roles: Output streams paired with semantic edge roles.
        action_signature_value: Optional canonical action signature.

    Returns:
        Updated immutable process graph.

    Example:
        graph = append_mixer_unit(graph, "impure_stream", "Feed", action, ((mixed, "open"),))
    """
    recycle_node_id = get_stream_node_id(graph, recycle_stream_id)
    if recycle_node_id is None:
        raise ValueError(f"Recycle stream '{recycle_stream_id}' is not present in graph.")
    feed_node_id = get_stream_node_id(graph, feed_stream_id)
    if feed_node_id is None:
        raise ValueError(f"Feed stream '{feed_stream_id}' is not present in graph.")

    unit_index = graph.next_node_index
    unit_node_id = _next_node_id("U", unit_index)
    unit_node = ProcessNode(
        id=unit_node_id,
        kind="unit",
        label=_action_label(action),
        data=(
            ("action_kind", getattr(action, "kind", "")),
            ("action_signature", action_signature_value or _raw_action_signature(action)),
        ),
    )
    edges = graph.edges + (
        ProcessEdge(source=recycle_node_id, target=unit_node_id, role="recycle"),
        ProcessEdge(source=feed_node_id, target=unit_node_id, role="feed"),
    )
    nodes = graph.nodes + (unit_node,)
    stream_node_ids = graph.stream_node_ids
    next_index = unit_index + 1

    for output_stream, role in output_streams_with_roles:
        stream_node_id = _next_node_id("S", next_index)
        stream_node = ProcessNode(
            id=stream_node_id,
            kind="stream",
            label=output_stream.id,
            data=(("role", role),),
        )
        nodes += (stream_node,)
        edges += (ProcessEdge(source=unit_node_id, target=stream_node_id, role=role),)
        stream_node_ids += ((output_stream.id, stream_node_id),)
        next_index += 1

    return ProcessGraph(
        nodes=nodes,
        edges=edges,
        root_node_ids=graph.root_node_ids,
        next_node_index=next_index,
        stream_node_ids=stream_node_ids,
    )


def append_product_assignment(
    graph: ProcessGraph,
    stream_id: str,
    role: str,
) -> ProcessGraph:
    """Append a product assignment node linked to an existing stream.

    Args:
        graph: Existing process graph.
        stream_id: Current display label of the accepted stream.
        role: Product role assigned to the stream.

    Returns:
        Updated graph with a product node and product edge.

    Example:
        graph = append_product_assignment(graph, "Feed_liquid", "Product")
    """
    stream_node_id = get_stream_node_id(graph, stream_id)
    if stream_node_id is None:
        raise ValueError(f"Product stream '{stream_id}' is not present in graph.")
    product_node_id = _next_node_id("P", graph.next_node_index)
    product_node = ProcessNode(
        id=product_node_id,
        kind="product",
        label=role,
        data=(("role", role),),
    )
    return ProcessGraph(
        nodes=graph.nodes + (product_node,),
        edges=graph.edges
        + (ProcessEdge(source=stream_node_id, target=product_node_id, role="product"),),
        root_node_ids=graph.root_node_ids,
        next_node_index=graph.next_node_index + 1,
        stream_node_ids=graph.stream_node_ids,
    )


def get_stream_node_id(graph: ProcessGraph, stream_label: str) -> str | None:
    """Look up the graph node id for the current display stream label.

    Args:
        graph: Process graph containing stream label mappings.
        stream_label: Current display stream identifier, e.g. ``Feed_hx_p10_1``.

    Returns:
        Internal graph stream node id, or None when the label is not present.

    Example:
        node_id = get_stream_node_id(graph, stream.id)
    """
    for label, node_id in reversed(graph.stream_node_ids):
        if label == stream_label:
            return node_id
    return None


def process_graph_diagnostics(
    graph: ProcessGraph,
    open_streams: tuple[StreamState, ...] = (),
    products: tuple[Any, ...] = (),
) -> list[dict[str, str]]:
    """Return structural diagnostics for a process graph.

    Args:
        graph: Process graph to validate.
        open_streams: Optional terminal open streams that should map to stream
            nodes in the graph.
        products: Optional product assignments whose streams should map to
            stream nodes and product roles in the graph.

    Returns:
        List of diagnostic dictionaries. An empty list means no structural
        issues were found.

    Example:
        issues = process_graph_diagnostics(state.process_graph, state.open_streams, state.products)
    """
    issues: list[dict[str, str]] = []
    nodes_by_id: dict[str, ProcessNode] = {}
    duplicate_node_ids: set[str] = set()
    for node in graph.nodes:
        if node.id in nodes_by_id:
            duplicate_node_ids.add(node.id)
        nodes_by_id[node.id] = node
    for node_id in sorted(duplicate_node_ids):
        _add_issue(issues, "error", "duplicate_node_id", f"Duplicate node id '{node_id}'.", node_id)

    for root_id in graph.root_node_ids:
        node = nodes_by_id.get(root_id)
        if node is None:
            _add_issue(issues, "error", "missing_root", f"Root node '{root_id}' is missing.", root_id)
        elif node.kind != "stream":
            _add_issue(
                issues,
                "error",
                "invalid_root_kind",
                f"Root node '{root_id}' has kind '{node.kind}', expected 'stream'.",
                root_id,
            )

    incoming: dict[str, list[ProcessEdge]] = {node.id: [] for node in graph.nodes}
    outgoing: dict[str, list[ProcessEdge]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes_by_id:
            _add_issue(
                issues,
                "error",
                "missing_edge_source",
                f"Edge source '{edge.source}' is missing.",
                edge.source,
            )
        else:
            outgoing.setdefault(edge.source, []).append(edge)
        if edge.target not in nodes_by_id:
            _add_issue(
                issues,
                "error",
                "missing_edge_target",
                f"Edge target '{edge.target}' is missing.",
                edge.target,
            )
        else:
            incoming.setdefault(edge.target, []).append(edge)

    for label, node_id in graph.stream_node_ids:
        node = nodes_by_id.get(node_id)
        if node is None:
            _add_issue(
                issues,
                "error",
                "missing_stream_mapping_target",
                f"Stream label '{label}' maps to missing node '{node_id}'.",
                node_id,
            )
        elif node.kind != "stream":
            _add_issue(
                issues,
                "error",
                "invalid_stream_mapping_target",
                f"Stream label '{label}' maps to non-stream node '{node_id}'.",
                node_id,
            )

    for node in graph.nodes:
        if node.kind == "unit":
            feed_edges = [edge for edge in incoming.get(node.id, ()) if edge.role == "feed"]
            if len(feed_edges) != 1:
                _add_issue(
                    issues,
                    "error",
                    "invalid_unit_feed_count",
                    f"Unit node '{node.id}' has {len(feed_edges)} feed edges, expected 1.",
                    node.id,
                )
            _diagnose_unit_outputs(issues, node, outgoing.get(node.id, ()))
        elif node.kind == "product":
            product_edges = [edge for edge in incoming.get(node.id, ()) if edge.role == "product"]
            if len(product_edges) != 1:
                _add_issue(
                    issues,
                    "error",
                    "invalid_product_edge_count",
                    f"Product node '{node.id}' has {len(product_edges)} product edges, expected 1.",
                    node.id,
                )
        elif node.kind == "stream" and node.id not in graph.root_node_ids:
            material_edges = [
                edge for edge in incoming.get(node.id, ()) if edge.role != "product"
            ]
            if len(material_edges) != 1:
                _add_issue(
                    issues,
                    "error",
                    "invalid_stream_source_count",
                    f"Stream node '{node.id}' has {len(material_edges)} material source edges, expected 1.",
                    node.id,
                )

    for stream in open_streams:
        if get_stream_node_id(graph, stream.id) is None:
            _add_issue(
                issues,
                "error",
                "unmapped_open_stream",
                f"Open stream '{stream.id}' is not mapped to a graph stream node.",
                stream.id,
            )

    product_roles = {
        node.label for node in graph.nodes if node.kind == "product"
    }
    for product in products:
        stream = getattr(product, "stream", None)
        role = getattr(product, "role", "")
        if stream is not None and get_stream_node_id(graph, stream.id) is None:
            _add_issue(
                issues,
                "error",
                "unmapped_product_stream",
                f"Product stream '{stream.id}' is not mapped to a graph stream node.",
                stream.id,
            )
        if role and role not in product_roles:
            _add_issue(
                issues,
                "error",
                "missing_product_role",
                f"Product role '{role}' is not represented by a product node.",
                role,
            )

    return issues


def _next_node_id(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def _action_label(action: Any) -> str:
    kind = getattr(action, "kind", "")
    if kind == "distillation":
        return f"distillation:{getattr(action, 'light_key', None)}/{getattr(action, 'heavy_key', None)}"
    return str(kind)


def _raw_action_signature(action: Any) -> tuple:
    return tuple(
        sorted(
            (name, value)
            for name, value in getattr(action, "__dict__", {}).items()
            if name != "stream_id" and value is not None
        )
    )


def _diagnose_unit_outputs(
    issues: list[dict[str, str]],
    node: ProcessNode,
    output_edges: tuple[ProcessEdge, ...] | list[ProcessEdge],
) -> None:
    action_kind = _node_data(node, "action_kind")
    roles = [edge.role for edge in output_edges]
    if not roles:
        _add_issue(
            issues,
            "error",
            "unit_without_outputs",
            f"Unit node '{node.id}' has no output edges.",
            node.id,
        )
        return
    expected_roles = {
        "hx": {"out"},
        "compressor": {"out"},
        "pump": {"out"},
        "valve": {"out"},
        "flash": {"vapor", "liquid"},
        "distillation": {"distillate", "bottoms"},
    }.get(action_kind)
    if expected_roles is None:
        return
    unexpected = sorted(set(roles) - expected_roles)
    if unexpected:
        _add_issue(
            issues,
            "error",
            "unexpected_unit_output_role",
            f"Unit node '{node.id}' has unexpected output roles: {', '.join(unexpected)}.",
            node.id,
        )
    if action_kind in {"hx", "compressor", "pump", "valve"} and roles != ["out"]:
        _add_issue(
            issues,
            "error",
            "invalid_single_output_roles",
            f"Unit node '{node.id}' has output roles {roles}, expected ['out'].",
            node.id,
        )


def _node_data(node: ProcessNode, key: str) -> Any:
    for item_key, value in node.data:
        if item_key == key:
            return value
    return None


def _add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    message: str,
    node_id: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "node_id": node_id,
        }
    )
