from __future__ import annotations

from ml import (
    ProcessEdge,
    ProcessGraph,
    ProcessNode,
    StreamState,
    UnitAction,
    action_signature,
    append_product_assignment,
    append_unit_operation,
    process_graph_diagnostics,
    process_graph_from_feed,
)
from ml.process_graph import get_stream_node_id


def _stream(stream_id: str) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )


def test_process_graph_from_feed_creates_stable_stream_root():
    graph = process_graph_from_feed(_stream("Feed"))

    assert graph.root_node_ids == ("S0",)
    assert graph.nodes[0].id == "S0"
    assert graph.nodes[0].kind == "stream"
    assert graph.nodes[0].label == "Feed"
    assert get_stream_node_id(graph, "Feed") == "S0"


def test_append_unit_operation_creates_feed_and_out_edges():
    feed = _stream("Feed")
    outlet = _stream("Feed_hx_p10_1")
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)

    graph = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        action,
        ((outlet, "out"),),
        action_signature(action),
    )

    assert [node.kind for node in graph.nodes] == ["stream", "unit", "stream"]
    assert [(edge.source, edge.target, edge.role) for edge in graph.edges] == [
        ("S0", "U1", "feed"),
        ("U1", "S2", "out"),
    ]
    assert get_stream_node_id(graph, "Feed_hx_p10_1") == "S2"


def test_append_flash_and_distillation_roles_are_explicit_edges():
    feed = _stream("Feed")
    flash = UnitAction(kind="flash", stream_id="Feed")
    distillation = UnitAction(
        kind="distillation",
        stream_id="Feed_liquid",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )

    graph = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        flash,
        ((_stream("Feed_vapor"), "vapor"), (_stream("Feed_liquid"), "liquid")),
        action_signature(flash),
    )
    graph = append_unit_operation(
        graph,
        "Feed_liquid",
        distillation,
        (
            (_stream("ColumnD"), "distillate"),
            (_stream("ColumnB"), "bottoms"),
        ),
        action_signature(distillation),
    )

    roles = [edge.role for edge in graph.edges]
    assert roles == ["feed", "vapor", "liquid", "feed", "distillate", "bottoms"]


def test_append_product_assignment_preserves_product_role():
    graph = process_graph_from_feed(_stream("Feed"))
    graph = append_product_assignment(graph, "Feed", "PropaneProduct")

    assert graph.nodes[-1].kind == "product"
    assert graph.nodes[-1].label == "PropaneProduct"
    assert graph.edges[-1].role == "product"


def test_process_graph_diagnostics_accept_valid_graph():
    feed = _stream("Feed")
    outlet = _stream("Feed_hx_p10_1")
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    graph = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        action,
        ((outlet, "out"),),
        action_signature(action),
    )

    assert process_graph_diagnostics(graph, open_streams=(outlet,)) == []


def test_process_graph_diagnostics_report_structural_issues():
    graph = ProcessGraph(
        nodes=(
            ProcessNode(id="S0", kind="stream", label="Feed"),
            ProcessNode(id="U1", kind="unit", label="hx", data=(("action_kind", "hx"),)),
            ProcessNode(id="S2", kind="stream", label="Outlet"),
        ),
        edges=(
            ProcessEdge(source="S0", target="U1", role="feed"),
            ProcessEdge(source="U1", target="S2", role="vapor"),
            ProcessEdge(source="missing", target="S2", role="out"),
        ),
        root_node_ids=("S0",),
        next_node_index=3,
        stream_node_ids=(("Feed", "S0"),),
    )

    issues = process_graph_diagnostics(graph, open_streams=(_stream("Outlet"),))
    codes = {issue["code"] for issue in issues}

    assert "missing_edge_source" in codes
    assert "unexpected_unit_output_role" in codes
    assert "invalid_single_output_roles" in codes
    assert "unmapped_open_stream" in codes
