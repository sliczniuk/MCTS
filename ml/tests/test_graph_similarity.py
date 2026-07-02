from __future__ import annotations

from ml import (
    MCTSConfig,
    MCTSResult,
    SearchState,
    StreamState,
    UnitAction,
    action_signature,
    append_unit_operation,
    process_graph_fingerprints,
    process_graph_from_feed,
    process_graph_similarity,
    rank_similar_process_graphs,
    stream_composition_cosine,
    stream_condition_cosine,
    stream_condition_vector,
    suspicious_similarity_report,
    terminal_stream_cosine_profile,
    terminal_streams_for_similarity,
    topology_feature_cosine,
)


def _stream(stream_id: str) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )


def _state_with_outlet(outlet: StreamState, delta_t: float = 10.0) -> SearchState:
    feed = _stream("Feed")
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=delta_t)
    graph = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        action,
        ((outlet, "out"),),
        action_signature(action),
    )
    return SearchState(
        open_streams=(outlet,),
        unit_sequence=(action,),
        process_graph=graph,
    )


def _hx_graph(feed_label: str, outlet_label: str, delta_t: float):
    feed = _stream(feed_label)
    outlet = _stream(outlet_label)
    action = UnitAction(kind="hx", stream_id=feed_label, delta_T_K=delta_t)
    return append_unit_operation(
        process_graph_from_feed(feed),
        feed_label,
        action,
        ((outlet, "out"),),
        action_signature(action),
    )


def test_process_graph_similarity_is_one_for_label_renamed_same_topology():
    left = _hx_graph("Feed", "Feed_hx_p10_1", 10.0)
    right = _hx_graph("RenamedFeed", "RenamedOutlet", 10.0)

    score = process_graph_similarity(left, right)

    assert score.overall == 1.0
    assert score.branch_similarity == 1.0
    assert score.unit_similarity == 1.0
    assert score.topology_similarity == 1.0


def test_process_graph_similarity_drops_when_action_parameters_differ():
    left = _hx_graph("Feed", "Feed_hx_p10_1", 10.0)
    right = _hx_graph("Feed", "Feed_hx_p20_1", 20.0)

    score = process_graph_similarity(left, right)

    assert 0.0 < score.overall < 1.0
    assert score.unit_similarity == 0.0
    assert score.edge_role_similarity == 1.0


def test_process_graph_similarity_is_invariant_to_independent_branch_order():
    feed = _stream("Feed")
    split = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    distillate = _stream("D")
    bottoms = _stream("B")
    distillate_hx = UnitAction(kind="hx", stream_id="D", delta_T_K=10.0)
    bottoms_hx = UnitAction(kind="hx", stream_id="B", delta_T_K=20.0)
    distillate_out = _stream("D_hot")
    bottoms_out = _stream("B_hot")
    base = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        split,
        ((distillate, "distillate"), (bottoms, "bottoms")),
        action_signature(split),
    )

    left = append_unit_operation(
        base,
        "D",
        distillate_hx,
        ((distillate_out, "out"),),
        action_signature(distillate_hx),
    )
    left = append_unit_operation(
        left,
        "B",
        bottoms_hx,
        ((bottoms_out, "out"),),
        action_signature(bottoms_hx),
    )
    right = append_unit_operation(
        base,
        "B",
        bottoms_hx,
        ((bottoms_out, "out"),),
        action_signature(bottoms_hx),
    )
    right = append_unit_operation(
        right,
        "D",
        distillate_hx,
        ((distillate_out, "out"),),
        action_signature(distillate_hx),
    )

    assert process_graph_similarity(left, right).overall == 1.0


def test_process_graph_fingerprints_accept_search_state():
    graph = _hx_graph("Feed", "Outlet", 10.0)
    state = SearchState(open_streams=(_stream("Outlet"),), process_graph=graph)

    features = process_graph_fingerprints(state)

    assert features["units"] == (("unit", ("hx", 10.0)),)
    assert features["edge_roles"] == (("feed",), ("out",))


def test_terminal_stream_similarity_changes_with_temperature():
    left = _state_with_outlet(_stream("Outlet"))
    hotter = StreamState(
        id="RenamedOutlet",
        temperature_K=330.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )
    right = _state_with_outlet(hotter)

    score = process_graph_similarity(left, right)

    assert score.topology_similarity == 1.0
    assert score.terminal_stream_similarity < 1.0
    assert score.overall < 1.0


def test_terminal_stream_similarity_changes_with_composition():
    left = _state_with_outlet(_stream("Outlet"))
    richer = StreamState(
        id="RenamedOutlet",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.9, "n-butane": 0.1},
    )
    right = _state_with_outlet(richer)

    score = process_graph_similarity(left, right)

    assert score.topology_similarity == 1.0
    assert score.terminal_stream_similarity < 1.0


def test_objective_similarity_compares_separation_quality_with_feed_stream():
    feed = StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=2.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )
    perfect = SearchState(
        open_streams=(
            StreamState("C3", 300.0, 101325.0, 1.0, {"propane": 1.0, "n-butane": 0.0}),
            StreamState("C4", 300.0, 101325.0, 1.0, {"propane": 0.0, "n-butane": 1.0}),
        ),
    )
    mixed = SearchState(
        open_streams=(
            StreamState("M1", 300.0, 101325.0, 1.0, {"propane": 0.5, "n-butane": 0.5}),
            StreamState("M2", 300.0, 101325.0, 1.0, {"propane": 0.5, "n-butane": 0.5}),
        ),
    )

    score = process_graph_similarity(
        perfect,
        mixed,
        feed_stream=feed,
        components=("propane", "n-butane"),
    )

    assert score.objective_similarity < 1.0
    assert score.overall < 1.0
    assert not any("separation objective unavailable" in item for item in score.limitations)


def test_objective_similarity_records_limitation_without_feed_stream():
    left = _state_with_outlet(_stream("Outlet"))
    right = _state_with_outlet(_stream("OtherOutlet"))

    score = process_graph_similarity(left, right)

    assert score.objective_similarity == 1.0
    assert "separation objective unavailable without feed stream and terminal streams" in score.limitations


def test_terminal_streams_for_similarity_accepts_mcts_result():
    state = _state_with_outlet(_stream("Outlet"))
    result = MCTSResult(
        best_state=state,
        best_reward=1.0,
        best_sequence=state.unit_sequence,
        product=None,
        iterations=1,
    )

    assert terminal_streams_for_similarity(result)[0].id == "Outlet"


def test_stream_cosine_diagnostics_report_composition_and_conditions_separately():
    config = MCTSConfig(
        min_temperature_K=50.0,
        max_temperature_K=500.0,
        min_pressure_Pa=100000.0,
        max_pressure_Pa=800000.0,
    )
    feed = StreamState(
        id="Feed",
        temperature_K=330.0,
        pressure_Pa=500000.0,
        molar_flow_mols=3.0,
        composition={"propane": 1 / 3, "n-butane": 1 / 3, "n-pentane": 1 / 3},
    )
    base = StreamState(
        id="A",
        temperature_K=330.0,
        pressure_Pa=500000.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.33, "n-butane": 0.34, "n-pentane": 0.33},
    )
    different_composition = StreamState(
        id="B",
        temperature_K=330.0,
        pressure_Pa=500000.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.95, "n-butane": 0.03, "n-pentane": 0.02},
    )
    different_conditions = StreamState(
        id="C",
        temperature_K=380.0,
        pressure_Pa=200000.0,
        molar_flow_mols=0.2,
        composition=dict(base.composition),
    )

    assert stream_condition_vector(base, config, feed) == stream_condition_vector(base, config=config, feed_stream=feed)
    assert stream_composition_cosine(base, different_composition) < 0.7
    assert stream_condition_cosine(base, different_composition, config, feed) == 1.0
    assert stream_composition_cosine(base, different_conditions) == 1.0
    assert stream_condition_cosine(base, different_conditions, config, feed) < 0.95


def test_terminal_stream_cosine_profile_matches_terminal_stream_sets():
    config = MCTSConfig(min_pressure_Pa=100000.0, max_pressure_Pa=800000.0)
    feed = _stream("Feed")
    left = _state_with_outlet(_stream("Outlet"))
    right = _state_with_outlet(_stream("OtherOutlet"))

    profile = terminal_stream_cosine_profile(left, right, config=config, feed_stream=feed)

    assert profile["matched_pair_count"] == 1
    assert profile["unmatched_stream_count"] == 0
    assert profile["minimum_composition_cosine"] == 1.0
    assert profile["minimum_condition_cosine"] == 1.0
    assert profile["matched_pairs"][0]["left_stream_id"] == "Outlet"


def test_topology_feature_cosine_is_high_for_same_feature_counts():
    reference = _hx_graph("Feed", "Outlet", 10.0)
    renamed = _hx_graph("OtherFeed", "OtherOutlet", 10.0)
    different = _hx_graph("Feed", "Outlet", 20.0)

    assert topology_feature_cosine(reference, renamed) == 1.0
    assert topology_feature_cosine(reference, different) < 1.0


def test_suspicious_similarity_report_classifies_duplicate_like_states():
    config = MCTSConfig(min_pressure_Pa=100000.0, max_pressure_Pa=800000.0)
    feed = _stream("Feed")
    left = _state_with_outlet(_stream("Outlet"))
    right = _state_with_outlet(_stream("RenamedOutlet"))

    report = suspicious_similarity_report(left, right, config=config, feed_stream=feed)

    assert report["classification"] == "exact_duplicate"
    assert report["exact_duplicate"] is True
    assert report["same_topology_hash"] is True
    assert report["similar_streams"] is True


def test_suspicious_similarity_report_distinguishes_same_topology_different_streams():
    config = MCTSConfig(min_pressure_Pa=100000.0, max_pressure_Pa=800000.0)
    feed = _stream("Feed")
    left = _state_with_outlet(_stream("Outlet"))
    hotter = StreamState(
        id="Hotter",
        temperature_K=430.0,
        pressure_Pa=800000.0,
        molar_flow_mols=0.1,
        composition={"propane": 0.95, "n-butane": 0.05},
    )
    right = _state_with_outlet(hotter)

    report = suspicious_similarity_report(left, right, config=config, feed_stream=feed)

    assert report["classification"] == "similar_topology_different_streams"
    assert report["same_topology_hash"] is True
    assert report["similar_streams"] is False


def test_rank_similar_process_graphs_orders_by_overall_score():
    reference = _hx_graph("Feed", "Outlet", 10.0)
    exact = _hx_graph("F0", "Out0", 10.0)
    different = _hx_graph("Feed", "Outlet", 20.0)

    rows = rank_similar_process_graphs(
        reference,
        {
            "different": different,
            "exact": exact,
        },
    )

    assert [row["label"] for row in rows] == ["exact", "different"]
    assert rows[0]["overall"] == 1.0
    assert rows[0]["topology_similarity"] == 1.0
    assert rows[0]["terminal_stream_similarity"] == 1.0
    assert rows[1]["overall"] < 1.0
