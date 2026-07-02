from __future__ import annotations

from ml import (
    MCTSDiagnostics,
    MCTSConfig,
    MCTSResult,
    StreamState,
    UnitAction,
    action_signature,
    append_unit_operation,
    build_pr_flasher,
    mcts_diagnostics_table,
    mcts_replay_graph,
    process_graph_cluster_table,
    process_graph_diagnostics_table,
    process_graph_similarity_pairs,
    process_graph_similarity_summary,
    process_graph_from_feed,
    stream_table,
    stream_table_from_state,
)
from ml.mcts import SearchState


COMPOUNDS = ["nitrogen", "propane", "n-butane"]


def _feed() -> StreamState:
    return StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=500000.0,
        molar_flow_mols=2.0,
        composition={"nitrogen": 0.1, "propane": 0.45, "n-butane": 0.45},
    )


def _rows(table):
    if hasattr(table, "to_dict"):
        return table.to_dict("records")
    return table


def test_stream_table_reports_components_and_roles_without_pandas_requirement():
    feed = _feed()
    table = stream_table(
        [feed],
        roles={"Feed": "feed"},
        as_dataframe=False,
    )

    assert table[0]["id"] == "Feed"
    assert table[0]["role"] == "feed"
    assert table[0]["dominant_component"] == "propane"
    assert table[0]["x_nitrogen"] == 0.1


def test_stream_table_from_state_includes_open_and_product_roles():
    feed = _feed()
    state = SearchState(open_streams=(feed,))

    table = stream_table_from_state(state, as_dataframe=False)

    assert table[0]["id"] == "Feed"
    assert table[0]["role"] == "open"


def test_mcts_replay_graph_represents_forked_distillation_outputs():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        enable_distillation_actions=True,
        require_flash_liquid_product=True,
        distillation_max_theoretical_stages=120.0,
    )
    sequence = (
        UnitAction(
            kind="distillation",
            stream_id="Feed",
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
            reflux_ratio_multiplier=1.5,
        ),
    )

    graph = mcts_replay_graph(
        _feed(),
        provider,
        config,
        sequence,
        as_dataframe=False,
    )

    edge_rows = _rows(graph.edges)
    stream_rows = _rows(graph.streams)

    assert graph.errors == ()
    assert "U01: distillation" in graph.text
    assert "distillate" in graph.text
    assert "bottoms" in graph.text
    assert "[feed]" in graph.text
    assert "[distillate]" in graph.text
    assert "[bottoms]" in graph.text
    assert any(row["edge"] == "distillate" for row in edge_rows)
    assert any(row["edge"] == "bottoms" for row in edge_rows)
    assert len(graph.final_state.open_streams) == 2
    assert {row["role"] for row in stream_rows} >= {"feed", "open"}


def test_mcts_diagnostics_table_compares_runs_without_pandas_requirement():
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    baseline = MCTSResult(
        best_state=state,
        best_reward=1.0,
        best_sequence=(action,),
        product=None,
        iterations=10,
        progress=({"elapsed_s": 2.0},),
        diagnostics=MCTSDiagnostics(n_expanded_nodes=10),
    )
    cached = MCTSResult(
        best_state=state,
        best_reward=1.5,
        best_sequence=(action,),
        product=None,
        iterations=10,
        progress=({"elapsed_s": 1.0},),
        diagnostics=MCTSDiagnostics(
            n_expanded_nodes=8,
            n_duplicate_states_skipped=2,
            duplicate_skip_rate=0.2,
            n_apply_action_cache_hits=4,
            n_apply_action_cache_misses=2,
            apply_action_cache_hit_rate=4 / 6,
            n_distillation_result_cache_hits=3,
            n_distillation_result_cache_misses=1,
            distillation_result_cache_hit_rate=0.75,
            n_valid_action_calls=10,
            n_valid_action_cache_hits=3,
            n_valid_action_cache_misses=2,
            valid_action_cache_hit_rate=0.6,
            n_valid_action_cache_entries=2,
            n_valid_actions_generated_total=12,
            max_valid_actions_generated_per_call=7,
            valid_actions_generated_by_kind=(("hx", 4), ("distillation", 8)),
            n_relative_volatility_cache_hits=5,
            n_relative_volatility_cache_misses=1,
            relative_volatility_cache_hit_rate=5 / 6,
            n_relative_volatility_cache_entries=1,
        ),
    )

    rows = mcts_diagnostics_table(
        {"baseline": baseline, "cached": cached},
        baseline_label="baseline",
        as_dataframe=False,
    )

    assert rows[0]["label"] == "baseline"
    assert rows[1]["label"] == "cached"
    assert rows[1]["reward_delta_vs_baseline"] == 0.5
    assert rows[1]["elapsed_ratio_vs_baseline"] == 0.5
    assert rows[1]["expanded_delta_vs_baseline"] == -2
    assert rows[1]["sequence_kinds"] == ("hx",)
    assert rows[1]["n_apply_action_cache_hits"] == 4
    assert rows[1]["n_distillation_result_cache_hits"] == 3
    assert rows[1]["n_valid_action_calls"] == 10
    assert rows[1]["n_valid_action_cache_hits"] == 3
    assert rows[1]["valid_action_cache_hit_rate"] == 0.6
    assert rows[1]["valid_actions_generated_by_kind"] == (
        ("hx", 4),
        ("distillation", 8),
    )
    assert rows[1]["n_relative_volatility_cache_hits"] == 5
    assert rows[1]["relative_volatility_cache_hit_rate"] == 5 / 6


def test_process_graph_diagnostics_table_summarises_states_and_similarity():
    feed = _feed()
    outlet = StreamState(
        id="Feed_hx_p10_1",
        temperature_K=310.0,
        pressure_Pa=feed.pressure_Pa,
        molar_flow_mols=feed.molar_flow_mols,
        composition=dict(feed.composition),
        history=("hx",),
    )
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    graph = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        action,
        ((outlet, "out"),),
        action_signature(action),
    )
    state = SearchState(
        open_streams=(outlet,),
        unit_sequence=(action,),
        process_graph=graph,
    )
    result = MCTSResult(
        best_state=state,
        best_reward=1.0,
        best_sequence=(action,),
        product=None,
        iterations=5,
    )

    rows = process_graph_diagnostics_table(
        {"state": state, "result": result},
        reference=state,
        as_dataframe=False,
    )

    assert rows[0]["label"] == "state"
    assert rows[0]["n_nodes"] == 3
    assert rows[0]["n_edges"] == 2
    assert rows[0]["n_stream_nodes"] == 2
    assert rows[0]["n_unit_nodes"] == 1
    assert rows[0]["graph_issue_count"] == 0
    assert rows[0]["similarity_overall"] == 1.0
    assert rows[0]["similarity_topology"] == 1.0
    assert rows[0]["similarity_terminal_streams"] == 1.0
    assert rows[0]["similarity_objective"] == 1.0
    assert "similarity_limitations" in rows[0]
    assert rows[1]["best_reward"] == 1.0
    assert rows[1]["sequence_kinds"] == ("hx",)


def test_process_graph_cluster_table_groups_exact_hashes_and_nearest_neighbors():
    feed = _feed()
    state_a = _hx_state(feed, outlet_id="Feed_hx_p10_1", delta_T_K=10.0)
    state_b = _hx_state(feed, outlet_id="Renamed_hx_outlet", delta_T_K=10.0)
    state_c = _hx_state(feed, outlet_id="Feed_hx_p20_1", delta_T_K=20.0)

    rows = process_graph_cluster_table(
        {"a": state_a, "b": state_b, "c": state_c},
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
        as_dataframe=False,
    )

    assert [row["label"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["topology_group_id"] == rows[1]["topology_group_id"]
    assert rows[0]["topology_group_size"] == 2
    assert rows[0]["state_group_id"] == rows[1]["state_group_id"]
    assert rows[0]["state_group_size"] == 2
    assert rows[2]["topology_group_size"] == 1
    assert rows[2]["state_group_size"] == 1
    assert rows[0]["nearest_label"] == "b"
    assert rows[0]["nearest_classification"] == "exact_duplicate"
    assert rows[0]["nearest_exact_duplicate"] is True
    assert rows[0]["nearest_min_composition_cosine"] == 1.0
    assert rows[0]["nearest_min_condition_cosine"] == 1.0
    assert rows[0]["nearest_topology_feature_cosine"] == 1.0


def test_process_graph_similarity_pairs_reports_meaningful_pairs_by_default():
    feed = _feed()
    duplicate_a = _hx_state(feed, outlet_id="Feed_hx_p10_1", delta_T_K=10.0)
    duplicate_b = _hx_state(feed, outlet_id="Renamed_hx_outlet", delta_T_K=10.0)
    similar_same_topology = _hx_state(
        feed,
        outlet_id="Slightly_warmer_hx_outlet",
        delta_T_K=10.0,
        actual_delta_T_K=10.01,
    )
    different = _hx_state(
        feed,
        outlet_id="Different_hx_outlet",
        delta_T_K=100.0,
        pressure_Pa=1_000_000.0,
        molar_flow_mols=0.1,
        composition={"nitrogen": 1.0, "propane": 0.0, "n-butane": 0.0},
    )

    rows = process_graph_similarity_pairs(
        {
            "duplicate_a": duplicate_a,
            "duplicate_b": duplicate_b,
            "similar_same_topology": similar_same_topology,
            "different": different,
        },
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
        as_dataframe=False,
    )

    labels = {(row["left_label"], row["right_label"]) for row in rows}
    classifications = {row["classification"] for row in rows}
    assert ("duplicate_a", "duplicate_b") in labels
    assert ("duplicate_a", "different") not in labels
    assert "different" not in classifications
    assert "exact_duplicate" in classifications
    assert "same_topology_similar_streams" in classifications
    exact_row = next(row for row in rows if row["classification"] == "exact_duplicate")
    same_topology_row = next(
        row for row in rows if row["classification"] == "same_topology_similar_streams"
    )
    assert exact_row["recommendation"] == "already handled by exact duplicate pruning"
    assert same_topology_row["recommendation"] == (
        "inspect; possible future deprioritization candidate"
    )


def test_process_graph_similarity_pairs_can_include_different_pairs():
    feed = _feed()
    reference = _hx_state(feed, outlet_id="Feed_hx_p10_1", delta_T_K=10.0)
    different = _hx_state(
        feed,
        outlet_id="Different_hx_outlet",
        delta_T_K=100.0,
        pressure_Pa=1_000_000.0,
        molar_flow_mols=0.1,
        composition={"nitrogen": 1.0, "propane": 0.0, "n-butane": 0.0},
    )

    rows = process_graph_similarity_pairs(
        {"reference": reference, "different": different},
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
        include_different=True,
        as_dataframe=False,
    )

    assert len(rows) == 1
    assert rows[0]["classification"] == "different"
    assert rows[0]["recommendation"] == "ignore"


def test_process_graph_similarity_summary_counts_all_pairs():
    feed = _feed()
    duplicate_a = _hx_state(feed, outlet_id="Feed_hx_p10_1", delta_T_K=10.0)
    duplicate_b = _hx_state(feed, outlet_id="Renamed_hx_outlet", delta_T_K=10.0)
    similar_same_topology = _hx_state(
        feed,
        outlet_id="Slightly_warmer_hx_outlet",
        delta_T_K=10.0,
        actual_delta_T_K=10.01,
    )
    different = _hx_state(
        feed,
        outlet_id="Different_hx_outlet",
        delta_T_K=100.0,
        pressure_Pa=1_000_000.0,
        molar_flow_mols=0.1,
        composition={"nitrogen": 1.0, "propane": 0.0, "n-butane": 0.0},
    )
    items = {
        "duplicate_a": duplicate_a,
        "duplicate_b": duplicate_b,
        "similar_same_topology": similar_same_topology,
        "different": different,
    }

    all_rows = process_graph_similarity_pairs(
        items,
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
        include_different=True,
        as_dataframe=False,
    )
    summary = process_graph_similarity_summary(
        items,
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
    )

    assert summary["n_items"] == 4
    assert summary["n_pairs"] == len(all_rows) == 6
    assert summary["n_exact_duplicate_pairs"] == 1
    assert summary["n_same_topology_similar_stream_pairs"] == 2
    assert summary["n_different_pairs"] == 3
    assert summary["n_suspicious_pairs"] == 3
    assert summary["exact_duplicate_pair_fraction"] == 1 / 6
    assert summary["suspicious_pair_fraction"] == 0.5


def test_process_graph_similarity_summary_handles_one_item():
    feed = _feed()
    state = _hx_state(feed, outlet_id="Feed_hx_p10_1", delta_T_K=10.0)

    rows = process_graph_similarity_pairs(
        {"only": state},
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
        as_dataframe=False,
    )
    summary = process_graph_similarity_summary(
        {"only": state},
        feed_stream=feed,
        components=COMPOUNDS,
        config=MCTSConfig(),
    )

    assert rows == []
    assert summary["n_items"] == 1
    assert summary["n_pairs"] == 0
    assert summary["n_suspicious_pairs"] == 0
    assert summary["exact_duplicate_pair_fraction"] == 0.0
    assert summary["suspicious_pair_fraction"] == 0.0


def _hx_state(
    feed: StreamState,
    *,
    outlet_id: str,
    delta_T_K: float,
    actual_delta_T_K: float | None = None,
    pressure_Pa: float | None = None,
    molar_flow_mols: float | None = None,
    composition: dict[str, float] | None = None,
) -> SearchState:
    outlet = StreamState(
        id=outlet_id,
        temperature_K=feed.temperature_K + (actual_delta_T_K or delta_T_K),
        pressure_Pa=pressure_Pa or feed.pressure_Pa,
        molar_flow_mols=molar_flow_mols or feed.molar_flow_mols,
        composition=dict(composition or feed.composition),
        history=("hx",),
    )
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=delta_T_K)
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
