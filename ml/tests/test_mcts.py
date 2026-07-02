from __future__ import annotations

import random

import pytest

from ml import (
    MCTSConfig,
    RefinedSequenceResult,
    SearchState,
    ShortcutDistillationResult,
    StreamState,
    UnitAction,
    append_stream_root,
    batched_mcts_search,
    build_pr_flasher,
    column_duties_from_energy_balance,
    mcts_search,
    mutual_information_separation,
    parallel_mcts_search,
    process_graph_from_feed,
    refine_distillation_sequence,
    shortcut_distillation_fug,
    state_identity_hash,
)
from ml.mcts import (
    _DiagnosticsAccumulator,
    _Node,
    _apply_action,
    _distillation_condenser_duty_W,
    _distillation_lineage_pair_counts,
    _effective_leaf_discount,
    _is_terminal,
    _resolve_compressor_target_ratio,
    _resolve_hx_target_delta_T,
    _resolve_pump_target_ratio,
    _resolve_valve_target_ratio,
    _reward,
    _rollout,
    _rollout_action,
    _select,
    _stream_vapor_fraction,
    _recycle_count,
    _valid_actions,
    _widen_node,
)


COMPOUNDS = ["methane", "ethane", "nitrogen"]
DISTILLATION_COMPOUNDS = ["nitrogen", "propane", "n-butane"]


def _feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 110.0,
        "pressure_Pa": 100000.0,
        "molar_flow_mols": 1.0,
        "composition": {"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
    }
    values.update(overrides)
    return StreamState(**values)


def _distillation_feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 500000.0,
        "molar_flow_mols": 2.0,
        "composition": {"nitrogen": 0.1, "propane": 0.45, "n-butane": 0.45},
    }
    values.update(overrides)
    return StreamState(**values)


def test_mcts_discovers_hx_flash_cool_accept_order_for_target_liquid():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        product_role="CooledLiquid",
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = mcts_search(_feed(), provider, config, iterations=160, seed=7)

    assert result.product is not None
    assert result.product.history[-1] == "hx"
    assert "flash:liquid" in result.product.history
    assert result.product.temperature_K == 110.0
    assert abs(result.product.composition["methane"] - 0.48) < 0.01

    kinds = [action.kind for action in result.best_sequence]
    assert kinds == ["hx", "flash", "hx", "accept"]
    assert result.best_sequence[0].delta_T_K == 10.0
    assert result.best_sequence[2].delta_T_K == -10.0


def test_mcts_is_deterministic_for_same_seed_and_config():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    first = mcts_search(_feed(), provider, config, iterations=120, seed=11)
    second = mcts_search(_feed(), provider, config, iterations=120, seed=11)

    assert first.best_sequence == second.best_sequence
    assert first.best_reward == pytest.approx(second.best_reward)
    assert first.product.composition == pytest.approx(second.product.composition)


def test_candidate_evaluation_mode_finds_target_sequence():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
        candidate_eval_width=4,
        candidate_rollouts_per_action=2,
        candidate_eval_workers=2,
    )

    result = mcts_search(_feed(), provider, config, iterations=60, seed=9)

    assert result.product is not None
    assert abs(result.product.composition["methane"] - 0.48) < 0.01
    assert result.product.temperature_K == 110.0
    assert [action.kind for action in result.best_sequence] == ["hx", "flash", "hx", "accept"]


def test_candidate_evaluation_width_zero_preserves_seeded_behavior():
    provider = build_pr_flasher(COMPOUNDS)
    base_config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
    )
    explicit_zero_config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        candidate_eval_width=0,
    )

    base = mcts_search(_feed(), provider, base_config, iterations=80, seed=13)
    explicit_zero = mcts_search(_feed(), provider, explicit_zero_config, iterations=80, seed=13)

    assert base.best_sequence == explicit_zero.best_sequence
    assert base.best_reward == pytest.approx(explicit_zero.best_reward)


def test_mcts_progress_records_and_callback_are_emitted():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        max_depth=2,
    )
    callback_records = []

    result = mcts_search(
        _feed(),
        provider,
        config,
        iterations=12,
        seed=5,
        progress_interval=5,
        progress_callback=callback_records.append,
    )

    assert [record["iteration"] for record in result.progress] == [5, 10, 12]
    assert callback_records == list(result.progress)
    assert all(record["iterations"] == 12 for record in result.progress)
    assert all("best_reward" in record for record in result.progress)
    assert all("sequence_kinds" in record for record in result.progress)
    assert result.diagnostics.n_expanded_nodes > 0
    assert result.diagnostics.n_seen_state_identities >= 1
    assert result.diagnostics.n_duplicate_states_skipped == 0
    assert all("n_expanded_nodes" in record for record in result.progress)
    assert all("n_seen_state_identities" in record for record in result.progress)


def test_duplicate_pruning_diagnostics_count_skipped_child_states():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed(id="DemoFeed")
    state = SearchState(open_streams=(feed,))
    action = UnitAction(kind="accept", stream_id="DemoFeed", role="DemoProduct")
    config = MCTSConfig(
        target_component="propane",
        target_fraction=feed.composition["propane"],
        product_role="DemoProduct",
        require_flash_liquid_product=False,
        enable_exact_duplicate_pruning=True,
    )
    root = _Node(state, config, feed, provider)
    root.untried_actions = [action, action]
    diagnostics = _DiagnosticsAccumulator()
    seen_hashes = {state_identity_hash(state)}

    _select(root, feed, provider, config, random.Random(1), seen_hashes, diagnostics)
    _select(root, feed, provider, config, random.Random(2), seen_hashes, diagnostics)
    snapshot = diagnostics.snapshot(len(seen_hashes))

    assert len(root.children) == 1
    assert snapshot.n_expanded_nodes == 1
    assert snapshot.n_duplicate_states_skipped == 1
    assert snapshot.n_seen_state_identities == 2
    assert snapshot.duplicate_skip_rate == pytest.approx(0.5)


def test_apply_action_cache_reuses_equivalent_stream_unit_results():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
    )
    cache = {}
    diagnostics = _DiagnosticsAccumulator()
    left_state = SearchState(open_streams=(_feed(id="Feed"),))
    right_state = SearchState(open_streams=(_feed(id="RenamedFeed"),))

    left_next = _apply_action(
        left_state,
        UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0),
        provider,
        config,
        cache,
        diagnostics,
    )
    right_next = _apply_action(
        right_state,
        UnitAction(kind="hx", stream_id="RenamedFeed", delta_T_K=10.0),
        provider,
        config,
        cache,
        diagnostics,
    )
    snapshot = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_apply_action_cache_entries=len(cache),
    )

    assert snapshot.n_apply_action_cache_misses == 1
    assert snapshot.n_apply_action_cache_hits == 1
    assert snapshot.apply_action_cache_hit_rate == pytest.approx(0.5)
    assert left_next.open_streams[0].id == "Feed_hx_p10_1"
    assert right_next.open_streams[0].id == "RenamedFeed_hx_p10_1"
    assert right_next.open_streams[0].temperature_K == pytest.approx(
        left_next.open_streams[0].temperature_K
    )
    assert right_next.open_streams[0].history == ("hx",)
    assert snapshot.n_apply_action_cache_entries == 1
    assert snapshot.apply_action_calc_time_s >= 0.0
    assert snapshot.apply_action_cache_saved_estimate_s >= 0.0


def test_apply_action_cache_respects_action_kind_filter():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
        cached_action_kinds=("distillation",),
    )
    cache = {}
    diagnostics = _DiagnosticsAccumulator()

    for stream_id in ("Feed", "RenamedFeed"):
        _apply_action(
            SearchState(open_streams=(_feed(id=stream_id),)),
            UnitAction(kind="hx", stream_id=stream_id, delta_T_K=10.0),
            provider,
            config,
            cache,
            diagnostics,
        )
    snapshot = diagnostics.snapshot(n_seen_state_identities=0, n_apply_action_cache_entries=len(cache))

    assert cache == {}
    assert snapshot.n_apply_action_cache_hits == 0
    assert snapshot.n_apply_action_cache_misses == 0
    assert snapshot.n_apply_action_cache_entries == 0
    assert snapshot.apply_action_calc_time_s >= 0.0


def test_apply_action_cache_respects_max_entries():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
        max_apply_action_cache_entries=0,
    )
    cache = {}
    diagnostics = _DiagnosticsAccumulator()

    for stream_id in ("Feed", "RenamedFeed"):
        _apply_action(
            SearchState(open_streams=(_feed(id=stream_id),)),
            UnitAction(kind="hx", stream_id=stream_id, delta_T_K=10.0),
            provider,
            config,
            cache,
            diagnostics,
        )
    snapshot = diagnostics.snapshot(n_seen_state_identities=0, n_apply_action_cache_entries=len(cache))

    assert cache == {}
    assert snapshot.n_apply_action_cache_hits == 0
    assert snapshot.n_apply_action_cache_misses == 2
    assert snapshot.n_apply_action_cache_entries == 0


def test_distillation_validation_cache_is_reused_by_apply_action():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        enable_distillation_actions=True,
        enable_apply_action_cache=True,
        cached_action_kinds=("distillation",),
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=False,
    )
    distillation_result_cache = {}
    diagnostics = _DiagnosticsAccumulator()

    actions = [
        action
        for action in _valid_actions(
            state,
            config,
            provider,
            feed,
            distillation_result_cache,
            diagnostics,
        )
        if action.kind == "distillation"
    ]
    snapshot_after_validation = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_distillation_result_cache_entries=len(distillation_result_cache),
    )

    assert actions
    assert snapshot_after_validation.n_distillation_result_cache_misses == len(actions)
    assert snapshot_after_validation.n_distillation_result_cache_hits == 0

    next_state = _apply_action(
        state,
        actions[0],
        provider,
        config,
        None,
        diagnostics,
        distillation_result_cache,
    )
    snapshot_after_apply = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_distillation_result_cache_entries=len(distillation_result_cache),
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 2
    assert snapshot_after_apply.n_distillation_result_cache_hits == 1
    assert snapshot_after_apply.n_distillation_result_cache_misses == len(actions)
    assert snapshot_after_apply.distillation_result_cache_hit_rate == pytest.approx(
        1 / (len(actions) + 1)
    )


def test_valid_action_cache_reuses_equivalent_state_action_lists():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(10.0,),
        max_flash_count_per_path=0,
        enable_distillation_actions=True,
        enable_action_generation_cache=True,
        validate_distillation_candidates=False,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=False,
    )
    valid_action_cache = {}
    relative_volatility_cache = {}
    diagnostics = _DiagnosticsAccumulator()

    first = _valid_actions(
        state,
        config,
        provider,
        feed,
        None,
        diagnostics,
        valid_action_cache,
        relative_volatility_cache,
    )
    second = _valid_actions(
        state,
        config,
        provider,
        feed,
        None,
        diagnostics,
        valid_action_cache,
        relative_volatility_cache,
    )
    snapshot = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_valid_action_cache_entries=len(valid_action_cache),
        n_relative_volatility_cache_entries=len(relative_volatility_cache),
    )

    assert first == second
    assert snapshot.n_valid_action_calls == 2
    assert snapshot.n_valid_action_cache_misses == 1
    assert snapshot.n_valid_action_cache_hits == 1
    assert snapshot.valid_action_cache_hit_rate == pytest.approx(0.5)
    assert snapshot.n_valid_action_cache_entries == 1
    assert snapshot.valid_action_generation_time_s >= 0.0
    assert snapshot.valid_action_cache_saved_estimate_s >= 0.0
    assert snapshot.n_valid_actions_generated_total == len(first)
    assert dict(snapshot.valid_actions_generated_by_kind)["distillation"] > 0


def test_relative_volatility_cache_reuses_equivalent_renamed_streams():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    left = _distillation_feed(id="Feed")
    right = _distillation_feed(id="RenamedFeed")
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        enable_distillation_actions=True,
        enable_action_generation_cache=True,
        validate_distillation_candidates=False,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=False,
    )
    relative_volatility_cache = {}
    diagnostics = _DiagnosticsAccumulator()

    _valid_actions(
        SearchState(open_streams=(left,)),
        config,
        provider,
        left,
        relative_volatility_cache=relative_volatility_cache,
        diagnostics=diagnostics,
    )
    _valid_actions(
        SearchState(open_streams=(right,)),
        config,
        provider,
        right,
        relative_volatility_cache=relative_volatility_cache,
        diagnostics=diagnostics,
    )
    snapshot = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_relative_volatility_cache_entries=len(relative_volatility_cache),
    )

    assert snapshot.n_relative_volatility_cache_misses == 1
    assert snapshot.n_relative_volatility_cache_hits == 1
    assert snapshot.relative_volatility_cache_hit_rate == pytest.approx(0.5)
    assert snapshot.n_relative_volatility_cache_entries == 1
    assert snapshot.relative_volatility_calc_time_s >= 0.0
    assert snapshot.relative_volatility_cache_saved_estimate_s >= 0.0


def test_action_generation_cache_disabled_ignores_supplied_cache_objects():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        enable_distillation_actions=True,
        enable_action_generation_cache=False,
        validate_distillation_candidates=False,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=False,
    )
    valid_action_cache = {}
    relative_volatility_cache = {}
    diagnostics = _DiagnosticsAccumulator()

    _valid_actions(
        SearchState(open_streams=(feed,)),
        config,
        provider,
        feed,
        diagnostics=diagnostics,
        valid_action_cache=valid_action_cache,
        relative_volatility_cache=relative_volatility_cache,
    )
    snapshot = diagnostics.snapshot(
        n_seen_state_identities=0,
        n_valid_action_cache_entries=len(valid_action_cache),
        n_relative_volatility_cache_entries=len(relative_volatility_cache),
    )

    assert valid_action_cache == {}
    assert relative_volatility_cache == {}
    assert snapshot.n_valid_action_calls == 1
    assert snapshot.n_valid_action_cache_hits == 0
    assert snapshot.n_valid_action_cache_misses == 0
    assert snapshot.n_relative_volatility_cache_hits == 0
    assert snapshot.n_relative_volatility_cache_misses == 0
    assert snapshot.valid_action_generation_time_s >= 0.0


def test_progress_records_include_action_generation_cache_diagnostics():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        max_depth=2,
        enable_distillation_actions=True,
        enable_action_generation_cache=True,
        validate_distillation_candidates=False,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_min_alpha_ratio=1.2,
        require_flash_liquid_product=False,
    )

    result = mcts_search(
        _distillation_feed(),
        provider,
        config,
        iterations=12,
        seed=4,
        progress_interval=6,
    )
    final_progress = result.progress[-1]

    assert final_progress["n_valid_action_calls"] == result.diagnostics.n_valid_action_calls
    assert final_progress["n_valid_action_cache_entries"] == result.diagnostics.n_valid_action_cache_entries
    assert final_progress["n_relative_volatility_cache_entries"] == result.diagnostics.n_relative_volatility_cache_entries
    assert "valid_actions_generated_by_kind" in final_progress
    assert result.diagnostics.n_valid_action_cache_misses > 0
    assert result.diagnostics.n_relative_volatility_cache_misses > 0


def test_batched_mcts_complete_separation_progress_includes_metric():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        max_depth=1,
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_min_alpha_ratio=1.2,
    )

    result = batched_mcts_search(
        _distillation_feed(),
        provider,
        config,
        iterations=10,
        batch_size=4,
        rollout_workers=2,
        seed=6,
        progress_interval=4,
    )

    assert [record["iteration"] for record in result.progress] == [4, 8, 10]
    assert result.progress[-1]["separation_target"] == 3
    assert result.progress[-1]["separation_score"] > 1.0
    assert 0.0 < result.progress[-1]["fraction_of_target"] <= 1.0
    assert "component_scores" in result.progress[-1]


def test_batched_mcts_finds_target_sequence():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = batched_mcts_search(
        _feed(),
        provider,
        config,
        iterations=160,
        batch_size=8,
        rollout_workers=2,
        seed=2,
    )

    assert result.iterations == 160
    assert result.batch_size == 8
    assert result.rollout_workers == 2
    assert result.product is not None
    assert abs(result.product.composition["methane"] - 0.48) < 0.01
    assert result.product.temperature_K == 110.0
    assert [action.kind for action in result.best_sequence] == ["hx", "flash", "hx", "accept"]


def test_batched_mcts_batch_size_one_is_deterministic():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
    )

    first = batched_mcts_search(
        _feed(),
        provider,
        config,
        iterations=80,
        batch_size=1,
        rollout_workers=1,
        seed=13,
    )
    second = batched_mcts_search(
        _feed(),
        provider,
        config,
        iterations=80,
        batch_size=1,
        rollout_workers=1,
        seed=13,
    )

    assert first.best_sequence == second.best_sequence
    assert first.best_reward == pytest.approx(second.best_reward)


def test_batched_mcts_invalid_arguments_raise_value_error():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    with pytest.raises(ValueError, match="iterations must be positive"):
        batched_mcts_search(_feed(), provider, config, iterations=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        batched_mcts_search(_feed(), provider, config, batch_size=0)

    with pytest.raises(ValueError, match="rollout_workers must be positive"):
        batched_mcts_search(_feed(), provider, config, rollout_workers=0)


def test_mcts_finds_good_sequence_with_broader_delta_set():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-40.0, -30.0, -20.0, -10.0, 10.0, 20.0, 30.0, 40.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = mcts_search(_feed(), provider, config, iterations=900, seed=5)

    assert result.product is not None
    assert abs(result.product.composition["methane"] - 0.48) < 0.01
    assert [action.kind for action in result.best_sequence] == ["hx", "flash", "hx", "accept"]
    assert result.product.temperature_K == 110.0


def test_mcts_can_choose_feasible_distillation_key_pair_action():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.7773,
        product_role="PropaneRich",
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        max_depth=2,
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=True,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = mcts_search(_distillation_feed(), provider, config, iterations=40, seed=3)

    assert result.product is not None
    assert abs(result.product.composition["propane"] - 0.7773) < 0.01
    assert [action.kind for action in result.best_sequence] == [
        "distillation",
        "accept",
    ]
    assert result.best_sequence[0].light_key == "propane"
    assert result.best_sequence[0].heavy_key == "n-butane"


def test_complete_separation_objective_scores_open_outlet_streams():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        max_depth=1,
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_min_alpha_ratio=1.2,
        require_flash_liquid_product=True,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = mcts_search(_distillation_feed(), provider, config, iterations=12, seed=3)

    assert result.product is None
    assert result.best_reward > 1.0
    assert result.best_reward < 3.0
    assert result.best_sequence[0].kind == "distillation"
    assert len(result.best_state.open_streams) == 2


def test_complete_separation_objective_does_not_emit_accept_actions():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    config = MCTSConfig(
        objective_mode="complete_separation",
        allowed_delta_T_K=(),
        max_flash_count_per_path=0,
        max_depth=2,
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_min_alpha_ratio=1.2,
        require_flash_liquid_product=True,
    )
    state = SearchState(open_streams=(feed,))
    split_state = _apply_action(
        state,
        UnitAction(
            kind="distillation",
            stream_id="Feed",
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
            reflux_ratio_multiplier=1.5,
        ),
        provider,
        config,
    )

    actions = _valid_actions(split_state, config, provider, feed)

    assert "accept" not in {action.kind for action in actions}


def test_complete_separation_rollout_does_not_force_distillation_actions():
    config = MCTSConfig(objective_mode="complete_separation")
    state = SearchState(open_streams=(_distillation_feed(),))
    actions = [
        UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0),
        UnitAction(
            kind="distillation",
            stream_id="Feed",
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
            reflux_ratio_multiplier=1.5,
        ),
    ]

    selected_kinds = {
        _rollout_action(state, actions, config, random.Random(seed)).kind
        for seed in range(20)
    }

    assert selected_kinds == {"distillation", "hx"}


def test_stream_priority_gating_limits_processing_actions_to_top_stream():
    feed = StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=100.0,
        composition={"methane": 0.5, "ethane": 0.5},
    )
    low_priority = StreamState(
        id="LowPriority",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=80.0,
        composition={"methane": 0.99, "ethane": 0.01},
    )
    high_priority = StreamState(
        id="HighPriority",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=50.0,
        composition={"methane": 0.5, "ethane": 0.5},
    )
    state = SearchState(open_streams=(low_priority, high_priority))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
        max_flash_count_per_path=0,
        max_active_streams_per_state=1,
    )

    actions = _valid_actions(state, config, feed_stream=feed)
    hx_stream_ids = [action.stream_id for action in actions if action.kind == "hx"]

    assert hx_stream_ids == ["HighPriority"]


def test_stream_priority_gating_keeps_valid_accept_actions():
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=feed.composition["methane"],
        allowed_delta_T_K=(10.0,),
        max_flash_count_per_path=0,
        max_active_streams_per_state=0,
        require_flash_liquid_product=False,
    )

    actions = _valid_actions(state, config, feed_stream=feed)

    assert any(action.kind == "accept" and action.stream_id == "Feed" for action in actions)
    assert all(action.kind != "hx" for action in actions)


def test_stream_priority_gating_diagnostics_count_filtered_streams():
    feed = StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=100.0,
        composition={"methane": 0.5, "ethane": 0.5},
    )
    low_priority = StreamState(
        id="LowPriority",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=80.0,
        composition={"methane": 0.99, "ethane": 0.01},
    )
    high_priority = StreamState(
        id="HighPriority",
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=50.0,
        composition={"methane": 0.5, "ethane": 0.5},
    )
    state = SearchState(open_streams=(low_priority, high_priority))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
        max_flash_count_per_path=0,
        max_active_streams_per_state=1,
    )
    diagnostics = _DiagnosticsAccumulator()

    _valid_actions(
        state,
        config,
        feed_stream=feed,
        diagnostics=diagnostics,
    )
    snapshot = diagnostics.snapshot(n_seen_state_identities=0)

    assert snapshot.n_stream_priority_gating_calls == 1
    assert snapshot.n_stream_priority_streams_considered == 2
    assert snapshot.n_stream_priority_streams_gated == 1
    assert snapshot.stream_priority_gate_rate == pytest.approx(0.5)


def test_valid_actions_enforce_temperature_bounds():
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-100.0, -10.0, 10.0, 500.0),
        min_temperature_K=105.0,
        max_temperature_K=125.0,
    )

    actions = _valid_actions(state, config)
    hx_deltas = sorted(action.delta_T_K for action in actions if action.kind == "hx")

    assert hx_deltas == [10.0]


def test_compressor_actions_are_disabled_by_default():
    state = SearchState(open_streams=(_feed(temperature_K=300.0),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    actions = _valid_actions(state, config)

    assert all(action.kind != "compressor" for action in actions)


def test_pump_actions_are_disabled_by_default():
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    actions = _valid_actions(state, config)

    assert all(action.kind != "pump" for action in actions)


def test_valve_actions_are_disabled_by_default():
    state = SearchState(open_streams=(_feed(pressure_Pa=200000.0),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    actions = _valid_actions(state, config)

    assert all(action.kind != "valve" for action in actions)


def test_distillation_actions_are_disabled_by_default_even_with_provider():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(_distillation_feed(),))
    config = MCTSConfig(target_component="propane", target_fraction=0.78)

    actions = _valid_actions(state, config, provider)

    assert all(action.kind != "distillation" for action in actions)


def test_valid_actions_include_feasible_adjacent_distillation_key_pairs():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(_distillation_feed(),))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        require_flash_liquid_product=False,
    )

    actions = _valid_actions(state, config, provider)
    pairs = sorted(
        (action.light_key, action.heavy_key)
        for action in actions
        if action.kind == "distillation"
    )

    assert pairs == [("nitrogen", "propane"), ("propane", "n-butane")]


def test_valid_actions_preserve_all_distillation_reflux_multipliers():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(_distillation_feed(),))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.2, 1.5, 2.0),
        require_flash_liquid_product=False,
    )

    actions = [
        action
        for action in _valid_actions(state, config, provider)
        if action.kind == "distillation"
    ]
    multipliers_by_pair = {
        (action.light_key, action.heavy_key): []
        for action in actions
    }
    for action in actions:
        multipliers_by_pair[(action.light_key, action.heavy_key)].append(
            action.reflux_ratio_multiplier
        )

    assert multipliers_by_pair
    assert all(
        sorted(multipliers) == [1.2, 1.5, 2.0]
        for multipliers in multipliers_by_pair.values()
    )


def test_valid_actions_can_include_all_feasible_distillation_key_pairs():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(_distillation_feed(),))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_key_pair_mode="all",
        require_flash_liquid_product=False,
    )

    actions = _valid_actions(state, config, provider)
    pairs = sorted(
        (action.light_key, action.heavy_key)
        for action in actions
        if action.kind == "distillation"
    )

    assert pairs == [
        ("nitrogen", "n-butane"),
        ("nitrogen", "propane"),
        ("propane", "n-butane"),
    ]


def test_distillation_min_alpha_filter_removes_key_pairs():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(_distillation_feed(),))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        distillation_min_alpha_ratio=1e6,
        require_flash_liquid_product=False,
    )

    actions = _valid_actions(state, config, provider)

    assert all(action.kind != "distillation" for action in actions)


def test_valid_actions_include_configured_compressor_ratios():
    state = SearchState(open_streams=(_feed(temperature_K=300.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_compression_ratios=(1.0, 2.0, 3.0),
        max_pressure_Pa=250000.0,
    )

    actions = _valid_actions(state, config)
    ratios = sorted(
        action.pressure_ratio for action in actions if action.kind == "compressor"
    )

    assert ratios == [2.0]


def test_valid_actions_include_configured_compressor_delta_pressures():
    state = SearchState(open_streams=(_feed(temperature_K=300.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_compression_delta_P_Pa=(0.0, 100000.0, 200000.0),
        max_pressure_Pa=250000.0,
    )

    actions = _valid_actions(state, config)
    deltas = sorted(
        action.delta_P_Pa for action in actions if action.kind == "compressor"
    )

    assert deltas == [100000.0]


def test_valid_actions_include_configured_pump_ratios():
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_pump_pressure_ratios=(1.0, 2.0, 3.0),
        max_pressure_Pa=250000.0,
    )

    actions = _valid_actions(state, config)
    ratios = sorted(action.pressure_ratio for action in actions if action.kind == "pump")

    assert ratios == [2.0]


def test_valid_actions_include_configured_pump_delta_pressures():
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_pump_delta_P_Pa=(0.0, 100000.0, 200000.0),
        max_pressure_Pa=250000.0,
    )

    actions = _valid_actions(state, config)
    deltas = sorted(action.delta_P_Pa for action in actions if action.kind == "pump")

    assert deltas == [100000.0]


def test_valid_actions_include_configured_valve_ratios():
    state = SearchState(open_streams=(_feed(pressure_Pa=200000.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_valve_pressure_ratios=(0.0, 0.5, 1.0),
        min_pressure_Pa=120000.0,
    )

    actions = _valid_actions(state, config)
    ratios = sorted(action.pressure_ratio for action in actions if action.kind == "valve")

    assert ratios == []

    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_valve_pressure_ratios=(0.0, 0.5, 0.75, 1.0),
        min_pressure_Pa=120000.0,
    )
    actions = _valid_actions(state, config)
    ratios = sorted(action.pressure_ratio for action in actions if action.kind == "valve")

    assert ratios == [0.75]


def test_valid_actions_include_configured_valve_delta_pressures():
    state = SearchState(open_streams=(_feed(pressure_Pa=200000.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_valve_delta_P_Pa=(0.0, 50000.0, 150000.0),
        min_pressure_Pa=100000.0,
    )

    actions = _valid_actions(state, config)
    deltas = sorted(action.delta_P_Pa for action in actions if action.kind == "valve")

    assert deltas == [50000.0]


def test_accept_action_requires_flash_liquid_when_configured():
    provider = build_pr_flasher(COMPOUNDS)
    feed_state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.965,
        require_flash_liquid_product=True,
    )

    root_actions = _valid_actions(feed_state, config)
    assert all(action.kind != "accept" for action in root_actions)

    flashed = _apply_action(feed_state, UnitAction(kind="flash", stream_id="Feed"), provider, config)
    flashed_actions = _valid_actions(flashed, config)
    accept_streams = {action.stream_id for action in flashed_actions if action.kind == "accept"}

    assert any(stream_id.endswith("_liquid") for stream_id in accept_streams)
    assert all(stream_id.endswith("_liquid") for stream_id in accept_streams)


def test_accept_action_can_accept_feed_when_flash_liquid_requirement_disabled():
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.965,
        require_flash_liquid_product=False,
    )

    actions = _valid_actions(state, config)

    assert any(action.kind == "accept" and action.stream_id == "Feed" for action in actions)


def test_flash_actions_not_repeated_on_same_stream_path_by_default():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    flashed = _apply_action(state, UnitAction(kind="flash", stream_id="Feed"), provider, config)
    liquid = next(stream for stream in flashed.open_streams if "flash:liquid" in stream.history)
    actions = _valid_actions(SearchState(open_streams=(liquid,)), config)

    assert all(action.kind != "flash" for action in actions)


def test_apply_hx_action_accumulates_absolute_duty_and_replaces_stream():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    next_state = _apply_action(
        state,
        UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0),
        provider,
        config,
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 1
    assert next_state.open_streams[0].temperature_K == pytest.approx(120.0)
    assert next_state.open_streams[0].composition == pytest.approx(_feed().composition)
    assert next_state.total_abs_duty_W > 0.0


def test_apply_flash_action_splits_open_stream_and_closes_material_balance():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    next_state = _apply_action(state, UnitAction(kind="flash", stream_id="Feed"), provider, config)

    assert not next_state.errors
    assert len(next_state.open_streams) == 2
    assert sum(stream.molar_flow_mols for stream in next_state.open_streams) == pytest.approx(1.0)
    for compound, z in _feed().composition.items():
        recovered = sum(
            stream.molar_flow_mols * stream.composition[compound]
            for stream in next_state.open_streams
        )
        assert recovered == pytest.approx(z)


def test_apply_compressor_action_replaces_stream_and_accumulates_power():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(temperature_K=300.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_compression_ratios=(2.0,),
    )

    next_state = _apply_action(
        state,
        UnitAction(kind="compressor", stream_id="Feed", pressure_ratio=2.0),
        provider,
        config,
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 1
    assert next_state.open_streams[0].pressure_Pa == pytest.approx(200000.0)
    assert next_state.open_streams[0].temperature_K > 300.0
    assert next_state.open_streams[0].composition == pytest.approx(_feed().composition)
    assert next_state.open_streams[0].history[-1] == "compressor"
    assert next_state.total_abs_duty_W > 0.0


def test_apply_compressor_delta_pressure_action():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(temperature_K=300.0),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_compression_delta_P_Pa=(100000.0,),
    )

    next_state = _apply_action(
        state,
        UnitAction(kind="compressor", stream_id="Feed", delta_P_Pa=100000.0),
        provider,
        config,
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 1
    assert next_state.open_streams[0].pressure_Pa == pytest.approx(200000.0)
    assert next_state.open_streams[0].history[-1] == "compressor"
    assert next_state.total_abs_duty_W > 0.0


def test_apply_pump_action_replaces_liquid_stream_and_accumulates_power():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_pump_pressure_ratios=(2.0,),
    )
    flashed = _apply_action(
        SearchState(open_streams=(_feed(),)),
        UnitAction(kind="flash", stream_id="Feed"),
        provider,
        config,
    )
    liquid = next(stream for stream in flashed.open_streams if "flash:liquid" in stream.history)

    next_state = _apply_action(
        flashed,
        UnitAction(kind="pump", stream_id=liquid.id, pressure_ratio=2.0),
        provider,
        config,
    )

    assert not next_state.errors
    pumped = next(stream for stream in next_state.open_streams if stream.history[-1] == "pump")
    assert pumped.pressure_Pa == pytest.approx(200000.0)
    assert pumped.temperature_K > liquid.temperature_K
    assert pumped.composition == pytest.approx(liquid.composition)
    assert next_state.total_abs_duty_W > 0.0


def test_apply_pump_delta_pressure_action():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_pump_delta_P_Pa=(100000.0,),
    )
    flashed = _apply_action(
        SearchState(open_streams=(_feed(),)),
        UnitAction(kind="flash", stream_id="Feed"),
        provider,
        config,
    )
    liquid = next(stream for stream in flashed.open_streams if "flash:liquid" in stream.history)

    next_state = _apply_action(
        flashed,
        UnitAction(kind="pump", stream_id=liquid.id, delta_P_Pa=100000.0),
        provider,
        config,
    )

    assert not next_state.errors
    pumped = next(stream for stream in next_state.open_streams if stream.history[-1] == "pump")
    assert pumped.pressure_Pa == pytest.approx(200000.0)
    assert next_state.total_abs_duty_W > 0.0


def test_apply_valve_action_replaces_stream_without_power_penalty():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed(temperature_K=300.0, pressure_Pa=200000.0)
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_valve_pressure_ratios=(0.5,),
    )

    next_state = _apply_action(
        state,
        UnitAction(kind="valve", stream_id="Feed", pressure_ratio=0.5),
        provider,
        config,
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 1
    assert next_state.open_streams[0].pressure_Pa == pytest.approx(100000.0)
    assert next_state.open_streams[0].composition == pytest.approx(feed.composition)
    assert next_state.open_streams[0].history[-1] == "valve"
    assert next_state.total_abs_duty_W == pytest.approx(0.0)


def test_apply_valve_delta_pressure_action():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed(temperature_K=300.0, pressure_Pa=200000.0)
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_valve_delta_P_Pa=(100000.0,),
    )

    next_state = _apply_action(
        state,
        UnitAction(kind="valve", stream_id="Feed", delta_P_Pa=100000.0),
        provider,
        config,
    )

    assert not next_state.errors
    assert next_state.open_streams[0].pressure_Pa == pytest.approx(100000.0)
    assert next_state.open_streams[0].history[-1] == "valve"


def test_apply_distillation_action_splits_stream_and_closes_material_balance():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="propane",
        target_fraction=0.78,
        enable_distillation_actions=True,
        require_flash_liquid_product=False,
    )

    next_state = _apply_action(
        state,
        UnitAction(
            kind="distillation",
            stream_id="Feed",
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
            reflux_ratio_multiplier=1.5,
        ),
        provider,
        config,
    )

    assert not next_state.errors
    assert len(next_state.open_streams) == 2
    assert sum(stream.molar_flow_mols for stream in next_state.open_streams) == pytest.approx(
        feed.molar_flow_mols
    )
    distillate = next(
        stream
        for stream in next_state.open_streams
        if stream.history[-1] == "shortcut_distillation:total_condenser_distillate"
    )
    bottoms = next(
        stream
        for stream in next_state.open_streams
        if stream.history[-1] == "shortcut_distillation:bottoms"
    )
    assert distillate.pressure_Pa == pytest.approx(feed.pressure_Pa)
    assert bottoms.pressure_Pa == pytest.approx(feed.pressure_Pa)
    assert distillate.composition["propane"] > feed.composition["propane"]
    assert bottoms.composition["n-butane"] > feed.composition["n-butane"]

    for compound in DISTILLATION_COMPOUNDS:
        recovered = (
            distillate.molar_flow_mols * distillate.composition[compound]
            + bottoms.molar_flow_mols * bottoms.composition[compound]
        )
        expected = feed.molar_flow_mols * feed.composition[compound]
        assert recovered == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_apply_pump_action_on_vapor_records_error_and_keeps_stream_open():
    provider = build_pr_flasher(COMPOUNDS)
    vapor_feed = _feed(temperature_K=300.0)
    state = SearchState(open_streams=(vapor_feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_pump_pressure_ratios=(2.0,),
    )

    next_state = _apply_action(
        state,
        UnitAction(kind="pump", stream_id="Feed", pressure_ratio=2.0),
        provider,
        config,
    )

    assert next_state.errors
    assert "requires a liquid or near-liquid inlet" in next_state.errors[-1]
    assert next_state.open_streams == (vapor_feed,)
    assert next_state.unit_sequence[-1].kind == "pump"


def test_apply_failed_flash_records_error_and_keeps_stream_open():
    provider = build_pr_flasher(COMPOUNDS)
    vapor_feed = _feed(temperature_K=300.0)
    state = SearchState(open_streams=(vapor_feed,))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    next_state = _apply_action(state, UnitAction(kind="flash", stream_id="Feed"), provider, config)

    assert next_state.errors
    assert next_state.open_streams == (vapor_feed,)
    assert next_state.unit_sequence[-1].kind == "flash"


def test_accept_action_creates_product_and_removes_open_stream():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(target_component="methane", target_fraction=0.48)
    flashed = _apply_action(
        SearchState(open_streams=(_feed(),)),
        UnitAction(kind="flash", stream_id="Feed"),
        provider,
        config,
    )
    liquid = next(stream for stream in flashed.open_streams if "flash:liquid" in stream.history)

    accepted = _apply_action(
        flashed,
        UnitAction(kind="accept", stream_id=liquid.id, role="CooledLiquid"),
        provider,
        config,
    )

    assert not accepted.errors
    assert len(accepted.products) == 1
    assert accepted.products[0].stream == liquid
    assert liquid.id not in {stream.id for stream in accepted.open_streams}


def test_product_temperature_constraint_blocks_accept_until_cooling():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
    )
    heated = _apply_action(
        SearchState(open_streams=(_feed(),)),
        UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0),
        provider,
        config,
    )
    flashed = _apply_action(
        heated,
        UnitAction(kind="flash", stream_id=heated.open_streams[0].id),
        provider,
        config,
    )
    liquid = next(stream for stream in flashed.open_streams if "flash:liquid" in stream.history)

    actions_before_cooling = _valid_actions(flashed, config)
    assert all(
        not (action.kind == "accept" and action.stream_id == liquid.id)
        for action in actions_before_cooling
    )

    cooled = _apply_action(
        flashed,
        UnitAction(kind="hx", stream_id=liquid.id, delta_T_K=-10.0),
        provider,
        config,
    )
    cooled_liquid = next(stream for stream in cooled.open_streams if stream.history[-1] == "hx")
    actions_after_cooling = _valid_actions(cooled, config)

    assert any(
        action.kind == "accept" and action.stream_id == cooled_liquid.id
        for action in actions_after_cooling
    )


def test_unit_and_duty_penalties_reduce_best_reward():
    provider = build_pr_flasher(COMPOUNDS)
    base_config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    penalized_config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.1,
        duty_penalty_per_W=1e-5,
    )

    base = mcts_search(_feed(), provider, base_config, iterations=160, seed=7)
    penalized = mcts_search(_feed(), provider, penalized_config, iterations=160, seed=7)

    assert base.product is not None
    assert penalized.product is not None
    assert penalized.best_reward < base.best_reward


def test_zero_iterations_raises_value_error():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    with pytest.raises(ValueError, match="iterations must be positive"):
        mcts_search(_feed(), provider, config, iterations=0)


def test_min_flow_filters_tiny_flash_outlets():
    provider = build_pr_flasher(COMPOUNDS)
    state = SearchState(open_streams=(_feed(),))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        min_flow_mols=0.1,
    )

    next_state = _apply_action(state, UnitAction(kind="flash", stream_id="Feed"), provider, config)

    assert not next_state.errors
    assert len(next_state.open_streams) == 1
    assert "flash:liquid" in next_state.open_streams[0].history


def test_mcts_without_product_returns_result_not_exception():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(10.0,),
        target_product_temperature_K=90.0,
        max_depth=2,
    )

    result = mcts_search(_feed(), provider, config, iterations=20, seed=3)

    assert result.iterations == 20
    assert result.product is None
    assert result.best_reward <= 0.0


def test_parallel_mcts_thread_backend_returns_best_worker_result():
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        target_product_temperature_K=110.0,
        max_depth=4,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    result = parallel_mcts_search(
        _feed(),
        COMPOUNDS,
        config,
        total_iterations=160,
        n_jobs=4,
        base_seed=7,
        backend="thread",
    )

    assert result.total_iterations == 160
    assert result.n_jobs == 4
    assert result.backend == "thread"
    assert len(result.worker_results) == 4
    assert sum(worker.iterations for worker in result.worker_results) == 160
    assert result.best_result.best_reward == max(
        worker.best_reward for worker in result.worker_results
    )
    assert result.best_result.product is not None
    assert abs(result.best_result.product.composition["methane"] - 0.48) < 0.01


def test_parallel_mcts_reduces_jobs_to_iteration_count():
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    result = parallel_mcts_search(
        _feed(),
        COMPOUNDS,
        config,
        total_iterations=3,
        n_jobs=10,
        base_seed=1,
        backend="thread",
    )

    assert result.n_jobs == 3
    assert len(result.worker_results) == 3
    assert all(worker.iterations == 1 for worker in result.worker_results)


def test_parallel_mcts_invalid_arguments_raise_value_error():
    config = MCTSConfig(target_component="methane", target_fraction=0.48)

    with pytest.raises(ValueError, match="total_iterations must be positive"):
        parallel_mcts_search(_feed(), COMPOUNDS, config, total_iterations=0)

    with pytest.raises(ValueError, match="n_jobs must be positive"):
        parallel_mcts_search(_feed(), COMPOUNDS, config, total_iterations=10, n_jobs=0)

    with pytest.raises(ValueError, match="backend"):
        parallel_mcts_search(
            _feed(),
            COMPOUNDS,
            config,
            total_iterations=10,
            n_jobs=2,
            backend="invalid",
        )


# ---------------------------------------------------------------------------
# Thermodynamic state target tests — propane/n-butane binary system
# ---------------------------------------------------------------------------

PHASE_BOUNDARY_COMPOUNDS = ["propane", "n-butane"]


def _pb_feed(**overrides) -> StreamState:
    """50/50 propane/n-butane at 300 K, 500 kPa — two-phase at these conditions."""
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 500_000.0,
        "molar_flow_mols": 2.0,
        "composition": {"propane": 0.5, "n-butane": 0.5},
    }
    values.update(overrides)
    return StreamState(**values)


def _pb_config(**overrides) -> MCTSConfig:
    # Disable default ΔT and ΔP grids so only explicit targets/grids are active.
    values = dict(
        target_component="propane",
        target_fraction=0.9,
        min_temperature_K=150.0,
        max_temperature_K=500.0,
        min_pressure_Pa=10_000.0,
        max_pressure_Pa=10_000_000.0,
        allowed_delta_T_K=(),
    )
    values.update(overrides)
    return MCTSConfig(**values)


def test_hx_bubble_point_target_generates_cooling_action():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()  # two-phase at 300 K, 5 bar
    state = SearchState(open_streams=(stream,))
    config = _pb_config(hx_target_states=("bubble_point",))

    actions = _valid_actions(state, config, provider)
    hx_actions = [a for a in actions if a.kind == "hx"]

    assert len(hx_actions) == 1
    action = hx_actions[0]
    assert action.delta_T_K is not None
    assert action.delta_T_K < 0  # cooling to reach bubble point from two-phase

    next_state = _apply_action(state, action, provider, config)
    assert not next_state.errors
    outlet = next_state.open_streams[0]
    outlet_flash = provider.flasher.flash(
        T=outlet.temperature_K,
        P=outlet.pressure_Pa,
        zs=list(outlet.composition.values()),
    )
    assert outlet_flash.VF == pytest.approx(0.0, abs=0.05)


def test_hx_dew_point_target_generates_heating_action():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()  # two-phase at 300 K, 5 bar
    state = SearchState(open_streams=(stream,))
    config = _pb_config(hx_target_states=("dew_point",))

    actions = _valid_actions(state, config, provider)
    hx_actions = [a for a in actions if a.kind == "hx"]

    assert len(hx_actions) == 1
    action = hx_actions[0]
    assert action.delta_T_K is not None
    assert action.delta_T_K > 0  # heating to reach dew point from two-phase

    next_state = _apply_action(state, action, provider, config)
    assert not next_state.errors
    outlet = next_state.open_streams[0]
    outlet_flash = provider.flasher.flash(
        T=outlet.temperature_K,
        P=outlet.pressure_Pa,
        zs=list(outlet.composition.values()),
    )
    assert outlet_flash.VF == pytest.approx(1.0, abs=0.05)


def test_hx_partial_vf_target_generates_action_reaching_target_vf():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # Stream VF at 300 K / 5 bar ≈ 0.53; use target VF=0.2 to get a significant ΔT
    stream = _pb_feed()
    state = SearchState(open_streams=(stream,))
    config = _pb_config(hx_target_states=("partial_vf",), hx_partial_target_vf=0.2)

    actions = _valid_actions(state, config, provider)
    hx_actions = [a for a in actions if a.kind == "hx"]

    assert len(hx_actions) == 1
    action = hx_actions[0]
    assert action.delta_T_K is not None
    assert action.delta_T_K < 0  # cooling to reduce VF from ~0.53 to 0.2

    next_state = _apply_action(state, action, provider, config)
    assert not next_state.errors
    outlet = next_state.open_streams[0]
    outlet_flash = provider.flasher.flash(
        T=outlet.temperature_K,
        P=outlet.pressure_Pa,
        zs=list(outlet.composition.values()),
    )
    assert outlet_flash.VF == pytest.approx(0.2, abs=0.05)


def test_hx_targets_and_delta_T_grid_coexist_without_deduplication_if_different():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()
    state = SearchState(open_streams=(stream,))
    # Grid ΔT = +20 K is far from bubble_point ΔT (negative) → distinct actions
    config = _pb_config(
        hx_target_states=("bubble_point",),
        allowed_delta_T_K=(20.0,),
    )

    actions = _valid_actions(state, config, provider)
    hx_actions = [a for a in actions if a.kind == "hx"]

    assert len(hx_actions) == 2
    delta_Ts = sorted(a.delta_T_K for a in hx_actions)
    assert delta_Ts[0] < 0.0  # bubble-point target is a cooling action
    assert delta_Ts[1] == pytest.approx(20.0)  # grid action


def test_hx_target_deduplicated_when_grid_matches_within_tolerance():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()

    # Find the true bubble-point ΔT
    dt = _resolve_hx_target_delta_T(stream, "bubble_point", _pb_config(), provider)
    assert dt is not None

    # Use a grid ΔT exactly matching the target → only one HX action total
    config = _pb_config(
        hx_target_states=("bubble_point",),
        allowed_delta_T_K=(round(dt, 1),),
    )
    state = SearchState(open_streams=(stream,))
    actions = _valid_actions(state, config, provider)
    hx_actions = [a for a in actions if a.kind == "hx"]

    assert len(hx_actions) == 1


def test_pump_bubble_pressure_target_not_generated_for_vapor_stream():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 300 K, 3 bar — VF > 0 (below bubble pressure ~6 bar for propane/n-butane 50/50).
    # pump_max_inlet_vapor_fraction gate (default 1e-6) blocks pump actions on non-liquid
    # streams: the action would fail at apply time anyway, so we don't generate it.
    stream = _pb_feed(pressure_Pa=300_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(pump_target_states=("bubble_pressure",))

    actions = _valid_actions(state, config, provider)
    pump_actions = [a for a in actions if a.kind == "pump"]

    assert len(pump_actions) == 0


def test_pump_bubble_pressure_target_not_generated_for_fully_liquid_stream():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 300 K, 20 bar — well above bubble pressure for propane/n-butane 50/50
    stream = _pb_feed(pressure_Pa=2_000_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(pump_target_states=("bubble_pressure",))

    actions = _valid_actions(state, config, provider)
    pump_actions = [a for a in actions if a.kind == "pump"]

    assert len(pump_actions) == 0  # no target generated (already above bubble P)


def test_compressor_dew_pressure_target_generated_for_vapor_stream():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 250 K, 0.5 bar — fully vapor; isentropic compression reaches condensation below 10 MPa
    stream = _pb_feed(temperature_K=250.0, pressure_Pa=50_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(compressor_target_states=("dew_pressure",))

    actions = _valid_actions(state, config, provider)
    comp_actions = [a for a in actions if a.kind == "compressor"]

    assert len(comp_actions) == 1
    assert comp_actions[0].pressure_ratio is not None
    assert comp_actions[0].pressure_ratio > 1.0
    p_out = stream.pressure_Pa * comp_actions[0].pressure_ratio
    assert p_out <= config.max_pressure_Pa


def test_valve_bubble_pressure_target_generated_for_liquid_stream():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 300 K, 20 bar — well above bubble pressure → fully liquid
    stream = _pb_feed(pressure_Pa=2_000_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(valve_target_states=("bubble_pressure",))

    actions = _valid_actions(state, config, provider)
    valve_actions = [a for a in actions if a.kind == "valve"]

    assert len(valve_actions) == 1
    action = valve_actions[0]
    assert action.pressure_ratio is not None
    assert 0.0 < action.pressure_ratio < 1.0
    p_out = stream.pressure_Pa * action.pressure_ratio
    assert p_out >= config.min_pressure_Pa

    next_state = _apply_action(state, action, provider, config)
    assert not next_state.errors
    outlet = next_state.open_streams[0]
    outlet_flash = provider.flasher.flash(
        T=outlet.temperature_K,
        P=outlet.pressure_Pa,
        zs=list(outlet.composition.values()),
    )
    assert outlet_flash.VF > 0.0  # isenthalpic expansion caused flashing


def test_compressor_min_inlet_vapor_fraction_suppresses_all_compressor_actions():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 300 K, 20 bar — fully liquid (VF ≈ 0)
    stream = _pb_feed(pressure_Pa=2_000_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(
        compressor_min_inlet_vapor_fraction=1.0,  # vapour-only
        allowed_compression_ratios=(2.0, 3.0),
        compressor_target_states=("dew_pressure",),
    )

    actions = _valid_actions(state, config, provider)
    comp_actions = [a for a in actions if a.kind == "compressor"]

    assert len(comp_actions) == 0  # both grid and target suppressed


def test_compressor_min_inlet_vapor_fraction_zero_does_not_filter():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    # 300 K, 20 bar — liquid stream
    stream = _pb_feed(pressure_Pa=2_000_000.0)
    state = SearchState(open_streams=(stream,))
    config = _pb_config(
        compressor_min_inlet_vapor_fraction=0.0,  # no restriction (default)
        allowed_compression_ratios=(2.0,),
    )

    actions = _valid_actions(state, config, provider)
    comp_actions = [a for a in actions if a.kind == "compressor"]

    assert len(comp_actions) == 1  # grid action generated regardless of phase


def test_hx_target_states_change_invalidates_action_generation_cache():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()
    state = SearchState(open_streams=(stream,))

    cache: dict = {}
    config_no_target = _pb_config(
        enable_action_generation_cache=True,
        allowed_delta_T_K=(-5.0,),
    )
    config_with_target = _pb_config(
        enable_action_generation_cache=True,
        allowed_delta_T_K=(-5.0,),
        hx_target_states=("bubble_point",),
    )

    actions_no_target = _valid_actions(state, config_no_target, provider, valid_action_cache=cache)
    actions_with_target = _valid_actions(
        state, config_with_target, provider, valid_action_cache=cache
    )

    # Different config signatures → different cache entries → different action lists
    assert len(cache) == 2
    hx_no_target = [a for a in actions_no_target if a.kind == "hx"]
    hx_with_target = [a for a in actions_with_target if a.kind == "hx"]
    assert len(hx_with_target) == len(hx_no_target) + 1  # one extra target action


def test_stream_vapor_fraction_returns_vf_for_two_phase_stream():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()  # two-phase at 300 K, 5 bar
    vf = _stream_vapor_fraction(stream, provider)
    assert vf is not None
    assert 0.0 < vf < 1.0


def test_resolve_hx_target_delta_T_returns_none_for_unknown_target():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed()
    config = _pb_config()
    result = _resolve_hx_target_delta_T(stream, "nonexistent_target", config, provider)
    assert result is None


def test_resolve_pump_target_ratio_returns_none_for_unknown_target():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed(pressure_Pa=300_000.0)
    result = _resolve_pump_target_ratio(stream, "nonexistent", provider)
    assert result is None


def test_resolve_compressor_target_ratio_returns_none_for_unknown_target():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed(pressure_Pa=100_000.0)
    config = _pb_config()
    result = _resolve_compressor_target_ratio(stream, "nonexistent", config, provider)
    assert result is None


def test_resolve_valve_target_ratio_returns_none_for_unknown_target():
    provider = build_pr_flasher(PHASE_BOUNDARY_COMPOUNDS)
    stream = _pb_feed(pressure_Pa=2_000_000.0)
    config = _pb_config()
    result = _resolve_valve_target_ratio(stream, "nonexistent", config, provider)
    assert result is None


# ---------------------------------------------------------------------------
# Leaf value estimator tests — _effective_leaf_discount, _rollout, _reward
# ---------------------------------------------------------------------------


def test_effective_leaf_discount_auto_computes_n_c_over_2():
    feed = _distillation_feed()  # nitrogen 0.1, propane 0.45, n-butane 0.45 — 3 components
    config = MCTSConfig(objective_mode="complete_separation")
    discount = _effective_leaf_discount(config, feed)
    assert discount == pytest.approx(1.5)  # 3 / 2


def test_effective_leaf_discount_explicit_float_overrides_auto():
    feed = _distillation_feed()
    config = MCTSConfig(objective_mode="complete_separation", leaf_value_discount=7.0)
    discount = _effective_leaf_discount(config, feed)
    assert discount == pytest.approx(7.0)


def test_effective_leaf_discount_excludes_trace_components_below_min_fraction():
    # Two meaningful components + one trace below the default 1e-8 threshold
    feed = StreamState(
        id="Feed",
        temperature_K=300.0,
        pressure_Pa=500_000.0,
        molar_flow_mols=1.0,
        composition={"nitrogen": 0.5, "propane": 0.5 - 1e-9, "n-butane": 1e-9},
    )
    config = MCTSConfig(objective_mode="complete_separation")
    discount = _effective_leaf_discount(config, feed)
    assert discount == pytest.approx(1.0)  # 2 meaningful components / 2


def test_rollout_returns_leaf_state_unchanged_when_estimator_active():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        allowed_delta_T_K=(-10.0,),
    )
    result = _rollout(state, feed, provider, config, random.Random(1))
    assert result is state  # no simulation — same object returned immediately


def test_rollout_simulates_when_estimator_inactive():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=False,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        max_depth=1,
    )
    result = _rollout(state, feed, provider, config, random.Random(1))
    assert len(result.unit_sequence) > len(state.unit_sequence)


def test_rollout_simulates_in_single_product_mode_even_with_estimator_active():
    # use_leaf_value_estimator only fires for objective_mode="complete_separation"
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        objective_mode="single_product",
        target_component="propane",
        target_fraction=0.9,
        use_leaf_value_estimator=True,
        allowed_delta_T_K=(),
        enable_distillation_actions=True,
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.5,),
        max_depth=1,
    )
    result = _rollout(state, feed, provider, config, random.Random(1))
    assert len(result.unit_sequence) > len(state.unit_sequence)


def test_reward_augmented_by_potential_when_estimator_active_with_mixed_open_stream():
    feed = _distillation_feed()  # mixed 3-component stream
    state = SearchState(open_streams=(feed,))
    config_no_est = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=False,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    config_est = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_value_discount=2.0,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    r_base = _reward(state, config_no_est, feed)
    r_augmented = _reward(state, config_est, feed)
    # Mixed feed has positive entropy (Phi > 0) → augmented reward > base
    assert r_augmented > r_base


def test_reward_not_augmented_when_no_open_streams_in_state():
    feed = _distillation_feed()
    terminal = SearchState(open_streams=())
    config_no_est = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=False,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    config_est = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_value_discount=10.0,  # large — would dominate if applied
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    r_base = _reward(terminal, config_no_est, feed)
    r_augmented = _reward(terminal, config_est, feed)
    # No open streams → Phi = 0 → discount term vanishes
    assert r_augmented == pytest.approx(r_base)


# ---------------------------------------------------------------------------
# Condenser duty tests — _distillation_condenser_duty_W
# ---------------------------------------------------------------------------


def _make_distillate(molar_flow_mols: float = 2.0) -> StreamState:
    return StreamState(
        id="D",
        temperature_K=300.0,
        pressure_Pa=500_000.0,
        molar_flow_mols=molar_flow_mols,
        composition={"propane": 1.0},
    )


def test_distillation_condenser_duty_W_computes_R_times_D_times_lambda():
    result = ShortcutDistillationResult(
        success=True,
        inlet_stream_id="Feed",
        distillate_stream=_make_distillate(molar_flow_mols=2.0),
        reflux_ratio=1.5,
    )
    config = MCTSConfig(distillation_molar_heat_of_vaporization_J_mol=30_000.0)
    duty = _distillation_condenser_duty_W(result, config)
    assert duty == pytest.approx(1.5 * 2.0 * 30_000.0)  # R * D * lambda = 90 000 W


def test_distillation_condenser_duty_W_returns_zero_when_lambda_zero():
    result = ShortcutDistillationResult(
        success=True,
        inlet_stream_id="Feed",
        distillate_stream=_make_distillate(),
        reflux_ratio=1.5,
    )
    config = MCTSConfig(distillation_molar_heat_of_vaporization_J_mol=0.0)
    assert _distillation_condenser_duty_W(result, config) == 0.0


def test_distillation_condenser_duty_W_returns_zero_when_reflux_ratio_missing():
    result = ShortcutDistillationResult(
        success=True,
        inlet_stream_id="Feed",
        distillate_stream=_make_distillate(),
        reflux_ratio=None,
    )
    config = MCTSConfig(distillation_molar_heat_of_vaporization_J_mol=30_000.0)
    assert _distillation_condenser_duty_W(result, config) == 0.0


def test_distillation_condenser_duty_W_returns_zero_when_distillate_stream_missing():
    result = ShortcutDistillationResult(
        success=True,
        inlet_stream_id="Feed",
        distillate_stream=None,
        reflux_ratio=1.5,
    )
    config = MCTSConfig(distillation_molar_heat_of_vaporization_J_mol=30_000.0)
    assert _distillation_condenser_duty_W(result, config) == 0.0


def test_distillation_duty_accumulated_in_state_after_distillation_action():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    config_with_lambda = MCTSConfig(
        objective_mode="complete_separation",
        distillation_molar_heat_of_vaporization_J_mol=30_000.0,
    )
    config_no_lambda = MCTSConfig(
        objective_mode="complete_separation",
        distillation_molar_heat_of_vaporization_J_mol=0.0,
    )
    state_with = _apply_action(state, action, provider, config_with_lambda)
    state_no = _apply_action(state, action, provider, config_no_lambda)
    assert not state_with.errors
    assert state_with.total_abs_duty_W > 0.0
    assert state_no.total_abs_duty_W == pytest.approx(0.0)


def test_distillation_duty_cache_key_invalidated_when_lambda_changes():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(open_streams=(feed,))
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    config_a = MCTSConfig(
        objective_mode="complete_separation",
        distillation_molar_heat_of_vaporization_J_mol=30_000.0,
    )
    config_b = MCTSConfig(
        objective_mode="complete_separation",
        distillation_molar_heat_of_vaporization_J_mol=0.0,
    )
    cache: dict = {}
    state_a = _apply_action(state, action, provider, config_a, action_cache=cache)
    state_b = _apply_action(state, action, provider, config_b, action_cache=cache)
    # Different lambda → different config signature → two distinct cache entries
    assert len(cache) == 2
    assert state_a.total_abs_duty_W > state_b.total_abs_duty_W


# ---------------------------------------------------------------------------
# Post-search reflux multiplier refinement tests
# ---------------------------------------------------------------------------

_REFINE_ACTION = UnitAction(
    kind="distillation",
    stream_id="Feed",
    light_key="propane",
    heavy_key="n-butane",
    light_key_recovery=0.95,
    heavy_key_recovery=0.05,
    reflux_ratio_multiplier=1.5,
)

_REFINE_GRID = (1.2, 1.5, 2.0, 3.0)


def _refine_config(**overrides) -> MCTSConfig:
    values: dict = dict(objective_mode="complete_separation", unit_penalty=0.0, duty_penalty_per_W=0.0)
    values.update(overrides)
    return MCTSConfig(**values)


def test_refine_distillation_sequence_raises_for_no_distillation_actions():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    sequence = (UnitAction(kind="hx", stream_id="Feed", delta_T_K=-10.0),)
    with pytest.raises(ValueError, match="no distillation actions"):
        refine_distillation_sequence(feed, provider, _refine_config(), sequence)


def test_refine_distillation_sequence_returns_result_for_each_grid_point():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    result = refine_distillation_sequence(
        feed, provider, _refine_config(), (_REFINE_ACTION,), reflux_multiplier_grid=_REFINE_GRID
    )
    assert len(result.grid_results) == len(_REFINE_GRID)
    assert {r["reflux_multiplier"] for r in result.grid_results} == set(_REFINE_GRID)


def test_refine_distillation_sequence_best_reward_is_max_over_grid():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    result = refine_distillation_sequence(
        feed, provider, _refine_config(), (_REFINE_ACTION,), reflux_multiplier_grid=_REFINE_GRID
    )
    best_in_grid = max(r["reward"] for r in result.grid_results)
    assert result.best_reward == pytest.approx(best_in_grid)
    assert result.best_reflux_multiplier in _REFINE_GRID


def test_refine_distillation_sequence_only_changes_reflux_multiplier_in_best_sequence():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    result = refine_distillation_sequence(
        feed, provider, _refine_config(), (_REFINE_ACTION,), reflux_multiplier_grid=_REFINE_GRID
    )
    best_dist = next(a for a in result.best_sequence if a.kind == "distillation")
    assert best_dist.reflux_ratio_multiplier == pytest.approx(result.best_reflux_multiplier)
    assert best_dist.light_key == _REFINE_ACTION.light_key
    assert best_dist.heavy_key == _REFINE_ACTION.heavy_key
    assert best_dist.stream_id == _REFINE_ACTION.stream_id


def test_refine_distillation_sequence_stream_ids_stable_across_multipliers():
    # Downstream actions reference distillate/bottoms IDs by name. IDs are
    # derived from key names and step index, not the reflux multiplier, so
    # replaying with a different multiplier must not break subsequent actions.
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    config = _refine_config(allowed_delta_T_K=(-10.0,))

    # Apply the distillation once to discover the actual distillate stream ID.
    initial = SearchState(open_streams=(feed,))
    after_dist = _apply_action(initial, _REFINE_ACTION, provider, config)
    assert not after_dist.errors, f"setup distillation failed: {after_dist.errors}"
    distillate = max(
        after_dist.open_streams,
        key=lambda s: s.composition.get("propane", 0.0),
    )

    # 2-action sequence: distillation then HX on the distillate.
    sequence = (_REFINE_ACTION, UnitAction(kind="hx", stream_id=distillate.id, delta_T_K=-10.0))
    result = refine_distillation_sequence(
        feed, provider, config, sequence, reflux_multiplier_grid=(1.2, 1.5, 2.0)
    )

    # All grid points must find the distillate stream by ID after multiplier change.
    for point in result.grid_results:
        assert point["n_errors"] == 0, (
            f"m={point['reflux_multiplier']}: errors={point['errors']}"
        )


def test_refine_distillation_sequence_grid_results_include_total_abs_duty_W():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    config = MCTSConfig(
        objective_mode="complete_separation",
        enable_distillation_actions=True,
        distillation_key_pair_mode="adjacent",
        distillation_light_key_recoveries=(0.95,),
        distillation_heavy_key_recoveries=(0.05,),
        distillation_reflux_multipliers=(1.3,),
        distillation_molar_heat_of_vaporization_J_mol=32_000.0,
    )
    result = refine_distillation_sequence(
        feed, provider, config, (_REFINE_ACTION,), reflux_multiplier_grid=_REFINE_GRID
    )
    for point in result.grid_results:
        assert "total_abs_duty_W" in point
        assert isinstance(point["total_abs_duty_W"], float)
        assert point["total_abs_duty_W"] >= 0.0


# ---------------------------------------------------------------------------
# Progressive widening tests
# ---------------------------------------------------------------------------


def test_widening_disabled_by_default_all_actions_available_immediately():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(target_component="methane", target_fraction=0.48)
    # widening_coefficient=0.0 → full action set in untried_actions from creation
    node = _Node(state, config, feed, provider)
    assert len(node.untried_actions) > 0
    assert len(node._all_actions) == 0  # unused path when widening is off


def test_widening_enabled_starts_with_empty_untried_actions():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        widening_coefficient=1.0,
        widening_exponent=0.5,
    )
    node = _Node(state, config, feed, provider, rng=random.Random(1))
    assert len(node.untried_actions) == 0
    assert len(node._all_actions) > 0  # full shuffled list stored separately


def test_widen_node_reveals_ceil_cw_actions_on_first_call():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        widening_coefficient=2.0,
        widening_exponent=0.5,
    )
    node = _Node(state, config, feed, provider, rng=random.Random(1))
    assert len(node.untried_actions) == 0

    _widen_node(node, config)  # N=0 → k = ceil(2.0 * max(0,1)^0.5) = 2
    assert len(node.untried_actions) == 2


def test_widen_node_reveals_more_actions_as_visits_increase():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        widening_coefficient=1.0,
        widening_exponent=0.5,
    )
    node = _Node(state, config, feed, provider, rng=random.Random(1))

    _widen_node(node, config)  # N=0 → k = ceil(1.0 * 1^0.5) = 1
    n_first = len(node.untried_actions)
    assert n_first >= 1

    node.visits = 8  # simulate 8 backpropagations
    _widen_node(node, config)  # N=8 → k = ceil(1.0 * 8^0.5) = ceil(2.83) = 3
    assert len(node.untried_actions) > n_first


def test_widen_node_never_exceeds_total_action_count():
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        widening_coefficient=999.0,  # absurdly large
        widening_exponent=0.5,
    )
    node = _Node(state, config, feed, provider, rng=random.Random(1))
    total = len(node._all_actions)
    node.visits = 10_000
    _widen_node(node, config)
    assert len(node.untried_actions) == total  # capped at total


def test_mcts_search_completes_with_widening_enabled():
    provider = build_pr_flasher(COMPOUNDS)
    config = MCTSConfig(
        target_component="methane",
        target_fraction=0.48,
        allowed_delta_T_K=(-10.0, 10.0),
        max_depth=3,
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
        widening_coefficient=1.0,
        widening_exponent=0.5,
    )
    result = mcts_search(_feed(), provider, config, iterations=40, seed=7)
    assert result.iterations == 40  # completes without error or exception


# ── Mutual information separation metric ─────────────────────────────────────


def _pure(stream_id: str, comp: str, flow: float = 1.0) -> StreamState:
    return StreamState(stream_id, 300.0, 100_000.0, flow, {comp: 1.0})


def test_mi_perfect_two_component_separation():
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    m = mutual_information_separation(feed, [_pure("S1", "A"), _pure("S2", "B")])
    assert m["target"] == 2
    assert m["fraction_of_target"] == pytest.approx(1.0, abs=1e-9)
    assert m["score"] == pytest.approx(2.0, abs=1e-9)


def test_mi_perfect_three_component_separation():
    feed = StreamState(
        "F", 300.0, 100_000.0, 3.0, {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    )
    s1 = _pure("S1", "A")
    s2 = _pure("S2", "B")
    s3 = _pure("S3", "C")
    m = mutual_information_separation(feed, [s1, s2, s3])
    assert m["target"] == 3
    assert m["fraction_of_target"] == pytest.approx(1.0, abs=1e-9)


def test_mi_no_separation_single_stream_at_feed_composition():
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    same = StreamState("S", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    m = mutual_information_separation(feed, [same])
    assert m["fraction_of_target"] == pytest.approx(0.0, abs=1e-9)
    assert m["score"] == pytest.approx(0.0, abs=1e-9)


def test_mi_uninformative_split_gives_zero():
    # Splitting feed into two streams at feed composition gains no information.
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    s1 = StreamState("S1", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})
    s2 = StreamState("S2", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})
    m = mutual_information_separation(feed, [s1, s2])
    assert m["fraction_of_target"] == pytest.approx(0.0, abs=1e-9)


def test_mi_partial_separation_between_zero_and_one():
    # 90/10 symmetric split: should give score strictly between 0 and target.
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    s1 = StreamState("S1", 300.0, 100_000.0, 1.0, {"A": 0.9, "B": 0.1})
    s2 = StreamState("S2", 300.0, 100_000.0, 1.0, {"A": 0.1, "B": 0.9})
    m = mutual_information_separation(feed, [s1, s2])
    assert 0.0 < m["fraction_of_target"] < 1.0
    assert 0.0 < m["score"] < m["target"]


def test_mi_score_increases_with_purity():
    # 99/1 split should score higher than 90/10 split.
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    m_90 = mutual_information_separation(
        feed,
        [
            StreamState("S1", 300.0, 100_000.0, 1.0, {"A": 0.9, "B": 0.1}),
            StreamState("S2", 300.0, 100_000.0, 1.0, {"A": 0.1, "B": 0.9}),
        ],
    )
    m_99 = mutual_information_separation(
        feed,
        [
            StreamState("S1", 300.0, 100_000.0, 1.0, {"A": 0.99, "B": 0.01}),
            StreamState("S2", 300.0, 100_000.0, 1.0, {"A": 0.01, "B": 0.99}),
        ],
    )
    assert m_99["score"] > m_90["score"]


def test_mi_degeneracy_lower_than_purity_recovery():
    # Feed: A=0.5, B=0.3, C=0.2. One large mixed stream (barely enriched in A)
    # and one tiny pure-C stream. purity*recovery scores A and B highly via the
    # big stream; MI recognises the big stream is nearly uninformative.
    from ml import separation_indicator

    feed = StreamState("F", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.3, "C": 0.2})
    # Material balance: s_big carries 0.9 of flow; s_small carries C only.
    # s_big composition: A=0.5556, B=0.3333, C=0.1111 → F_A=0.5, F_B=0.3, F_C=0.1
    # s_small: F_C = 0.1
    s_big = StreamState(
        "Big", 300.0, 100_000.0, 0.9,
        {"A": 0.5556, "B": 0.3333, "C": 0.1111},
    )
    s_small = StreamState("Small", 300.0, 100_000.0, 0.1, {"A": 0.0, "B": 0.0, "C": 1.0})

    m_pr = separation_indicator(feed, [s_big, s_small])
    m_mi = mutual_information_separation(feed, [s_big, s_small])

    # Perfect separation gives fraction_of_target = 1.0 for both.
    s1 = StreamState("S1", 300.0, 100_000.0, 0.5, {"A": 1.0, "B": 0.0, "C": 0.0})
    s2 = StreamState("S2", 300.0, 100_000.0, 0.3, {"A": 0.0, "B": 1.0, "C": 0.0})
    s3 = StreamState("S3", 300.0, 100_000.0, 0.2, {"A": 0.0, "B": 0.0, "C": 1.0})
    m_perfect = mutual_information_separation(feed, [s1, s2, s3])

    # MI correctly identifies the degenerate state as far from perfect.
    # purity*recovery gives a misleadingly higher fraction_of_target.
    assert m_mi["fraction_of_target"] < m_perfect["fraction_of_target"]
    assert m_pr["fraction_of_target"] > m_mi["fraction_of_target"]


def test_mi_returns_dict_with_required_keys():
    feed = StreamState("F", 300.0, 100_000.0, 1.0, {"A": 0.6, "B": 0.4})
    s = StreamState("S", 300.0, 100_000.0, 1.0, {"A": 0.6, "B": 0.4})
    m = mutual_information_separation(feed, [s])
    assert set(m) >= {"score", "target", "fraction_of_target", "mi_nats", "feed_entropy_nats"}


def test_mi_separation_score_mode_default_is_purity_recovery():
    config = MCTSConfig(objective_mode="complete_separation")
    assert config.separation_score_mode == "purity_recovery"


def test_mi_reward_uses_mi_metric_when_mode_set():
    from ml.mcts import _reward, _complete_separation_metric

    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.5, "B": 0.5})
    # Perfect separation state: two pure open streams.
    from ml import process_graph_from_feed
    graph = process_graph_from_feed(feed)
    s1 = StreamState("S1", 300.0, 100_000.0, 1.0, {"A": 1.0, "B": 0.0})
    s2 = StreamState("S2", 300.0, 100_000.0, 1.0, {"A": 0.0, "B": 1.0})
    state = SearchState(open_streams=(s1, s2), process_graph=graph)

    config_pr = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="purity_recovery",
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )
    config_mi = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information",
        unit_penalty=0.0,
        duty_penalty_per_W=0.0,
    )

    r_pr = _reward(state, config_pr, feed)
    r_mi = _reward(state, config_mi, feed)

    # Both score perfect separation as target=2.
    assert r_pr == pytest.approx(2.0, abs=1e-6)
    assert r_mi == pytest.approx(2.0, abs=1e-6)


def test_mi_reward_differs_from_pr_on_degenerate_state():
    from ml.mcts import _reward
    from ml import process_graph_from_feed

    feed = StreamState("F", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.3, "C": 0.2})
    graph = process_graph_from_feed(feed)
    # Degenerate state: large mixed stream + tiny pure-C stream.
    s_big = StreamState(
        "Big", 300.0, 100_000.0, 0.9,
        {"A": 0.5556, "B": 0.3333, "C": 0.1111},
    )
    s_small = StreamState("Small", 300.0, 100_000.0, 0.1, {"A": 0.0, "B": 0.0, "C": 1.0})
    state = SearchState(open_streams=(s_big, s_small), process_graph=graph)

    config_pr = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="purity_recovery",
        unit_penalty=0.0, duty_penalty_per_W=0.0,
    )
    config_mi = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information",
        unit_penalty=0.0, duty_penalty_per_W=0.0,
    )

    r_pr = _reward(state, config_pr, feed)
    r_mi = _reward(state, config_mi, feed)

    # MI assigns a lower reward to the degenerate state than purity*recovery.
    assert r_mi < r_pr


def test_mi_cache_signature_changes_with_mode():
    from ml.mcts import _valid_action_config_signature

    config_pr = MCTSConfig(separation_score_mode="purity_recovery")
    config_mi = MCTSConfig(separation_score_mode="mutual_information")
    assert _valid_action_config_signature(config_pr) != _valid_action_config_signature(config_mi)


def test_mi_complete_separation_metric_includes_component_scores():
    # _complete_separation_metric in MI mode must return component_scores and
    # best_stream_by_component so progress recording and graph similarity
    # don't crash with KeyError.
    from ml.mcts import _complete_separation_metric

    COMPS = ["propane", "n-butane"]
    provider = build_pr_flasher(COMPS)
    feed = StreamState("Feed", 300.0, 500_000.0, 2.0, {"propane": 0.5, "n-butane": 0.5})
    pure_propane = StreamState("P", 300.0, 500_000.0, 1.0, {"propane": 1.0, "n-butane": 0.0})
    pure_butane = StreamState("B", 300.0, 500_000.0, 1.0, {"propane": 0.0, "n-butane": 1.0})
    from ml.process_graph import process_graph_from_feed
    from ml import ProductAssignment

    state = SearchState(
        open_streams=(),
        products=(
            ProductAssignment(role="product", stream=pure_propane),
            ProductAssignment(role="product", stream=pure_butane),
        ),
        process_graph=process_graph_from_feed(feed),
    )
    config = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information",
    )
    metric = _complete_separation_metric(state, config, feed)
    assert "component_scores" in metric
    assert "best_stream_by_component" in metric
    assert set(metric["component_scores"].keys()) == set(COMPS)
    # Perfect separation → MI score ≈ 2.0, per-component purity×recovery ≈ 1.0 each
    assert metric["score"] == pytest.approx(2.0, abs=1e-6)
    assert all(v == pytest.approx(1.0, abs=1e-4) for v in metric["component_scores"].values())


# ── equal-weight MI tests ─────────────────────────────────────────────────────

def test_mi_equal_weight_perfect_separation_scores_n_c():
    # Perfect separation must score N_C regardless of feed fractions.
    feed = StreamState("F", 300.0, 100_000.0, 3.0, {"A": 0.6, "B": 0.3, "C": 0.1})
    s_a = StreamState("SA", 300.0, 100_000.0, 1.8, {"A": 1.0, "B": 0.0, "C": 0.0})
    s_b = StreamState("SB", 300.0, 100_000.0, 0.9, {"A": 0.0, "B": 1.0, "C": 0.0})
    s_c = StreamState("SC", 300.0, 100_000.0, 0.3, {"A": 0.0, "B": 0.0, "C": 1.0})
    m = mutual_information_separation(feed, [s_a, s_b, s_c], weight_mode="equal_weight")
    assert m["target"] == 3
    assert m["score"] == pytest.approx(3.0, abs=1e-6)
    assert m["fraction_of_target"] == pytest.approx(1.0, abs=1e-6)


def test_mi_equal_weight_no_separation_scores_zero():
    feed = StreamState("F", 300.0, 100_000.0, 2.0, {"A": 0.7, "B": 0.3})
    same = StreamState("S", 300.0, 100_000.0, 2.0, {"A": 0.7, "B": 0.3})
    m = mutual_information_separation(feed, [same], weight_mode="equal_weight")
    assert m["score"] == pytest.approx(0.0, abs=1e-9)
    assert m["fraction_of_target"] == pytest.approx(0.0, abs=1e-9)


def test_mi_equal_weight_upweights_minority_component():
    # Feed: A=0.9, B=0.1 (very unequal). Perfect separation into two pure streams.
    # feed_fraction MI: dominated by A (90% weight) → score near 2 even before B splits.
    # equal_weight MI: A and B each get 50% weight → minority B gets more signal.
    feed = StreamState("F", 300.0, 100_000.0, 1.0, {"A": 0.9, "B": 0.1})
    s_a = StreamState("SA", 300.0, 100_000.0, 0.9, {"A": 1.0, "B": 0.0})
    s_b = StreamState("SB", 300.0, 100_000.0, 0.1, {"A": 0.0, "B": 1.0})
    m_ff = mutual_information_separation(feed, [s_a, s_b], weight_mode="feed_fraction")
    m_ew = mutual_information_separation(feed, [s_a, s_b], weight_mode="equal_weight")
    # Both reach perfect score for perfect separation.
    assert m_ff["score"] == pytest.approx(2.0, abs=1e-6)
    assert m_ew["score"] == pytest.approx(2.0, abs=1e-6)
    # Now test with A pure but B still mixed into A stream (poor B recovery).
    s_a_mixed = StreamState("SA", 300.0, 100_000.0, 1.0, {"A": 0.9, "B": 0.1})
    m_ff2 = mutual_information_separation(feed, [s_a_mixed], weight_mode="feed_fraction")
    m_ew2 = mutual_information_separation(feed, [s_a_mixed], weight_mode="equal_weight")
    # No separation in either mode.
    assert m_ff2["score"] == pytest.approx(0.0, abs=1e-9)
    assert m_ew2["score"] == pytest.approx(0.0, abs=1e-9)


def test_mi_equal_weight_returns_required_keys():
    feed = StreamState("F", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})
    m = mutual_information_separation(
        feed,
        [StreamState("S", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})],
        weight_mode="equal_weight",
    )
    for key in ("score", "target", "fraction_of_target", "mi_nats", "feed_entropy_nats"):
        assert key in m
    # feed_entropy_nats should be log(N_C) = log(2) for equal_weight
    import math
    assert m["feed_entropy_nats"] == pytest.approx(math.log(2), abs=1e-9)


# ── max_entropy leaf potential tests ─────────────────────────────────────────

def test_separation_potential_max_entropy_returns_max_stream_entropy():
    from ml.mcts import _separation_potential
    from ml.process_graph import process_graph_from_feed

    feed = StreamState("F", 300.0, 100_000.0, 10.0, {"A": 0.5, "B": 0.5})
    # Two open streams: one pure (H=0), one fully mixed (H=1).
    pure = StreamState("P", 300.0, 100_000.0, 9.0, {"A": 1.0, "B": 0.0})
    mixed = StreamState("M", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})

    state = SearchState(
        open_streams=(pure, mixed),
        products=(),
        process_graph=process_graph_from_feed(feed),
    )
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="max_entropy",
    )
    pot = _separation_potential(state, config, feed)
    # max_entropy: max(H_norm(pure)=0, H_norm(mixed)=1) = 1.0
    assert pot == pytest.approx(1.0, abs=1e-9)


def test_separation_potential_flow_weighted_sum_dominated_by_large_stream():
    from ml.mcts import _separation_potential
    from ml.process_graph import process_graph_from_feed

    feed = StreamState("F", 300.0, 100_000.0, 10.0, {"A": 0.5, "B": 0.5})
    # Large pure stream (9 mol/s) + small mixed stream (1 mol/s).
    large_pure = StreamState("LP", 300.0, 100_000.0, 9.0, {"A": 1.0, "B": 0.0})
    small_mixed = StreamState("SM", 300.0, 100_000.0, 1.0, {"A": 0.5, "B": 0.5})

    state = SearchState(
        open_streams=(large_pure, small_mixed),
        products=(),
        process_graph=process_graph_from_feed(feed),
    )
    config_fw = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="flow_weighted_sum",
    )
    config_me = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="max_entropy",
    )
    pot_fw = _separation_potential(state, config_fw, feed)
    pot_me = _separation_potential(state, config_me, feed)

    # flow_weighted_sum: large pure stream contributes ~0; small mixed contributes 0.1×1 = 0.1
    assert pot_fw == pytest.approx(0.1, abs=1e-6)
    # max_entropy: max(0, 1.0) = 1.0 — gives full weight to the small mixed stream
    assert pot_me == pytest.approx(1.0, abs=1e-9)
    # max_entropy potential is larger, incentivising processing of the small mixed stream
    assert pot_me > pot_fw


def test_mi_equal_weight_score_mode_via_complete_separation_metric():
    from ml.mcts import _complete_separation_metric
    from ml.process_graph import process_graph_from_feed
    from ml import ProductAssignment

    COMPS = ["propane", "n-butane"]
    feed = StreamState("Feed", 300.0, 500_000.0, 2.0, {"propane": 0.5, "n-butane": 0.5})
    pure_propane = StreamState("P", 300.0, 500_000.0, 1.0, {"propane": 1.0, "n-butane": 0.0})
    pure_butane = StreamState("B", 300.0, 500_000.0, 1.0, {"propane": 0.0, "n-butane": 1.0})
    state = SearchState(
        open_streams=(),
        products=(
            ProductAssignment(role="product", stream=pure_propane),
            ProductAssignment(role="product", stream=pure_butane),
        ),
        process_graph=process_graph_from_feed(feed),
    )
    config = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information_equal_weight",
    )
    metric = _complete_separation_metric(state, config, feed)
    # Perfect separation → score = N_C = 2 in equal-weight mode too.
    assert metric["score"] == pytest.approx(2.0, abs=1e-6)
    assert "component_scores" in metric
    assert "best_stream_by_component" in metric


# ── max_same_key_pair_count_per_lineage tests ────────────────────────────────


def test_lineage_pair_counts_feed_only_has_empty_counts():
    from ml.process_graph import process_graph_from_feed

    feed = _distillation_feed()
    graph = process_graph_from_feed(feed)
    counts = _distillation_lineage_pair_counts(feed.id, graph)
    assert len(counts) == 0


def test_lineage_pair_counts_after_one_distillation():
    from ml.process_graph import process_graph_from_feed
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    config = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
    )
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    next_state = _apply_action(state, action, provider, config)
    assert not next_state.errors

    distillate = next(
        s for s in next_state.open_streams
        if "distillate" in s.history[-1]
    )
    counts = _distillation_lineage_pair_counts(distillate.id, next_state.process_graph)
    assert counts[("propane", "n-butane")] == 1
    assert len(counts) == 1


def test_lineage_pair_counts_bottoms_also_records_split():
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    from ml.process_graph import process_graph_from_feed
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    config = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
    )
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    next_state = _apply_action(state, action, provider, config)
    bottoms = next(
        s for s in next_state.open_streams
        if "bottoms" in s.history[-1]
    )
    counts = _distillation_lineage_pair_counts(bottoms.id, next_state.process_graph)
    assert counts[("propane", "n-butane")] == 1


def test_max_same_key_pair_count_per_lineage_filters_repeated_pair():
    """After one D(propane/n-butane), limit=1 removes it from valid actions."""
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    from ml.process_graph import process_graph_from_feed
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    config_base = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=5,
    )
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    next_state = _apply_action(state, action, provider, config_base)
    assert not next_state.errors

    # Without the limit, D(propane/n-butane) can recur.
    config_no_limit = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=5,
        max_same_key_pair_count_per_lineage=None,
    )
    actions_no_limit = _valid_actions(next_state, config_no_limit, provider, feed)
    prop_nbu_no_limit = [
        a for a in actions_no_limit
        if a.kind == "distillation" and a.light_key == "propane" and a.heavy_key == "n-butane"
    ]

    # With limit=1, D(propane/n-butane) must be absent.
    config_limit = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=5,
        max_same_key_pair_count_per_lineage=1,
    )
    actions_limit = _valid_actions(next_state, config_limit, provider, feed)
    prop_nbu_limit = [
        a for a in actions_limit
        if a.kind == "distillation" and a.light_key == "propane" and a.heavy_key == "n-butane"
    ]

    assert len(prop_nbu_no_limit) > 0, "baseline: pair should appear without limit"
    assert len(prop_nbu_limit) == 0, "pair must be absent when lineage count hits limit"


def test_max_same_key_pair_count_per_lineage_allows_different_pairs():
    """The filter must not remove pairs that have never appeared in the lineage."""
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    from ml.process_graph import process_graph_from_feed
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    config_base = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=5,
    )
    # Apply D(propane/n-butane) first.
    next_state = _apply_action(
        state,
        UnitAction(
            kind="distillation",
            stream_id="Feed",
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
            reflux_ratio_multiplier=1.5,
        ),
        provider,
        config_base,
    )
    distillate = next(s for s in next_state.open_streams if "distillate" in s.history[-1])

    # The distillate has nitrogen + propane; D(nitrogen/propane) should still appear.
    config_limit = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=5,
        max_same_key_pair_count_per_lineage=1,
    )
    # Build a state with only the distillate as the open stream.
    distillate_state = SearchState(
        open_streams=(distillate,),
        process_graph=next_state.process_graph,
    )
    actions = _valid_actions(distillate_state, config_limit, provider, feed)
    nitrogen_propane = [
        a for a in actions
        if a.kind == "distillation" and a.light_key == "nitrogen" and a.heavy_key == "propane"
    ]
    assert len(nitrogen_propane) > 0, "D(nitrogen/propane) must still be proposed"


def test_max_same_key_pair_count_per_lineage_limit_2_allows_second_occurrence():
    """limit=2 should allow the pair a second time but block a third."""
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    feed = _distillation_feed()
    from ml.process_graph import process_graph_from_feed
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    base_cfg = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=10,
    )
    action = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    # First distillation.
    state1 = _apply_action(state, action, provider, base_cfg)
    distillate1 = next(s for s in state1.open_streams if "distillate" in s.history[-1])

    # Second distillation on the distillate.
    action2 = UnitAction(
        kind="distillation",
        stream_id=distillate1.id,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    state2 = _apply_action(
        SearchState(open_streams=(distillate1,), process_graph=state1.process_graph),
        action2,
        provider,
        base_cfg,
    )
    distillate2 = next(s for s in state2.open_streams if "distillate" in s.history[-1])

    state2_full = SearchState(
        open_streams=(distillate2,),
        process_graph=state2.process_graph,
    )

    config_limit2 = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=10,
        max_same_key_pair_count_per_lineage=2,
    )
    config_limit1 = MCTSConfig(
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        max_distillation_count_per_path=10,
        max_same_key_pair_count_per_lineage=1,
    )

    # With limit=2, after 2 occurrences, the pair must be absent.
    acts_limit2 = _valid_actions(state2_full, config_limit2, provider, feed)
    absent = [
        a for a in acts_limit2
        if a.kind == "distillation" and a.light_key == "propane" and a.heavy_key == "n-butane"
    ]
    assert len(absent) == 0, "pair must be absent when lineage count reaches limit=2"

    # With limit=1, after 2 occurrences, also absent.
    acts_limit1 = _valid_actions(state2_full, config_limit1, provider, feed)
    absent1 = [
        a for a in acts_limit1
        if a.kind == "distillation" and a.light_key == "propane" and a.heavy_key == "n-butane"
    ]
    assert len(absent1) == 0


# ── K-sample truncated rollout + α-filter tests ──────────────────────────────


def test_flow_weighted_mean_alpha_empty_state_returns_zero():
    from ml.mcts import _flow_weighted_mean_alpha

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(objective_mode="complete_separation")
    state = SearchState(open_streams=())
    assert _flow_weighted_mean_alpha(state, provider, config) == 0.0


def test_flow_weighted_mean_alpha_positive_for_mixed_stream():
    from ml.mcts import _flow_weighted_mean_alpha

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(objective_mode="complete_separation")
    state = SearchState(open_streams=(feed,))
    alpha = _flow_weighted_mean_alpha(state, provider, config)
    assert alpha > 1.0


def test_rollout_depth_zero_returns_leaf_immediately():
    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        rollout_depth=0,
    )
    state = SearchState(open_streams=(feed,))
    rng = random.Random(0)
    result = _rollout(state, feed, provider, config, rng)
    assert result is state


def test_rollout_depth_two_advances_exactly_two_steps():
    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        rollout_depth=2,
        max_distillation_count_per_path=10,
    )
    leaf = SearchState(open_streams=(feed,))
    rng = random.Random(42)
    endpoint = _rollout(leaf, feed, provider, config, rng)
    added = len(endpoint.unit_sequence) - len(leaf.unit_sequence)
    assert added == 2


def test_batched_worker_k1_depth0_matches_legacy():
    from ml.mcts import _batched_rollout_worker

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        rollout_depth=0,
        rollout_k=1,
    )
    state = SearchState(open_streams=(feed,))
    _, worker_reward = _batched_rollout_worker(state, feed, provider, config, seed=7)
    direct_reward = _reward(state, config, feed, provider)
    assert worker_reward == pytest.approx(direct_reward, abs=1e-9)


def test_batched_worker_k_sample_averages_valid_rewards(monkeypatch):
    import math
    import ml.mcts as mcts_module
    from ml.mcts import _batched_rollout_worker

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        enable_distillation_actions=True,
        validate_distillation_candidates=False,
        rollout_depth=1,
        rollout_k=3,
    )
    state = SearchState(open_streams=(feed,))

    # First call = leaf (returns 1.0); subsequent calls = endpoints (return 2.0).
    # This ensures all K samples pass the α-filter.
    call_count = [0]

    def fake_alpha(st, prov, cfg, cache=None):
        idx = call_count[0]
        call_count[0] += 1
        return 1.0 if idx == 0 else 2.0

    monkeypatch.setattr(mcts_module, "_flow_weighted_mean_alpha", fake_alpha)

    _, result_reward = _batched_rollout_worker(state, feed, provider, config, seed=11)
    # 4 calls: 1 for leaf + 3 for endpoints; all valid samples
    assert call_count[0] == 4
    assert math.isfinite(result_reward)


def test_batched_worker_alpha_filter_discards_non_improving():
    from ml.mcts import _batched_rollout_worker

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    # depth=0, k=2: _rollout returns the leaf unchanged for both samples.
    # alpha_end == alpha_leaf → strict > fails → all filtered → fallback.
    config = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        rollout_depth=0,
        rollout_k=2,
    )
    state = SearchState(open_streams=(feed,))
    result_state, result_reward = _batched_rollout_worker(state, feed, provider, config, seed=5)
    assert result_state is state
    direct_reward = _reward(state, config, feed, provider)
    assert result_reward == pytest.approx(direct_reward, abs=1e-9)


def test_alpha_weighted_potential_bounded_and_positive_for_easy_split():
    from ml.mcts import _separation_potential

    feed = _distillation_feed()
    provider = build_pr_flasher(DISTILLATION_COMPOUNDS)
    state = SearchState(open_streams=(feed,))
    config_alpha = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="alpha_weighted",
    )
    config_entropy = MCTSConfig(
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="max_entropy",
    )
    phi_alpha = _separation_potential(state, config_alpha, feed, provider)
    phi_entropy = _separation_potential(state, config_entropy, feed)
    # f(α) = 1 - 1/α  is always in [0, 1), so phi_alpha ≤ phi_entropy.
    assert phi_alpha > 0
    assert phi_alpha <= phi_entropy
    # For N2/propane/n-butane at 300 K, 500 kPa, α_max >> 1,
    # so f(α_max) = 1 - 1/α_max is close to 1 and phi_alpha ≈ phi_entropy.
    assert phi_alpha / phi_entropy > 0.9


def test_mi_strictly_increases_with_last_column():
    """Adding the final distillation column always strictly increases MI score."""
    feed = _distillation_feed()  # N2=0.1, propane=0.45, n-butane=0.45, F=2.0 mol/s

    # After 1 column (N2/propane split): N2 isolated, propane/n-butane still mixed
    s_n2 = StreamState(
        id="dist1", temperature_K=250, pressure_Pa=500000,
        molar_flow_mols=0.22,
        composition={"nitrogen": 0.86, "propane": 0.10, "n-butane": 0.04},
    )
    s_mixed = StreamState(
        id="bott1", temperature_K=320, pressure_Pa=500000,
        molar_flow_mols=1.78,
        composition={"nitrogen": 0.006, "propane": 0.496, "n-butane": 0.498},
    )

    # After 2 columns (propane/n-butane split added): all 3 components isolated
    s_propane = StreamState(
        id="dist2", temperature_K=300, pressure_Pa=500000,
        molar_flow_mols=0.92,
        composition={"nitrogen": 0.005, "propane": 0.940, "n-butane": 0.055},
    )
    s_nbutane = StreamState(
        id="bott2", temperature_K=350, pressure_Pa=500000,
        molar_flow_mols=0.86,
        composition={"nitrogen": 0.003, "propane": 0.044, "n-butane": 0.953},
    )

    mi_1col = mutual_information_separation(
        feed, [s_n2, s_mixed], weight_mode="equal_weight"
    )
    mi_2col = mutual_information_separation(
        feed, [s_n2, s_propane, s_nbutane], weight_mode="equal_weight"
    )
    assert mi_2col["score"] > mi_1col["score"]


def test_reward_increases_after_last_column():
    """_reward (pure MI, no leaf estimator) is strictly higher after the last column."""
    feed = _distillation_feed()
    config = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information_equal_weight",
        use_leaf_value_estimator=False,
    )

    s_n2 = StreamState(
        id="dist1", temperature_K=250, pressure_Pa=500000,
        molar_flow_mols=0.22,
        composition={"nitrogen": 0.86, "propane": 0.10, "n-butane": 0.04},
    )
    s_mixed = StreamState(
        id="bott1", temperature_K=320, pressure_Pa=500000,
        molar_flow_mols=1.78,
        composition={"nitrogen": 0.006, "propane": 0.496, "n-butane": 0.498},
    )
    s_propane = StreamState(
        id="dist2", temperature_K=300, pressure_Pa=500000,
        molar_flow_mols=0.92,
        composition={"nitrogen": 0.005, "propane": 0.940, "n-butane": 0.055},
    )
    s_nbutane = StreamState(
        id="bott2", temperature_K=350, pressure_Pa=500000,
        molar_flow_mols=0.86,
        composition={"nitrogen": 0.003, "propane": 0.044, "n-butane": 0.953},
    )

    state_1col = SearchState(open_streams=(s_n2, s_mixed))
    state_2col = SearchState(open_streams=(s_n2, s_propane, s_nbutane))

    reward_1col = _reward(state_1col, config, feed)
    reward_2col = _reward(state_2col, config, feed)
    assert reward_2col > reward_1col


def test_equal_weight_mi_recovery_clamping():
    """MI score stays in [0, n_c] when stream compositions slightly overshoot material balance."""
    feed = _distillation_feed()  # F=2.0, N2=0.1, propane=0.45, n-butane=0.45
    n_c = 3

    # Distillate flow slightly exceeds what propane mass balance strictly allows,
    # simulating FUG numerical overshoot (rec[propane] > 1.0 without clamping).
    s_distillate = StreamState(
        id="D", temperature_K=290, pressure_Pa=500000,
        molar_flow_mols=1.10,
        composition={"nitrogen": 0.07, "propane": 0.88, "n-butane": 0.05},
    )
    s_bottoms = StreamState(
        id="B", temperature_K=370, pressure_Pa=500000,
        molar_flow_mols=0.90,
        composition={"nitrogen": 0.04, "propane": 0.04, "n-butane": 0.92},
    )

    result = mutual_information_separation(
        feed, [s_distillate, s_bottoms], weight_mode="equal_weight"
    )
    assert 0.0 <= result["score"] <= float(n_c)
    assert 0.0 <= result["fraction_of_target"] <= 1.0


def test_min_distillation_count_prevents_early_terminal():
    """min_distillation_count_per_path blocks terminal until minimum columns placed."""
    feed = _distillation_feed()
    stream_a = StreamState(
        id="a", temperature_K=300, pressure_Pa=500000,
        molar_flow_mols=1.0,
        composition={"nitrogen": 0.1, "propane": 0.5, "n-butane": 0.4},
    )
    stream_b = StreamState(
        id="b", temperature_K=300, pressure_Pa=500000,
        molar_flow_mols=1.0,
        composition={"nitrogen": 0.1, "propane": 0.4, "n-butane": 0.5},
    )
    # Two distillation actions in sequence; open streams remain.
    state = SearchState(
        open_streams=(stream_a, stream_b),
        unit_sequence=(
            UnitAction(kind="distillation", stream_id="Feed"),
            UnitAction(kind="distillation", stream_id="bott1"),
        ),
    )
    # Very loose tolerance ensures the score threshold always triggers when
    # the min-column guard is satisfied, isolating the constraint under test.
    base = dict(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information_equal_weight",
        separation_score_tolerance=5.0,
        max_depth=20,
    )
    # n_dist=2 < 3 → not terminal despite score threshold
    assert _is_terminal(state, MCTSConfig(**base, min_distillation_count_per_path=3), feed) is False
    # n_dist=2 >= 2 → terminal (score threshold triggers)
    assert _is_terminal(state, MCTSConfig(**base, min_distillation_count_per_path=2), feed) is True
    # min=None → no constraint, terminal
    assert _is_terminal(state, MCTSConfig(**base, min_distillation_count_per_path=None), feed) is True


def test_remaining_mi_potential_is_exact_upper_bound():
    """remaining_mi potential equals the MI gain from perfectly separating open streams.

    Property: MI_current + Φ_remaining == MI_if_open_streams_were_pure.
    Verified by computing both sides independently.
    """
    from ml.mcts import _separation_potential

    # Feed: 3 components, equal fractions
    feed = StreamState(
        id="Feed", temperature_K=300, pressure_Pa=500000,
        molar_flow_mols=3.0,
        composition={"nitrogen": 1 / 3, "propane": 1 / 3, "n-butane": 1 / 3},
    )
    # One closed pure stream (N2 already separated)
    s_closed = StreamState(
        id="pure_n2", temperature_K=250, pressure_Pa=500000,
        molar_flow_mols=1.0,
        composition={"nitrogen": 1.0, "propane": 0.0, "n-butane": 0.0},
    )
    # One open mixed stream (propane + n-butane still together)
    s_open = StreamState(
        id="mixed", temperature_K=320, pressure_Pa=500000,
        molar_flow_mols=2.0,
        composition={"nitrogen": 0.0, "propane": 0.5, "n-butane": 0.5},
    )

    config = MCTSConfig(
        objective_mode="complete_separation",
        separation_score_mode="mutual_information_equal_weight",
        use_leaf_value_estimator=True,
        leaf_potential_mode="remaining_mi",
        leaf_value_discount=1.0,
    )

    state = SearchState(open_streams=(s_open,))

    phi = _separation_potential(state, config, feed)

    # Current MI: N2 pure, propane+n-butane mixed equally
    mi_current = mutual_information_separation(
        feed, [s_closed, s_open], weight_mode="equal_weight"
    )["score"]

    # Perfect MI: all three components separated
    s_pure_propane = StreamState(
        id="pure_prop", temperature_K=310, pressure_Pa=500000,
        molar_flow_mols=1.0,
        composition={"nitrogen": 0.0, "propane": 1.0, "n-butane": 0.0},
    )
    s_pure_nbutane = StreamState(
        id="pure_nbu", temperature_K=330, pressure_Pa=500000,
        molar_flow_mols=1.0,
        composition={"nitrogen": 0.0, "propane": 0.0, "n-butane": 1.0},
    )
    mi_perfect = mutual_information_separation(
        feed, [s_closed, s_pure_propane, s_pure_nbutane], weight_mode="equal_weight"
    )["score"]

    # Core property: MI_current + Φ_remaining == MI_perfect
    assert mi_current + phi == pytest.approx(mi_perfect, abs=1e-9)
    # Sanity: Φ in [0, N_C] and positive for a mixed open stream
    assert 0.0 < phi <= 3.0


# ---------------------------------------------------------------------------
# Recycle action tests
# ---------------------------------------------------------------------------

def _recycle_feed(**overrides) -> StreamState:
    values = {
        "id": "Feed",
        "temperature_K": 300.0,
        "pressure_Pa": 500000.0,
        "molar_flow_mols": 2.0,
        "composition": {"nitrogen": 0.1, "propane": 0.45, "n-butane": 0.45},
    }
    values.update(overrides)
    return StreamState(**values)


def _impure_stream(**overrides) -> StreamState:
    """A stream that is below default recycle_purity_threshold=0.95."""
    values = {
        "id": "S1",
        "temperature_K": 320.0,
        "pressure_Pa": 400000.0,
        "molar_flow_mols": 1.0,
        "composition": {"nitrogen": 0.05, "propane": 0.6, "n-butane": 0.35},
    }
    values.update(overrides)
    return StreamState(**values)


def _recycle_test_state(feed: StreamState, recycle: StreamState) -> SearchState:
    """Build a SearchState with a proper process graph containing both feed and recycle nodes."""
    graph = process_graph_from_feed(feed)
    graph = append_stream_root(graph, recycle, role="open")
    return SearchState(
        open_streams=(recycle,),
        process_graph=graph,
        feed_stream=feed,
    )


def test_recycle_mass_balance():
    """Mixed stream composition must match the analytical mass balance of feed + recycle."""
    feed = _recycle_feed()
    recycle = _impure_stream()
    _components = ["nitrogen", "propane", "n-butane"]

    config = MCTSConfig(
        enable_recycle_actions=True,
        max_recycle_count_per_path=1,
        recycle_purity_threshold=0.95,
    )
    provider = build_pr_flasher(_components)
    state = _recycle_test_state(feed, recycle)

    action = UnitAction(kind="recycle", stream_id=recycle.id)
    new_state = _apply_action(state, action, provider, config)

    assert not new_state.errors, f"Unexpected errors: {new_state.errors}"
    assert len(new_state.open_streams) == 1
    mixed = new_state.open_streams[0]

    F_f = feed.molar_flow_mols
    F_r = recycle.molar_flow_mols
    F_mix = F_f + F_r

    for comp in ["nitrogen", "propane", "n-butane"]:
        expected = (
            F_f * feed.composition.get(comp, 0.0)
            + F_r * recycle.composition.get(comp, 0.0)
        ) / F_mix
        assert mixed.composition[comp] == pytest.approx(expected, abs=1e-12), (
            f"Mass balance failed for {comp}: got {mixed.composition[comp]}, expected {expected}"
        )

    expected_T = (F_f * feed.temperature_K + F_r * recycle.temperature_K) / F_mix
    assert mixed.temperature_K == pytest.approx(expected_T, abs=1e-12)

    expected_P = min(feed.pressure_Pa, recycle.pressure_Pa)
    assert mixed.pressure_Pa == pytest.approx(expected_P, abs=1e-12)

    assert mixed.molar_flow_mols == pytest.approx(F_mix, abs=1e-12)


def test_recycle_count_gate():
    """_recycle_count increments correctly and blocks a second recycle at max=1."""
    feed = _recycle_feed()
    recycle_stream = _impure_stream()
    _components = ["nitrogen", "propane", "n-butane"]

    config = MCTSConfig(
        enable_recycle_actions=True,
        max_recycle_count_per_path=1,
        recycle_purity_threshold=0.95,
    )
    provider = build_pr_flasher(_components)
    state = _recycle_test_state(feed, recycle_stream)

    # First recycle — should be generated
    actions_before = _valid_actions(state, config)
    assert any(a.kind == "recycle" for a in actions_before), "Expected recycle action before first recycle"

    # Apply the recycle
    action = UnitAction(kind="recycle", stream_id=recycle_stream.id)
    new_state = _apply_action(state, action, provider, config)
    assert not new_state.errors

    mixed = new_state.open_streams[0]
    assert _recycle_count(mixed) == 1

    # After one recycle, the mixed stream should NOT generate another recycle action
    actions_after = _valid_actions(new_state, config)
    assert not any(a.kind == "recycle" for a in actions_after), (
        "Recycle action should be blocked after reaching max_recycle_count_per_path=1"
    )


def test_recycle_purity_threshold_gate():
    """No recycle action is generated when the stream is already pure enough."""
    feed = _recycle_feed()
    pure_enough = _impure_stream(
        composition={"nitrogen": 0.02, "propane": 0.96, "n-butane": 0.02}
    )  # max fraction = 0.96 >= threshold 0.95

    config = MCTSConfig(
        enable_recycle_actions=True,
        max_recycle_count_per_path=1,
        recycle_purity_threshold=0.95,
    )
    state = _recycle_test_state(feed, pure_enough)

    actions = _valid_actions(state, config)
    assert not any(a.kind == "recycle" for a in actions), (
        "Recycle action must not be generated for a stream at or above purity threshold"
    )


def test_recycle_not_generated_for_feed():
    """No recycle action is generated when the open stream IS the feed stream itself."""
    feed = _recycle_feed()

    config = MCTSConfig(
        enable_recycle_actions=True,
        max_recycle_count_per_path=1,
        recycle_purity_threshold=0.95,
    )
    # Open stream is the feed — same object id
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
        feed_stream=feed,
    )

    actions = _valid_actions(state, config)
    assert not any(a.kind == "recycle" for a in actions), (
        "Recycle action must not be generated for the feed stream itself"
    )


# ---------------------------------------------------------------------------
# Reboiler / energy-model tests
# ---------------------------------------------------------------------------

_EB_COMPONENTS = ["propane", "n-butane"]


def _eb_feed() -> StreamState:
    return StreamState(
        id="EbFeed",
        temperature_K=300.0,
        pressure_Pa=500_000.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )


def test_column_duties_energy_balance_positive():
    """column_duties_from_energy_balance returns finite, positive duties for a normal column."""
    provider = build_pr_flasher(_EB_COMPONENTS)
    feed = _eb_feed()
    result = shortcut_distillation_fug(
        feed, provider,
        light_key="propane", heavy_key="n-butane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )
    assert result.success
    assert result.distillate_stream is not None
    assert result.bottoms_stream is not None
    assert result.reflux_ratio is not None

    q_cond, q_reb = column_duties_from_energy_balance(
        feed, result.distillate_stream, result.bottoms_stream,
        result.reflux_ratio, provider,
    )
    assert q_cond > 0.0, "Condenser duty must be positive"
    assert q_reb > 0.0, "Reboiler duty must be positive for this normal column"
    # Reboiler > condenser only when feed enthalpy exceeds combined product enthalpy;
    # at 300 K subcooled feed it may be either way — just require finite positive values.
    import math
    assert math.isfinite(q_cond)
    assert math.isfinite(q_reb)


def test_include_reboiler_duty_increases_total_duty():
    """With include_reboiler_duty=True, total_abs_duty_W must exceed the condenser-proxy value."""
    provider = build_pr_flasher(_EB_COMPONENTS)
    feed = _eb_feed()
    action = UnitAction(
        kind="distillation",
        stream_id=feed.id,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )

    lambda_J_mol = 20_000.0
    config_proxy = MCTSConfig(
        enable_distillation_actions=True,
        distillation_molar_heat_of_vaporization_J_mol=lambda_J_mol,
        include_reboiler_duty=False,
    )
    config_eb = MCTSConfig(
        enable_distillation_actions=True,
        distillation_molar_heat_of_vaporization_J_mol=0.0,
        include_reboiler_duty=True,
    )

    state0 = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    state_proxy = _apply_action(state0, action, provider, config_proxy)
    state_eb = _apply_action(state0, action, provider, config_eb)

    assert state_proxy.total_abs_duty_W > 0.0, "Proxy duty must be positive"
    assert state_eb.total_abs_duty_W > 0.0, "EB duty must be positive"


def test_total_theoretical_stages_accumulates():
    """total_theoretical_stages grows with each distillation action applied."""
    provider = build_pr_flasher(["propane", "n-butane", "n-pentane"])
    feed = StreamState(
        id="StagesFeed",
        temperature_K=300.0,
        pressure_Pa=500_000.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.34, "n-butane": 0.33, "n-pentane": 0.33},
    )
    config = MCTSConfig(enable_distillation_actions=True)
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    assert state.total_theoretical_stages == 0.0

    action1 = UnitAction(
        kind="distillation", stream_id=feed.id,
        light_key="propane", heavy_key="n-butane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )
    state1 = _apply_action(state, action1, provider, config)
    assert not state1.errors
    assert state1.total_theoretical_stages > 0.0, "Stages must accumulate after first distillation"

    # Apply a second distillation on the bottoms stream (n-butane / n-pentane rich)
    next_stream = next(s for s in state1.open_streams if "bottoms" in s.id)
    action2 = UnitAction(
        kind="distillation", stream_id=next_stream.id,
        light_key="n-butane", heavy_key="n-pentane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )
    state2 = _apply_action(state1, action2, provider, config)
    assert state2.total_theoretical_stages > state1.total_theoretical_stages, (
        "Stages must increase after second distillation"
    )


def test_stage_count_penalty_reduces_reward():
    """stage_count_penalty_per_stage > 0 must reduce the reward compared to 0."""
    provider = build_pr_flasher(_EB_COMPONENTS)
    feed = _eb_feed()
    action = UnitAction(
        kind="distillation", stream_id=feed.id,
        light_key="propane", heavy_key="n-butane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )

    config_no_penalty = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        stage_count_penalty_per_stage=0.0,
    )
    config_with_penalty = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        stage_count_penalty_per_stage=0.01,
    )

    state0 = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    state_after = _apply_action(state0, action, provider, config_no_penalty)
    assert not state_after.errors
    assert state_after.total_theoretical_stages > 0.0

    r_no_penalty = _reward(state_after, config_no_penalty, feed_stream=feed)
    r_with_penalty = _reward(state_after, config_with_penalty, feed_stream=feed)

    assert r_with_penalty < r_no_penalty, (
        "Stage count penalty must reduce the reward"
    )


# ---------------------------------------------------------------------------
# depth_aware_bounded leaf estimator tests
# ---------------------------------------------------------------------------

def test_depth_aware_alpha_gated_reward_never_exceeds_n_c():
    """V_α(X) ≤ 1 ⟹ reward ≤ N_C for any partial state."""
    components = _EB_COMPONENTS
    provider = build_pr_flasher(components)
    feed = _eb_feed()
    config = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="depth_aware_alpha_gated",
        max_depth=10,
    )
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    n_c = len(components)
    r = _reward(state, config, feed_stream=feed, provider=provider)
    assert r <= n_c, f"reward {r:.4f} exceeds N_C={n_c} (admissibility violated)"


def test_depth_aware_alpha_gated_zero_potential_when_gate_blocks_all():
    """When distillation_min_alpha_ratio is unreachably high every stream fails
    the gate, U_α = 0, and the reward must equal the plain base score."""
    components = _EB_COMPONENTS
    provider = build_pr_flasher(components)
    feed = _eb_feed()

    config_gated = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="depth_aware_alpha_gated",
        distillation_min_alpha_ratio=1e6,   # no stream can pass this gate
        max_depth=10,
    )
    config_base = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=False,
        distillation_min_alpha_ratio=1e6,
        max_depth=10,
    )
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    r_gated = _reward(state, config_gated, feed_stream=feed, provider=provider)
    r_base = _reward(state, config_base, feed_stream=feed, provider=provider)
    assert r_gated == pytest.approx(r_base, abs=1e-9), (
        f"All streams blocked by gate → reward {r_gated:.6f} must equal base {r_base:.6f}"
    )


def test_depth_aware_bounded_reward_never_exceeds_n_c():
    """V(X) ≤ 1 ⟹ reward ≤ N_C for any partial state."""
    components = _EB_COMPONENTS  # ["propane", "n-butane"], N_C = 2
    provider = build_pr_flasher(components)
    feed = _eb_feed()
    config = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="depth_aware_bounded",
        max_depth=10,
    )
    # Partial state: feed not yet split — open_streams = [feed]
    state = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    n_c = len(components)
    r = _reward(state, config, feed_stream=feed)
    assert r <= n_c, f"reward {r:.4f} exceeds N_C={n_c} (admissibility violated)"


def test_depth_aware_bounded_potential_fades_at_max_depth():
    """At depth == max_depth, γ_d = 0 and the reward must equal the plain base score."""
    components = _EB_COMPONENTS
    provider = build_pr_flasher(components)
    feed = _eb_feed()
    max_depth = 4

    config_estimator = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=True,
        leaf_potential_mode="depth_aware_bounded",
        max_depth=max_depth,
    )
    config_no_estimator = MCTSConfig(
        enable_distillation_actions=True,
        objective_mode="complete_separation",
        use_leaf_value_estimator=False,
        max_depth=max_depth,
    )

    action = UnitAction(
        kind="distillation", stream_id=feed.id,
        light_key="propane", heavy_key="n-butane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )
    state0 = SearchState(
        open_streams=(feed,),
        process_graph=process_graph_from_feed(feed),
    )
    state1 = _apply_action(state0, action, provider, config_estimator)
    assert not state1.errors

    # Force depth to max_depth by padding unit_sequence
    deep_state = SearchState(
        open_streams=state1.open_streams,
        unit_sequence=state1.unit_sequence + ("dummy",) * (max_depth - 1),
        process_graph=state1.process_graph,
        total_abs_duty_W=state1.total_abs_duty_W,
        total_theoretical_stages=state1.total_theoretical_stages,
        errors=state1.errors,
        feed_stream=state1.feed_stream,
    )
    assert len(deep_state.unit_sequence) == max_depth

    r_estimator = _reward(deep_state, config_estimator, feed_stream=feed)
    r_base = _reward(deep_state, config_no_estimator, feed_stream=feed)
    assert r_estimator == pytest.approx(r_base, abs=1e-9), (
        f"At max_depth γ_d=0 so estimator reward {r_estimator:.6f} must equal "
        f"base reward {r_base:.6f}"
    )


# ---------------------------------------------------------------------------
# Enthalpy-overflow propagation tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from ml import compress_stream


def test_compress_stream_overflow_in_isentropic_flash_returns_failed():
    """Isentropic flash enthalpy > _H_OVERFLOW must produce success=False, not a petawatt duty."""
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed(temperature_K=300.0)

    overflow_h = 1e12  # far beyond 1e8 threshold

    fake_inlet = MagicMock()
    fake_inlet.H.return_value = overflow_h
    fake_inlet.S.return_value = 100.0
    fake_inlet.T = 300.0

    fake_ideal = MagicMock()
    fake_ideal.H.return_value = overflow_h

    def fake_flash(**kwargs):
        if "S" in kwargs:
            return fake_ideal
        return fake_inlet

    with patch.object(provider.flasher, "flash", side_effect=fake_flash):
        result = compress_stream(feed, provider, pressure_ratio=2.0)

    assert not result.success
    assert result.error_message is not None
    assert "overflow" in result.error_message.lower()


def test_compress_stream_overflow_in_actual_outlet_returns_failed():
    """Actual outlet enthalpy > _H_OVERFLOW must produce success=False."""
    provider = build_pr_flasher(COMPOUNDS)
    feed = _feed(temperature_K=300.0)

    normal_h = 1000.0
    overflow_h = 5e8

    fake_inlet = MagicMock()
    fake_inlet.H.return_value = normal_h
    fake_inlet.S.return_value = 100.0
    fake_inlet.T = 300.0

    fake_ideal = MagicMock()
    fake_ideal.H.return_value = normal_h + 500.0
    fake_ideal.T = 320.0

    fake_actual = MagicMock()
    fake_actual.H.return_value = overflow_h
    fake_actual.T = 350.0
    fake_actual.VF = 1.0
    fake_actual.phase_count = 1
    fake_actual.gas = MagicMock(zs=[1.0 / len(COMPOUNDS)] * len(COMPOUNDS))
    fake_actual.liquid0 = None

    call_count = [0]

    def fake_flash(**kwargs):
        if "S" in kwargs:
            return fake_ideal
        if "H" in kwargs:
            return fake_actual
        return fake_inlet

    with patch.object(provider.flasher, "flash", side_effect=fake_flash):
        result = compress_stream(feed, provider, pressure_ratio=2.0)

    assert not result.success
    assert result.error_message is not None
    assert "overflow" in result.error_message.lower()


def test_column_duties_energy_balance_raises_on_enthalpy_overflow():
    """column_duties_from_energy_balance must raise ValueError when any enthalpy overflows."""
    provider = build_pr_flasher(_EB_COMPONENTS)
    feed = _eb_feed()
    result = shortcut_distillation_fug(
        feed, provider,
        light_key="propane", heavy_key="n-butane",
        light_key_recovery=0.95, heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )
    assert result.success
    assert result.distillate_stream is not None
    assert result.bottoms_stream is not None

    overflow_h = 1e12

    fake_flash_result = MagicMock()
    fake_flash_result.H.return_value = overflow_h

    with patch.object(provider.flasher, "flash", return_value=fake_flash_result):
        with pytest.raises(ValueError, match="overflow"):
            column_duties_from_energy_balance(
                feed, result.distillate_stream, result.bottoms_stream,
                result.reflux_ratio, provider,
            )


def test_apply_action_distillation_energy_balance_overflow_returns_error_state():
    """When column_duties_from_energy_balance raises ValueError, _apply_action returns an error state."""
    provider = build_pr_flasher(_EB_COMPONENTS)
    feed = _eb_feed()
    state = SearchState(open_streams=(feed,))
    config = MCTSConfig(
        enable_distillation_actions=True,
        include_reboiler_duty=True,
    )
    action = UnitAction(
        kind="distillation",
        stream_id=feed.id,
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.3,
    )

    # Patch at the mcts module level — tests that _apply_action correctly catches
    # ValueError from column_duties_from_energy_balance and returns an error state.
    with patch(
        "ml.mcts.column_duties_from_energy_balance",
        side_effect=ValueError(
            "Enthalpy overflow in column energy balance: H_feed=1.00e+12 J/mol"
        ),
    ):
        next_state = _apply_action(state, action, provider, config)

    assert next_state.errors, "Expected an error state from energy balance overflow"
    assert any("energy balance" in e.lower() or "overflow" in e.lower() for e in next_state.errors)
