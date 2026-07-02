from __future__ import annotations

from ml import (
    MCTSConfig,
    ProductAssignment,
    SearchState,
    StreamState,
    UnitAction,
    action_signature,
    append_product_assignment,
    append_unit_operation,
    process_graph_from_feed,
    state_identity_hash,
    state_topology_hash,
)
from ml.mcts import mcts_progress_record


def _stream(stream_id: str, history: tuple[str, ...] = ()) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=300.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
        history=history,
    )


def test_action_signature_ignores_stream_id():
    left = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    right = UnitAction(kind="hx", stream_id="Other", delta_T_K=10.0)

    assert action_signature(left) == action_signature(right)


def test_state_identity_hash_ignores_generated_stream_ids():
    left_state = SearchState(
        open_streams=(
            _stream("Feed_hx_p10_1_vapor", ("hx", "flash:vapor")),
            _stream("Feed_hx_p10_1_liquid", ("hx", "flash:liquid")),
        ),
        unit_sequence=(
            UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0),
            UnitAction(kind="flash", stream_id="Feed_hx_p10_1"),
        ),
    )
    right_state = SearchState(
        open_streams=(
            _stream("F0_hx_p10_1_vapor", ("hx", "flash:vapor")),
            _stream("F0_hx_p10_1_liquid", ("hx", "flash:liquid")),
        ),
        unit_sequence=(
            UnitAction(kind="hx", stream_id="F0", delta_T_K=10.0),
            UnitAction(kind="flash", stream_id="F0_hx_p10_1"),
        ),
    )

    assert state_identity_hash(left_state) == state_identity_hash(right_state)


def test_graph_backed_topology_hash_ignores_stream_labels():
    action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    left_outlet = _stream("Feed_hx_p10_1", ("hx",))
    right_outlet = _stream("RenamedOutlet", ("hx",))

    left_graph = append_unit_operation(
        process_graph_from_feed(_stream("Feed")),
        "Feed",
        action,
        ((left_outlet, "out"),),
        action_signature(action),
    )
    right_graph = append_unit_operation(
        process_graph_from_feed(_stream("RenamedFeed")),
        "RenamedFeed",
        UnitAction(kind="hx", stream_id="RenamedFeed", delta_T_K=10.0),
        ((right_outlet, "out"),),
        action_signature(action),
    )

    left_state = SearchState(open_streams=(left_outlet,), process_graph=left_graph)
    right_state = SearchState(open_streams=(right_outlet,), process_graph=right_graph)

    assert state_topology_hash(left_state) == state_topology_hash(right_state)


def test_topology_hash_is_invariant_to_independent_branch_order():
    split = UnitAction(
        kind="distillation",
        stream_id="Feed",
        light_key="propane",
        heavy_key="n-butane",
        light_key_recovery=0.95,
        heavy_key_recovery=0.05,
        reflux_ratio_multiplier=1.5,
    )
    distillate = "Feed_dist_propane_over_n-butane_1_distillate"
    bottoms = "Feed_dist_propane_over_n-butane_1_bottoms"
    left_state = SearchState(
        open_streams=(
            _stream(f"{distillate}_hx_p10_2", ("shortcut_distillation:total_condenser_distillate", "hx")),
            _stream(f"{bottoms}_hx_p20_3", ("shortcut_distillation:bottoms", "hx")),
        ),
        unit_sequence=(
            split,
            UnitAction(kind="hx", stream_id=distillate, delta_T_K=10.0),
            UnitAction(kind="hx", stream_id=bottoms, delta_T_K=20.0),
        ),
    )
    right_state = SearchState(
        open_streams=(
            _stream(f"{bottoms}_hx_p20_2", ("shortcut_distillation:bottoms", "hx")),
            _stream(f"{distillate}_hx_p10_3", ("shortcut_distillation:total_condenser_distillate", "hx")),
        ),
        unit_sequence=(
            split,
            UnitAction(kind="hx", stream_id=bottoms, delta_T_K=20.0),
            UnitAction(kind="hx", stream_id=distillate, delta_T_K=10.0),
        ),
    )

    assert state_topology_hash(left_state) == state_topology_hash(right_state)


def test_graph_backed_topology_hash_is_invariant_to_independent_branch_order():
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
    distillate = _stream("Feed_D", ("shortcut_distillation:total_condenser_distillate",))
    bottoms = _stream("Feed_B", ("shortcut_distillation:bottoms",))
    distillate_hx = UnitAction(kind="hx", stream_id="Feed_D", delta_T_K=10.0)
    bottoms_hx = UnitAction(kind="hx", stream_id="Feed_B", delta_T_K=20.0)
    distillate_out = _stream("Feed_D_hot", distillate.history + ("hx",))
    bottoms_out = _stream("Feed_B_hot", bottoms.history + ("hx",))

    base = append_unit_operation(
        process_graph_from_feed(feed),
        "Feed",
        split,
        ((distillate, "distillate"), (bottoms, "bottoms")),
        action_signature(split),
    )
    left_graph = append_unit_operation(
        base,
        "Feed_D",
        distillate_hx,
        ((distillate_out, "out"),),
        action_signature(distillate_hx),
    )
    left_graph = append_unit_operation(
        left_graph,
        "Feed_B",
        bottoms_hx,
        ((bottoms_out, "out"),),
        action_signature(bottoms_hx),
    )

    right_graph = append_unit_operation(
        base,
        "Feed_B",
        bottoms_hx,
        ((bottoms_out, "out"),),
        action_signature(bottoms_hx),
    )
    right_graph = append_unit_operation(
        right_graph,
        "Feed_D",
        distillate_hx,
        ((distillate_out, "out"),),
        action_signature(distillate_hx),
    )

    left_state = SearchState(open_streams=(distillate_out, bottoms_out), process_graph=left_graph)
    right_state = SearchState(open_streams=(bottoms_out, distillate_out), process_graph=right_graph)

    assert state_topology_hash(left_state) == state_topology_hash(right_state)


def test_state_identity_hash_includes_terminal_stream_conditions():
    left_state = SearchState(open_streams=(_stream("Feed"),))
    hotter = StreamState(
        id="Feed",
        temperature_K=310.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )
    right_state = SearchState(open_streams=(hotter,))

    assert state_topology_hash(left_state) == state_topology_hash(right_state)
    assert state_identity_hash(left_state) != state_identity_hash(right_state)


def test_graph_backed_state_identity_hash_includes_terminal_stream_conditions():
    feed = _stream("Feed")
    hotter_feed = StreamState(
        id="Feed",
        temperature_K=310.0,
        pressure_Pa=101325.0,
        molar_flow_mols=1.0,
        composition={"propane": 0.5, "n-butane": 0.5},
    )
    graph = process_graph_from_feed(feed)
    left_state = SearchState(open_streams=(feed,), process_graph=graph)
    right_state = SearchState(open_streams=(hotter_feed,), process_graph=graph)

    assert state_topology_hash(left_state) == state_topology_hash(right_state)
    assert state_identity_hash(left_state) != state_identity_hash(right_state)


def test_product_role_changes_graph_backed_topology_hash():
    feed = _stream("Feed")
    product = ProductAssignment(role="Product", stream=feed)
    reject = ProductAssignment(role="Reject", stream=feed)
    product_state = SearchState(
        open_streams=(),
        products=(product,),
        process_graph=append_product_assignment(process_graph_from_feed(feed), "Feed", "Product"),
    )
    reject_state = SearchState(
        open_streams=(),
        products=(reject,),
        process_graph=append_product_assignment(process_graph_from_feed(feed), "Feed", "Reject"),
    )

    assert state_topology_hash(product_state) != state_topology_hash(reject_state)


def test_progress_record_includes_topology_and_state_identity_hashes():
    state = SearchState(
        open_streams=(),
        products=(ProductAssignment(role="Product", stream=_stream("Feed")),),
    )

    record = mcts_progress_record(
        1,
        1,
        0.0,
        state,
        1.0,
        MCTSConfig(target_component="propane", product_role="Product"),
        _stream("Feed"),
    )

    assert record["topology_hash"] == state_topology_hash(state)
    assert record["state_identity_hash"] == state_identity_hash(state)
