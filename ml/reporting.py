"""Reporting helpers for MCTS flowsheet results."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .flash import ThermoFlashProvider
from .graph_identity import state_identity_hash, state_topology_hash
from .graph_similarity import process_graph_similarity, suspicious_similarity_report
from .mcts import (
    BatchedMCTSResult,
    MCTSConfig,
    MCTSResult,
    SearchState,
    UnitAction,
    _apply_action,
)
from .process_graph import ProcessGraph, ProcessNode, process_graph_diagnostics
from .types import StreamState


MCTSLikeResult = MCTSResult | BatchedMCTSResult


@dataclass(frozen=True)
class FlowsheetGraph:
    """Replay-derived flowsheet graph.

    Args:
        text: Compact forked text tree.
        edges: Edge table as a pandas DataFrame when pandas is available,
            otherwise a list of dicts.
        streams: Stream table as a pandas DataFrame when pandas is available,
            otherwise a list of dicts.
        final_state: Final state obtained by replaying the sequence.
        errors: Non-fatal replay errors.

    Returns:
        Structured graph report for notebook display or logging.
    """

    text: str
    edges: Any
    streams: Any
    final_state: SearchState
    errors: tuple[str, ...] = ()


def mcts_diagnostics_table(
    results: Mapping[str, MCTSLikeResult] | Sequence[MCTSLikeResult],
    labels: Sequence[str] | None = None,
    baseline_label: str | None = None,
    as_dataframe: bool = True,
) -> Any:
    """Create a comparison table for MCTS run diagnostics.

    Args:
        results: Either a mapping of label -> MCTS result or a sequence of
            MCTS results.
        labels: Labels used when results is a sequence. Defaults to run_1,
            run_2, ...
        baseline_label: Optional label used to compute reward, elapsed-time,
            and expansion deltas. Defaults to the first row when omitted.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        table = mcts_diagnostics_table(
            {"baseline": base_result, "cached": cached_result}
        )
        display(table)
    """
    labelled_results = _normalise_labelled_results(results, labels)
    if not labelled_results:
        return _maybe_dataframe([], as_dataframe)

    baseline_index = 0
    if baseline_label is not None:
        labels_by_name = {label: index for index, (label, _) in enumerate(labelled_results)}
        if baseline_label not in labels_by_name:
            raise ValueError(
                f"baseline_label '{baseline_label}' was not found in result labels."
            )
        baseline_index = labels_by_name[baseline_label]

    baseline = labelled_results[baseline_index][1]
    baseline_elapsed_s = _result_elapsed_s(baseline)
    baseline_expanded = baseline.diagnostics.n_expanded_nodes
    baseline_reward = baseline.best_reward

    rows = []
    for label, result in labelled_results:
        diagnostics = result.diagnostics
        elapsed_s = _result_elapsed_s(result)
        rows.append(
            {
                "label": label,
                "best_reward": result.best_reward,
                "reward_delta_vs_baseline": result.best_reward - baseline_reward,
                "elapsed_s": elapsed_s,
                "elapsed_ratio_vs_baseline": _safe_ratio(elapsed_s, baseline_elapsed_s),
                "iterations": result.iterations,
                "sequence_length": len(result.best_sequence),
                "sequence_kinds": tuple(action.kind for action in result.best_sequence),
                "topology_hash": state_topology_hash(result.best_state),
                "state_identity_hash": state_identity_hash(result.best_state),
                "n_open_streams": len(result.best_state.open_streams),
                "n_products": len(result.best_state.products),
                "n_errors": len(result.best_state.errors),
                "n_expanded_nodes": diagnostics.n_expanded_nodes,
                "expanded_delta_vs_baseline": (
                    diagnostics.n_expanded_nodes - baseline_expanded
                ),
                "n_duplicate_states_skipped": diagnostics.n_duplicate_states_skipped,
                "duplicate_skip_rate": diagnostics.duplicate_skip_rate,
                "n_seen_state_identities": diagnostics.n_seen_state_identities,
                "n_apply_action_cache_hits": diagnostics.n_apply_action_cache_hits,
                "n_apply_action_cache_misses": diagnostics.n_apply_action_cache_misses,
                "apply_action_cache_hit_rate": diagnostics.apply_action_cache_hit_rate,
                "n_apply_action_cache_entries": diagnostics.n_apply_action_cache_entries,
                "apply_action_calc_time_s": diagnostics.apply_action_calc_time_s,
                "apply_action_cache_saved_estimate_s": (
                    diagnostics.apply_action_cache_saved_estimate_s
                ),
                "n_distillation_result_cache_hits": (
                    diagnostics.n_distillation_result_cache_hits
                ),
                "n_distillation_result_cache_misses": (
                    diagnostics.n_distillation_result_cache_misses
                ),
                "distillation_result_cache_hit_rate": (
                    diagnostics.distillation_result_cache_hit_rate
                ),
                "n_distillation_result_cache_entries": (
                    diagnostics.n_distillation_result_cache_entries
                ),
                "distillation_result_calc_time_s": (
                    diagnostics.distillation_result_calc_time_s
                ),
                "distillation_result_cache_saved_estimate_s": (
                    diagnostics.distillation_result_cache_saved_estimate_s
                ),
                "n_stream_priority_gating_calls": (
                    diagnostics.n_stream_priority_gating_calls
                ),
                "n_stream_priority_streams_considered": (
                    diagnostics.n_stream_priority_streams_considered
                ),
                "n_stream_priority_streams_gated": (
                    diagnostics.n_stream_priority_streams_gated
                ),
                "stream_priority_gate_rate": diagnostics.stream_priority_gate_rate,
                "n_valid_action_calls": diagnostics.n_valid_action_calls,
                "n_valid_action_cache_hits": diagnostics.n_valid_action_cache_hits,
                "n_valid_action_cache_misses": diagnostics.n_valid_action_cache_misses,
                "valid_action_cache_hit_rate": diagnostics.valid_action_cache_hit_rate,
                "n_valid_action_cache_entries": diagnostics.n_valid_action_cache_entries,
                "valid_action_generation_time_s": (
                    diagnostics.valid_action_generation_time_s
                ),
                "valid_action_cache_saved_estimate_s": (
                    diagnostics.valid_action_cache_saved_estimate_s
                ),
                "n_valid_actions_generated_total": (
                    diagnostics.n_valid_actions_generated_total
                ),
                "max_valid_actions_generated_per_call": (
                    diagnostics.max_valid_actions_generated_per_call
                ),
                "valid_actions_generated_by_kind": (
                    diagnostics.valid_actions_generated_by_kind
                ),
                "n_relative_volatility_cache_hits": (
                    diagnostics.n_relative_volatility_cache_hits
                ),
                "n_relative_volatility_cache_misses": (
                    diagnostics.n_relative_volatility_cache_misses
                ),
                "relative_volatility_cache_hit_rate": (
                    diagnostics.relative_volatility_cache_hit_rate
                ),
                "n_relative_volatility_cache_entries": (
                    diagnostics.n_relative_volatility_cache_entries
                ),
                "relative_volatility_calc_time_s": (
                    diagnostics.relative_volatility_calc_time_s
                ),
                "relative_volatility_cache_saved_estimate_s": (
                    diagnostics.relative_volatility_cache_saved_estimate_s
                ),
                "n_distillation_action_generation_calls": (
                    diagnostics.n_distillation_action_generation_calls
                ),
                "n_distillation_candidate_key_pairs": (
                    diagnostics.n_distillation_candidate_key_pairs
                ),
                "n_distillation_candidate_grid_actions": (
                    diagnostics.n_distillation_candidate_grid_actions
                ),
                "n_distillation_candidate_actions_generated": (
                    diagnostics.n_distillation_candidate_actions_generated
                ),
                "n_distillation_candidates_filtered_alpha": (
                    diagnostics.n_distillation_candidates_filtered_alpha
                ),
                "n_distillation_candidates_filtered_invalid_recovery": (
                    diagnostics.n_distillation_candidates_filtered_invalid_recovery
                ),
                "n_distillation_candidates_filtered_infeasible": (
                    diagnostics.n_distillation_candidates_filtered_infeasible
                ),
            }
        )

    return _maybe_dataframe(rows, as_dataframe)


def process_graph_diagnostics_table(
    items: Mapping[str, Any] | Sequence[Any],
    labels: Sequence[str] | None = None,
    reference: Any | None = None,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    as_dataframe: bool = True,
) -> Any:
    """Create a graph identity, validation, and similarity diagnostics table.

    Args:
        items: Mapping or sequence of MCTS results, SearchState objects, or
            ProcessGraph objects.
        labels: Labels used when items is a sequence. Defaults to graph_1,
            graph_2, ...
        reference: Optional result/state/graph used to compute similarity
            scores for every row.
        feed_stream: Optional feed stream used for separation-objective
            similarity.
        components: Optional component order for separation-objective
            similarity.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        table = process_graph_diagnostics_table(
            {"run_a": result_a, "run_b": result_b},
            reference=result_a,
        )
        display(table)
    """
    labelled_items = _normalise_labelled_items(items, labels, default_prefix="graph")
    rows: list[dict[str, object]] = []
    for label, item in labelled_items:
        state = _state_from_item(item)
        graph = _graph_from_item(item)
        issues = process_graph_diagnostics(
            graph,
            open_streams=state.open_streams if state is not None else (),
            products=state.products if state is not None else (),
        )
        row = _process_graph_summary_row(label, item, state, graph, issues)
        if reference is not None:
            similarity = process_graph_similarity(
                reference,
                item,
                feed_stream=feed_stream,
                components=components,
            )
            row.update(
                {
                    "similarity_overall": similarity.overall,
                    "similarity_topology": similarity.topology_similarity,
                    "similarity_terminal_streams": similarity.terminal_stream_similarity,
                    "similarity_objective": similarity.objective_similarity,
                    "similarity_branches": similarity.branch_similarity,
                    "similarity_units": similarity.unit_similarity,
                    "similarity_edge_roles": similarity.edge_role_similarity,
                    "similarity_product_roles": similarity.product_role_similarity,
                    "similarity_limitations": similarity.limitations,
                }
            )
        rows.append(row)
    return _maybe_dataframe(rows, as_dataframe)


def process_graph_cluster_table(
    items: Mapping[str, Any] | Sequence[Any],
    labels: Sequence[str] | None = None,
    *,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    config: MCTSConfig | None = None,
    composition_threshold: float = 0.995,
    condition_threshold: float = 0.98,
    topology_threshold: float = 0.95,
    as_dataframe: bool = True,
) -> Any:
    """Create exact-identity groups and nearest similarity diagnostics.

    Args:
        items: Mapping or sequence of MCTS results, SearchState objects, or
            ProcessGraph objects.
        labels: Labels used when items is a sequence. Defaults to item_1,
            item_2, ...
        feed_stream: Optional feed stream used to scale terminal stream flow.
        components: Optional component order for composition vectors.
        config: Optional MCTSConfig-like object used for T/P bounds when
            scaling stream condition vectors.
        composition_threshold: Minimum terminal composition cosine used by the
            suspicious-similarity classifier.
        condition_threshold: Minimum terminal condition cosine used by the
            suspicious-similarity classifier.
        topology_threshold: Minimum topology feature cosine used by the
            suspicious-similarity classifier.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        table = process_graph_cluster_table(
            {"baseline": result_a, "candidate": result_b},
            feed_stream=feed,
            components=["propane", "n-butane"],
        )
        display(table)
    """
    labelled_items = _normalise_labelled_items(items, labels, default_prefix="item")
    entries: list[dict[str, Any]] = []
    for label, item in labelled_items:
        state = _state_from_item(item)
        graph = _graph_from_item(item)
        issues = process_graph_diagnostics(
            graph,
            open_streams=state.open_streams if state is not None else (),
            products=state.products if state is not None else (),
        )
        entries.append(
            {
                "item": item,
                "row": _process_graph_summary_row(label, item, state, graph, issues),
            }
        )

    topology_groups = _group_ids(
        [entry["row"]["topology_hash"] for entry in entries],
        prefix="T",
    )
    state_groups = _group_ids(
        [entry["row"]["state_identity_hash"] for entry in entries],
        prefix="S",
    )

    for entry in entries:
        row = entry["row"]
        topology_group_id, topology_group_size = topology_groups[row["topology_hash"]]
        state_group_id, state_group_size = state_groups[row["state_identity_hash"]]
        row.update(
            {
                "topology_group_id": topology_group_id,
                "topology_group_size": topology_group_size,
                "state_group_id": state_group_id,
                "state_group_size": state_group_size,
            }
        )

    for index, entry in enumerate(entries):
        row = entry["row"]
        best: tuple[tuple[Any, ...], int, dict[str, Any]] | None = None
        for other_index, other_entry in enumerate(entries):
            if other_index == index:
                continue
            report = suspicious_similarity_report(
                entry["item"],
                other_entry["item"],
                components=components,
                config=config,
                feed_stream=feed_stream,
                composition_threshold=composition_threshold,
                condition_threshold=condition_threshold,
                topology_threshold=topology_threshold,
            )
            sort_key = _suspicious_neighbor_sort_key(report, other_index)
            if best is None or sort_key > best[0]:
                best = (sort_key, other_index, report)

        if best is None:
            row.update(_empty_neighbor_row())
            continue

        _, other_index, report = best
        stream_profile = report["stream_profile"]
        row.update(
            {
                "nearest_label": entries[other_index]["row"]["label"],
                "nearest_classification": report["classification"],
                "nearest_exact_duplicate": report["exact_duplicate"],
                "nearest_same_topology_hash": report["same_topology_hash"],
                "nearest_similar_streams": report["similar_streams"],
                "nearest_similar_topology_features": report[
                    "similar_topology_features"
                ],
                "nearest_min_composition_cosine": report[
                    "minimum_composition_cosine"
                ],
                "nearest_min_condition_cosine": report["minimum_condition_cosine"],
                "nearest_topology_feature_cosine": report[
                    "topology_feature_cosine"
                ],
                "nearest_matched_stream_count": stream_profile[
                    "matched_pair_count"
                ],
                "nearest_unmatched_stream_count": stream_profile[
                    "unmatched_stream_count"
                ],
            }
        )

    return _maybe_dataframe([entry["row"] for entry in entries], as_dataframe)


def process_graph_similarity_pairs(
    items: Mapping[str, Any] | Sequence[Any],
    labels: Sequence[str] | None = None,
    *,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    config: MCTSConfig | None = None,
    composition_threshold: float = 0.995,
    condition_threshold: float = 0.98,
    topology_threshold: float = 0.95,
    include_different: bool = False,
    as_dataframe: bool = True,
) -> Any:
    """Create pairwise suspicious-similarity diagnostics.

    Args:
        items: Mapping or sequence of MCTS results, SearchState objects, or
            ProcessGraph objects.
        labels: Labels used when items is a sequence. Defaults to item_1,
            item_2, ...
        feed_stream: Optional feed stream used to scale terminal stream flow.
        components: Optional component order for composition vectors.
        config: Optional MCTSConfig-like object used for T/P bounds when
            scaling stream condition vectors.
        composition_threshold: Minimum terminal composition cosine used by the
            suspicious-similarity classifier.
        condition_threshold: Minimum terminal condition cosine used by the
            suspicious-similarity classifier.
        topology_threshold: Minimum topology feature cosine used by the
            suspicious-similarity classifier.
        include_different: Include pairs classified as ``different``.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        pairs = process_graph_similarity_pairs(
            {"run_a": result_a, "run_b": result_b},
            feed_stream=feed,
        )
        display(pairs)
    """
    labelled_items = _normalise_labelled_items(items, labels, default_prefix="item")
    rows = _process_graph_similarity_pair_rows(
        labelled_items,
        feed_stream=feed_stream,
        components=components,
        config=config,
        composition_threshold=composition_threshold,
        condition_threshold=condition_threshold,
        topology_threshold=topology_threshold,
        include_different=include_different,
    )
    return _maybe_dataframe(rows, as_dataframe)


def process_graph_similarity_summary(
    items: Mapping[str, Any] | Sequence[Any],
    labels: Sequence[str] | None = None,
    *,
    feed_stream: StreamState | None = None,
    components: Sequence[str] | None = None,
    config: MCTSConfig | None = None,
    composition_threshold: float = 0.995,
    condition_threshold: float = 0.98,
    topology_threshold: float = 0.95,
) -> dict[str, object]:
    """Summarize pairwise suspicious-similarity diagnostics.

    Args:
        items: Mapping or sequence of MCTS results, SearchState objects, or
            ProcessGraph objects.
        labels: Labels used when items is a sequence. Defaults to item_1,
            item_2, ...
        feed_stream: Optional feed stream used to scale terminal stream flow.
        components: Optional component order for composition vectors.
        config: Optional MCTSConfig-like object used for T/P bounds when
            scaling stream condition vectors.
        composition_threshold: Minimum terminal composition cosine used by the
            suspicious-similarity classifier.
        condition_threshold: Minimum terminal condition cosine used by the
            suspicious-similarity classifier.
        topology_threshold: Minimum topology feature cosine used by the
            suspicious-similarity classifier.

    Returns:
        Plain dictionary with pair counts and diagnostic fractions.

    Example:
        summary = process_graph_similarity_summary(results, feed_stream=feed)
    """
    labelled_items = _normalise_labelled_items(items, labels, default_prefix="item")
    rows = _process_graph_similarity_pair_rows(
        labelled_items,
        feed_stream=feed_stream,
        components=components,
        config=config,
        composition_threshold=composition_threshold,
        condition_threshold=condition_threshold,
        topology_threshold=topology_threshold,
        include_different=True,
    )
    counts = {
        "exact_duplicate": 0,
        "same_topology_similar_streams": 0,
        "similar_streams_different_topology": 0,
        "similar_topology_different_streams": 0,
        "different": 0,
    }
    for row in rows:
        classification = str(row["classification"])
        counts[classification] = counts.get(classification, 0) + 1

    n_pairs = len(rows)
    n_suspicious_pairs = n_pairs - counts.get("different", 0)
    return {
        "n_items": len(labelled_items),
        "n_pairs": n_pairs,
        "n_exact_duplicate_pairs": counts.get("exact_duplicate", 0),
        "n_same_topology_similar_stream_pairs": counts.get(
            "same_topology_similar_streams",
            0,
        ),
        "n_similar_streams_different_topology_pairs": counts.get(
            "similar_streams_different_topology",
            0,
        ),
        "n_similar_topology_different_streams_pairs": counts.get(
            "similar_topology_different_streams",
            0,
        ),
        "n_different_pairs": counts.get("different", 0),
        "n_suspicious_pairs": n_suspicious_pairs,
        "exact_duplicate_pair_fraction": _safe_ratio(
            counts.get("exact_duplicate", 0),
            n_pairs,
        )
        or 0.0,
        "suspicious_pair_fraction": _safe_ratio(n_suspicious_pairs, n_pairs) or 0.0,
    }


def stream_table(
    streams: Sequence[StreamState],
    components: Sequence[str] | None = None,
    roles: dict[str, str] | None = None,
    include_history: bool = True,
    as_dataframe: bool = True,
) -> Any:
    """Create a stream summary table.

    Args:
        streams: Streams to report.
        components: Optional component order. Defaults to sorted components
            found in the streams.
        roles: Optional mapping from stream id to status/role label.
        include_history: Include compact stream history text when True.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        table = stream_table(result.best_state.open_streams)
        display(table)
    """
    component_order = tuple(components or _components_from_streams(streams))
    rows: list[dict[str, object]] = []
    role_map = roles or {}

    for stream in streams:
        dominant_component, dominant_x = _dominant_component(stream)
        row: dict[str, object] = {
            "id": stream.id,
            "role": role_map.get(stream.id, ""),
            "T_K": stream.temperature_K,
            "P_Pa": stream.pressure_Pa,
            "F_mol_s": stream.molar_flow_mols,
            "dominant_component": dominant_component,
            "dominant_x": dominant_x,
        }
        for component in component_order:
            row[f"x_{component}"] = stream.composition.get(component, 0.0)
        if include_history:
            row["history"] = " -> ".join(stream.history)
        rows.append(row)

    return _maybe_dataframe(rows, as_dataframe)


def stream_table_from_state(
    state: SearchState,
    components: Sequence[str] | None = None,
    include_history: bool = True,
    as_dataframe: bool = True,
) -> Any:
    """Create a stream table from open streams and accepted products.

    Args:
        state: MCTS search state.
        components: Optional component order.
        include_history: Include compact stream history text when True.
        as_dataframe: Return a pandas DataFrame when pandas is installed.

    Returns:
        pandas DataFrame when available and requested, otherwise list of dicts.

    Example:
        display(stream_table_from_state(result.best_state))
    """
    streams = [product.stream for product in state.products]
    streams.extend(state.open_streams)
    roles = {product.stream.id: product.role for product in state.products}
    for stream in state.open_streams:
        roles.setdefault(stream.id, "open")
    return stream_table(
        streams,
        components=components,
        roles=roles,
        include_history=include_history,
        as_dataframe=as_dataframe,
    )


def mcts_replay_graph(
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    sequence: Sequence[UnitAction],
    components: Sequence[str] | None = None,
    as_dataframe: bool = True,
) -> FlowsheetGraph:
    """Replay an MCTS sequence and build a forked flowsheet graph.

    Multi-output actions such as flash and shortcut distillation are represented
    as stream -> unit -> multiple child stream edges.

    Args:
        feed_stream: Initial feed stream.
        provider: Thermo provider used to replay actions.
        config: MCTS configuration used to replay actions.
        sequence: Unit action sequence to replay.
        components: Optional component order for the stream table.
        as_dataframe: Return pandas DataFrames for edges/streams when pandas is
            installed.

    Returns:
        FlowsheetGraph with text, edge table, stream table, and final state.

    Example:
        graph = mcts_replay_graph(feed, provider, config, result.best_sequence)
        print(graph.text)
        display(graph.streams)
    """
    state = SearchState(open_streams=(feed_stream,))
    known_streams: dict[str, StreamState] = {feed_stream.id: feed_stream}
    stream_roles: dict[str, str] = {feed_stream.id: "feed"}
    edges: list[dict[str, object]] = []
    adjacency: dict[str, list[str]] = {}
    unit_children: dict[str, list[str]] = {}
    node_labels: dict[str, str] = {feed_stream.id: _stream_label(feed_stream, "feed")}
    errors: list[str] = []

    for index, action in enumerate(sequence, start=1):
        before_open = {stream.id for stream in state.open_streams}
        before_products = len(state.products)
        input_stream = _find_stream(state, action.stream_id) or known_streams.get(
            action.stream_id
        )
        unit_id = f"U{index:02d}"
        node_labels[unit_id] = _action_label(index, action)
        edges.append(
            {
                "from": action.stream_id,
                "to": unit_id,
                "edge": "feed",
                "action_index": index,
                "action_kind": action.kind,
            }
        )
        adjacency.setdefault(action.stream_id, []).append(unit_id)

        next_state = _apply_action(state, action, provider, config)
        new_errors = next_state.errors[len(state.errors) :]
        errors.extend(new_errors)

        outputs: list[tuple[StreamState, str]] = []
        for stream in next_state.open_streams:
            if stream.id not in before_open:
                outputs.append((stream, "open"))
        for product in next_state.products[before_products:]:
            outputs.append((product.stream, product.role))

        if not outputs and input_stream is not None and new_errors:
            outputs.append((input_stream, "failed-open"))

        for output_index, (stream, role) in enumerate(outputs, start=1):
            known_streams[stream.id] = stream
            stream_roles[stream.id] = role
            node_labels[stream.id] = _stream_label(stream, role)
            edge_role = _output_role(action.kind, stream, role, output_index)
            edges.append(
                {
                    "from": unit_id,
                    "to": stream.id,
                    "edge": edge_role,
                    "action_index": index,
                    "action_kind": action.kind,
                }
            )
            unit_children.setdefault(unit_id, []).append(stream.id)

        state = next_state

    graph_text = _process_graph_text(state.process_graph, known_streams, stream_roles, sequence)
    graph_edges = _process_graph_edge_rows(state.process_graph)
    if graph_text:
        text = graph_text
    else:
        text = _render_tree(feed_stream.id, node_labels, adjacency, unit_children)
    if graph_edges:
        edges = graph_edges
    stream_rows = stream_table(
        list(known_streams.values()),
        components=components,
        roles=stream_roles,
        include_history=True,
        as_dataframe=as_dataframe,
    )
    return FlowsheetGraph(
        text=text,
        edges=_maybe_dataframe(edges, as_dataframe),
        streams=stream_rows,
        final_state=state,
        errors=tuple(errors),
    )


def _process_graph_edge_rows(graph: ProcessGraph) -> list[dict[str, object]]:
    if not graph.nodes:
        return []
    nodes = {node.id: node for node in graph.nodes}
    rows = []
    for edge in graph.edges:
        source = nodes.get(edge.source)
        target = nodes.get(edge.target)
        rows.append(
            {
                "from": edge.source,
                "to": edge.target,
                "from_label": source.label if source else edge.source,
                "to_label": target.label if target else edge.target,
                "edge": edge.role,
                "source_kind": source.kind if source else "",
                "target_kind": target.kind if target else "",
            }
        )
    return rows


def _process_graph_text(
    graph: ProcessGraph,
    known_streams: dict[str, StreamState],
    stream_roles: dict[str, str],
    sequence: Sequence[UnitAction],
) -> str:
    if not graph.nodes:
        return ""
    nodes = {node.id: node for node in graph.nodes}
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append((edge.role, edge.target))

    unit_labels: dict[str, str] = {}
    unit_index = 0
    for node in graph.nodes:
        if node.kind != "unit":
            continue
        unit_index += 1
        if unit_index <= len(sequence):
            unit_labels[node.id] = _action_label(unit_index, sequence[unit_index - 1])
        else:
            unit_labels[node.id] = f"{node.id}: {node.label}"

    def label_for(node: ProcessNode) -> str:
        if node.kind == "stream":
            stream = known_streams.get(node.label)
            role = stream_roles.get(node.label) or _node_data(node, "role") or ""
            if stream is not None:
                return _stream_label(stream, str(role))
            role_text = f" [{role}]" if role else ""
            return f"{node.label}{role_text}"
        if node.kind == "unit":
            return unit_labels.get(node.id, f"{node.id}: {node.label}")
        if node.kind == "product":
            return f"{node.label} [product]"
        return f"{node.id}: {node.label}"

    lines: list[str] = []

    def walk(node_id: str, prefix: str) -> None:
        child_edges = sorted(
            outgoing.get(node_id, ()),
            key=lambda item: (item[0], label_for(nodes[item[1]]) if item[1] in nodes else item[1]),
        )
        for index, (role, child_id) in enumerate(child_edges):
            child = nodes.get(child_id)
            if child is None:
                continue
            is_last = index == len(child_edges) - 1
            connector = "`-- " if is_last else "|-- "
            edge_label = f"[{role}] " if role else ""
            lines.append(f"{prefix}{connector}{edge_label}{label_for(child)}")
            next_prefix = f"{prefix}{'    ' if is_last else '|   '}"
            walk(child_id, next_prefix)

    for root_index, root_id in enumerate(graph.root_node_ids):
        root = nodes.get(root_id)
        if root is None:
            continue
        if root_index:
            lines.append("")
        lines.append(label_for(root))
        walk(root_id, "")
    return "\n".join(lines)


def _normalise_labelled_results(
    results: Mapping[str, MCTSLikeResult] | Sequence[MCTSLikeResult],
    labels: Sequence[str] | None,
) -> list[tuple[str, MCTSLikeResult]]:
    if isinstance(results, Mapping):
        if labels is not None:
            raise ValueError("labels must be omitted when results is a mapping.")
        return [(str(label), result) for label, result in results.items()]

    if labels is None:
        labels = tuple(f"run_{index}" for index in range(1, len(results) + 1))
    if len(labels) != len(results):
        raise ValueError("labels length must match results length.")
    return [(str(label), result) for label, result in zip(labels, results)]


def _normalise_labelled_items(
    items: Mapping[str, Any] | Sequence[Any],
    labels: Sequence[str] | None,
    default_prefix: str,
) -> list[tuple[str, Any]]:
    if isinstance(items, Mapping):
        if labels is not None:
            raise ValueError("labels must be omitted when items is a mapping.")
        return [(str(label), item) for label, item in items.items()]

    if labels is None:
        labels = tuple(f"{default_prefix}_{index}" for index in range(1, len(items) + 1))
    if len(labels) != len(items):
        raise ValueError("labels length must match items length.")
    return [(str(label), item) for label, item in zip(labels, items)]


def _process_graph_summary_row(
    label: str,
    item: Any,
    state: SearchState | None,
    graph: ProcessGraph,
    issues: list[dict[str, str]],
) -> dict[str, object]:
    nodes = graph.nodes
    unit_nodes = [node for node in nodes if node.kind == "unit"]
    stream_nodes = [node for node in nodes if node.kind == "stream"]
    product_nodes = [node for node in nodes if node.kind == "product"]
    row: dict[str, object] = {
        "label": label,
        "topology_hash": _topology_hash_for_item(item, state, graph),
        "state_identity_hash": _state_identity_hash_for_item(item, state, graph),
        "n_nodes": len(nodes),
        "n_edges": len(graph.edges),
        "n_roots": len(graph.root_node_ids),
        "n_stream_nodes": len(stream_nodes),
        "n_unit_nodes": len(unit_nodes),
        "n_product_nodes": len(product_nodes),
        "n_open_streams": len(state.open_streams) if state is not None else None,
        "n_products": len(state.products) if state is not None else None,
        "n_errors": len(state.errors) if state is not None else None,
        "graph_issue_count": len(issues),
        "graph_issue_codes": tuple(issue["code"] for issue in issues),
    }
    if isinstance(item, (MCTSResult, BatchedMCTSResult)):
        row.update(
            {
                "best_reward": item.best_reward,
                "iterations": item.iterations,
                "sequence_length": len(item.best_sequence),
                "sequence_kinds": tuple(action.kind for action in item.best_sequence),
                "n_expanded_nodes": item.diagnostics.n_expanded_nodes,
                "n_duplicate_states_skipped": item.diagnostics.n_duplicate_states_skipped,
            }
        )
    return row


def _topology_hash_for_item(item: Any, state: SearchState | None, graph: ProcessGraph) -> str:
    if state is not None:
        return state_topology_hash(state)
    return state_topology_hash(_GraphOnlyState(process_graph=graph))


def _state_identity_hash_for_item(
    item: Any,
    state: SearchState | None,
    graph: ProcessGraph,
) -> str:
    if state is not None:
        return state_identity_hash(state)
    return state_identity_hash(_GraphOnlyState(process_graph=graph))


@dataclass(frozen=True)
class _GraphOnlyState:
    process_graph: ProcessGraph
    open_streams: tuple[StreamState, ...] = ()
    products: tuple[Any, ...] = ()
    errors: tuple[str, ...] = ()


def _state_from_item(item: Any) -> SearchState | None:
    if isinstance(item, (MCTSResult, BatchedMCTSResult)):
        return item.best_state
    if isinstance(item, SearchState):
        return item
    return None


def _graph_from_item(item: Any) -> ProcessGraph:
    if isinstance(item, ProcessGraph):
        return item
    state = _state_from_item(item)
    if state is not None:
        return state.process_graph
    graph = getattr(item, "process_graph", None)
    if isinstance(graph, ProcessGraph):
        return graph
    return ProcessGraph.empty()


def _result_elapsed_s(result: MCTSLikeResult) -> float | None:
    if not result.progress:
        return None
    return float(result.progress[-1].get("elapsed_s", 0.0))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _group_ids(values: Sequence[object], prefix: str) -> dict[object, tuple[str, int]]:
    ids: dict[object, str] = {}
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if value not in ids:
            ids[value] = f"{prefix}{len(ids) + 1}"
    return {value: (group_id, counts[value]) for value, group_id in ids.items()}


def _process_graph_similarity_pair_rows(
    labelled_items: Sequence[tuple[str, Any]],
    *,
    feed_stream: StreamState | None,
    components: Sequence[str] | None,
    config: MCTSConfig | None,
    composition_threshold: float,
    condition_threshold: float,
    topology_threshold: float,
    include_different: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_index, (left_label, left_item) in enumerate(labelled_items):
        for right_label, right_item in labelled_items[left_index + 1 :]:
            report = suspicious_similarity_report(
                left_item,
                right_item,
                components=components,
                config=config,
                feed_stream=feed_stream,
                composition_threshold=composition_threshold,
                condition_threshold=condition_threshold,
                topology_threshold=topology_threshold,
            )
            classification = str(report["classification"])
            if classification == "different" and not include_different:
                continue
            stream_profile = report["stream_profile"]
            rows.append(
                {
                    "left_label": left_label,
                    "right_label": right_label,
                    "classification": classification,
                    "exact_duplicate": report["exact_duplicate"],
                    "same_topology_hash": report["same_topology_hash"],
                    "similar_streams": report["similar_streams"],
                    "similar_topology_features": report[
                        "similar_topology_features"
                    ],
                    "minimum_composition_cosine": report[
                        "minimum_composition_cosine"
                    ],
                    "minimum_condition_cosine": report["minimum_condition_cosine"],
                    "topology_feature_cosine": report["topology_feature_cosine"],
                    "matched_stream_count": stream_profile["matched_pair_count"],
                    "unmatched_stream_count": stream_profile[
                        "unmatched_stream_count"
                    ],
                    "recommendation": _similarity_recommendation(classification),
                }
            )
    return rows


def _similarity_recommendation(classification: str) -> str:
    return {
        "exact_duplicate": "already handled by exact duplicate pruning",
        "same_topology_similar_streams": (
            "inspect; possible future deprioritization candidate"
        ),
        "similar_streams_different_topology": "keep; convergent alternative",
        "similar_topology_different_streams": (
            "keep; topology alone is insufficient"
        ),
        "different": "ignore",
    }.get(classification, "inspect")


def _suspicious_neighbor_sort_key(
    report: dict[str, Any],
    candidate_index: int,
) -> tuple[int, int, float, float, float, int]:
    classification_priority = {
        "exact_duplicate": 5,
        "same_topology_similar_streams": 4,
        "similar_streams_different_topology": 3,
        "similar_topology_different_streams": 2,
        "different": 1,
    }
    return (
        classification_priority.get(str(report.get("classification")), 0),
        int(bool(report.get("same_topology_hash"))),
        _similarity_value(report.get("minimum_composition_cosine")),
        _similarity_value(report.get("minimum_condition_cosine")),
        _similarity_value(report.get("topology_feature_cosine")),
        -candidate_index,
    )


def _similarity_value(value: object) -> float:
    if value is None:
        return -1.0
    return float(value)


def _empty_neighbor_row() -> dict[str, object]:
    return {
        "nearest_label": None,
        "nearest_classification": None,
        "nearest_exact_duplicate": False,
        "nearest_same_topology_hash": False,
        "nearest_similar_streams": False,
        "nearest_similar_topology_features": False,
        "nearest_min_composition_cosine": None,
        "nearest_min_condition_cosine": None,
        "nearest_topology_feature_cosine": None,
        "nearest_matched_stream_count": 0,
        "nearest_unmatched_stream_count": 0,
    }


def _components_from_streams(streams: Sequence[StreamState]) -> tuple[str, ...]:
    components: set[str] = set()
    for stream in streams:
        components.update(stream.composition)
    return tuple(sorted(components))


def _dominant_component(stream: StreamState) -> tuple[str | None, float]:
    if not stream.composition:
        return None, 0.0
    component = max(stream.composition, key=stream.composition.get)
    return component, float(stream.composition[component])


def _find_stream(state: SearchState, stream_id: str) -> StreamState | None:
    for stream in state.open_streams:
        if stream.id == stream_id:
            return stream
    for product in state.products:
        if product.stream.id == stream_id:
            return product.stream
    return None


def _stream_label(stream: StreamState, role: str) -> str:
    dominant_component, dominant_x = _dominant_component(stream)
    suffix = ""
    if dominant_component is not None:
        suffix = f" | main={dominant_component} x={dominant_x:.3g}"
    role_text = f" [{role}]" if role else ""
    return (
        f"{stream.id}{role_text} | F={stream.molar_flow_mols:.6g} mol/s "
        f"| T={stream.temperature_K:.3g} K | P={stream.pressure_Pa:.6g} Pa"
        f"{suffix}"
    )


def _action_label(index: int, action: UnitAction) -> str:
    if action.kind == "hx":
        detail = f"dT={action.delta_T_K:g} K"
    elif action.kind in {"compressor", "pump", "valve"}:
        if action.delta_P_Pa is not None:
            detail = f"dP={action.delta_P_Pa:g} Pa"
        else:
            detail = f"ratio={action.pressure_ratio:g}"
    elif action.kind == "distillation":
        detail = (
            f"LK/HK={action.light_key}/{action.heavy_key}, "
            f"rec={action.light_key_recovery:g}/{action.heavy_key_recovery:g}, "
            f"R/Rmin={action.reflux_ratio_multiplier:g}"
        )
    elif action.kind == "flash":
        detail = "PT flash"
    elif action.kind == "accept":
        detail = f"role={action.role}"
    else:
        detail = ""
    return f"U{index:02d}: {action.kind}({detail})"


def _node_data(node: ProcessNode, key: str) -> Any:
    for item_key, value in node.data:
        if item_key == key:
            return value
    return None


def _output_role(
    action_kind: str,
    stream: StreamState,
    role: str,
    output_index: int,
) -> str:
    if action_kind == "flash":
        if stream.history and stream.history[-1] == "flash:vapor":
            return "vapor"
        if stream.history and stream.history[-1] == "flash:liquid":
            return "liquid"
    if action_kind == "distillation":
        if stream.history and stream.history[-1] == "shortcut_distillation:total_condenser_distillate":
            return "distillate"
        if stream.history and stream.history[-1] == "shortcut_distillation:bottoms":
            return "bottoms"
    if action_kind == "accept":
        return role
    return "out" if output_index == 1 else f"out{output_index}"


def _render_tree(
    root_id: str,
    node_labels: dict[str, str],
    adjacency: dict[str, list[str]],
    unit_children: dict[str, list[str]],
) -> str:
    lines = [node_labels[root_id]]

    def children(node_id: str) -> list[str]:
        if node_id.startswith("U"):
            return unit_children.get(node_id, [])
        return adjacency.get(node_id, [])

    def walk(node_id: str, prefix: str) -> None:
        child_ids = children(node_id)
        for index, child_id in enumerate(child_ids):
            is_last = index == len(child_ids) - 1
            connector = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{node_labels.get(child_id, child_id)}")
            next_prefix = f"{prefix}{'    ' if is_last else '|   '}"
            walk(child_id, next_prefix)

    walk(root_id, "")
    return "\n".join(lines)


def _maybe_dataframe(rows: list[dict[str, object]], as_dataframe: bool) -> Any:
    if not as_dataframe:
        return rows
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)
