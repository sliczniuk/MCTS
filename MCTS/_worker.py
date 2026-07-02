"""Process worker for Phase 1 parallel data generation.

Must live in a proper module (not a notebook cell) because multiprocessing on
Windows uses the 'spawn' start method, which re-imports from scratch and cannot
access functions defined inside Jupyter cells.

Usage (from notebook):
    from concurrent.futures import ProcessPoolExecutor
    from study_leaf_estimator._worker import run_and_save

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        list(tqdm(pool.map(run_and_save, pending), total=len(pending)))
"""

from __future__ import annotations

import json
import pickle


def _unit_action_to_dict(action) -> dict:
    """Return a JSON-safe representation of an MCTS UnitAction."""
    return {
        "kind": getattr(action, "kind", None),
        "stream_id": getattr(action, "stream_id", None),
        "delta_T_K": getattr(action, "delta_T_K", None),
        "pressure_ratio": getattr(action, "pressure_ratio", None),
        "delta_P_Pa": getattr(action, "delta_P_Pa", None),
        "light_key": getattr(action, "light_key", None),
        "heavy_key": getattr(action, "heavy_key", None),
        "light_key_recovery": getattr(action, "light_key_recovery", None),
        "heavy_key_recovery": getattr(action, "heavy_key_recovery", None),
        "reflux_ratio_multiplier": getattr(action, "reflux_ratio_multiplier", None),
        "role": getattr(action, "role", None),
    }


def run_and_save(args: tuple) -> dict:
    """Run one MCTS search, save JSON, and save a training-node shard pickle.

    Called inside a subprocess — builds its own ThermoFlashProvider and
    extracts training nodes before returning, so the tree never needs to be
    pickled across process boundaries.

    Args:
        args: (system, method, condition, seed) 4-tuple, or
            (system, method, condition, seed, force_training_shard) when a
            schema change requires rebuilding training_data.pkl from new tree
            extracts even if the JSON summary already exists.

    Returns:
        Summary dict (same top-level fields as the JSON, minus "progress").
        Returns immediately from cached JSON if the result file already exists.
    """
    system, method, condition, seed = args[:4]
    force_training_shard = bool(args[4]) if len(args) > 4 else False

    # All imports deferred to avoid loading heavy modules in the main process
    # before forking, and to work correctly under spawn (Windows default).
    from dataclasses import replace as dc_replace

    from study_leaf_estimator.config import (
        SYSTEMS,
        METHOD_CONFIGS,
        MCTS_ITERATIONS,
        RESULTS_DIR,
        SUCCESS_THRESHOLD,
        make_feed_for_system,
        result_path,
    )
    from study_leaf_estimator.training_data import (
        TRAINING_SCHEMA_VERSION,
        extract_training_nodes,
    )
    from ml import build_pr_flasher, mcts_search

    out_path = result_path(method, condition, seed, system)
    shard_path = (
        RESULTS_DIR
        / f"_shard_{system}_{method}_{condition}_{seed:04d}.pkl"
    )
    if out_path.exists() and not force_training_shard:
        rec = json.loads(out_path.read_text())
        return {k: rec[k] for k in rec if k != "progress"}

    components = SYSTEMS[system]["components"]
    provider   = build_pr_flasher(components)
    # Adjust max distillation count to match this system's component count
    config = dc_replace(
        METHOD_CONFIGS[method],
        max_total_distillation_count=len(components) - 1,
    )
    feed     = make_feed_for_system(system, condition)
    is_uct   = method in {"full_rollout"}

    result = mcts_search(
        feed,
        provider,
        config,
        iterations=MCTS_ITERATIONS,
        seed=seed,
        progress_interval=10,
        return_tree=is_uct,
    )

    diag = result.diagnostics
    n_thermo_proxy = (
        diag.n_apply_action_cache_misses
        + diag.n_distillation_result_cache_misses
    )

    iter_to_success: int | None = None
    thermo_at_success: int | None = None
    for rec in result.progress:
        fot = rec.get("fraction_of_target")
        if fot is not None and fot >= SUCCESS_THRESHOLD:
            iter_to_success = int(rec["iteration"])
            thermo_at_success = int(
                rec.get("n_apply_action_cache_misses", 0)
                + rec.get("n_distillation_result_cache_misses", 0)
            )
            break

    last        = result.progress[-1] if result.progress else {}
    final_fot   = last.get("fraction_of_target")
    wall_time_s = last.get("elapsed_s")

    progress_compressed = [
        {
            "iteration":          int(r["iteration"]),
            "fraction_of_target": r.get("fraction_of_target"),
            "elapsed_s":          r.get("elapsed_s"),
            "n_thermo_proxy":     int(
                r.get("n_apply_action_cache_misses", 0)
                + r.get("n_distillation_result_cache_misses", 0)
            ),
        }
        for r in result.progress
    ]

    record = {
        "system":             system,
        "method":             method,
        "condition":          condition,
        "seed":               seed,
        "best_reward":        float(result.best_reward),
        "fraction_of_target": float(final_fot) if final_fot is not None else None,
        "success":            (final_fot is not None and final_fot >= SUCCESS_THRESHOLD),
        "iter_to_success":    iter_to_success,
        "thermo_at_success":  thermo_at_success,
        "sequence_kinds":     [a.kind for a in result.best_sequence],
        "best_sequence_actions": [
            _unit_action_to_dict(a) for a in result.best_sequence
        ],
        "topology_hash":      str(last.get("topology_hash", "")),
        "n_thermo_proxy":     n_thermo_proxy,
        "wall_time_s":        float(wall_time_s) if wall_time_s is not None else None,
        "iterations":         MCTS_ITERATIONS,
        "progress":           progress_compressed,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
    }
    out_path.write_text(json.dumps(record, indent=2))

    # Extract training nodes in-process — avoids pickling the tree across the
    # process boundary.
    if is_uct and result.tree_root is not None:
        nodes = extract_training_nodes(
            result.tree_root,
            provider=provider,
            config=config,
            feed_stream=feed,
            rv_cache=result.relative_volatility_cache,
            run_id=f"{system}__{method}__{condition}__{seed:04d}",
            run_metadata={
                "system": system,
                "method": method,
                "condition": condition,
                "seed": seed,
            },
        )
        if nodes:
            with open(shard_path, "wb") as f:
                pickle.dump(nodes, f)

    return {k: record[k] for k in record if k != "progress"}
