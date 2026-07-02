"""MCTS search over simplified process-unit actions."""

from __future__ import annotations

import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from .compressor import compress_stream
from .distillation import column_duties_from_energy_balance, estimate_relative_volatilities, shortcut_distillation_fug
from .flash import ThermoFlashProvider, build_pr_flasher, flash_split
from .graph_identity import action_signature, state_identity_hash, state_topology_hash, stream_signature
from .hx import heat_stream
from .process_graph import (
    ProcessGraph,
    append_mixer_unit,
    append_product_assignment,
    append_stream_root,
    append_unit_operation,
    process_graph_from_feed,
)
from .pump import pump_stream
from .separation_metrics import mutual_information_separation, separation_indicator
from .stream_priority import rank_streams_by_priority, stream_priority, stream_composition_entropy
from .types import ShortcutDistillationResult, StreamState
from .valve import valve_stream


ObjectiveMode = Literal["single_product", "complete_separation"]
SeparationScoreMode = Literal[
    "purity_recovery",
    "mutual_information",
    "mutual_information_equal_weight",
]
LeafPotentialMode = Literal["flow_weighted_sum", "max_entropy", "alpha_weighted", "remaining_mi", "depth_aware_bounded", "depth_aware_alpha_gated"]
DistillationKeyPairMode = Literal["adjacent", "all"]
ProgressCallback = Callable[[dict[str, object]], None]
ActionKind = Literal[
    "hx",
    "flash",
    "compressor",
    "pump",
    "valve",
    "distillation",
    "accept",
    "recycle",
]


@dataclass(frozen=True)
class UnitAction:
    """Discrete unit action considered by MCTS.

    Args:
        kind: Action type: "hx", "flash", "compressor", "pump", "valve",
            "distillation", or "accept".
        stream_id: Open stream consumed by the action.
        delta_T_K: HX temperature change [K], required for "hx".
        pressure_ratio: Pressure ratio, required for "compressor" and "pump".
        delta_P_Pa: Pressure change magnitude [Pa]. It is a pressure increase
            for "compressor" and "pump", and a pressure drop for "valve".
        light_key: Distillation light key component.
        heavy_key: Distillation heavy key component.
        light_key_recovery: Fraction of light key recovered to distillate.
        heavy_key_recovery: Fraction of heavy key recovered to distillate.
        reflux_ratio_multiplier: Distillation R/R_min multiplier.
        role: Product role, required for "accept".

    Returns:
        Immutable action record used in search traces.

    Example:
        action = UnitAction(kind="hx", stream_id="Feed", delta_T_K=10.0)
    """

    kind: ActionKind
    stream_id: str
    delta_T_K: float | None = None
    pressure_ratio: float | None = None
    delta_P_Pa: float | None = None
    light_key: str | None = None
    heavy_key: str | None = None
    light_key_recovery: float | None = None
    heavy_key_recovery: float | None = None
    reflux_ratio_multiplier: float | None = None
    role: str | None = None


@dataclass(frozen=True)
class ProductAssignment:
    """Accepted product stream assignment."""

    role: str
    stream: StreamState


@dataclass(frozen=True)
class SearchState:
    """MCTS state for simplified flowsheet synthesis.

    Args:
        open_streams: Streams still available for processing.
        products: Accepted products.
        unit_sequence: Actions applied to reach this state.
        total_abs_duty_W: Sum of absolute HX duties [W].
        errors: Non-fatal action failures on the path.
        process_graph: Explicit topology graph used for canonical graph/state
            identity and duplicate pruning.

    Returns:
        Immutable state record.
    """

    open_streams: tuple[StreamState, ...]
    products: tuple[ProductAssignment, ...] = field(default_factory=tuple)
    unit_sequence: tuple[UnitAction, ...] = field(default_factory=tuple)
    total_abs_duty_W: float = 0.0
    total_theoretical_stages: float = 0.0
    errors: tuple[str, ...] = field(default_factory=tuple)
    process_graph: ProcessGraph = field(default_factory=ProcessGraph.empty)
    feed_stream: StreamState | None = None


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for v1 MCTS unit-order search.

    Args:
        target_component: Component whose product mole fraction is targeted.
        target_fraction: Target mole fraction in the accepted product.
        product_role: Product role to create, e.g. "CooledLiquid".
        allowed_delta_T_K: Discrete HX temperature changes [K].
        allowed_compression_ratios: Discrete compressor P_out/P_in values.
            Empty disables compressor actions.
        allowed_compression_delta_P_Pa: Discrete compressor pressure increases
            [Pa]. Empty disables compressor delta-P actions.
        allowed_pump_pressure_ratios: Discrete pump P_out/P_in values.
            Empty disables pump actions.
        allowed_pump_delta_P_Pa: Discrete pump pressure increases [Pa]. Empty
            disables pump delta-P actions.
        allowed_valve_pressure_ratios: Discrete valve P_out/P_in values.
            Values must be between 0 and 1. Empty disables valve ratio actions.
        allowed_valve_delta_P_Pa: Discrete valve pressure drops [Pa]. Empty
            disables valve delta-P actions.
        enable_distillation_actions: If True, generate shortcut distillation
            actions from feasible volatility key pairs.
        distillation_light_key_recoveries: Candidate LK recoveries to distillate.
        distillation_heavy_key_recoveries: Candidate HK recoveries to distillate.
        distillation_reflux_multipliers: Candidate R/R_min multipliers.
        distillation_key_pair_mode: "adjacent" proposes adjacent key pairs by
            relative-volatility order. "all" proposes every feasible lighter /
            heavier key pair.
        validate_distillation_candidates: If True, run the shortcut column model
            during action generation to filter infeasible candidates. Disable
            for very broad searches where rollout/application penalties should
            handle failed column candidates.
        distillation_min_key_flow_mols: Minimum component molar flow for LK/HK.
        distillation_min_alpha_ratio: Minimum alpha_LK/alpha_HK for a candidate
            adjacent key pair.
        distillation_max_theoretical_stages: Maximum Gilliland stages retained
            during action generation and application.
        max_distillation_count_per_path: Maximum shortcut columns allowed in a
            stream history.
        max_same_key_pair_count_per_lineage: Maximum number of times the same
            (light_key, heavy_key) pair may appear in the ancestor lineage of
            an open stream. When set, distillation actions that would exceed
            this count are suppressed during action generation. ``None``
            (default) disables the constraint. Set to 1 to prevent any
            repeated key pairs, which structurally forces distinct separations
            at each depth level and prevents the MCTS from getting stuck in
            degenerate solutions that reuse the same split repeatedly.
        compressor_isentropic_efficiency: Compressor isentropic efficiency.
        compressor_mechanical_efficiency: Compressor mechanical efficiency.
        pump_isentropic_efficiency: Pump isentropic efficiency.
        pump_mechanical_efficiency: Pump mechanical efficiency.
        pump_max_inlet_vapor_fraction: Maximum inlet vapor fraction accepted
            by pump actions.
        min_pressure_Pa: Lower pressure bound for valve actions.
        max_pressure_Pa: Upper pressure bound for compressor and pump actions.
        target_product_temperature_K: Optional required product temperature [K].
        product_temperature_tolerance_K: Acceptance tolerance for product
            temperature when target_product_temperature_K is set.
        min_temperature_K: Lower temperature bound for HX actions.
        max_temperature_K: Upper temperature bound for HX actions.
        min_flow_mols: Minimum stream flow retained in open_streams.
        max_active_streams_per_state: Optional cap on the number of open
            streams that can receive processing actions during valid-action
            generation. Accept actions are still generated when valid.
        min_stream_priority: Minimum flow-weighted composition-entropy priority
            required for a stream to receive processing actions. Defaults to
            zero, preserving current behavior.
        use_leaf_value_estimator: If True, replace random rollouts with a
            closed-form leaf value estimate combining the current separation
            score with a discounted stream-priority potential. Only active in
            ``complete_separation`` mode.
        leaf_value_discount: Discount applied to the stream-priority potential
            in the leaf value estimate. ``None`` (default) sets it automatically
            to ``0.5``, meaning "at most 50% extra optimism above S_norm" in the
            normalised [0, 1] reward space. Pass an explicit float to override.
            Ignored when ``use_leaf_value_estimator`` is False.
        leaf_potential_mode: How to compute the stream-priority potential used
            in the leaf value estimator.
            ``"flow_weighted_sum"`` (default) sums flow-weighted composition
            entropy across all open streams — large streams dominate.
            ``"max_entropy"`` takes the maximum normalised composition entropy
            over all open streams, giving equal attention to any single
            highly-mixed stream regardless of its flow rate. This prevents
            small but important streams (e.g. a C6/C7 bottoms at 13 % of feed)
            from being overshadowed by larger streams.
            ``"alpha_weighted"`` multiplies each stream's flow-weighted entropy
            by the best relative volatility α available for that stream at its
            current T and P. This makes the potential sensitive to thermodynamic
            conditioning: a cooled stream gains credit because its α is higher.
            Requires ``provider`` to be passed to ``_reward``; falls back to
            ``"max_entropy"`` if provider is unavailable.
            ``"depth_aware_bounded"`` computes
            ``V = S + min(U, 1 − S)`` where S = fraction_of_target
            (normalized MI ∈ [0,1]) and U = Σ_k (F_k/F_0) H_norm(z_k) is the
            normalized remaining-MI potential ∈ [0,1].  The cap min(U, 1−S)
            ensures V ≤ 1 (reward ≤ N_C) — admissibility holds without a depth
            discount because S + (1−S) = 1 already.  ``leaf_value_discount``
            is ignored for this mode.
            ``"depth_aware_alpha_gated"`` extends ``"depth_aware_bounded"``
            with a soft separability gate: each open stream's entropy
            contribution is scaled by min(1, α_max / α_threshold), giving
            partial credit to streams that are below the hard distillation
            threshold but could be unlocked by pressure change. A hard binary
            gate gave zero credit to compressed-but-not-yet-distilled
            intermediates, making compression appear worthless to the estimator.
            Requires ``provider``; falls back to ``"depth_aware_bounded"`` (no
            gate) when provider is unavailable.
            Ignored when ``use_leaf_value_estimator`` is False.
        rollout_depth: Number of random simulation steps to take from each leaf
            before evaluating the leaf estimator reward. ``0`` (default) returns
            the leaf state immediately — pure leaf estimator, no thermodynamic
            calls during rollout. Positive values enable truncated rollouts:
            exactly ``rollout_depth`` actions are applied, then the estimator is
            evaluated on the resulting state. Only meaningful when
            ``use_leaf_value_estimator`` is True and
            ``objective_mode`` is ``"complete_separation"``.
        rollout_k: Number of independent truncated rollouts to average per leaf
            visit. ``1`` (default) is a single sample — current behaviour.
            Values greater than 1 enable K-sample averaging with an α-filter:
            each sample is evaluated independently; only samples where the
            flow-weighted mean relative volatility of open streams improved over
            the leaf are included in the average. Falls back to the pure leaf
            estimator when all K samples are filtered out. Each sample uses a
            deterministic seed offset for reproducibility.
        distillation_molar_heat_of_vaporization_J_mol: Molar heat of
            vaporization [J/mol] used to estimate condenser duty for each
            shortcut distillation column as ``R * D * lambda``. Zero (default)
            disables distillation duty tracking.
        max_depth: Maximum number of actions in a sequence.
        max_flash_count_per_path: Maximum flashes allowed in a stream history.
        exploration_weight: UCT exploration coefficient.  Ignored when
            ``use_thompson_sampling`` is True.
        use_thompson_sampling: If True, replace the UCT selection rule with
            Thompson Sampling.  Each child's expected value is modelled as a
            Beta posterior — Beta(1 + V, 1 + n − V) where V is the accumulated
            reward and n the visit count — and selection draws one sample per
            child, picking the argmax.  The uninformative prior Beta(1, 1) is
            used for unvisited children, giving them naturally high exploration
            probability without a separate "expand untried first" rule.
            Meaningful only when rewards are stochastic (full_rollout,
            truncated_rollout); for deterministic leaf estimators (score_only,
            bounded_depth_aware, fug_gated) UCT and Thompson Sampling are
            equivalent and UCT is cheaper.
        unit_penalty: Reward penalty per action.
        duty_penalty_per_W: Reward penalty per W of absolute HX duty.
        missing_product_penalty: Penalty when terminal state has no product.
        require_flash_liquid_product: If True, accept only streams descended
            from a flash liquid outlet.
        candidate_eval_width: Number of untried actions to pre-score at
            expansion. Zero disables parallel candidate evaluation.
        candidate_rollouts_per_action: Rollout samples used to score each
            candidate action.
        candidate_eval_workers: Thread workers used during candidate scoring.
        objective_mode: "single_product" keeps the original target-component
            objective. "complete_separation" rewards purity/recovery across all
            meaningful feed components.
        separation_score_mode: Score function used in complete_separation mode.
            "purity_recovery" (default) sums max(purity * recovery) per
            component. "mutual_information" uses N_C * I(C;K) / H(C), a
            weight-free information-theoretic metric that naturally penalises
            degenerate states where one stream ranks first for multiple
            components. Both return scores in [0, N_C]. See
            ``mutual_information_separation`` for details.
        separation_score_tolerance: Terminal tolerance below the ideal
            complete-separation score.
        min_component_fraction: Minimum feed mole fraction included in
            complete-separation scoring.
        enable_exact_duplicate_pruning: If True, skip expanding child states
            whose canonical state identity hash has already been seen in the
            current MCTS tree.
        enable_apply_action_cache: If True, cache deterministic unit
            calculation results by canonical inlet stream and action signature.
        cached_action_kinds: Action kinds eligible for apply-action caching.
        max_apply_action_cache_entries: Optional maximum number of cached
            unit-action outcomes retained during one search.
        enable_action_generation_cache: If True, cache deterministic valid
            action lists and relative-volatility estimates during action
            generation.
        max_valid_action_cache_entries: Optional maximum number of cached
            valid-action lists retained during one search.
        max_relative_volatility_cache_entries: Optional maximum number of
            cached relative-volatility estimates retained during one search.
        product_purity_threshold: Dominant-component mole fraction at or above
            which a stream is considered pure enough and receives no further
            actions.  Set to 1.0 (default) to disable — all open streams are
            always eligible for processing.  When enabled, streams meeting the
            threshold are auto-closed, reducing the branching factor at every
            node where some streams are already near-pure.  Streams below the
            threshold are never auto-closed: pressure/temperature-changing
            actions (compressor, valve, HX) remain available even when
            distillation is blocked by an insufficient α ratio, preserving the
            pressure-swing pathway.  Flash actions are additionally gated on
            the stream having a two-phase vapour fraction (0 < VF < 1) at
            current T/P — single-phase streams carry no flash separation
            potential regardless of the threshold.

    Returns:
        Search configuration for mcts_search().

    Example:
        config = MCTSConfig(
            target_component="methane",
            target_fraction=0.48,
            target_product_temperature_K=110.0,
        )
    """

    target_component: str = ""
    target_fraction: float = 1.0
    product_role: str = "CooledLiquid"
    allowed_delta_T_K: tuple[float, ...] = (
        -40.0,
        -30.0,
        -20.0,
        -10.0,
        10.0,
        20.0,
        30.0,
        40.0,
    )
    allowed_compression_ratios: tuple[float, ...] = ()
    allowed_compression_delta_P_Pa: tuple[float, ...] = ()
    allowed_pump_pressure_ratios: tuple[float, ...] = ()
    allowed_pump_delta_P_Pa: tuple[float, ...] = ()
    allowed_valve_pressure_ratios: tuple[float, ...] = ()
    allowed_valve_delta_P_Pa: tuple[float, ...] = ()
    hx_target_states: tuple[str, ...] = ()
    # "bubble_point", "dew_point", "partial_vf"
    hx_partial_target_vf: float = 0.5
    pump_target_states: tuple[str, ...] = ()
    # "bubble_pressure"
    compressor_target_states: tuple[str, ...] = ()
    # "dew_pressure"
    compressor_min_inlet_vapor_fraction: float = 0.0
    # 0.0 = no restriction; 0.5 = require at least 50% vapour; 1.0 = vapour-only
    valve_target_states: tuple[str, ...] = ()
    # "bubble_pressure"
    enable_distillation_actions: bool = False
    distillation_light_key_recoveries: tuple[float, ...] = (0.9, 0.95, 0.98)
    distillation_heavy_key_recoveries: tuple[float, ...] = (0.01, 0.05, 0.1)
    distillation_reflux_multipliers: tuple[float, ...] = (1.2, 1.5, 2.0)
    distillation_key_pair_mode: DistillationKeyPairMode = "adjacent"
    validate_distillation_candidates: bool = True
    distillation_min_key_flow_mols: float = 1e-9
    distillation_min_alpha_ratio: float = 1.2
    distillation_max_theoretical_stages: float = 80.0
    distillation_max_reflux_ratio: float = 50.0
    max_distillation_count_per_path: int = 1
    min_distillation_count_per_path: int | None = None
    max_total_distillation_count: int | None = None
    max_same_key_pair_count_per_lineage: int | None = None
    widening_coefficient: float = 0.0
    # 0.0 = disabled (all actions available immediately, current behaviour).
    # Positive value enables progressive widening: k(v) = ceil(c_w * N(v)^alpha)
    # actions are exposed after N(v) visits. Actions are shuffled once at node
    # creation so the subset revealed first is random but deterministic.
    widening_exponent: float = 0.5
    compressor_isentropic_efficiency: float = 0.75
    compressor_mechanical_efficiency: float = 1.0
    pump_isentropic_efficiency: float = 0.75
    pump_mechanical_efficiency: float = 1.0
    pump_max_inlet_vapor_fraction: float = 1e-6
    min_pressure_Pa: float = 1.0
    max_pressure_Pa: float = 1.0e7
    target_product_temperature_K: float | None = None
    product_temperature_tolerance_K: float = 1e-6
    min_temperature_K: float = 50.0
    max_temperature_K: float = 500.0
    min_flow_mols: float = 1e-9
    max_active_streams_per_state: int | None = None
    min_stream_priority: float = 0.0
    use_leaf_value_estimator: bool = False
    leaf_value_discount: float | None = None
    leaf_potential_mode: LeafPotentialMode = "flow_weighted_sum"
    rollout_depth: int = 0
    rollout_k: int = 1
    distillation_molar_heat_of_vaporization_J_mol: float = 0.0
    include_reboiler_duty: bool = False
    stage_count_penalty_per_stage: float = 0.0
    max_depth: int = 5
    max_flash_count_per_path: int = 1
    enable_recycle_actions: bool = False
    max_recycle_count_per_path: int = 1
    recycle_purity_threshold: float = 0.95
    product_purity_threshold: float = 1.0
    exploration_weight: float = 1.4
    use_thompson_sampling: bool = False
    unit_penalty: float = 0.01
    duty_penalty_per_W: float = 1e-5
    missing_product_penalty: float = 10.0
    require_flash_liquid_product: bool = True
    candidate_eval_width: int = 0
    candidate_rollouts_per_action: int = 2
    candidate_eval_workers: int = 4
    objective_mode: ObjectiveMode = "single_product"
    separation_score_mode: SeparationScoreMode = "purity_recovery"
    separation_score_tolerance: float = 1e-3
    min_component_fraction: float = 1e-8
    enable_exact_duplicate_pruning: bool = False
    enable_apply_action_cache: bool = False
    cached_action_kinds: tuple[ActionKind, ...] = (
        "hx",
        "flash",
        "compressor",
        "pump",
        "valve",
        "distillation",
    )
    max_apply_action_cache_entries: int | None = None
    enable_action_generation_cache: bool = False
    max_valid_action_cache_entries: int | None = None
    max_relative_volatility_cache_entries: int | None = None


@dataclass(frozen=True)
class MCTSDiagnostics:
    """Search-tree diagnostics for MCTS expansion and duplicate pruning."""

    n_expanded_nodes: int = 0
    n_duplicate_states_skipped: int = 0
    n_seen_state_identities: int = 0
    duplicate_skip_rate: float = 0.0
    n_apply_action_cache_hits: int = 0
    n_apply_action_cache_misses: int = 0
    apply_action_cache_hit_rate: float = 0.0
    n_apply_action_cache_entries: int = 0
    apply_action_calc_time_s: float = 0.0
    apply_action_cache_saved_estimate_s: float = 0.0
    n_distillation_result_cache_hits: int = 0
    n_distillation_result_cache_misses: int = 0
    distillation_result_cache_hit_rate: float = 0.0
    n_distillation_result_cache_entries: int = 0
    distillation_result_calc_time_s: float = 0.0
    distillation_result_cache_saved_estimate_s: float = 0.0
    n_stream_priority_gating_calls: int = 0
    n_stream_priority_streams_considered: int = 0
    n_stream_priority_streams_gated: int = 0
    stream_priority_gate_rate: float = 0.0
    n_valid_action_calls: int = 0
    n_valid_action_cache_hits: int = 0
    n_valid_action_cache_misses: int = 0
    valid_action_cache_hit_rate: float = 0.0
    n_valid_action_cache_entries: int = 0
    valid_action_generation_time_s: float = 0.0
    valid_action_cache_saved_estimate_s: float = 0.0
    n_valid_actions_generated_total: int = 0
    max_valid_actions_generated_per_call: int = 0
    valid_actions_generated_by_kind: tuple[tuple[str, int], ...] = ()
    n_relative_volatility_cache_hits: int = 0
    n_relative_volatility_cache_misses: int = 0
    relative_volatility_cache_hit_rate: float = 0.0
    n_relative_volatility_cache_entries: int = 0
    relative_volatility_calc_time_s: float = 0.0
    relative_volatility_cache_saved_estimate_s: float = 0.0
    n_distillation_action_generation_calls: int = 0
    n_distillation_candidate_key_pairs: int = 0
    n_distillation_candidate_grid_actions: int = 0
    n_distillation_candidate_actions_generated: int = 0
    n_distillation_candidates_filtered_alpha: int = 0
    n_distillation_candidates_filtered_invalid_recovery: int = 0
    n_distillation_candidates_filtered_infeasible: int = 0


@dataclass(frozen=True)
class MCTSResult:
    """Result from MCTS unit-order search."""

    best_state: SearchState
    best_reward: float
    best_sequence: tuple[UnitAction, ...]
    product: StreamState | None
    iterations: int
    progress: tuple[dict[str, object], ...] = ()
    diagnostics: MCTSDiagnostics = field(default_factory=MCTSDiagnostics)
    tree_root: object | None = None  # _Node if return_tree=True was passed to mcts_search
    relative_volatility_cache: dict | None = None  # populated when return_tree=True


@dataclass(frozen=True)
class BatchedMCTSResult:
    """Result from MCTS with parallel batched rollouts.

    Args:
        best_state: Best terminal state found during rollout evaluation.
        best_reward: Reward of best_state.
        best_sequence: Unit sequence that created best_state.
        product: Accepted target product stream, if found.
        iterations: Number of requested MCTS iterations.
        batch_size: Number of selection paths evaluated per batch.
        rollout_workers: Thread workers used for rollout evaluation.
        progress: Optional progress records captured during the run.

    Returns:
        Aggregate result from batched rollout MCTS.

    Example:
        result = batched_mcts_search(feed, provider, config, iterations=500)
        print(result.best_sequence)
    """

    best_state: SearchState
    best_reward: float
    best_sequence: tuple[UnitAction, ...]
    product: StreamState | None
    iterations: int
    batch_size: int
    rollout_workers: int
    progress: tuple[dict[str, object], ...] = ()
    diagnostics: MCTSDiagnostics = field(default_factory=MCTSDiagnostics)


ParallelBackend = Literal["thread", "process"]


@dataclass(frozen=True)
class ParallelMCTSResult:
    """Result from root-parallel MCTS search.

    Args:
        best_result: Best independent worker result by reward.
        worker_results: All worker results in completion order.
        total_iterations: Total requested iterations across workers.
        n_jobs: Number of workers used.
        backend: Parallel backend used.

    Returns:
        Aggregate result from independent root searches.

    Example:
        result = parallel_mcts_search(feed, compounds, config, total_iterations=2000)
        print(result.best_result.best_sequence)
    """

    best_result: MCTSResult
    worker_results: tuple[MCTSResult, ...]
    total_iterations: int
    n_jobs: int
    backend: ParallelBackend


@dataclass(frozen=True)
class RefinedSequenceResult:
    """Result from post-search reflux multiplier refinement.

    Args:
        best_state: State with highest reward across the multiplier grid.
        best_reward: Reward of best_state.
        best_sequence: Action sequence that produced best_state.
        best_reflux_multiplier: Reflux multiplier value that gave best_state.
        grid_results: Per-grid-point diagnostics (multiplier, reward, errors, total_abs_duty_W).

    Returns:
        Refinement result from refine_distillation_sequence().

    Example:
        refined = refine_distillation_sequence(feed, provider, config,
                                               result.best_sequence)
        print(refined.best_reflux_multiplier)
    """

    best_state: SearchState
    best_reward: float
    best_sequence: tuple[UnitAction, ...]
    best_reflux_multiplier: float
    grid_results: tuple[dict[str, object], ...]


@dataclass
class _DiagnosticsAccumulator:
    n_expanded_nodes: int = 0
    n_duplicate_states_skipped: int = 0
    n_apply_action_cache_hits: int = 0
    n_apply_action_cache_misses: int = 0
    apply_action_calc_time_s: float = 0.0
    apply_action_cache_saved_estimate_s: float = 0.0
    n_distillation_result_cache_hits: int = 0
    n_distillation_result_cache_misses: int = 0
    distillation_result_calc_time_s: float = 0.0
    distillation_result_cache_saved_estimate_s: float = 0.0
    n_stream_priority_gating_calls: int = 0
    n_stream_priority_streams_considered: int = 0
    n_stream_priority_streams_gated: int = 0
    n_valid_action_calls: int = 0
    n_valid_action_cache_hits: int = 0
    n_valid_action_cache_misses: int = 0
    valid_action_generation_time_s: float = 0.0
    valid_action_cache_saved_estimate_s: float = 0.0
    n_valid_actions_generated_total: int = 0
    max_valid_actions_generated_per_call: int = 0
    valid_actions_generated_by_kind: Counter[str] = field(default_factory=Counter)
    n_relative_volatility_cache_hits: int = 0
    n_relative_volatility_cache_misses: int = 0
    relative_volatility_calc_time_s: float = 0.0
    relative_volatility_cache_saved_estimate_s: float = 0.0
    n_distillation_action_generation_calls: int = 0
    n_distillation_candidate_key_pairs: int = 0
    n_distillation_candidate_grid_actions: int = 0
    n_distillation_candidate_actions_generated: int = 0
    n_distillation_candidates_filtered_alpha: int = 0
    n_distillation_candidates_filtered_invalid_recovery: int = 0
    n_distillation_candidates_filtered_infeasible: int = 0

    def snapshot(
        self,
        n_seen_state_identities: int,
        n_apply_action_cache_entries: int = 0,
        n_distillation_result_cache_entries: int = 0,
        n_valid_action_cache_entries: int = 0,
        n_relative_volatility_cache_entries: int = 0,
    ) -> MCTSDiagnostics:
        considered = self.n_expanded_nodes + self.n_duplicate_states_skipped
        duplicate_skip_rate = (
            0.0
            if considered == 0
            else self.n_duplicate_states_skipped / considered
        )
        cache_lookups = self.n_apply_action_cache_hits + self.n_apply_action_cache_misses
        apply_action_cache_hit_rate = (
            0.0
            if cache_lookups == 0
            else self.n_apply_action_cache_hits / cache_lookups
        )
        distillation_cache_lookups = (
            self.n_distillation_result_cache_hits
            + self.n_distillation_result_cache_misses
        )
        distillation_result_cache_hit_rate = (
            0.0
            if distillation_cache_lookups == 0
            else self.n_distillation_result_cache_hits / distillation_cache_lookups
        )
        stream_priority_gate_rate = (
            0.0
            if self.n_stream_priority_streams_considered == 0
            else self.n_stream_priority_streams_gated
            / self.n_stream_priority_streams_considered
        )
        valid_action_cache_lookups = (
            self.n_valid_action_cache_hits + self.n_valid_action_cache_misses
        )
        valid_action_cache_hit_rate = (
            0.0
            if valid_action_cache_lookups == 0
            else self.n_valid_action_cache_hits / valid_action_cache_lookups
        )
        relative_volatility_cache_lookups = (
            self.n_relative_volatility_cache_hits
            + self.n_relative_volatility_cache_misses
        )
        relative_volatility_cache_hit_rate = (
            0.0
            if relative_volatility_cache_lookups == 0
            else self.n_relative_volatility_cache_hits
            / relative_volatility_cache_lookups
        )
        return MCTSDiagnostics(
            n_expanded_nodes=self.n_expanded_nodes,
            n_duplicate_states_skipped=self.n_duplicate_states_skipped,
            n_seen_state_identities=n_seen_state_identities,
            duplicate_skip_rate=duplicate_skip_rate,
            n_apply_action_cache_hits=self.n_apply_action_cache_hits,
            n_apply_action_cache_misses=self.n_apply_action_cache_misses,
            apply_action_cache_hit_rate=apply_action_cache_hit_rate,
            n_apply_action_cache_entries=n_apply_action_cache_entries,
            apply_action_calc_time_s=self.apply_action_calc_time_s,
            apply_action_cache_saved_estimate_s=self.apply_action_cache_saved_estimate_s,
            n_distillation_result_cache_hits=self.n_distillation_result_cache_hits,
            n_distillation_result_cache_misses=self.n_distillation_result_cache_misses,
            distillation_result_cache_hit_rate=distillation_result_cache_hit_rate,
            n_distillation_result_cache_entries=n_distillation_result_cache_entries,
            distillation_result_calc_time_s=self.distillation_result_calc_time_s,
            distillation_result_cache_saved_estimate_s=(
                self.distillation_result_cache_saved_estimate_s
            ),
            n_stream_priority_gating_calls=self.n_stream_priority_gating_calls,
            n_stream_priority_streams_considered=(
                self.n_stream_priority_streams_considered
            ),
            n_stream_priority_streams_gated=self.n_stream_priority_streams_gated,
            stream_priority_gate_rate=stream_priority_gate_rate,
            n_valid_action_calls=self.n_valid_action_calls,
            n_valid_action_cache_hits=self.n_valid_action_cache_hits,
            n_valid_action_cache_misses=self.n_valid_action_cache_misses,
            valid_action_cache_hit_rate=valid_action_cache_hit_rate,
            n_valid_action_cache_entries=n_valid_action_cache_entries,
            valid_action_generation_time_s=self.valid_action_generation_time_s,
            valid_action_cache_saved_estimate_s=(
                self.valid_action_cache_saved_estimate_s
            ),
            n_valid_actions_generated_total=self.n_valid_actions_generated_total,
            max_valid_actions_generated_per_call=(
                self.max_valid_actions_generated_per_call
            ),
            valid_actions_generated_by_kind=tuple(
                sorted(self.valid_actions_generated_by_kind.items())
            ),
            n_relative_volatility_cache_hits=(
                self.n_relative_volatility_cache_hits
            ),
            n_relative_volatility_cache_misses=(
                self.n_relative_volatility_cache_misses
            ),
            relative_volatility_cache_hit_rate=relative_volatility_cache_hit_rate,
            n_relative_volatility_cache_entries=(
                n_relative_volatility_cache_entries
            ),
            relative_volatility_calc_time_s=self.relative_volatility_calc_time_s,
            relative_volatility_cache_saved_estimate_s=(
                self.relative_volatility_cache_saved_estimate_s
            ),
            n_distillation_action_generation_calls=(
                self.n_distillation_action_generation_calls
            ),
            n_distillation_candidate_key_pairs=(
                self.n_distillation_candidate_key_pairs
            ),
            n_distillation_candidate_grid_actions=(
                self.n_distillation_candidate_grid_actions
            ),
            n_distillation_candidate_actions_generated=(
                self.n_distillation_candidate_actions_generated
            ),
            n_distillation_candidates_filtered_alpha=(
                self.n_distillation_candidates_filtered_alpha
            ),
            n_distillation_candidates_filtered_invalid_recovery=(
                self.n_distillation_candidates_filtered_invalid_recovery
            ),
            n_distillation_candidates_filtered_infeasible=(
                self.n_distillation_candidates_filtered_infeasible
            ),
        )


@dataclass(frozen=True)
class _CachedActionOutcome:
    output_streams: tuple[StreamState, ...] = ()
    total_abs_duty_delta_W: float = 0.0
    error: str | None = None
    calculation_time_s: float = 0.0


@dataclass(frozen=True)
class _CachedDistillationResult:
    result: ShortcutDistillationResult
    calculation_time_s: float = 0.0


@dataclass(frozen=True)
class _CachedValidActions:
    actions: tuple[UnitAction, ...]
    calculation_time_s: float = 0.0


@dataclass(frozen=True)
class _CachedRelativeVolatilities:
    alphas: dict[str, float]
    calculation_time_s: float = 0.0


class _Node:
    def __init__(
        self,
        state: SearchState,
        config: MCTSConfig,
        feed_stream: StreamState | None = None,
        provider: ThermoFlashProvider | None = None,
        parent: "_Node | None" = None,
        action: UnitAction | None = None,
        diagnostics: _DiagnosticsAccumulator | None = None,
        distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
        valid_action_cache: dict[tuple, _CachedValidActions] | None = None,
        relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[_Node] = []
        all_actions = _valid_actions(
            state,
            config,
            provider,
            feed_stream,
            distillation_result_cache,
            diagnostics,
            valid_action_cache,
            relative_volatility_cache,
        )
        if config.widening_coefficient > 0.0:
            if rng is not None:
                rng.shuffle(all_actions)
            self._all_actions: list[UnitAction] = all_actions
            self._next_action_index: int = 0
            self.untried_actions: list[UnitAction] = []
        else:
            self._all_actions = []
            self._next_action_index = 0
            self.untried_actions = all_actions
        self.visits = 0
        self.value = 0.0


def mcts_search(
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    iterations: int = 500,
    seed: int = 42,
    progress_interval: int = 0,
    progress_callback: ProgressCallback | None = None,
    return_tree: bool = False,
) -> MCTSResult:
    """Search for an ordered unit sequence using discrete MCTS actions.

    Args:
        feed_stream: Initial feed stream.
        provider: ThermoFlashProvider used by HX and flash actions.
        config: Search configuration and target definition.
        iterations: Number of MCTS iterations.
        seed: Random seed for deterministic search.
        progress_interval: Emit/store a progress record every N iterations.
            Zero disables progress tracking unless progress_callback is set,
            in which case a ten-record default cadence is used.
        progress_callback: Optional callable invoked with each progress record.

    Returns:
        MCTSResult with the best state found during rollouts.

    Example:
        result = mcts_search(feed, provider, config, iterations=500)
        print(result.best_sequence)
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    progress_interval = _normalise_progress_interval(
        iterations,
        progress_interval,
        progress_callback,
    )

    started_at = time.monotonic()
    rng = random.Random(seed)
    root_state = SearchState(
        open_streams=(feed_stream,),
        process_graph=process_graph_from_feed(feed_stream),
        feed_stream=feed_stream,
    )
    seen_state_hashes = {state_identity_hash(root_state)}
    diagnostics = _DiagnosticsAccumulator()
    action_cache: dict[tuple, _CachedActionOutcome] | None = (
        {} if config.enable_apply_action_cache else None
    )
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = (
        {} if _distillation_result_cache_enabled(config) else None
    )
    valid_action_cache: dict[tuple, _CachedValidActions] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )
    root = _Node(
        root_state,
        config,
        feed_stream,
        provider,
        diagnostics=diagnostics,
        distillation_result_cache=distillation_result_cache,
        valid_action_cache=valid_action_cache,
        relative_volatility_cache=relative_volatility_cache,
        rng=rng,
    )

    best_state = root_state
    # Initialise with the actual metric score only (no leaf estimator).
    # depth_aware_bounded and depth_aware_alpha_gated produce V == n_c at the
    # root (γ_d=1, U_norm=1, S_norm=0 for a fresh equimolar feed), which equals
    # the theoretical maximum reward.  No rollout from any child can beat it
    # because γ_d < 1 at depth > 0, so the search would return seq=() forever.
    _init_config = (
        replace(config, use_leaf_value_estimator=False)
        if config.use_leaf_value_estimator
        else config
    )
    best_reward = _reward(root_state, _init_config, feed_stream)
    progress: list[dict[str, object]] = []

    for iteration in range(1, iterations + 1):
        node = _select(
            root,
            feed_stream,
            provider,
            config,
            rng,
            seen_state_hashes,
            diagnostics,
            action_cache,
            distillation_result_cache,
            valid_action_cache,
            relative_volatility_cache,
        )
        reward_state = _rollout(
            node.state,
            feed_stream,
            provider,
            config,
            rng,
            action_cache,
            diagnostics,
            distillation_result_cache,
            valid_action_cache,
            relative_volatility_cache,
        )
        reward = _reward(reward_state, config, feed_stream, provider, relative_volatility_cache)

        # UCT back-propagation uses the full estimated reward so the tree
        # favours states with high separation potential.
        _backpropagate(node, reward)

        # best_state tracking uses the actual metric score (no leaf estimator
        # boost) so that terminal states are never beaten by shallow non-terminal
        # states whose potential has been inflated by depth_aware_bounded /
        # depth_aware_alpha_gated.  For full_rollout and score_only this is the
        # same as reward; the extra call is skipped for those modes.
        tracking_reward = (
            _reward(reward_state, _init_config, feed_stream)
            if config.use_leaf_value_estimator
            else reward
        )
        if tracking_reward > best_reward:
            best_state = reward_state
            best_reward = tracking_reward
        if _should_emit_progress(iteration, iterations, progress_interval):
            _emit_progress(
                progress,
                progress_callback,
                iteration,
                iterations,
                started_at,
                best_state,
                best_reward,
                config,
                feed_stream,
                diagnostics.snapshot(
                    len(seen_state_hashes),
                    _apply_action_cache_size(action_cache),
                    _distillation_result_cache_size(distillation_result_cache),
                    _valid_action_cache_size(valid_action_cache),
                    _relative_volatility_cache_size(relative_volatility_cache),
                ),
            )

    # Compare the greedy tree path against the best rollout seen.
    # For leaf-estimator methods this can surface consistently-good paths
    # that were never the single best rollout but have the highest average Q.
    # For full-rollout the greedy leaf is usually non-terminal so its bare
    # metric is lower and the comparison is harmless (never overwrites).
    tree_state = _best_tree_path(root, config, feed_stream)
    tree_reward = _reward(tree_state, _init_config, feed_stream)
    if tree_reward > best_reward:
        best_state = tree_state
        best_reward = tree_reward

    product = _product(best_state, config.product_role)
    diagnostics_snapshot = diagnostics.snapshot(
        len(seen_state_hashes),
        _apply_action_cache_size(action_cache),
        _distillation_result_cache_size(distillation_result_cache),
        _valid_action_cache_size(valid_action_cache),
        _relative_volatility_cache_size(relative_volatility_cache),
    )
    return MCTSResult(
        best_state=best_state,
        best_reward=best_reward,
        best_sequence=best_state.unit_sequence,
        product=product.stream if product else None,
        iterations=iterations,
        progress=tuple(progress),
        diagnostics=diagnostics_snapshot,
        tree_root=root if return_tree else None,
        relative_volatility_cache=relative_volatility_cache if return_tree else None,
    )


def parallel_mcts_search(
    feed_stream: StreamState,
    compounds: list[str],
    config: MCTSConfig,
    total_iterations: int = 2000,
    n_jobs: int | None = None,
    base_seed: int = 42,
    backend: ParallelBackend = "thread",
    return_tree: bool = False,
) -> ParallelMCTSResult:
    """Run multiple independent root MCTS searches in parallel.

    Each worker builds its own thermo provider from compounds and runs a serial
    MCTS search with a distinct seed. Results are merged by best_reward. This
    avoids shared-tree synchronization and keeps the serial MCTS implementation
    unchanged.

    Args:
        feed_stream: Initial feed stream.
        compounds: thermo component identifiers used to build each worker flasher.
        config: Search configuration and target definition.
        total_iterations: Total iterations distributed across workers.
        n_jobs: Number of workers. Defaults to min(cpu_count, total_iterations).
        base_seed: Seed used to derive per-worker seeds.
        backend: "thread" or "process". Thread is default to avoid process
            startup overhead for small searches.
        return_tree: If True, each worker result carries tree_root. Use when
            training-node extraction is needed after the search.

    Returns:
        ParallelMCTSResult with the best worker result and all worker results.

    Raises:
        ValueError: if arguments are invalid.

    Example:
        result = parallel_mcts_search(
            feed, ["methane", "ethane", "nitrogen"], config,
            total_iterations=2000, n_jobs=4,
        )
    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive.")
    if backend not in ("thread", "process"):
        raise ValueError("backend must be 'thread' or 'process'.")

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, total_iterations)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive.")
    n_jobs = min(n_jobs, total_iterations)

    iteration_counts = _split_iterations(total_iterations, n_jobs)
    seeds = [base_seed + i for i in range(n_jobs)]
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    results: list[MCTSResult] = []
    with executor_cls(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _parallel_worker,
                feed_stream,
                list(compounds),
                config,
                iteration_counts[i],
                seeds[i],
                return_tree,
            )
            for i in range(n_jobs)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    best = max(results, key=lambda result: result.best_reward)
    return ParallelMCTSResult(
        best_result=best,
        worker_results=tuple(results),
        total_iterations=total_iterations,
        n_jobs=n_jobs,
        backend=backend,
    )


def batched_mcts_search(
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    iterations: int = 1000,
    batch_size: int = 16,
    rollout_workers: int = 4,
    seed: int = 42,
    progress_interval: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> BatchedMCTSResult:
    """Search with serial tree policy and parallel rollout evaluation.

    The tree is still selected, expanded, and backpropagated on one thread.
    Only rollout simulations from the selected/expanded nodes are evaluated in
    parallel. Results are backpropagated in selection order to keep seeded runs
    deterministic while still overlapping expensive thermo calculations.

    Args:
        feed_stream: Initial feed stream.
        provider: ThermoFlashProvider used by HX and flash actions.
        config: Search configuration and target definition.
        iterations: Number of MCTS iterations.
        batch_size: Maximum selected nodes evaluated per rollout batch.
        rollout_workers: Thread workers used for rollout evaluation.
        seed: Random seed for deterministic selection and rollout seeds.
        progress_interval: Emit/store a progress record every N completed
            rollouts. Zero disables progress tracking unless progress_callback
            is set, in which case a ten-record default cadence is used.
        progress_callback: Optional callable invoked with each progress record.

    Returns:
        BatchedMCTSResult with the best state found during rollouts.

    Raises:
        ValueError: if iterations, batch_size, or rollout_workers are invalid.

    Example:
        result = batched_mcts_search(
            feed, provider, config, iterations=1000, batch_size=16
        )
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if rollout_workers <= 0:
        raise ValueError("rollout_workers must be positive.")
    progress_interval = _normalise_progress_interval(
        iterations,
        progress_interval,
        progress_callback,
    )

    started_at = time.monotonic()
    rng = random.Random(seed)
    root_state = SearchState(
        open_streams=(feed_stream,),
        process_graph=process_graph_from_feed(feed_stream),
        feed_stream=feed_stream,
    )
    seen_state_hashes = {state_identity_hash(root_state)}
    diagnostics = _DiagnosticsAccumulator()
    action_cache: dict[tuple, _CachedActionOutcome] | None = (
        {} if config.enable_apply_action_cache else None
    )
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = (
        {} if _distillation_result_cache_enabled(config) else None
    )
    valid_action_cache: dict[tuple, _CachedValidActions] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )
    root = _Node(
        root_state,
        config,
        feed_stream,
        provider,
        diagnostics=diagnostics,
        distillation_result_cache=distillation_result_cache,
        valid_action_cache=valid_action_cache,
        relative_volatility_cache=relative_volatility_cache,
        rng=rng,
    )

    best_state = root_state
    _init_config_b = (
        replace(config, use_leaf_value_estimator=False)
        if config.use_leaf_value_estimator
        else config
    )
    best_reward = _reward(root_state, _init_config_b, feed_stream)
    completed = 0
    progress: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=rollout_workers) as executor:
        while completed < iterations:
            current_batch_size = min(batch_size, iterations - completed)
            nodes = [
                _select(
                    root,
                    feed_stream,
                    provider,
                    config,
                    rng,
                    seen_state_hashes,
                    diagnostics,
                    action_cache,
                    distillation_result_cache,
                    valid_action_cache,
                    relative_volatility_cache,
                )
                for _ in range(current_batch_size)
            ]
            seeds = [rng.randrange(2**31 - 1) for _ in nodes]
            futures = [
                executor.submit(
                    _batched_rollout_worker,
                    node.state,
                    feed_stream,
                    provider,
                    config,
                    rollout_seed,
                )
                for node, rollout_seed in zip(nodes, seeds)
            ]

            for node, future in zip(nodes, futures):
                reward_state, reward = future.result()
                _backpropagate(node, reward)
                tracking_reward = (
                    _reward(reward_state, _init_config_b, feed_stream)
                    if config.use_leaf_value_estimator
                    else reward
                )
                if tracking_reward > best_reward:
                    best_state = reward_state
                    best_reward = tracking_reward

            completed += current_batch_size
            if _should_emit_progress(completed, iterations, progress_interval):
                _emit_progress(
                    progress,
                    progress_callback,
                    completed,
                    iterations,
                    started_at,
                    best_state,
                    best_reward,
                    config,
                    feed_stream,
                    diagnostics.snapshot(
                        len(seen_state_hashes),
                        _apply_action_cache_size(action_cache),
                        _distillation_result_cache_size(distillation_result_cache),
                        _valid_action_cache_size(valid_action_cache),
                        _relative_volatility_cache_size(relative_volatility_cache),
                    ),
                )

    product = _product(best_state, config.product_role)
    diagnostics_snapshot = diagnostics.snapshot(
        len(seen_state_hashes),
        _apply_action_cache_size(action_cache),
        _distillation_result_cache_size(distillation_result_cache),
        _valid_action_cache_size(valid_action_cache),
        _relative_volatility_cache_size(relative_volatility_cache),
    )
    return BatchedMCTSResult(
        best_state=best_state,
        best_reward=best_reward,
        best_sequence=best_state.unit_sequence,
        product=product.stream if product else None,
        iterations=iterations,
        batch_size=batch_size,
        rollout_workers=rollout_workers,
        progress=tuple(progress),
        diagnostics=diagnostics_snapshot,
    )


def _normalise_progress_interval(
    iterations: int,
    progress_interval: int,
    progress_callback: ProgressCallback | None,
) -> int:
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative.")
    if progress_interval == 0 and progress_callback is not None:
        return max(1, iterations // 10)
    return progress_interval


def _should_emit_progress(
    iteration: int,
    iterations: int,
    progress_interval: int,
) -> bool:
    if progress_interval <= 0:
        return False
    return iteration == iterations or iteration % progress_interval == 0


def _emit_progress(
    progress: list[dict[str, object]],
    progress_callback: ProgressCallback | None,
    iteration: int,
    iterations: int,
    started_at: float,
    best_state: SearchState,
    best_reward: float,
    config: MCTSConfig,
    feed_stream: StreamState,
    diagnostics: MCTSDiagnostics | None = None,
) -> None:
    record = mcts_progress_record(
        iteration,
        iterations,
        time.monotonic() - started_at,
        best_state,
        best_reward,
        config,
        feed_stream,
        diagnostics,
    )
    progress.append(record)
    if progress_callback is not None:
        progress_callback(record)


def mcts_progress_record(
    iteration: int,
    iterations: int,
    elapsed_s: float,
    best_state: SearchState,
    best_reward: float,
    config: MCTSConfig,
    feed_stream: StreamState,
    diagnostics: MCTSDiagnostics | None = None,
) -> dict[str, object]:
    """Build a plain progress record for serial or batched MCTS.

    Args:
        iteration: Completed iteration count.
        iterations: Requested iteration count.
        elapsed_s: Elapsed wall time [s].
        best_state: Current best state.
        best_reward: Current best reward.
        config: MCTS configuration.
        feed_stream: Original feed stream.
        diagnostics: Optional search-tree diagnostics snapshot.

    Returns:
        Plain dict suitable for printing, logging, or notebook display.

    Example:
        record = mcts_progress_record(i, n, elapsed, state, reward, config, feed)
        print(record["iteration"], record["best_reward"])
    """
    record: dict[str, object] = {
        "iteration": iteration,
        "iterations": iterations,
        "elapsed_s": elapsed_s,
        "best_reward": best_reward,
        "sequence_length": len(best_state.unit_sequence),
        "sequence_kinds": tuple(action.kind for action in best_state.unit_sequence),
        "n_open_streams": len(best_state.open_streams),
        "n_products": len(best_state.products),
        "n_errors": len(best_state.errors),
        "total_abs_duty_W": best_state.total_abs_duty_W,
        "topology_hash": state_topology_hash(best_state),
        "state_identity_hash": state_identity_hash(best_state),
    }
    if diagnostics is not None:
        record.update(_diagnostics_record(diagnostics))

    if config.objective_mode == "complete_separation":
        metric = _complete_separation_metric(best_state, config, feed_stream)
        record.update(
            {
                "separation_score": metric["score"],
                "separation_target": metric["target"],
                "fraction_of_target": metric["fraction_of_target"],
                "component_scores": metric["component_scores"],
                "best_stream_by_component": metric["best_stream_by_component"],
            }
        )
    else:
        product = _product(best_state, config.product_role)
        record["product_id"] = product.stream.id if product else None
        record["product_role"] = product.role if product else None
        if product and config.target_component in product.stream.composition:
            record["target_component_fraction"] = product.stream.composition[
                config.target_component
            ]
        else:
            record["target_component_fraction"] = None

    return record


def _diagnostics_record(diagnostics: MCTSDiagnostics) -> dict[str, object]:
    return {
        "n_expanded_nodes": diagnostics.n_expanded_nodes,
        "n_duplicate_states_skipped": diagnostics.n_duplicate_states_skipped,
        "n_seen_state_identities": diagnostics.n_seen_state_identities,
        "duplicate_skip_rate": diagnostics.duplicate_skip_rate,
        "n_apply_action_cache_hits": diagnostics.n_apply_action_cache_hits,
        "n_apply_action_cache_misses": diagnostics.n_apply_action_cache_misses,
        "apply_action_cache_hit_rate": diagnostics.apply_action_cache_hit_rate,
        "n_apply_action_cache_entries": diagnostics.n_apply_action_cache_entries,
        "apply_action_calc_time_s": diagnostics.apply_action_calc_time_s,
        "apply_action_cache_saved_estimate_s": (
            diagnostics.apply_action_cache_saved_estimate_s
        ),
        "n_distillation_result_cache_hits": diagnostics.n_distillation_result_cache_hits,
        "n_distillation_result_cache_misses": (
            diagnostics.n_distillation_result_cache_misses
        ),
        "distillation_result_cache_hit_rate": (
            diagnostics.distillation_result_cache_hit_rate
        ),
        "n_distillation_result_cache_entries": (
            diagnostics.n_distillation_result_cache_entries
        ),
        "distillation_result_calc_time_s": diagnostics.distillation_result_calc_time_s,
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
        "valid_action_generation_time_s": diagnostics.valid_action_generation_time_s,
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


def print_mcts_progress(record: dict[str, object]) -> None:
    """Print a compact one-line MCTS progress record.

    Args:
        record: Progress record from mcts_search() or batched_mcts_search().

    Returns:
        None.

    Example:
        result = batched_mcts_search(..., progress_interval=50,
                                     progress_callback=print_mcts_progress)
    """
    if "separation_score" in record:
        score = float(record["separation_score"])
        target = int(record["separation_target"])
        fraction = float(record["fraction_of_target"])
        objective = f"score={score:.6g}/{target} ({fraction:.3%})"
    else:
        fraction_value = record.get("target_component_fraction")
        if fraction_value is None:
            objective = "product=None"
        else:
            objective = f"x_target={float(fraction_value):.6g}"

    print(
        "MCTS "
        f"{record['iteration']}/{record['iterations']} "
        f"elapsed={float(record['elapsed_s']):.1f}s "
        f"reward={float(record['best_reward']):.6g} "
        f"{objective} "
        f"seq={record['sequence_kinds']}"
    )


def refine_distillation_sequence(
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    sequence: tuple[UnitAction, ...] | list[UnitAction],
    reflux_multiplier_grid: tuple[float, ...] | list[float] | None = None,
) -> RefinedSequenceResult:
    """Sweep a reflux multiplier grid on a fixed distillation sequence.

    Replaces the reflux multiplier in every distillation action in the sequence
    with each value from the grid, replays the full sequence from scratch, and
    returns the highest-reward result. Stream IDs are preserved across the sweep
    because column naming depends on key names and step index, not the multiplier.

    Args:
        feed_stream: Original feed stream to replay from.
        provider: ThermoFlashProvider for unit calculations.
        config: MCTS configuration used for reward evaluation.
        sequence: Fixed action sequence to replay (typically best_sequence from
            an MCTS result).
        reflux_multiplier_grid: Reflux R/R_min multipliers to evaluate.
            Defaults to a nine-point grid from 1.05 to 3.0.

    Returns:
        RefinedSequenceResult with best state, reward, multiplier, and per-grid
        diagnostics.

    Raises:
        ValueError: If sequence contains no distillation actions.

    Example:
        refined = refine_distillation_sequence(
            feed, provider, config, mcts_result.best_sequence
        )
        print(f"Best R/Rmin = {refined.best_reflux_multiplier:.2f}")
    """
    sequence = tuple(sequence)
    if not any(a.kind == "distillation" for a in sequence):
        raise ValueError(
            "sequence contains no distillation actions to refine. "
            "Only sequences with at least one distillation action can be refined."
        )
    if reflux_multiplier_grid is None:
        reflux_multiplier_grid = (1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0)

    best_state: SearchState | None = None
    best_reward = -math.inf
    best_sequence = sequence
    best_multiplier = float(reflux_multiplier_grid[0])
    grid_results: list[dict[str, object]] = []

    for multiplier in reflux_multiplier_grid:
        refined = tuple(
            UnitAction(
                kind=a.kind,
                stream_id=a.stream_id,
                delta_T_K=a.delta_T_K,
                pressure_ratio=a.pressure_ratio,
                delta_P_Pa=a.delta_P_Pa,
                light_key=a.light_key,
                heavy_key=a.heavy_key,
                light_key_recovery=a.light_key_recovery,
                heavy_key_recovery=a.heavy_key_recovery,
                reflux_ratio_multiplier=(
                    float(multiplier) if a.kind == "distillation" else a.reflux_ratio_multiplier
                ),
                role=a.role,
            )
            for a in sequence
        )
        state = SearchState(
            open_streams=(feed_stream,),
            process_graph=process_graph_from_feed(feed_stream),
        )
        for action in refined:
            state = _apply_action(state, action, provider, config)
        reward = _reward(state, config, feed_stream)
        grid_results.append(
            {
                "reflux_multiplier": float(multiplier),
                "reward": reward,
                "n_errors": len(state.errors),
                "errors": state.errors,
                "sequence_length": len(state.unit_sequence),
                "total_abs_duty_W": state.total_abs_duty_W,
            }
        )
        if reward > best_reward:
            best_reward = reward
            best_state = state
            best_sequence = refined
            best_multiplier = float(multiplier)

    if best_state is None:
        best_state = SearchState(
            open_streams=(feed_stream,),
            process_graph=process_graph_from_feed(feed_stream),
        )

    return RefinedSequenceResult(
        best_state=best_state,
        best_reward=best_reward,
        best_sequence=best_sequence,
        best_reflux_multiplier=best_multiplier,
        grid_results=tuple(grid_results),
    )


def _parallel_worker(
    feed_stream: StreamState,
    compounds: list[str],
    config: MCTSConfig,
    iterations: int,
    seed: int,
    return_tree: bool = False,
) -> MCTSResult:
    provider = build_pr_flasher(compounds)
    return mcts_search(
        feed_stream=feed_stream,
        provider=provider,
        config=config,
        iterations=iterations,
        seed=seed,
        return_tree=return_tree,
    )


def _batched_rollout_worker(
    state: SearchState,
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    seed: int,
) -> tuple[SearchState, float]:
    k = max(1, config.rollout_k)
    use_k_sample = k > 1 or config.rollout_depth > 0

    valid_action_cache: dict[tuple, _CachedValidActions] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = (
        {} if _action_generation_cache_enabled(config) else None
    )

    if not use_k_sample:
        rng = random.Random(seed)
        reward_state = _rollout(
            state, feed_stream, provider, config, rng,
            valid_action_cache=valid_action_cache,
            relative_volatility_cache=relative_volatility_cache,
        )
        reward = _reward(reward_state, config, feed_stream, provider)
        return reward_state, reward

    # K-sample truncated rollout with α-filter
    alpha_leaf = _flow_weighted_mean_alpha(state, provider, config, relative_volatility_cache)

    valid_samples: list[tuple[SearchState, float]] = []
    best_sample_reward: float = -math.inf
    best_sample_state: SearchState = state

    for i in range(k):
        rng_i = random.Random(seed + i * 1_000_003)
        sample_va_cache: dict[tuple, _CachedValidActions] | None = (
            {} if _action_generation_cache_enabled(config) else None
        )
        sample_rv_cache: dict[tuple, _CachedRelativeVolatilities] | None = (
            {} if _action_generation_cache_enabled(config) else None
        )
        endpoint = _rollout(
            state, feed_stream, provider, config, rng_i,
            valid_action_cache=sample_va_cache,
            relative_volatility_cache=sample_rv_cache,
        )
        r = _reward(endpoint, config, feed_stream, provider)
        alpha_end = _flow_weighted_mean_alpha(endpoint, provider, config, sample_rv_cache)
        if alpha_end > alpha_leaf:
            valid_samples.append((endpoint, r))
        if r > best_sample_reward:
            best_sample_reward = r
            best_sample_state = endpoint

    if valid_samples:
        avg_reward = sum(r for _, r in valid_samples) / len(valid_samples)
        best_valid = max(valid_samples, key=lambda x: x[1])[0]
        return best_valid, avg_reward

    # All samples filtered — fall back to pure leaf estimator at the leaf node
    fallback_reward = _reward(state, config, feed_stream, provider)
    return state, fallback_reward


def _split_iterations(total_iterations: int, n_jobs: int) -> list[int]:
    base = total_iterations // n_jobs
    remainder = total_iterations % n_jobs
    return [base + (1 if i < remainder else 0) for i in range(n_jobs)]


def _widen_node(node: _Node, config: MCTSConfig) -> None:
    """Reveal actions up to k(v) = ceil(c_w * max(N,1)^alpha) for node v."""
    if config.widening_coefficient <= 0.0 or not node._all_actions:
        return
    k = math.ceil(
        config.widening_coefficient * max(node.visits, 1) ** config.widening_exponent
    )
    k = min(k, len(node._all_actions))
    while node._next_action_index < k:
        node.untried_actions.append(node._all_actions[node._next_action_index])
        node._next_action_index += 1


def _select(
    node: _Node,
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    rng: random.Random,
    seen_state_hashes: set[str] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    action_cache: dict[tuple, _CachedActionOutcome] | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    valid_action_cache: dict[tuple, _CachedValidActions] | None = None,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
) -> _Node:
    while not _is_terminal(node.state, config, feed_stream):
        _widen_node(node, config)
        if node.untried_actions:
            action = _choose_untried_action(node, feed_stream, provider, config, rng)
            node.untried_actions.remove(action)
            child_state = _apply_action(
                node.state,
                action,
                provider,
                config,
                action_cache,
                diagnostics,
                distillation_result_cache,
            )
            if seen_state_hashes is not None:
                child_hash = state_identity_hash(child_state)
                if config.enable_exact_duplicate_pruning and child_hash in seen_state_hashes:
                    if diagnostics is not None:
                        diagnostics.n_duplicate_states_skipped += 1
                    continue
                if child_hash not in seen_state_hashes:
                    seen_state_hashes.add(child_hash)
            child = _Node(
                child_state,
                config,
                feed_stream,
                provider,
                parent=node,
                action=action,
                diagnostics=diagnostics,
                distillation_result_cache=distillation_result_cache,
                valid_action_cache=valid_action_cache,
                relative_volatility_cache=relative_volatility_cache,
                rng=rng,
            )
            node.children.append(child)
            if diagnostics is not None:
                diagnostics.n_expanded_nodes += 1
            return child
        if not node.children:
            return node
        node = (
            _best_thompson_child(node, rng)
            if config.use_thompson_sampling
            else _best_uct_child(node, config)
        )
    return node


def _choose_untried_action(
    node: _Node,
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    rng: random.Random,
) -> UnitAction:
    if config.candidate_eval_width <= 0:
        return rng.choice(node.untried_actions)

    candidates = _sample_actions(node.untried_actions, config.candidate_eval_width, rng)
    scored = _evaluate_candidate_actions(
        node.state,
        candidates,
        feed_stream,
        provider,
        config,
        rng,
    )
    return max(scored, key=lambda item: item[1])[0]


def _evaluate_candidate_actions(
    state: SearchState,
    actions: list[UnitAction],
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    rng: random.Random,
) -> list[tuple[UnitAction, float]]:
    rollouts_per_action = max(1, int(config.candidate_rollouts_per_action))
    workers = max(1, min(int(config.candidate_eval_workers), len(actions)))
    seeds = {
        action: [rng.randrange(2**31 - 1) for _ in range(rollouts_per_action)]
        for action in actions
    }

    def score_action(action: UnitAction) -> tuple[UnitAction, float]:
        child_state = _apply_action(state, action, provider, config)
        rewards = []
        for seed in seeds[action]:
            rollout_rng = random.Random(seed)
            terminal = _rollout(child_state, feed_stream, provider, config, rollout_rng)
            rewards.append(_reward(terminal, config, feed_stream))
        return action, sum(rewards) / len(rewards)

    if workers == 1:
        return [score_action(action) for action in actions]

    scored: list[tuple[UnitAction, float]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_action, action) for action in actions]
        for future in as_completed(futures):
            scored.append(future.result())
    return scored


def _sample_actions(
    actions: list[UnitAction],
    width: int,
    rng: random.Random,
) -> list[UnitAction]:
    if width >= len(actions):
        return list(actions)
    return rng.sample(actions, width)


def _best_tree_path(
    root: _Node,
    config: MCTSConfig,
    feed_stream: StreamState,
) -> SearchState:
    """Return the state reached by greedy descent from root.

    At each node, selects the visited child with the highest average Q-value
    (value / visits).  Stops at a terminal state or when no visited children
    remain (unvisited children are ignored — they carry no statistical signal).

    For full-rollout runs the returned state is typically a shallow tree leaf
    (non-terminal), so its bare metric score will be lower than the best
    rollout terminal seen during search.  For leaf-estimator methods the leaf
    IS the intended evaluation point and the comparison is meaningful.

    Args:
        root: Root node of the completed search tree.
        config: Search configuration (used only for ``_is_terminal``).
        feed_stream: Original feed (used only for ``_is_terminal``).

    Returns:
        SearchState of the deepest node on the greedy-Q path.
    """
    node = root
    while not _is_terminal(node.state, config, feed_stream):
        visited = [c for c in node.children if c.visits > 0]
        if not visited:
            break
        node = max(visited, key=lambda c: c.value / c.visits)
    return node.state


def _rollout(
    state: SearchState,
    feed_stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    rng: random.Random,
    action_cache: dict[tuple, _CachedActionOutcome] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    valid_action_cache: dict[tuple, _CachedValidActions] | None = None,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
) -> SearchState:
    if (
        config.use_leaf_value_estimator
        and config.objective_mode == "complete_separation"
        and config.rollout_depth == 0
    ):
        return state
    current = state
    steps = 0
    max_steps = config.rollout_depth if config.rollout_depth > 0 else None
    while not _is_terminal(current, config, feed_stream):
        if max_steps is not None and steps >= max_steps:
            break
        actions = _valid_actions(
            current,
            config,
            provider,
            feed_stream,
            distillation_result_cache,
            diagnostics,
            valid_action_cache,
            relative_volatility_cache,
        )
        if not actions:
            break
        action = _rollout_action(current, actions, config, rng)
        current = _apply_action(
            current,
            action,
            provider,
            config,
            action_cache,
            diagnostics,
            distillation_result_cache,
        )
        steps += 1
    return current


def _backpropagate(node: _Node, reward: float) -> None:
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent


def _best_uct_child(node: _Node, config: MCTSConfig) -> _Node:
    log_parent = math.log(max(node.visits, 1))

    def score(child: _Node) -> float:
        if child.visits == 0:
            return math.inf
        exploit = child.value / child.visits
        explore = config.exploration_weight * math.sqrt(log_parent / child.visits)
        return exploit + explore

    return max(node.children, key=score)


def _best_thompson_child(node: _Node, rng: random.Random) -> _Node:
    """Select a child by Thompson Sampling over Beta posteriors.

    Each child's expected reward is modelled as Beta(1 + V, 1 + n − V) where
    V = accumulated reward and n = visit count.  Unvisited children use the
    uninformative prior Beta(1, 1) = Uniform(0, 1), which gives them a high
    probability of being selected, replacing the explicit "expand untried
    first" heuristic with a principled Bayesian equivalent.

    Meaningful when rewards are stochastic (full_rollout, truncated_rollout).
    For deterministic leaf estimators the posterior collapses to a point mass
    after the first visit and Thompson Sampling degenerates to UCT.
    """
    best_child: _Node | None = None
    best_sample = -1.0
    for child in node.children:
        alpha = 1.0 + child.value
        beta  = 1.0 + child.visits - child.value
        # beta must stay positive; value is clipped by reward bounds [0,1]
        # but numerical edge cases can push it slightly below 0.
        sample = rng.betavariate(max(alpha, 1e-6), max(beta, 1e-6))
        if sample > best_sample:
            best_sample = sample
            best_child = child
    # Fallback — should never trigger because _select only calls this when
    # node.children is non-empty.
    return best_child if best_child is not None else node.children[0]


def _valid_actions(
    state: SearchState,
    config: MCTSConfig,
    provider: ThermoFlashProvider | None = None,
    feed_stream: StreamState | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    valid_action_cache: dict[tuple, _CachedValidActions] | None = None,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
) -> list[UnitAction]:
    if diagnostics is not None:
        diagnostics.n_valid_action_calls += 1

    use_cache = _valid_action_cache_enabled(config, valid_action_cache)
    cache_key = None
    if use_cache:
        cache_key = _valid_action_cache_key(state, config, provider, feed_stream)
        cached = valid_action_cache.get(cache_key)
        if cached is not None:
            if diagnostics is not None:
                diagnostics.n_valid_action_cache_hits += 1
                diagnostics.valid_action_cache_saved_estimate_s += (
                    cached.calculation_time_s
                )
            return list(cached.actions)
        if diagnostics is not None:
            diagnostics.n_valid_action_cache_misses += 1

    started_at = time.monotonic()
    actions = _valid_actions_uncached(
        state,
        config,
        provider,
        feed_stream,
        distillation_result_cache,
        diagnostics,
        relative_volatility_cache,
    )
    calculation_time_s = time.monotonic() - started_at
    if diagnostics is not None:
        diagnostics.valid_action_generation_time_s += calculation_time_s
        _record_valid_actions_generated(diagnostics, actions)
    if (
        use_cache
        and cache_key is not None
        and _can_store_valid_action_cache_entry(valid_action_cache, config)
    ):
        valid_action_cache[cache_key] = _CachedValidActions(
            actions=tuple(actions),
            calculation_time_s=calculation_time_s,
        )
    return actions


_HX_TARGET_VF: dict[str, float] = {
    "bubble_point": 0.0,
    "dew_point": 1.0,
}

_BISECT_ITERS = 40


def _stream_vapor_fraction(
    stream: StreamState,
    provider: ThermoFlashProvider,
) -> float | None:
    try:
        result = provider.flasher.flash(
            T=stream.temperature_K,
            P=stream.pressure_Pa,
            zs=list(stream.composition.values()),
        )
        return result.VF
    except Exception:
        return None


def _resolve_hx_target_delta_T(
    stream: StreamState,
    target: str,
    config: MCTSConfig,
    provider: ThermoFlashProvider,
) -> float | None:
    vf = config.hx_partial_target_vf if target == "partial_vf" else _HX_TARGET_VF.get(target)
    if vf is None:
        return None
    try:
        result = provider.flasher.flash(
            P=stream.pressure_Pa,
            VF=vf,
            zs=list(stream.composition.values()),
        )
        T_target = result.T
    except Exception:
        return None
    if T_target is None or not math.isfinite(T_target):
        return None
    return T_target - stream.temperature_K


def _resolve_pump_target_ratio(
    stream: StreamState,
    target: str,
    provider: ThermoFlashProvider,
) -> float | None:
    if target != "bubble_pressure":
        return None
    try:
        result = provider.flasher.flash(
            T=stream.temperature_K,
            VF=0.0,
            zs=list(stream.composition.values()),
        )
        P_target = result.P
    except Exception:
        return None
    if P_target is None or not math.isfinite(P_target) or P_target <= stream.pressure_Pa:
        return None
    return P_target / stream.pressure_Pa


def _resolve_compressor_target_ratio(
    stream: StreamState,
    target: str,
    config: MCTSConfig,
    provider: ThermoFlashProvider,
) -> float | None:
    if target != "dew_pressure":
        return None
    zs = list(stream.composition.values())
    try:
        S_in = provider.flasher.flash(
            T=stream.temperature_K,
            P=stream.pressure_Pa,
            zs=zs,
        ).S()
    except Exception:
        return None

    def vf_at_P(P: float) -> float | None:
        try:
            return provider.flasher.flash(S=S_in, P=P, zs=zs).VF
        except Exception:
            return None

    P_lo, P_hi = stream.pressure_Pa, config.max_pressure_Pa
    vf_hi = vf_at_P(P_hi)
    if vf_hi is None or vf_hi >= 1.0:
        return None  # condensation never reachable within max pressure

    for _ in range(_BISECT_ITERS):
        P_mid = 0.5 * (P_lo + P_hi)
        vf = vf_at_P(P_mid)
        if vf is None:
            return None
        if vf >= 1.0:
            P_lo = P_mid
        else:
            P_hi = P_mid
        if (P_hi - P_lo) / stream.pressure_Pa < 1e-6:
            break

    P_dew = P_hi
    ratio = P_dew / stream.pressure_Pa
    return ratio if ratio > 1.0 else None


def _resolve_valve_target_ratio(
    stream: StreamState,
    target: str,
    config: MCTSConfig,
    provider: ThermoFlashProvider,
) -> float | None:
    if target != "bubble_pressure":
        return None
    zs = list(stream.composition.values())
    try:
        H_in = provider.flasher.flash(
            T=stream.temperature_K,
            P=stream.pressure_Pa,
            zs=zs,
        ).H()
    except Exception:
        return None

    def vf_at_P(P: float) -> float | None:
        try:
            return provider.flasher.flash(H=H_in, P=P, zs=zs).VF
        except Exception:
            return None

    P_lo, P_hi = config.min_pressure_Pa, stream.pressure_Pa
    vf_lo = vf_at_P(P_lo)
    if vf_lo is None or vf_lo <= 0.0:
        return None  # liquid never flashes within min pressure

    vf_hi = vf_at_P(P_hi)
    if vf_hi is None or vf_hi > 1e-6:
        return None  # inlet already two-phase → no meaningful bubble P below P_in

    for _ in range(_BISECT_ITERS):
        P_mid = 0.5 * (P_lo + P_hi)
        vf = vf_at_P(P_mid)
        if vf is None:
            return None
        if vf <= 0.0:
            P_hi = P_mid
        else:
            P_lo = P_mid
        if (P_hi - P_lo) / stream.pressure_Pa < 1e-6:
            break

    P_bubble = P_lo
    ratio = P_bubble / stream.pressure_Pa
    return ratio if 0.0 < ratio < 1.0 else None


def _valid_actions_uncached(
    state: SearchState,
    config: MCTSConfig,
    provider: ThermoFlashProvider | None = None,
    feed_stream: StreamState | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
) -> list[UnitAction]:
    if _is_terminal(state, config, feed_stream):
        return []

    actions: list[UnitAction] = []
    product_exists = _product(state, config.product_role) is not None
    _total_dist_count = sum(1 for a in state.unit_sequence if a.kind == "distillation")
    _distillation_allowed = config.enable_distillation_actions and (
        config.max_total_distillation_count is None
        or _total_dist_count < config.max_total_distillation_count
    )

    for stream in state.open_streams:
        if stream.molar_flow_mols < config.min_flow_mols:
            continue

        if (
            config.objective_mode == "single_product"
            and not product_exists
            and _can_accept(stream, config)
        ):
            actions.append(
                UnitAction(kind="accept", stream_id=stream.id, role=config.product_role)
            )

    for stream in _processing_streams_for_valid_actions(
        state,
        config,
        feed_stream,
        diagnostics,
    ):
        # Condition 1: stream meets product purity threshold — no further processing.
        if _stream_is_pure_enough(stream, config):
            continue

        # Condition 2 (flash): only offer flash when two-phase split is possible.
        # Single-phase streams (VF=0 or VF=1) carry no flash separation potential.
        # P/T-changing actions (compressor, valve, HX) are still offered below so
        # that streams blocked by an insufficient α ratio can reach a pressure/
        # temperature where distillation or flash becomes feasible.
        if (
            _flash_count(stream) < config.max_flash_count_per_path
            and _stream_has_flash_potential(stream, provider)
        ):
            actions.append(UnitAction(kind="flash", stream_id=stream.id))

        for delta_T in config.allowed_delta_T_K:
            target_T = stream.temperature_K + delta_T
            if config.min_temperature_K <= target_T <= config.max_temperature_K:
                actions.append(
                    UnitAction(kind="hx", stream_id=stream.id, delta_T_K=float(delta_T))
                )

        if provider is not None and config.hx_target_states:
            _seen_hx_dt: set[float] = {
                a.delta_T_K
                for a in actions
                if a.kind == "hx" and a.stream_id == stream.id and a.delta_T_K is not None
            }
            for target in config.hx_target_states:
                dt = _resolve_hx_target_delta_T(stream, target, config, provider)
                if dt is None or abs(dt) < 0.5:
                    continue
                target_T = stream.temperature_K + dt
                if not (config.min_temperature_K <= target_T <= config.max_temperature_K):
                    continue
                if any(abs(dt - seen) < 0.1 for seen in _seen_hx_dt):
                    continue
                _seen_hx_dt.add(dt)
                actions.append(UnitAction(kind="hx", stream_id=stream.id, delta_T_K=dt))

        _stream_vf: float | None = None
        _stream_vf_computed = False

        def _get_stream_vf() -> float | None:
            nonlocal _stream_vf, _stream_vf_computed
            if not _stream_vf_computed:
                _stream_vf = (
                    _stream_vapor_fraction(stream, provider) if provider is not None else None
                )
                _stream_vf_computed = True
            return _stream_vf

        _compressor_phase_ok = True
        if config.compressor_min_inlet_vapor_fraction > 0.0:
            vf = _get_stream_vf()
            if vf is not None and vf < config.compressor_min_inlet_vapor_fraction:
                _compressor_phase_ok = False

        if _compressor_phase_ok:
            for pressure_ratio in config.allowed_compression_ratios:
                target_pressure = stream.pressure_Pa * pressure_ratio
                if pressure_ratio > 1.0 and target_pressure <= config.max_pressure_Pa:
                    actions.append(
                        UnitAction(
                            kind="compressor",
                            stream_id=stream.id,
                            pressure_ratio=float(pressure_ratio),
                        )
                    )

            for delta_P in config.allowed_compression_delta_P_Pa:
                target_pressure = stream.pressure_Pa + delta_P
                if delta_P > 0.0 and target_pressure <= config.max_pressure_Pa:
                    actions.append(
                        UnitAction(
                            kind="compressor",
                            stream_id=stream.id,
                            delta_P_Pa=float(delta_P),
                        )
                    )

            if provider is not None and config.compressor_target_states:
                _seen_comp_ratio: set[float] = set()
                for a in actions:
                    if a.kind != "compressor" or a.stream_id != stream.id:
                        continue
                    if a.pressure_ratio is not None:
                        _seen_comp_ratio.add(a.pressure_ratio)
                    elif a.delta_P_Pa is not None:
                        _seen_comp_ratio.add(
                            (stream.pressure_Pa + a.delta_P_Pa) / stream.pressure_Pa
                        )
                for target in config.compressor_target_states:
                    ratio = _resolve_compressor_target_ratio(stream, target, config, provider)
                    if ratio is None or ratio <= 1.0:
                        continue
                    if stream.pressure_Pa * ratio > config.max_pressure_Pa:
                        continue
                    if any(abs(ratio - r) < 1e-4 for r in _seen_comp_ratio):
                        continue
                    _seen_comp_ratio.add(ratio)
                    actions.append(
                        UnitAction(kind="compressor", stream_id=stream.id, pressure_ratio=ratio)
                    )

        _pump_phase_ok = True
        if config.pump_max_inlet_vapor_fraction < 1.0:
            vf = _get_stream_vf()
            if vf is not None and vf > config.pump_max_inlet_vapor_fraction:
                _pump_phase_ok = False

        if _pump_phase_ok:
            for pressure_ratio in config.allowed_pump_pressure_ratios:
                target_pressure = stream.pressure_Pa * pressure_ratio
                if pressure_ratio > 1.0 and target_pressure <= config.max_pressure_Pa:
                    actions.append(
                        UnitAction(
                            kind="pump",
                            stream_id=stream.id,
                            pressure_ratio=float(pressure_ratio),
                        )
                    )

            for delta_P in config.allowed_pump_delta_P_Pa:
                target_pressure = stream.pressure_Pa + delta_P
                if delta_P > 0.0 and target_pressure <= config.max_pressure_Pa:
                    actions.append(
                        UnitAction(
                            kind="pump",
                            stream_id=stream.id,
                            delta_P_Pa=float(delta_P),
                        )
                    )

            if provider is not None and config.pump_target_states:
                _seen_pump_ratio: set[float] = set()
                for a in actions:
                    if a.kind != "pump" or a.stream_id != stream.id:
                        continue
                    if a.pressure_ratio is not None:
                        _seen_pump_ratio.add(a.pressure_ratio)
                    elif a.delta_P_Pa is not None:
                        _seen_pump_ratio.add(
                            (stream.pressure_Pa + a.delta_P_Pa) / stream.pressure_Pa
                        )
                for target in config.pump_target_states:
                    ratio = _resolve_pump_target_ratio(stream, target, provider)
                    if ratio is None or ratio <= 1.0:
                        continue
                    if stream.pressure_Pa * ratio > config.max_pressure_Pa:
                        continue
                    if any(abs(ratio - r) < 1e-4 for r in _seen_pump_ratio):
                        continue
                    _seen_pump_ratio.add(ratio)
                    actions.append(
                        UnitAction(kind="pump", stream_id=stream.id, pressure_ratio=ratio)
                    )

        for pressure_ratio in config.allowed_valve_pressure_ratios:
            target_pressure = stream.pressure_Pa * pressure_ratio
            if 0.0 < pressure_ratio < 1.0 and target_pressure >= config.min_pressure_Pa:
                actions.append(
                    UnitAction(
                        kind="valve",
                        stream_id=stream.id,
                        pressure_ratio=float(pressure_ratio),
                    )
                )

        for delta_P in config.allowed_valve_delta_P_Pa:
            target_pressure = stream.pressure_Pa - delta_P
            if delta_P > 0.0 and target_pressure >= config.min_pressure_Pa:
                actions.append(
                    UnitAction(
                        kind="valve",
                        stream_id=stream.id,
                        delta_P_Pa=float(delta_P),
                    )
                )

        if provider is not None and config.valve_target_states:
            _seen_valve_ratio: set[float] = set()
            for a in actions:
                if a.kind != "valve" or a.stream_id != stream.id:
                    continue
                if a.pressure_ratio is not None:
                    _seen_valve_ratio.add(a.pressure_ratio)
                elif a.delta_P_Pa is not None:
                    _seen_valve_ratio.add(
                        (stream.pressure_Pa - a.delta_P_Pa) / stream.pressure_Pa
                    )
            for target in config.valve_target_states:
                ratio = _resolve_valve_target_ratio(stream, target, config, provider)
                if ratio is None or not (0.0 < ratio < 1.0):
                    continue
                if stream.pressure_Pa * ratio < config.min_pressure_Pa:
                    continue
                if any(abs(ratio - r) < 1e-4 for r in _seen_valve_ratio):
                    continue
                _seen_valve_ratio.add(ratio)
                actions.append(
                    UnitAction(kind="valve", stream_id=stream.id, pressure_ratio=ratio)
                )

        if provider is not None and _distillation_allowed:
            actions.extend(
                _distillation_actions(
                    stream,
                    provider,
                    config,
                    distillation_result_cache,
                    diagnostics,
                    relative_volatility_cache,
                    process_graph=state.process_graph,
                )
            )

        if (
            config.enable_recycle_actions
            and state.feed_stream is not None
            and stream.id != state.feed_stream.id
            and _recycle_count(stream) < config.max_recycle_count_per_path
            and max(stream.composition.values(), default=1.0) < config.recycle_purity_threshold
        ):
            actions.append(UnitAction(kind="recycle", stream_id=stream.id))

    return actions


def _processing_streams_for_valid_actions(
    state: SearchState,
    config: MCTSConfig,
    feed_stream: StreamState | None,
    diagnostics: _DiagnosticsAccumulator | None = None,
) -> tuple[StreamState, ...]:
    streams = tuple(
        stream
        for stream in state.open_streams
        if stream.molar_flow_mols >= config.min_flow_mols
    )
    if config.max_active_streams_per_state is not None and config.max_active_streams_per_state < 0:
        raise ValueError("max_active_streams_per_state must be non-negative or None.")
    if config.min_stream_priority < 0.0:
        raise ValueError("min_stream_priority must be non-negative.")
    if not _stream_priority_gating_enabled(config):
        return streams

    ranked = rank_streams_by_priority(
        streams,
        feed_stream=feed_stream,
        min_component_fraction=config.min_component_fraction,
    )
    selected = [
        stream
        for stream, priority in ranked
        if priority >= config.min_stream_priority
    ]
    if config.max_active_streams_per_state is not None:
        selected = selected[: config.max_active_streams_per_state]

    if diagnostics is not None:
        diagnostics.n_stream_priority_gating_calls += 1
        diagnostics.n_stream_priority_streams_considered += len(streams)
        diagnostics.n_stream_priority_streams_gated += len(streams) - len(selected)
    return tuple(selected)


def _stream_priority_gating_enabled(config: MCTSConfig) -> bool:
    return (
        config.max_active_streams_per_state is not None
        or config.min_stream_priority > 0.0
    )


def _stream_is_pure_enough(stream: StreamState, config: MCTSConfig) -> bool:
    """Return True when the stream's dominant component meets the purity threshold.

    When True, the stream receives no further actions — it is treated as an
    auto-closed product.  Returns False when ``product_purity_threshold >= 1.0``
    (disabled) or when the stream has no composition data.
    """
    if config.product_purity_threshold >= 1.0 or not stream.composition:
        return False
    return max(stream.composition.values()) >= config.product_purity_threshold


def _stream_has_flash_potential(
    stream: StreamState,
    provider: "ThermoFlashProvider | None",
) -> bool:
    """Return True when the stream is two-phase (0 < VF < 1) at current T/P.

    A single-phase stream (VF = 0 or VF = 1) produces no net composition
    split from a flash and should not receive a flash action.  Returns True
    when no provider is available (conservative: assume flash might work).
    """
    if provider is None:
        return True
    vf = _stream_vapor_fraction(stream, provider)
    if vf is None:
        return True   # flash computation failed — don't suppress the action
    return 0.0 < vf < 1.0


def _distillation_lineage_pair_counts(
    stream_id: str,
    graph: "ProcessGraph",
) -> "Counter[tuple[str, str]]":
    """Count (lk, hk) distillation pairs in the ancestor lineage of stream_id.

    Walks backwards through the process graph from the given stream node,
    following feed edges through each unit operation, and records the
    (light_key, heavy_key) pair for every distillation unit encountered.
    """
    counts: Counter[tuple[str, str]] = Counter()

    nodes_by_id = {node.id: node for node in graph.nodes}
    edges_by_target: dict[str, list] = {}
    for edge in graph.edges:
        edges_by_target.setdefault(edge.target, []).append(edge)

    stream_node_id = None
    for label, nid in reversed(graph.stream_node_ids):
        if label == stream_id:
            stream_node_id = nid
            break
    if stream_node_id is None:
        return counts

    current_node_id = stream_node_id
    while True:
        parent_edges = edges_by_target.get(current_node_id, [])
        if not parent_edges:
            break
        unit_node_id = parent_edges[0].source
        unit_node = nodes_by_id.get(unit_node_id)
        if unit_node is None or unit_node.kind != "unit":
            break

        unit_data = dict(unit_node.data)
        if unit_data.get("action_kind") == "distillation":
            sig = unit_data.get("action_signature", ())
            if len(sig) >= 3 and sig[1] is not None and sig[2] is not None:
                counts[(sig[1], sig[2])] += 1

        feed_edges = [e for e in edges_by_target.get(unit_node_id, []) if e.role == "feed"]
        if not feed_edges:
            break
        current_node_id = feed_edges[0].source

    return counts


def _distillation_actions(
    stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
    process_graph: "ProcessGraph | None" = None,
) -> list[UnitAction]:
    if diagnostics is not None:
        diagnostics.n_distillation_action_generation_calls += 1
    if _distillation_count(stream) >= config.max_distillation_count_per_path:
        return []
    try:
        alphas = _estimate_relative_volatilities_cached(
            stream,
            provider,
            config,
            relative_volatility_cache,
            diagnostics,
        )
    except ValueError:
        return []

    meaningful = [
        compound
        for compound in provider.compounds
        if stream.molar_flow_mols * stream.composition.get(compound, 0.0)
        >= config.distillation_min_key_flow_mols
        and alphas.get(compound, 0.0) > 0.0
    ]
    if len(meaningful) < 2:
        return []

    ordered = sorted(meaningful, key=lambda compound: alphas[compound], reverse=True)
    if config.distillation_key_pair_mode == "adjacent":
        key_pairs = list(zip(ordered, ordered[1:]))
    elif config.distillation_key_pair_mode == "all":
        key_pairs = [
            (light_key, heavy_key)
            for light_index, light_key in enumerate(ordered)
            for heavy_key in ordered[light_index + 1 :]
        ]
    else:
        raise ValueError(
            "distillation_key_pair_mode must be 'adjacent' or 'all'."
        )
    if diagnostics is not None:
        diagnostics.n_distillation_candidate_key_pairs += len(key_pairs)

    lineage_counts: Counter[tuple[str, str]] | None = None
    if config.max_same_key_pair_count_per_lineage is not None and process_graph is not None:
        lineage_counts = _distillation_lineage_pair_counts(stream.id, process_graph)

    actions: list[UnitAction] = []
    for light_key, heavy_key in key_pairs:
        if (
            lineage_counts is not None
            and lineage_counts.get((light_key, heavy_key), 0)
            >= config.max_same_key_pair_count_per_lineage  # type: ignore[operator]
        ):
            continue

        alpha_ratio = alphas[light_key] / alphas[heavy_key]
        if alpha_ratio < config.distillation_min_alpha_ratio:
            if diagnostics is not None:
                diagnostics.n_distillation_candidates_filtered_alpha += 1
            continue

        for lk_recovery in config.distillation_light_key_recoveries:
            for hk_recovery in config.distillation_heavy_key_recoveries:
                if not 0.0 < hk_recovery < lk_recovery < 1.0:
                    if diagnostics is not None:
                        diagnostics.n_distillation_candidates_filtered_invalid_recovery += len(
                            config.distillation_reflux_multipliers
                        )
                    continue
                for reflux_multiplier in config.distillation_reflux_multipliers:
                    if diagnostics is not None:
                        diagnostics.n_distillation_candidate_grid_actions += 1
                    if reflux_multiplier <= 1.0:
                        if diagnostics is not None:
                            diagnostics.n_distillation_candidates_filtered_invalid_recovery += 1
                        continue
                    candidate = UnitAction(
                        kind="distillation",
                        stream_id=stream.id,
                        light_key=light_key,
                        heavy_key=heavy_key,
                        light_key_recovery=float(lk_recovery),
                        heavy_key_recovery=float(hk_recovery),
                        reflux_ratio_multiplier=float(reflux_multiplier),
                    )
                    if (
                        not config.validate_distillation_candidates
                        or _distillation_candidate_is_feasible(
                            stream,
                            provider,
                            config,
                            candidate,
                            alphas,
                            distillation_result_cache,
                            diagnostics,
                        )
                    ):
                        actions.append(candidate)
                        if diagnostics is not None:
                            diagnostics.n_distillation_candidate_actions_generated += 1
                    elif diagnostics is not None:
                        diagnostics.n_distillation_candidates_filtered_infeasible += 1
    return actions


def _distillation_candidate_is_feasible(
    stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    action: UnitAction,
    alphas: dict[str, float],
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
) -> bool:
    try:
        result = _shortcut_distillation_fug_cached(
            stream,
            provider,
            config,
            action,
            relative_volatilities=alphas,
            distillation_result_cache=distillation_result_cache,
            diagnostics=diagnostics,
        )
    except (ArithmeticError, ValueError):
        return False
    if not result.success:
        return False
    if result.theoretical_stages is None:
        return False
    if result.theoretical_stages > config.distillation_max_theoretical_stages:
        return False
    if result.distillate_stream is None or result.bottoms_stream is None:
        return False
    if result.distillate_stream.temperature_K < config.min_temperature_K:
        return False
    if result.bottoms_stream.temperature_K < config.min_temperature_K:
        return False
    return (
        result.distillate_stream.molar_flow_mols >= config.min_flow_mols
        and result.bottoms_stream.molar_flow_mols >= config.min_flow_mols
    )


def _apply_action(
    state: SearchState,
    action: UnitAction,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    action_cache: dict[tuple, _CachedActionOutcome] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
) -> SearchState:
    stream = _open_stream(state, action.stream_id)
    if stream is None:
        return _with_error(state, action, f"Open stream not found: {action.stream_id}")
    if action.kind == "accept":
        return _apply_action_uncached(
            state,
            action,
            provider,
            config,
            stream,
            distillation_result_cache,
            diagnostics,
        )
    if action_cache is None or action.kind not in config.cached_action_kinds:
        return _apply_action_uncached_timed(
            state,
            action,
            provider,
            config,
            stream,
            diagnostics,
            distillation_result_cache,
        )

    key = _apply_action_cache_key(stream, action, provider, config)
    cached = action_cache.get(key)
    if cached is not None:
        if diagnostics is not None:
            diagnostics.n_apply_action_cache_hits += 1
            diagnostics.apply_action_cache_saved_estimate_s += cached.calculation_time_s
        return _state_from_cached_action_outcome(state, stream, action, cached)

    if diagnostics is not None:
        diagnostics.n_apply_action_cache_misses += 1
    started_at = time.monotonic()
    next_state = _apply_action_uncached(
        state,
        action,
        provider,
        config,
        stream,
        distillation_result_cache,
        diagnostics,
    )
    calculation_time_s = time.monotonic() - started_at
    if diagnostics is not None:
        diagnostics.apply_action_calc_time_s += calculation_time_s
    if _can_store_apply_action_cache_entry(action_cache, config):
        action_cache[key] = _cached_action_outcome_from_transition(
            state,
            stream,
            next_state,
            calculation_time_s,
        )
    return next_state


def _apply_action_uncached_timed(
    state: SearchState,
    action: UnitAction,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    stream: StreamState,
    diagnostics: _DiagnosticsAccumulator | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
) -> SearchState:
    started_at = time.monotonic()
    next_state = _apply_action_uncached(
        state,
        action,
        provider,
        config,
        stream,
        distillation_result_cache,
        diagnostics,
    )
    if diagnostics is not None:
        diagnostics.apply_action_calc_time_s += time.monotonic() - started_at
    return next_state


def _apply_action_uncached(
    state: SearchState,
    action: UnitAction,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    stream: StreamState,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
) -> SearchState:
    if action.kind == "hx":
        if action.delta_T_K is None:
            return _with_error(state, action, "HX action missing delta_T_K.")
        result = heat_stream(
            stream,
            provider,
            delta_T_K=action.delta_T_K,
            outlet_stream_id=_hx_stream_id(stream, action, len(state.unit_sequence) + 1),
        )
        if not result.success or result.outlet_stream is None:
            return _with_error(state, action, result.error_message or "HX action failed.")
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            ((result.outlet_stream, "out"),),
            total_abs_duty_delta_W=abs(result.duty_W or 0.0),
        )

    if action.kind == "compressor":
        if (action.pressure_ratio is None) == (action.delta_P_Pa is None):
            return _with_error(
                state,
                action,
                "Compressor action requires exactly one of pressure_ratio or delta_P_Pa.",
            )
        result = compress_stream(
            stream,
            provider,
            pressure_ratio=action.pressure_ratio,
            delta_P_Pa=action.delta_P_Pa,
            isentropic_efficiency=config.compressor_isentropic_efficiency,
            mechanical_efficiency=config.compressor_mechanical_efficiency,
            outlet_stream_id=_compressor_stream_id(
                stream,
                action,
                len(state.unit_sequence) + 1,
            ),
        )
        if not result.success or result.outlet_stream is None:
            return _with_error(
                state,
                action,
                result.error_message or "Compressor action failed.",
            )
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            ((result.outlet_stream, "out"),),
            total_abs_duty_delta_W=abs(result.shaft_power_W or 0.0),
        )

    if action.kind == "pump":
        if (action.pressure_ratio is None) == (action.delta_P_Pa is None):
            return _with_error(
                state,
                action,
                "Pump action requires exactly one of pressure_ratio or delta_P_Pa.",
            )
        result = pump_stream(
            stream,
            provider,
            pressure_ratio=action.pressure_ratio,
            delta_P_Pa=action.delta_P_Pa,
            isentropic_efficiency=config.pump_isentropic_efficiency,
            mechanical_efficiency=config.pump_mechanical_efficiency,
            max_inlet_vapor_fraction=config.pump_max_inlet_vapor_fraction,
            outlet_stream_id=_pump_stream_id(
                stream,
                action,
                len(state.unit_sequence) + 1,
            ),
        )
        if not result.success or result.outlet_stream is None:
            return _with_error(
                state,
                action,
                result.error_message or "Pump action failed.",
            )
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            ((result.outlet_stream, "out"),),
            total_abs_duty_delta_W=abs(result.shaft_power_W or 0.0),
        )

    if action.kind == "valve":
        if (action.pressure_ratio is None) == (action.delta_P_Pa is None):
            return _with_error(
                state,
                action,
                "Valve action requires exactly one of pressure_ratio or delta_P_Pa.",
            )
        result = valve_stream(
            stream,
            provider,
            pressure_ratio=action.pressure_ratio,
            delta_P_Pa=action.delta_P_Pa,
            outlet_stream_id=_valve_stream_id(
                stream,
                action,
                len(state.unit_sequence) + 1,
            ),
        )
        if not result.success or result.outlet_stream is None:
            return _with_error(
                state,
                action,
                result.error_message or "Valve action failed.",
            )
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            ((result.outlet_stream, "out"),),
        )

    if action.kind == "distillation":
        missing = [
            name
            for name, value in (
                ("light_key", action.light_key),
                ("heavy_key", action.heavy_key),
                ("light_key_recovery", action.light_key_recovery),
                ("heavy_key_recovery", action.heavy_key_recovery),
                ("reflux_ratio_multiplier", action.reflux_ratio_multiplier),
            )
            if value is None
        ]
        if missing:
            return _with_error(
                state,
                action,
                f"Distillation action missing required fields: {missing}.",
            )
        try:
            result = _shortcut_distillation_fug_cached(
                stream,
                provider,
                config,
                action,
                distillation_result_cache=distillation_result_cache,
                diagnostics=diagnostics,
            )
        except (ArithmeticError, ValueError) as exc:
            return _with_error(
                state,
                action,
                f"Distillation action failed during shortcut calculation: {exc}",
            )
        if not result.success or result.distillate_stream is None or result.bottoms_stream is None:
            return _with_error(
                state,
                action,
                result.error_message or "Distillation action failed.",
            )
        if (
            result.theoretical_stages is not None
            and result.theoretical_stages > config.distillation_max_theoretical_stages
        ):
            return _with_error(
                state,
                action,
                "Distillation action exceeded max theoretical stages: "
                f"{result.theoretical_stages:g} > "
                f"{config.distillation_max_theoretical_stages:g}.",
            )
        distillate_stream, bottoms_stream = _rebased_distillation_streams(
            result,
            stream,
            action,
            len(state.unit_sequence) + 1,
        )
        children_with_roles = tuple(
            (child, role)
            for child, role in (
                (distillate_stream, "distillate"),
                (bottoms_stream, "bottoms"),
            )
            if child.molar_flow_mols >= config.min_flow_mols
        )
        children = [child for child, _ in children_with_roles]
        if not children:
            return _with_error(
                state,
                action,
                f"Distillation on '{stream.id}' produced no retained streams.",
            )
        n_stages = float(result.theoretical_stages or 0.0)
        if config.include_reboiler_duty and result.reflux_ratio is not None:
            try:
                q_cond_eb, q_reb_eb = column_duties_from_energy_balance(
                    stream, distillate_stream, bottoms_stream, result.reflux_ratio, provider
                )
                duty_delta_W = abs(q_cond_eb) + abs(q_reb_eb)
            except ValueError as exc:
                return _with_error(
                    state, action,
                    f"Distillation energy balance failed for '{stream.id}': {exc}",
                )
        else:
            duty_delta_W = _distillation_condenser_duty_W(result, config)
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            children_with_roles,
            total_abs_duty_delta_W=duty_delta_W,
            total_theoretical_stages_delta=n_stages,
        )

    if action.kind == "flash":
        result = flash_split(stream, provider)
        if (
            not result.success
            or result.phase_state != "two_phase"
            or result.vapor_stream is None
            or result.liquid_stream is None
        ):
            return _with_error(
                state,
                action,
                result.error_message or f"Flash on '{stream.id}' did not produce two phases.",
            )
        children_with_roles = tuple(
            (child, role)
            for child, role in (
                (result.vapor_stream, "vapor"),
                (result.liquid_stream, "liquid"),
            )
            if child is not None and child.molar_flow_mols >= config.min_flow_mols
        )
        children = [child for child, _ in children_with_roles]
        if not children:
            return _with_error(state, action, f"Flash on '{stream.id}' produced no retained streams.")
        return _state_with_unit_outputs(
            state,
            stream,
            action,
            children_with_roles,
            total_abs_duty_delta_W=abs(result.duty_W or 0.0),
        )

    if action.kind == "accept":
        role = action.role or config.product_role
        if not _can_accept(stream, config):
            return _with_error(state, action, f"Stream '{stream.id}' cannot be accepted as {role}.")
        return _state_with_product(state, stream, action, role)

    if action.kind == "recycle":
        feed = state.feed_stream
        if feed is None:
            return _with_error(state, action, "Recycle action requires feed_stream in SearchState.")
        F_r = stream.molar_flow_mols
        F_f = feed.molar_flow_mols
        F_mix = F_r + F_f
        all_components = set(feed.composition) | set(stream.composition)
        z_mix = {
            c: (F_f * feed.composition.get(c, 0.0) + F_r * stream.composition.get(c, 0.0)) / F_mix
            for c in all_components
        }
        T_mix = (F_f * feed.temperature_K + F_r * stream.temperature_K) / F_mix
        P_mix = min(feed.pressure_Pa, stream.pressure_Pa)
        mixed_id = _recycle_stream_id(stream, len(state.unit_sequence) + 1)
        mixed_stream = StreamState(
            id=mixed_id,
            temperature_K=T_mix,
            pressure_Pa=P_mix,
            molar_flow_mols=F_mix,
            composition=z_mix,
            history=stream.history + ("recycle:feed_merge",),
        )
        outputs = ((mixed_stream, "open"),)
        graph = append_mixer_unit(
            _process_graph_for_state(state),
            stream.id,
            feed.id,
            action,
            outputs,
            action_signature(action),
        )
        return _state_with_unit_outputs(state, stream, action, outputs, process_graph=graph)

    return _with_error(state, action, f"Unsupported action kind: {action.kind}")


def _rollout_action(
    state: SearchState,
    actions: list[UnitAction],
    config: MCTSConfig,
    rng: random.Random,
) -> UnitAction:
    accept_actions = [action for action in actions if action.kind == "accept"]
    if accept_actions:
        return min(
            accept_actions,
            key=lambda action: _stream_target_error(_open_stream(state, action.stream_id), config),
        )

    flash_actions = [action for action in actions if action.kind == "flash"]
    has_hx = any(action.kind == "hx" for action in state.unit_sequence)
    if has_hx and flash_actions and rng.random() < 0.55:
        return rng.choice(flash_actions)

    return rng.choice(actions)


def _distillation_condenser_duty_W(
    result: ShortcutDistillationResult,
    config: MCTSConfig,
) -> float:
    """Estimate condenser duty for a distillation column as R * D * lambda.

    Returns zero when the heat-of-vaporization parameter is disabled or
    when the result does not contain the required fields.
    """
    if config.distillation_molar_heat_of_vaporization_J_mol <= 0.0:
        return 0.0
    if result.reflux_ratio is None or result.distillate_stream is None:
        return 0.0
    return (
        result.reflux_ratio
        * float(result.distillate_stream.molar_flow_mols)
        * config.distillation_molar_heat_of_vaporization_J_mol
    )


def _effective_leaf_discount(config: MCTSConfig, feed_stream: StreamState) -> float:
    """Return the effective leaf_value_discount for a given search.

    When ``config.leaf_value_discount`` is ``None`` (the default), returns 0.5.
    This represents "at most 50% extra optimism on top of S_norm" in the
    normalised [0, 1] reward space, giving V_max ≈ 1.5 for original_s+γU.
    The old auto value (N_C/2) was in the unnormalised [0, N_C] space and
    produced Q-values up to 7.5 for a 5-component system, inconsistent with
    the [0, 1] base and making exploration_weight c hard to tune.
    """
    if config.leaf_value_discount is not None:
        return config.leaf_value_discount
    return 0.5


def _flow_weighted_mean_alpha(
    state: SearchState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    relative_volatility_cache: "dict[tuple, _CachedRelativeVolatilities] | None" = None,
) -> float:
    """Flow-weighted mean of the best relative volatility per open stream.

    Used as the α-filter threshold in K-sample rollouts. A rollout sample is
    considered productive when this value at the endpoint exceeds the value at
    the leaf — meaning the bulk separability of open streams improved.

    Using flow-weighted mean rather than max prevents a small high-α distillate
    from carrying a rollout that left a large low-α bottoms unimproved.

    Returns 0.0 when there are no open streams.
    """
    if not state.open_streams:
        return 0.0
    F_total = sum(s.molar_flow_mols for s in state.open_streams)
    if F_total <= 0:
        return 0.0
    total = 0.0
    for stream in state.open_streams:
        try:
            alphas = _estimate_relative_volatilities_cached(
                stream, provider, config, relative_volatility_cache
            )
            alpha_max = max(alphas.values()) if alphas else 1.0
        except ValueError:
            alpha_max = 1.0
        total += (stream.molar_flow_mols / F_total) * alpha_max
    return total


def _separation_potential(
    state: SearchState,
    config: MCTSConfig,
    feed_stream: StreamState,
    provider: "ThermoFlashProvider | None" = None,
    relative_volatility_cache: "dict[tuple, _CachedRelativeVolatilities] | None" = None,
) -> float:
    """Estimate remaining separable work for the leaf value estimator.

    Three modes controlled by ``config.leaf_potential_mode``:

    ``"flow_weighted_sum"`` (default):
        Sum of flow-weighted normalised composition entropy across all open
        streams.  Large streams dominate the signal.

    ``"max_entropy"``:
        Maximum normalised composition entropy over all open streams.
        A single highly-mixed stream (regardless of its flow rate) drives
        the potential, preventing small but important streams from being
        overshadowed by larger ones.

    ``"alpha_weighted"``:
        Flow-weighted sum of ``H_norm × f(α_best)`` per open stream where
        ``f(α) = 1 − 1/α`` maps ``[1, ∞) → [0, 1)``.  H_norm measures
        compositional disorder; f(α_best) captures thermodynamic
        separability at current T/P.  The product is bounded in ``[0, 1]``:
        a single-phase stream (α = 1) contributes 0; a highly separable
        stream (α → ∞) contributes H_norm.  Requires ``provider``; falls
        back to ``"max_entropy"`` when provider is None.

    ``"remaining_mi"``:
        Exact upper bound on the MI still recoverable from open streams,
        assuming each open stream could be perfectly separated::

            Φ = N_C × Σ_{open k} (F_k / F_0) × H_norm(z_k)

        This is derived directly from the equal-weight MI formula: if every
        open stream were split into pure components, their conditional
        entropy would drop to zero and MI would increase by exactly this
        amount.  With ``leaf_value_discount=1.0`` the total reward becomes
        ``MI_current + Φ_remaining``, an optimistic-but-consistent upper
        bound on the achievable MI from this state.  No thermodynamic calls
        are needed.

    Returns zero when there are no open streams.
    """
    if not state.open_streams:
        return 0.0

    if config.leaf_potential_mode == "alpha_weighted":
        component_order = tuple(
            c for c, z in feed_stream.composition.items()
            if z >= config.min_component_fraction
        )
        if provider is None:
            # fallback: max_entropy (no thermodynamic calls available)
            return max(
                stream_composition_entropy(
                    s,
                    components=component_order,
                    normalise=True,
                    min_component_fraction=config.min_component_fraction,
                )
                for s in state.open_streams
            )
        F_0 = feed_stream.molar_flow_mols
        total = 0.0
        for stream in state.open_streams:
            h = stream_composition_entropy(
                stream,
                components=component_order,
                normalise=True,
                min_component_fraction=config.min_component_fraction,
            )
            try:
                alphas = _estimate_relative_volatilities_cached(
                    stream, provider, config, relative_volatility_cache
                )
                alpha_max = max(alphas.values()) if alphas else 1.0
            except ValueError:
                alpha_max = 1.0
            # f(α) = 1 − 1/α  maps [1, ∞) → [0, 1); single-phase (α=1) → 0
            f_alpha = 1.0 - 1.0 / alpha_max if alpha_max > 1.0 else 0.0
            weight = stream.molar_flow_mols / F_0 if F_0 > 0 else 0.0
            total += weight * h * f_alpha
        return total

    if config.leaf_potential_mode == "max_entropy":
        component_order = tuple(
            c for c, z in feed_stream.composition.items()
            if z >= config.min_component_fraction
        )
        return max(
            stream_composition_entropy(
                s,
                components=component_order,
                normalise=True,
                min_component_fraction=config.min_component_fraction,
            )
            for s in state.open_streams
        )

    if config.leaf_potential_mode == "remaining_mi":
        component_order = tuple(
            c for c, z in feed_stream.composition.items()
            if z >= config.min_component_fraction
        )
        n_c = len(component_order)
        F_0 = feed_stream.molar_flow_mols
        if n_c == 0 or F_0 <= 0:
            return 0.0
        return n_c * sum(
            (s.molar_flow_mols / F_0) * stream_composition_entropy(
                s,
                components=component_order,
                normalise=True,
                min_component_fraction=config.min_component_fraction,
            )
            for s in state.open_streams
        )

    # "flow_weighted_sum" (default)
    return sum(
        stream_priority(
            s,
            feed_stream=feed_stream,
            min_component_fraction=config.min_component_fraction,
        )
        for s in state.open_streams
    )


def _reward(
    state: SearchState,
    config: MCTSConfig,
    feed_stream: StreamState | None = None,
    provider: "ThermoFlashProvider | None" = None,
    relative_volatility_cache: "dict[tuple, _CachedRelativeVolatilities] | None" = None,
) -> float:
    if config.objective_mode == "complete_separation":
        if feed_stream is None:
            return -config.missing_product_penalty
        metric = _complete_separation_metric(state, config, feed_stream)
        # All rewards in complete_separation mode are normalised to [0, 1] so
        # that exploration_weight c is independent of N_C and comparable across
        # problem sizes.  Penalties are divided by n_c to stay on the same scale.
        n_c = max(1, metric["target"])
        action_penalty = config.unit_penalty * len(state.unit_sequence) / n_c
        duty_penalty = config.duty_penalty_per_W * state.total_abs_duty_W / n_c
        stage_penalty = config.stage_count_penalty_per_stage * state.total_theoretical_stages / n_c
        error_penalty = 0.5 * len(state.errors) / n_c
        base = metric["fraction_of_target"] - action_penalty - duty_penalty - stage_penalty - error_penalty
        if config.use_leaf_value_estimator and state.open_streams:
            if config.leaf_potential_mode == "depth_aware_bounded":
                S_norm = metric["fraction_of_target"]   # ∈ [0, 1]
                component_order = tuple(
                    c for c, z_val in feed_stream.composition.items()
                    if z_val >= config.min_component_fraction
                )
                F_0 = feed_stream.molar_flow_mols
                U = (
                    sum(
                        (s.molar_flow_mols / F_0) * stream_composition_entropy(
                            s,
                            components=component_order,
                            normalise=True,
                            min_component_fraction=config.min_component_fraction,
                        )
                        for s in state.open_streams
                    )
                    if F_0 > 0 and n_c > 0
                    else 0.0
                )
                # V = S + 0.5·min(U, 1−S) ≤ 0.5·(1+S) ≤ 1  (admissible; avoids
                # saturation at S=0.80: V=0.90 not 1.0, preserving gradient signal)
                bounded_potential = 0.5 * min(U, max(0.0, 1.0 - S_norm))
                return base + bounded_potential
            if config.leaf_potential_mode == "depth_aware_alpha_gated":
                S_norm = metric["fraction_of_target"]
                component_order = tuple(
                    c for c, z_val in feed_stream.composition.items()
                    if z_val >= config.min_component_fraction
                )
                F_0 = feed_stream.molar_flow_mols
                if provider is None or F_0 <= 0 or n_c == 0:
                    # No thermo available — fall back to ungated depth_aware_bounded
                    U_alpha = (
                        sum(
                            (s.molar_flow_mols / F_0) * stream_composition_entropy(
                                s,
                                components=component_order,
                                normalise=True,
                                min_component_fraction=config.min_component_fraction,
                            )
                            for s in state.open_streams
                        )
                        if F_0 > 0 and n_c > 0
                        else 0.0
                    )
                else:
                    U_alpha = 0.0
                    for s in state.open_streams:
                        try:
                            alphas = _estimate_relative_volatilities_cached(
                                s, provider, config, relative_volatility_cache
                            )
                            alpha_max = max(alphas.values()) if alphas else 0.0
                            # Soft gate: partial credit proportional to α/α_threshold.
                            # Hard gate (0/1) gave zero credit to streams blocked at
                            # current P, making compression appear worthless. Soft gate
                            # preserves the separability signal while still down-weighting
                            # thermodynamically difficult streams.
                            q = min(1.0, alpha_max / config.distillation_min_alpha_ratio) if alpha_max > 1.0 else 0.0
                        except Exception:
                            q = 0.0
                        U_alpha += q * (s.molar_flow_mols / F_0) * stream_composition_entropy(
                            s,
                            components=component_order,
                            normalise=True,
                            min_component_fraction=config.min_component_fraction,
                        )
                # Same gamma=0.5 cap as depth_aware_bounded for consistent admissibility
                bounded_potential = 0.5 * min(U_alpha, max(0.0, 1.0 - S_norm))
                return base + bounded_potential
            potential = _separation_potential(
                state, config, feed_stream, provider, relative_volatility_cache
            )
            discount = _effective_leaf_discount(config, feed_stream)
            return base + discount * potential
        return base

    stage_penalty = config.stage_count_penalty_per_stage * state.total_theoretical_stages
    product = _product(state, config.product_role)
    action_penalty = config.unit_penalty * len(state.unit_sequence)
    duty_penalty = config.duty_penalty_per_W * state.total_abs_duty_W
    error_penalty = 0.5 * len(state.errors)

    if product is None:
        partial = _best_partial_error(state, config)
        partial_bonus = 0.0 if partial is None else max(0.0, 1.0 - partial)
        return (
            -config.missing_product_penalty
            + partial_bonus
            - action_penalty
            - duty_penalty
            - stage_penalty
            - error_penalty
        )

    x = product.stream.composition.get(config.target_component)
    if x is None:
        return -config.missing_product_penalty - action_penalty - duty_penalty - stage_penalty - error_penalty

    composition_error = abs(x - config.target_fraction)
    temperature_penalty = _temperature_penalty(product.stream, config)
    return (
        10.0
        - 100.0 * composition_error
        - temperature_penalty
        - action_penalty
        - duty_penalty
        - stage_penalty
        - error_penalty
    )


def _best_partial_error(state: SearchState, config: MCTSConfig) -> float | None:
    candidates = [
        abs(stream.composition[config.target_component] - config.target_fraction)
        for stream in state.open_streams
        if config.target_component in stream.composition and _can_accept(stream, config)
    ]
    if not candidates:
        return None
    return min(candidates)


def _complete_separation_metric(
    state: SearchState,
    config: MCTSConfig,
    feed_stream: StreamState,
) -> dict:
    outlet_streams = [product.stream for product in state.products]
    outlet_streams.extend(state.open_streams)
    if config.separation_score_mode in (
        "mutual_information",
        "mutual_information_equal_weight",
    ):
        weight_mode = (
            "equal_weight"
            if config.separation_score_mode == "mutual_information_equal_weight"
            else "feed_fraction"
        )
        mi = mutual_information_separation(
            feed_stream,
            outlet_streams,
            min_component_fraction=config.min_component_fraction,
            weight_mode=weight_mode,
        )
        pr = separation_indicator(
            feed_stream,
            outlet_streams,
            min_component_fraction=config.min_component_fraction,
        )
        # MI fields are the primary objective; purity×recovery fields are
        # included so that progress recording and graph-similarity comparisons
        # that expect component_scores / best_stream_by_component don't crash.
        return {
            **mi,
            "component_scores": pr["component_scores"],
            "best_stream_by_component": pr["best_stream_by_component"],
            "purities": pr["purities"],
            "recoveries": pr.get("recoveries", {}),
        }
    return separation_indicator(
        feed_stream,
        outlet_streams,
        min_component_fraction=config.min_component_fraction,
    )


def _temperature_penalty(stream: StreamState, config: MCTSConfig) -> float:
    if config.target_product_temperature_K is None:
        return 0.0
    return abs(stream.temperature_K - config.target_product_temperature_K)


def _is_terminal(
    state: SearchState,
    config: MCTSConfig,
    feed_stream: StreamState | None = None,
) -> bool:
    # Hard budget stop always applies regardless of other constraints.
    if len(state.unit_sequence) >= config.max_depth:
        return True

    # Enforce minimum distillation columns before any success-based termination.
    if config.min_distillation_count_per_path is not None:
        n_dist = sum(1 for u in state.unit_sequence if u.kind == "distillation")
        if n_dist < config.min_distillation_count_per_path:
            return False

    if config.objective_mode == "complete_separation" and feed_stream is not None:
        metric = _complete_separation_metric(state, config, feed_stream)
        if metric["score"] >= metric["target"] - config.separation_score_tolerance:
            return True
    if (
        config.objective_mode == "single_product"
        and _product(state, config.product_role) is not None
    ):
        return True
    return len(state.open_streams) == 0


def _can_accept(stream: StreamState, config: MCTSConfig) -> bool:
    if config.objective_mode == "complete_separation":
        if not stream.composition:
            return False
        if config.require_flash_liquid_product and not _has_accepted_liquid_history(stream):
            return False
        return True

    if config.target_component not in stream.composition:
        return False
    if config.require_flash_liquid_product and not _has_accepted_liquid_history(stream):
        return False
    if config.target_product_temperature_K is not None:
        delta = abs(stream.temperature_K - config.target_product_temperature_K)
        if delta > config.product_temperature_tolerance_K:
            return False
    return True


def _dominant_component(stream: StreamState) -> str:
    if not stream.composition:
        return "Product"
    return max(stream.composition, key=stream.composition.get)


def _has_accepted_liquid_history(stream: StreamState) -> bool:
    return any(
        item
        in {
            "flash:liquid",
            "shortcut_distillation:total_condenser_distillate",
            "shortcut_distillation:bottoms",
        }
        for item in stream.history
    )


def _product(state: SearchState, role: str) -> ProductAssignment | None:
    for product in state.products:
        if product.role == role:
            return product
    return None


def _open_stream(state: SearchState, stream_id: str) -> StreamState | None:
    for stream in state.open_streams:
        if stream.id == stream_id:
            return stream
    return None


def _state_with_unit_outputs(
    state: SearchState,
    input_stream: StreamState,
    action: UnitAction,
    output_streams_with_roles: tuple[tuple[StreamState, str], ...],
    total_abs_duty_delta_W: float = 0.0,
    total_theoretical_stages_delta: float = 0.0,
    process_graph: ProcessGraph | None = None,
) -> SearchState:
    if process_graph is None:
        process_graph = append_unit_operation(
            _process_graph_for_state(state),
            input_stream.id,
            action,
            output_streams_with_roles,
            action_signature(action),
        )
    outputs = [stream for stream, _ in output_streams_with_roles]
    return SearchState(
        open_streams=_replace_open_stream(state, input_stream.id, outputs),
        products=state.products,
        unit_sequence=state.unit_sequence + (action,),
        total_abs_duty_W=state.total_abs_duty_W + total_abs_duty_delta_W,
        total_theoretical_stages=state.total_theoretical_stages + total_theoretical_stages_delta,
        errors=state.errors,
        process_graph=process_graph,
        feed_stream=state.feed_stream,
    )


def _state_with_product(
    state: SearchState,
    stream: StreamState,
    action: UnitAction,
    role: str,
) -> SearchState:
    graph = append_product_assignment(_process_graph_for_state(state), stream.id, role)
    return SearchState(
        open_streams=_remove_open_stream(state, stream.id),
        products=state.products + (ProductAssignment(role=role, stream=stream),),
        unit_sequence=state.unit_sequence + (action,),
        total_abs_duty_W=state.total_abs_duty_W,
        total_theoretical_stages=state.total_theoretical_stages,
        errors=state.errors,
        process_graph=graph,
        feed_stream=state.feed_stream,
    )


def _process_graph_for_state(state: SearchState) -> ProcessGraph:
    if state.process_graph.nodes:
        return state.process_graph
    graph = ProcessGraph.empty()
    for stream in state.open_streams:
        graph = append_stream_root(graph, stream)
    for product in state.products:
        graph = append_stream_root(graph, product.stream, role="product")
        graph = append_product_assignment(graph, product.stream.id, product.role)
    return graph


def _replace_open_stream(
    state: SearchState,
    stream_id: str,
    replacements: list[StreamState],
) -> tuple[StreamState, ...]:
    streams: list[StreamState] = []
    for stream in state.open_streams:
        if stream.id == stream_id:
            streams.extend(replacements)
        else:
            streams.append(stream)
    return tuple(streams)


def _remove_open_stream(state: SearchState, stream_id: str) -> tuple[StreamState, ...]:
    return tuple(stream for stream in state.open_streams if stream.id != stream_id)


def _with_error(state: SearchState, action: UnitAction, error: str) -> SearchState:
    return SearchState(
        open_streams=state.open_streams,
        products=state.products,
        unit_sequence=state.unit_sequence + (action,),
        total_abs_duty_W=state.total_abs_duty_W,
        total_theoretical_stages=state.total_theoretical_stages,
        errors=state.errors + (error,),
        process_graph=_process_graph_for_state(state),
        feed_stream=state.feed_stream,
    )


def _record_valid_actions_generated(
    diagnostics: _DiagnosticsAccumulator,
    actions: list[UnitAction],
) -> None:
    diagnostics.n_valid_actions_generated_total += len(actions)
    diagnostics.max_valid_actions_generated_per_call = max(
        diagnostics.max_valid_actions_generated_per_call,
        len(actions),
    )
    diagnostics.valid_actions_generated_by_kind.update(action.kind for action in actions)


def _estimate_relative_volatilities_cached(
    stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
) -> dict[str, float]:
    use_cache = _relative_volatility_cache_enabled(config, relative_volatility_cache)
    cache_key = None
    if use_cache:
        cache_key = _relative_volatility_cache_key(stream, provider)
        cached = relative_volatility_cache.get(cache_key)
        if cached is not None:
            if diagnostics is not None:
                diagnostics.n_relative_volatility_cache_hits += 1
                diagnostics.relative_volatility_cache_saved_estimate_s += (
                    cached.calculation_time_s
                )
            return cached.alphas
        if diagnostics is not None:
            diagnostics.n_relative_volatility_cache_misses += 1

    started_at = time.monotonic()
    try:
        alphas, _, _ = estimate_relative_volatilities(stream, provider)
    finally:
        calculation_time_s = time.monotonic() - started_at
        if diagnostics is not None:
            diagnostics.relative_volatility_calc_time_s += calculation_time_s

    if (
        use_cache
        and cache_key is not None
        and _can_store_relative_volatility_cache_entry(relative_volatility_cache, config)
    ):
        relative_volatility_cache[cache_key] = _CachedRelativeVolatilities(
            alphas=alphas,
            calculation_time_s=calculation_time_s,
        )
    return alphas


def _apply_action_cache_key(
    stream: StreamState,
    action: UnitAction,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
) -> tuple:
    return (
        _provider_signature(provider),
        stream_signature(stream),
        action_signature(action),
        _apply_action_config_signature(action.kind, config),
    )


def _valid_action_cache_key(
    state: SearchState,
    config: MCTSConfig,
    provider: ThermoFlashProvider | None,
    feed_stream: StreamState | None,
) -> tuple:
    return (
        _provider_signature(provider) if provider is not None else None,
        stream_signature(feed_stream) if feed_stream is not None else None,
        state_identity_hash(state),
        _valid_action_config_signature(config),
    )


def _relative_volatility_cache_key(
    stream: StreamState,
    provider: ThermoFlashProvider,
) -> tuple:
    return (
        _provider_signature(provider),
        stream_signature(stream),
    )


def lookup_relative_volatilities(
    stream: StreamState,
    provider: ThermoFlashProvider,
    cache: dict,
) -> dict[str, float] | None:
    """Look up pre-computed relative volatilities from an MCTSResult cache.

    Args:
        stream: Stream whose volatilities to retrieve.
        provider: Provider used during the originating mcts_search call.
        cache: MCTSResult.relative_volatility_cache from a return_tree=True run.

    Returns:
        Dict {compound: alpha} normalised to the default reference compound,
        or None if this stream was not encountered during the search.

    Example:
        alphas = lookup_relative_volatilities(stream, provider, result.relative_volatility_cache)
    """
    if cache is None:
        return None
    entry = cache.get(_relative_volatility_cache_key(stream, provider))
    return entry.alphas if entry is not None else None


def _provider_signature(provider: ThermoFlashProvider) -> tuple:
    return ("thermo_pr", provider.compounds)


def _valid_action_config_signature(config: MCTSConfig) -> tuple:
    return (
        ("objective_mode", config.objective_mode),
        ("separation_score_mode", config.separation_score_mode),
        ("target_component", config.target_component),
        ("target_fraction", config.target_fraction),
        ("product_role", config.product_role),
        ("target_product_temperature_K", config.target_product_temperature_K),
        ("product_temperature_tolerance_K", config.product_temperature_tolerance_K),
        ("require_flash_liquid_product", config.require_flash_liquid_product),
        ("allowed_delta_T_K", config.allowed_delta_T_K),
        ("allowed_compression_ratios", config.allowed_compression_ratios),
        ("allowed_compression_delta_P_Pa", config.allowed_compression_delta_P_Pa),
        ("allowed_pump_pressure_ratios", config.allowed_pump_pressure_ratios),
        ("allowed_pump_delta_P_Pa", config.allowed_pump_delta_P_Pa),
        ("allowed_valve_pressure_ratios", config.allowed_valve_pressure_ratios),
        ("allowed_valve_delta_P_Pa", config.allowed_valve_delta_P_Pa),
        ("hx_target_states", config.hx_target_states),
        ("hx_partial_target_vf", config.hx_partial_target_vf),
        ("pump_target_states", config.pump_target_states),
        ("compressor_target_states", config.compressor_target_states),
        ("compressor_min_inlet_vapor_fraction", config.compressor_min_inlet_vapor_fraction),
        ("valve_target_states", config.valve_target_states),
        ("min_pressure_Pa", config.min_pressure_Pa),
        ("max_pressure_Pa", config.max_pressure_Pa),
        ("min_temperature_K", config.min_temperature_K),
        ("max_temperature_K", config.max_temperature_K),
        ("min_flow_mols", config.min_flow_mols),
        ("max_active_streams_per_state", config.max_active_streams_per_state),
        ("min_stream_priority", config.min_stream_priority),
        ("max_depth", config.max_depth),
        ("max_flash_count_per_path", config.max_flash_count_per_path),
        ("separation_score_tolerance", config.separation_score_tolerance),
        ("min_component_fraction", config.min_component_fraction),
        ("enable_distillation_actions", config.enable_distillation_actions),
        ("distillation_light_key_recoveries", config.distillation_light_key_recoveries),
        ("distillation_heavy_key_recoveries", config.distillation_heavy_key_recoveries),
        ("distillation_reflux_multipliers", config.distillation_reflux_multipliers),
        ("distillation_key_pair_mode", config.distillation_key_pair_mode),
        ("validate_distillation_candidates", config.validate_distillation_candidates),
        ("distillation_min_key_flow_mols", config.distillation_min_key_flow_mols),
        ("distillation_min_alpha_ratio", config.distillation_min_alpha_ratio),
        ("distillation_max_theoretical_stages", config.distillation_max_theoretical_stages),
        ("max_distillation_count_per_path", config.max_distillation_count_per_path),
        ("max_total_distillation_count", config.max_total_distillation_count),
        ("enable_recycle_actions", config.enable_recycle_actions),
        ("max_recycle_count_per_path", config.max_recycle_count_per_path),
        ("recycle_purity_threshold", config.recycle_purity_threshold),
    )


def _apply_action_config_signature(action_kind: str, config: MCTSConfig) -> tuple:
    if action_kind == "compressor":
        return (
            ("compressor_isentropic_efficiency", config.compressor_isentropic_efficiency),
            ("compressor_mechanical_efficiency", config.compressor_mechanical_efficiency),
        )
    if action_kind == "pump":
        return (
            ("pump_isentropic_efficiency", config.pump_isentropic_efficiency),
            ("pump_mechanical_efficiency", config.pump_mechanical_efficiency),
            ("pump_max_inlet_vapor_fraction", config.pump_max_inlet_vapor_fraction),
        )
    if action_kind == "distillation":
        return (
            ("distillation_max_theoretical_stages", config.distillation_max_theoretical_stages),
            ("min_flow_mols", config.min_flow_mols),
            ("distillation_molar_heat_of_vaporization_J_mol", config.distillation_molar_heat_of_vaporization_J_mol),
            ("include_reboiler_duty", config.include_reboiler_duty),
        )
    if action_kind == "flash":
        return (("min_flow_mols", config.min_flow_mols),)
    return ()


def _apply_action_cache_size(
    action_cache: dict[tuple, _CachedActionOutcome] | None,
) -> int:
    return 0 if action_cache is None else len(action_cache)


def _valid_action_cache_size(
    valid_action_cache: dict[tuple, _CachedValidActions] | None,
) -> int:
    return 0 if valid_action_cache is None else len(valid_action_cache)


def _relative_volatility_cache_size(
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None,
) -> int:
    return 0 if relative_volatility_cache is None else len(relative_volatility_cache)


def _action_generation_cache_enabled(config: MCTSConfig) -> bool:
    return config.enable_action_generation_cache


def _valid_action_cache_enabled(
    config: MCTSConfig,
    valid_action_cache: dict[tuple, _CachedValidActions] | None,
) -> bool:
    return config.enable_action_generation_cache and valid_action_cache is not None


def _relative_volatility_cache_enabled(
    config: MCTSConfig,
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities] | None,
) -> bool:
    return (
        config.enable_action_generation_cache
        and relative_volatility_cache is not None
    )


def _can_store_apply_action_cache_entry(
    action_cache: dict[tuple, _CachedActionOutcome],
    config: MCTSConfig,
) -> bool:
    max_entries = config.max_apply_action_cache_entries
    return max_entries is None or len(action_cache) < max_entries


def _can_store_valid_action_cache_entry(
    valid_action_cache: dict[tuple, _CachedValidActions],
    config: MCTSConfig,
) -> bool:
    max_entries = config.max_valid_action_cache_entries
    return max_entries is None or len(valid_action_cache) < max_entries


def _can_store_relative_volatility_cache_entry(
    relative_volatility_cache: dict[tuple, _CachedRelativeVolatilities],
    config: MCTSConfig,
) -> bool:
    max_entries = config.max_relative_volatility_cache_entries
    return max_entries is None or len(relative_volatility_cache) < max_entries


def _distillation_result_cache_enabled(config: MCTSConfig) -> bool:
    return config.enable_apply_action_cache and "distillation" in config.cached_action_kinds


def _distillation_result_cache_size(
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None,
) -> int:
    return 0 if distillation_result_cache is None else len(distillation_result_cache)


def _distillation_result_cache_key(
    stream: StreamState,
    action: UnitAction,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
) -> tuple:
    return (
        _provider_signature(provider),
        stream_signature(stream),
        action_signature(action),
        _apply_action_config_signature("distillation", config),
    )


def _shortcut_distillation_fug_cached(
    stream: StreamState,
    provider: ThermoFlashProvider,
    config: MCTSConfig,
    action: UnitAction,
    relative_volatilities: dict[str, float] | None = None,
    distillation_result_cache: dict[tuple, _CachedDistillationResult] | None = None,
    diagnostics: _DiagnosticsAccumulator | None = None,
) -> ShortcutDistillationResult:
    if distillation_result_cache is None:
        return shortcut_distillation_fug(
            stream,
            provider,
            light_key=action.light_key or "",
            heavy_key=action.heavy_key or "",
            light_key_recovery=float(action.light_key_recovery or 0.0),
            heavy_key_recovery=float(action.heavy_key_recovery or 0.0),
            relative_volatilities=relative_volatilities,
            reflux_ratio_multiplier=float(action.reflux_ratio_multiplier or 0.0),
            max_reflux_ratio=config.distillation_max_reflux_ratio,
        )

    key = _distillation_result_cache_key(stream, action, provider, config)
    cached = distillation_result_cache.get(key)
    if cached is not None:
        if diagnostics is not None:
            diagnostics.n_distillation_result_cache_hits += 1
            diagnostics.distillation_result_cache_saved_estimate_s += (
                cached.calculation_time_s
            )
        return cached.result

    if diagnostics is not None:
        diagnostics.n_distillation_result_cache_misses += 1
    started_at = time.monotonic()
    result = shortcut_distillation_fug(
        stream,
        provider,
        light_key=action.light_key or "",
        heavy_key=action.heavy_key or "",
        light_key_recovery=float(action.light_key_recovery or 0.0),
        heavy_key_recovery=float(action.heavy_key_recovery or 0.0),
        relative_volatilities=relative_volatilities,
        reflux_ratio_multiplier=float(action.reflux_ratio_multiplier or 0.0),
        max_reflux_ratio=config.distillation_max_reflux_ratio,
    )
    calculation_time_s = time.monotonic() - started_at
    if diagnostics is not None:
        diagnostics.distillation_result_calc_time_s += calculation_time_s
    if _can_store_distillation_result_cache_entry(distillation_result_cache, config):
        distillation_result_cache[key] = _CachedDistillationResult(
            result=result,
            calculation_time_s=calculation_time_s,
        )
    return result


def _can_store_distillation_result_cache_entry(
    distillation_result_cache: dict[tuple, _CachedDistillationResult],
    config: MCTSConfig,
) -> bool:
    max_entries = config.max_apply_action_cache_entries
    return max_entries is None or len(distillation_result_cache) < max_entries


def _rebased_distillation_streams(
    result: ShortcutDistillationResult,
    input_stream: StreamState,
    action: UnitAction,
    step: int,
) -> tuple[StreamState, StreamState]:
    if result.distillate_stream is None or result.bottoms_stream is None:
        raise ValueError("Cannot rebase failed distillation result without outlet streams.")
    prefix = _distillation_column_id(input_stream, action, step)
    return (
        _rebased_distillation_stream(
            result.distillate_stream,
            input_stream,
            f"{prefix}_distillate",
            "shortcut_distillation:total_condenser_distillate",
        ),
        _rebased_distillation_stream(
            result.bottoms_stream,
            input_stream,
            f"{prefix}_bottoms",
            "shortcut_distillation:bottoms",
        ),
    )


def _rebased_distillation_stream(
    cached_stream: StreamState,
    input_stream: StreamState,
    stream_id: str,
    history_item: str,
) -> StreamState:
    return StreamState(
        id=stream_id,
        temperature_K=cached_stream.temperature_K,
        pressure_Pa=cached_stream.pressure_Pa,
        molar_flow_mols=cached_stream.molar_flow_mols,
        composition=dict(cached_stream.composition),
        vapor_fraction=cached_stream.vapor_fraction,
        history=input_stream.history + (history_item,),
    )


def _cached_action_outcome_from_transition(
    state: SearchState,
    stream: StreamState,
    next_state: SearchState,
    calculation_time_s: float,
) -> _CachedActionOutcome:
    if len(next_state.errors) > len(state.errors):
        return _CachedActionOutcome(
            error=next_state.errors[-1],
            calculation_time_s=calculation_time_s,
        )
    output_streams = _transition_output_streams(state, stream.id, next_state)
    return _CachedActionOutcome(
        output_streams=tuple(output_streams),
        total_abs_duty_delta_W=next_state.total_abs_duty_W - state.total_abs_duty_W,
        calculation_time_s=calculation_time_s,
    )


def _transition_output_streams(
    state: SearchState,
    stream_id: str,
    next_state: SearchState,
) -> tuple[StreamState, ...]:
    input_index = next(
        index for index, stream in enumerate(state.open_streams) if stream.id == stream_id
    )
    replacement_count = len(next_state.open_streams) - (len(state.open_streams) - 1)
    if replacement_count <= 0:
        return ()
    return tuple(next_state.open_streams[input_index : input_index + replacement_count])


def _state_from_cached_action_outcome(
    state: SearchState,
    stream: StreamState,
    action: UnitAction,
    outcome: _CachedActionOutcome,
) -> SearchState:
    if outcome.error is not None:
        return _with_error(state, action, outcome.error)
    step = len(state.unit_sequence) + 1
    output_streams = [
        _rebased_cached_output_stream(stream, action, cached_stream, step)
        for cached_stream in outcome.output_streams
    ]
    output_streams_with_roles = tuple(
        (output_stream, _cached_output_role(action, output_stream))
        for output_stream in output_streams
    )
    return _state_with_unit_outputs(
        state,
        stream,
        action,
        output_streams_with_roles,
        total_abs_duty_delta_W=outcome.total_abs_duty_delta_W,
    )


def _rebased_cached_output_stream(
    input_stream: StreamState,
    action: UnitAction,
    cached_stream: StreamState,
    step: int,
) -> StreamState:
    history_item = cached_stream.history[-1] if cached_stream.history else action.kind
    return StreamState(
        id=_cached_output_stream_id(input_stream, action, history_item, step),
        temperature_K=cached_stream.temperature_K,
        pressure_Pa=cached_stream.pressure_Pa,
        molar_flow_mols=cached_stream.molar_flow_mols,
        composition=dict(cached_stream.composition),
        vapor_fraction=cached_stream.vapor_fraction,
        history=input_stream.history + (history_item,),
    )


def _cached_output_role(action: UnitAction, cached_stream: StreamState) -> str:
    history_item = cached_stream.history[-1] if cached_stream.history else ""
    if action.kind == "flash":
        if history_item == "flash:vapor":
            return "vapor"
        if history_item == "flash:liquid":
            return "liquid"
    if action.kind == "distillation":
        if history_item == "shortcut_distillation:total_condenser_distillate":
            return "distillate"
        if history_item == "shortcut_distillation:bottoms":
            return "bottoms"
    return "out"


def _cached_output_stream_id(
    input_stream: StreamState,
    action: UnitAction,
    history_item: str,
    step: int,
) -> str:
    if action.kind == "hx":
        return _hx_stream_id(input_stream, action, step)
    if action.kind == "compressor":
        return _compressor_stream_id(input_stream, action, step)
    if action.kind == "pump":
        return _pump_stream_id(input_stream, action, step)
    if action.kind == "valve":
        return _valve_stream_id(input_stream, action, step)
    if action.kind == "flash":
        suffix = "vapor" if history_item == "flash:vapor" else "liquid"
        return f"{input_stream.id}_{suffix}"
    if action.kind == "distillation":
        prefix = _distillation_column_id(input_stream, action, step)
        if history_item == "shortcut_distillation:total_condenser_distillate":
            return f"{prefix}_distillate"
        return f"{prefix}_bottoms"
    return f"{input_stream.id}_{action.kind}_{step}"


def _hx_stream_id(stream: StreamState, action: UnitAction, step: int) -> str:
    delta = action.delta_T_K or 0.0
    sign = "p" if delta >= 0 else "m"
    return f"{stream.id}_hx_{sign}{abs(delta):g}_{step}"


def _compressor_stream_id(stream: StreamState, action: UnitAction, step: int) -> str:
    if action.pressure_ratio is not None:
        return f"{stream.id}_comp_r{action.pressure_ratio:g}_{step}"
    return f"{stream.id}_comp_dP{action.delta_P_Pa:g}_{step}"


def _pump_stream_id(stream: StreamState, action: UnitAction, step: int) -> str:
    if action.pressure_ratio is not None:
        return f"{stream.id}_pump_r{action.pressure_ratio:g}_{step}"
    return f"{stream.id}_pump_dP{action.delta_P_Pa:g}_{step}"


def _valve_stream_id(stream: StreamState, action: UnitAction, step: int) -> str:
    if action.pressure_ratio is not None:
        return f"{stream.id}_valve_r{action.pressure_ratio:g}_{step}"
    return f"{stream.id}_valve_dP{action.delta_P_Pa:g}_{step}"


def _distillation_column_id(stream: StreamState, action: UnitAction, step: int) -> str:
    return f"{stream.id}_dist_{action.light_key}_over_{action.heavy_key}_{step}"


def _flash_count(stream: StreamState) -> int:
    return sum(1 for item in stream.history if item in {"flash:vapor", "flash:liquid"})


def _recycle_count(stream: StreamState) -> int:
    return sum(1 for item in stream.history if item == "recycle:feed_merge")


def _recycle_stream_id(stream: StreamState, step: int) -> str:
    return f"{stream.id}_recycle_feed_{step}"


def _distillation_count(stream: StreamState) -> int:
    return sum(
        1
        for item in stream.history
        if item
        in {
            "shortcut_distillation:total_condenser_distillate",
            "shortcut_distillation:bottoms",
        }
    )


def _stream_target_error(stream: StreamState | None, config: MCTSConfig) -> float:
    if stream is None or config.target_component not in stream.composition:
        return math.inf
    return abs(stream.composition[config.target_component] - config.target_fraction)
