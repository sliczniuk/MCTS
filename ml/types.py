"""Shared data shapes for simplified process models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamState:
    """State of a material stream used by search-side unit models.

    Args:
        id: Stable stream identifier.
        temperature_K: Stream temperature [K].
        pressure_Pa: Stream pressure [Pa].
        molar_flow_mols: Total molar flow [mol/s].
        composition: Mole-fraction mapping keyed by component identifier.
        history: Unit/action identifiers that produced this stream.

    Returns:
        Immutable stream state for MCTS unit calculations.

    Example:
        stream = StreamState(
            id="Feed",
            temperature_K=110.0,
            pressure_Pa=100000.0,
            molar_flow_mols=1.0,
            composition={"methane": 0.965, "ethane": 0.018, "nitrogen": 0.017},
        )
    """

    id: str
    temperature_K: float
    pressure_Pa: float
    molar_flow_mols: float
    composition: dict[str, float]
    vapor_fraction: float | None = None
    history: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FlashResult:
    """Result from a simplified flash-drum calculation.

    Args:
        success: True when the flash calculation completed.
        inlet_stream_id: Identifier of the feed stream.
        phase_state: "two_phase", "vapor", "liquid", or "unknown".
        vapor_fraction: Vapor fraction of the inlet molar flow.
        liquid_fraction: Liquid fraction of the inlet molar flow.
        vapor_stream: Vapor outlet stream when a split exists.
        liquid_stream: Liquid outlet stream when a split exists.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.
        raw_phase_count: Phase count reported by thermo when available.

    Returns:
        Plain Python result object for MCTS scoring and graph expansion.

    Example:
        result = flash_split(feed, provider)
        if result.success and result.phase_state == "two_phase":
            open_streams.extend([result.vapor_stream, result.liquid_stream])
    """

    success: bool
    inlet_stream_id: str
    phase_state: str
    vapor_fraction: float | None = None
    liquid_fraction: float | None = None
    vapor_stream: StreamState | None = None
    liquid_stream: StreamState | None = None
    duty_W: float | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    raw_phase_count: int | None = None


@dataclass(frozen=True)
class HXResult:
    """Result from a simplified single-stream heat exchanger calculation.

    Args:
        success: True when inlet and outlet thermo flashes completed.
        inlet_stream_id: Identifier of the inlet stream.
        outlet_stream: Outlet stream at the requested temperature and inlet pressure.
        duty_W: Heat added to the stream [W]. Negative means cooling.
        inlet_enthalpy_Jmol: Inlet molar enthalpy [J/mol].
        outlet_enthalpy_Jmol: Outlet molar enthalpy [J/mol].
        phase_state: Outlet phase state: "two_phase", "vapor", "liquid", or "unknown".
        vapor_fraction: Outlet vapor fraction.
        liquid_fraction: Outlet liquid fraction.
        phase_compositions: Outlet phase compositions keyed by phase name.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.
        raw_phase_count: Outlet phase count reported by thermo when available.

    Returns:
        Plain Python result object for MCTS heat-conditioning actions.

    Example:
        result = heat_stream(feed, provider, outlet_temperature_K=350.0)
        if result.success:
            next_stream = result.outlet_stream
    """

    success: bool
    inlet_stream_id: str
    outlet_stream: StreamState | None = None
    duty_W: float | None = None
    inlet_enthalpy_Jmol: float | None = None
    outlet_enthalpy_Jmol: float | None = None
    phase_state: str = "unknown"
    vapor_fraction: float | None = None
    liquid_fraction: float | None = None
    phase_compositions: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    raw_phase_count: int | None = None


@dataclass(frozen=True)
class CompressionResult:
    """Result from a simplified compressor calculation.

    Args:
        success: True when inlet, isentropic outlet, and actual outlet flashes completed.
        inlet_stream_id: Identifier of the inlet stream.
        outlet_stream: Outlet stream at calculated discharge temperature and pressure.
        outlet_pressure_Pa: Compressor discharge pressure [Pa].
        delta_P_Pa: Compressor pressure increase [Pa].
        pressure_ratio: Outlet pressure divided by inlet pressure.
        isentropic_efficiency: Compressor isentropic efficiency.
        mechanical_efficiency: Shaft-to-fluid mechanical efficiency.
        fluid_power_W: Power added to the fluid [W].
        shaft_power_W: Shaft power required [W].
        inlet_enthalpy_Jmol: Inlet molar enthalpy [J/mol].
        inlet_entropy_JmolK: Inlet molar entropy [J/mol/K].
        ideal_outlet_enthalpy_Jmol: Isentropic outlet molar enthalpy [J/mol].
        actual_outlet_enthalpy_Jmol: Actual outlet molar enthalpy [J/mol].
        ideal_outlet_temperature_K: Isentropic outlet temperature [K].
        phase_state: Actual outlet phase state.
        vapor_fraction: Actual outlet vapor fraction.
        liquid_fraction: Actual outlet liquid fraction.
        phase_compositions: Actual outlet phase compositions keyed by phase name.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.
        raw_phase_count: Actual outlet phase count reported by thermo when available.

    Returns:
        Plain Python result object for MCTS pressure-conditioning actions.

    Example:
        result = compress_stream(feed, provider, pressure_ratio=2.0)
        if result.success:
            outlet = result.outlet_stream
    """

    success: bool
    inlet_stream_id: str
    outlet_stream: StreamState | None = None
    outlet_pressure_Pa: float | None = None
    delta_P_Pa: float | None = None
    pressure_ratio: float | None = None
    isentropic_efficiency: float | None = None
    mechanical_efficiency: float | None = None
    fluid_power_W: float | None = None
    shaft_power_W: float | None = None
    inlet_enthalpy_Jmol: float | None = None
    inlet_entropy_JmolK: float | None = None
    ideal_outlet_enthalpy_Jmol: float | None = None
    actual_outlet_enthalpy_Jmol: float | None = None
    ideal_outlet_temperature_K: float | None = None
    phase_state: str = "unknown"
    vapor_fraction: float | None = None
    liquid_fraction: float | None = None
    phase_compositions: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    raw_phase_count: int | None = None


@dataclass(frozen=True)
class PumpResult:
    """Result from a simplified liquid pump calculation.

    Args:
        success: True when inlet, isentropic outlet, and actual outlet flashes completed.
        inlet_stream_id: Identifier of the inlet stream.
        outlet_stream: Outlet stream at calculated discharge temperature and pressure.
        outlet_pressure_Pa: Pump discharge pressure [Pa].
        delta_P_Pa: Pump pressure increase [Pa].
        pressure_ratio: Outlet pressure divided by inlet pressure.
        isentropic_efficiency: Pump isentropic efficiency.
        mechanical_efficiency: Shaft-to-fluid mechanical efficiency.
        max_inlet_vapor_fraction: Maximum inlet vapor fraction accepted by the pump.
        fluid_power_W: Power added to the fluid [W].
        shaft_power_W: Shaft power required [W].
        inlet_enthalpy_Jmol: Inlet molar enthalpy [J/mol].
        inlet_entropy_JmolK: Inlet molar entropy [J/mol/K].
        ideal_outlet_enthalpy_Jmol: Isentropic outlet molar enthalpy [J/mol].
        actual_outlet_enthalpy_Jmol: Actual outlet molar enthalpy [J/mol].
        ideal_outlet_temperature_K: Isentropic outlet temperature [K].
        phase_state: Actual outlet phase state.
        inlet_vapor_fraction: Inlet vapor fraction.
        vapor_fraction: Actual outlet vapor fraction.
        liquid_fraction: Actual outlet liquid fraction.
        phase_compositions: Actual outlet phase compositions keyed by phase name.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.
        raw_phase_count: Actual outlet phase count reported by thermo when available.

    Returns:
        Plain Python result object for MCTS liquid pressure-conditioning actions.

    Example:
        result = pump_stream(liquid, provider, pressure_ratio=2.0)
        if result.success:
            outlet = result.outlet_stream
    """

    success: bool
    inlet_stream_id: str
    outlet_stream: StreamState | None = None
    outlet_pressure_Pa: float | None = None
    delta_P_Pa: float | None = None
    pressure_ratio: float | None = None
    isentropic_efficiency: float | None = None
    mechanical_efficiency: float | None = None
    max_inlet_vapor_fraction: float | None = None
    fluid_power_W: float | None = None
    shaft_power_W: float | None = None
    inlet_enthalpy_Jmol: float | None = None
    inlet_entropy_JmolK: float | None = None
    ideal_outlet_enthalpy_Jmol: float | None = None
    actual_outlet_enthalpy_Jmol: float | None = None
    ideal_outlet_temperature_K: float | None = None
    phase_state: str = "unknown"
    inlet_vapor_fraction: float | None = None
    vapor_fraction: float | None = None
    liquid_fraction: float | None = None
    phase_compositions: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    raw_phase_count: int | None = None


@dataclass(frozen=True)
class ValveResult:
    """Result from a simplified pressure-reduction valve calculation.

    Args:
        success: True when inlet and outlet flashes completed.
        inlet_stream_id: Identifier of the inlet stream.
        outlet_stream: Outlet stream at calculated outlet temperature and pressure.
        outlet_pressure_Pa: Valve outlet pressure [Pa].
        delta_P_Pa: Positive pressure drop P_in - P_out [Pa].
        pressure_ratio: Outlet pressure divided by inlet pressure.
        inlet_enthalpy_Jmol: Inlet molar enthalpy [J/mol].
        outlet_enthalpy_Jmol: Outlet molar enthalpy [J/mol].
        phase_state: Outlet phase state.
        vapor_fraction: Outlet vapor fraction.
        liquid_fraction: Outlet liquid fraction.
        phase_compositions: Outlet phase compositions keyed by phase name.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.
        raw_phase_count: Outlet phase count reported by thermo when available.

    Returns:
        Plain Python result object for MCTS pressure-reduction actions.

    Example:
        result = valve_stream(feed, provider, pressure_ratio=0.5)
        if result.success:
            outlet = result.outlet_stream
    """

    success: bool
    inlet_stream_id: str
    outlet_stream: StreamState | None = None
    outlet_pressure_Pa: float | None = None
    delta_P_Pa: float | None = None
    pressure_ratio: float | None = None
    inlet_enthalpy_Jmol: float | None = None
    outlet_enthalpy_Jmol: float | None = None
    phase_state: str = "unknown"
    vapor_fraction: float | None = None
    liquid_fraction: float | None = None
    phase_compositions: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    raw_phase_count: int | None = None


@dataclass(frozen=True)
class ShortcutDistillationResult:
    """Result from a Fenske-Underwood-Gilliland shortcut column.

    Args:
        success: True when material split and FUG estimates completed.
        inlet_stream_id: Identifier of the feed stream.
        distillate_stream: Overhead product stream.
        bottoms_stream: Bottoms product stream.
        light_key: Light key component identifier.
        heavy_key: Heavy key component identifier.
        pressure_Pa: Column pressure [Pa].
        component_recoveries: Component recoveries to distillate.
        relative_volatilities: Volatilities normalized to heavy_key = 1.
        k_values: K-values used to estimate relative volatilities when available.
        minimum_stages: Fenske minimum theoretical stages.
        underwood_theta: Underwood root between heavy and light keys.
        minimum_reflux_ratio: Underwood minimum reflux ratio.
        reflux_ratio: Actual reflux ratio used for Gilliland.
        theoretical_stages: Gilliland theoretical stages estimate.
        feed_quality: Underwood feed quality q computed from the inlet flash.
        reflux_ratio_multiplier: Multiplier used when reflux_ratio is not provided.
        warnings: Non-fatal issues encountered during calculation.
        error_message: Actionable error when success is False.

    Returns:
        Plain Python result object for shortcut distillation actions.
    """

    success: bool
    inlet_stream_id: str
    distillate_stream: StreamState | None = None
    bottoms_stream: StreamState | None = None
    light_key: str | None = None
    heavy_key: str | None = None
    pressure_Pa: float | None = None
    component_recoveries: dict[str, float] = field(default_factory=dict)
    relative_volatilities: dict[str, float] = field(default_factory=dict)
    k_values: dict[str, float] = field(default_factory=dict)
    minimum_stages: float | None = None
    underwood_theta: float | None = None
    minimum_reflux_ratio: float | None = None
    reflux_ratio: float | None = None
    theoretical_stages: float | None = None
    feed_quality: float | None = None
    reflux_ratio_multiplier: float | None = None
    condenser_duty_W: float | None = None
    reboiler_duty_W: float | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
