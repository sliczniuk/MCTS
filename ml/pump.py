"""Thermo-based single-stream liquid pump model for flowsheet search."""

from __future__ import annotations

from typing import Any

from .flash import ThermoFlashProvider
from .types import PumpResult, StreamState


_COMPOSITION_TOL = 1e-12
_PHASE_TOL = 1e-9


def pump_stream(
    stream: StreamState,
    provider: ThermoFlashProvider,
    outlet_pressure_Pa: float | None = None,
    pressure_ratio: float | None = None,
    delta_P_Pa: float | None = None,
    isentropic_efficiency: float = 0.75,
    mechanical_efficiency: float = 1.0,
    max_inlet_vapor_fraction: float = 1e-6,
    outlet_stream_id: str | None = None,
) -> PumpResult:
    """Pump one liquid stream using thermo isentropic and PH flashes.

    The model follows the same thermodynamic structure as the compressor:
    flash the inlet at T/P, reject vapor-rich inlet states, flash the ideal
    outlet at inlet entropy and discharge pressure, convert the ideal enthalpy
    rise to an actual enthalpy rise with the isentropic efficiency, then flash
    the actual outlet at H/P. Composition and molar flow are unchanged.

    Args:
        stream: Inlet stream state.
        provider: ThermoFlashProvider from build_pr_flasher().
        outlet_pressure_Pa: Required discharge pressure [Pa]. Mutually
            exclusive with pressure_ratio and delta_P_Pa.
        pressure_ratio: Required P_out/P_in. Mutually exclusive with
            outlet_pressure_Pa and delta_P_Pa.
        delta_P_Pa: Required pressure increase P_out - P_in [Pa]. Mutually
            exclusive with outlet_pressure_Pa and pressure_ratio.
        isentropic_efficiency: Pump isentropic efficiency, 0 < eta <= 1.
        mechanical_efficiency: Shaft-to-fluid efficiency, 0 < eta <= 1.
        max_inlet_vapor_fraction: Maximum inlet VF allowed for pump service.
        outlet_stream_id: Optional id for the outlet stream.

    Returns:
        PumpResult with outlet stream, outlet state, and power demand.

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
        result = pump_stream(liquid, provider, pressure_ratio=2.0)
    """
    target_error = _target_pressure_error(outlet_pressure_Pa, pressure_ratio, delta_P_Pa)
    if target_error:
        return _failed(stream.id, target_error)

    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        return _failed(stream.id, validation_error)

    efficiency_error = _validate_efficiencies(
        isentropic_efficiency,
        mechanical_efficiency,
        max_inlet_vapor_fraction,
    )
    if efficiency_error:
        return _failed(stream.id, efficiency_error)

    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        return _failed(stream.id, zs_or_error)
    zs = zs_or_error

    outlet_pressure = _outlet_pressure(stream, outlet_pressure_Pa, pressure_ratio, delta_P_Pa)
    if outlet_pressure <= stream.pressure_Pa:
        return _failed(
            stream.id,
            "Pump outlet pressure must be greater than inlet pressure; "
            f"got P_in={stream.pressure_Pa} Pa and P_out={outlet_pressure} Pa.",
        )

    try:
        inlet = provider.flasher.flash(
            T=float(stream.temperature_K),
            P=float(stream.pressure_Pa),
            zs=zs,
        )
        inlet_vf = _safe_float(getattr(inlet, "VF", None))
        if inlet_vf is None:
            return _failed(
                stream.id,
                f"thermo pump inlet for stream '{stream.id}' did not report VF.",
            )
        inlet_vf = min(1.0, max(0.0, inlet_vf))
        if inlet_vf > max_inlet_vapor_fraction:
            return _failed(
                stream.id,
                "Pump requires a liquid or near-liquid inlet; "
                f"stream '{stream.id}' has VF={inlet_vf:g}, above "
                f"max_inlet_vapor_fraction={max_inlet_vapor_fraction:g}. "
                "Use a compressor for vapor streams.",
            )

        inlet_h = _enthalpy_Jmol(inlet)
        inlet_s = _entropy_JmolK(inlet)
        if inlet_h is None or inlet_s is None:
            return _failed(
                stream.id,
                f"thermo did not report inlet enthalpy/entropy for stream '{stream.id}'.",
            )

        ideal_outlet = provider.flasher.flash(
            S=inlet_s,
            P=outlet_pressure,
            zs=zs,
        )
        ideal_h = _enthalpy_Jmol(ideal_outlet)
        if ideal_h is None:
            return _failed(
                stream.id,
                f"thermo did not report isentropic outlet enthalpy for stream '{stream.id}'.",
            )

        ideal_delta_h = ideal_h - inlet_h
        if ideal_delta_h < 0:
            return _failed(
                stream.id,
                "Isentropic pump enthalpy rise was negative; check inlet "
                "state and outlet pressure.",
            )

        actual_h = inlet_h + ideal_delta_h / float(isentropic_efficiency)
        actual_outlet = provider.flasher.flash(
            H=actual_h,
            P=outlet_pressure,
            zs=zs,
        )
    except Exception as exc:
        return _failed(
            stream.id,
            f"thermo pump flash failed for stream '{stream.id}': {exc}",
        )

    actual_t = _safe_float(getattr(actual_outlet, "T", None))
    if actual_t is None or actual_t <= 0:
        return _failed(
            stream.id,
            f"thermo did not report a valid pump outlet temperature for '{stream.id}'.",
        )

    actual_h_reported = _enthalpy_Jmol(actual_outlet)
    if actual_h_reported is None:
        return _failed(
            stream.id,
            f"thermo did not report actual outlet enthalpy for stream '{stream.id}'.",
        )

    vf = _safe_float(getattr(actual_outlet, "VF", None))
    if vf is None:
        return _failed(
            stream.id,
            f"thermo pump outlet for stream '{stream.id}' did not report VF.",
        )
    vf = min(1.0, max(0.0, vf))
    lf = 1.0 - vf

    phase_count = _safe_int(getattr(actual_outlet, "phase_count", None))
    phase_compositions = _phase_compositions(actual_outlet, provider.compounds)
    ideal_t = _safe_float(getattr(ideal_outlet, "T", None))

    outlet_stream = StreamState(
        id=outlet_stream_id or f"{stream.id}_pump",
        temperature_K=actual_t,
        pressure_Pa=outlet_pressure,
        molar_flow_mols=stream.molar_flow_mols,
        composition=dict(zip(provider.compounds, zs)),
        vapor_fraction=vf,
        history=stream.history + ("pump",),
    )

    fluid_power = stream.molar_flow_mols * (actual_h_reported - inlet_h)
    shaft_power = fluid_power / float(mechanical_efficiency)

    return PumpResult(
        success=True,
        inlet_stream_id=stream.id,
        outlet_stream=outlet_stream,
        outlet_pressure_Pa=outlet_pressure,
        delta_P_Pa=outlet_pressure - stream.pressure_Pa,
        pressure_ratio=outlet_pressure / stream.pressure_Pa,
        isentropic_efficiency=float(isentropic_efficiency),
        mechanical_efficiency=float(mechanical_efficiency),
        max_inlet_vapor_fraction=float(max_inlet_vapor_fraction),
        fluid_power_W=fluid_power,
        shaft_power_W=shaft_power,
        inlet_enthalpy_Jmol=inlet_h,
        inlet_entropy_JmolK=inlet_s,
        ideal_outlet_enthalpy_Jmol=ideal_h,
        actual_outlet_enthalpy_Jmol=actual_h_reported,
        ideal_outlet_temperature_K=ideal_t,
        phase_state=_phase_state(
            vf,
            getattr(actual_outlet, "gas", None),
            getattr(actual_outlet, "liquid0", None),
            phase_count,
        ),
        inlet_vapor_fraction=inlet_vf,
        vapor_fraction=vf,
        liquid_fraction=lf,
        phase_compositions=phase_compositions,
        raw_phase_count=phase_count,
    )


def _target_pressure_error(
    outlet_pressure_Pa: float | None,
    pressure_ratio: float | None,
    delta_P_Pa: float | None,
) -> str | None:
    specs = [
        outlet_pressure_Pa is not None,
        pressure_ratio is not None,
        delta_P_Pa is not None,
    ]
    if sum(specs) != 1:
        return "Provide exactly one of outlet_pressure_Pa, pressure_ratio, or delta_P_Pa."
    if outlet_pressure_Pa is not None and outlet_pressure_Pa <= 0:
        return f"outlet_pressure_Pa must be positive, got {outlet_pressure_Pa}."
    if pressure_ratio is not None and pressure_ratio <= 1.0:
        return f"pressure_ratio must be greater than 1 for pumping, got {pressure_ratio}."
    if delta_P_Pa is not None and delta_P_Pa <= 0:
        return f"delta_P_Pa must be positive for pumping, got {delta_P_Pa}."
    return None


def _outlet_pressure(
    stream: StreamState,
    outlet_pressure_Pa: float | None,
    pressure_ratio: float | None,
    delta_P_Pa: float | None,
) -> float:
    if outlet_pressure_Pa is not None:
        return float(outlet_pressure_Pa)
    if pressure_ratio is not None:
        return float(stream.pressure_Pa) * float(pressure_ratio)
    return float(stream.pressure_Pa) + float(delta_P_Pa)


def _validate_stream_conditions(stream: StreamState) -> str | None:
    if not stream.id:
        return "stream id must not be blank."
    if stream.temperature_K <= 0:
        return (
            f"Stream '{stream.id}' has non-positive temperature_K="
            f"{stream.temperature_K}; provide temperature in K."
        )
    if stream.pressure_Pa <= 0:
        return (
            f"Stream '{stream.id}' has non-positive pressure_Pa="
            f"{stream.pressure_Pa}; provide pressure in Pa."
        )
    if stream.molar_flow_mols <= 0:
        return (
            f"Stream '{stream.id}' has non-positive molar_flow_mols="
            f"{stream.molar_flow_mols}; provide molar flow in mol/s."
        )
    return None


def _validate_efficiencies(
    isentropic_efficiency: float,
    mechanical_efficiency: float,
    max_inlet_vapor_fraction: float,
) -> str | None:
    if not 0.0 < isentropic_efficiency <= 1.0:
        return (
            "isentropic_efficiency must satisfy 0 < eta <= 1, got "
            f"{isentropic_efficiency}."
        )
    if not 0.0 < mechanical_efficiency <= 1.0:
        return (
            "mechanical_efficiency must satisfy 0 < eta <= 1, got "
            f"{mechanical_efficiency}."
        )
    if not 0.0 <= max_inlet_vapor_fraction <= 1.0:
        return (
            "max_inlet_vapor_fraction must satisfy 0 <= VF <= 1, got "
            f"{max_inlet_vapor_fraction}."
        )
    return None


def _composition_vector(
    stream: StreamState,
    compounds: tuple[str, ...],
) -> list[float] | str:
    unknown = sorted(set(stream.composition) - set(compounds))
    if unknown:
        return (
            f"Stream '{stream.id}' composition contains components not in the "
            f"flasher: {unknown}. Flasher compounds: {list(compounds)}."
        )

    values = []
    for compound in compounds:
        value = float(stream.composition.get(compound, 0.0))
        if value < 0:
            return (
                f"Stream '{stream.id}' composition for '{compound}' is negative "
                f"({value}); mole fractions must be non-negative."
            )
        values.append(value)

    total = sum(values)
    if total <= _COMPOSITION_TOL:
        return (
            f"Stream '{stream.id}' composition sum is zero; provide at least one "
            "positive mole fraction."
        )

    return [value / total for value in values]


def _phase_compositions(
    flash: Any,
    compounds: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    phases = {}
    gas = getattr(flash, "gas", None)
    liquid = getattr(flash, "liquid0", None)

    if gas is not None:
        phases["vapor"] = dict(zip(compounds, _normalise_list([float(v) for v in gas.zs])))
    if liquid is not None:
        phases["liquid"] = dict(zip(compounds, _normalise_list([float(v) for v in liquid.zs])))
    return phases


def _phase_state(vf: float, gas_phase: Any, liquid_phase: Any, phase_count: int | None) -> str:
    if phase_count == 2 and gas_phase is not None and liquid_phase is not None:
        return "two_phase"
    if vf >= 1.0 - _PHASE_TOL or (gas_phase is not None and liquid_phase is None):
        return "vapor"
    if vf <= _PHASE_TOL or (liquid_phase is not None and gas_phase is None):
        return "liquid"
    return "unknown"


def _normalise_list(values: list[float]) -> list[float]:
    values = [max(0.0, value) for value in values]
    total = sum(values)
    if total <= _COMPOSITION_TOL:
        raise ValueError("phase composition sum is zero")
    return [value / total for value in values]


def _enthalpy_Jmol(flash: Any) -> float | None:
    try:
        return float(flash.H())
    except Exception:
        return None


def _entropy_JmolK(flash: Any) -> float | None:
    try:
        return float(flash.S())
    except Exception:
        return None


def _failed(stream_id: str, error_message: str) -> PumpResult:
    return PumpResult(
        success=False,
        inlet_stream_id=stream_id,
        error_message=error_message,
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
