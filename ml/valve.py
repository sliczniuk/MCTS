"""Thermo-based pressure-reduction valve model for flowsheet search."""

from __future__ import annotations

from typing import Any

from .flash import ThermoFlashProvider
from .types import StreamState, ValveResult


_COMPOSITION_TOL = 1e-12
_PHASE_TOL = 1e-9


def valve_stream(
    stream: StreamState,
    provider: ThermoFlashProvider,
    outlet_pressure_Pa: float | None = None,
    pressure_ratio: float | None = None,
    delta_P_Pa: float | None = None,
    outlet_stream_id: str | None = None,
) -> ValveResult:
    """Reduce one stream pressure through an isenthalpic valve.

    The model flashes the inlet at T/P, keeps the inlet molar enthalpy fixed,
    and flashes the outlet at H/P. Composition and molar flow are unchanged.
    For valves, delta_P_Pa is a positive pressure drop: P_in - P_out.

    Args:
        stream: Inlet stream state.
        provider: ThermoFlashProvider from build_pr_flasher().
        outlet_pressure_Pa: Required outlet pressure [Pa]. Mutually exclusive
            with pressure_ratio and delta_P_Pa.
        pressure_ratio: Required P_out/P_in, with 0 < ratio < 1. Mutually
            exclusive with outlet_pressure_Pa and delta_P_Pa.
        delta_P_Pa: Required pressure drop P_in - P_out [Pa]. Mutually
            exclusive with outlet_pressure_Pa and pressure_ratio.
        outlet_stream_id: Optional id for the outlet stream.

    Returns:
        ValveResult with outlet stream and outlet phase information.

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
        result = valve_stream(feed, provider, pressure_ratio=0.5)
    """
    target_error = _target_pressure_error(outlet_pressure_Pa, pressure_ratio, delta_P_Pa)
    if target_error:
        return _failed(stream.id, target_error)

    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        return _failed(stream.id, validation_error)

    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        return _failed(stream.id, zs_or_error)
    zs = zs_or_error

    outlet_pressure = _outlet_pressure(stream, outlet_pressure_Pa, pressure_ratio, delta_P_Pa)
    if outlet_pressure >= stream.pressure_Pa:
        return _failed(
            stream.id,
            "Valve outlet pressure must be lower than inlet pressure; "
            f"got P_in={stream.pressure_Pa} Pa and P_out={outlet_pressure} Pa.",
        )
    if outlet_pressure <= 0:
        return _failed(
            stream.id,
            f"Valve outlet pressure must be positive, got {outlet_pressure} Pa.",
        )

    try:
        inlet = provider.flasher.flash(
            T=float(stream.temperature_K),
            P=float(stream.pressure_Pa),
            zs=zs,
        )
        inlet_h = _enthalpy_Jmol(inlet)
        if inlet_h is None:
            return _failed(
                stream.id,
                f"thermo did not report inlet enthalpy for stream '{stream.id}'.",
            )
        outlet = provider.flasher.flash(
            H=inlet_h,
            P=outlet_pressure,
            zs=zs,
        )
    except Exception as exc:
        return _failed(
            stream.id,
            f"thermo valve flash failed for stream '{stream.id}': {exc}",
        )

    outlet_t = _safe_float(getattr(outlet, "T", None))
    if outlet_t is None or outlet_t <= 0:
        return _failed(
            stream.id,
            f"thermo did not report a valid valve outlet temperature for '{stream.id}'.",
        )

    outlet_h = _enthalpy_Jmol(outlet)
    if outlet_h is None:
        return _failed(
            stream.id,
            f"thermo did not report outlet enthalpy for stream '{stream.id}'.",
        )

    vf = _safe_float(getattr(outlet, "VF", None))
    if vf is None:
        return _failed(
            stream.id,
            f"thermo valve outlet for stream '{stream.id}' did not report VF.",
        )
    vf = min(1.0, max(0.0, vf))
    lf = 1.0 - vf

    phase_count = _safe_int(getattr(outlet, "phase_count", None))
    phase_compositions = _phase_compositions(outlet, provider.compounds)

    outlet_stream = StreamState(
        id=outlet_stream_id or f"{stream.id}_valve",
        temperature_K=outlet_t,
        pressure_Pa=outlet_pressure,
        molar_flow_mols=stream.molar_flow_mols,
        composition=dict(zip(provider.compounds, zs)),
        vapor_fraction=vf,
        history=stream.history + ("valve",),
    )

    return ValveResult(
        success=True,
        inlet_stream_id=stream.id,
        outlet_stream=outlet_stream,
        outlet_pressure_Pa=outlet_pressure,
        delta_P_Pa=stream.pressure_Pa - outlet_pressure,
        pressure_ratio=outlet_pressure / stream.pressure_Pa,
        inlet_enthalpy_Jmol=inlet_h,
        outlet_enthalpy_Jmol=outlet_h,
        phase_state=_phase_state(
            vf,
            getattr(outlet, "gas", None),
            getattr(outlet, "liquid0", None),
            phase_count,
        ),
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
    if pressure_ratio is not None and not 0.0 < pressure_ratio < 1.0:
        return (
            "pressure_ratio must satisfy 0 < ratio < 1 for pressure reduction, "
            f"got {pressure_ratio}."
        )
    if delta_P_Pa is not None and delta_P_Pa <= 0:
        return f"delta_P_Pa must be a positive pressure drop, got {delta_P_Pa}."
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
    return float(stream.pressure_Pa) - float(delta_P_Pa)


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


def _failed(stream_id: str, error_message: str) -> ValveResult:
    return ValveResult(
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
