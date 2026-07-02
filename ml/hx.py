"""Thermo-based single-stream heat exchanger model for flowsheet search."""

from __future__ import annotations

from typing import Any

from .flash import ThermoFlashProvider
from .types import HXResult, StreamState


_COMPOSITION_TOL = 1e-12
_PHASE_TOL = 1e-9


def heat_stream(
    stream: StreamState,
    provider: ThermoFlashProvider,
    outlet_temperature_K: float | None = None,
    delta_T_K: float | None = None,
    outlet_stream_id: str | None = None,
) -> HXResult:
    """Heat or cool one stream at constant pressure using thermo enthalpy.

    This model represents the search-side behaviour of a heater, cooler, or
    single-stream HX duty. It does not pair hot/cold streams and does not change
    pressure or total composition.

    Args:
        stream: Inlet stream state.
        provider: ThermoFlashProvider from build_pr_flasher().
        outlet_temperature_K: Required outlet temperature [K].
        delta_T_K: Temperature change [K]. Mutually exclusive with
            outlet_temperature_K.
        outlet_stream_id: Optional id for the outlet stream.

    Returns:
        HXResult with outlet stream, duty [W], and outlet phase information.

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
        result = heat_stream(feed, provider, delta_T_K=10.0)
    """
    target_error = _target_temperature_error(outlet_temperature_K, delta_T_K)
    if target_error:
        return _failed(stream.id, target_error)

    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        return _failed(stream.id, validation_error)

    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        return _failed(stream.id, zs_or_error)
    zs = zs_or_error

    target_T = (
        float(outlet_temperature_K)
        if outlet_temperature_K is not None
        else float(stream.temperature_K) + float(delta_T_K)
    )
    if target_T <= 0:
        return _failed(
            stream.id,
            f"Outlet temperature must be positive, got {target_T} K.",
        )

    try:
        inlet = provider.flasher.flash(
            T=float(stream.temperature_K),
            P=float(stream.pressure_Pa),
            zs=zs,
        )
        outlet = provider.flasher.flash(
            T=target_T,
            P=float(stream.pressure_Pa),
            zs=zs,
        )
    except Exception as exc:
        return _failed(
            stream.id,
            f"thermo PT flash failed for heat-stream calculation on "
            f"'{stream.id}': {exc}",
        )

    inlet_h = _enthalpy_Jmol(inlet)
    outlet_h = _enthalpy_Jmol(outlet)
    if inlet_h is None or outlet_h is None:
        return _failed(
            stream.id,
            f"thermo did not report molar enthalpy for stream '{stream.id}'.",
        )

    vf = _safe_float(getattr(outlet, "VF", None))
    if vf is None:
        return _failed(
            stream.id,
            f"thermo outlet flash for stream '{stream.id}' did not report VF.",
        )
    vf = min(1.0, max(0.0, vf))
    lf = 1.0 - vf

    phase_count = _safe_int(getattr(outlet, "phase_count", None))
    phase_compositions = _phase_compositions(outlet, provider.compounds)

    outlet_stream = StreamState(
        id=outlet_stream_id or f"{stream.id}_hx",
        temperature_K=target_T,
        pressure_Pa=stream.pressure_Pa,
        molar_flow_mols=stream.molar_flow_mols,
        composition=dict(zip(provider.compounds, zs)),
        vapor_fraction=vf,
        history=stream.history + ("hx",),
    )

    return HXResult(
        success=True,
        inlet_stream_id=stream.id,
        outlet_stream=outlet_stream,
        duty_W=stream.molar_flow_mols * (outlet_h - inlet_h),
        inlet_enthalpy_Jmol=inlet_h,
        outlet_enthalpy_Jmol=outlet_h,
        phase_state=_phase_state(vf, getattr(outlet, "gas", None), getattr(outlet, "liquid0", None), phase_count),
        vapor_fraction=vf,
        liquid_fraction=lf,
        phase_compositions=phase_compositions,
        raw_phase_count=phase_count,
    )


def _target_temperature_error(
    outlet_temperature_K: float | None,
    delta_T_K: float | None,
) -> str | None:
    if outlet_temperature_K is None and delta_T_K is None:
        return "Provide exactly one of outlet_temperature_K or delta_T_K."
    if outlet_temperature_K is not None and delta_T_K is not None:
        return "Provide only one of outlet_temperature_K or delta_T_K, not both."
    return None


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


def _failed(stream_id: str, error_message: str) -> HXResult:
    return HXResult(
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
