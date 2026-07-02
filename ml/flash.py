"""Thermo-based flash drum model for flowsheet search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import FlashResult, StreamState

try:
    from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL, PRMIX
    from thermo.interaction_parameters import IPDB
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "thermo is required for ml.flash. Install it with: pip install thermo"
    ) from exc


_COMPOSITION_TOL = 1e-12
_PHASE_TOL = 1e-9


@dataclass(frozen=True)
class ThermoFlashProvider:
    """Reusable thermo FlashVL object with component ordering metadata.

    Args:
        compounds: Component identifiers in thermo order.
        flasher: Configured thermo FlashVL object.
        constants: thermo ChemicalConstantsPackage used by the flasher.
        properties: thermo PropertyCorrelationsPackage used by the flasher.

    Returns:
        Provider consumed by flash_split().

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
    """

    compounds: tuple[str, ...]
    flasher: Any
    constants: Any
    properties: Any


def build_pr_flasher(compounds: list[str]) -> ThermoFlashProvider:
    """Build a Peng-Robinson thermo flasher for VLE calculations.

    Args:
        compounds: thermo component identifiers in the desired composition order.

    Returns:
        ThermoFlashProvider with a reusable FlashVL flasher.

    Raises:
        ValueError: if compounds are empty, duplicated, or not recognised by thermo.

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
    """
    if not compounds:
        raise ValueError("compounds must contain at least one component.")

    normalized = tuple(str(compound).strip() for compound in compounds)
    if any(not compound for compound in normalized):
        raise ValueError("compounds must not contain blank identifiers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"compounds must be unique, got {list(normalized)}.")

    try:
        constants, properties = ChemicalConstantsPackage.from_IDs(list(normalized))
    except Exception as exc:
        raise ValueError(
            f"Could not create thermo constants for compounds {list(normalized)}: "
            f"{exc}. Use thermo component identifiers such as 'methane'."
        ) from exc

    try:
        kijs = IPDB.get_ip_asymmetric_matrix("ChemSep PR", constants.CASs, "kij")
    except Exception:
        kijs = None

    eos_kwargs = {
        "Pcs": constants.Pcs,
        "Tcs": constants.Tcs,
        "omegas": constants.omegas,
    }
    if kijs is not None:
        eos_kwargs["kijs"] = kijs

    try:
        gas = CEOSGas(
            PRMIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        liquid = CEOSLiquid(
            PRMIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        flasher = FlashVL(constants, properties, gas=gas, liquid=liquid)
    except Exception as exc:
        raise ValueError(
            f"Could not build Peng-Robinson thermo flasher for "
            f"{list(normalized)}: {exc}"
        ) from exc

    return ThermoFlashProvider(
        compounds=normalized,
        flasher=flasher,
        constants=constants,
        properties=properties,
    )


def flash_split(stream: StreamState, provider: ThermoFlashProvider) -> FlashResult:
    """Run a split-only PT flash for one stream.

    The flash drum does not change stream temperature or pressure. Conditioning
    actions such as heaters, coolers, pumps, and valves should run upstream.

    Args:
        stream: Feed stream state.
        provider: ThermoFlashProvider from build_pr_flasher().

    Returns:
        FlashResult containing child streams for two-phase flashes, or a
        single-phase no-split result.

    Example:
        provider = build_pr_flasher(["methane", "ethane", "nitrogen"])
        result = flash_split(feed, provider)
    """
    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        return _failed(stream.id, zs_or_error)
    zs = zs_or_error

    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        return _failed(stream.id, validation_error)

    try:
        flash = provider.flasher.flash(
            T=float(stream.temperature_K),
            P=float(stream.pressure_Pa),
            zs=zs,
        )
    except Exception as exc:
        return _failed(
            stream.id,
            f"thermo PT flash failed for stream '{stream.id}': {exc}",
        )

    phase_count = _safe_int(getattr(flash, "phase_count", None))
    vf = _safe_float(getattr(flash, "VF", None))
    if vf is None:
        return _failed(
            stream.id,
            f"thermo PT flash for stream '{stream.id}' did not report VF.",
        )

    vf = min(1.0, max(0.0, vf))
    lf = 1.0 - vf

    gas_phase = getattr(flash, "gas", None)
    liquid_phase = getattr(flash, "liquid0", None)

    if phase_count == 2 and gas_phase is not None and liquid_phase is not None:
        try:
            ys = _normalise_list([float(v) for v in gas_phase.zs])
            xs = _normalise_list([float(v) for v in liquid_phase.zs])
        except Exception as exc:
            return _failed(
                stream.id,
                f"thermo PT flash returned invalid phase compositions: {exc}",
            )

        vapor_stream = _child_stream(
            stream,
            suffix="vapor",
            molar_flow_mols=stream.molar_flow_mols * vf,
            compounds=provider.compounds,
            composition_values=ys,
            vapor_fraction=1.0,
        )
        liquid_stream = _child_stream(
            stream,
            suffix="liquid",
            molar_flow_mols=stream.molar_flow_mols * lf,
            compounds=provider.compounds,
            composition_values=xs,
            vapor_fraction=0.0,
        )

        # Latent heat of the vapor fraction: F_total × VF × (H_gas − H_liq).
        # H_gas > H_liq at same T, P → duty is always ≥ 0 and comparable in
        # scale to distillation condenser/reboiler duty for the same feed.
        try:
            h_gas = float(gas_phase.H())
            h_liq = float(liquid_phase.H())
            duty_W: float | None = stream.molar_flow_mols * vf * max(0.0, h_gas - h_liq)
        except Exception:
            duty_W = None

        return FlashResult(
            success=True,
            inlet_stream_id=stream.id,
            phase_state="two_phase",
            vapor_fraction=vf,
            liquid_fraction=lf,
            vapor_stream=vapor_stream,
            liquid_stream=liquid_stream,
            duty_W=duty_W,
            raw_phase_count=phase_count,
        )

    phase_state = _single_phase_state(vf, gas_phase, liquid_phase)
    warnings = []
    if phase_count not in (1, None):
        warnings.append(
            f"thermo reported phase_count={phase_count}; v1 flash only emits "
            "two-phase VLE splits or single-phase no-split results."
        )

    return FlashResult(
        success=True,
        inlet_stream_id=stream.id,
        phase_state=phase_state,
        vapor_fraction=vf,
        liquid_fraction=lf,
        warnings=warnings,
        raw_phase_count=phase_count,
    )


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


def _child_stream(
    parent: StreamState,
    suffix: str,
    molar_flow_mols: float,
    compounds: tuple[str, ...],
    composition_values: list[float],
    vapor_fraction: float | None = None,
) -> StreamState:
    return StreamState(
        id=f"{parent.id}_{suffix}",
        temperature_K=parent.temperature_K,
        pressure_Pa=parent.pressure_Pa,
        molar_flow_mols=molar_flow_mols,
        composition=dict(zip(compounds, composition_values)),
        vapor_fraction=vapor_fraction,
        history=parent.history + (f"flash:{suffix}",),
    )


def _single_phase_state(vf: float, gas_phase: Any, liquid_phase: Any) -> str:
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


def _failed(stream_id: str, error_message: str) -> FlashResult:
    return FlashResult(
        success=False,
        inlet_stream_id=stream_id,
        phase_state="unknown",
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
