"""Fenske-Underwood-Gilliland shortcut distillation model."""

from __future__ import annotations

import math
from typing import Any

from .flash import ThermoFlashProvider
from .types import ShortcutDistillationResult, StreamState


_COMPOSITION_TOL = 1e-12
_ROOT_TOL = 1e-12
_H_OVERFLOW = 1e8  # 100 MJ/mol — PR flash singularity guard (hydrocarbons physically ±5×10^5 J/mol)
_MAX_PRACTICAL_REFLUX = 50.0  # default for max_reflux_ratio parameter


def estimate_relative_volatilities(
    stream: StreamState,
    provider: ThermoFlashProvider,
    pressure_Pa: float | None = None,
    vapor_fraction: float = 0.5,
    heavy_key: str | None = None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Estimate relative volatilities from a representative thermo VLE flash.

    The calculation uses the configured thermo flasher, so EOS choice and
    binary interaction coefficients are inherited from the provider. K-values
    are computed as y_i/x_i from the requested P/VF flash and relative
    volatilities are K_i/K_heavy_key when a heavy key is provided.

    Args:
        stream: Feed stream state.
        provider: ThermoFlashProvider from build_pr_flasher().
        pressure_Pa: Column pressure [Pa]. Defaults to stream pressure.
        vapor_fraction: Representative flash vapor fraction, 0 < VF < 1.
        heavy_key: Optional component used as alpha = 1 reference.

    Returns:
        Tuple of (relative_volatilities, k_values, warnings).

    Raises:
        ValueError: if inputs are invalid or thermo cannot provide a VLE split.

    Example:
        alphas, k_values, warnings = estimate_relative_volatilities(
            feed, provider, heavy_key="n-butane"
        )
    """
    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        raise ValueError(validation_error)
    if heavy_key is not None and heavy_key not in provider.compounds:
        raise ValueError(
            f"heavy_key '{heavy_key}' is not in provider compounds "
            f"{list(provider.compounds)}."
        )
    if not 0.0 < vapor_fraction < 1.0:
        raise ValueError(f"vapor_fraction must satisfy 0 < VF < 1, got {vapor_fraction}.")

    pressure = float(stream.pressure_Pa if pressure_Pa is None else pressure_Pa)
    if pressure <= 0:
        raise ValueError(f"pressure_Pa must be positive, got {pressure}.")

    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        raise ValueError(zs_or_error)
    zs = zs_or_error

    warnings: list[str] = []
    try:
        flash = provider.flasher.flash(P=pressure, VF=float(vapor_fraction), zs=zs)
    except Exception as exc:
        warnings.append(
            f"thermo P/VF flash failed at P={pressure} Pa and VF={vapor_fraction}: "
            f"{exc}. Falling back to stream T/P flash."
        )
        try:
            flash = provider.flasher.flash(
                T=float(stream.temperature_K),
                P=pressure,
                zs=zs,
            )
        except Exception as fallback_exc:
            raise ValueError(
                "Could not estimate relative volatilities because thermo failed "
                f"both P/VF and T/P flashes: {fallback_exc}"
            ) from fallback_exc

    k_values = _k_values_from_flash(flash, provider.compounds)
    if not k_values:
        raise ValueError(
            "Could not estimate relative volatilities because thermo did not "
            "return both vapor and liquid phase compositions."
        )

    reference = heavy_key or provider.compounds[-1]
    reference_k = k_values.get(reference)
    if reference_k is None or reference_k <= 0:
        raise ValueError(
            f"Could not normalize relative volatilities because K[{reference}] "
            f"is not positive ({reference_k})."
        )

    relative_volatilities = {
        compound: value / reference_k for compound, value in k_values.items()
    }
    return relative_volatilities, k_values, warnings


def shortcut_distillation_fug(
    stream: StreamState,
    provider: ThermoFlashProvider,
    light_key: str,
    heavy_key: str,
    light_key_recovery: float = 0.95,
    heavy_key_recovery: float = 0.05,
    pressure_Pa: float | None = None,
    relative_volatilities: dict[str, float] | None = None,
    feed_quality: float | None = None,
    reflux_ratio: float | None = None,
    reflux_ratio_multiplier: float = 1.5,
    column_id: str | None = None,
    max_reflux_ratio: float = _MAX_PRACTICAL_REFLUX,
) -> ShortcutDistillationResult:
    """Run a Fenske-Underwood-Gilliland shortcut distillation estimate.

    The model consumes one feed stream at its current pressure, computes feed
    quality from the inlet T/P flash, assumes a total condenser, assigns
    light/heavy key recoveries to the distillate, estimates non-key
    distribution with the Fenske relation, and returns distillate/bottoms
    streams plus FUG stage and reflux estimates.

    Args:
        stream: Feed stream state.
        provider: ThermoFlashProvider from build_pr_flasher().
        light_key: Component expected to be more volatile than heavy_key.
        heavy_key: Component used as relative-volatility reference.
        light_key_recovery: Fraction of light key recovered to distillate.
        heavy_key_recovery: Fraction of heavy key recovered to distillate.
        pressure_Pa: Unsupported override retained for compatibility. Column
            pressure is always stream.pressure_Pa in this v1 model.
        relative_volatilities: Optional alpha values. When omitted, alpha is
            estimated from thermo K-values at stream.pressure_Pa and VF=0.5.
        feed_quality: Unsupported override retained for compatibility. Underwood
            q is computed as the inlet liquid fraction from the stream T/P flash.
        reflux_ratio: Actual reflux ratio. If omitted, uses
            reflux_ratio_multiplier * R_min.
        reflux_ratio_multiplier: Multiplier for R_min when reflux_ratio is omitted.
        column_id: Optional id prefix for outlet streams.
        max_reflux_ratio: Upper bound on R_min. R_min above this value means the
            separation is thermodynamically infeasible at these conditions (relative
            volatilities have collapsed) or the Underwood solver converged to a
            degenerate root. Defaults to _MAX_PRACTICAL_REFLUX (50). No real
            column operates above ~20; values above 50 indicate an infeasible split.

    Returns:
        ShortcutDistillationResult with product streams and FUG estimates.

    Example:
        result = shortcut_distillation_fug(
            feed,
            provider,
            light_key="propane",
            heavy_key="n-butane",
            light_key_recovery=0.95,
            heavy_key_recovery=0.05,
        )
    """
    validation_error = _validate_inputs(
        stream,
        provider,
        light_key,
        heavy_key,
        light_key_recovery,
        heavy_key_recovery,
        pressure_Pa,
        feed_quality,
        reflux_ratio,
        reflux_ratio_multiplier,
    )
    if validation_error:
        return _failed(stream.id, validation_error)

    zs_or_error = _composition_vector(stream, provider.compounds)
    if isinstance(zs_or_error, str):
        return _failed(stream.id, zs_or_error)
    zs = zs_or_error
    pressure = float(stream.pressure_Pa)

    feed_quality_result = _feed_quality_from_stream(stream, provider, zs)
    if isinstance(feed_quality_result, str):
        return _failed(stream.id, feed_quality_result)
    computed_feed_quality = feed_quality_result

    warnings: list[str] = []
    k_values: dict[str, float] = {}
    if relative_volatilities is None:
        try:
            alphas, k_values, alpha_warnings = estimate_relative_volatilities(
                stream,
                provider,
                pressure_Pa=pressure,
                vapor_fraction=0.5,
                heavy_key=heavy_key,
            )
            warnings.extend(alpha_warnings)
        except ValueError as exc:
            return _failed(
                stream.id,
                "Could not estimate relative volatilities from thermo. Provide "
                f"relative_volatilities explicitly or adjust pressure: {exc}",
            )
    else:
        alpha_result = _normalise_relative_volatilities(
            relative_volatilities,
            provider.compounds,
            heavy_key,
        )
        if isinstance(alpha_result, str):
            return _failed(stream.id, alpha_result)
        alphas = alpha_result

    alpha_lk = alphas[light_key]
    alpha_hk = alphas[heavy_key]
    if not math.isclose(alpha_hk, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return _failed(
            stream.id,
            f"Internal alpha normalization failed; expected alpha[{heavy_key}]=1.",
        )
    if alpha_lk <= 1.0:
        return _failed(
            stream.id,
            f"light_key '{light_key}' must be more volatile than heavy_key "
            f"'{heavy_key}'. Got alpha_LK/HK={alpha_lk}.",
        )

    feed_moles = {
        compound: stream.molar_flow_mols * z
        for compound, z in zip(provider.compounds, zs)
    }
    if feed_moles[light_key] <= _COMPOSITION_TOL:
        return _failed(stream.id, f"light_key '{light_key}' has zero feed flow.")
    if feed_moles[heavy_key] <= _COMPOSITION_TOL:
        return _failed(stream.id, f"heavy_key '{heavy_key}' has zero feed flow.")

    minimum_stages = _fenske_minimum_stages(
        feed_moles,
        light_key,
        heavy_key,
        light_key_recovery,
        heavy_key_recovery,
        alpha_lk,
    )
    if minimum_stages <= 0 or not math.isfinite(minimum_stages):
        return _failed(
            stream.id,
            f"Fenske minimum stages were invalid ({minimum_stages}); check key "
            "recoveries and relative volatility.",
        )

    distillate_moles, bottoms_moles = _component_splits(
        feed_moles,
        alphas,
        light_key,
        heavy_key,
        light_key_recovery,
        heavy_key_recovery,
        minimum_stages,
    )
    total_distillate = sum(distillate_moles.values())
    total_bottoms = sum(bottoms_moles.values())
    if total_distillate <= _COMPOSITION_TOL or total_bottoms <= _COMPOSITION_TOL:
        return _failed(
            stream.id,
            "Shortcut distillation produced an empty product stream; adjust key "
            "recoveries or feed composition.",
        )

    x_distillate = _normalise_dict(distillate_moles)
    x_bottoms = _normalise_dict(bottoms_moles)

    theta = _underwood_root(
        alphas,
        dict(zip(provider.compounds, zs)),
        computed_feed_quality,
        heavy_key,
        light_key,
    )
    if theta is None:
        return _failed(
            stream.id,
            "Underwood root could not be bracketed between heavy and light keys; "
            "check relative volatilities, keys, and feed_quality.",
        )

    minimum_reflux = _minimum_reflux_ratio(alphas, x_distillate, theta)
    if minimum_reflux is None or minimum_reflux <= 0.0:
        return _failed(
            stream.id,
            f"Underwood minimum reflux ratio was invalid ({minimum_reflux}); "
            "check key selection and relative volatilities.",
        )
    if minimum_reflux > float(max_reflux_ratio):
        return _failed(
            stream.id,
            f"Underwood R_min={minimum_reflux:.3g} exceeds max_reflux_ratio="
            f"{max_reflux_ratio}; the {light_key}/{heavy_key} split is "
            "thermodynamically infeasible at these conditions (relative "
            "volatilities have collapsed or the Underwood root is degenerate).",
        )

    actual_reflux = (
        float(reflux_ratio)
        if reflux_ratio is not None
        else float(reflux_ratio_multiplier) * minimum_reflux
    )
    if actual_reflux <= minimum_reflux:
        return _failed(
            stream.id,
            f"reflux_ratio must be greater than R_min={minimum_reflux}, got "
            f"{actual_reflux}.",
        )

    theoretical_stages = _gilliland_stages(minimum_stages, minimum_reflux, actual_reflux)
    if theoretical_stages is None:
        return _failed(
            stream.id,
            "Gilliland stage estimate failed; check reflux_ratio and R_min.",
        )

    distillate_temperature = _estimate_product_temperature(
        provider,
        pressure,
        x_distillate,
        vapor_fraction=0.0,
        warnings=warnings,
        product_name="total-condenser distillate",
    )
    if distillate_temperature is None:
        return _failed(
            stream.id,
            "Distillate bubble-point flash failed at column pressure "
            f"({pressure:.0f} Pa); product cannot exist as liquid at this pressure.",
        )
    condenser_error = _condenser_feasible(
        provider, pressure, distillate_temperature, x_distillate
    )
    if condenser_error:
        return _failed(stream.id, f"Total condenser infeasible: {condenser_error}")
    bottoms_temperature = _estimate_product_temperature(
        provider,
        pressure,
        x_bottoms,
        vapor_fraction=0.0,
        warnings=warnings,
        product_name="bottoms",
    )
    if bottoms_temperature is None:
        return _failed(
            stream.id,
            "Bottoms bubble-point flash failed at column pressure "
            f"({pressure:.0f} Pa); product cannot exist as liquid at this pressure.",
        )

    if distillate_temperature >= bottoms_temperature:
        return _failed(
            stream.id,
            f"Condenser ({distillate_temperature:.1f} K) ≥ reboiler "
            f"({bottoms_temperature:.1f} K) at {pressure:.0f} Pa — "
            "column violates the second law; check component ordering or pressure.",
        )

    prefix = column_id or stream.id
    distillate_stream = StreamState(
        id=f"{prefix}_distillate",
        temperature_K=distillate_temperature,
        pressure_Pa=pressure,
        molar_flow_mols=total_distillate,
        composition=x_distillate,
        vapor_fraction=0.0,
        history=stream.history + ("shortcut_distillation:total_condenser_distillate",),
    )
    bottoms_stream = StreamState(
        id=f"{prefix}_bottoms",
        temperature_K=bottoms_temperature,
        pressure_Pa=pressure,
        molar_flow_mols=total_bottoms,
        composition=x_bottoms,
        vapor_fraction=0.0,
        history=stream.history + ("shortcut_distillation:bottoms",),
    )

    component_recoveries = {
        compound: (
            distillate_moles[compound] / feed_moles[compound]
            if feed_moles[compound] > _COMPOSITION_TOL
            else 0.0
        )
        for compound in provider.compounds
    }

    return ShortcutDistillationResult(
        success=True,
        inlet_stream_id=stream.id,
        distillate_stream=distillate_stream,
        bottoms_stream=bottoms_stream,
        light_key=light_key,
        heavy_key=heavy_key,
        pressure_Pa=pressure,
        component_recoveries=component_recoveries,
        relative_volatilities=alphas,
        k_values=k_values,
        minimum_stages=minimum_stages,
        underwood_theta=theta,
        minimum_reflux_ratio=minimum_reflux,
        reflux_ratio=actual_reflux,
        theoretical_stages=theoretical_stages,
        feed_quality=computed_feed_quality,
        reflux_ratio_multiplier=(
            None if reflux_ratio is not None else float(reflux_ratio_multiplier)
        ),
        warnings=warnings,
    )


def column_duties_from_energy_balance(
    feed_stream: StreamState,
    distillate_stream: StreamState,
    bottoms_stream: StreamState,
    reflux_ratio: float,
    provider: ThermoFlashProvider,
) -> tuple[float, float]:
    """Compute condenser and reboiler duties via column energy balance.

    Uses thermo to evaluate stream enthalpies. For a total condenser:

        Q_condenser = (R + 1) * F_dist * λ_dist
        λ_dist = h_vapor_dist − h_liq_dist  (latent heat at column pressure)

    Reboiler from the overall column energy balance:

        Q_reboiler = Q_condenser + F_dist*h_liq_dist + F_bot*h_liq_bot − F_feed*h_feed

    Args:
        feed_stream: Feed stream at its actual T, P, and composition.
        distillate_stream: Distillate product stream (at bubble point).
        bottoms_stream: Bottoms product stream (at bubble point).
        reflux_ratio: Actual reflux ratio from the FUG calculation.
        provider: ThermoFlashProvider used for the column.

    Returns:
        Tuple (condenser_duty_W, reboiler_duty_W) in watts. Both are positive
        for a conventional column.

    Raises:
        ValueError: if any flash fails or any enthalpy exceeds _H_OVERFLOW
            (100 MJ/mol — far beyond physical values for hydrocarbons ±5×10^5
            J/mol), indicating a PR flash singularity.  The caller should treat
            this as a failed energy-balance action rather than silencing it.

    Example:
        q_cond, q_reb = column_duties_from_energy_balance(
            feed, result.distillate_stream, result.bottoms_stream,
            result.reflux_ratio, provider
        )
    """
    def _zs(stream: StreamState) -> list[float]:
        raw = [float(stream.composition.get(c, 0.0)) for c in provider.compounds]
        total = sum(raw)
        return raw if total <= 0.0 else [v / total for v in raw]

    z_feed = _zs(feed_stream)
    z_dist = _zs(distillate_stream)
    z_bot = _zs(bottoms_stream)

    try:
        h_feed = float(
            provider.flasher.flash(
                T=float(feed_stream.temperature_K),
                P=float(feed_stream.pressure_Pa),
                zs=z_feed,
            ).H()
        )
        P_col = float(distillate_stream.pressure_Pa)
        h_liq_dist = float(provider.flasher.flash(P=P_col, VF=0.0, zs=z_dist).H())
        h_vap_dist = float(provider.flasher.flash(P=P_col, VF=1.0, zs=z_dist).H())
        h_liq_bot = float(provider.flasher.flash(P=P_col, VF=0.0, zs=z_bot).H())
    except Exception as exc:
        raise ValueError(
            f"Flash failed in column energy balance: {exc}"
        ) from exc

    if (
        abs(h_feed) > _H_OVERFLOW
        or abs(h_liq_dist) > _H_OVERFLOW
        or abs(h_vap_dist) > _H_OVERFLOW
        or abs(h_liq_bot) > _H_OVERFLOW
    ):
        raise ValueError(
            f"Enthalpy overflow in column energy balance: "
            f"H_feed={h_feed:.3e}, H_liq_dist={h_liq_dist:.3e}, "
            f"H_vap_dist={h_vap_dist:.3e}, H_liq_bot={h_liq_bot:.3e} J/mol "
            f"(physical max ~±5×10^5 J/mol for hydrocarbons)."
        )

    lambda_dist = h_vap_dist - h_liq_dist
    F_dist = float(distillate_stream.molar_flow_mols)
    F_bot = float(bottoms_stream.molar_flow_mols)
    F_feed = float(feed_stream.molar_flow_mols)

    q_condenser = (float(reflux_ratio) + 1.0) * F_dist * lambda_dist
    delta_H = F_dist * h_liq_dist + F_bot * h_liq_bot - F_feed * h_feed
    q_reboiler = q_condenser + delta_H

    if not (math.isfinite(q_condenser) and math.isfinite(q_reboiler)):
        raise ValueError(
            f"Non-finite column duty: q_cond={q_condenser:.3e}, "
            f"q_reb={q_reboiler:.3e} W."
        )
    return q_condenser, q_reboiler


def _validate_inputs(
    stream: StreamState,
    provider: ThermoFlashProvider,
    light_key: str,
    heavy_key: str,
    light_key_recovery: float,
    heavy_key_recovery: float,
    pressure_Pa: float | None,
    feed_quality: float | None,
    reflux_ratio: float | None,
    reflux_ratio_multiplier: float,
) -> str | None:
    validation_error = _validate_stream_conditions(stream)
    if validation_error:
        return validation_error
    if light_key == heavy_key:
        return "light_key and heavy_key must be different components."
    for name, compound in (("light_key", light_key), ("heavy_key", heavy_key)):
        if compound not in provider.compounds:
            return (
                f"{name} '{compound}' is not in provider compounds "
                f"{list(provider.compounds)}."
            )
    if not 0.0 < light_key_recovery < 1.0:
        return (
            "light_key_recovery must satisfy 0 < recovery < 1, got "
            f"{light_key_recovery}."
        )
    if not 0.0 < heavy_key_recovery < 1.0:
        return (
            "heavy_key_recovery must satisfy 0 < recovery < 1, got "
            f"{heavy_key_recovery}."
        )
    if light_key_recovery <= heavy_key_recovery:
        return (
            "light_key_recovery must be greater than heavy_key_recovery for a "
            "normal distillation split."
        )
    if pressure_Pa is not None and not math.isclose(
        float(pressure_Pa),
        float(stream.pressure_Pa),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        return (
            "pressure_Pa is not an MCTS decision variable in this v1 column; "
            "column pressure is taken from stream.pressure_Pa. Use an upstream "
            "pump, compressor, or valve to change pressure."
        )
    if feed_quality is not None:
        return (
            "feed_quality is not an MCTS decision variable in this v1 column; "
            "it is computed from the inlet stream thermo flash. Use upstream "
            "heating, cooling, or pressure conditioning to change feed quality."
        )
    if reflux_ratio is not None and reflux_ratio <= 0:
        return f"reflux_ratio must be positive when provided, got {reflux_ratio}."
    if reflux_ratio is None and reflux_ratio_multiplier <= 1.0:
        return (
            "reflux_ratio_multiplier must be greater than 1 when reflux_ratio "
            f"is omitted, got {reflux_ratio_multiplier}."
        )
    return None


def _feed_quality_from_stream(
    stream: StreamState,
    provider: ThermoFlashProvider,
    zs: list[float],
) -> float | str:
    try:
        flash = provider.flasher.flash(
            T=float(stream.temperature_K),
            P=float(stream.pressure_Pa),
            zs=zs,
        )
    except Exception as exc:
        return (
            f"Could not compute feed_quality from inlet stream '{stream.id}' "
            f"T/P flash: {exc}"
        )

    vapor_fraction = _safe_float(getattr(flash, "VF", None))
    if vapor_fraction is None:
        return (
            f"Could not compute feed_quality from inlet stream '{stream.id}' "
            "because thermo did not report VF."
        )

    vapor_fraction = min(1.0, max(0.0, vapor_fraction))
    return 1.0 - vapor_fraction


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


def _k_values_from_flash(flash: Any, compounds: tuple[str, ...]) -> dict[str, float]:
    gas = getattr(flash, "gas", None)
    liquid = getattr(flash, "liquid0", None)
    if gas is None or liquid is None:
        return {}

    try:
        ys = _normalise_list([float(value) for value in gas.zs])
        xs = _normalise_list([float(value) for value in liquid.zs])
    except Exception:
        return {}

    k_values = {}
    for compound, y_value, x_value in zip(compounds, ys, xs):
        if x_value <= _COMPOSITION_TOL:
            return {}
        k_values[compound] = y_value / x_value
    return k_values


def _normalise_relative_volatilities(
    relative_volatilities: dict[str, float],
    compounds: tuple[str, ...],
    heavy_key: str,
) -> dict[str, float] | str:
    missing = [compound for compound in compounds if compound not in relative_volatilities]
    if missing:
        return (
            "relative_volatilities must include every provider compound; missing "
            f"{missing}."
        )

    raw = {}
    for compound in compounds:
        try:
            value = float(relative_volatilities[compound])
        except (TypeError, ValueError):
            return f"relative_volatilities[{compound!r}] must be numeric."
        if value <= 0:
            return (
                f"relative_volatilities[{compound!r}] must be positive, got "
                f"{value}."
            )
        raw[compound] = value

    reference = raw[heavy_key]
    return {compound: value / reference for compound, value in raw.items()}


def _fenske_minimum_stages(
    feed_moles: dict[str, float],
    light_key: str,
    heavy_key: str,
    light_key_recovery: float,
    heavy_key_recovery: float,
    alpha_lk_hk: float,
) -> float:
    distillate_lk = light_key_recovery * feed_moles[light_key]
    bottoms_lk = (1.0 - light_key_recovery) * feed_moles[light_key]
    distillate_hk = heavy_key_recovery * feed_moles[heavy_key]
    bottoms_hk = (1.0 - heavy_key_recovery) * feed_moles[heavy_key]
    separation_factor = (distillate_lk / bottoms_lk) / (distillate_hk / bottoms_hk)
    return math.log(separation_factor) / math.log(alpha_lk_hk)


def _component_splits(
    feed_moles: dict[str, float],
    alphas: dict[str, float],
    light_key: str,
    heavy_key: str,
    light_key_recovery: float,
    heavy_key_recovery: float,
    minimum_stages: float,
) -> tuple[dict[str, float], dict[str, float]]:
    distillate = {}
    bottoms = {}
    distillate[light_key] = light_key_recovery * feed_moles[light_key]
    bottoms[light_key] = feed_moles[light_key] - distillate[light_key]
    distillate[heavy_key] = heavy_key_recovery * feed_moles[heavy_key]
    bottoms[heavy_key] = feed_moles[heavy_key] - distillate[heavy_key]

    hk_ratio = distillate[heavy_key] / bottoms[heavy_key]
    for compound, feed_value in feed_moles.items():
        if compound in {light_key, heavy_key}:
            continue
        if feed_value <= _COMPOSITION_TOL:
            distillate[compound] = 0.0
            bottoms[compound] = 0.0
            continue
        distribution_ratio = hk_ratio * alphas[compound] ** minimum_stages
        distillate[compound] = feed_value * distribution_ratio / (1.0 + distribution_ratio)
        bottoms[compound] = feed_value - distillate[compound]

    return distillate, bottoms


def _underwood_root(
    alphas: dict[str, float],
    feed_zs: dict[str, float],
    feed_quality: float,
    heavy_key: str,
    light_key: str,
) -> float | None:
    alpha_hk = alphas[heavy_key]
    alpha_lk = alphas[light_key]
    if alpha_hk >= alpha_lk:
        return None

    # The active Underwood root φ satisfies α_LK > φ > α_HK (Mazzotti eq. 20).
    # Within this interval the feed function f(φ) = Σ αᵢzᵢ/(αᵢ−φ) is strictly
    # monotonically increasing (df/dφ > 0, eq. 19) and has exactly one root —
    # but only when LK and HK are *adjacent* in the volatility scale (no
    # component α lies strictly between α_HK and α_LK).
    #
    # For non-adjacent key pairs a distributing component creates a pole inside
    # (α_HK, α_LK).  Each pole-free sub-interval has its own root; using one
    # root underestimates R_min.  Return None so the caller rejects the split;
    # it should be decomposed into a sequence of adjacent-pair columns.
    above_hk = sorted(a for a in alphas.values() if a > alpha_hk + 1e-10)
    if not above_hk:
        return None
    alpha_next = above_hk[0]
    if not math.isclose(alpha_next, alpha_lk, rel_tol=1e-6):
        return None  # distributing component between HK and LK — non-adjacent pair

    target = 1.0 - float(feed_quality)
    return _newton_underwood(alphas, feed_zs, target, alpha_hk, alpha_lk)


def _newton_underwood(
    alphas: dict[str, float],
    feed_zs: dict[str, float],
    target: float,
    lower: float,
    upper: float,
    max_iter: int = 50,
) -> float | None:
    """Safe Newton-Raphson root of Σ αᵢzᵢ/(αᵢ−φ) = target on (lower, upper).

    Within a pole-free sub-interval the Underwood feed function is strictly
    monotonically increasing (df/dφ = Σ αᵢzᵢ/(αᵢ−φ)² > 0, Mazzotti eq. 19),
    guaranteeing quadratic convergence from any interior starting point.  When a
    Newton step escapes the bracket, a bisection fallback keeps φ inside
    (lower, upper).  f and f' are evaluated in one pass by reusing the
    intermediate t = αᵢzᵢ/(αᵢ−φ), so that f' = Σ t/(αᵢ−φ).
    """
    theta = 0.5 * (lower + upper)
    for _ in range(max_iter):
        f_val = 0.0
        df_val = 0.0
        for compound in alphas:
            a = alphas[compound]
            z = feed_zs[compound]
            d = a - theta
            if abs(d) < 1e-14:
                return None  # at a pole — excluded by adjacency check upstream
            t = a * z / d
            f_val += t
            df_val += t / d  # == a*z/d²
        f_val -= target

        if abs(f_val) < _ROOT_TOL:
            return theta

        if df_val <= 0.0 or not math.isfinite(df_val):
            return None

        theta_new = theta - f_val / df_val

        if theta_new <= lower or theta_new >= upper:
            # Newton step escaped — fall back to bisection
            if f_val > 0.0:
                upper = theta
            else:
                lower = theta
            theta = 0.5 * (lower + upper)
        else:
            if f_val > 0.0:
                upper = theta
            else:
                lower = theta
            theta = theta_new

        if (upper - lower) < _ROOT_TOL:
            return 0.5 * (lower + upper)

    return 0.5 * (lower + upper)


def _minimum_reflux_ratio(
    alphas: dict[str, float],
    distillate_zs: dict[str, float],
    theta: float,
) -> float | None:
    try:
        value = sum(
            alphas[compound] * distillate_zs[compound] / (alphas[compound] - theta)
            for compound in alphas
        ) - 1.0
    except ZeroDivisionError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _gilliland_stages(
    minimum_stages: float,
    minimum_reflux: float,
    reflux_ratio: float,
) -> float | None:
    if reflux_ratio <= minimum_reflux or reflux_ratio <= -1.0:
        return None
    x_value = (reflux_ratio - minimum_reflux) / (reflux_ratio + 1.0)
    x_value = min(0.999999999, max(1e-12, x_value))
    exponent = ((1.0 + 54.4 * x_value) / (11.0 + 117.2 * x_value))
    exponent *= (x_value - 1.0) / math.sqrt(x_value)
    y_value = 1.0 - math.exp(exponent)
    if not 0.0 <= y_value < 1.0:
        return None
    return (minimum_stages + y_value) / (1.0 - y_value)


def _estimate_product_temperature(
    provider: ThermoFlashProvider,
    pressure_Pa: float,
    composition: dict[str, float],
    vapor_fraction: float,
    warnings: list[str],
    product_name: str,
) -> float | None:
    """Return bubble/dew-point temperature, or None if the flash fails.

    Returning None signals that the product cannot exist as a liquid at this
    pressure (e.g. CO2 below its triple-point pressure of 5.18 bar), and the
    caller should treat the column as infeasible rather than falling back to the
    feed temperature.
    """
    zs = [composition[compound] for compound in provider.compounds]
    try:
        flash = provider.flasher.flash(P=pressure_Pa, VF=vapor_fraction, zs=zs)
        temperature = _safe_float(getattr(flash, "T", None))
        if temperature is not None and temperature > 0:
            return temperature
    except Exception as exc:
        warnings.append(
            f"Could not estimate {product_name} product temperature from "
            f"P/VF flash: {exc}. Column treated as infeasible."
        )
    return None


def _condenser_feasible(
    provider: ThermoFlashProvider,
    pressure_Pa: float,
    condenser_temperature_K: float,
    composition: dict[str, float],
    min_fraction: float = 0.05,
) -> str | None:
    """Return an error string if any significant distillate component cannot be condensed.

    Applies two thermodynamic checks sourced from provider.constants
    (thermo ChemicalConstantsPackage — no hardcoded values):

    1. P < P_triple  →  no liquid-vapour coexistence line exists at this pressure.
       The component can only be solid or vapour; a total condenser is physically
       impossible.  PR-EOS returns a fictitious liquid root here (no solid model),
       so the check is applied explicitly using library triple-point data.

    2. T_condenser < T_triple  →  even if pressure is above P_triple, the computed
       condenser temperature falls in the solid region; the component would
       freeze rather than condense.

    Falls back to a pure-component EOS flash for components whose triple-point
    data is absent from the library (catches edge cases where the EOS itself
    signals infeasibility through an exception).

    Args:
        provider: ThermoFlashProvider carrying the thermo constants.
        pressure_Pa: Column operating pressure (= feed stream pressure).
        condenser_temperature_K: Distillate bubble-point temperature.
        composition: Distillate mole-fraction dict.
        min_fraction: Ignore components below this mole fraction.

    Returns:
        None if all significant components are condensable; an error string
        identifying the first infeasible component otherwise.
    """
    Pts: list[float | None] = getattr(provider.constants, "Pts", None) or []
    Tts: list[float | None] = getattr(provider.constants, "Tts", None) or []

    for i, compound in enumerate(provider.compounds):
        fraction = composition.get(compound, 0.0)
        if fraction < min_fraction:
            continue

        p_triple = Pts[i] if i < len(Pts) else None
        t_triple = Tts[i] if i < len(Tts) else None

        # Check 1: pressure below triple-point pressure → no liquid phase
        if p_triple is not None and pressure_Pa < p_triple:
            return (
                f"Component '{compound}' (x={fraction:.3f}) has no liquid phase: "
                f"column P={pressure_Pa:.0f} Pa < P_triple={p_triple:.0f} Pa."
            )

        # Check 2: condenser temperature below triple-point temperature → solid forms
        if t_triple is not None and condenser_temperature_K < t_triple:
            return (
                f"Component '{compound}' (x={fraction:.3f}) would solidify: "
                f"condenser T={condenser_temperature_K:.1f} K < "
                f"T_triple={t_triple:.1f} K."
            )

        # Check 3: EOS fallback for components missing triple-point library data
        if p_triple is None and t_triple is None:
            pure_zs = [1.0 if c == compound else 0.0 for c in provider.compounds]
            try:
                flash = provider.flasher.flash(P=pressure_Pa, VF=0.0, zs=pure_zs)
                T = _safe_float(getattr(flash, "T", None))
                if T is None or T <= 0:
                    return (
                        f"Component '{compound}' (x={fraction:.3f}) has no valid "
                        f"liquid phase at {pressure_Pa:.0f} Pa."
                    )
            except Exception:
                return (
                    f"Component '{compound}' (x={fraction:.3f}) cannot be condensed "
                    f"at {pressure_Pa:.0f} Pa — pure-component flash failed."
                )
    return None


def _normalise_dict(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= _COMPOSITION_TOL:
        raise ValueError("composition sum is zero")
    return {compound: max(0.0, value) / total for compound, value in values.items()}


def _normalise_list(values: list[float]) -> list[float]:
    values = [max(0.0, value) for value in values]
    total = sum(values)
    if total <= _COMPOSITION_TOL:
        raise ValueError("phase composition sum is zero")
    return [value / total for value in values]


def _failed(stream_id: str, error_message: str) -> ShortcutDistillationResult:
    return ShortcutDistillationResult(
        success=False,
        inlet_stream_id=stream_id,
        error_message=error_message,
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
