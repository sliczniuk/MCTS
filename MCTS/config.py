"""Shared configuration for the leaf-estimator comparison study.

Import this module from any analysis notebook to get the system definition,
feed conditions, per-method MCTS configs, and experiment budget — without
repeating simulation setup in each notebook.

Usage (single-system, backward-compatible)
------------------------------------------
    from study_leaf_estimator.config import (
        COMPONENTS, N_COMPONENTS, PROVIDER,
        make_feed, FEED_CONDITIONS,
        METHOD_CONFIGS, METHOD_NAMES, CONDITION_NAMES,
        MCTS_ITERATIONS, SEEDS,
        SUCCESS_THRESHOLD, RESULTS_DIR,
        result_path, run_is_complete,
    )

Usage (multi-system)
--------------------
    from study_leaf_estimator.config import SYSTEMS, make_feed_for_system

    for sys_name, sys_def in SYSTEMS.items():
        provider = sys_def["provider"]
        for cond_name in sys_def["feed_conditions"]:
            feed = make_feed_for_system(sys_name, cond_name)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = next(
    p for p in [Path(__file__).parent.parent, *Path(__file__).parent.parent.parents]
    if (p / "ml").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import MCTSConfig, StreamState, build_pr_flasher

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Multi-system definitions
#
# Each system entry contains:
#   components      : list[str]   — thermo identifiers for build_pr_flasher
#   provider        : ThermoFlashProvider
#   feed_conditions : dict[str, dict]  — same schema as original FEED_CONDITIONS
#
# All BIPs verified from ChemSep PR database via build_pr_flasher / IPDB.
# Normal boiling points at 1 bar (K):
#   methane 111 | ethane 184 | propane 231 | isobutane 261 |
#   n-butane 273 | isopentane 301 | n-pentane 309 | n-hexane 342
#   carbon dioxide 195 (sublimation) | nitrogen 77
# ---------------------------------------------------------------------------
SYSTEMS: dict[str, dict] = {}

# ── System 1: original 5-component (CO2 / C2 / C3 / iC4 / nC4) ─────────────
# Validated baseline.  N_C-1 = 4 distillation columns for complete separation.
_comps_orig = ["carbon dioxide", "propane", "n-butane", "isobutane", "ethane"]
SYSTEMS["5comp_original"] = {
    "components": _comps_orig,
    "provider": build_pr_flasher(_comps_orig),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.20, "propane": 0.20, "n-butane": 0.20,
                "isobutane": 0.20, "ethane": 0.20,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 100_000.0,
            "molar_flow_mols": 1.0,
        },
        "light_rich": {
            "composition": {
                "carbon dioxide": 0.30, "ethane": 0.25, "propane": 0.20,
                "isobutane": 0.15, "n-butane": 0.10,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 100_000.0,
            "molar_flow_mols": 1.0,
        },
        "heavy_rich": {
            "composition": {
                "carbon dioxide": 0.10, "ethane": 0.15, "propane": 0.20,
                "isobutane": 0.25, "n-butane": 0.30,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 100_000.0,
            "molar_flow_mols": 1.0,
        },
        "low_pressure": {
            "composition": {
                "carbon dioxide": 0.20, "propane": 0.20, "n-butane": 0.20,
                "isobutane": 0.20, "ethane": 0.20,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 50_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "carbon dioxide": 0.20, "propane": 0.20, "n-butane": 0.20,
                "isobutane": 0.20, "ethane": 0.20,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 200_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 2: 5-component NGL (C2 / C3 / nC4 / iC4 / nC5) ──────────────────
# Classical natural-gas-liquid fractionation feed.  No CO2.
# N_C-1 = 4 distillation columns.  Heavier than original (nC5 replaces CO2).
# All BIPs in ChemSep PR database.
_comps_ngl5 = ["ethane", "propane", "n-butane", "isobutane", "n-pentane"]
SYSTEMS["5comp_NGL"] = {
    "components": _comps_ngl5,
    "provider": build_pr_flasher(_comps_ngl5),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "ethane": 0.20, "propane": 0.20, "n-butane": 0.20,
                "isobutane": 0.20, "n-pentane": 0.20,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 200_000.0,   # 2 bar — keeps nC5 mostly liquid
            "molar_flow_mols": 1.0,
        },
        "ethane_rich": {
            "composition": {
                "ethane": 0.40, "propane": 0.25, "n-butane": 0.15,
                "isobutane": 0.12, "n-pentane": 0.08,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 200_000.0,
            "molar_flow_mols": 1.0,
        },
        "pentane_rich": {
            "composition": {
                "ethane": 0.08, "propane": 0.12, "n-butane": 0.20,
                "isobutane": 0.20, "n-pentane": 0.40,
            },
            "temperature_K": 320.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "ethane": 0.20, "propane": 0.20, "n-butane": 0.20,
                "isobutane": 0.20, "n-pentane": 0.20,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 500_000.0,   # 5 bar
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 3: 4-component CO2 + light hydrocarbons (CO2 / C2 / C3 / nC4) ───
# CO2 capture / acid-gas removal context.  N_C-1 = 3 distillation columns.
# High CO2 fraction makes the separation thermodynamically distinct from NGL.
_comps_co2 = ["carbon dioxide", "ethane", "propane", "n-butane"]
SYSTEMS["4comp_CO2"] = {
    "components": _comps_co2,
    "provider": build_pr_flasher(_comps_co2),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.25, "ethane": 0.25,
                "propane": 0.25, "n-butane": 0.25,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 150_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO2_rich": {
            "composition": {
                "carbon dioxide": 0.55, "ethane": 0.20,
                "propane": 0.15, "n-butane": 0.10,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 150_000.0,
            "molar_flow_mols": 1.0,
        },
        "hydrocarbon_rich": {
            "composition": {
                "carbon dioxide": 0.10, "ethane": 0.25,
                "propane": 0.35, "n-butane": 0.30,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 150_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "carbon dioxide": 0.25, "ethane": 0.25,
                "propane": 0.25, "n-butane": 0.25,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 500_000.0,   # 5 bar — CO2 remains supercritical-adjacent
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 4: 3-component (C2 / C3 / nC4) ────────────────────────────────────
# Simplest hydrocarbon separation.  N_C-1 = 2 distillation columns.
# Useful for debugging the MCTS policy and as a low-complexity training system.
_comps_3 = ["ethane", "propane", "n-butane"]
SYSTEMS["3comp_C2C4"] = {
    "components": _comps_3,
    "provider": build_pr_flasher(_comps_3),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "ethane": 1/3, "propane": 1/3, "n-butane": 1/3,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 100_000.0,
            "molar_flow_mols": 1.0,
        },
        "ethane_rich": {
            "composition": {
                "ethane": 0.60, "propane": 0.25, "n-butane": 0.15,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 100_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "ethane": 1/3, "propane": 1/3, "n-butane": 1/3,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 400_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}


# ── System 5: 4-component aromatics / BTX (benzene / toluene / o-xylene / p-xylene)
# Petrochemical reformate fractionation.  N_C-1 = 3 distillation columns.
# BPs at 1 bar: benzene 353 K | toluene 384 K | p-xylene 411 K | o-xylene 417 K.
# o-/p-xylene differ by only 6 K — the classic difficult aromatic split.
# All BIPs in ChemSep PR database.  PR EOS suitable (non-polar aromatics).
_comps_btx = ["benzene", "toluene", "o-xylene", "p-xylene"]
SYSTEMS["4comp_BTX"] = {
    "components": _comps_btx,
    "provider": build_pr_flasher(_comps_btx),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "benzene": 0.25, "toluene": 0.25,
                "o-xylene": 0.25, "p-xylene": 0.25,
            },
            "temperature_K": 400.0,    # above benzene/toluene BPs; xylenes still liquid
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "benzene_rich": {
            "composition": {
                "benzene": 0.50, "toluene": 0.30,
                "o-xylene": 0.10, "p-xylene": 0.10,
            },
            "temperature_K": 390.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "xylene_rich": {
            "composition": {
                "benzene": 0.10, "toluene": 0.15,
                "o-xylene": 0.40, "p-xylene": 0.35,
            },
            "temperature_K": 420.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "benzene": 0.25, "toluene": 0.25,
                "o-xylene": 0.25, "p-xylene": 0.25,
            },
            "temperature_K": 400.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 6: 4-component olefins / paraffins (ethylene / ethane / propylene / propane)
# C2-C3 splitter train — critical in steam cracker / refinery.
# N_C-1 = 3 distillation columns.
# BPs at 1 bar: ethylene 169 K | ethane 184 K | propylene 225 K | propane 231 K.
# High pressure required to keep feed partially condensed at 280 K.
_comps_olef = ["ethylene", "ethane", "propylene", "propane"]
SYSTEMS["4comp_olefins"] = {
    "components": _comps_olef,
    "provider": build_pr_flasher(_comps_olef),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "ethylene": 0.25, "ethane": 0.25,
                "propylene": 0.25, "propane": 0.25,
            },
            "temperature_K": 280.0,
            "pressure_Pa": 1_000_000.0,   # 10 bar — keeps C3 stream condensed
            "molar_flow_mols": 1.0,
        },
        "ethylene_rich": {
            "composition": {
                "ethylene": 0.45, "ethane": 0.25,
                "propylene": 0.20, "propane": 0.10,
            },
            "temperature_K": 270.0,
            "pressure_Pa": 1_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "propylene_rich": {
            "composition": {
                "ethylene": 0.10, "ethane": 0.15,
                "propylene": 0.40, "propane": 0.35,
            },
            "temperature_K": 290.0,
            "pressure_Pa": 1_500_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "ethylene": 0.25, "ethane": 0.25,
                "propylene": 0.25, "propane": 0.25,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 2_000_000.0,   # 20 bar
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 7: 4-component C5-C8 alkanes (nC5 / nC6 / nC7 / nC8)
# Gasoline-fraction / naphtha separation.  N_C-1 = 3 distillation columns.
# BPs: nC5 309 K | nC6 342 K | nC7 371 K | nC8 399 K — all liquid at room T.
# Purely non-polar alkanes; PR EOS highly accurate.
_comps_c5c8 = ["n-pentane", "n-hexane", "n-heptane", "n-octane"]
SYSTEMS["4comp_C5C8"] = {
    "components": _comps_c5c8,
    "provider": build_pr_flasher(_comps_c5c8),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "n-pentane": 0.25, "n-hexane": 0.25,
                "n-heptane": 0.25, "n-octane": 0.25,
            },
            "temperature_K": 360.0,    # between nC6 and nC7 BPs → two-phase feed
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "light_rich": {
            "composition": {
                "n-pentane": 0.45, "n-hexane": 0.30,
                "n-heptane": 0.15, "n-octane": 0.10,
            },
            "temperature_K": 340.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "heavy_rich": {
            "composition": {
                "n-pentane": 0.10, "n-hexane": 0.15,
                "n-heptane": 0.30, "n-octane": 0.45,
            },
            "temperature_K": 390.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "n-pentane": 0.25, "n-hexane": 0.25,
                "n-heptane": 0.25, "n-octane": 0.25,
            },
            "temperature_K": 360.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 8: 3-component solvents (n-hexane / benzene / cyclohexane)
# Classic solvent-recovery / aromatic-aliphatic separation.
# N_C-1 = 2 distillation columns.
# BPs: hexane 342 K | benzene 353 K | cyclohexane 354 K.
# Benzene and cyclohexane differ by < 1 K — nearly azeotropic at low pressure;
# this makes it a hard separation that stresses compressor/valve actions.
_comps_solv = ["n-hexane", "benzene", "cyclohexane"]
SYSTEMS["3comp_solvents"] = {
    "components": _comps_solv,
    "provider": build_pr_flasher(_comps_solv),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "n-hexane": 1/3, "benzene": 1/3, "cyclohexane": 1/3,
            },
            "temperature_K": 360.0,    # slightly above all BPs → mostly vapour feed
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "hexane_rich": {
            "composition": {
                "n-hexane": 0.60, "benzene": 0.20, "cyclohexane": 0.20,
            },
            "temperature_K": 355.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "benzene_rich": {
            "composition": {
                "n-hexane": 0.20, "benzene": 0.55, "cyclohexane": 0.25,
            },
            "temperature_K": 360.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "n-hexane": 1/3, "benzene": 1/3, "cyclohexane": 1/3,
            },
            "temperature_K": 360.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}


# ── System 9: 3-component chlorinated solvents (CHCl3 / CCl4 / DCM)
# Fine chemistry: halogenated solvents for organic extraction and purification.
# N_C-1 = 2 distillation columns.
# BPs at 1 bar: DCM 313 K | CHCl3 334 K | CCl4 350 K — good volatility spread.
# All three are structurally similar chlorinated methanes; kij≈0 is physically
# justified (non-polar, similar polarity and molecular volume). PR EOS accurate.
# ChemSep PR BIPs are zero for all pairs but the approximation error is small
# (literature kij for CCl4/CHCl3 ≈ 0.003, CHCl3/DCM ≈ 0.002).
_comps_chl = ["methylene chloride", "chloroform", "carbon tetrachloride"]
SYSTEMS["3comp_chlorinated"] = {
    "components": _comps_chl,
    "provider": build_pr_flasher(_comps_chl),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "methylene chloride": 1/3, "chloroform": 1/3,
                "carbon tetrachloride": 1/3,
            },
            "temperature_K": 335.0,    # between CHCl3 and CCl4 BPs
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "DCM_rich": {
            "composition": {
                "methylene chloride": 0.60, "chloroform": 0.25,
                "carbon tetrachloride": 0.15,
            },
            "temperature_K": 325.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "CCl4_rich": {
            "composition": {
                "methylene chloride": 0.15, "chloroform": 0.25,
                "carbon tetrachloride": 0.60,
            },
            "temperature_K": 345.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "methylene chloride": 1/3, "chloroform": 1/3,
                "carbon tetrachloride": 1/3,
            },
            "temperature_K": 335.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# ── System 10: 3-component chromatography / extraction solvents (EtOAc / nC6 / nC7)
# Fine chemistry: solvent-gradient purification; ethyl acetate fractions eluted
# with hexane/heptane mixtures are the most common normal-phase separation solvent
# system in pharmaceutical synthesis workup.  N_C-1 = 2 distillation columns.
# BPs at 1 bar: nC6 342 K | EtOAc 350 K | nC7 371 K.
# Ethyl acetate and n-hexane differ by only 8 K — a challenging split that
# drives compressor and valve exploration in the MCTS tree.
# Missing BIPs: EtOAc/nC6 and EtOAc/nC7 (literature kij ≈ 0.01-0.02 — minor
# non-polarity correction; error in predicted α is < 5 %).
_comps_ext = ["n-hexane", "ethyl acetate", "n-heptane"]
SYSTEMS["3comp_extraction"] = {
    "components": _comps_ext,
    "provider": build_pr_flasher(_comps_ext),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "n-hexane": 1/3, "ethyl acetate": 1/3, "n-heptane": 1/3,
            },
            "temperature_K": 355.0,    # between EtOAc and nC7 BPs — two-phase feed
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "hexane_rich": {
            "composition": {
                "n-hexane": 0.60, "ethyl acetate": 0.25, "n-heptane": 0.15,
            },
            "temperature_K": 348.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "EtOAc_rich": {
            "composition": {
                "n-hexane": 0.20, "ethyl acetate": 0.60, "n-heptane": 0.20,
            },
            "temperature_K": 355.0,
            "pressure_Pa": 101_325.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "n-hexane": 1/3, "ethyl acetate": 1/3, "n-heptane": 1/3,
            },
            "temperature_K": 355.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# -- System 11: 4-component syngas with inert nitrogen (H2 / CO / CH4 / N2)
# Hydrogen recovery, purge-gas cleanup, and reformer recycle context. This set
# has complete non-zero ChemSep PR kij coverage.
_comps_syngas = ["hydrogen", "carbon monoxide", "methane", "nitrogen"]
SYSTEMS["4comp_syngas_inerts"] = {
    "components": _comps_syngas,
    "provider": build_pr_flasher(_comps_syngas),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "hydrogen": 0.45, "carbon monoxide": 0.25,
                "methane": 0.20, "nitrogen": 0.10,
            },
            "temperature_K": 220.0,
            "pressure_Pa": 3_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "hydrogen_rich": {
            "composition": {
                "hydrogen": 0.70, "carbon monoxide": 0.12,
                "methane": 0.10, "nitrogen": 0.08,
            },
            "temperature_K": 210.0,
            "pressure_Pa": 3_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO_rich": {
            "composition": {
                "hydrogen": 0.30, "carbon monoxide": 0.45,
                "methane": 0.15, "nitrogen": 0.10,
            },
            "temperature_K": 220.0,
            "pressure_Pa": 2_500_000.0,
            "molar_flow_mols": 1.0,
        },
        "methane_rich": {
            "composition": {
                "hydrogen": 0.20, "carbon monoxide": 0.15,
                "methane": 0.55, "nitrogen": 0.10,
            },
            "temperature_K": 240.0,
            "pressure_Pa": 4_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "nitrogen_diluted": {
            "composition": {
                "hydrogen": 0.35, "carbon monoxide": 0.20,
                "methane": 0.20, "nitrogen": 0.25,
            },
            "temperature_K": 230.0,
            "pressure_Pa": 1_000_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# -- System 12: 4-component CO2 + light alkanes (CO2 / C1 / C2 / C3)
# Acid-gas removal, natural-gas liquids, and rich-gas conditioning context.
# Complete non-zero ChemSep PR kij coverage.
_comps_co2_lights = ["carbon dioxide", "methane", "ethane", "propane"]
SYSTEMS["4comp_CO2_light_alkanes"] = {
    "components": _comps_co2_lights,
    "provider": build_pr_flasher(_comps_co2_lights),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.25, "methane": 0.25,
                "ethane": 0.25, "propane": 0.25,
            },
            "temperature_K": 260.0,
            "pressure_Pa": 2_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO2_rich": {
            "composition": {
                "carbon dioxide": 0.50, "methane": 0.20,
                "ethane": 0.15, "propane": 0.15,
            },
            "temperature_K": 260.0,
            "pressure_Pa": 2_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "methane_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methane": 0.55,
                "ethane": 0.20, "propane": 0.10,
            },
            "temperature_K": 240.0,
            "pressure_Pa": 3_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "propane_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methane": 0.15,
                "ethane": 0.25, "propane": 0.45,
            },
            "temperature_K": 300.0,
            "pressure_Pa": 1_500_000.0,
            "molar_flow_mols": 1.0,
        },
        "low_pressure": {
            "composition": {
                "carbon dioxide": 0.25, "methane": 0.25,
                "ethane": 0.25, "propane": 0.25,
            },
            "temperature_K": 280.0,
            "pressure_Pa": 500_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# -- System 13: 4-component CO2 / light olefin recovery (CO2 / C1 / C2 / C2=)
# Cracked-gas cleanup and ethylene recovery context. Complete non-zero ChemSep
# PR kij coverage, unlike the common ethylene/propylene/paraffin set.
_comps_co2_olefin = ["carbon dioxide", "methane", "ethane", "ethylene"]
SYSTEMS["4comp_CO2_olefin_recovery"] = {
    "components": _comps_co2_olefin,
    "provider": build_pr_flasher(_comps_co2_olefin),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.25, "methane": 0.25,
                "ethane": 0.25, "ethylene": 0.25,
            },
            "temperature_K": 240.0,
            "pressure_Pa": 2_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "ethylene_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methane": 0.15,
                "ethane": 0.20, "ethylene": 0.50,
            },
            "temperature_K": 230.0,
            "pressure_Pa": 2_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO2_rich": {
            "composition": {
                "carbon dioxide": 0.50, "methane": 0.20,
                "ethane": 0.15, "ethylene": 0.15,
            },
            "temperature_K": 250.0,
            "pressure_Pa": 3_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "methane_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methane": 0.55,
                "ethane": 0.15, "ethylene": 0.15,
            },
            "temperature_K": 230.0,
            "pressure_Pa": 1_500_000.0,
            "molar_flow_mols": 1.0,
        },
        "high_pressure": {
            "composition": {
                "carbon dioxide": 0.25, "methane": 0.25,
                "ethane": 0.25, "ethylene": 0.25,
            },
            "temperature_K": 260.0,
            "pressure_Pa": 5_000_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# -- System 14: 4-component CO2 aromatic/solvent extraction
# Supercritical CO2 extraction, naphtha dearomatisation, and solvent recovery
# context. Complete non-zero ChemSep PR kij coverage.
_comps_co2_arom = ["carbon dioxide", "n-pentane", "benzene", "cyclohexane"]
SYSTEMS["4comp_CO2_aromatic_solvent"] = {
    "components": _comps_co2_arom,
    "provider": build_pr_flasher(_comps_co2_arom),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.20, "n-pentane": 0.30,
                "benzene": 0.25, "cyclohexane": 0.25,
            },
            "temperature_K": 340.0,
            "pressure_Pa": 1_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO2_rich": {
            "composition": {
                "carbon dioxide": 0.55, "n-pentane": 0.15,
                "benzene": 0.15, "cyclohexane": 0.15,
            },
            "temperature_K": 320.0,
            "pressure_Pa": 4_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "pentane_rich": {
            "composition": {
                "carbon dioxide": 0.10, "n-pentane": 0.60,
                "benzene": 0.15, "cyclohexane": 0.15,
            },
            "temperature_K": 330.0,
            "pressure_Pa": 500_000.0,
            "molar_flow_mols": 1.0,
        },
        "benzene_rich": {
            "composition": {
                "carbon dioxide": 0.10, "n-pentane": 0.20,
                "benzene": 0.55, "cyclohexane": 0.15,
            },
            "temperature_K": 360.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
        "cyclohexane_rich": {
            "composition": {
                "carbon dioxide": 0.10, "n-pentane": 0.20,
                "benzene": 0.15, "cyclohexane": 0.55,
            },
            "temperature_K": 360.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# -- System 15: 3-component CO2 / methanol / water
# Methanol synthesis effluent, CO2 capture solvent regeneration, and polar
# oxygenate cleanup context. Complete non-zero ChemSep PR kij coverage.
_comps_co2_meoh_water = ["carbon dioxide", "methanol", "water"]
SYSTEMS["3comp_CO2_methanol_water"] = {
    "components": _comps_co2_meoh_water,
    "provider": build_pr_flasher(_comps_co2_meoh_water),
    "feed_conditions": {
        "nominal": {
            "composition": {
                "carbon dioxide": 0.30, "methanol": 0.45, "water": 0.25,
            },
            "temperature_K": 330.0,
            "pressure_Pa": 1_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "CO2_rich": {
            "composition": {
                "carbon dioxide": 0.65, "methanol": 0.25, "water": 0.10,
            },
            "temperature_K": 310.0,
            "pressure_Pa": 2_000_000.0,
            "molar_flow_mols": 1.0,
        },
        "methanol_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methanol": 0.70, "water": 0.15,
            },
            "temperature_K": 340.0,
            "pressure_Pa": 500_000.0,
            "molar_flow_mols": 1.0,
        },
        "water_rich": {
            "composition": {
                "carbon dioxide": 0.15, "methanol": 0.25, "water": 0.60,
            },
            "temperature_K": 370.0,
            "pressure_Pa": 300_000.0,
            "molar_flow_mols": 1.0,
        },
        "low_pressure": {
            "composition": {
                "carbon dioxide": 0.30, "methanol": 0.45, "water": 0.25,
            },
            "temperature_K": 330.0,
            "pressure_Pa": 150_000.0,
            "molar_flow_mols": 1.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible aliases (original 5-component system)
# ---------------------------------------------------------------------------
COMPONENTS = SYSTEMS["5comp_original"]["components"]
N_COMPONENTS = len(COMPONENTS)
PROVIDER = SYSTEMS["5comp_original"]["provider"]
FEED_CONDITIONS = SYSTEMS["5comp_original"]["feed_conditions"]


def make_feed(condition_name: str) -> StreamState:
    """Return a StreamState for the original 5-component system.

    Args:
        condition_name: Key in FEED_CONDITIONS.

    Returns:
        StreamState ready for mcts_search().

    Example:
        feed = make_feed("nominal")
    """
    return make_feed_for_system("5comp_original", condition_name)


def make_feed_for_system(system_name: str, condition_name: str) -> StreamState:
    """Return a StreamState for the named system and feed condition.

    Args:
        system_name: Key in SYSTEMS.
        condition_name: Key in SYSTEMS[system_name]["feed_conditions"].

    Returns:
        StreamState ready for mcts_search().

    Example:
        feed = make_feed_for_system("5comp_NGL", "ethane_rich")
    """
    sys_def = SYSTEMS[system_name]
    c = sys_def["feed_conditions"][condition_name]
    return StreamState(
        id="Feed",
        temperature_K=c["temperature_K"],
        pressure_Pa=c["pressure_Pa"],
        molar_flow_mols=c["molar_flow_mols"],
        composition=c["composition"],
    )


SYSTEM_NAMES: list[str] = list(SYSTEMS.keys())
CONDITION_NAMES: list[str] = list(FEED_CONDITIONS.keys())   # original system


# ---------------------------------------------------------------------------
# Base MCTS configuration
#
# Identical to test_ml_5comp.ipynb except:
#   - adjacent key-pair mode (N_C-1 = 4 pairs per stream; baseline used "all")
#   - all penalties zeroed (duty and stages tracked separately for analysis)
#   - energy balance enabled (include_reboiler_duty=True)
#   - use_leaf_value_estimator=False  ->  full-rollout baseline
# ---------------------------------------------------------------------------
_BASE_CONFIG = MCTSConfig(
    # -- Objective -----------------------------------------------------------
    objective_mode="complete_separation",
    separation_score_mode="mutual_information_equal_weight",
    min_component_fraction=1e-8,
    product_purity_threshold=0.9,

    # -- Distillation --------------------------------------------------------
    enable_distillation_actions=True,
    distillation_light_key_recoveries=(0.98,),
    distillation_heavy_key_recoveries=(0.02,),
    distillation_reflux_multipliers=(1.3,),
    distillation_key_pair_mode="adjacent",
    distillation_min_alpha_ratio=1.05,
    distillation_max_theoretical_stages=100.0,
    distillation_max_reflux_ratio=50.0,
    max_total_distillation_count=N_COMPONENTS - 1,
    max_distillation_count_per_path=10,
    distillation_molar_heat_of_vaporization_J_mol=0.0,
    include_reboiler_duty=True,
    distillation_min_key_flow_mols=1e-8,
    validate_distillation_candidates=True,

    # -- Flash ---------------------------------------------------------------
    max_flash_count_per_path=3,
    require_flash_liquid_product=True,

    # -- HX -----------------------------------------------------------------
    allowed_delta_T_K=(-60.0, -40.0, -20.0, +20.0, +40.0, 60.0),
    hx_target_states=("bubble_point", "dew_point"),

    # -- Valve --------------------------------------------------------------
    allowed_valve_pressure_ratios=(0.1, 0.25, 0.5, 0.75),
    valve_target_states=(),

    # -- Compressor / Pump --------------------------------------------------
    allowed_compression_ratios=(2.0, 5.0, 10.0),
    allowed_pump_pressure_ratios=(2.0, 5.0, 10.0),

    # -- Pressure / temperature bounds --------------------------------------
    min_pressure_Pa=100_000.0,
    max_pressure_Pa=5_000_000.0,
    min_temperature_K=150.0,
    max_temperature_K=450.0,
    min_flow_mols=0.01,

    # -- Search structure ---------------------------------------------------
    max_depth=10,
    unit_penalty=0.01,
    duty_penalty_per_W=1e-6,
    stage_count_penalty_per_stage=0.001,

    # -- UCT ----------------------------------------------------------------
    exploration_weight=1.41,

    # -- Caching ------------------------------------------------------------
    enable_apply_action_cache=True,
    enable_action_generation_cache=True,

    # -- Recycle ------------------------------------------------------------
    enable_recycle_actions=False,

    # -- Leaf estimator -----------------------------------------------------
    use_leaf_value_estimator=False,
)

# ---------------------------------------------------------------------------
# Per-method configurations
# ---------------------------------------------------------------------------
METHOD_CONFIGS: dict[str, MCTSConfig] = {
    "full_rollout": replace(_BASE_CONFIG, exploration_weight=3.0)
}

METHOD_NAMES: list[str] = list(METHOD_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Experiment budget
# ---------------------------------------------------------------------------
MCTS_ITERATIONS: int = 1500
SEEDS: list[int] = list(range(3,6))
SUCCESS_THRESHOLD: float = 0.90

# ---------------------------------------------------------------------------
# Results file helpers
# ---------------------------------------------------------------------------

def result_path(
    method: str,
    condition: str,
    seed: int,
    system: str = "5comp_original",
) -> Path:
    """Return the JSON path for a single experiment run.

    Args:
        method: Key in METHOD_CONFIGS.
        condition: Key in FEED_CONDITIONS (or the system's feed_conditions).
        seed: Integer seed (0-indexed).
        system: Key in SYSTEMS. Defaults to "5comp_original" for
            backward compatibility (original files have no system prefix).

    Returns:
        Path of the form results/{method}__{condition}__{seed:04d}.json
        or results/{system}__{method}__{condition}__{seed:04d}.json for
        non-original systems.
    """
    if system == "5comp_original":
        return RESULTS_DIR / f"{method}__{condition}__{seed:04d}.json"
    return RESULTS_DIR / f"{system}__{method}__{condition}__{seed:04d}.json"


def run_is_complete(
    method: str,
    condition: str,
    seed: int,
    system: str = "5comp_original",
) -> bool:
    """Return True if the result file for this run already exists."""
    return result_path(method, condition, seed, system).exists()


def pending_runs(
    methods: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    systems: list[str] | None = None,
) -> list[tuple[str, str, str, int]]:
    """Return list of (system, method, condition, seed) tuples not yet completed.

    Args:
        methods: Subset of METHOD_NAMES to consider. None = all.
        conditions: Subset of condition names. None = all conditions per system.
        seeds: Subset of SEEDS. None = all.
        systems: Subset of SYSTEM_NAMES. None = all.

    Returns:
        List of (system, method, condition, seed) triples in consistent order.

    Example (original system only, backward-compatible):
        runs = pending_runs(systems=["5comp_original"])
    """
    _methods = methods or METHOD_NAMES
    _seeds = seeds or SEEDS
    _systems = systems or SYSTEM_NAMES
    result = []
    for sys_name in _systems:
        sys_conditions = conditions or list(SYSTEMS[sys_name]["feed_conditions"].keys())
        for m in _methods:
            for cond in sys_conditions:
                for s in _seeds:
                    if not run_is_complete(m, cond, s, sys_name):
                        result.append((sys_name, m, cond, s))
    return result
