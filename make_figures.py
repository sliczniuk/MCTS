"""
Generate figures for Report 1 (MCTS-Based Flowsheet Synthesis).

Run from the project root:
    python reports/report_01/make_figures.py

Outputs written to:  reports/report_01/figures/
"""

import json
import glob
import shutil
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent  # project root
RESULTS   = ROOT / "study_leaf_estimator" / "results"
FIGS_OUT  = pathlib.Path(__file__).resolve().parent / "figures"
FIGS_OUT.mkdir(exist_ok=True)

CSV_PATH  = RESULTS / "full_rollout_summary.csv"
TOPO_SRC  = ROOT / "study_leaf_estimator" / "00_flowsheet_full_rollout.png"
TOPO_DST  = FIGS_OUT / "fig_topology.png"

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "lines.linewidth":  1.2,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

CONDITION_COLORS = {
    "nominal":       "#2166ac",
    "light_rich":    "#4dac26",
    "heavy_rich":    "#d01c8b",
    "low_pressure":  "#f1a340",
    "high_pressure": "#7b3294",
}
CONDITION_LABELS = {
    "nominal":       "Nominal",
    "light_rich":    "Light-rich",
    "heavy_rich":    "Heavy-rich",
    "low_pressure":  "Low pressure (0.5 bar)",
    "high_pressure": "High pressure (2 bar)",
}
SEED_LINESTYLES = ["solid", "dashed", "dotted"]

# ── helper ────────────────────────────────────────────────────────────────────
def load_5comp_jsons():
    """Load all 5comp_original JSON files (those without a system prefix)."""
    pattern = str(RESULTS / "full_rollout__*__000*.json")
    files = glob.glob(pattern)
    records = []
    for path in sorted(files):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("system") != "5comp_original":
            continue
        prog = pd.DataFrame(d["progress"])
        prog["condition"] = d["condition"]
        prog["seed"]      = d["seed"]
        prog["success"]   = d["success"]
        records.append(prog)
    if not records:
        raise RuntimeError(f"No 5comp JSON files found under {RESULTS}")
    return pd.concat(records, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Convergence curves for 5comp_original
# ═══════════════════════════════════════════════════════════════════════════════
def fig_convergence():
    data = load_5comp_jsons()
    conditions = [
        "nominal", "light_rich", "heavy_rich", "low_pressure", "high_pressure"
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for cond in conditions:
        cdata = data[data["condition"] == cond]
        seeds = sorted(cdata["seed"].unique())
        color = CONDITION_COLORS.get(cond, "gray")
        for i, seed in enumerate(seeds):
            sdata = cdata[cdata["seed"] == seed].sort_values("iteration")
            ls = SEED_LINESTYLES[i % len(SEED_LINESTYLES)]
            label = CONDITION_LABELS.get(cond, cond) if i == 0 else None
            ax.plot(
                sdata["iteration"], sdata["fraction_of_target"],
                color=color, linestyle=ls, alpha=0.85, label=label,
            )

    # success threshold line
    ax.axhline(0.90, color="black", linewidth=1.0, linestyle="--",
               label="Success threshold (0.90)")

    ax.set_xlabel("MCTS iteration")
    ax.set_ylabel(r"Fraction of target $S_\mathrm{norm}$")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.05, 0.1))

    # legend: condition patches + linestyle note
    handles, labels = ax.get_legend_handles_labels()
    # add linestyle legend entries for seeds
    seed_handles = [
        mlines.Line2D([], [], color="gray", linestyle=ls,
                      label=f"Seed {i+1}")
        for i, ls in enumerate(SEED_LINESTYLES)
    ]
    ax.legend(
        handles + seed_handles,
        labels + [h.get_label() for h in seed_handles],
        loc="lower right", ncol=2, framealpha=0.9,
    )

    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_title(
        r"5-component system: $S_\mathrm{norm}$ vs.\ MCTS iteration"
        "\n(pilot: 3 seeds per condition, 1 500 iterations)"
    )

    out = FIGS_OUT / "fig_convergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Success-rate bar chart across all 9 systems
# ═══════════════════════════════════════════════════════════════════════════════
def fig_success_rates():
    df = pd.read_csv(CSV_PATH)
    df["system"] = df["system"].fillna("5comp_original")

    stats = (
        df.groupby("system")
          .agg(success_rate=("success", "mean"),
               mean_fot=("fraction_of_target", "mean"))
          .reset_index()
    )

    SYSTEM_LABELS = {
        "5comp_original":            r"5-comp (CO$_2$/C$_2$-C$_4$)",
        "3comp_CO2_methanol_water":  r"3-comp (CO$_2$/MeOH/H$_2$O)",
        "4comp_CO2_light_alkanes":   r"4-comp (CO$_2$/C$_1$-C$_3$)",
        "4comp_CO2_olefin_recovery": r"4-comp (CO$_2$/olefins)",
        "4comp_CO2":                 r"4-comp (CO$_2$/hydrocarbons)",
        "3comp_C2C4":                r"3-comp (C$_2$/C$_4$)",
        "4comp_CO2_aromatic_solvent":r"4-comp (CO$_2$/aromatics)",
        "3comp_solvents":            r"3-comp (solvents)",
        "4comp_syngas_inerts":       r"4-comp (syngas)",
    }

    # sort by success rate descending, then mean_fot
    stats = stats.sort_values(
        ["success_rate", "mean_fot"], ascending=[False, False]
    ).reset_index(drop=True)

    sys_labels = [SYSTEM_LABELS.get(s, s) for s in stats["system"]]
    x = np.arange(len(stats))

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars_sr  = ax.bar(x - 0.2, stats["success_rate"] * 100, 0.38,
                      label="Success rate (%)", color="#2166ac", alpha=0.85)
    bars_fot = ax.bar(x + 0.2, stats["mean_fot"] * 100, 0.38,
                      label=r"Mean $S_\mathrm{norm}$ (× 100)", color="#d7191c",
                      alpha=0.70)

    ax.axhline(90, color="black", linewidth=0.8, linestyle="--",
               label="90 % threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(sys_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(
        "Success rate and mean separation score by chemical system\n"
        "(pilot: 3 seeds per condition, 1 500 MCTS iterations)"
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    out = FIGS_OUT / "fig_success_rates.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Thermo evaluations to success (box / strip plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_thermo_cost():
    df = pd.read_csv(CSV_PATH)
    df["system"] = df["system"].fillna("5comp_original")
    success = df[df["success"]].copy()

    SYSTEM_SHORT = {
        "5comp_original":            "5-comp",
        "3comp_CO2_methanol_water":  "3-comp\n(CO$_2$/MeOH)",
        "4comp_CO2_light_alkanes":   "4-comp\n(light alk.)",
        "4comp_CO2_olefin_recovery": "4-comp\n(olefins)",
    }

    systems = [s for s in SYSTEM_SHORT if s in success["system"].values]
    if not systems:
        print("No successful runs found — skipping thermo-cost figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))

    for ax, col, ylabel in zip(
        axes,
        ["thermo_at_success", "iter_to_success"],
        ["Thermo evaluations to success", "MCTS iterations to success"],
    ):
        data_by_sys = [
            success[success["system"] == s][col].dropna().values
            for s in systems
        ]
        labels = [SYSTEM_SHORT[s] for s in systems]
        bp = ax.boxplot(data_by_sys, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        colors = ["#2166ac", "#4dac26", "#f1a340", "#d01c8b"]
        for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle(
        "Computational cost for successful runs\n"
        "(pilot scale — few data points per system)"
    )
    fig.tight_layout()
    out = FIGS_OUT / "fig_thermo_cost.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — FOT distribution for all 9 systems (box plots)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fot_distribution():
    df = pd.read_csv(CSV_PATH)
    df["system"] = df["system"].fillna("5comp_original")

    SYSTEM_LABELS = {
        "5comp_original":             "5-comp\n(NGL)",
        "3comp_CO2_methanol_water":   "3-comp\n(CO2/MeOH/H2O)",
        "4comp_CO2_light_alkanes":    "4-comp\n(CO2/alk.)",
        "4comp_CO2_olefin_recovery":  "4-comp\n(olefins)",
        "4comp_CO2":                  "4-comp\n(CO2)",
        "3comp_C2C4":                 "3-comp\n(C2/C4)",
        "4comp_CO2_aromatic_solvent": "4-comp\n(aromatics)",
        "3comp_solvents":             "3-comp\n(solvents)",
        "4comp_syngas_inerts":        "4-comp\n(syngas)",
    }
    # order: sort by median FOT descending
    order = (
        df.groupby("system")["fraction_of_target"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )

    data_by_sys  = [df[df["system"] == s]["fraction_of_target"].values for s in order]
    tick_labels  = [SYSTEM_LABELS.get(s, s) for s in order]
    success_rate = [df[df["system"] == s]["success"].mean() for s in order]

    # colour by success rate: green → yellow → red
    cmap = plt.cm.RdYlGn
    colors = [cmap(sr) for sr in success_rate]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bp = ax.boxplot(
        data_by_sys,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker="o", markersize=3, linestyle="none",
                        markerfacecolor="gray", alpha=0.6),
        widths=0.55,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # overlay individual data points
    for i, vals in enumerate(data_by_sys, start=1):
        ax.scatter(
            np.full_like(vals, i) + np.random.uniform(-0.12, 0.12, len(vals)),
            vals,
            s=16, color="black", alpha=0.45, zorder=3,
        )

    ax.axhline(0.90, color="green", linewidth=1.0, linestyle="--", alpha=0.8,
               label="Success threshold (0.90)")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_ylabel(r"Final $S_\mathrm{norm}$")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(
        "Distribution of final separation score $S_{\\mathrm{norm}}$ per system\n"
        "(pilot study — box colour: green = high success rate, red = low/zero)"
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    out = FIGS_OUT / "fig_fot_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Cumulative convergence CDF for 5comp_original
# ═══════════════════════════════════════════════════════════════════════════════
def fig_convergence_cdf():
    data = load_5comp_jsons()

    # For each run, determine the first iteration at which FOT >= 0.90
    threshold = 0.90
    conditions = [
        "nominal", "light_rich", "heavy_rich", "low_pressure", "high_pressure"
    ]
    iters = np.arange(0, 1510, 10)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for cond in conditions:
        cdata = data[data["condition"] == cond]
        seeds  = sorted(cdata["seed"].unique())
        n_seeds = len(seeds)
        frac_success = np.zeros(len(iters))
        for seed in seeds:
            sdata = cdata[cdata["seed"] == seed].sort_values("iteration")
            # cumulative max FOT up to each iteration
            cum_max = sdata["fraction_of_target"].cummax().values
            for k, t in enumerate(iters):
                idx = np.searchsorted(sdata["iteration"].values, t, side="right") - 1
                if idx >= 0 and cum_max[idx] >= threshold:
                    frac_success[k] += 1.0
        frac_success /= n_seeds
        ax.step(iters, frac_success * 100, where="post",
                color=CONDITION_COLORS.get(cond, "gray"),
                linewidth=1.5,
                label=CONDITION_LABELS.get(cond, cond))

    ax.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel("MCTS iteration")
    ax.set_ylabel("Fraction of seeds succeeded (%)")
    ax.set_xlim(0, 1500)
    ax.set_ylim(-5, 110)
    ax.set_title(
        r"Cumulative success rate vs.\ iteration — 5-component NGL system"
        "\n(pilot: 3 seeds per condition)"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(linewidth=0.4, alpha=0.5)

    out = FIGS_OUT / "fig_convergence_cdf.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Best flowsheet topology schematic (programmatic, readable)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_topology_schematic():
    """Draw a clean box-and-arrow schematic of the best nominal run topology."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    UNIT_COLORS = {
        "compressor": "#74c476",   # green
        "valve":      "#9ecae1",   # light blue
        "hx":         "#fdae6b",   # orange
        "distillation": "#6baed6", # blue
        "product":    "#f7f7f7",   # near-white
        "feed":       "#fee090",   # yellow
    }
    UNIT_ABBR = {
        "compressor": "COMP",
        "valve":      "VALVE",
        "hx":         "HX",
        "distillation": "DIST",
    }

    # Nodes: (x, y, kind, label_line1, label_line2, label_line3)
    # Main chain y = 2.0; lower branch y = 0.5; product streams further out
    nodes = {
        "feed": (0.0, 2.0, "feed",        "Feed",         "",               ""),
        "u01":  (1.2, 2.0, "compressor",  "COMP",         "×10",            "10 bar"),
        "u02":  (2.6, 2.0, "distillation","DIST",         "C3 | iC4",       ""),
        "u03":  (3.8, 0.6, "distillation","DIST",         "iC4 | nC4",      ""),
        "u04":  (3.8, 2.0, "valve",       "VALVE",        "×0.75",          ""),
        "u05":  (5.0, 2.0, "hx",          "HX",           "+20 K",          ""),
        "u06":  (6.2, 2.0, "compressor",  "COMP",         "×2",             ""),
        "u07":  (7.4, 2.0, "distillation","DIST",         "CO₂ | C₂",       ""),
        "u08":  (8.6, 2.0, "hx",          "HX",           "−20 K",          ""),
        "u09":  (9.8, 2.0, "distillation","DIST",         "C₂ | C₃",        ""),
        # products
        "p_ic4": (4.8, 0.9, "product",    "iC4",          "",               ""),
        "p_nc4": (4.8, 0.2, "product",    "nC4",          "",               ""),
        "p_co2": (7.4, 3.3, "product",    "CO₂",          "",               ""),
        "p_c2":  (11.0, 2.5, "product",   "C₂ (eth.)",    "",               ""),
        "p_c3":  (11.0, 1.5, "product",   "C₃ (prop.)",   "",               ""),
    }

    # Edges: (from, to, label)
    edges = [
        ("feed", "u01",  ""),
        ("u01",  "u02",  "10 bar"),
        ("u02",  "u04",  "dist."),        # main chain: distillate (C3-rich)
        ("u02",  "u03",  "bot."),         # branch: bottoms (iC4/nC4)
        ("u03",  "p_ic4",""),
        ("u03",  "p_nc4",""),
        ("u04",  "u05",  ""),
        ("u05",  "u06",  ""),
        ("u06",  "u07",  ""),
        ("u07",  "p_co2","dist."),        # CO2 distillate
        ("u07",  "u08",  "bot."),         # bottoms (C2/C3)
        ("u08",  "u09",  ""),
        ("u09",  "p_c2", "dist."),
        ("u09",  "p_c3", "bot."),
    ]

    W, H = 0.80, 0.42   # box width, height
    PROD_W, PROD_H = 0.70, 0.36

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.5, 11.8)
    ax.set_ylim(-0.3, 4.0)
    ax.axis("off")
    ax.set_aspect("equal")

    # Draw edges first (behind boxes)
    pos = {k: (v[0], v[1]) for k, v in nodes.items()}

    for src, dst, lbl in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        is_prod = dst.startswith("p_")
        # choose arrow start/end on box border
        bw = PROD_W if src.startswith("p_") else W
        bh = PROD_H if src.startswith("p_") else H
        # simple midpoint routing
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                lw=0.9,
                connectionstyle="arc3,rad=0.0" if abs(y1-y0) < 0.05
                    else ("arc3,rad=-0.3" if x1-x0 > 0 and y1 > y0 else "arc3,rad=0.0"),
                shrinkA=18, shrinkB=16,
            ),
        )
        if lbl:
            mx, my = (x0+x1)/2, (y0+y1)/2 + 0.14
            ax.text(mx, my, lbl, ha="center", va="bottom", fontsize=6.5,
                    color="#333333")

    # Draw unit-op boxes
    for nid, (x, y, kind, l1, l2, l3) in nodes.items():
        bw = PROD_W if nid.startswith("p_") else W
        bh = PROD_H if nid.startswith("p_") else H
        color = UNIT_COLORS.get(kind, "#eeeeee")
        rect = FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.03",
            linewidth=0.8,
            edgecolor="#333333",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)
        # text
        lines = [l for l in [l1, l2, l3] if l]
        n = len(lines)
        for j, line in enumerate(lines):
            dy = (n - 1) * 0.08 - j * 0.16
            ax.text(x, y + dy, line,
                    ha="center", va="center",
                    fontsize=7.5 if j == 0 else 6.5,
                    fontweight="bold" if j == 0 else "normal",
                    color="#111111", zorder=4)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=UNIT_COLORS["feed"],        label="Feed"),
        mpatches.Patch(facecolor=UNIT_COLORS["compressor"],  label="Compressor"),
        mpatches.Patch(facecolor=UNIT_COLORS["distillation"],label="Distillation"),
        mpatches.Patch(facecolor=UNIT_COLORS["hx"],          label="Heat exchanger"),
        mpatches.Patch(facecolor=UNIT_COLORS["valve"],       label="Valve"),
        mpatches.Patch(facecolor=UNIT_COLORS["product"],     label="Product"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7,
              framealpha=0.9, ncol=3)

    ax.set_title(
        r"Best flowsheet: 5-component NGL (nominal, $S_\mathrm{norm}=0.903$)  "
        "— pressure-swing sequence",
        fontsize=9, pad=6,
    )

    out = FIGS_OUT / "fig_topology_schematic.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Copy topology PNG (kept for reference; schematic replaces it in report)
# ═══════════════════════════════════════════════════════════════════════════════
def copy_topology():
    if TOPO_SRC.exists():
        shutil.copy(TOPO_SRC, TOPO_DST)
        print(f"Copied topology figure -> {TOPO_DST}")
    else:
        print(f"WARNING: topology figure not found at {TOPO_SRC}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Report 1 ...")
    fig_convergence()
    fig_success_rates()
    fig_thermo_cost()
    fig_fot_distribution()
    fig_convergence_cdf()
    fig_topology_schematic()
    copy_topology()
    print("Done.")
