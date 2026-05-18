"""Generate figures and statistical tables for the diploma chapter 4.

Reads parquet files from data/experiments/experiments/<exp_id>/_runs.parquet,
maps them to E1/E2/E3/E4/conf experiments via experiment.json names, then
emits PNG figures into arch/diploma/figures/ and a JSON report with
two-way ANOVA on E1 and Cohen's d on key pairwise comparisons.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "arch" / "diploma" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 140,
})

CORPUS_LABEL = {
    "humaneval": "HumanEval",
    "gsm8k": "GSM8K",
    "commongen": "CommonGen",
    "dabench": "DABench",
}
TOPO_ORDER = ["chain", "star", "debate", "hierarchical", "mesh"]
ROLE_ORDER = ["coordinator", "reviewer", "judge", "peer", "monitor"]
CORPUS_ORDER = ["humaneval", "gsm8k", "commongen", "dabench"]


DEDUP_LAST_N = 15  # last N per (task_id, seed), matches original analysis


def load_experiments() -> dict[str, pd.DataFrame]:
    """Map experiment.json[name] → labelled DataFrame.

    For runs whose grid duplicated retries (E3 router-* and the confirmation
    sets), apply the same dedup as the original analysis: keep only completed
    rows with a non-null quality_score and take the last 15 per
    (task_id, seed). For E1/E2/E4 the parquet is already at planned cardinality
    (900 / 2295 / 540), so the dedup is a no-op there.
    """
    DEDUP_NAMES = {
        "e3_full_router_oracle", "e3_full_router_rule", "e3_full_router_llm",
        "confirmation_e3_rule", "confirmation_e3_llm",
    }
    mapping: dict[str, pd.DataFrame] = {}
    for cfg in sorted(glob.glob(str(REPO / "data/experiments/experiments/*/experiment.json"))):
        meta = json.loads(Path(cfg).read_text())
        name = meta["name"]
        runs = Path(cfg).parent / "_runs.parquet"
        if not runs.exists():
            continue
        df = pd.read_parquet(runs).sort_values("started_at").reset_index(drop=True)
        if name in DEDUP_NAMES:
            df = df[df["status"] == "completed"].dropna(subset=["quality_score"])
            df = df.sort_values("started_at").groupby(["task_id", "seed"]).tail(DEDUP_LAST_N).reset_index(drop=True)
        mapping[name] = df
    return mapping


def split_e4_by_router(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """E4 sweep order is fixed → rule → llm, 180 rows each (4×15×3)."""
    chunk = 180
    return {
        "fixed": df.iloc[0 * chunk : 1 * chunk],
        "rule": df.iloc[1 * chunk : 2 * chunk],
        "llm": df.iloc[2 * chunk : 3 * chunk],
    }


def bootstrap_ci(x: np.ndarray, y: np.ndarray, *, n: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the difference of means (x - y)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = rng.choice(x, size=len(x), replace=True).mean() - rng.choice(y, size=len(y), replace=True).mean()
    return float(np.mean(x) - np.mean(y)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


# --------------------------------------------------------------------------
# Figure 1: E1 quality by topology × corpus
# --------------------------------------------------------------------------
def fig_e1_bars(e1: pd.DataFrame, path: Path) -> None:
    pivot = e1.pivot_table(index="topology", columns="task_id", values="quality_score", aggfunc="mean")
    pivot = pivot.reindex(TOPO_ORDER)[CORPUS_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(TOPO_ORDER))
    width = 0.18
    colors = ["#3a6ea5", "#c0392b", "#27ae60", "#e67e22"]
    for i, corpus in enumerate(CORPUS_ORDER):
        ax.bar(x + (i - 1.5) * width, pivot[corpus].values, width,
               label=CORPUS_LABEL[corpus], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TOPO_ORDER])
    ax.set_ylabel("Quality (q)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Quality of static topologies by corpus (E1)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: Pareto-front quality vs cost (mean per cell)
# --------------------------------------------------------------------------
def fig_pareto(e1: pd.DataFrame, e3: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # E1 statics (per topology averaged across corpora)
    g = e1.groupby("topology").agg(q=("quality_score", "mean"), c=("budget_spent_usd", "mean")).reindex(TOPO_ORDER)
    ax.scatter(g["c"], g["q"], s=85, c="#3a6ea5", marker="o", label="Static topology (E1)", zorder=3)
    for t, row in g.iterrows():
        ax.annotate(t, (row["c"], row["q"]), xytext=(6, 4), textcoords="offset points", fontsize=8)

    # Per-task best-of (oracle of statics)
    per_task = e1.groupby(["task_id", "topology"])["quality_score"].mean().reset_index()
    best_per = per_task.loc[per_task.groupby("task_id")["quality_score"].idxmax()]
    oracle_q = best_per["quality_score"].mean()
    oracle_c = e1.groupby(["task_id", "topology"])["budget_spent_usd"].mean().reset_index().merge(
        best_per[["task_id", "topology"]], on=["task_id", "topology"])
    oracle_c_val = oracle_c["budget_spent_usd"].mean()
    ax.scatter([oracle_c_val], [oracle_q], s=140, marker="*", c="#f1c40f",
               edgecolor="black", linewidth=0.8, label="Best static per task (oracle)", zorder=4)

    # E3 routers
    e3_pts = {
        "Adaptive — oracle router": (e3["oracle"], "#27ae60", "s"),
        "Adaptive — rule router":   (e3["rule"],   "#e67e22", "D"),
        "Adaptive — LLM router":    (e3["llm"],    "#c0392b", "^"),
    }
    for label, (df, color, marker) in e3_pts.items():
        q = df["quality_score"].mean()
        c = df["budget_spent_usd"].mean()
        ax.scatter([c], [q], s=110, marker=marker, c=color, label=label, zorder=3,
                   edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Cost per run, USD (mean)")
    ax.set_ylabel("Quality (q)")
    ax.set_title("Pareto plane — quality vs cost")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: HITL lift per (topology × corpus) cell (E2 vs E1)
# --------------------------------------------------------------------------
def fig_hitl_lift(e1: pd.DataFrame, e2: pd.DataFrame, path: Path) -> None:
    base = e1.groupby(["topology", "task_id"])["quality_score"].mean().reset_index().rename(
        columns={"quality_score": "q_e1"})
    e2_best = e2.groupby(["topology", "task_id"])["quality_score"].mean().reset_index().rename(
        columns={"quality_score": "q_e2"})
    merged = base.merge(e2_best, on=["topology", "task_id"], how="inner")
    merged["lift"] = merged["q_e2"] - merged["q_e1"]
    pivot = merged.pivot(index="topology", columns="task_id", values="lift").reindex(TOPO_ORDER)[CORPUS_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(TOPO_ORDER))
    width = 0.18
    colors = ["#3a6ea5", "#c0392b", "#27ae60", "#e67e22"]
    for i, corpus in enumerate(CORPUS_ORDER):
        ax.bar(x + (i - 1.5) * width, pivot[corpus].values, width,
               label=CORPUS_LABEL[corpus], color=colors[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TOPO_ORDER])
    ax.set_ylabel(r"$\Delta q$  (E2 best role — E1)")
    ax.set_title("HITL lift per cell (E2 best role vs E1)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: Bootstrap forest plot
# --------------------------------------------------------------------------
def fig_bootstrap(e3: dict[str, pd.DataFrame], e4: dict[str, pd.DataFrame],
                  conf_e3: dict[str, pd.DataFrame], conf_e4: dict[str, pd.DataFrame],
                  path: Path) -> None:
    rows = []

    def add(label: str, x: np.ndarray, y: np.ndarray) -> None:
        d, lo, hi = bootstrap_ci(x, y)
        rows.append((label, d, lo, hi))

    add("E3: rule − llm (q, gpt-oss)",
        e3["rule"]["quality_score"].values, e3["llm"]["quality_score"].values)
    add("E3: oracle − rule (q, gpt-oss)",
        e3["oracle"]["quality_score"].values, e3["rule"]["quality_score"].values)
    add("E4: rule − fixed (q, gpt-oss)",
        e4["rule"]["quality_score"].values, e4["fixed"]["quality_score"].values)
    add("conf_e3: rule − llm (q, Qwen)",
        conf_e3["rule"]["quality_score"].values, conf_e3["llm"]["quality_score"].values)
    add("conf_e4: rule − fixed (q, Qwen)",
        conf_e4["rule"]["quality_score"].values, conf_e4["fixed"]["quality_score"].values)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ys = np.arange(len(rows))[::-1]
    for y, (label, point, lo, hi) in zip(ys, rows):
        color = "#c0392b" if (lo > 0 or hi < 0) else "#555"
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o",
                    color=color, ecolor=color, capsize=4, lw=1.6, ms=6)
        ax.text(hi + 0.005, y, f"[{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(r"$\Delta q$ (95% bootstrap CI)")
    ax.set_title("Bootstrap CI forest plot — key comparisons")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5: Cross-family comparison (gpt-oss vs Qwen)
# --------------------------------------------------------------------------
def fig_crossfamily(e3: dict[str, pd.DataFrame], conf_e3: dict[str, pd.DataFrame], path: Path) -> None:
    families = ["gpt-oss-120b", "Qwen-3-235b"]
    rule_q = [e3["rule"]["quality_score"].mean(), conf_e3["rule"]["quality_score"].mean()]
    llm_q = [e3["llm"]["quality_score"].mean(), conf_e3["llm"]["quality_score"].mean()]
    rule_c = [e3["rule"]["budget_spent_usd"].mean(), conf_e3["rule"]["budget_spent_usd"].mean()]
    llm_c = [e3["llm"]["budget_spent_usd"].mean(), conf_e3["llm"]["budget_spent_usd"].mean()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    x = np.arange(2)
    w = 0.35
    ax1.bar(x - w / 2, rule_q, w, color="#27ae60", label="Rule router")
    ax1.bar(x + w / 2, llm_q, w, color="#c0392b", label="LLM router")
    ax1.set_xticks(x)
    ax1.set_xticklabels(families)
    ax1.set_ylabel("Quality (q)")
    ax1.set_title("Quality of routers by family")
    ax1.legend(framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    ratio = [r / l for r, l in zip(rule_c, llm_c)]
    ax2.bar(x, ratio, color=["#3a6ea5", "#e67e22"])
    ax2.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(families)
    ax2.set_ylabel(r"$c_{\mathrm{rule}} / c_{\mathrm{llm}}$")
    ax2.set_title("Cost ratio — collapse on Qwen")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# ANOVA + Cohen's d
# --------------------------------------------------------------------------
def run_anova(e1: pd.DataFrame) -> dict:
    df = e1[["quality_score", "topology", "task_id"]].copy()
    df["topology"] = df["topology"].astype("category")
    df["task_id"] = df["task_id"].astype("category")
    model = ols("quality_score ~ C(topology) + C(task_id) + C(topology):C(task_id)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)

    # Compute partial eta-squared per factor.
    ss_resid = table.loc["Residual", "sum_sq"]
    rows = []
    for factor in ["C(topology)", "C(task_id)", "C(topology):C(task_id)"]:
        ss = table.loc[factor, "sum_sq"]
        df_eff = table.loc[factor, "df"]
        F = table.loc[factor, "F"]
        p = table.loc[factor, "PR(>F)"]
        eta2_p = ss / (ss + ss_resid)
        rows.append({
            "factor": factor.replace("C(", "").replace(")", "").replace(":", "×"),
            "df": int(df_eff),
            "F": float(F),
            "p": float(p),
            "eta2_partial": float(eta2_p),
        })
    return {"anova": rows, "n": len(df)}


def run_cohens_d(e1: pd.DataFrame, e3: dict, e4: dict, conf_e4: dict) -> list[dict]:
    rows: list[dict] = []
    # E1: chain vs mesh overall (largest spread observed)
    q_chain = e1.loc[e1["topology"] == "chain", "quality_score"].values
    q_mesh = e1.loc[e1["topology"] == "mesh", "quality_score"].values
    rows.append({"pair": "E1: chain vs mesh (overall)", "d": cohens_d(q_chain, q_mesh), "n1": len(q_chain), "n2": len(q_mesh)})

    # E1: chain vs debate on HumanEval (largest topology contrast on one corpus)
    he = e1[e1["task_id"] == "humaneval"]
    rows.append({
        "pair": "E1: chain vs debate (HumanEval)",
        "d": cohens_d(
            he.loc[he["topology"] == "chain", "quality_score"].values,
            he.loc[he["topology"] == "debate", "quality_score"].values,
        ),
        "n1": int((he["topology"] == "chain").sum()),
        "n2": int((he["topology"] == "debate").sum()),
    })

    # E3: oracle vs llm router
    rows.append({
        "pair": "E3: oracle vs llm router",
        "d": cohens_d(e3["oracle"]["quality_score"].values, e3["llm"]["quality_score"].values),
        "n1": len(e3["oracle"]), "n2": len(e3["llm"]),
    })

    # E4: rule vs fixed
    rows.append({
        "pair": "E4: rule vs fixed router (gpt-oss)",
        "d": cohens_d(e4["rule"]["quality_score"].values, e4["fixed"]["quality_score"].values),
        "n1": len(e4["rule"]), "n2": len(e4["fixed"]),
    })

    # Cross-family same comparison
    rows.append({
        "pair": "conf_e4: rule vs fixed router (Qwen)",
        "d": cohens_d(conf_e4["rule"]["quality_score"].values, conf_e4["fixed"]["quality_score"].values),
        "n1": len(conf_e4["rule"]), "n2": len(conf_e4["fixed"]),
    })

    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    exps = load_experiments()
    e1 = exps["e1_full"]
    e2 = exps["e2_full"]
    e3 = {
        "oracle": exps["e3_full_router_oracle"],
        "rule": exps["e3_full_router_rule"],
        "llm": exps["e3_full_router_llm"],
    }
    e4 = split_e4_by_router(exps["e4_full"])
    conf_e3 = {
        "rule": exps["confirmation_e3_rule"],
        "llm": exps["confirmation_e3_llm"],
    }
    conf_e4 = split_e4_by_router(exps["confirmation_e4"])

    fig_e1_bars(e1, OUT / "fig_e1_quality_bars.png")
    fig_pareto(e1, e3, OUT / "fig_pareto.png")
    fig_hitl_lift(e1, e2, OUT / "fig_hitl_lift.png")
    fig_bootstrap(e3, e4, conf_e3, conf_e4, OUT / "fig_bootstrap_forest.png")
    fig_crossfamily(e3, conf_e3, OUT / "fig_crossfamily.png")

    stats_out = {
        "anova_e1": run_anova(e1),
        "cohens_d": run_cohens_d(e1, e3, e4, conf_e4),
    }
    (OUT / "stats_report.json").write_text(json.dumps(stats_out, indent=2, ensure_ascii=False))

    print("\n=== Two-way ANOVA on E1 (quality ~ topology × corpus) ===")
    for r in stats_out["anova_e1"]["anova"]:
        print(f"  {r['factor']:<25s}  F={r['F']:7.2f}  df={r['df']:3d}  p={r['p']:.3e}  η²p={r['eta2_partial']:.3f}")
    print(f"  N = {stats_out['anova_e1']['n']}")

    print("\n=== Cohen's d for key pairs ===")
    for r in stats_out["cohens_d"]:
        print(f"  {r['pair']:<40s}  d={r['d']:+.3f}  (n1={r['n1']}, n2={r['n2']})")

    print(f"\nFigures saved to {OUT}")


if __name__ == "__main__":
    main()
