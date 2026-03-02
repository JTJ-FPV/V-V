#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example:
# python process_task3.py --root ./data --out ./result/task3 --exp_cp ./data/Experiment_Cp_AoA_5_Freestream.dat --exp_usecols 0,1

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Parsers
# ============================================================

def parse_usecols(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("exp_usecols must be like '0,1'")
    return int(parts[0]), int(parts[1])


def load_exp_cp_data(path: Path, usecols=(0, 1)) -> pd.DataFrame:
    data = np.loadtxt(path, usecols=usecols)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Experimental Cp file must contain at least two columns: {path}")
    return pd.DataFrame({"x": data[:, 0].astype(float), "cp": data[:, 1].astype(float)})


def parse_fluent_rfile(path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*([+-]?\d+)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )

    iterations: List[int] = []
    values: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                iterations.append(int(m.group(1)))
                values.append(float(m.group(2)))

    if not iterations:
        raise ValueError(f"Failed to parse Fluent report file: {path}")

    return pd.DataFrame({"iteration": iterations, "value": values}).sort_values("iteration").reset_index(drop=True)


def parse_tecplot_xy_pairs(path: Path) -> pd.DataFrame:
    pair = re.compile(
        r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )

    xs: List[float] = []
    ys: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pair.match(line.replace("\t", " ").strip())
            if m:
                xs.append(float(m.group(1)))
                ys.append(float(m.group(2)))

    if not xs:
        raise ValueError(f"Failed to parse XY file: {path}")

    return pd.DataFrame({"x": xs, "cp": ys})


# ============================================================
# Helpers
# ============================================================

def split_surfaces_by_cp(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cp_med = np.median(df["cp"].to_numpy())
    idx_suction = df["cp"] <= cp_med
    idx_pressure = ~idx_suction

    n = len(df)
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = df["cp"] < 0.0
        idx_pressure = ~idx_suction

    suction = (
        df.loc[idx_suction]
        .copy()
        .sort_values("x")
        .drop_duplicates(subset="x")
        .reset_index(drop=True)
    )
    pressure = (
        df.loc[idx_pressure]
        .copy()
        .sort_values("x")
        .drop_duplicates(subset="x")
        .reset_index(drop=True)
    )

    return {"suction": suction, "pressure": pressure}


def tail_stats(series: pd.Series, tail: int = 200) -> Dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        return {
            "n_total": 0,
            "last": np.nan,
            "mean_tail": np.nan,
            "std_tail": np.nan,
            "range_tail": np.nan,
        }

    t = s.iloc[-tail:] if len(s) > tail else s
    return {
        "n_total": int(len(s)),
        "last": float(s.iloc[-1]),
        "mean_tail": float(t.mean()),
        "std_tail": float(t.std(ddof=1) if len(t) > 1 else 0.0),
        "range_tail": float(t.max() - t.min()),
    }


def cp_uncertainty_vs_exp(cp_df: pd.DataFrame, exp_df: pd.DataFrame, n_grid: int = 300) -> Dict[str, float]:
    cp_split = split_surfaces_by_cp(cp_df)
    exp_split = split_surfaces_by_cp(exp_df)

    errs_all = []
    out = {}

    for branch in ["pressure", "suction"]:
        cfd = cp_split[branch]
        exp = exp_split[branch]

        if len(cfd) < 2 or len(exp) < 2:
            out[f"u_rms_{branch}"] = np.nan
            out[f"u_max_{branch}"] = np.nan
            continue

        x_min = max(cfd["x"].min(), exp["x"].min())
        x_max = min(cfd["x"].max(), exp["x"].max())
        if x_max <= x_min:
            out[f"u_rms_{branch}"] = np.nan
            out[f"u_max_{branch}"] = np.nan
            continue

        x_common = np.linspace(x_min, x_max, n_grid)
        y_cfd = np.interp(x_common, cfd["x"].to_numpy(), cfd["cp"].to_numpy())
        y_exp = np.interp(x_common, exp["x"].to_numpy(), exp["cp"].to_numpy())

        err = y_cfd - y_exp
        errs_all.append(err)

        out[f"u_rms_{branch}"] = float(np.sqrt(np.mean(err**2)))
        out[f"u_max_{branch}"] = float(np.max(np.abs(err)))

    if errs_all:
        all_err = np.concatenate(errs_all)
        out["u_rms_all"] = float(np.sqrt(np.mean(all_err**2)))
        out["u_max_all"] = float(np.max(np.abs(all_err)))
    else:
        out["u_rms_all"] = np.nan
        out["u_max_all"] = np.nan

    return out


# ============================================================
# Data structures
# ============================================================

@dataclass
class Task3Case:
    model: str           # kw / sa
    variant: str         # baseline / ama_300
    cd_file: Path
    cl_file: Path
    cp_file: Path


def discover_task3_cases(root: Path) -> List[Task3Case]:
    cases: List[Task3Case] = []

    # kw
    kw_dir = root / "kw" / "task3"
    if kw_dir.exists():
        f = kw_dir / "cd_1_1_1_AUSM.out"
        l = kw_dir / "cl_1_1_1_AUSM.out"
        p = kw_dir / "kw_1_1_1_AUSM.txt"
        if f.exists() and l.exists() and p.exists():
            cases.append(Task3Case("kw", "baseline", f, l, p))

        f = kw_dir / "cd_1_1_1_ama_300.out"
        l = kw_dir / "cl_1_1_1_ama_300.out"
        p = kw_dir / "kw_1_1_1_ama_300.txt"
        if f.exists() and l.exists() and p.exists():
            cases.append(Task3Case("kw", "ama_300", f, l, p))

    # sa
    sa_dir = root / "sa" / "task3"
    if sa_dir.exists():
        f = sa_dir / "cd_1_2_AUSM.out"
        l = sa_dir / "cl_1_2_AUSM.out"
        p = sa_dir / "sa_1_2_AUSM.txt"
        if f.exists() and l.exists() and p.exists():
            cases.append(Task3Case("sa", "baseline", f, l, p))

        f = sa_dir / "cd_1_2_ama_300.out"
        l = sa_dir / "cl_1_2_ama_300.out"
        p = sa_dir / "sa_1_2_ama_300.txt"
        if f.exists() and l.exists() and p.exists():
            cases.append(Task3Case("sa", "ama_300", f, l, p))

    return cases

def build_task3_improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compare ama_300 against baseline for each model.

    Output columns:
      model
      baseline_u_rms_all
      ama_300_u_rms_all
      delta_u_rms_all
      baseline_u_max_all
      ama_300_u_max_all
      delta_u_max_all
    """
    rows = []

    for model in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model].copy()

        row_b = sub[sub["variant"] == "baseline"]
        row_a = sub[sub["variant"] == "ama_300"]

        if row_b.empty or row_a.empty:
            continue

        b = row_b.iloc[0]
        a = row_a.iloc[0]

        rows.append({
            "model": model,
            "baseline_u_rms_all": float(b["u_rms_all"]),
            "ama_300_u_rms_all": float(a["u_rms_all"]),
            "delta_u_rms_all": float(a["u_rms_all"] - b["u_rms_all"]),
            "baseline_u_max_all": float(b["u_max_all"]),
            "ama_300_u_max_all": float(a["u_max_all"]),
            "delta_u_max_all": float(a["u_max_all"] - b["u_max_all"]),
        })

    return pd.DataFrame(rows)

def plot_task3_uq_baseline_vs_ama(improve_df: pd.DataFrame, out_png: Path) -> None:
    """
    For each model, compare baseline vs ama_300 uncertainty.
    """
    if improve_df.empty:
        return

    x = np.arange(len(improve_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width/2, improve_df["baseline_u_rms_all"], width, label="Baseline RMS vs Exp")
    ax.bar(x + width/2, improve_df["ama_300_u_rms_all"], width, label="ama_300 RMS vs Exp")

    ax.set_xticks(x)
    ax.set_xticklabels(improve_df["model"])
    ax.set_ylabel("RMS uncertainty")
    ax.set_title("Task3: Baseline vs adaptive-mesh uncertainty")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_task3_uq_improvement_delta(improve_df: pd.DataFrame, out_png: Path) -> None:
    """
    Delta uncertainty = ama_300 - baseline.
    Negative means improvement.
    """
    if improve_df.empty:
        return

    x = np.arange(len(improve_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width/2, improve_df["delta_u_rms_all"], width, label="Δ RMS (ama_300 - baseline)")
    ax.bar(x + width/2, improve_df["delta_u_max_all"], width, label="Δ Max (ama_300 - baseline)")

    ax.axhline(0.0, color="k", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(improve_df["model"])
    ax.set_ylabel("Change in uncertainty")
    ax.set_title("Task3: Change in uncertainty after adaptive mesh")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

# ============================================================
# Unified plotting styles (consistent with process_task2_task4.py)
# ============================================================

CASE_STYLE_MAP = {
    # keep the same visual logic as task4_Cp_all_models
    "kw_baseline": {"color": "C0", "linestyle": "-",  "linewidth": 1.5},
    "sa_baseline": {"color": "C1", "linestyle": "--", "linewidth": 1.5},
    "kw_ama_300":  {"color": "C2", "linestyle": "-",  "linewidth": 1.5},
    "sa_ama_300":  {"color": "C3", "linestyle": "--", "linewidth": 1.5},
}

EXP_STYLE = {
    "color": "k",
    "linestyle": "-",
    "linewidth": 1.8,
}


# ============================================================
# Plotting
# ============================================================

def plot_two_histories(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ylabel: str,
    title: str,
    out_png: Path,
    case1_key: str,
    case2_key: str,
    label1: str = "baseline",
    label2: str = "ama_300",
) -> None:
    style1 = CASE_STYLE_MAP.get(case1_key, {"color": "C2", "linestyle": "-", "linewidth": 1.8})
    style2 = CASE_STYLE_MAP.get(case2_key, {"color": "C3", "linestyle": "--", "linewidth": 1.8})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df1["iteration"], df1["value"],
        color=style1["color"], linestyle=style1["linestyle"], linewidth=style1["linewidth"],
        label=label1,
    )
    ax.plot(
        df2["iteration"], df2["value"],
        color=style2["color"], linestyle=style2["linestyle"], linewidth=style2["linewidth"],
        label=label2,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_cp_model_variants_vs_exp(
    cp_baseline: pd.DataFrame,
    cp_ama: pd.DataFrame,
    exp_df: pd.DataFrame,
    title: str,
    out_png: Path,
    model_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    baseline_key = f"{model_name}_baseline"
    ama_key = f"{model_name}_ama_300"

    for case_key, cp_df, legend_label in [
        (baseline_key, cp_baseline, f"{model_name}_baseline"),
        (ama_key, cp_ama, f"{model_name}_ama_300"),
    ]:
        style = CASE_STYLE_MAP.get(case_key, {"color": "C2", "linestyle": "-", "linewidth": 1.8})
        cp_split = split_surfaces_by_cp(cp_df)

        first = True
        for branch in ["pressure", "suction"]:
            df = cp_split[branch]
            ax.plot(
                df["x"], df["cp"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker="o",
                markersize=3,
                label=legend_label if first else None,
            )
            first = False

    exp_split = split_surfaces_by_cp(exp_df)
    first = True
    for branch in ["pressure", "suction"]:
        df = exp_split[branch]
        ax.plot(
            df["x"], df["cp"],
            color=EXP_STYLE["color"],
            linestyle=EXP_STYLE["linestyle"],
            linewidth=EXP_STYLE["linewidth"],
            label="Exp" if first else None,
        )
        first = False

    ax.invert_yaxis()
    ax.set_xlabel("x/c (Position)")
    ax.set_ylabel("Cp")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def split_into_segments_by_x_jumps(x: np.ndarray, jump_thresh: float = 0.25):
    if len(x) < 2:
        return [slice(0, len(x))]

    cuts = [0]
    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > jump_thresh:
            cuts.append(i)
    cuts.append(len(x))

    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 3:
            segs.append(slice(a, b))

    return segs if segs else [slice(0, len(x))]

def plot_cp_all_cases_vs_exp(cp_case_map: Dict[str, pd.DataFrame], exp_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    # CFD curves: use one explicit style per case, same logic as task4
    for label, df in cp_case_map.items():
        style = CASE_STYLE_MAP.get(label, {"color": "C6", "linestyle": "-.", "linewidth": 1.2})

        x = df["x"].to_numpy()
        cp = df["cp"].to_numpy()

        # same segmentation logic as process_task2_task4.py
        segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)

        first = True
        for s in segs:
            ax.plot(
                x[s],
                cp[s],
                marker="o",
                markersize=2.5,
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=label if first else None,
            )
            first = False

    # Experimental Cp: same style for both surfaces, only one legend entry
    exp_split = split_surfaces_by_cp(exp_df)

    first = True
    for key in ["pressure", "suction"]:
        if key in exp_split and len(exp_split[key]) > 0:
            df = exp_split[key]
            ax.plot(
                df["x"],
                df["cp"],
                color=EXP_STYLE["color"],
                linestyle=EXP_STYLE["linestyle"],
                linewidth=EXP_STYLE["linewidth"],
                label="Exp" if first else None,
            )
            first = False

    ax.invert_yaxis()
    ax.set_xlabel("x/c (Position)")
    ax.set_ylabel("Cp")
    ax.set_title("Task3: Cp comparison of all cases vs experiment")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_task3_uncertainty_bar(summary: pd.DataFrame, out_png: Path) -> None:
    df = summary.sort_values("u_rms_all").reset_index(drop=True)

    labels = [f"{m}_{v}" for m, v in zip(df["model"], df["variant"])]
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["u_rms_all"], width, label="RMS vs Exp")
    ax.bar(x + width/2, df["u_max_all"], width, label="Max abs error vs Exp")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Model uncertainty")
    ax.set_title("Task3: Per-case uncertainty relative to experimental Cp")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ============================================================
# Main processing
# ============================================================

def run_task3(root: Path, outdir: Path, exp_cp_path: Path, exp_usecols=(0, 1), tail: int = 200) -> pd.DataFrame:
    cases = discover_task3_cases(root)
    if not cases:
        raise RuntimeError(f"No Task3 cases found under: {root}")

    if not exp_cp_path.exists():
        raise FileNotFoundError(f"Experimental Cp file not found: {exp_cp_path}")

    exp_df = load_exp_cp_data(exp_cp_path, usecols=exp_usecols)

    rows = []
    cp_case_map: Dict[str, pd.DataFrame] = {}

    # 先逐个case处理
    parsed = {}
    for case in cases:
        cd_df = parse_fluent_rfile(case.cd_file)
        cl_df = parse_fluent_rfile(case.cl_file)
        cp_df = parse_tecplot_xy_pairs(case.cp_file)

        key = f"{case.model}_{case.variant}"
        parsed[key] = {
            "cd": cd_df,
            "cl": cl_df,
            "cp": cp_df,
        }
        cp_case_map[key] = cp_df

        row = {
            "model": case.model,
            "variant": case.variant,
            "cd_last": float(cd_df["value"].iloc[-1]),
            "cl_last": float(cl_df["value"].iloc[-1]),
            **{f"cd_{k}": v for k, v in tail_stats(cd_df["value"], tail=tail).items()},
            **{f"cl_{k}": v for k, v in tail_stats(cl_df["value"], tail=tail).items()},
        }
        row.update(cp_uncertainty_vs_exp(cp_df, exp_df, n_grid=300))
        rows.append(row)

    summary = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "task3_summary.csv", index=False)

    # Build improvement table: does ama_300 improve agreement with experiment?
    improve_df = build_task3_improvement_table(summary)
    improve_df.to_csv(outdir / "task3_improvement_summary.csv", index=False)

    # 每个模型内部：baseline vs ama_300
    for model in ["kw", "sa"]:
        key_b = f"{model}_baseline"
        key_a = f"{model}_ama_300"
        if key_b in parsed and key_a in parsed:
            plot_two_histories(
                parsed[key_b]["cl"],
                parsed[key_a]["cl"],
                "Cl",
                f"Task3 {model.upper()}: Cl convergence (baseline vs ama_300)",
                outdir / "figures" / model / f"{model}_task3_Cl_baseline_vs_ama300.png",
                case1_key=key_b,
                case2_key=key_a,
                label1="baseline",
                label2="ama_300",
            )
            plot_two_histories(
                parsed[key_b]["cd"],
                parsed[key_a]["cd"],
                "Cd",
                f"Task3 {model.upper()}: Cd convergence (baseline vs ama_300)",
                outdir / "figures" / model / f"{model}_task3_Cd_baseline_vs_ama300.png",
                case1_key=key_b,
                case2_key=key_a,
                label1="baseline",
                label2="ama_300",
            )
            plot_cp_model_variants_vs_exp(
                parsed[key_b]["cp"],
                parsed[key_a]["cp"],
                exp_df,
                f"Task3 {model.upper()}: Cp comparison (baseline vs ama_300 vs experiment)",
                outdir / "figures" / model / f"{model}_task3_Cp_baseline_ama300_Exp.png",
                model_name=model,
            )

    # 所有case总合并图
    plot_cp_all_cases_vs_exp(
        cp_case_map,
        exp_df,
        outdir / "figures" / "combined" / "task3_Cp_all_cases_vs_exp.png",
    )

    # 不确定性柱状图
    plot_task3_uncertainty_bar(
        summary,
        outdir / "figures" / "combined" / "task3_model_uncertainty.png",
    )

    # Baseline vs adaptive-mesh uncertainty comparison
    plot_task3_uq_baseline_vs_ama(
        improve_df,
        outdir / "figures" / "combined" / "task3_uq_baseline_vs_ama.png",
    )

    # Direct improvement / deterioration after adaptive mesh
    plot_task3_uq_improvement_delta(
        improve_df,
        outdir / "figures" / "combined" / "task3_uq_improvement_delta.png",
    )

    return summary, improve_df


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process Task3 data: baseline vs adaptive-mesh Cp comparison against experiment."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory containing kw/ and sa/")
    parser.add_argument("--out", type=str, default="result/task3", help="Output directory")
    parser.add_argument("--exp_cp", type=str, required=True, help="Experimental Cp file path")
    parser.add_argument("--exp_usecols", type=str, default="0,1", help="Columns for x/c and Cp in experimental file")
    parser.add_argument("--tail", type=int, default=200, help="Last N iterations for convergence tail statistics")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    exp_cp_path = Path(args.exp_cp).resolve()
    exp_usecols = parse_usecols(args.exp_usecols)

    summary, improve_df = run_task3(
        root=root,
        outdir=outdir,
        exp_cp_path=exp_cp_path,
        exp_usecols=exp_usecols,
        tail=args.tail,
    )

    print(f"[OK] Task3 summary            : {outdir / 'task3_summary.csv'}")
    print(f"[OK] Task3 improvement table : {outdir / 'task3_improvement_summary.csv'}")
    print(f"[OK] Figures dir             : {outdir / 'figures'}")

    cols = ["model", "variant", "u_rms_all", "u_max_all", "cl_mean_tail", "cd_mean_tail"]
    keep = [c for c in cols if c in summary.columns]
    print(summary[keep].to_string(index=False))

    if not improve_df.empty:
        print("\n[Adaptive mesh improvement summary]")
        print(improve_df.to_string(index=False))


if __name__ == "__main__":
    main()