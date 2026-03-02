#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_case3_task4.py  (Case3 only, Task4-style validation)

Usage:
  python process_case3_task4.py --root ./data --out ./result/case3_task4 --exp_cp ./data/Experiment_Cp_AoA_3_6_degree_hc_0_224_ge.dat --exp_usecols 0,1 --tail 200

Input folders:
  ./data/kw/case3
  ./data/sa/case3

Outputs:
  - figures/case3/combined/case3_Cp_kw_sa_vs_exp.png
  - figures/case3/combined/case3_cp_uncertainty_per_model.png
  - figures/case3/combined/case3_Cl_kw_vs_sa.png
  - figures/case3/combined/case3_Cd_kw_vs_sa.png
  - figures/case3/combined/case3_std_tail_bar.png
  - figures/case3/combined/case3_range_tail_bar.png
  - figures/case3/combined/case3_abs_last_minus_mean_bar.png
  - case3_tail_stats.csv
  - case3_cp_uncertainty.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    return (int(parts[0]), int(parts[1]))


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
# Cp splitting + line segmentation (same as task2_task4)
# ============================================================

def split_exp_surfaces_by_cp(exp: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cp_med = np.median(exp["cp"].to_numpy())
    idx_suction = exp["cp"] <= cp_med
    idx_pressure = ~idx_suction

    n = len(exp)
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = exp["cp"] < 0.0
        idx_pressure = ~idx_suction

    suction = exp.loc[idx_suction].copy().sort_values("x").reset_index(drop=True)
    pressure = exp.loc[idx_pressure].copy().sort_values("x").reset_index(drop=True)
    return {"suction": suction, "pressure": pressure}


def split_cp_surfaces_by_median(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cp_med = np.median(df["cp"].to_numpy())
    idx_suction = df["cp"] <= cp_med
    idx_pressure = ~idx_suction

    n = len(df)
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = df["cp"] < 0.0
        idx_pressure = ~idx_suction

    suction = df.loc[idx_suction].copy().sort_values("x").reset_index(drop=True)
    pressure = df.loc[idx_pressure].copy().sort_values("x").reset_index(drop=True)
    return {"suction": suction, "pressure": pressure}


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


# ============================================================
# Statistics (same as task2_task4)
# ============================================================

def tail_stats(series: pd.Series, tail: int = 200) -> Dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        return {
            "n_total": 0,
            "last": np.nan,
            "mean_tail": np.nan,
            "std_tail": np.nan,
            "min_tail": np.nan,
            "max_tail": np.nan,
            "range_tail": np.nan,
            "abs_last_minus_mean": np.nan,
        }

    t = s.iloc[-tail:] if len(s) > tail else s
    last = float(s.iloc[-1])
    mean_tail = float(t.mean())
    std_tail = float(t.std(ddof=1) if len(t) > 1 else 0.0)
    min_tail = float(t.min())
    max_tail = float(t.max())
    range_tail = max_tail - min_tail

    return {
        "n_total": int(len(s)),
        "last": last,
        "mean_tail": mean_tail,
        "std_tail": std_tail,
        "min_tail": min_tail,
        "max_tail": max_tail,
        "range_tail": range_tail,
        "abs_last_minus_mean": abs(last - mean_tail),
    }


def compute_single_model_uncertainty_vs_exp(
    cp_df: pd.DataFrame,
    exp_split: Dict[str, pd.DataFrame],
    n_grid: int = 300,
) -> Dict[str, float]:
    cp_split = split_cp_surfaces_by_median(cp_df)
    all_err = []
    branch_metrics: Dict[str, float] = {}

    for branch_name in ["pressure", "suction"]:
        cfd_branch = cp_split[branch_name].copy().sort_values("x").drop_duplicates(subset="x")
        exp_branch = exp_split[branch_name].copy().sort_values("x").drop_duplicates(subset="x")

        x_cfd = cfd_branch["x"].to_numpy()
        y_cfd = cfd_branch["cp"].to_numpy()
        x_exp = exp_branch["x"].to_numpy()
        y_exp = exp_branch["cp"].to_numpy()

        if len(x_cfd) < 2 or len(x_exp) < 2:
            branch_metrics[f"u_rms_{branch_name}"] = np.nan
            branch_metrics[f"u_max_{branch_name}"] = np.nan
            continue

        x_min = max(np.min(x_cfd), np.min(x_exp))
        x_max = min(np.max(x_cfd), np.max(x_exp))
        if x_max <= x_min:
            branch_metrics[f"u_rms_{branch_name}"] = np.nan
            branch_metrics[f"u_max_{branch_name}"] = np.nan
            continue

        x_common = np.linspace(x_min, x_max, n_grid)
        y_cfd_i = np.interp(x_common, x_cfd, y_cfd)
        y_exp_i = np.interp(x_common, x_exp, y_exp)

        err = y_cfd_i - y_exp_i
        abs_err = np.abs(err)

        branch_metrics[f"u_rms_{branch_name}"] = float(np.sqrt(np.mean(err**2)))
        branch_metrics[f"u_max_{branch_name}"] = float(np.max(abs_err))

        all_err.append(err)

    if len(all_err) == 0:
        branch_metrics["u_rms_all"] = np.nan
        branch_metrics["u_max_all"] = np.nan
    else:
        err_all = np.concatenate(all_err)
        branch_metrics["u_rms_all"] = float(np.sqrt(np.mean(err_all**2)))
        branch_metrics["u_max_all"] = float(np.max(np.abs(err_all)))

    return branch_metrics


# ============================================================
# Data discovery (Case3 only)
# ============================================================

@dataclass
class Case3Case:
    model: str
    cd_file: Path
    cl_file: Path
    cp_file: Path


def _pick_first_existing(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dir_path.glob(pat))
        for h in hits:
            if h.is_file():
                return h
    return None


def discover_case3_cases(root: Path) -> List[Case3Case]:
    cases: List[Case3Case] = []
    kw_dir = root / "kw" / "case3"
    sa_dir = root / "sa" / "case3"

    cd_pats = ["cd*.out", "Cd*.out", "*cd*.out"]
    cl_pats = ["cl*.out", "Cl*.out", "*cl*.out"]
    kw_cp_pats = ["kw*.txt", "*kw*.txt", "*Cp*.txt", "*.txt"]
    sa_cp_pats = ["sa*.txt", "*sa*.txt", "*Cp*.txt", "*.txt"]

    if kw_dir.exists():
        cd = _pick_first_existing(kw_dir, cd_pats)
        cl = _pick_first_existing(kw_dir, cl_pats)
        cp = _pick_first_existing(kw_dir, kw_cp_pats)
        if cd and cl and cp:
            cases.append(Case3Case("kw", cd, cl, cp))

    if sa_dir.exists():
        cd = _pick_first_existing(sa_dir, cd_pats)
        cl = _pick_first_existing(sa_dir, cl_pats)
        cp = _pick_first_existing(sa_dir, sa_cp_pats)
        if cd and cl and cp:
            cases.append(Case3Case("sa", cd, cl, cp))

    return cases


# ============================================================
# Plotting (same visual style as task2_task4)
# ============================================================

def plot_combined_history(data_map: Dict[str, pd.DataFrame], ylabel: str, title: str, out_png: Path) -> None:
    style_map = {"kw": {"color": "C0", "linestyle": "-"}, "sa": {"color": "C1", "linestyle": "--"}}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in data_map.items():
        st = style_map.get(label, {"color": "C2", "linestyle": "-."})
        ax.plot(df["iteration"], df["value"], linewidth=1.8, color=st["color"], linestyle=st["linestyle"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_metric_bar_grouped(summary: pd.DataFrame, metric: str, title: str, out_png: Path) -> None:
    sub = summary[summary["quantity"].isin(["Cl", "Cd"])].copy()
    quantities = ["Cl", "Cd"]
    models = sorted(sub["model"].unique())
    x = np.arange(len(quantities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        vals = []
        for q in quantities:
            row = sub[(sub["model"] == model) & (sub["quantity"] == q)]
            vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        ax.bar(x + i * width - width / 2, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(quantities)
    ax.set_xlabel("Quantity")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_task4_style_cp(
    data_map: Dict[str, pd.DataFrame],
    title: str,
    out_png: Path,
    exp_split: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    style_map = {
        "kw": {"color": "C0", "linestyle": "-", "linewidth": 1.5},
        "sa": {"color": "C1", "linestyle": "--", "linewidth": 1.5},
    }
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, df in data_map.items():
        st = style_map.get(label, {"color": "C6", "linestyle": "-.", "linewidth": 1.2})
        x = df["x"].to_numpy()
        cp = df["cp"].to_numpy()
        segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)

        first = True
        for s in segs:
            ax.plot(
                x[s], cp[s],
                marker="o", markersize=2.5,
                linewidth=st["linewidth"], color=st["color"], linestyle=st["linestyle"],
                label=label if first else None,
            )
            first = False

    if exp_split is not None:
        first = True
        for key in ["pressure", "suction"]:
            if key in exp_split and len(exp_split[key]) > 0:
                dfe = exp_split[key]
                ax.plot(
                    dfe["x"], dfe["cp"],
                    color="k", linestyle="-", linewidth=1.8,
                    label="Exp" if first else None,
                )
                first = False

    ax.invert_yaxis()
    ax.set_xlabel("x/c (Position)")
    ax.set_ylabel("Cp")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_per_model_uncertainty(summary: pd.DataFrame, out_png: Path) -> None:
    df = summary.copy().sort_values("u_rms_all").reset_index(drop=True)
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["u_rms_all"], width, label="RMS vs Exp")
    ax.bar(x + width/2, df["u_max_all"], width, label="Max abs error vs Exp")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Validation error vs Exp")
    ax.set_title("Case3: Per-model uncertainty relative to experimental Cp")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ============================================================
# Main workflow (Case3 only)
# ============================================================

def run_case3(
    root: Path,
    outdir: Path,
    exp_cp_path: Path,
    exp_usecols=(0, 1),
    tail: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cases = discover_case3_cases(root)
    if not cases:
        raise RuntimeError(f"No Case3 cases found under: {root}/kw/case3 and {root}/sa/case3")

    exp_df = load_exp_cp_data(exp_cp_path, usecols=exp_usecols)
    exp_split = split_exp_surfaces_by_cp(exp_df)

    # maps for combined plots
    cl_map: Dict[str, pd.DataFrame] = {}
    cd_map: Dict[str, pd.DataFrame] = {}
    cp_map: Dict[str, pd.DataFrame] = {}

    iter_rows: List[Dict[str, object]] = []
    uq_rows: List[Dict[str, object]] = []

    for case in cases:
        cl_df = parse_fluent_rfile(case.cl_file)
        cd_df = parse_fluent_rfile(case.cd_file)
        cp_df = parse_tecplot_xy_pairs(case.cp_file)

        cl_map[case.model] = cl_df
        cd_map[case.model] = cd_df
        cp_map[case.model] = cp_df

        # tail stats for Cl/Cd
        iter_rows.append({"model": case.model, "quantity": "Cl", "file": str(case.cl_file.relative_to(root)), **tail_stats(cl_df["value"], tail=tail)})
        iter_rows.append({"model": case.model, "quantity": "Cd", "file": str(case.cd_file.relative_to(root)), **tail_stats(cd_df["value"], tail=tail)})

        # Cp UQ vs exp
        u = compute_single_model_uncertainty_vs_exp(cp_df, exp_split, n_grid=300)
        uq_rows.append({"label": f"{case.model}_case3", "model": case.model, "file": str(case.cp_file.relative_to(root)), "n_points": int(len(cp_df)), **u})

    iter_summary = pd.DataFrame(iter_rows)
    cp_uq_summary = pd.DataFrame(uq_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    iter_summary.to_csv(outdir / "case3_tail_stats.csv", index=False)
    cp_uq_summary.to_csv(outdir / "case3_cp_uncertainty.csv", index=False)

    # combined figures
    plot_combined_history(cl_map, "Cl", "Case3: Cl convergence (kw vs sa)", outdir / "figures" / "case3" / "combined" / "case3_Cl_kw_vs_sa.png")
    plot_combined_history(cd_map, "Cd", "Case3: Cd convergence (kw vs sa)", outdir / "figures" / "case3" / "combined" / "case3_Cd_kw_vs_sa.png")
    plot_task4_style_cp(cp_map, "Case3: Cp comparison (kw/sa) with experimental data", outdir / "figures" / "case3" / "combined" / "case3_Cp_kw_sa_vs_exp.png", exp_split=exp_split)

    # iterative uncertainty bars
    plot_metric_bar_grouped(iter_summary, "std_tail", "Case3 iterative uncertainty: tail standard deviation", outdir / "figures" / "case3" / "combined" / "case3_std_tail_bar.png")
    plot_metric_bar_grouped(iter_summary, "range_tail", "Case3 iterative uncertainty: tail range", outdir / "figures" / "case3" / "combined" / "case3_range_tail_bar.png")
    plot_metric_bar_grouped(iter_summary, "abs_last_minus_mean", "Case3 iterative error: |last - mean_tail|", outdir / "figures" / "case3" / "combined" / "case3_abs_last_minus_mean_bar.png")

    # per-model Cp uncertainty bar
    plot_per_model_uncertainty(cp_uq_summary, outdir / "figures" / "case3" / "combined" / "case3_cp_uncertainty_per_model.png")

    return iter_summary, cp_uq_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Case3-only postprocess (Task4-style validation): Cp vs Exp + per-model uncertainty, plus Cl/Cd tail stats.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing kw/ and sa/ (expects kw/case3, sa/case3)")
    parser.add_argument("--out", type=str, default="result_case3_task4", help="Output directory")
    parser.add_argument("--tail", type=int, default=200, help="Last N iterations used for tail statistics (Cl/Cd)")
    parser.add_argument("--exp_cp", type=str, required=True, help="Experimental Cp file path")
    parser.add_argument("--exp_usecols", type=str, default="0,1", help="Which two columns in exp_cp are x/c and Cp")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    exp_path = Path(args.exp_cp).resolve()
    exp_usecols = parse_usecols(args.exp_usecols)

    run_case3(root=root, outdir=outdir, exp_cp_path=exp_path, exp_usecols=exp_usecols, tail=args.tail)

    print(f"[OK] Case3 tail stats      : {outdir / 'case3_tail_stats.csv'}")
    print(f"[OK] Case3 Cp uncertainty : {outdir / 'case3_cp_uncertainty.csv'}")
    print(f"[OK] Figures dir           : {outdir / 'figures' / 'case3' / 'combined'}")


if __name__ == "__main__":
    main()