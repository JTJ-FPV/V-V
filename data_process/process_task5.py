#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example:
# python process_task5.py --root ./data --out ./result/task5 \
#   --exp_cp ./data/Experiment_Cp_AoA_5_Freestream.dat --exp_usecols 0,1 \
#   --kw_ncells 23976,95904,383616 --sa_ncells 23976,95904,383616
# python process_task5.py --root ./data --out ./result/task5 --exp_cp ./data/Experiment_Cp_AoA_5_Freestream.dat --exp_usecols 0,1 --kw_ncells 23976,95904,383616 --sa_ncells 23976,95904,383616
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def parse_float_triplet(s: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values, e.g. 1.0,0.7071,0.5")
    return parts[0], parts[1], parts[2]


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

def split_cp_surfaces_by_median(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cp_med = np.median(df["cp"].to_numpy())
    idx_suction = df["cp"] <= cp_med
    idx_pressure = ~idx_suction

    n = len(df)
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = df["cp"] < 0.0
        idx_pressure = ~idx_suction

    suction = (
        df.loc[idx_suction].copy()
        .sort_values("x")
        .drop_duplicates(subset="x")
        .reset_index(drop=True)
    )
    pressure = (
        df.loc[idx_pressure].copy()
        .sort_values("x")
        .drop_duplicates(subset="x")
        .reset_index(drop=True)
    )

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


def h_from_ncells_2d(ncells: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(float(n) ** (-0.5) for n in ncells)


# ============================================================
# Richardson / GCI
# ============================================================

def safe_sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def apparent_order_three_grid(f1: float, f2: float, f3: float, r21: float, r32: float) -> float:
    """
    Grid 1 = finest, 2 = medium, 3 = coarsest
    """
    eps32 = f3 - f2
    eps21 = f2 - f1

    if abs(eps32) < 1e-30 or abs(eps21) < 1e-30:
        return np.nan

    s = safe_sign(eps32 / eps21)
    if s == 0:
        return np.nan

    p = abs(np.log(abs(eps32 / eps21)) / np.log(r21))
    for _ in range(100):
        denom = (r32**p - s)
        if abs(denom) < 1e-30:
            return np.nan
        numerator = abs(eps32 / eps21) + (r21**p - s) / denom
        if numerator <= 0:
            return np.nan
        p_new = abs(np.log(numerator) / np.log(r21))
        if abs(p_new - p) < 1e-10:
            return p_new
        p = p_new
    return p


def richardson_extrapolation(f1: float, f2: float, r21: float, p: float) -> float:
    denom = r21**p - 1.0
    if abs(denom) < 1e-30:
        return np.nan
    return f1 + (f1 - f2) / denom


def gci_fine(f1: float, f2: float, r21: float, p: float, Fs: float = 1.25) -> float:
    if abs(f1) < 1e-30:
        return np.nan
    denom = r21**p - 1.0
    if abs(denom) < 1e-30:
        return np.nan
    ea21 = abs((f1 - f2) / f1)
    return 100.0 * Fs * ea21 / denom


def asymptotic_ratio(gci21: float, gci32: float, r21: float, p: float) -> float:
    denom = gci21 * (r21**p)
    if abs(denom) < 1e-30:
        return np.nan
    return gci32 / denom


def scalar_grid_convergence(
    fine: float,
    medium: float,
    coarse: float,
    h_fine: float,
    h_medium: float,
    h_coarse: float,
    Fs: float = 1.25,
) -> Dict[str, float]:
    grid = sorted(
        [(h_fine, fine), (h_medium, medium), (h_coarse, coarse)],
        key=lambda x: x[0]
    )
    h1, f1 = grid[0]
    h2, f2 = grid[1]
    h3, f3 = grid[2]

    r21 = h2 / h1
    r32 = h3 / h2

    p = apparent_order_three_grid(f1, f2, f3, r21, r32)
    if not np.isfinite(p):
        return {
            "f1": f1, "f2": f2, "f3": f3,
            "h1": h1, "h2": h2, "h3": h3,
            "r21": r21, "r32": r32,
            "p": np.nan,
            "f_ext": np.nan,
            "ea21_percent": np.nan,
            "ea32_percent": np.nan,
            "gci21_percent": np.nan,
            "gci32_percent": np.nan,
            "asymptotic_ratio": np.nan,
        }

    f_ext = richardson_extrapolation(f1, f2, r21, p)
    ea21 = 100.0 * abs((f1 - f2) / f1) if abs(f1) > 1e-30 else np.nan
    ea32 = 100.0 * abs((f2 - f3) / f2) if abs(f2) > 1e-30 else np.nan
    gci21 = gci_fine(f1, f2, r21, p, Fs=Fs)
    gci32 = gci_fine(f2, f3, r32, p, Fs=Fs)
    ar = asymptotic_ratio(gci21, gci32, r21, p)

    return {
        "f1": f1, "f2": f2, "f3": f3,
        "h1": h1, "h2": h2, "h3": h3,
        "r21": r21, "r32": r32,
        "p": p,
        "f_ext": f_ext,
        "ea21_percent": ea21,
        "ea32_percent": ea32,
        "gci21_percent": gci21,
        "gci32_percent": gci32,
        "asymptotic_ratio": ar,
    }


# ============================================================
# Cp vs experiment metrics
# ============================================================

def compute_single_model_uncertainty_vs_exp(
    cp_df: pd.DataFrame,
    exp_split: Dict[str, pd.DataFrame],
    n_grid: int = 300,
) -> Dict[str, float]:
    """
    Return:
      u_rms_all
      u_max_all
      u_rms_pressure
      u_max_pressure
      u_rms_suction
      u_max_suction
    """
    cp_split = split_cp_surfaces_by_median(cp_df)

    all_err = []
    branch_metrics = {}

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

        u_rms = float(np.sqrt(np.mean(err**2)))
        u_max = float(np.max(abs_err))

        branch_metrics[f"u_rms_{branch_name}"] = u_rms
        branch_metrics[f"u_max_{branch_name}"] = u_max
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
# Data structures
# ============================================================

@dataclass
class Task5Case:
    model: str
    variant: str   # baseline / refine1 / refine2
    cd_file: Path
    cl_file: Path
    cp_file: Path


def discover_task5_cases(root: Path) -> List[Task5Case]:
    cases: List[Task5Case] = []

    kw_dir = root / "kw" / "task5"
    if kw_dir.exists():
        mapping = {
            "baseline": (
                kw_dir / "cd_1_1_1_AUSM.out",
                kw_dir / "cl_1_1_1_AUSM.out",
                kw_dir / "kw_1_1_1_AUSM.txt",
            ),
            "refine1": (
                kw_dir / "cd_1_1_1_refine1.out",
                kw_dir / "cl_1_1_1_refine1.out",
                kw_dir / "kw_1_1_1_refine1.txt",
            ),
            "refine2": (
                kw_dir / "cd_1_1_1_refine2.out",
                kw_dir / "cl_1_1_1_refine2.out",
                kw_dir / "kw_1_1_1_refine2.txt",
            ),
        }
        for variant, (cdf, clf, cpf) in mapping.items():
            if cdf.exists() and clf.exists() and cpf.exists():
                cases.append(Task5Case("kw", variant, cdf, clf, cpf))

    sa_dir = root / "sa" / "task5"
    if sa_dir.exists():
        mapping = {
            "baseline": (
                sa_dir / "cd_1_2_AUSM.out",
                sa_dir / "cl_1_2_AUSM.out",
                sa_dir / "sa_1_2_AUSM.txt",
            ),
            "refine1": (
                sa_dir / "cd_1_2_refine1.out",
                sa_dir / "cl_1_2_refine1.out",
                sa_dir / "sa_1_2_refine1.txt",
            ),
            "refine2": (
                sa_dir / "cd_1_2_refine2.out",
                sa_dir / "cl_1_2_refine2.out",
                sa_dir / "sa_1_2_refine2.txt",
            ),
        }
        for variant, (cdf, clf, cpf) in mapping.items():
            if cdf.exists() and clf.exists() and cpf.exists():
                cases.append(Task5Case("sa", variant, cdf, clf, cpf))

    return cases


# ============================================================
# Plotting
# ============================================================

TASK5_STYLE_MAP = {
    "baseline": {"color": "C0", "linestyle": "-",  "linewidth": 1.8},
    "refine1":  {"color": "C1", "linestyle": "--", "linewidth": 1.8},
    "refine2":  {"color": "C2", "linestyle": "-.", "linewidth": 1.8},
}

EXP_STYLE = {"color": "k", "linestyle": "-", "linewidth": 1.8}


def plot_three_histories(data_map: Dict[str, pd.DataFrame], ylabel: str, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ["baseline", "refine1", "refine2"]:
        if variant not in data_map:
            continue
        df = data_map[variant]
        style = TASK5_STYLE_MAP[variant]
        ax.plot(
            df["iteration"], df["value"],
            color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
            label=variant,
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


def plot_cp_three_meshes(
    cp_map: Dict[str, pd.DataFrame],
    title: str,
    out_png: Path,
    exp_df: Optional[pd.DataFrame] = None
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for variant in ["baseline", "refine1", "refine2"]:
        if variant not in cp_map:
            continue
        df = cp_map[variant]
        style = TASK5_STYLE_MAP[variant]

        x = df["x"].to_numpy()
        cp = df["cp"].to_numpy()
        segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)

        first = True
        for s in segs:
            ax.plot(
                x[s], cp[s],
                marker="o", markersize=2.5,
                color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
                label=variant if first else None,
            )
            first = False

    if exp_df is not None:
        exp_split = split_cp_surfaces_by_median(exp_df)
        first = True
        for key in ["pressure", "suction"]:
            d = exp_split[key]
            ax.plot(
                d["x"], d["cp"],
                color=EXP_STYLE["color"], linestyle=EXP_STYLE["linestyle"], linewidth=EXP_STYLE["linewidth"],
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


def plot_gci_bar(summary_df: pd.DataFrame, quantity: str, out_png: Path) -> None:
    sub = summary_df[summary_df["quantity"] == quantity].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    width = 0.35

    ax.bar(x - width/2, sub["gci21_percent"], width, label="GCI fine-medium")
    ax.bar(x + width/2, sub["gci32_percent"], width, label="GCI medium-coarse")

    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"])
    ax.set_ylabel("GCI (%)")
    ax.set_title(f"Task5: {quantity} Grid Convergence Index")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_scalar_vs_mesh(summary_df: pd.DataFrame, quantity: str, out_png: Path) -> None:
    sub = summary_df[summary_df["quantity"] == quantity].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sub.iterrows():
        xs = [row["h3"], row["h2"], row["h1"]]
        ys = [row["f3"], row["f2"], row["f1"]]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=row["model"])

        if np.isfinite(row["f_ext"]):
            ax.axhline(row["f_ext"], linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Representative grid size h")
    ax.set_ylabel(quantity)
    ax.set_title(f"Task5: {quantity} vs mesh size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_metric_bar_by_variant(summary: pd.DataFrame, metric: str, title: str, out_png: Path) -> None:
    variants = ["baseline", "refine1", "refine2"]
    models = ["kw", "sa"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.22

    for i, variant in enumerate(variants):
        vals = []
        for model in models:
            row = summary[(summary["model"] == model) & (summary["variant"] == variant)]
            vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_metric_trend(summary: pd.DataFrame, metric: str, title: str, out_png: Path) -> None:
    variants = ["baseline", "refine1", "refine2"]
    models = ["kw", "sa"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        vals = []
        for variant in variants:
            row = summary[(summary["model"] == model) & (summary["variant"] == variant)]
            vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        ax.plot(variants, vals, marker="o", linewidth=1.8, label=model)

    ax.set_xlabel("Mesh level")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ============================================================
# Main processing
# ============================================================

def run_task5(
    root: Path,
    outdir: Path,
    kw_h: Optional[Tuple[float, float, float]] = None,
    sa_h: Optional[Tuple[float, float, float]] = None,
    kw_ncells: Optional[Tuple[float, float, float]] = None,
    sa_ncells: Optional[Tuple[float, float, float]] = None,
    exp_cp_path: Optional[Path] = None,
    exp_usecols=(0, 1),
    tail: int = 200,
    Fs: float = 1.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = discover_task5_cases(root)
    if not cases:
        raise RuntimeError(f"No Task5 cases found under: {root}")

    outdir.mkdir(parents=True, exist_ok=True)

    if kw_h is None and kw_ncells is not None:
        kw_h = h_from_ncells_2d(kw_ncells)
    if sa_h is None and sa_ncells is not None:
        sa_h = h_from_ncells_2d(sa_ncells)

    exp_df = None
    exp_split = None
    if exp_cp_path is not None and exp_cp_path.exists():
        exp_df = load_exp_cp_data(exp_cp_path, usecols=exp_usecols)
        exp_split = split_cp_surfaces_by_median(exp_df)

    parsed: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {"kw": {}, "sa": {}}

    # -------------------------
    # Parse all cases
    # -------------------------
    tail_rows = []
    validation_rows = []

    for case in cases:
        cd_df = parse_fluent_rfile(case.cd_file)
        cl_df = parse_fluent_rfile(case.cl_file)
        cp_df = parse_tecplot_xy_pairs(case.cp_file)

        parsed[case.model][case.variant] = {
            "Cd": cd_df,
            "Cl": cl_df,
            "Cp": cp_df,
        }

        tail_rows.append({
            "model": case.model,
            "variant": case.variant,
            "quantity": "Cd",
            **tail_stats(cd_df["value"], tail=tail),
        })
        tail_rows.append({
            "model": case.model,
            "variant": case.variant,
            "quantity": "Cl",
            **tail_stats(cl_df["value"], tail=tail),
        })

        if exp_split is not None:
            uq = compute_single_model_uncertainty_vs_exp(cp_df, exp_split, n_grid=300)
            validation_rows.append({
                "model": case.model,
                "variant": case.variant,
                **uq,
            })

    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(outdir / "task5_tail_stats.csv", index=False)

    validation_df = pd.DataFrame(validation_rows)
    if not validation_df.empty:
        validation_df.to_csv(outdir / "task5_validation_cp_vs_exp.csv", index=False)

    # -------------------------
    # Per-model figures
    # -------------------------
    for model in ["kw", "sa"]:
        if model not in parsed or not parsed[model]:
            continue

        cd_map = {k: v["Cd"] for k, v in parsed[model].items()}
        cl_map = {k: v["Cl"] for k, v in parsed[model].items()}
        cp_map = {k: v["Cp"] for k, v in parsed[model].items()}

        plot_three_histories(
            cd_map, "Cd",
            f"Task5 {model.upper()}: Cd convergence on three meshes",
            outdir / "figures" / model / f"{model}_task5_Cd_three_meshes.png"
        )
        plot_three_histories(
            cl_map, "Cl",
            f"Task5 {model.upper()}: Cl convergence on three meshes",
            outdir / "figures" / model / f"{model}_task5_Cl_three_meshes.png"
        )
        plot_cp_three_meshes(
            cp_map,
            f"Task5 {model.upper()}: Cp comparison on three meshes",
            outdir / "figures" / model / f"{model}_task5_Cp_three_meshes.png",
            exp_df=exp_df,
        )

    # -------------------------
    # Richardson / GCI for Cl and Cd
    # -------------------------
    summary_rows = []
    model_h_map = {"kw": kw_h, "sa": sa_h}

    for model in ["kw", "sa"]:
        hvals = model_h_map.get(model)
        if hvals is None:
            continue
        if model not in parsed:
            continue

        needed = ["baseline", "refine1", "refine2"]
        if not all(v in parsed[model] for v in needed):
            continue

        h_coarse, h_medium, h_fine = hvals

        for quantity in ["Cd", "Cl"]:
            f_coarse = float(parsed[model]["baseline"][quantity]["value"].iloc[-1])
            f_medium = float(parsed[model]["refine1"][quantity]["value"].iloc[-1])
            f_fine = float(parsed[model]["refine2"][quantity]["value"].iloc[-1])

            res = scalar_grid_convergence(
                fine=f_fine,
                medium=f_medium,
                coarse=f_coarse,
                h_fine=h_fine,
                h_medium=h_medium,
                h_coarse=h_coarse,
                Fs=Fs,
            )
            summary_rows.append({
                "model": model,
                "quantity": quantity,
                **res,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "task5_richardson_summary.csv", index=False)

    for quantity in ["Cd", "Cl"]:
        plot_gci_bar(
            summary_df, quantity,
            outdir / "figures" / "combined" / f"task5_{quantity}_gci_bar.png"
        )
        plot_scalar_vs_mesh(
            summary_df, quantity,
            outdir / "figures" / "combined" / f"task5_{quantity}_vs_h.png"
        )

    # -------------------------
    # Cp validation figures
    # -------------------------
    if not validation_df.empty:
        metric_title_map = {
            "u_rms_all": "Task5: u_rms_all vs experiment",
            "u_max_all": "Task5: u_max_all vs experiment",
            "u_rms_pressure": "Task5: u_rms_pressure vs experiment",
            "u_max_pressure": "Task5: u_max_pressure vs experiment",
            "u_rms_suction": "Task5: u_rms_suction vs experiment",
            "u_max_suction": "Task5: u_max_suction vs experiment",
        }

        for metric, title in metric_title_map.items():
            plot_metric_bar_by_variant(
                validation_df,
                metric=metric,
                title=title,
                out_png=outdir / "figures" / "combined" / f"task5_{metric}_bar.png"
            )
            plot_metric_trend(
                validation_df,
                metric=metric,
                title=f"Task5: {metric} across three meshes",
                out_png=outdir / "figures" / "combined" / f"task5_{metric}_trend.png"
            )

    return tail_df, summary_df, validation_df


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process Task5 data: grid convergence + Richardson extrapolation + GCI + Cp validation vs experiment."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory containing kw/ and sa/")
    parser.add_argument("--out", type=str, default="result/task5", help="Output directory")
    parser.add_argument("--tail", type=int, default=200, help="Last N iterations used for tail statistics")
    parser.add_argument("--Fs", type=float, default=1.25, help="Safety factor for GCI")

    parser.add_argument("--kw_h", type=str, default="", help="Three mesh sizes for kw: coarse,medium,fine")
    parser.add_argument("--sa_h", type=str, default="", help="Three mesh sizes for sa: coarse,medium,fine")
    parser.add_argument("--kw_ncells", type=str, default="", help="Three cell counts for kw: coarse,medium,fine")
    parser.add_argument("--sa_ncells", type=str, default="", help="Three cell counts for sa: coarse,medium,fine")

    parser.add_argument("--exp_cp", type=str, default="", help="Optional experimental Cp file path")
    parser.add_argument("--exp_usecols", type=str, default="0,1", help="Columns in exp_cp for x/c and Cp")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    kw_h = parse_float_triplet(args.kw_h) if args.kw_h else None
    sa_h = parse_float_triplet(args.sa_h) if args.sa_h else None
    kw_ncells = parse_float_triplet(args.kw_ncells) if args.kw_ncells else None
    sa_ncells = parse_float_triplet(args.sa_ncells) if args.sa_ncells else None

    exp_path = Path(args.exp_cp).resolve() if args.exp_cp else None
    exp_usecols = parse_usecols(args.exp_usecols)

    tail_df, summary_df, validation_df = run_task5(
        root=root,
        outdir=outdir,
        kw_h=kw_h,
        sa_h=sa_h,
        kw_ncells=kw_ncells,
        sa_ncells=sa_ncells,
        exp_cp_path=exp_path,
        exp_usecols=exp_usecols,
        tail=args.tail,
        Fs=args.Fs,
    )

    print(f"[OK] Richardson summary : {outdir / 'task5_richardson_summary.csv'}")
    print(f"[OK] Tail stats         : {outdir / 'task5_tail_stats.csv'}")
    if not validation_df.empty:
        print(f"[OK] Cp validation      : {outdir / 'task5_validation_cp_vs_exp.csv'}")
    print(f"[OK] Figures dir        : {outdir / 'figures'}")

    if not summary_df.empty:
        print("\n===== Richardson / GCI summary =====")
        print(summary_df.to_string(index=False))

    if not validation_df.empty:
        print("\n===== Cp vs experiment summary =====")
        print(validation_df.to_string(index=False))


if __name__ == "__main__":
    main()