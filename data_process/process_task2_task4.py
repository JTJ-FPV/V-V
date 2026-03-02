#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python process_task2_task4.py --root .\data\ --out .\result\task2_task4\ --exp_cp .\data\Experiment_Cp_AoA_5_Freestream.dat --exp_usecols 0,1

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Parsers
# ============================================================
def load_exp_cp_data(path: Path, usecols=(0, 1)) -> pd.DataFrame:
    """
    Load experimental Cp data from a plain text file.
    Default columns: x/c, Cp
    """
    data = np.loadtxt(path, usecols=usecols)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Experimental Cp file must contain at least two columns: {path}")

    return pd.DataFrame({"x": data[:, 0].astype(float), "cp": data[:, 1].astype(float)})


def split_exp_surfaces_by_cp(exp: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split mixed experimental Cp points into two branches using median Cp.
    This avoids connecting upper and lower surfaces incorrectly.
    """
    cp_med = np.median(exp["cp"].to_numpy())
    idx_suction = exp["cp"] <= cp_med
    idx_pressure = ~idx_suction

    # fallback if one side is too small
    n = len(exp)
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = exp["cp"] < 0.0
        idx_pressure = ~idx_suction

    suction = exp.loc[idx_suction].copy().sort_values("x").reset_index(drop=True)
    pressure = exp.loc[idx_pressure].copy().sort_values("x").reset_index(drop=True)

    return {"suction": suction, "pressure": pressure}

def split_cp_surfaces_by_median(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a mixed Cp curve into two branches using Cp median.
    Used for CFD curves as well as experimental curves.
    """
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


def parse_usecols(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("exp_usecols must be like '0,1'")
    return (int(parts[0]), int(parts[1]))


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
# Statistics
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
    """
    For one CFD Cp curve, compute uncertainty metrics against experimental Cp.

    Returns:
      u_rms_all      : RMS error over both branches
      u_max_all      : max abs error over both branches
      u_rms_pressure : RMS error on pressure branch
      u_rms_suction  : RMS error on suction branch
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

        # 共有区间
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

def compute_model_uncertainty_envelope(
    cp_map: Dict[str, pd.DataFrame],
    n_grid: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scheme B:
      U_model(x) = (max_i Cp_i(x) - min_i Cp_i(x)) / 2

    We split each CFD Cp file into suction / pressure branches,
    interpolate each branch onto a common x-grid, then compute
    the half-envelope width at each x.

    Returns:
      branch_df: rows for each branch and x-grid point
      global_df: one-row summary with mean/max uncertainty
    """
    branch_results = []

    for branch_name in ["pressure", "suction"]:
        interp_curves = []
        labels = []
        common_x = None

        for label, df in cp_map.items():
            split = split_cp_surfaces_by_median(df)
            bdf = split[branch_name].copy()

            # 去重并按 x 升序
            bdf = bdf.sort_values("x").drop_duplicates(subset="x", keep="mean" if False else "first")
            x = bdf["x"].to_numpy()
            y = bdf["cp"].to_numpy()

            if len(x) < 2:
                continue

            # 每条曲线自己的有效区间
            x_min = np.min(x)
            x_max = np.max(x)

            if common_x is None:
                common_x = np.linspace(x_min, x_max, n_grid)
            else:
                # 只保留所有模型都共有的区间
                common_x = common_x[(common_x >= x_min) & (common_x <= x_max)]

            interp_curves.append((label, x, y))
            labels.append(label)

        if common_x is None or len(common_x) < 2 or len(interp_curves) < 2:
            continue

        # 重新在共有区间插值
        values = []
        valid_labels = []
        for label, x, y in interp_curves:
            mask = (common_x >= np.min(x)) & (common_x <= np.max(x))
            if np.count_nonzero(mask) < 2:
                continue

            y_interp = np.interp(common_x, x, y)
            values.append(y_interp)
            valid_labels.append(label)

        if len(values) < 2:
            continue

        vals = np.vstack(values)  # shape: [n_models, n_x]
        cp_min = np.min(vals, axis=0)
        cp_max = np.max(vals, axis=0)
        u_model = 0.5 * (cp_max - cp_min)

        df_branch = pd.DataFrame({
            "branch": branch_name,
            "x": common_x,
            "cp_min": cp_min,
            "cp_max": cp_max,
            "u_model": u_model,
        })
        branch_results.append(df_branch)

    if not branch_results:
        raise RuntimeError("Failed to compute model uncertainty envelope: not enough overlapping Cp branches.")

    branch_df = pd.concat(branch_results, ignore_index=True)

    global_df = pd.DataFrame([{
        "u_model_mean": float(branch_df["u_model"].mean()),
        "u_model_max": float(branch_df["u_model"].max()),
        "n_points": int(len(branch_df)),
    }])

    return branch_df, global_df

# ============================================================
# Data structures
# ============================================================

@dataclass
class Task2Case:
    model: str
    cd_file: Path
    cl_file: Path
    cp_file: Path


@dataclass
class CpCase:
    label: str
    file: Path


def discover_task2_cases(root: Path) -> List[Task2Case]:
    cases: List[Task2Case] = []

    kw_dir = root / "kw" / "task2"
    sa_dir = root / "sa" / "task2"

    if kw_dir.exists():
        cases.append(
            Task2Case(
                model="kw",
                cd_file=kw_dir / "cd_1_1_1_AUSM.out",
                cl_file=kw_dir / "cl_1_1_1_AUSM.out",
                cp_file=kw_dir / "kw_1_1_1_AUSM.txt",
            )
        )

    if sa_dir.exists():
        cases.append(
            Task2Case(
                model="sa",
                cd_file=sa_dir / "cd_1_2_AUSM.out",
                cl_file=sa_dir / "cl_1_2_AUSM.out",
                cp_file=sa_dir / "sa_1_2_AUSM.txt",
            )
        )

    return cases


def discover_task4_cp_cases(root: Path) -> List[CpCase]:
    cases: List[CpCase] = []

    # baseline models from task4
    kw_file = root / "kw" / "task4" / "kw_1_1_1_AUSM.txt"
    sa_file = root / "sa" / "task4" / "sa_1_2_AUSM.txt"

    if kw_file.exists():
        cases.append(CpCase(label="kw_AUSM_111", file=kw_file))
    if sa_file.exists():
        cases.append(CpCase(label="sa_AUSM_12", file=sa_file))

    # RNG from teammate
    rng_dir = root / "kw" / "task4"
    rng_names = [
        "RNG_5_first_AUSM.txt",
        "RNG_5_first_Roe.txt",
        "RNG_5_second_AUSM.txt",
        "RNG_5_second_Roe.txt",
    ]
    for name in rng_names:
        f = rng_dir / name
        if f.exists():
            cases.append(CpCase(label=name.replace(".txt", ""), file=f))

    return cases


# ============================================================
# Plotting helpers
# ============================================================

def plot_task4_single_model_uncertainty(summary: pd.DataFrame, out_png: Path) -> None:
    """
    Plot one uncertainty value per model.
    """
    df = summary.copy()

    # 按整体 RMS 从小到大排序
    df = df.sort_values("u_rms_all").reset_index(drop=True)

    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width/2, df["u_rms_all"], width, label="RMS vs Exp")
    ax.bar(x + width/2, df["u_max_all"], width, label="Max abs error vs Exp")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Model uncertainty")
    ax.set_title("Task4: Per-model uncertainty relative to experimental Cp")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_task4_model_uncertainty(
    branch_df: pd.DataFrame,
    out_png: Path,
    title: str = "Task4: Model uncertainty envelope based on Cp spread",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    style_map = {
        "pressure": {"color": "C0", "label": "Pressure branch"},
        "suction": {"color": "C1", "label": "Suction branch"},
    }

    for branch_name in ["pressure", "suction"]:
        sub = branch_df[branch_df["branch"] == branch_name].copy()
        if sub.empty:
            continue

        style = style_map.get(branch_name, {"color": "C2", "label": branch_name})

        ax.plot(
            sub["x"],
            sub["u_model"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )
        ax.fill_between(
            sub["x"],
            0.0,
            sub["u_model"],
            color=style["color"],
            alpha=0.20,
        )

    ax.set_xlabel("x/c (Position)")
    ax.set_ylabel(r"$U_{\mathrm{model}}(x)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_single_history(df: pd.DataFrame, ylabel: str, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["iteration"], df["value"], linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_combined_history(data_map: Dict[str, pd.DataFrame], ylabel: str, title: str, out_png: Path) -> None:
    style_map = {
        "kw": {"color": "C0", "linestyle": "-"},
        "sa": {"color": "C1", "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in data_map.items():
        style = style_map.get(label, {"color": "C2", "linestyle": "-."})
        ax.plot(
            df["iteration"],
            df["value"],
            linewidth=1.8,
            color=style["color"],
            linestyle=style["linestyle"],
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_single_cp(df: pd.DataFrame, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["x"].to_numpy()
    cp = df["cp"].to_numpy()

    segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)
    first = True
    for s in segs:
        ax.plot(x[s], cp[s], marker="o", markersize=3, linewidth=1.2, label="Cp" if first else None)
        first = False

    ax.invert_yaxis()
    ax.set_xlabel("x/c (Position)")
    ax.set_ylabel("Cp")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_combined_cp_task2(data_map: Dict[str, pd.DataFrame], title: str, out_png: Path) -> None:
    # kw一个颜色，sa一个颜色
    style_map = {
        "kw": {"color": "C0", "linestyle": "-"},
        "sa": {"color": "C1", "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in data_map.items():
        style = style_map.get(label, {"color": "C2", "linestyle": "-."})
        x = df["x"].to_numpy()
        cp = df["cp"].to_numpy()
        segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)

        first = True
        for s in segs:
            ax.plot(
                x[s],
                cp[s],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=style["color"],
                linestyle=style["linestyle"],
                label=label if first else None,
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
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_combined_cp_task4(
    data_map: Dict[str, pd.DataFrame],
    title: str,
    out_png: Path,
    exp_split: Dict[str, pd.DataFrame] | None = None,
) -> None:
    # 固定风格
    style_map = {
        "kw_AUSM_111": {"color": "C0", "linestyle": "-", "linewidth": 1.5},
        "sa_AUSM_12": {"color": "C1", "linestyle": "--", "linewidth": 1.5},
        "RNG_5_first_AUSM": {"color": "C2", "linestyle": "-", "linewidth": 1.5},
        "RNG_5_first_Roe": {"color": "C3", "linestyle": "--", "linewidth": 1.5},
        "RNG_5_second_AUSM": {"color": "C4", "linestyle": "-", "linewidth": 1.5},
        "RNG_5_second_Roe": {"color": "C5", "linestyle": "--", "linewidth": 1.5},
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    # CFD curves
    for label, df in data_map.items():
        style = style_map.get(label, {"color": "C6", "linestyle": "-.", "linewidth": 1.2})
        x = df["x"].to_numpy()
        cp = df["cp"].to_numpy()
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
    if exp_split is not None:
        exp_color = "k"
        exp_ls = "-"
        exp_lw = 1.8

        first = True
        for key in ["pressure", "suction"]:
            if key in exp_split and len(exp_split[key]) > 0:
                df = exp_split[key]
                ax.plot(
                    df["x"],
                    df["cp"],
                    color=exp_color,
                    linestyle=exp_ls,
                    linewidth=exp_lw,
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


# ============================================================
# Task2
# ============================================================

def run_task2(root: Path, outdir: Path, tail: int = 200) -> pd.DataFrame:
    cases = discover_task2_cases(root)
    if not cases:
        raise RuntimeError(f"No Task2 cases found under: {root}")

    summary_rows: List[Dict[str, object]] = []

    cl_map: Dict[str, pd.DataFrame] = {}
    cd_map: Dict[str, pd.DataFrame] = {}
    cp_map: Dict[str, pd.DataFrame] = {}

    for case in cases:
        # Cd
        cd_df = parse_fluent_rfile(case.cd_file)
        cd_map[case.model] = cd_df
        cd_stats = tail_stats(cd_df["value"], tail=tail)
        plot_single_history(cd_df, "Cd", f"Task2 {case.model.upper()}: Cd convergence",
                            outdir / "figures" / "task2" / case.model / f"{case.model}_task2_Cd.png")
        summary_rows.append({"model": case.model, "quantity": "Cd", "file": str(case.cd_file.relative_to(root)), **cd_stats})

        # Cl
        cl_df = parse_fluent_rfile(case.cl_file)
        cl_map[case.model] = cl_df
        cl_stats = tail_stats(cl_df["value"], tail=tail)
        plot_single_history(cl_df, "Cl", f"Task2 {case.model.upper()}: Cl convergence",
                            outdir / "figures" / "task2" / case.model / f"{case.model}_task2_Cl.png")
        summary_rows.append({"model": case.model, "quantity": "Cl", "file": str(case.cl_file.relative_to(root)), **cl_stats})

        # Cp
        cp_df = parse_tecplot_xy_pairs(case.cp_file)
        cp_map[case.model] = cp_df
        plot_single_cp(cp_df, f"Task2 {case.model.upper()}: Cp distribution",
                       outdir / "figures" / "task2" / case.model / f"{case.model}_task2_Cp.png")
        summary_rows.append(
            {
                "model": case.model,
                "quantity": "Cp",
                "file": str(case.cp_file.relative_to(root)),
                "n_total": int(len(cp_df)),
                "last": np.nan,
                "mean_tail": float(cp_df["cp"].mean()),
                "std_tail": float(cp_df["cp"].std(ddof=1) if len(cp_df) > 1 else 0.0),
                "min_tail": float(cp_df["cp"].min()),
                "max_tail": float(cp_df["cp"].max()),
                "range_tail": float(cp_df["cp"].max() - cp_df["cp"].min()),
                "abs_last_minus_mean": np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "task2_summary.csv", index=False)

    coef = summary[summary["quantity"].isin(["Cl", "Cd"])].copy()
    if not coef.empty:
        pivot = coef.pivot_table(
            index=["model"],
            columns="quantity",
            values=["last", "mean_tail", "std_tail", "range_tail", "abs_last_minus_mean"],
            aggfunc="first",
        )
        pivot.to_csv(outdir / "task2_summary_pivot.csv")

    # Combined plots
    plot_combined_history(cl_map, "Cl", "Task2: Cl convergence (kw vs sa)",
                          outdir / "figures" / "task2" / "combined" / "task2_Cl_kw_vs_sa.png")
    plot_combined_history(cd_map, "Cd", "Task2: Cd convergence (kw vs sa)",
                          outdir / "figures" / "task2" / "combined" / "task2_Cd_kw_vs_sa.png")
    plot_combined_cp_task2(cp_map, "Task2: Cp distribution (kw vs sa)",
                           outdir / "figures" / "task2" / "combined" / "task2_Cp_kw_vs_sa.png")

    # Iterative error / uncertainty plots
    plot_metric_bar_grouped(summary, "std_tail",
                            "Task2 iterative uncertainty: tail standard deviation",
                            outdir / "figures" / "task2" / "combined" / "task2_std_tail_bar.png")
    plot_metric_bar_grouped(summary, "range_tail",
                            "Task2 iterative uncertainty: tail range",
                            outdir / "figures" / "task2" / "combined" / "task2_range_tail_bar.png")
    plot_metric_bar_grouped(summary, "abs_last_minus_mean",
                            "Task2 iterative error: |last - mean_tail|",
                            outdir / "figures" / "task2" / "combined" / "task2_abs_last_minus_mean_bar.png")

    return summary


# ============================================================
# Task4
# ============================================================
def run_task4(
    root: Path,
    outdir: Path,
    exp_cp_path: Path | None = None,
    exp_usecols=(0, 1),
) -> pd.DataFrame:
    cp_cases = discover_task4_cp_cases(root)
    if not cp_cases:
        raise RuntimeError(f"No Task4 cp cases found under: {root}")

    cp_map: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    # experimental Cp is required for per-model uncertainty
    exp_split = None
    if exp_cp_path is not None and exp_cp_path.exists():
        exp_df = load_exp_cp_data(exp_cp_path, usecols=exp_usecols)
        exp_split = split_exp_surfaces_by_cp(exp_df)
    else:
        raise RuntimeError("Task4 per-model uncertainty requires experimental Cp data (--exp_cp).")

    for case in cp_cases:
        df = parse_tecplot_xy_pairs(case.file)
        cp_map[case.label] = df

        # basic Cp stats
        row = {
            "label": case.label,
            "file": str(case.file.relative_to(root)),
            "n_points": int(len(df)),
            "cp_mean": float(df["cp"].mean()),
            "cp_std": float(df["cp"].std(ddof=1) if len(df) > 1 else 0.0),
            "cp_min": float(df["cp"].min()),
            "cp_max": float(df["cp"].max()),
        }

        # per-model uncertainty vs experiment
        u = compute_single_model_uncertainty_vs_exp(df, exp_split, n_grid=300)
        row.update(u)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "task4_cp_summary.csv", index=False)

    # 全部模型 + 实验
    plot_combined_cp_task4(
        cp_map,
        "Task4: Cp comparison among SA, kw and RNG variants",
        outdir / "figures" / "task4" / "combined" / "task4_Cp_all_models.png",
        exp_split=exp_split,
    )

    # 只比较 RNG 四条 + 实验
    rng_only = {k: v for k, v in cp_map.items() if k.startswith("RNG_")}
    if rng_only:
        plot_combined_cp_task4(
            rng_only,
            "Task4: Cp comparison among RNG variants",
            outdir / "figures" / "task4" / "combined" / "task4_Cp_rng_only.png",
            exp_split=exp_split,
        )

    # 每个模型一个不确定性图
    plot_task4_single_model_uncertainty(
        summary,
        outdir / "figures" / "task4" / "combined" / "task4_model_uncertainty_per_model.png",
    )

    return summary

# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process Task2 and Task4 data: Task2 kw/sa comparison + Task4 turbulence-model Cp comparison."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory containing kw/ and sa/")
    parser.add_argument("--out", type=str, default="postprocess_task2_task4", help="Output directory")
    parser.add_argument("--tail", type=int, default=200, help="Last N iterations used for Task2 tail statistics")
    parser.add_argument("--exp_cp", type=str, default="", help="Optional experimental Cp file path")
    parser.add_argument("--exp_usecols", type=str, default="0,1", help="Which two columns in exp_cp are x/c and Cp")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    exp_path = Path(args.exp_cp).resolve() if args.exp_cp else None
    exp_usecols = parse_usecols(args.exp_usecols)

    task2_summary = run_task2(root=root, outdir=outdir, tail=args.tail)
    task4_summary = run_task4(root=root, outdir=outdir, exp_cp_path=exp_path, exp_usecols=exp_usecols)

    print(f"[OK] Task2 summary : {outdir / 'task2_summary.csv'}")
    print(f"[OK] Task4 summary : {outdir / 'task4_cp_summary.csv'}")
    print(f"[OK] Task4 UQ fig  : {outdir / 'figures' / 'task4' / 'combined' / 'task4_model_uncertainty_per_model.png'}")
    print(f"[OK] Figures dir   : {outdir / 'figures'}")

if __name__ == "__main__":
    main()