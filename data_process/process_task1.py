#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python process_task1.py --root .\data\ --out .\result\task1\ --exp_cp .\data\Experiment_Cp_AoA_5_Freestream.dat --overlay_exp --exp_usecols 0,1

"""
Task 1 post-processing (Fluent RANS assignment):
- Parse Fluent report files: cd_*.out / cl_*.out  (iteration, value)
- Parse Cp files: sa_*.txt / kw_*.txt (Tecplot/Fluent XY pairs, may contain multiple segments)
- Generate:
  (A) Single-case plots (Cl/Cd convergence, Cp distribution)
  (B) MERGED plots:
      1) Flux merged: AUSM vs Roe under same discretisation scheme
      2) Discretisation merged: different (x,y[,z]) under same flux
  (C) Summary CSVs for report writing
- Experimental Cp overlay:
      Load .dat with numpy.loadtxt, split into upper/pressure + lower/suction surfaces,
      then plot as TWO curves to avoid connecting mixed points.

Folder + naming conventions follow your description:
  sa/task1/{1,2}/...  with files:
    cd_x_y_{Flux}.out, cl_x_y_{Flux}.out, sa_x_y_{Flux}.txt
  kw/task1/{1,2}/...  with files:
    cd_x_y_z_{Flux}.out, cl_x_y_z_{Flux}.out, kw_x_y_z_{Flux}.txt
Where:
  SA: x=Flow order, y=Modified Turbulent Viscosity order
  KW: x=Flow order, y=k order, z=omega order
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parsing utilities
# -----------------------------

def parse_fluent_rfile(path: Path) -> pd.DataFrame:
    """Parse Fluent report file to DataFrame: iteration(int), value(float)."""
    iters, vals = [], []
    pat = re.compile(r"^\s*([+-]?\d+)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                iters.append(int(m.group(1)))
                vals.append(float(m.group(2)))
    if not iters:
        raise ValueError(f"[parse_fluent_rfile] No numeric data found: {path}")
    df = pd.DataFrame({"iteration": iters, "value": vals}).sort_values("iteration").reset_index(drop=True)
    return df


def parse_tecplot_xy_pairs(path: Path) -> pd.DataFrame:
    """
    Parse Tecplot-style XY exported text:
      (title ...)
      ...
      x  y
      ...
    Returns DataFrame: x, y
    """
    xs, ys = [], []
    pair = re.compile(
        r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.replace("\t", " ").strip()
            m = pair.match(line)
            if m:
                xs.append(float(m.group(1)))
                ys.append(float(m.group(2)))
    if not xs:
        raise ValueError(f"[parse_tecplot_xy_pairs] No numeric data found: {path}")
    return pd.DataFrame({"x": xs, "y": ys})


def split_into_segments_by_x_jumps(x: np.ndarray, jump_thresh: float = 0.25) -> List[slice]:
    """
    Cp export may contain multiple curve segments in one file.
    Split whenever |x[i]-x[i-1]| > jump_thresh.
    """
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


def robust_tail_stats(series: pd.Series, tail: int = 200) -> Dict[str, float]:
    """Tail stats for convergence: last value, mean/std in last 'tail' samples."""
    s = series.dropna()
    if len(s) == 0:
        return {"last": np.nan, "mean_tail": np.nan, "std_tail": np.nan}
    t = s.iloc[-tail:] if len(s) > tail else s
    return {
        "last": float(s.iloc[-1]),
        "mean_tail": float(t.mean()),
        "std_tail": float(t.std(ddof=1) if len(t) > 1 else 0.0),
    }


# -----------------------------
# Experimental Cp parsing & splitting
# -----------------------------

def load_exp_cp_data(path: Path, usecols: Tuple[int, int] = (0, 1)) -> pd.DataFrame:
    """
    Load experimental Cp data with numpy.loadtxt.
    Default assumes columns: x/c, Cp.
    If your file has extra columns, set --exp_usecols like "1,2".
    """
    data = np.loadtxt(path, usecols=usecols)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")
    x = data[:, 0].astype(float)
    cp = data[:, 1].astype(float)
    return pd.DataFrame({"x": x, "cp": cp})


def split_exp_surfaces_by_cp(exp: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split mixed experimental Cp points into two surfaces using Cp clustering by median.
    Returns: {"raw","pressure","suction"} with pressure/suction sorted by x.
    """
    x = exp["x"].to_numpy()
    cp = exp["cp"].to_numpy()

    cp_med = np.median(cp)
    idx_suction = cp <= cp_med
    idx_pressure = ~idx_suction

    n = len(cp)
    # Fallback if too imbalanced: sign split
    if min(idx_suction.sum(), idx_pressure.sum()) < max(3, int(0.2 * n)):
        idx_suction = cp < 0.0
        idx_pressure = ~idx_suction

    suction = exp.loc[idx_suction].copy().sort_values("x").reset_index(drop=True)
    pressure = exp.loc[idx_pressure].copy().sort_values("x").reset_index(drop=True)
    return {"raw": exp.reset_index(drop=True), "suction": suction, "pressure": pressure}


def save_split_exp_surfaces(split: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """Save split experimental surfaces for checking / later use."""
    outdir.mkdir(parents=True, exist_ok=True)
    split["suction"].to_csv(outdir / "cp_lower_suction_surface.csv", index=False)
    split["pressure"].to_csv(outdir / "cp_upper_pressure_surface.csv", index=False)


# -----------------------------
# Case metadata from filenames
# -----------------------------

@dataclass(frozen=True)
class CaseMeta:
    turb: str           # "sa" or "kw"
    task: int
    group: int          # task1/{1,2}
    flux: str           # "AUSM" or "Roe"
    x: int              # Flow order
    y: int              # SA: mu_t order; KW: k order
    z: Optional[int]    # KW: omega order; SA: None


def parse_case_meta_from_path(path: Path) -> Optional[CaseMeta]:
    name = path.name
    flux_m = re.search(r"_(AUSM|Roe)\.(out|txt)$", name, re.IGNORECASE)
    if not flux_m:
        return None
    flux = flux_m.group(1)

    parts = [p.lower() for p in path.parts]
    turb = "sa" if "sa" in parts else ("kw" if "kw" in parts else None)
    if turb is None:
        return None

    task = None
    group = None
    for p in parts:
        m = re.match(r"task(\d+)", p)
        if m:
            task = int(m.group(1))
            break
    if task is None:
        return None

    for i, p in enumerate(parts):
        if p == f"task{task}":
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                group = int(parts[i + 1])
            break
    if group is None:
        return None

    if turb == "sa":
        m = re.match(r"^(cd|cl|sa)_(\d+)_(\d+)_(AUSM|Roe)\.(out|txt)$", name, re.IGNORECASE)
        if not m:
            return None
        x = int(m.group(2))
        y = int(m.group(3))
        return CaseMeta(turb=turb, task=task, group=group, flux=flux, x=x, y=y, z=None)

    if turb == "kw":
        m = re.match(r"^(cd|cl|kw)_(\d+)_(\d+)_(\d+)_(AUSM|Roe)\.(out|txt)$", name, re.IGNORECASE)
        if not m:
            return None
        x = int(m.group(2))
        y = int(m.group(3))
        z = int(m.group(4))
        return CaseMeta(turb=turb, task=task, group=group, flux=flux, x=x, y=y, z=z)

    return None


def case_id(meta: CaseMeta) -> str:
    if meta.turb == "sa":
        return f"{meta.turb}_task{meta.task}_g{meta.group}_x{meta.x}_y{meta.y}_{meta.flux}"
    return f"{meta.turb}_task{meta.task}_g{meta.group}_x{meta.x}_y{meta.y}_z{meta.z}_{meta.flux}"


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_convergence(df: pd.DataFrame, title: str, ylab: str, out_png: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["iteration"], df["value"], label=ylab)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_cp_segments_single_color(cp: pd.DataFrame,
                                 title: str,
                                 out_png: Path,
                                 color: str = "C0",
                                 exp_split: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Plot Cp segments in ONE color (avoid rainbow segments).
    Optionally overlay experimental as two surfaces.
    """
    x = cp["x"].to_numpy()
    y = cp["y"].to_numpy()
    segs = split_into_segments_by_x_jumps(x, jump_thresh=0.25)

    fig, ax = plt.subplots()
    first_seg = True
    for s in segs:
        ax.plot(x[s], y[s],
                marker="o", markersize=2, linewidth=1,
                color=color,
                label="CFD" if first_seg else None)
        first_seg = False

    # Experimental overlay: SAME style for both surfaces, single legend entry
    if exp_split is not None:
        p = exp_split.get("pressure", None)
        sfc = exp_split.get("suction", None)

        exp_color = "k"        # or any color you like
        exp_ls = "-"           # same linestyle for both
        exp_lw = 1.8

        first = True
        if p is not None and len(p) > 0:
            ax.plot(p["x"], p["cp"], color=exp_color, linestyle=exp_ls, linewidth=exp_lw,
                    label="Exp" if first else None)
            first = False
        if sfc is not None and len(sfc) > 0:
            ax.plot(sfc["x"], sfc["cp"], color=exp_color, linestyle=exp_ls, linewidth=exp_lw,
                    label="Exp" if first else None)

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


def plot_overlay_iter_curves(items: List[Tuple[str, Optional[pd.DataFrame]]],
                            quantity: str,
                            out_png: Path,
                            title: str) -> None:
    fig, ax = plt.subplots()
    for lab, df in items:
        if df is None or len(df) == 0:
            continue
        ax.plot(df["iteration"], df["value"], label=lab)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(quantity)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_overlay_cp(items: List[Tuple[str, Optional[pd.DataFrame]]],
                    out_png: Path,
                    title: str,
                    exp_split: Optional[Dict[str, pd.DataFrame]] = None,
                    jump_thresh: float = 0.25) -> None:
    """
    Overlay multiple CFD Cp curves (each may have multiple segments).
    Fix rainbow issue by using ONE color per CFD curve (per label), across all its segments.
    Overlay experimental as two surfaces (lines) to avoid mixed-point connection.
    """
    fig, ax = plt.subplots()

    # Matplotlib default cycle colors (C0..); we pick one per CFD curve
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

    cidx = 0
    for lab, cp in items:
        if cp is None or len(cp) == 0:
            continue
        color = color_cycle[cidx % len(color_cycle)]
        cidx += 1

        x = cp["x"].to_numpy()
        y = cp["y"].to_numpy()
        segs = split_into_segments_by_x_jumps(x, jump_thresh=jump_thresh)

        first = True
        for s in segs:
            ax.plot(x[s], y[s],
                    marker="o", markersize=2, linewidth=1,
                    color=color,
                    label=lab if first else None)
            first = False

    # Experimental overlay: two surfaces
    # if exp_split is not None:
    #     p = exp_split.get("pressure", None)
    #     sfc = exp_split.get("suction", None)
    #     if p is not None and len(p) > 0:
    #         ax.plot(p["x"], p["cp"], linewidth=1.2, label="Exp upper/pressure")
    #     if sfc is not None and len(sfc) > 0:
    #         ax.plot(sfc["x"], sfc["cp"], linewidth=1.2, label="Exp lower/suction")

    # Experimental overlay: SAME style for both surfaces, single legend entry
    if exp_split is not None:
        p = exp_split.get("pressure", None)
        sfc = exp_split.get("suction", None)

        exp_color = "k"
        exp_ls = "-"
        exp_lw = 1.8

        first = True
        if p is not None and len(p) > 0:
            ax.plot(p["x"], p["cp"], color=exp_color, linestyle=exp_ls, linewidth=exp_lw,
                    label="Exp" if first else None)
            first = False
        if sfc is not None and len(sfc) > 0:
            ax.plot(sfc["x"], sfc["cp"], color=exp_color, linestyle=exp_ls, linewidth=exp_lw,
                    label="Exp" if first else None)

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


# -----------------------------
# File discovery
# -----------------------------

def find_task1_files(root: Path) -> List[Path]:
    patterns = [
        "**/task1/**/cd_*_*.out",
        "**/task1/**/cl_*_*.out",
        "**/task1/**/sa_*_*.txt",
        "**/task1/**/kw_*_*.txt",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(set([p for p in files if p.is_file()]))


# -----------------------------
# Main pipeline
# -----------------------------

def run(root: Path,
        outdir: Path,
        tail: int = 200,
        exp_cp_dat: Optional[Path] = None,
        overlay_exp_on_cp: bool = False,
        exp_usecols: Tuple[int, int] = (0, 1),
        save_exp_split: bool = True) -> None:

    files = find_task1_files(root)
    if not files:
        raise RuntimeError(f"No task1 files found under: {root}")

    # Experimental data
    exp_split = None
    if overlay_exp_on_cp and exp_cp_dat is not None and exp_cp_dat.exists():
        exp_df = load_exp_cp_data(exp_cp_dat, usecols=exp_usecols)
        exp_split = split_exp_surfaces_by_cp(exp_df)
        if save_exp_split:
            save_split_exp_surfaces(exp_split, outdir / "experimental_cp_split")

    # case_data[cid] = {"meta": meta, "Cl": df, "Cd": df, "Cp": df}
    case_data: Dict[str, Dict[str, object]] = {}
    records = []

    for f in files:
        meta = parse_case_meta_from_path(f)
        if meta is None:
            continue
        cid = case_id(meta)

        if cid not in case_data:
            case_data[cid] = {"meta": meta, "Cl": None, "Cd": None, "Cp": None}

        # ---- Cd / Cl ----
        if f.suffix.lower() == ".out" and f.name.lower().startswith("cd_"):
            df = parse_fluent_rfile(f)
            case_data[cid]["Cd"] = df

            stats = robust_tail_stats(df["value"], tail=tail)
            plot_convergence(
                df,
                title=f"{cid} Cd",
                ylab="Cd",
                out_png=outdir / "figures" / meta.turb / "task1_single" / f"{cid}_Cd.png"
            )
            records.append({
                "case_id": cid, "turb": meta.turb, "task": meta.task, "group": meta.group,
                "flux": meta.flux, "x": meta.x, "y": meta.y, "z": meta.z,
                "quantity": "Cd", **stats, "file": str(f.relative_to(root))
            })

        elif f.suffix.lower() == ".out" and f.name.lower().startswith("cl_"):
            df = parse_fluent_rfile(f)
            case_data[cid]["Cl"] = df

            stats = robust_tail_stats(df["value"], tail=tail)
            plot_convergence(
                df,
                title=f"{cid} Cl",
                ylab="Cl",
                out_png=outdir / "figures" / meta.turb / "task1_single" / f"{cid}_Cl.png"
            )
            records.append({
                "case_id": cid, "turb": meta.turb, "task": meta.task, "group": meta.group,
                "flux": meta.flux, "x": meta.x, "y": meta.y, "z": meta.z,
                "quantity": "Cl", **stats, "file": str(f.relative_to(root))
            })

        # ---- Cp ----
        elif f.suffix.lower() == ".txt" and (f.name.lower().startswith("sa_") or f.name.lower().startswith("kw_")):
            cp = parse_tecplot_xy_pairs(f)
            case_data[cid]["Cp"] = cp

            # Use a stable single color for single-case Cp (C0)
            plot_cp_segments_single_color(
                cp,
                title=f"{cid} Cp",
                out_png=outdir / "figures" / meta.turb / "task1_single" / f"{cid}_Cp.png",
                color="C0",
                exp_split=exp_split
            )

            records.append({
                "case_id": cid, "turb": meta.turb, "task": meta.task, "group": meta.group,
                "flux": meta.flux, "x": meta.x, "y": meta.y, "z": meta.z,
                "quantity": "Cp_points",
                "last": float(len(cp)),
                "mean_tail": float(cp["y"].mean()),
                "std_tail": float(cp["y"].std(ddof=1) if len(cp) > 1 else 0.0),
                "file": str(f.relative_to(root))
            })

    if not records:
        raise RuntimeError("No parsable files found. Check naming conventions and folder structure.")

    outdir.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(records)
    df_all.to_csv(outdir / "task1_summary_all.csv", index=False)

    df_coef = df_all[df_all["quantity"].isin(["Cl", "Cd"])].copy()
    if len(df_coef) > 0:
        pivot = df_coef.pivot_table(
            index=["turb", "group", "flux", "x", "y", "z"],
            columns="quantity",
            values=["last", "mean_tail", "std_tail"],
            aggfunc="first"
        )
        pivot.to_csv(outdir / "task1_summary_coef_pivot.csv")

    # -----------------------------
    # MERGED plots for Task 1
    # -----------------------------

    # (1) Flux merged: same scheme overlay AUSM vs Roe
    by_scheme = defaultdict(list)  # key=(turb,task,group,x,y,z) -> list of (flux, pack)
    for cid, pack in case_data.items():
        m: CaseMeta = pack["meta"]  # type: ignore
        key = (m.turb, m.task, m.group, m.x, m.y, m.z)
        by_scheme[key].append((m.flux, pack))

    for key, lst in by_scheme.items():
        turb, task, group, x, y, z = key
        if task != 1:
            continue
        fluxes = {fx.lower() for fx, _ in lst}
        if not ("ausm" in fluxes and "roe" in fluxes):
            continue

        def get_pack(flux_name: str):
            for fx, p in lst:
                if fx.lower() == flux_name.lower():
                    return p
            return None

        pA = get_pack("AUSM")
        pR = get_pack("Roe")
        if pA is None or pR is None:
            continue

        scheme = f"x{x}_y{y}" if turb == "sa" else f"x{x}_y{y}_z{z}"
        base = outdir / "figures" / turb / "task1_merged" / "flux_compare"

        plot_overlay_iter_curves(
            [("AUSM", pA.get("Cl")), ("Roe", pR.get("Cl"))],  # type: ignore
            "Cl",
            base / f"{turb}_g{group}_{scheme}_Cl_AUSM_vs_Roe.png",
            f"{turb.upper()} Task1 Group{group} {scheme}: Cl (AUSM vs Roe)"
        )
        plot_overlay_iter_curves(
            [("AUSM", pA.get("Cd")), ("Roe", pR.get("Cd"))],  # type: ignore
            "Cd",
            base / f"{turb}_g{group}_{scheme}_Cd_AUSM_vs_Roe.png",
            f"{turb.upper()} Task1 Group{group} {scheme}: Cd (AUSM vs Roe)"
        )
        plot_overlay_cp(
            [("AUSM", pA.get("Cp")), ("Roe", pR.get("Cp"))],  # type: ignore
            base / f"{turb}_g{group}_{scheme}_Cp_AUSM_vs_Roe.png",
            f"{turb.upper()} Task1 Group{group} {scheme}: Cp (AUSM vs Roe)",
            exp_split=exp_split
        )

    # (2) Discretisation merged: same flux overlay different schemes
    by_flux = defaultdict(list)  # key=(turb,task,group,flux) -> list of packs
    for cid, pack in case_data.items():
        m: CaseMeta = pack["meta"]  # type: ignore
        key = (m.turb, m.task, m.group, m.flux)
        by_flux[key].append(pack)

    def scheme_sort_key(meta: CaseMeta):
        return (meta.x, meta.y) if meta.turb == "sa" else (meta.x, meta.y, meta.z if meta.z is not None else -1)

    for key, packs in by_flux.items():
        turb, task, group, flux = key
        if task != 1:
            continue

        packs = sorted(packs, key=lambda p: scheme_sort_key(p["meta"]))  # type: ignore
        items_cl, items_cd, items_cp = [], [], []

        for p in packs:
            m: CaseMeta = p["meta"]  # type: ignore
            lab = f"x{m.x}_y{m.y}" if m.turb == "sa" else f"x{m.x}_y{m.y}_z{m.z}"
            items_cl.append((lab, p.get("Cl")))  # type: ignore
            items_cd.append((lab, p.get("Cd")))  # type: ignore
            items_cp.append((lab, p.get("Cp")))  # type: ignore

        base = outdir / "figures" / turb / "task1_merged" / "discretisation_compare"
        plot_overlay_iter_curves(
            items_cl, "Cl",
            base / f"{turb}_g{group}_{flux}_Cl_schemes.png",
            f"{turb.upper()} Task1 Group{group} {flux}: Cl (different discretisations)"
        )
        plot_overlay_iter_curves(
            items_cd, "Cd",
            base / f"{turb}_g{group}_{flux}_Cd_schemes.png",
            f"{turb.upper()} Task1 Group{group} {flux}: Cd (different discretisations)"
        )
        plot_overlay_cp(
            items_cp,
            base / f"{turb}_g{group}_{flux}_Cp_schemes.png",
            f"{turb.upper()} Task1 Group{group} {flux}: Cp (different discretisations)",
            exp_split=exp_split
        )

    print(f"[OK] CSV: {outdir / 'task1_summary_all.csv'}")
    print(f"[OK] CSV: {outdir / 'task1_summary_coef_pivot.csv'}")
    print(f"[OK] Single figs: {outdir / 'figures'}/<sa|kw>/task1_single/")
    print(f"[OK] Merged figs: {outdir / 'figures'}/<sa|kw>/task1_merged/")
    if exp_split is not None:
        print(f"[OK] Experimental Cp overlaid (split upper/lower) from: {exp_cp_dat}")
        if save_exp_split:
            print(f"[OK] Saved split exp surfaces to: {outdir / 'experimental_cp_split'}")


def parse_usecols(s: str) -> Tuple[int, int]:
    """Parse --exp_usecols like '0,1' -> (0,1)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("exp_usecols must be two integers separated by comma, e.g. 0,1")
    return (int(parts[0]), int(parts[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root dir containing sa/ and kw/ (e.g., Experiment_Cp_AoA_5_Freestream)")
    ap.add_argument("--out", type=str, default="postprocess_task1",
                    help="Output directory")
    ap.add_argument("--tail", type=int, default=200,
                    help="Tail iterations used for mean/std stats")
    ap.add_argument("--exp_cp", type=str, default="",
                    help="Optional experimental Cp .dat path (e.g., Experiment_Cp_AoA_5_Freestream.dat)")
    ap.add_argument("--overlay_exp", action="store_true",
                    help="Overlay experimental Cp onto Cp plots (single + merged)")
    ap.add_argument("--exp_usecols", type=str, default="0,1",
                    help="Which two columns to read from exp .dat, e.g. '0,1' or '1,2'")
    ap.add_argument("--no_save_exp_split", action="store_true",
                    help="Do not save split experimental surfaces CSV")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()

    exp_path = Path(args.exp_cp).resolve() if args.exp_cp else None
    usecols = parse_usecols(args.exp_usecols)

    run(
        root=root,
        outdir=outdir,
        tail=args.tail,
        exp_cp_dat=exp_path,
        overlay_exp_on_cp=args.overlay_exp,
        exp_usecols=usecols,
        save_exp_split=(not args.no_save_exp_split),
    )


if __name__ == "__main__":
    main()



