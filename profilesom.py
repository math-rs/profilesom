# SPDX-License-Identifier: MIT
# Copyright (c) 2025
# Matheus Santos (ORCID: 0000-0002-1604-381X)

"""
Hexagonal SOM + KMeans (auto-K via Davies–Bouldin on a sample-weighted codebook)
================================================================================

Overview
--------
This script trains a hexagonal Self-Organizing Map (SOM) on z-scored features,
clusters the SOM prototypes (codebook) with KMeans, and selects the number of
clusters k that minimizes the Davies–Bouldin (DB) index computed on a
sample-weighted codebook. Weighting is achieved by replicating each
prototype according to its BMU hit count, thereby preserving data density
while avoiding per-sample BMU noise during K selection.

Quick start
-----------
python som_hex_kmeans.py --xlsx data.xlsx --sheet Data \
    --som_m 6 --som_n 6 --som_iters 1000 --k_min 2 --k_max 10 \
    --id_col "Sample" --depth_col "h" --profile_col "Profile"

Input
-----
- Excel spreadsheet with header on row 2 ('header=1'): first row ignored.
- All columns are treated as variables, except:
  * a column named "h" (case-insensitive) — used as depth/height (optional plots),
  * an optional profile/borehole ID column (for vertical strips),
  * the chosen sample ID column.

Outputs (under --out_dir)
-------------------------
- log/          : run log and environment snapshot
- data/         : assignments, codebook vectors, summaries, and fitted artifacts
- diagnostics/  : K-scan metrics and PCA biplots
- plots/        : figures in subfolders: umatrix/, hits/, clusters/, components/, profile/, boxes/

K selection policy (DB-only)
----------------------------
- For each k, KMeans is trained on the sample-weighted codebook
  (prototypes replicated by BMU hit counts).
- scikit-learn selects the best initialization by inertia (SSE).
- We compute **Davies–Bouldin** on the weighted codebook and choose the k that minimizes it.
- Final cluster labels for plots are obtained by predicting labels on the
  unweighted codebook using the model at the selected k.

Notes
-----
- BMUs are used only for hit counts and to assign samples to clusters via the BMU
  (for reporting), not to select k.
- Requires Python 3.9+.
"""

from __future__ import annotations

# ---- BLAS threads (avoid oversubscription on MKL/OMP boxes) -----------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- Standard libs ----------------------------------------------------------
import argparse
import json
import logging
import math
import re
import sys
import platform
import pickle
from collections import deque
from pathlib import Path
from typing import Iterable

# ---- Third-party ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms as mtrans
from matplotlib.transforms import blended_transform_factory
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: Path, *, level: int = logging.INFO) -> None:
    """
    Configure logging to both console and file: outputs/log/run.log.
    Creates the directory if needed.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "run.log"

    # Root logger reset (idempotent if re-running in notebooks)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logging.basicConfig(level=level, handlers=[fh, ch])
    logging.info("Logging initialized at %s", logfile)


# =============================================================================
# I/O & Column Handling
# =============================================================================

def read_excel_header_row2(xlsx_path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Read an Excel sheet whose header row is the 2nd row (pandas header=1)."""
    if sheet is None:
        sheet = pd.ExcelFile(xlsx_path).sheet_names[0]
    return pd.read_excel(xlsx_path, sheet_name=sheet, header=1)


def _make_unique_labels(s: pd.Series) -> pd.Series:
    """Ensure labels are unique by appending #2, #3... when duplicates occur."""
    seen: dict[str, int] = {}
    out: list[str | None] = []
    for v in s.astype(str):
        v = v.strip()
        if v == "" or v.lower() in {"na", "nan", "none", "-"}:
            out.append(None)
            continue
        k = seen.get(v, 0)
        out.append(v if k == 0 else f"{v}#{k+1}")
        seen[v] = k + 1
    return pd.Series(out, index=s.index)


def choose_sample_id(df: pd.DataFrame, prefer: str | None = None) -> pd.Series:
    """
    Pick a human-readable ID column, or build S1..SN.
    Precedence:
      (1) explicit --id_col, (2) common names, (3) most-distinct object column, (4) S1..SN.
    Returns a **string** Series, unique and non-empty.
    """
    def _finish(s: pd.Series) -> pd.Series:
        s = _make_unique_labels(s)
        s = s.where(s.notna(), None)
        if s.isna().any():
            s = s.fillna(pd.Series([f"S{i+1}" for i in range(len(s))], index=s.index))
        return s.astype(str)

    if prefer:
        m = {c.lower(): c for c in df.columns}
        src = prefer if prefer in df.columns else m.get(prefer.lower())
        if src is not None:
            return _finish(df[src].astype(str))

    candidates = {
        "sample", "sample id", "sampleid", "id", "name", "code", "label", "tag",
        "amostra", "codigo", "código", "cod amostra", "codigo amostra",
        "sample_code", "sample code", "sample_name", "sample name",
    }
    by_lower = {c.lower().strip(): c for c in df.columns}
    for key in candidates:
        if key in by_lower:
            return _finish(df[by_lower[key]].astype(str))

    objish: list[tuple[int, str]] = []
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or df[c].dtype == "string":
            objish.append((df[c].nunique(dropna=True), c))
    if objish:
        objish.sort(reverse=True)
        return _finish(df[objish[0][1]].astype(str))

    return pd.Series([f"S{i+1}" for i in range(len(df))], name="SampleID", dtype=str)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert columns to numeric with censoring rules:
      - '< number' (e.g., '<0.12' or '< 0,12') -> number/2
      - '<LD' and NaN -> half of the finite minimum observed numeric in that column
        (if no numeric value exists, remains NaN and is imputed later).
    """
    lt_num_re = re.compile(r"^\s*<\s*([0-9]+(?:[.,][0-9]+)?(?:[eE][+-]?\d+)?)\s*$", re.IGNORECASE)
    ld_re = re.compile(r"^\s*<\s*LD\s*$", re.IGNORECASE)

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue

        s_raw = out[c]
        vals = pd.to_numeric(s_raw, errors="coerce")

        s_norm = s_raw.astype(str).str.strip().str.replace(",", ".", regex=False)

        # Case A: '< number' -> number/2
        extract = s_norm.str.extract(lt_num_re)[0]
        mask_lt_num = extract.notna()
        if mask_lt_num.any():
            vals.loc[mask_lt_num] = extract[mask_lt_num].astype(float).values / 2.0

        finite_min_series = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna()
        finite_min = finite_min_series.min() if not finite_min_series.empty else np.nan

        # Case B/C: '<LD' or NaN -> half of finite_min (if defined)
        mask_ld = s_norm.str.contains(ld_re)
        mask_nan = vals.isna()
        if (mask_ld.any() or mask_nan.any()) and pd.notna(finite_min):
            vals.loc[mask_ld | mask_nan] = float(finite_min) / 2.0

        out[c] = vals

    return out


def force_finite(X: np.ndarray) -> np.ndarray:
    """Replace any NaN/±inf by the column median (or 0 if undefined)."""
    X = np.asarray(X, dtype=float)
    col_med = np.nanmedian(X, axis=0)
    col_med[~np.isfinite(col_med)] = 0.0
    mask = ~np.isfinite(X)
    if mask.any():
        cols = np.where(mask)[1]
        X[mask] = col_med[cols]
    X[~np.isfinite(X)] = 0.0
    return X


# =============================================================================
# Hex Geometry & Helpers
# =============================================================================

def _hex_centers(m: int, n: int, R: float) -> tuple[np.ndarray, np.ndarray]:
    """Centers for a pointy-top hex grid with **odd-r** layout (m rows, n cols)."""
    rows = np.arange(m)[:, None]
    cols = np.arange(n)[None, :]
    Xc = (np.sqrt(3) * R) * (cols + 0.5 * (rows % 2))
    Yc = (1.5 * R) * rows * np.ones((1, n))
    return Xc, Yc


def _hex_vertices(xc: float, yc: float, R: float) -> np.ndarray:
    """Vertices of a pointy-top hex centered at (xc, yc) with radius R."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    xs = xc + R * np.cos(angles)
    ys = yc + R * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def _best_text_color(rgba) -> str:
    """Choose black/white text to contrast with a facecolor (sRGB luminance)."""
    r, g, b = rgba[:3]
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if L > 0.6 else "white"


def isotope_superscript(name: str,
                        _sup=str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")) -> str:
    """Render isotope mass numbers as superscripts (e.g., '40Ar' → '⁴⁰Ar')."""
    s = str(name)
    m = re.match(r"^\s*(\d+)\s*([A-Za-z].*)$", s)
    return (m.group(1).translate(_sup) + m.group(2)) if m else s


# =============================================================================
# Core Plotting: Packed Hex Grid + Boundaries/Labels
# =============================================================================

def _draw_hex_map_packed(
    data: np.ndarray,
    title: str,
    *,
    discrete: bool = False,
    cbar_label: str = "",
    R: float = 1.0,
    face_overfill: float = 1.0,
    cmap_name: str = "viridis",
    counts: np.ndarray | None = None,
    text_size: int = 8,
    show_zeros: bool = False,
    cbar_integer_ticks: bool = False,
    cbar_tick_offset: int = 0,
    cmap_override: ListedColormap | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, float, list[list[np.ndarray]]]:
    """
    Draw a packed hex map and return drawable context (fig, ax, Xc, Yc, R_face, verts_grid).
    `verts_grid[i][j]` stores the exact vertices used to paint cell (i, j).
    """
    m, n = data.shape
    if discrete:
        k = int(np.nanmax(data)) + 1
        cmap = (cmap_override if cmap_override is not None
                else plt.get_cmap(cmap_name, max(k, 2)))
        norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
        tick_vals = np.arange(0, k)
    else:
        cmap = (cmap_override if cmap_override is not None else plt.get_cmap(cmap_name))
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        tick_vals = None
        if cbar_integer_ticks:
            lo = int(np.floor(np.nanmin(data)))
            hi = int(np.ceil(np.nanmax(data)))
            tick_vals = np.arange(lo, hi + 1)

    Xc, Yc = _hex_centers(m, n, R)
    R_face = R * face_overfill

    verts, colors = [], []
    verts_grid = [[None] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            V = _hex_vertices(Xc[i, j], Yc[i, j], R_face)
            verts.append(V)
            verts_grid[i][j] = V
            colors.append(cmap(norm(float(data[i, j]))))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    pc = PolyCollection(verts, facecolors=colors, edgecolors="none",
                        linewidths=0.0, antialiased=False, closed=True)
    ax.add_collection(pc)

    # Optional per-cell text (e.g., hits)
    if counts is not None:
        for i in range(m):
            for j in range(n):
                c = int(counts[i, j])
                if c == 0 and not show_zeros:
                    continue
                rgba = cmap(norm(float(data[i, j])))
                ax.text(Xc[i, j], Yc[i, j], str(c),
                        ha="center", va="center",
                        color=_best_text_color(rgba), fontsize=text_size, fontweight="bold")

    ax.set_aspect("equal")
    ax.set_title(title)
    pad = R * 1.25
    ax.set_xlim(Xc.min() - pad, Xc.max() + pad)
    ax.set_ylim(Yc.min() - pad, Yc.max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    if cbar_label:
        cbar.set_label(cbar_label)
    if tick_vals is not None:
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([str(t + cbar_tick_offset) for t in tick_vals])

    return fig, ax, Xc, Yc, R_face, verts_grid


def _add_cluster_boundaries(
    ax: plt.Axes,
    Xc: np.ndarray,
    Yc: np.ndarray,
    R_face: float,
    cluster_grid: np.ndarray,
    *,
    color: str = "white",
    lw: float = 3.0,
    verts_grid: list[list[np.ndarray]] | None = None,
) -> None:
    """
    Draw cluster boundaries by locating shared edges between cells of different labels.
    If `verts_grid` is provided, it uses the *same* vertices as the faces to guarantee
    pixel-perfect alignment; otherwise recomputes with `_hex_vertices`.
    """
    m, n = cluster_grid.shape
    vcache = verts_grid
    if vcache is None:
        vcache = [[None] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                vcache[i][j] = _hex_vertices(Xc[i, j], Yc[i, j], R_face)

    def edge_midpoints(V: np.ndarray) -> np.ndarray:
        mids = []
        for a in range(6):
            b = (a + 1) % 6
            mids.append(0.5 * (V[a] + V[b]))
        return np.asarray(mids)

    segs: list[list[np.ndarray]] = []

    for i in range(m):
        row_even = (i % 2 == 0)
        for j in range(n):
            lab = int(cluster_grid[i, j])
            V = vcache[i][j]
            M = edge_midpoints(V)

            # Candidate neighbors to avoid double-drawing (E, SE, SW in odd-r)
            nbrs: list[tuple[int, int]] = []
            # E
            ii, jj = i, j + 1
            if jj < n:
                nbrs.append((ii, jj))
            # SE
            ii, jj = i + 1, j + (0 if row_even else 1)
            if ii < m and 0 <= jj < n:
                nbrs.append((ii, jj))
            # SW
            ii, jj = i + 1, j - (1 if row_even else 0)
            if ii < m and 0 <= jj < n:
                nbrs.append((ii, jj))

            for ii, jj in nbrs:
                if int(cluster_grid[ii, jj]) == lab:
                    continue
                Vn = vcache[ii][jj]
                Mn = edge_midpoints(Vn)

                D = np.sum((M[:, None, :] - Mn[None, :, :]) ** 2, axis=2)
                a_idx, b_idx = np.unravel_index(np.argmin(D), D.shape)
                # Draw this cell’s edge from vertex a_idx to a_idx+1
                a2 = (a_idx + 1) % 6
                segs.append([V[a_idx], V[a2]])

    if segs:
        ax.add_collection(LineCollection(
            segs,
            colors=color,
            linewidths=lw,
            antialiased=False,
            capstyle="round",
            joinstyle="round",
            zorder=6,
        ))


def _label_cluster_components(ax: plt.Axes,
                              Xc: np.ndarray,
                              Yc: np.ndarray,
                              cluster_grid: np.ndarray,
                              text_size: int = 12) -> None:
    """
    Label each connected component of a cluster.

    Strategy:
      • Among interior cells (graph distance > 0 to the boundary), choose the one whose
        center is closest to the component centroid.
      • If there are no interior cells, pick the cell farthest from the boundary.
      • As a last resort, use the cell closest to the centroid.

    This keeps labels centered and inside the component.
    """
    m, n = cluster_grid.shape
    visited = np.zeros((m, n), dtype=bool)

    def neighbors(i: int, j: int):
        even = (i % 2 == 0)
        offs = [(+1, 0), (-1, 0), (0, +1), (0, -1),
                (+1, -1 if even else +1), (-1, -1 if even else +1)]
        for di, dj in offs:
            ii, jj = i + di, j + dj
            if 0 <= ii < m and 0 <= jj < n:
                yield ii, jj

    for i in range(m):
        for j in range(n):
            if visited[i, j]:
                continue
            lab = cluster_grid[i, j]
            # DFS: componente conectado
            comp, stack = [], [(i, j)]
            visited[i, j] = True
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                for ii, jj in neighbors(ci, cj):
                    if (not visited[ii, jj]) and cluster_grid[ii, jj] == lab:
                        visited[ii, jj] = True
                        stack.append((ii, jj))

            comp_set = set(comp)
            # Células de borda: tocam algum vizinho fora do componente
            boundary = []
            for ci, cj in comp:
                for ii, jj in neighbors(ci, cj):
                    if (ii, jj) not in comp_set:
                        boundary.append((ci, cj))
                        break

            # BFS multi-origem para distância à borda
            dist = {rc: np.inf for rc in comp}
            dq = deque()
            for rc in boundary:
                dist[rc] = 0
                dq.append(rc)
            while dq:
                u = dq.popleft()
                for v in neighbors(*u):
                    if v in comp_set and dist[v] is np.inf:
                        dist[v] = dist[u] + 1
                        dq.append(v)

            # Centrós dos hexes do componente
            xs = np.array([Xc[ci, cj] for (ci, cj) in comp], dtype=float)
            ys = np.array([Yc[ci, cj] for (ci, cj) in comp], dtype=float)
            cx0, cy0 = float(xs.mean()), float(ys.mean())

            # Preferir células internas (dist > 0); escolher a mais próxima do centróide
            internal = [(ci, cj) for (ci, cj) in comp if np.isfinite(dist[(ci, cj)]) and dist[(ci, cj)] > 0]
            if internal:
                ci, cj = min(
                    internal,
                    key=lambda rc: ( (Xc[rc[0], rc[1]] - cx0)**2 + (Yc[rc[0], rc[1]] - cy0)**2 )
                )
            else:
                # Fallback: a mais distante da borda
                finite_any = [rc for rc in comp if np.isfinite(dist[rc])]
                if finite_any:
                    ci, cj = max(finite_any, key=lambda rc: dist[rc])
                else:
                    # Fallback extremo: centroide discreto
                    # (pouquíssimo provável; mantido por robustez)
                    d2 = (xs - cx0)**2 + (ys - cy0)**2
                    ci, cj = comp[int(np.argmin(d2))]

            cx, cy = float(Xc[ci, cj]), float(Yc[ci, cj])
            ax.text(cx, cy, str(int(lab) + 1),
                    ha="center", va="center",
                    fontsize=text_size, fontweight="bold", color="black",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    zorder=7)


# =============================================================================
# U-matrix (dense hex) and High-level Plots
# =============================================================================

def compute_dense_umatrix(som: MiniSom, center_mode: str = "builtin") -> np.ndarray:
    """
    Build a dense U-matrix (2m−1 × 2n−1) for hex (**odd-r**) layout.

    center_mode:
      - "builtin": MiniSom.distance_map() (mean over neighbors)
      - "mean":    mean of the 6 surrounding edge midpoints
      - "median":  median of the 6 surrounding edge midpoints (SOM-Toolbox default)
    """
    W = som.get_weights()  # (m, n, p)
    m, n, _ = W.shape
    U = np.full((2 * m - 1, 2 * n - 1), np.nan, dtype=float)

    def dist(a, b) -> float:
        d = a - b
        return float(np.sqrt(np.dot(d, d)))

    # edge midpoints
    for i in range(m):
        even = (i % 2 == 0)
        for j in range(n):
            if i + 1 < m:
                U[2 * i + 1, 2 * j] = dist(W[i, j], W[i + 1, j])        # E
            if j + 1 < n:
                U[2 * i, 2 * j + 1] = dist(W[i, j], W[i, j + 1])        # S
            if even:
                if i + 1 < m and j - 1 >= 0:
                    U[2 * i + 1, 2 * j - 1] = dist(W[i, j], W[i + 1, j - 1])  # SE
                if i - 1 >= 0 and j - 1 >= 0:
                    U[2 * i - 1, 2 * j - 1] = dist(W[i, j], W[i - 1, j - 1])  # SW
            else:
                if i + 1 < m and j + 1 < n:
                    U[2 * i + 1, 2 * j + 1] = dist(W[i, j], W[i + 1, j + 1])  # NE
                if i - 1 >= 0 and j + 1 < n:
                    U[2 * i - 1, 2 * j + 1] = dist(W[i, j], W[i - 1, j + 1])  # NW

    # centers
    if center_mode == "builtin":
        U[0::2, 0::2] = som.distance_map()
    else:
        for i in range(m):
            for j in range(n):
                vals = []
                if j - 1 >= 0:
                    vals.append(U[2 * i, 2 * j - 1])
                if j + 1 < n:
                    vals.append(U[2 * i, 2 * j + 1])
                if i - 1 >= 0:
                    if (i % 2 == 0):
                        if j - 1 >= 0:
                            vals.append(U[2 * i - 1, 2 * j - 1])
                        vals.append(U[2 * i - 1, 2 * j])
                    else:
                        vals.append(U[2 * i - 1, 2 * j])
                        if j + 1 < n:
                            vals.append(U[2 * i - 1, 2 * j + 1])
                if i + 1 < m:
                    if (i % 2 == 0):
                        if j - 1 >= 0:
                            vals.append(U[2 * i + 1, 2 * j - 1])
                        vals.append(U[2 * i + 1, 2 * j])
                    else:
                        vals.append(U[2 * i + 1, 2 * j])
                        if j + 1 < n:
                            vals.append(U[2 * i + 1, 2 * j + 1])
                if vals:
                    U[2 * i, 2 * j] = (np.mean(vals) if center_mode == "mean"
                                       else np.median(vals))
    return U


def plot_umatrix_hex_dense(
    som: MiniSom,
    out_png: Path,
    *,
    R: float = 1.0,
    face_overfill: float = 1.0,
    cmap_name: str = "viridis",
    title: str = "SOM U-matrix (dense hex)",
    center_mode: str = "builtin",   # 'builtin' | 'mean' | 'median'
) -> None:
    """Draw a dense U-matrix on a hex grid (centers + edge cells) with uniform tile size."""
    U = compute_dense_umatrix(som, center_mode=center_mode)   # (2m-1, 2n-1)
    m2, n2 = U.shape

    Xc, Yc = _hex_centers(m2, n2, R)
    R_draw = R * face_overfill

    vmin, vmax = np.nanmin(U), np.nanmax(U)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    verts, colors = [], []
    for i in range(m2):
        for j in range(n2):
            # Uniform radius for all tiles (no visual scaling)
            verts.append(_hex_vertices(Xc[i, j], Yc[i, j], R_draw))
            colors.append((0.98, 0.98, 0.99, 1.0) if np.isnan(U[i, j])
                          else cmap(norm(float(U[i, j]))))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    pc = PolyCollection(verts, facecolors=colors, edgecolors="none",
                        linewidths=0.0, antialiased=False, closed=True)
    ax.add_collection(pc)

    ax.set_aspect("equal")
    pad = R * 1.25
    ax.set_xlim(Xc.min() - pad, Xc.max() + pad)
    ax.set_ylim(Yc.min() - pad, Yc.max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Distance")

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _discrete_cmap_for_clusters(k: int) -> tuple[ListedColormap, BoundaryNorm]:
    """Return a discrete colormap & norm for integer cluster labels 0..k-1."""
    cmap = _distinct_cmap(k)
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
    return cmap, norm


def _distinct_cmap(k: int) -> ListedColormap:
    """
    Qualitative palette: inter-color distance is high.
    Strategy: take even indices from tab20 then odd; if k>20, extend with HSV.
    """
    base = plt.get_cmap("tab20").colors  # 20 RGBA tuples
    order = [0,2,4,6,8,10,12,14,16,18, 1,3,5,7,9,11,13,15,17,19]
    cols = [base[i] for i in order][:min(k, 20)]
    if k > 20:
        for t in np.linspace(0, 1, k - 20, endpoint=False):
            cols.append(tuple(plt.cm.hsv(t)))
    return ListedColormap(cols, name=f"distinct_{k}")


# ---- High-level wrappers -----------------------------------------------------

def plot_umatrix_hex(som: MiniSom, Xs: np.ndarray, out_png: Path, title: str = "SOM U-Matrix (hex)") -> None:
    um = som.distance_map()
    hits = _compute_hits_grid(som, Xs)
    fig, ax, Xc, Yc, R_face, _ = _draw_hex_map_packed(
        um, title, discrete=False, cbar_label="Distance",
        R=1.0, face_overfill=1.0, counts=hits, text_size=8, show_zeros=False
    )
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_hits_hex(som: MiniSom, Xs: np.ndarray, out_png: Path, title: str = "SOM Hits (hex)") -> None:
    hits = _compute_hits_grid(som, Xs)

    k = int(np.nanmax(hits)) + 1
    k = max(k, 2)
    vals = np.linspace(0.10, 0.90, k)
    cmap_bw = ListedColormap([(v, v, v, 1.0) for v in vals])

    fig, ax, Xc, Yc, R_face, _ = _draw_hex_map_packed(
        hits, title,
        discrete=True,
        cbar_label="# samples",
        R=1.0,
        face_overfill=1.0,
        counts=hits,
        text_size=8,
        show_zeros=True,
        cbar_tick_offset=0,
        cmap_override=cmap_bw,
    )
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)

def plot_clusters_hex(
    cluster_grid: np.ndarray,
    som: MiniSom,
    Xs: np.ndarray,
    out_png: Path,
    *,
    title: str = "KMeans clusters on SOM (hex)",
    boundary_color: str = "white",
    boundary_lw: float = 3.0,
) -> None:
    """
    Discrete cluster map with integer legend and boundaries.
    No component labels here (labels are reserved for the final
    “Clusters” panel in the component-planes figure).
    Boundaries use the same vertices used to paint the faces.
    """
    hits = _compute_hits_grid(som, Xs)
    k = int(np.nanmax(cluster_grid)) + 1
    cmap_disc, _ = _discrete_cmap_for_clusters(k)

    fig, ax, Xc, Yc, R_face, verts_grid = _draw_hex_map_packed(
        cluster_grid, title, discrete=True, cbar_label="Cluster ID",
        R=1.0, face_overfill=1.0, counts=hits, text_size=8, show_zeros=False,
        cbar_tick_offset=1, cmap_override=cmap_disc
    )

    # Boundaries aligned with painted faces
    _add_cluster_boundaries(
        ax, Xc, Yc, R_face, cluster_grid,
        color=boundary_color, lw=boundary_lw, verts_grid=verts_grid
    )

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_hits_hex_with_ids(
    som: MiniSom,
    Xs: np.ndarray,
    sample_ids: Iterable[str],
    out_png: Path,
    *,
    title: str = "SOM Hits (sample IDs per BMU)",
    R: float = 1.0,
    face_overfill: float = 1.0,
    max_ids_per_cell: int = 8,
    max_chars_per_id: int = 12,
    tick_vals: Iterable[int] | None = None,
) -> None:
    """Plot hit counts with truncated per-cell sample IDs (black & white)."""
    m, n = som.get_weights().shape[:2]

    cell_ids = [[[] for _ in range(n)] for _ in range(m)]
    hits = np.zeros((m, n), dtype=int)
    for sid, x in zip(sample_ids, Xs):
        i, j = som.winner(x)
        cell_ids[i][j].append(f"{int(sid):02d}" if str(sid).isdigit() else str(sid))
        hits[i, j] += 1

    k = int(np.nanmax(hits)) + 1
    k = max(k, 2)
    vals = np.linspace(0.10, 0.90, k)
    cmap = ListedColormap([(v, v, v, 1.0) for v in vals])
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
    if tick_vals is None:
        tick_vals = np.arange(0, k)

    Xc, Yc = _hex_centers(m, n, R)
    R_face = R * face_overfill

    verts, colors = [], []
    for i in range(m):
        for j in range(n):
            verts.append(_hex_vertices(Xc[i, j], Yc[i, j], R_face))
            colors.append(cmap(norm(float(hits[i, j]))))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    pc = PolyCollection(verts, facecolors=colors, edgecolors="none",
                        antialiased=False, closed=True)
    ax.add_collection(pc)

    def _shorten(s: str, L: int) -> str:
        return s if len(s) <= L else (s[:L-1] + "…")

    for i in range(m):
        for j in range(n):
            if not cell_ids[i][j]:
                continue
            ids_disp = [_shorten(s, max_chars_per_id) for s in cell_ids[i][j][:max_ids_per_cell]]
            extra = len(cell_ids[i][j]) - len(ids_disp)
            if extra > 0:
                ids_disp.append(f"+{extra} more")
            rgba = cmap(norm(float(hits[i, j])))
            fs = 9 if len(ids_disp) <= 3 else (8 if len(ids_disp) <= 6 else 7)
            ax.text(Xc[i, j], Yc[i, j], "\n".join(ids_disp),
                    ha="center", va="center",
                    color=_best_text_color(rgba), fontsize=fs,
                    linespacing=0.9)

    ax.set_aspect("equal")
    ax.set_title(title)
    pad = R * 1.25
    ax.set_xlim(Xc.min() - pad, Xc.max() + pad)
    ax.set_ylim(Yc.min() - pad, Yc.max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("# samples")
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(t) for t in tick_vals])

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _compute_hits_grid(som: MiniSom, Xs: np.ndarray) -> np.ndarray:
    """Count how many samples map to each neuron (BMU hits)."""
    m, n = som.get_weights().shape[:2]
    hits = np.zeros((m, n), dtype=int)
    for x in Xs:
        i, j = som.winner(x)
        hits[i, j] += 1
    return hits


# =============================================================================
# “Extra” Plots (Boxes, Profile, K diagnostics, Component planes)
# =============================================================================

def _discrete_palette_from_labels(labels: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    uniq = sorted(set(int(x) for x in labels))
    k = int(max(uniq)) + 1
    cmap_disc, norm_disc = _discrete_cmap_for_clusters(k)
    return {c: cmap_disc(norm_disc(c)) for c in uniq}


def plot_box_by_cluster(
    data_df: pd.DataFrame,
    feature_cols: list[str],
    clusters: pd.Series | np.ndarray,
    out_dir: Path,
    *,
    box_alpha: float = 1.0,
    box_linewidth: float = 1.2,
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    median_linewidth: float = 1.6,
    show_points: bool = False,
    point_size: float = 16.0,
    point_alpha: float = 0.55,
    jitter: float = 0.18,
    edgecolor: str = "black",
    random_seed: int = 42,
) -> None:
    """One PNG per feature: cluster-wise boxplots (optionally overlay points)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.Series(clusters).astype(int).reset_index(drop=True)
    uniq = sorted(clusters.dropna().unique().tolist())
    if not uniq:
        return

    colors = _discrete_palette_from_labels(uniq)
    rng = np.random.default_rng(random_seed)
    xcenters = {c: i for i, c in enumerate(uniq)}  # category -> x location

    def _one_axis(ax, feat: str):
        y = data_df[feat].reset_index(drop=True)
        msk = pd.notna(y) & pd.notna(clusters)
        if not msk.any():
            ax.axis("off"); return

        vals = y[msk].to_numpy(float)
        clus = clusters[msk].to_numpy(int)
        data_by_c = [vals[clus == c] for c in uniq]

        bp = ax.boxplot(
            data_by_c,
            positions=[xcenters[c] for c in uniq],
            widths=0.6,
            notch=False,
            vert=True,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )

        for i, c in enumerate(uniq):
            col = colors[c]
            bp["boxes"][i].set(
                facecolor=col, alpha=box_alpha, edgecolor=edgecolor, linewidth=box_linewidth
            )
            bp["medians"][i].set(color="black", linewidth=median_linewidth)
            bp["whiskers"][2 * i].set(color=edgecolor, linewidth=whisker_linewidth)
            bp["whiskers"][2 * i + 1].set(color=edgecolor, linewidth=whisker_linewidth)
            bp["caps"][2 * i].set(color=edgecolor, linewidth=cap_linewidth)
            bp["caps"][2 * i + 1].set(color=edgecolor, linewidth=cap_linewidth)

        if show_points:
            xs = np.empty_like(vals, dtype=float)
            for c in uniq:
                mask = (clus == c)
                n = int(mask.sum())
                if n == 0:
                    continue
                base = float(xcenters[c])
                w = float(jitter) * 0.5
                xs[mask] = base + rng.uniform(-w, +w, size=n)

            for c in uniq:
                mask = (clus == c)
                if not np.any(mask):
                    continue
                ax.scatter(
                    xs[mask], vals[mask],
                    s=point_size, alpha=point_alpha,
                    c=[colors[c]], edgecolors=edgecolor, linewidths=0.5,
                    zorder=2
                )

        ax.set_title(str(isotope_superscript(feat)), fontsize=10, pad=2)
        ax.set_xlabel("Cluster"); ax.set_ylabel("ppm")
        ax.set_xticks([xcenters[c] for c in uniq])
        ax.set_xticklabels([str(c + 1) for c in uniq])
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        if ymin == ymax:
            dy = 1.0 if ymax == 0 else abs(ymax) * 0.1
            ax.set_ylim(ymin - dy, ymax + dy)
        else:
            pad = 0.06 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for s in ax.spines.values():
            s.set_visible(False)

    for feat in feature_cols:
        fig_w = max(3.2, 1.2 * len(uniq) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 3.4))
        _one_axis(ax, feat)
        safe = re.sub(r"[^\w\-\.]+", "_", str(feat)).strip("_")
        fig.savefig(out_dir / f"box_{safe}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"box_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_box_rows_by_scale(
    data_df: pd.DataFrame,
    feature_cols: list[str],
    clusters: pd.Series | np.ndarray,
    out_png: Path,
    *,
    n_rows: int = 5,
    box_alpha: float = 1.0,
    box_linewidth: float = 1.2,
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    median_linewidth: float = 1.6,
    show_fliers: bool = False,
    figsize_each_row: float = 3.0,
    show_row_titles: bool = False,
    show_xlabel: bool = False
) -> None:
    """
    Panel with N rows; variables ordered by span and split into balanced rows.
    Each variable shows side-by-side boxplots across clusters.
    """
    clusters = pd.Series(clusters).astype(int).reset_index(drop=True)
    valid_feats = [c for c in feature_cols if c in data_df.columns]
    if not valid_feats or clusters.dropna().empty:
        return

    # span per variable
    spans = {}
    for f in valid_feats:
        s = pd.to_numeric(data_df[f], errors="coerce")
        if s.notna().any():
            spans[f] = float(s.max() - s.min())
    if not spans:
        return

    # sort by span and divide into n_rows blocks with (roughly) equal counts
    feats_sorted = [f for f, _ in sorted(spans.items(), key=lambda kv: kv[1])]
    q, r = divmod(len(feats_sorted), n_rows)
    sizes = [q + 1] * r + [q] * (n_rows - r)
    groups, p = [], 0
    for s in sizes:
        groups.append(feats_sorted[p:p + s]); p += s

    # ranges (for optional per-row titles)
    ranges = []
    for g in groups:
        if g:
            vals = [spans[f] for f in g]
            ranges.append((min(vals), max(vals)))
        else:
            ranges.append((np.nan, np.nan))

    uniq_clusters = sorted(clusters.dropna().unique().tolist())
    colors = _discrete_palette_from_labels(uniq_clusters)

    nrows = len(groups)
    fig_h = max(3.0, figsize_each_row * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, fig_h), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for r, (ax, feats) in enumerate(zip(axes, groups)):
        if not feats:
            ax.axis("off"); continue

        n_vars = len(feats)
        gwidth = max(1.1, 0.6 + 0.4 * len(uniq_clusters))   # width of one group
        gap = 0.6                                           # space between groups
        base_positions = np.arange(n_vars) * (gwidth + gap)

        for ci_idx, ci in enumerate(uniq_clusters):
            pos = base_positions + (ci_idx - (len(uniq_clusters)-1)/2.0) * (gwidth/len(uniq_clusters))
            data = []
            for f in feats:
                y = pd.to_numeric(data_df[f], errors="coerce")
                m = y.notna() & clusters.notna() & (clusters == ci)
                data.append(y[m].to_numpy(float))

            bp = ax.boxplot(
                data, positions=pos,
                widths=(gwidth/len(uniq_clusters))*0.8,
                vert=True, patch_artist=True,
                showfliers=show_fliers, whis=1.5
            )
            col = colors[ci]
            for b in bp["boxes"]:
                b.set(facecolor=col, alpha=box_alpha, edgecolor="black", linewidth=box_linewidth)
            for w in bp["whiskers"]:
                w.set(color="black", linewidth=whisker_linewidth)
            for c in bp["caps"]:
                c.set(color="black", linewidth=cap_linewidth)
            for mline in bp["medians"]:
                mline.set(color="black", linewidth=median_linewidth)

        ax.set_xticks(base_positions)
        ax.set_xticklabels([isotope_superscript(f) for f in feats], fontsize=16, ha="center")

        if show_row_titles:
            lo, hi = ranges[r]
            lo_txt = f"{0 if np.isnan(lo) else lo:.1f}"
            hi_txt = "∞" if np.isinf(hi) else f"{hi:.1f}"
            ax.set_title(f"Variable span ~ {lo_txt}–{hi_txt}")

        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for s in ax.spines.values():
            s.set_visible(False)

        if show_xlabel:
            ax.set_xlabel("variable")
        ax.set_ylabel("ppm")

    handles = []
    for ci in uniq_clusters:
        patch = plt.Line2D([0], [0], marker='s', linestyle='',
                           markersize=10, markerfacecolor=colors[ci],
                           markeredgecolor='black', label=str(ci + 1), alpha=box_alpha)
        handles.append(patch)
    axes[0].legend(handles=handles, title="Cluster", frameon=True, loc="upper left")

    fig.suptitle("Per-variable distributions (grouped by span, balanced rows)", y=1.02, fontsize=16)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_codebook_pca_biplot(codebook: np.ndarray,
                             feature_names: list[str],
                             labels: np.ndarray,
                             out_pdf: Path,
                             *,
                             title: str = "Codebook PCA — Biplot",
                             point_size: float = 58.0,
                             arrow_lw: float = 0.9,
                             head_frac: float = 0.012,        # small arrowhead
                             label_offset_frac: float = 0.018, # tiny nudge beyond tip
                             font_size: float = 8.5,
                             label_stroke: float = 1.6
                             ) -> None:
    """
    PCA (2D) of the ORIGINAL codebook for visualization only.
    • Points = codebook prototypes (PC1/PC2), colored by cluster
    • Arrows = feature loadings
    • Labels = placed at the **end of each arrow** (horizontal), with a tiny outward offset
    • Output: **PDF** (for editing in Illustrator)
    """

    # --- PCA on standardized codebook ---
    Z = StandardScaler().fit_transform(codebook)
    pca = PCA(n_components=2, random_state=0).fit(Z)
    scores = pca.transform(Z)
    load = pca.components_.T * np.sqrt(pca.explained_variance_)   # loadings scaled by eigenvalues

    # scale loadings so arrows live in the scores extent
    smax = float(np.max(np.abs(scores))) or 1.0
    L = np.linalg.norm(load, axis=1)
    scale = 0.85 * smax / (L.max() or 1.0)
    load_s = load * scale

    # palette for clusters
    k = int(np.nanmax(labels)) + 1
    cmap_disc, norm_disc = _discrete_cmap_for_clusters(k)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8.6, 7.0), constrained_layout=True)

    # scatter by cluster
    for c in range(k):
        m = (labels == c)
        if np.any(m):
            ax.scatter(scores[m, 0], scores[m, 1],
                       s=point_size, c=[cmap_disc(norm_disc(c))],
                       edgecolors="black", linewidths=0.45, alpha=0.9,
                       label=f"{c+1}", zorder=2)

    # axes lines
    ax.axhline(0, color="0.7", lw=0.7, ls="--")
    ax.axvline(0, color="0.7", lw=0.7, ls="--")

    head = head_frac * smax
    label_off = label_offset_frac * smax

    # arrows + labels at tips
    for i in range(load_s.shape[0]):
        x, y = load_s[i, 0], load_s[i, 1]

        # arrow
        ax.arrow(0, 0, x, y, length_includes_head=True,
                 head_width=head, head_length=head,
                 lw=arrow_lw, color="black", alpha=0.95, zorder=1)

        # small outward nudge along arrow direction
        r = np.hypot(x, y) or 1.0
        ux, uy = x / r, y / r
        lx, ly = x + ux * label_off, y + uy * label_off

        # horizontal label
        ax.text(lx, ly, isotope_superscript(str(feature_names[i])),
                ha="center", va="center", fontsize=font_size, color="black", zorder=3,
                path_effects=[pe.withStroke(linewidth=label_stroke, foreground="white")])

    # cosmetics
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(
        title="Cluster",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),        # 2% from left/top inside the axes
        bbox_transform=ax.transAxes,        # interpret anchor in axes coords
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        fontsize=9, borderpad=0.3, handlelength=0.9, handletextpad=0.4
    )
    leg.get_title().set_fontsize(10)

    # --- PDF only ---
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_k_diagnostics(metrics_df: pd.DataFrame, out_dir: Path, k_db: int | None = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_df["k"].to_numpy(), metrics_df["db_weighted"].to_numpy(), marker="o")
    if k_db is not None:
        plt.axvline(k_db, color="tab:red", ls="--", lw=1.5)
    plt.title("Davies–Bouldin (weighted codebook) — lower is better")
    plt.xlabel("k"); plt.ylabel("DB")
    plt.tight_layout()
    png = out_dir / "k_db_weighted.png"
    plt.savefig(png, dpi=300); plt.savefig(png.with_suffix(".pdf"))
    plt.close()


def plot_component_planes_hex(
    som: MiniSom,
    feature_names: list[str],
    out_png: Path,
    *,
    R: float = 1.0,
    face_overfill: float = 1.0,
    cmap_name: str = "viridis",
    ncols: int = 6,
    cluster_grid: np.ndarray | None = None,
    boundary_color: str = "white",
    boundary_lw: float = 3.0,
    clusters_text_size: int = 12,
) -> None:
    """
    Component planes (per feature), min–max normalized per plane.
    If cluster_grid is provided, draw boundaries and add a final “Clusters” panel.
    """
    W = som.get_weights()   # (m, n, p)
    m, n, p = W.shape
    assert p == len(feature_names)

    Xc, Yc = _hex_centers(m, n, R)
    R_face = R * face_overfill
    verts = [_hex_vertices(Xc[i, j], Yc[i, j], R_face) for i in range(m) for j in range(n)]

    total_panels = p + (1 if cluster_grid is not None else 0)
    ncols = max(1, min(ncols, total_panels))
    nrows = int(math.ceil(total_panels / ncols))

    panel = 2.35
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.002, h_pad=0.002, wspace=0.0, hspace=0.0)

    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()

    cmap = plt.get_cmap(cmap_name)
    norm_shared = Normalize(vmin=0.0, vmax=1.0)

    V = np.concatenate(verts, axis=0)
    xmin, ymin = V.min(0); xmax, ymax = V.max(0)
    dx, dy = xmax - xmin, ymax - ymin
    xlo, xhi = xmin - 0.01 * dx, xmax + 0.01 * dx
    ylo, yhi = ymin - 0.01 * dy, ymax + 0.01 * dy

    plane_axes: list[plt.Axes] = []

    # Per-feature planes
    for idx in range(p):
        ax = flat_axes[idx]
        plane = W[:, :, idx]
        vmin, vmax = np.nanmin(plane), np.nanmax(plane)
        plane_norm = (plane - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(plane)

        colors = [cmap(norm_shared(float(plane_norm[i, j]))) for i in range(m) for j in range(n)]
        pc = PolyCollection(verts, facecolors=colors, edgecolors="none",
                            linewidths=0.0, antialiased=False, closed=True)
        ax.add_collection(pc)

        if cluster_grid is not None:
            _add_cluster_boundaries(ax, Xc, Yc, R_face, cluster_grid,
                                    color=boundary_color, lw=boundary_lw)

        ax.set_aspect("equal")
        ax.set_title(isotope_superscript(feature_names[idx]), fontsize=16, pad=2)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        plane_axes.append(ax)

    # Final “Clusters” panel
    if cluster_grid is not None:
        axc = flat_axes[p]
        k = int(np.nanmax(cluster_grid)) + 1

        vals = np.linspace(0.20, 0.85, k) if k > 1 else np.array([0.5])
        cmap_disc = ListedColormap([(v, v, v, 1.0) for v in vals])
        norm_disc = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap_disc.N)
        sm_disc = ScalarMappable(norm=norm_disc, cmap=cmap_disc)

        colors_c = [sm_disc.to_rgba(cluster_grid[i, j]) for i in range(m) for j in range(n)]
        pc = PolyCollection(verts, facecolors=colors_c, edgecolors="none",
                            antialiased=False, closed=True, zorder=1)
        axc.add_collection(pc)

        _add_cluster_boundaries(axc, Xc, Yc, R_face, cluster_grid, color="black", lw=2.0)

        _label_cluster_components(axc, Xc, Yc, cluster_grid, text_size=clusters_text_size)

        axc.set_aspect("equal")
        axc.set_title(f"Clusters (k={k})", fontsize=16, pad=2)
        axc.set_xlim(xlo, xhi); axc.set_ylim(ylo, yhi)
        axc.set_xticks([]); axc.set_yticks([])
        for s in axc.spines.values():
            s.set_visible(False)

    # Hide any unused subplots
    for idx in range(total_panels, nrows * ncols):
        flat_axes[idx].axis("off")

    # Shared colorbar for component planes
    sm = plt.cm.ScalarMappable(norm=norm_shared, cmap=cmap)
    sm.set_array([])
    if plane_axes:
        cbar = fig.colorbar(sm, ax=plane_axes, fraction=0.03, pad=0.02)
        cbar.set_label("Min–Max (per parameter)", fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    fig.suptitle("Component Planes (min–max normalized)", fontsize=18, y=1.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# K Scan & Metrics
# =============================================================================

def kmeans_scan_metrics_weighted(codebook: np.ndarray,
                                 X_weighted: np.ndarray,
                                 k_min: int = 2,
                                 k_max: int = 9,
                                 seed: int = 42,
                                 n_init: int = 50):
    """
    Scan K on the sample-weighted codebook using the Davies–Bouldin index.
    Returns: (labels_best_on_codebook, k_best, db_best, metrics_df, labels_per_k)
    """
    rows, labels_per_k = [], {}
    best_db, best_k, best_labels_cb = float("inf"), None, None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        labels_w = km.fit_predict(X_weighted)
        dbw = float(davies_bouldin_score(X_weighted, labels_w))
        labels_cb = km.predict(codebook)

        labels_per_k[k] = labels_cb
        rows.append({"k": k, "db_weighted": dbw, "inertia": float(km.inertia_)})

        if dbw < best_db:
            best_db, best_k, best_labels_cb = dbw, k, labels_cb

    metrics_df = pd.DataFrame(rows)
    return best_labels_cb, int(best_k), float(best_db), metrics_df, labels_per_k


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=("Hex SOM + KMeans (DB-based K selection on sample-weighted codebook). "
                     "Reads Excel with header on row 2; excludes a column named 'h' from features.")
    )
    ap.add_argument("--xlsx", required=True, help="Path to the Excel file")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--out_dir", default="outputs", help="Root output directory")
    ap.add_argument("--som_m", type=int, default=10, help="SOM grid rows (m)")
    ap.add_argument("--som_n", type=int, default=10, help="SOM grid cols (n)")
    ap.add_argument("--som_iters", type=int, default=1500, help="SOM training iterations")
    ap.add_argument("--sigma", type=float, default=1.2, help="Neighborhood sigma")
    ap.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    ap.add_argument("--k_min", type=int, default=2, help="Min K to scan")
    ap.add_argument("--k_max", type=int, default=9, help="Max K to scan")
    ap.add_argument("--k_inits", type=int, default=50, help="KMeans n_init per K")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--k", type=int, default=-1, help="Optional fixed K for an extra cluster map")
    ap.add_argument("--u_center", choices=["builtin", "mean", "median"], default="builtin",
                    help="How to compute dense U-matrix centers")
    ap.add_argument("--depth_col", default="h",
                    help="Name of the depth/height column (used only for vertical plots).")
    ap.add_argument("--profile_col", default="",
                    help="Column that identifies the vertical profile/borehole (optional).")
    ap.add_argument("--depth_bins", type=int, default=12,
                    help="Number of depth bins for the stacked summary (reserved).")
    ap.add_argument("--id_col", default="",
                    help="Column to use as sample ID (overrides autodetect).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ----- Output layout ------------------------------------------------------
    out_root = Path(args.out_dir)
    plots = out_root / "plots"
    paths = {
        "log": out_root / "log",
        "data": out_root / "data",
        "diagn": out_root / "diagnostics",
        "umatrix": plots / "umatrix",
        "hits": plots / "hits",
        "clusters": plots / "clusters",
        "components": plots / "components",
        "profile": plots / "profile",
        "boxes": plots / "boxes",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    setup_logging(paths["log"])
    
    # --- Environment & versions (for reproducibility) ---
    import matplotlib, sklearn, minisom  # already imported elsewhere; here just for versions
    logging.info("Environment: Python=%s | OS=%s", sys.version.split()[0], platform.platform())
    logging.info("Versions: numpy=%s, pandas=%s, scikit-learn=%s, matplotlib=%s, minisom=%s",
                 np.__version__, pd.__version__, sklearn.__version__, matplotlib.__version__,
                 getattr(minisom, "__version__", "unknown"))

    (env_txt := paths["log"] / "environment.txt").write_text(
        "\n".join([
            f"Python=={sys.version.split()[0]}",
            f"OS=={platform.platform()}",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"matplotlib=={matplotlib.__version__}",
            f"minisom=={getattr(minisom, '__version__', 'unknown')}",
        ]) + "\n",
        encoding="utf-8"
    )

    # ----- Read & prepare data ----------------------------------------------
    xlsx = Path(args.xlsx)
    logging.info("Reading Excel: %s (sheet=%s)", xlsx, args.sheet or "<first>")
    df = read_excel_header_row2(xlsx, sheet=args.sheet)

    sample_id = choose_sample_id(df, prefer=(args.id_col or None))
    df = df.copy()
    df["SampleID"] = sample_id.values
    logging.info("Sample ID source: %s; unique=%d/%d",
                 "--id_col" if args.id_col else "auto", df["SampleID"].nunique(), len(df))

    # Build feature list: exclude ID, depth, profile, and common ID synonyms
    ID_SYNONYMS = {
        "sample", "sample id", "sampleid", "id", "name", "code", "label", "tag",
        "amostra", "codigo", "código", "cod amostra", "codigo amostra",
        "sample_code", "sample code", "sample_name", "sample name"
    }

    # start with fixed exclusions
    non_feat = {"sampleid", "h"}
    if args.depth_col:
        non_feat.add(str(args.depth_col).strip().lower())
    if args.profile_col:
        non_feat.add(str(args.profile_col).strip().lower())
    if getattr(args, "id_col", ""):
        non_feat.add(str(args.id_col).strip().lower())

    # also exclude any column whose name matches ID synonyms
    non_feat |= {str(c).strip().lower() for c in df.columns
                 if str(c).strip().lower() in ID_SYNONYMS}

    # additionally exclude any column that is effectively identical to SampleID
    def _strip_suffix_num(s: str) -> str:
        # remove "#2" etc. suffixes and trim spaces
        return re.sub(r"#\d+$", "", str(s).strip())

    if "SampleID" in df.columns:
        _sid_base = df["SampleID"].astype(str).map(_strip_suffix_num)
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in non_feat or c == "SampleID":
                continue
            try:
                if df[c].astype(str).map(_strip_suffix_num).equals(_sid_base):
                    non_feat.add(cl)
            except Exception:
                pass  # ignore columns that cannot be compared reliably

    # final feature list
    feature_cols = [c for c in df.columns
                    if str(c).strip().lower() not in non_feat]
    if not feature_cols:
        raise RuntimeError("No feature columns found (after excluding IDs/profile/depth).")

    # Coerce numerics with censoring rules, drop unusable columns
    df = coerce_numeric(df, feature_cols)
    keep_cols = [c for c in feature_cols if df[c].notna().any()]
    keep_cols = [c for c in keep_cols if df[c].nunique(dropna=True) > 1]
    dropped_cols = sorted(set(feature_cols) - set(keep_cols))
    feature_cols = keep_cols
    if dropped_cols:
        logging.info("Dropped non-informative columns: %s", ", ".join(dropped_cols))
    if len(feature_cols) < 2:
        raise RuntimeError(f"Not enough usable feature columns (found {len(feature_cols)}).")

    df = df[df[feature_cols].notna().any(axis=1)].reset_index(drop=True)

    # Impute per-column medians, z-score scale, and ensure finite
    med = df[feature_cols].median(numeric_only=True).fillna(0.0)
    X = df[feature_cols].fillna(med).to_numpy(dtype=float)
    X = force_finite(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ----- Train MiniSom (hex) ----------------------------------------------
    logging.info("Training MiniSom: m=%d, n=%d, iters=%d, sigma=%.3f, lr=%.3f",
                 args.som_m, args.som_n, args.som_iters, args.sigma, args.lr)
    som = MiniSom(args.som_m, args.som_n, Xs.shape[1],
                  sigma=args.sigma, learning_rate=args.lr,
                  topology="hexagonal", random_seed=args.seed)
    som.random_weights_init(Xs)
    som.train_random(Xs, args.som_iters, verbose=False)

    # ----- BMUs, codebook, K scan on codebook -------------------------------
    bmus = np.array([som.winner(x) for x in Xs])  # (n_samples, 2)
    codebook = som.get_weights().reshape(-1, Xs.shape[1])
    codebook = force_finite(codebook)
    
    # --- Sample-weighted codebook (replicate prototypes by BMU hits) ---
    hits_grid = _compute_hits_grid(som, Xs)               # (m, n)
    hits_flat = hits_grid.reshape(-1).astype(int)         # (m*n,)
    mask_nz = hits_flat > 0
    codebook_nz = codebook[mask_nz]
    reps = hits_flat[mask_nz]
    X_weighted = np.repeat(codebook_nz, reps, axis=0)     # acts like sample-level data

    labels_db, k_db, db_best, metrics_df, labels_per_k = kmeans_scan_metrics_weighted(
        codebook, X_weighted,
        k_min=args.k_min, k_max=args.k_max,
        seed=args.seed, n_init=args.k_inits
    )
    metrics_df.to_csv(paths["diagn"] / "k_selection_metrics_weighted.csv", index=False)
    logging.info("Best-by-DB (on sample-weighted codebook): k=%d (DBw=%.3f)", k_db, db_best)
                 
    # --- Save artifacts for full reproducibility ---
    # refit a best-k model on full X_weighted for persistence
    km_best = KMeans(n_clusters=k_db, n_init=args.k_inits, random_state=args.seed).fit(X_weighted)

    np.save(paths["data"] / "som_weights.npy", som.get_weights())
    with open(paths["data"] / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(paths["data"] / "kmeans_best.pkl", "wb") as f:
        pickle.dump(km_best, f)

    # Save simple config
    cfg = {
        "som_m": args.som_m, "som_n": args.som_n, "som_iters": args.som_iters,
        "sigma": args.sigma, "lr": args.lr, "k_min": args.k_min, "k_max": args.k_max,
        "k_inits": args.k_inits, "seed": args.seed, "u_center": args.u_center,
        "chosen_k_db_weighted": int(k_db)
    }
    (paths["data"] / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ----- Diagnostics: K plot ---------------------------------------------
    plot_k_diagnostics(metrics_df, paths["diagn"], k_db=k_db)

    # ----- Grid of cluster labels -------------------------------------------
    cluster_grid_db = labels_db.reshape(args.som_m, args.som_n)

    # Optional extra user-specified K
    labels_k = None
    if args.k and args.k >= 2:
        km = KMeans(n_clusters=args.k, n_init=args.k_inits, random_state=args.seed)
        labels_k = km.fit_predict(codebook)
        logging.info("Extra KMeans with user K=%d: inertia=%.2f", args.k, km.inertia_)

    # PCA biplot of the original codebook (feature arrows; labels at arrow tips)
    plot_codebook_pca_biplot(
        codebook=codebook,
        feature_names=feature_cols,
        labels=labels_db,
        out_pdf=paths["diagn"] / f"codebook_pca_biplot_k{k_db}.pdf",
        title=f"Codebook PCA with Variable Loadings (k={k_db})"
    )

    # Optional: also for user-specified K
    if labels_k is not None:
        plot_codebook_pca_biplot(
            codebook=codebook,
            feature_names=feature_cols,
            labels=labels_k,
            out_pdf=paths["diagn"] / f"codebook_pca_biplot_k{args.k}.pdf",
            title=f"Codebook PCA — Biplot (k={args.k})"
        )

    # ----- Assign samples to clusters via BMUs -------------------------------
    assign = pd.DataFrame({
        "SampleID": df["SampleID"].tolist(),
        "BMU_i": [i for (i, _) in bmus],
        "BMU_j": [j for (_, j) in bmus],
        "Cluster_db": [cluster_grid_db[i, j] for (i, j) in bmus],
    })
    if labels_k is not None:
        cluster_grid_k = labels_k.reshape(args.som_m, args.som_n)
        assign["Cluster_k"] = [cluster_grid_k[i, j] for (i, j) in bmus]

    # Attach optional columns
    if args.depth_col and args.depth_col in df.columns:
        assign[args.depth_col] = df[args.depth_col]
    if args.profile_col and args.profile_col in df.columns:
        assign[args.profile_col] = df[args.profile_col]

    # Save data artifacts
    assign.to_csv(paths["data"] / "som_sample_assignments.csv", index=False)
    codebook_df = pd.DataFrame(codebook, columns=feature_cols)
    codebook_df["Cluster_db"] = labels_db
    if labels_k is not None:
        codebook_df["Cluster_k"] = labels_k
    codebook_df.to_csv(paths["data"] / "som_codebook_vectors.csv", index=False)

    assign.groupby("Cluster_db").size().rename("n_samples").to_csv(paths["data"] / "cluster_sizes_db.csv")
    if labels_k is not None:
        assign.groupby("Cluster_k").size().rename("n_samples").to_csv(paths["data"] / f"cluster_sizes_k{args.k}.csv")

    # --- Per-cluster summaries (median & IQR) ---
    df_stats = df[feature_cols].copy()
    df_stats["Cluster_db"] = assign["Cluster_db"].to_numpy()
    rows = []
    for c in sorted(df_stats["Cluster_db"].unique().tolist()):
        sub = df_stats[df_stats["Cluster_db"] == c]
        rec = {"Cluster_db": int(c)}
        for feat in feature_cols:
            s = pd.to_numeric(sub[feat], errors="coerce").dropna()
            if s.empty:
                rec[f"{feat}__median"] = np.nan
                rec[f"{feat}__IQR"]    = np.nan
            else:
                rec[f"{feat}__median"] = float(np.median(s))
                rec[f"{feat}__IQR"]    = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(paths["data"] / "cluster_feature_summary.csv", index=False)

    # ----- Plots: U-matrix, hits, clusters, components, boxes, profile -----
    plot_umatrix_hex_dense(
        som, paths["umatrix"] / "som_umatrix_hex_dense.png",
        R=1.0, face_overfill=1.0, center_mode=args.u_center
    )
    plot_hits_hex(som, Xs, paths["hits"] / "som_hits_hex.png")
    plot_hits_hex_with_ids(
        som, Xs, df["SampleID"].tolist(),
        paths["hits"] / "som_hits_hex_ids.png",
        R=1.0, face_overfill=1.0,
        max_ids_per_cell=8, max_chars_per_id=12
    )
    plot_clusters_hex(
        cluster_grid_db, som, Xs,
        paths["clusters"] / f"som_clusters_hex_db_k{k_db}.png",
        title=f"KMeans clusters (best DB on codebook: k={k_db}, DB={db_best:.3f})",
        boundary_color="white", boundary_lw=3.0
    )
    if labels_k is not None:
        plot_clusters_hex(
            cluster_grid_k, som, Xs,
            paths["clusters"] / f"som_clusters_hex_k{args.k}.png",
            title=f"KMeans clusters (user K={args.k})",
            boundary_color="white", boundary_lw=3.0
        )

    # Per-k cluster maps
    for k, lab in labels_per_k.items():
        grid = lab.reshape(args.som_m, args.som_n)
        plot_clusters_hex(
            grid, som, Xs,
            paths["clusters"] / f"som_clusters_hex_k{k}.png",
            title=f"KMeans clusters (k={k})",
            boundary_color="white", boundary_lw=3.0
        )

    # Component planes (+ final clusters panel)
    plot_component_planes_hex(
        som, feature_cols, paths["components"] / "som_component_planes_hex.png",
        cluster_grid=cluster_grid_db, boundary_color="white",
        boundary_lw=3.0, ncols=6, clusters_text_size=12
    )

    # Per-feature individual planes with boundaries
    W = som.get_weights(); m, n, p = W.shape
    Xc, Yc = _hex_centers(m, n, 1.0)
    R_draw = 1.0
    cmap = plt.get_cmap("viridis"); norm_shared = Normalize(vmin=0.0, vmax=1.0)
    for idx in range(p):
        plane = W[:, :, idx]
        vmin, vmax = np.nanmin(plane), np.nanmax(plane)
        plane_norm = (plane - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(plane)
        verts = [_hex_vertices(Xc[i, j], Yc[i, j], R_draw) for i in range(m) for j in range(n)]
        colors = [cmap(norm_shared(float(plane_norm[i, j]))) for i in range(m) for j in range(n)]
        fig, ax = plt.subplots(figsize=(8, 8))
        pc = PolyCollection(verts, facecolors=colors, edgecolors="none",
                            linewidths=0.0, antialiased=False, closed=True)
        ax.add_collection(pc)
        sm = plt.cm.ScalarMappable(norm=norm_shared, cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Min–Max (per parameter)")
        _add_cluster_boundaries(ax, Xc, Yc, R_draw, cluster_grid_db, color="white", lw=3.0)
        ax.set_aspect("equal"); pad = 1.25
        ax.set_xlim(Xc.min() - pad, Xc.max() + pad)
        ax.set_ylim(Yc.min() - pad, Yc.max() + pad)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(isotope_superscript(feature_cols[idx]), fontsize=10)
        safe = re.sub(r"[^\w\-\.]+", "_", str(feature_cols[idx])).strip("_")
        fig.savefig(paths["components"] / f"component_{safe}.png", dpi=300, bbox_inches="tight")
        fig.savefig(paths["components"] / f"component_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Profile (if depth is available)
    if args.depth_col and args.depth_col in assign.columns:
        plot_cluster_strips_by_profile(
            assign, paths["profile"],
            cluster_col="Cluster_db",
            depth_col=args.depth_col,
            profile_col=(args.profile_col if args.profile_col and args.profile_col in assign.columns else None),
            fig_width=3.0
        )

    # Boxes
    plot_box_by_cluster(
        data_df=df,
        feature_cols=feature_cols,
        clusters=assign["Cluster_db"],
        out_dir=paths["boxes"]
    )
    plot_box_rows_by_scale(
        data_df=df,
        feature_cols=feature_cols,
        clusters=assign["Cluster_db"],
        out_png=paths["boxes"] / "boxes_rows_by_scale.png",
    )

    # ----- Summary -----------------------------------------------------------
    logging.info("OK — Outputs in: %s", out_root.resolve())
    logging.info("Used features (excluding %r): %s", args.depth_col, ", ".join(feature_cols))
    logging.info("Artifacts:")
    logging.info("  log/: run.log, environment.txt")
    logging.info("  data/: som_sample_assignments.csv, som_codebook_vectors.csv, "
                 "cluster_feature_summary.csv, cluster_sizes_db.csv, "
                 "cluster_sizes_k<userK>.csv (if --k provided), "
                 "som_weights.npy, scaler.pkl, kmeans_best.pkl, run_config.json")
    logging.info("  diagnostics/: k_selection_metrics_weighted.csv, k_db_weighted.png/pdf, "
                 "codebook_pca_biplot_k*.pdf")
    logging.info("  plots/umatrix/: som_umatrix_hex_dense.png/pdf")
    logging.info("  plots/hits/: som_hits_hex.png/pdf, som_hits_hex_ids.png/pdf")
    logging.info("  plots/clusters/: som_clusters_hex_db_k*.png/pdf, "
                 "som_clusters_hex_k<k>.png/pdf (for each scanned k and optional --k)")
    logging.info("  plots/components/: som_component_planes_hex.png/pdf, component_<feature>.png/pdf")
    logging.info("  plots/profile/: profile_<id>.png (if depth/profile provided)")
    logging.info("  plots/boxes/: per-feature box_*.png, boxes_rows_by_scale.png/pdf")


# =============================================================================
# Profile strips (vertical arrangement)
# =============================================================================

def plot_cluster_strips_by_profile(assign_df: pd.DataFrame,
                                   out_dir: Path,
                                   *,
                                   cluster_col: str = "Cluster_db",
                                   depth_col: str = "h",
                                   profile_col: str | None = None,
                                   fig_width: float = 3.0,
                                   dot_size: int = 260,
                                   glue_factor: float = 1.00,
                                   overlap_pct: float = 0.05,
                                   inter_gap_factor: float = 0.20,
                                   min_gap_px: float = 10.0,
                                   zero_labels: tuple[str, str] = ("Irati Fm.", "Diabase")) -> None:
    """
    Profile strips:
      • Same depth & same cluster → dots visually glued (tiny overlap).
      • Same depth but different clusters → small fixed gap between blocks.
      • X uses axes-fraction; Y uses data units (depth h).
      • Each dot shows the last two digits of SampleID.
      • Baseline at h=0 with labels above/below.
    """
    if assign_df.empty or depth_col not in assign_df.columns or cluster_col not in assign_df.columns:
        return

    df = assign_df.dropna(subset=[depth_col, cluster_col]).copy()
    if df.empty:
        return

    if "SampleID" not in df.columns:
        df["SampleID"] = [f"S{i+1}" for i in range(len(df))]

    k_here = int(np.nanmax(df[cluster_col])) + 1
    cmap_disc, norm_disc = _discrete_cmap_for_clusters(k_here)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = (df.groupby(profile_col) if profile_col and profile_col in df.columns
              else [("All samples", df)])

    def _last2(s: str) -> str:
        s = str(s).strip()
        m = re.match(r'^\D*(\d+)', s)
        if m:
            digits = m.group(1)
        else:
            ds = re.findall(r'\d+', s)
            digits = ''.join(ds) if ds else ''
        if digits:
            return digits[-2:].zfill(2)
        return s[-2:] if len(s) >= 2 else s

    for prof_id, g in groups:
        if g.empty:
            continue

        rows = []
        for depth_val, sub in g.groupby(depth_col, sort=True):
            sub = sub.sort_values([cluster_col, "SampleID"])
            blocks = [(int(ci), subc["SampleID"].astype(str).tolist())
                      for ci, subc in sub.groupby(cluster_col, sort=True)]
            rows.append((float(depth_val), blocks))
        if not rows:
            continue

        # Figure height based on vertical range & typical spacing (unit-agnostic)
        vals = pd.to_numeric(g[depth_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            v_sorted = np.unique(np.sort(vals))
            med_step = float(np.median(np.diff(v_sorted))) if v_sorted.size > 1 else 1.0
            med_step = med_step if med_step > 0 else 1.0
            yrange = float(v_sorted[-1] - v_sorted[0])
            approx_levels = yrange / med_step
        else:
            approx_levels, yrange = 1.0, 1.0

        # Keep height reasonable regardless of units; clamp to [4, 10] inches
        fig_h = float(np.clip(3.5 + 0.06 * approx_levels, 4.0, 10.0))

        fig, ax = plt.subplots(figsize=(fig_width, fig_h))

        trans = blended_transform_factory(ax.transAxes, ax.transData)

        fig.canvas.draw()
        diam_pt = 2.0 * np.sqrt(float(dot_size) / np.pi)     # diameter in points
        diam_px = diam_pt * fig.dpi / 72.0
        ax_w_px = ax.bbox.width
        marker_diam_frac = diam_px / ax_w_px

        touch_step = max(0.0, (glue_factor * (1.0 - overlap_pct)) * marker_diam_frac)
        gap_frac = max(inter_gap_factor * marker_diam_frac, min_gap_px / ax_w_px)

        Xf, Yd, C, TXT = [], [], [], []
        for depth_val, blocks in rows:
            if not blocks:
                continue
            xs_frac, cs, ids_all = [], [], []
            cursor = 0.0
            for ci, ids in blocks:
                n = len(ids)
                if n <= 0:
                    continue
                xs_block = [cursor + j * touch_step for j in range(n)]
                xs_frac.extend(xs_block)
                cs.extend([ci] * n)
                ids_all.extend(ids)
                cursor = xs_block[-1] + marker_diam_frac + gap_frac
            if not xs_frac:
                continue
            mid = 0.5 * (min(xs_frac) + max(xs_frac))
            xs_frac = [0.5 + (x - mid) for x in xs_frac]

            Xf.extend(xs_frac)
            Yd.extend([depth_val] * len(xs_frac))
            C.extend(cs)
            TXT.extend([_last2(s) for s in ids_all])

        if not Xf:
            plt.close(fig); continue

        Xf = np.asarray(Xf, float)
        Yd = np.asarray(Yd, float)
        C = np.asarray(C, int)
        TXT = np.asarray(TXT, str)

        for ci in sorted(np.unique(C).tolist()):
            msk = (C == ci)
            col = cmap_disc(norm_disc(int(ci)))
            ax.scatter(Xf[msk], Yd[msk], s=dot_size, c=[col], marker="o",
                       edgecolors="none", alpha=0.98, zorder=2,
                       transform=trans, label=str(int(ci) + 1))
            tcol = _best_text_color(col)
            for xf, y, t in zip(Xf[msk], Yd[msk], TXT[msk]):
                ax.text(xf, y, t, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=tcol,
                        transform=trans, zorder=3)

        ymin_data = float(np.nanmin(Yd))
        ymax_data = float(np.nanmax(Yd))
        yr = max(1e-9, ymax_data - ymin_data)
        ypad = 0.06 * yr
        ax.set_ylim(ymin_data - ypad, ymax_data + ypad)
        # Dynamic ticks: ~8–10 major ticks regardless of units (m, cm, etc.)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))  # nice ticks
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0])                   # drop negatives
        ax.set_ylabel(f"{depth_col} (cm)")

        ax.set_xlim(0.0, 1.0); ax.set_xticks([])

        ax.axhline(0.0, color="k", lw=2, zorder=1)
        x_anchor = 0.99; offset = 8
        ax.annotate(zero_labels[0], xy=(x_anchor, 0.0), xycoords=('axes fraction', 'data'),
                    xytext=(0, +offset), textcoords='offset points',
                    rotation=90, ha='center', va='bottom', clip_on=False)
        ax.annotate(zero_labels[1], xy=(x_anchor, 0.0), xycoords=('axes fraction', 'data'),
                    xytext=(0, -offset + 2), textcoords='offset points',
                    rotation=90, ha='center', va='top', clip_on=False)

        ax.grid(axis="y", linewidth=0.5, alpha=0.35)
        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_title(f"Profile: {prof_id}")

        trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        leg = ax.legend(
            title="Cluster", ncol=2, loc="upper left",
            bbox_to_anchor=(-0.2, -0.15), bbox_transform=trans,
            frameon=True, fancybox=True, framealpha=0.9,
            fontsize=8, markerscale=0.5,
            handlelength=0.9, handletextpad=0.3,
            columnspacing=0.5, labelspacing=0.2, borderpad=0.3
        )
        leg.get_title().set_fontsize(8)

        plt.tight_layout()
        plt.savefig((Path(out_dir) / f"profile_{str(prof_id)}.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.savefig((Path(out_dir) / f"profile_{str(prof_id)}.pdf"),
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# =============================================================================

if __name__ == "__main__":

    main()

