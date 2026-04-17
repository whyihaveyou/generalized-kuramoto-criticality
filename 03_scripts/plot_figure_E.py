#!/usr/bin/env python3
"""
图 E：磁化率 χ(K) 峰在耦合方向上的宽度 vs 系统尺寸 — 1D LR3 / 2D LRDG。

纵轴为峰在 K 方向的宽度 ΔK（半峰高定义，图中不用 FWHM 缩写）。

**定稿版式**（与草稿 E04 一致）：tab20 离散色 + 图例区分 σ；横轴为真实系统尺寸 N、
对数坐标，刻度标注整数 N（如 256、4096），便于一眼读出尺寸。

输出：
  1d_lr3/D_2/plots/figure_E/figure_E_chi_peak_width_vs_N_1d.png
  2d_lrdg/D_2/plots/figure_E/figure_E_chi_peak_width_vs_N_2d.png

运行：python3 -u plot_chi_peak_width_draft.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from plot_figure_A import mask_k_has_raw_folder
from config import (
    RAW_1D_BASE,
    STAT_1D, STAT_2D,
    DATA_1D, DATA_2D, N_1D, N_2D,
    OUT_1D_E, OUT_2D_E,
)

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.linewidth"] = 0.8

LINEWIDTH = 1.2
MARKER_SIZE = 22
MEW = 0.55

YLABEL = r"Peak width $\Delta K$ (half-maximum of $\chi(K)$)"
XLABEL_N = r"System size $N$ (log scale)"


def load_chi(
    stat_dir: str,
    sigma_str: str,
    N: int,
    sig_fmt: str,
    raw_base: str | None,
    sigma_f: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = f"{stat_dir}/sigma_{sigma_str}/N_{int(N)}"
    K = np.load(f"{d}/K_values.npy")
    chi = np.load(f"{d}/magnetic_susceptibilities.npy")
    ok = np.isfinite(K) & np.isfinite(chi) & (chi > 0)
    K, chi = K[ok], chi[ok]
    if raw_base is not None:
        m = mask_k_has_raw_folder(raw_base, sigma_f, sig_fmt, int(N), K)
        K, chi = K[m], chi[m]
    order = np.argsort(K)
    return K[order], chi[order]


def peak_width_delta_K(K: np.ndarray, chi: np.ndarray) -> float:
    """Half-maximum width in K: baseline χ_min–χ_max, half level in between."""
    if len(K) < 4:
        return float("nan")
    chi_max = float(np.max(chi))
    chi_min = float(np.min(chi))
    if chi_max <= chi_min:
        return float("nan")
    half = chi_min + 0.5 * (chi_max - chi_min)
    peak_idx = int(np.argmax(chi))

    def cross(k0: float, c0: float, k1: float, c1: float) -> float:
        if (c0 - half) * (c1 - half) > 0:
            return float("nan")
        if abs(c1 - c0) < 1e-18:
            return 0.5 * (k0 + k1)
        t = (half - c0) / (c1 - c0)
        return float(k0 + t * (k1 - k0))

    left_k = float("nan")
    for i in range(peak_idx):
        kk = cross(K[i], chi[i], K[i + 1], chi[i + 1])
        if np.isfinite(kk):
            left_k = kk
            break

    right_k = float("nan")
    for i in range(peak_idx, len(K) - 1):
        kk = cross(K[i], chi[i], K[i + 1], chi[i + 1])
        if np.isfinite(kk):
            right_k = kk
            break

    if not (np.isfinite(left_k) and np.isfinite(right_k)):
        return float("nan")
    w = right_k - left_k
    return float(w) if w > 0 else float("nan")


def collect_widths(
    stat_dir: str,
    sigmas: list[float],
    Ns: np.ndarray,
    sig_fmt: str,
    raw_base: str | None,
) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for sigma in sigmas:
        row = []
        for N in Ns:
            sigma_str = f"{sigma:{sig_fmt}}"
            try:
                K, chi = load_chi(stat_dir, sigma_str, int(N), sig_fmt, raw_base, sigma)
                row.append(peak_width_delta_K(K, chi))
            except (FileNotFoundError, OSError):
                row.append(float("nan"))
        out[sigma] = np.array(row, dtype=float)
    return out


def plot_one_dim(
    *,
    data_rows: list[tuple],
    stat_dir: str,
    Ns: np.ndarray,
    sig_fmt: str,
    raw_base: str | None,
    out_path: str,
    title_tag: str,
) -> None:
    sigmas = sorted(r[0] for r in data_rows)
    widths_map = collect_widths(stat_dir, list(sigmas), Ns, sig_fmt, raw_base)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for i, sigma in enumerate(sigmas):
        w = widths_map[sigma]
        c = cmap(i % 20)
        m = np.isfinite(w)
        if m.sum() < 2:
            continue
        lab = rf"$\sigma$={sigma:g}"
        ax.plot(Ns[m], w[m], "-", color=c, lw=LINEWIDTH, label=lab, zorder=2)
        ax.scatter(
            Ns[m],
            w[m],
            s=MARKER_SIZE,
            facecolors="none",
            edgecolors=c,
            linewidths=MEW,
            zorder=3,
            clip_on=False,
        )

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels(
        [f"{int(round(float(n)))}" for n in Ns],
        rotation=40,
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel(XLABEL_N, fontsize=12)
    ax.set_ylabel(YLABEL, fontsize=12)
    ax.grid(True, alpha=0.15, lw=0.4, which="major")
    ax.minorticks_off()
    ax.tick_params(labelsize=10)
    ax.legend(ncol=2, fontsize=7.5, loc="best", framealpha=0.92)

    fig.suptitle(
        r"$\mathsf{"
        + title_tag
        + r"}$ — susceptibility $\chi$: peak width in $K$ vs $N$",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    raw_1d = os.path.abspath(RAW_1D_BASE)

    out1 = os.path.join(OUT_1D_E, "figure_E_chi_peak_width_vs_N_1d.png")
    plot_one_dim(
        data_rows=DATA_1D,
        stat_dir=STAT_1D,
        Ns=np.array(N_1D, dtype=float),
        sig_fmt=".3f",
        raw_base=raw_1d,
        out_path=out1,
        title_tag="1D LR3",
    )

    out2 = os.path.join(OUT_2D_E, "figure_E_chi_peak_width_vs_N_2d.png")
    plot_one_dim(
        data_rows=DATA_2D,
        stat_dir=STAT_2D,
        Ns=np.array(N_2D, dtype=float),
        sig_fmt=".2f",
        raw_base=None,
        out_path=out2,
        title_tag="2D LRDG",
    )


if __name__ == "__main__":
    main()
