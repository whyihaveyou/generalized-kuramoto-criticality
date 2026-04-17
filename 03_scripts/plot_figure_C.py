#!/usr/bin/env python3
r"""
图 C：有限尺度塌缩（与 figure_C_tuning_tail5_6.ipynb 塌缩段相同的作图方式）。

横轴 x = (K − K_c) N^{1/(ν d_eff)}，**左右**双面板：左 U₄，右 m N^{β/(ν d_eff)}。

**参数**与 plot_exponents_vs_ds（临界指数 vs d_s）一致：表列 (K_c, ν)，β 由最近邻 K_c 的
ln⟨m⟩–ln N 得 β = |s| ν D_U（D_U=5），d_eff = min(d_s, 5)。标题中不写统计目录名或 tail5_6。

版式：两子图均为完整四边框；不画 x=0 竖线；不显示网格。

默认 statistics 路径与 plot_exponents_vs_ds 相同（可用 --stat-dir 覆盖）。

用法：
  python3 plot_figureC.py --dim 2d --sigma 0.95
  python3 plot_figureC.py --dim 2d --sigma 0.40 0.85 0.95
  python3 plot_figureC.py --dim 2d --preview
  python3 plot_figureC.py --dim 1d --all
  python3 plot_figureC.py --dim 1d --all --1d-only-k-with-raw
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from plot_figure_A import mask_k_has_raw_folder
from config import (
    RAW_1D_BASE, spectral_ds,
    STAT_1D, STAT_2D,
    OUT_1D_C as OUT_1D, OUT_2D_C as OUT_2D,
    DATA_1D, DATA_2D, N_1D, N_2D,
)
from plot_figure_D import (
    D_UPPER,
    beta_from_slope,
    d_eff as d_eff_fn,
)

# 与 notebook figure_C_tuning_tail5_6 单面板塌缩段一致（7 色循环）
NB_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
NB_MARKERS = ["o", "s", "^", "D", "v", "<", "P"]

# --preview 时批量作图（覆盖若干 σ，含 d_eff=d_s 与 d_eff=d_u 的情形）
PREVIEW_SIGMAS_1D = [0.200, 0.350, 0.415]
PREVIEW_SIGMAS_2D = [0.40, 0.70, 0.85, 0.95]


def collapse_axis_labels(d_s: float) -> tuple[str, str]:
    """
    d_eff = min(d_s, d_u)，轴上按定义写出符号（不写数值）：
    d_s ≤ d_u 时指数里用 d_s；d_s > d_u 时指数里用 d_u（上临界维）。
    """
    if d_s <= float(D_UPPER) + 1e-9:
        xlab = r"$(K - K_c)\, N^{1/(\nu d_s)}$"
        ylab_m = r"$m\, N^{\beta/(\nu d_s)}$"
    else:
        xlab = r"$(K - K_c)\, N^{1/(\nu d_{\mathrm{u}})}$"
        ylab_m = r"$m\, N^{\beta/(\nu d_{\mathrm{u}})}$"
    return xlab, ylab_m


def table_rows_to_sigma_kc_nu(rows: list[tuple[float, float, float]]) -> dict[float, tuple[float, float]]:
    return {float(r[0]): (float(r[1]), float(r[2])) for r in rows}


def resolve_sigma_key(table: dict[float, tuple[float, float]], sigma: float) -> float | None:
    """表键为浮点字面量时，用容差匹配用户输入。"""
    for s in table:
        if abs(float(s) - float(sigma)) < 1e-9:
            return float(s)
    return None


def load_curve(
    stat_dir: str,
    sigma_str: str,
    N: int,
    sig_fmt: str,
    raw_base: str | None,
    sigma_f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, m, U₄（与 notebook 一致：仅用散点，无误差棒；K 过滤同 merged+raw）。"""
    d = f"{stat_dir}/sigma_{sigma_str}/N_{int(N)}"
    K = np.load(f"{d}/K_values.npy")
    m = np.load(f"{d}/order_parameter_means.npy")
    U4 = np.load(f"{d}/binder_cumulants.npy")
    valid = np.isfinite(K) & np.isfinite(m) & (m > 0) & np.isfinite(U4) & (U4 > 0)
    K, m, U4 = K[valid], m[valid], U4[valid]
    if raw_base is not None:
        ok = mask_k_has_raw_folder(raw_base, sigma_f, sig_fmt, int(N), K)
        K, m, U4 = K[ok], m[ok], U4[ok]
    order = np.argsort(K)
    return K[order], m[order], U4[order]


def plot_figure_c(
    sigma: float,
    Kc: float,
    nu: float,
    stat_dir: str,
    out_dir: str,
    N_list: list[int],
    sig_fmt: str,
    *,
    dim_1d: bool,
    raw_base: str | None,
) -> None:
    beta, _dk_max, _dbe = beta_from_slope(
        stat_dir, sig_fmt, sigma, np.asarray(N_list, dtype=float), Kc, nu, raw_base
    )
    if not np.isfinite(beta):
        print(f"跳过 σ={sigma:{sig_fmt}}：无法估计 β", file=sys.stderr)
        return

    d_s = spectral_ds(dim_1d, sigma)
    deff = d_eff_fn(dim_1d, sigma)
    scale_exp = nu * deff
    if scale_exp < 1e-12:
        print(f"跳过 σ={sigma:{sig_fmt}}：ν d_eff 过小", file=sys.stderr)
        return

    dim_int = 1 if dim_1d else 2
    fig_lab = "LR3" if dim_1d else "LRDG"

    # notebook 单面板塌缩：serif、尺寸、无误差棒
    style = {
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.labelsize": 14,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "mathtext.fontset": "cm",
    }

    with plt.rc_context(style):
        fig, (ax_u, ax_m) = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=False)

        for j, N in enumerate(N_list):
            sigma_str = f"{sigma:{sig_fmt}}"
            try:
                K, m, U4 = load_curve(
                    stat_dir, sigma_str, int(N), sig_fmt, raw_base, float(sigma)
                )
            except (FileNotFoundError, OSError):
                continue
            if len(K) < 2:
                continue
            n = float(N)
            x = (K - Kc) * (n ** (1.0 / scale_exp))
            y_m = m * (n ** (beta / scale_exp))
            c = NB_COLORS[j % 7]
            mk = NB_MARKERS[j % 7]
            kw = dict(
                marker=mk,
                color=c,
                ms=5.5,
                mfc="none",
                mew=1.0,
                ls="none",
                label=f"N={int(N)}",
            )
            ax_u.plot(x, U4, **kw)
            ax_m.plot(x, y_m, **kw)

        xlab, ylab_m = collapse_axis_labels(d_s)

        for ax in (ax_u, ax_m):
            ax.legend(fontsize=9, ncol=1, loc="best", framealpha=0.9)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)

        ax_u.set_ylabel("$U_4$", fontsize=14)
        ax_u.tick_params(direction="in")

        ax_m.set_ylabel(ylab_m, fontsize=14)
        ax_m.tick_params(direction="in")

        ax_u.set_xlabel(xlab, fontsize=14)
        ax_m.set_xlabel(xlab, fontsize=14)

        # 不含统计分支名 / tail5_6；谱维数写作 $d_s$（小写）
        fig.suptitle(
            f"{dim_int}D {fig_lab}   σ = {sigma:{sig_fmt}}   ($d_s$ = {d_s:.2f})   "
            f"$K_c$ = {Kc}   ν = {nu}   β = {beta:.2f}",
            fontsize=13,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.subplots_adjust(wspace=0.22)

        os.makedirs(out_dir, exist_ok=True)
        outpath = f"{out_dir}/figure_C_sigma_{sigma:{sig_fmt}}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"✅ σ={sigma:{sig_fmt}} → {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="图 C：U₄ 与序参量塌缩（左右双面板）")
    parser.add_argument("--dim", required=True, choices=["1d", "2d"])
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="*",
        default=None,
        metavar="SIGMA",
        help="一个或多个 σ（均须在 DATA 表中）；可省略与 --preview/--all 联用",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="批量绘制预设 σ（2D 与 1D 各一套列表，见脚本内 PREVIEW_SIGMAS_*）",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--stat-dir",
        type=str,
        default=None,
        help="覆盖 statistics 根目录（默认与 plot_exponents_vs_ds 相同）",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="覆盖 figure_C 输出目录")
    parser.add_argument(
        "--1d-only-k-with-raw",
        action="store_true",
        dest="only_k_with_raw",
        help="仅 1D：只使用 raw 下存在 K 目录的 K 点",
    )
    parser.add_argument(
        "--raw-base",
        type=str,
        default=None,
        help="raw 根目录（默认 1d_lr3/D_2/raw）",
    )
    args = parser.parse_args()

    dim_1d = args.dim == "1d"
    if dim_1d:
        table = table_rows_to_sigma_kc_nu(DATA_1D)
        stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir or STAT_1D))
        out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_1D))
        sig_fmt = ".3f"
        N_list = [int(x) for x in N_1D.tolist()]
        raw_base = None
        if args.only_k_with_raw:
            raw_base = os.path.abspath(os.path.expanduser(args.raw_base or RAW_1D_BASE))
    else:
        table = table_rows_to_sigma_kc_nu(DATA_2D)
        stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir or STAT_2D))
        out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_2D))
        sig_fmt = ".2f"
        N_list = [int(x) for x in N_2D.tolist()]
        raw_base = None
        if args.only_k_with_raw:
            print("警告: --1d-only-k-with-raw 仅对 --dim 1d 生效，已忽略", file=sys.stderr)

    sigmas: list[float]
    if args.preview:
        preview_list = PREVIEW_SIGMAS_1D if dim_1d else PREVIEW_SIGMAS_2D
        resolved: list[float] = []
        for s in preview_list:
            sk = resolve_sigma_key(table, s)
            if sk is not None:
                resolved.append(sk)
        sigmas = sorted(set(resolved))
        if not sigmas:
            parser.error("--preview 列表与当前维度的 DATA 表无交集")
    elif args.sigma is not None and len(args.sigma) > 0:
        sigmas = []
        for s in args.sigma:
            sk = resolve_sigma_key(table, s)
            if sk is None:
                parser.error(
                    f"σ={s} 不在临界指数表（plot_exponents_vs_ds 的 DATA_1D/DATA_2D）中"
                )
            sigmas.append(sk)
    elif args.all:
        sigmas = sorted(table.keys())
    else:
        parser.error("需要 --sigma（可多个）、--all 或 --preview")

    for sigma in sigmas:
        Kc, nu = table[sigma]
        plot_figure_c(
            sigma,
            Kc,
            nu,
            stat_dir,
            out_dir,
            N_list,
            sig_fmt,
            dim_1d=dim_1d,
            raw_base=raw_base,
        )


if __name__ == "__main__":
    main()
