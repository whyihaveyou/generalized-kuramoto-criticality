#!/usr/bin/env python3
"""
图A：序参量/磁化率/Binder累积量 vs K（多尺寸有限尺度标度分析）

用途：展示不同系统尺寸N下，物理量随耦合强度K的变化，Kc处灰色虚线标记临界点。
风格确定版（2026-03-20）：
  - 空心标记 + 细边框(mew=0.6)
  - 配色：粉红/蓝/绿/紫/橙/棕/红（对应N从小到大）
  - 全局字体12pt，图例9pt，LaTeX数学字体标题
  - 三子图顺序：⟨m⟩ → χ → U₄
  - 所有物理量带误差棒

用法：
  # 2D单个sigma
  python plot_figureA.py --dim 2d --sigma 0.40
  # 2D全部
  python plot_figureA.py --dim 2d --all
  # 1D全部
  python plot_figureA.py --dim 1d --all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, matplotlib, argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from config import (
    STAT_1D, STAT_2D,
    OUT_1D_A as OUT_1D, OUT_2D_A as OUT_2D,
    RAW_1D_BASE, KC_1D, KC_2D, N_1D, N_2D,
    PANELS, spectral_ds,
)

# ====== 全局字体/样式 ======
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.linewidth'] = 0.8

# ====== 配色方案：空心+细边框 ======
COLORS = ['#f781bf', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#e41a1c']
MARKERS = ['s', '^', 'D', 'o', 'v', 'p', 'h']
MEW = 0.6       # 标记边框宽度
MS  = 5          # 标记大小
ELINEW = 0.5     # 误差棒线宽
LEGEND_FS = 9    # 图例字号


def mask_k_has_raw_folder(raw_base, sigma, sig_fmt, N, ks):
    """每个 K 仅在对应 raw 目录 K_{k:.2f} 存在时保留（与 merged_tail5_6 联用，避免无 raw 继承旧窗造成锯齿）。"""
    sigma_str = f'{sigma:{sig_fmt}}'
    base = f'{raw_base}/sigma_{sigma_str}/N_{int(N)}'
    out = np.zeros(len(ks), dtype=bool)
    for i, k in enumerate(ks):
        if not np.isfinite(k):
            continue
        out[i] = os.path.isdir(f'{base}/K_{float(k):.2f}')
    return out


def plot_figureA(
    sigma,
    stat_dir,
    Kc,
    N_list,
    out_dir,
    dim_label,
    sig_fmt,
    d_s,
    raw_base=None,
    only_k_with_raw=False,
):
    """绘制单个sigma的图A"""
    colors_N = COLORS[:len(N_list)]
    markers  = MARKERS[:len(N_list)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r'$\mathsf{' + dim_label + r'},\ \sigma = ' + f'{sigma:{sig_fmt}}'
        + r',\ d_{\mathrm{s}} = ' + f'{d_s:.2f}' + '$',
        fontsize=14, fontweight='bold',
    )

    for col, (title, ylabel, fname, std_fname) in enumerate(PANELS):
        ax = axes[col]
        for j, N in enumerate(N_list):
            d = f'{stat_dir}/sigma_{sigma:{sig_fmt}}/N_{N}'
            try:
                ks   = np.load(f'{d}/K_values.npy')
                data = np.load(f'{d}/{fname}.npy')
                std  = np.load(f'{d}/{std_fname}.npy')
            except FileNotFoundError:
                continue

            mec = colors_N[j]
            valid = np.isfinite(data) & (data > 0)
            if only_k_with_raw and raw_base is not None:
                valid = valid & mask_k_has_raw_folder(
                    raw_base, sigma, sig_fmt, N, ks
                )
            if valid.sum() < 2:
                continue

            ax.errorbar(ks[valid], data[valid], yerr=std[valid],
                       marker=markers[j], color=mec, markerfacecolor='none',
                       markeredgecolor=mec, markeredgewidth=MEW,
                       ms=MS, alpha=0.9, lw=1.0,
                       elinewidth=ELINEW, ecolor='0.6',
                       label=f'$N={N}$')

        # Kc虚线（灰色，不标注文字）
        ax.axvline(Kc, color='0.4', ls='--', lw=1.2, alpha=0.6)
        ax.set_xlabel('$K$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, lw=0.4)

    # 图例放最后一个子图内，单列
    axes[2].legend(fontsize=LEGEND_FS, loc='best', ncol=1, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    outpath = f'{out_dir}/figure_A_sigma_{sigma:{sig_fmt}}.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ σ={sigma:{sig_fmt}}, Kc={Kc} → {outpath}')

def main():
    parser = argparse.ArgumentParser(description='图A批量绘制')
    parser.add_argument('--dim', required=True, choices=['1d', '2d'])
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument(
        '--stat-dir',
        type=str,
        default=None,
        help='覆盖 statistics 根目录（1D/2D 均可用；默认各用 STAT_1D / STAT_2D）',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='覆盖 figure_A 输出根目录（1D/2D 均可用）',
    )
    parser.add_argument(
        '--1d-only-k-with-raw',
        action='store_true',
        dest='only_k_with_raw',
        help='仅 1D：只绘制 raw 下存在 K_xx.xx 目录的数据点（配合 statistics_merged_tail5_6）',
    )
    parser.add_argument(
        '--raw-base',
        type=str,
        default=None,
        help='raw 根目录；默认 1d_lr3/D_2/raw（仅在与 --1d-only-k-with-raw 联用时生效）',
    )
    args = parser.parse_args()

    if args.dim == '1d':
        kc_dict, N_list, stat_dir, out_dir, sig_fmt = KC_1D, N_1D, STAT_1D, OUT_1D, '.3f'
        dim_label = '1D LR3'
        if args.stat_dir is not None:
            stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir))
        if args.out_dir is not None:
            out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        raw_base = None
        if args.only_k_with_raw:
            raw_base = os.path.abspath(
                os.path.expanduser(args.raw_base or RAW_1D_BASE)
            )
    else:
        kc_dict, N_list, stat_dir, out_dir, sig_fmt = KC_2D, N_2D, STAT_2D, OUT_2D, '.2f'
        dim_label = '2D LRDG'
        if args.stat_dir is not None:
            stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir))
        if args.out_dir is not None:
            out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        raw_base = None
        if args.only_k_with_raw:
            print(
                '警告: --1d-only-k-with-raw 仅对 --dim 1d 生效，已忽略',
                file=sys.stderr,
            )

    sigmas = sorted(kc_dict.keys())
    if args.sigma is not None:
        sigmas = [args.sigma]
    elif not args.all:
        parser.error('需要 --sigma 或 --all')

    dim_1d = args.dim == '1d'
    for sigma in sigmas:
        Kc = kc_dict[sigma]
        plot_figureA(
            sigma,
            stat_dir,
            Kc,
            N_list,
            out_dir,
            dim_label,
            sig_fmt,
            spectral_ds(dim_1d, sigma),
            raw_base=raw_base if dim_1d else None,
            only_k_with_raw=args.only_k_with_raw and dim_1d,
        )

if __name__ == '__main__':
    main()
