#!/usr/bin/env python3
"""
图B：ln(物理量) vs ln(N) — 有限尺度标度分析

2×2布局：
  (a) ln U₄ vs ln N，不同K值散点+拟合线（K色条区分）
  (b) ln⟨m⟩ vs ln N
  (c) U₄斜率 vs K（含polyfit误差棒 + Kc灰色虚线）
  (d) ⟨m⟩斜率 vs K（含polyfit误差棒 + Kc灰色虚线）
  底部K色条：稀疏均匀刻度(0.5间隔)

风格确定版（2026-03-20）：
  - RdYlBu_r色条，K值颜色映射
  - 方案A误差棒（灰色细棒+cap）
  - Kc灰色虚线，不标注文字
  - 全部K值直接用statistics数据（含临界区密集采样）

用法：
  python3 plot_figureB.py --dim 2d --sigma 0.40
  python3 plot_figureB.py --dim 2d --all
  python3 plot_figureB.py --dim 1d --all
  # 1D + merged 统计：只使用 raw 中存在 K 目录的点（与 plot_figureA 一致）
  python3 plot_figureB.py --dim 1d --all --stat-dir .../statistics_merged_tail5_6 \\
      --out-dir .../figure_B_merged_tail5_6_rawKonly --1d-only-k-with-raw
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, matplotlib, argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings('ignore')

from plot_figure_A import mask_k_has_raw_folder
from config import (
    RAW_1D_BASE, spectral_ds,
    STAT_1D, STAT_2D,
    OUT_1D_B as OUT_1D, OUT_2D_B as OUT_2D,
    KC_1D, KC_2D, N_1D, N_2D,
)

# ====== 全局字体/样式 ======
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['figure.titlesize'] = 16

# 底部 K 色条字号（标签 + 刻度）
CBAR_LABEL_FONTSIZE = 14
CBAR_TICK_FONTSIZE = 12

# N_1D/N_2D from config are lists; convert to np.array for this script
N_1D = np.array(N_1D, dtype=float)
N_2D = np.array(N_2D, dtype=float)

def load_stats(stat_dir, sigma_str, N, filter_raw=None):
    """加载 statistics；filter_raw=(raw_base, sigma_float, sig_fmt) 时只保留 raw 下存在 K 目录的点。"""
    d = f'{stat_dir}/sigma_{sigma_str}/N_{int(N)}'
    K = np.load(f'{d}/K_values.npy')
    m = np.load(f'{d}/order_parameter_means.npy')
    U4 = np.load(f'{d}/binder_cumulants.npy')
    chi = np.load(f'{d}/magnetic_susceptibilities.npy')
    if filter_raw is not None:
        raw_base, sigma_f, sig_fmt = filter_raw
        ok = mask_k_has_raw_folder(raw_base, sigma_f, sig_fmt, int(N), K)
        K, m, U4, chi = K[ok], m[ok], U4[ok], chi[ok]
    return K, m, U4, chi

def get_Kc_auto(stat_dir, sigma_str, N_values, filter_raw=None):
    """用最大N的χ峰值自动估计Kc（备用）"""
    K, _, _, chi = load_stats(
        stat_dir, sigma_str, int(N_values[-1]), filter_raw=filter_raw
    )
    valid = (chi > 0) & np.isfinite(chi)
    return K[valid][np.argmax(chi[valid])]

def get_vals_at_K(stat_dir, sigma_str, N_values, K_val, quantity, filter_raw=None):
    """在指定K值处对所有N插值获取物理量"""
    ln_N = np.log(N_values)
    vals = np.full(len(N_values), np.nan)
    for i, N in enumerate(N_values):
        K, m, U4, chi = load_stats(
            stat_dir, sigma_str, int(N), filter_raw=filter_raw
        )
        valid = np.isfinite(U4) & np.isfinite(m) & (U4 > 0)
        src = U4[valid] if quantity == 'U4' else m[valid]
        if len(K[valid]) < 2: continue
        f = interp1d(K[valid], src, kind='linear', bounds_error=False, fill_value=np.nan)
        v = float(f(K_val))
        if np.isfinite(v) and v > 0: vals[i] = v
    return vals

def plot_figureB(
    sigma,
    stat_dir,
    out_dir,
    N_values,
    kc_dict,
    sig_fmt,
    dim_label,
    d_s,
    filter_raw=None,
):
    """绘制单个sigma的图B；filter_raw 同 load_stats。"""
    sigma_str = f'{sigma:{sig_fmt}}'
    ln_N = np.log(N_values)
    if sigma in kc_dict:
        Kc_user = kc_dict[sigma]
    else:
        Kc_user = get_Kc_auto(stat_dir, sigma_str, N_values, filter_raw=filter_raw)

    # 收集所有实际K值（union across N，可选仅含 raw 目录对应的 K）
    all_K = set()
    for N in N_values:
        K, _, U4, _ = load_stats(
            stat_dir, sigma_str, int(N), filter_raw=filter_raw
        )
        valid = np.isfinite(U4) & (U4 > 0)
        all_K.update(K[valid].tolist())
    K_sel = np.array(sorted(all_K))
    nK = len(K_sel)

    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=K_sel.min(), vmax=K_sel.max())
    colors = [cmap(norm(k))[:3] for k in K_sel]

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.32, bottom=0.09, top=0.92)
    ax_U4  = fig.add_subplot(gs[0, 0])
    ax_m   = fig.add_subplot(gs[0, 1])
    ax_U4s = fig.add_subplot(gs[1, 0])
    ax_ms  = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        r'$\mathsf{' + dim_label + r'},\ \sigma = ' + f'{sigma:{sig_fmt}}'
        + r',\ d_{\mathrm{s}} = ' + f'{d_s:.2f}' + '$',
        fontsize=17, fontweight='bold', y=0.97,
    )

    # (a) ln U4 vs ln N
    sl_U4, ks_U4 = [], []
    for j, K_val in enumerate(K_sel):
        vals = get_vals_at_K(
            stat_dir, sigma_str, N_values, K_val, 'U4', filter_raw=filter_raw
        )
        valid = np.isfinite(vals)
        if valid.sum() < 3: continue
        ln_vals = np.log(vals[valid])
        c = colors[j]
        ax_U4.scatter(ln_N[valid], ln_vals, c=[c], s=28, zorder=5,
                      edgecolors='0.2', linewidths=0.4)
        p, V = np.polyfit(ln_N[valid], ln_vals, 1, cov=True)
        xx = np.linspace(ln_N[valid].min()-0.05, ln_N[valid].max()+0.05, 50)
        ax_U4.plot(xx, p[0]*xx+p[1], '-', color=c, lw=1.2, alpha=0.8)
        sl_U4.append(p[0]); ks_U4.append(K_val)

    ax_U4.set_xlabel(r'$\ln N$'); ax_U4.set_ylabel(r'$\ln U_4$')
    ax_U4.set_title(r'(a) $\ln U_4$ vs $\ln N$ at various $K$')
    ax_U4.grid(True, alpha=0.10, linewidth=0.5)

    # (b) ln⟨m⟩ vs ln N
    sl_m, ks_m = [], []
    for j, K_val in enumerate(K_sel):
        vals = get_vals_at_K(
            stat_dir, sigma_str, N_values, K_val, 'm', filter_raw=filter_raw
        )
        valid = np.isfinite(vals) & (vals > 0.001)
        if valid.sum() < 3: continue
        ln_vals = np.log(vals[valid])
        c = colors[j]
        ax_m.scatter(ln_N[valid], ln_vals, c=[c], s=28, zorder=5,
                     edgecolors='0.2', linewidths=0.4)
        p, V = np.polyfit(ln_N[valid], ln_vals, 1, cov=True)
        xx = np.linspace(ln_N[valid].min()-0.05, ln_N[valid].max()+0.05, 50)
        ax_m.plot(xx, p[0]*xx+p[1], '-', color=c, lw=1.2, alpha=0.8)
        sl_m.append(p[0]); ks_m.append(K_val)

    ax_m.set_xlabel(r'$\ln N$'); ax_m.set_ylabel(r'$\ln\langle m \rangle$')
    ax_m.set_title(r'(b) $\ln\langle m \rangle$ vs $\ln N$ at various $K$')
    ax_m.grid(True, alpha=0.10, linewidth=0.5)

    # (c) U4 slope vs K — 方案A误差棒 + Kc虚线
    sl_U4, ks_U4 = np.array(sl_U4), np.array(ks_U4)
    idx = np.argsort(ks_U4)
    for j in idx:
        vals = get_vals_at_K(
            stat_dir, sigma_str, N_values, ks_U4[j], 'U4', filter_raw=filter_raw
        )
        valid = np.isfinite(vals)
        if valid.sum() < 3: continue
        p, V = np.polyfit(ln_N[valid], np.log(vals[valid]), 1, cov=True)
        ax_U4s.errorbar(ks_U4[j], sl_U4[j], yerr=np.sqrt(V[0,0]),
                       fmt='o', ms=5, zorder=5,
                       color=colors[list(K_sel).index(ks_U4[j])],
                       ecolor='0.5', elinewidth=1.0, capsize=4, capthick=0.8,
                       markeredgecolor='k', markeredgewidth=0.5)
    ax_U4s.axhline(0, color='0.3', ls='-', lw=0.6)
    ax_U4s.axvline(Kc_user, color='0.4', ls='--', lw=1.2, alpha=0.6)
    ax_U4s.set_xlabel(r'$K$'); ax_U4s.set_ylabel(r'Slope of $\ln U_4$ vs $\ln N$')
    ax_U4s.set_title(r'(c) Slope of $\ln U_4$ vs $\ln N$ [from (a)]')
    ax_U4s.grid(True, alpha=0.10, linewidth=0.5)

    # (d) ⟨m⟩ slope vs K — 方案A误差棒 + Kc虚线
    sl_m, ks_m = np.array(sl_m), np.array(ks_m)
    idx = np.argsort(ks_m)
    for j in idx:
        vals = get_vals_at_K(
            stat_dir, sigma_str, N_values, ks_m[j], 'm', filter_raw=filter_raw
        )
        valid = np.isfinite(vals) & (vals > 0.001)
        if valid.sum() < 3: continue
        p, V = np.polyfit(ln_N[valid], np.log(vals[valid]), 1, cov=True)
        ax_ms.errorbar(ks_m[j], sl_m[j], yerr=np.sqrt(V[0,0]),
                      fmt='o', ms=5, zorder=5,
                      color=colors[list(K_sel).index(ks_m[j])],
                      ecolor='0.5', elinewidth=1.0, capsize=4, capthick=0.8,
                      markeredgecolor='k', markeredgewidth=0.5)
    ax_ms.axvline(Kc_user, color='0.4', ls='--', lw=1.2, alpha=0.6)
    ax_ms.set_xlabel(r'$K$'); ax_ms.set_ylabel(r'Slope of $\ln\langle m \rangle$ vs $\ln N$')
    ax_ms.set_title(r'(d) Slope of $\ln\langle m \rangle$ vs $\ln N$ [from (b)]')
    ax_ms.grid(True, alpha=0.10, linewidth=0.5)

    # Colorbar：稀疏均匀刻度(0.5间隔)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.025, 0.70, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$K$', fontsize=CBAR_LABEL_FONTSIZE, labelpad=4)
    k_lo = np.ceil(K_sel.min() * 2) / 2
    k_hi = np.floor(K_sel.max() * 2) / 2
    ticks = list(np.arange(k_lo, k_hi + 0.01, 0.5))
    cbar.set_ticks(ticks); cbar.set_ticklabels(ticks)
    cbar.ax.tick_params(
        labelsize=CBAR_TICK_FONTSIZE, direction='in', length=4, pad=3
    )
    cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor('0.5')

    os.makedirs(out_dir, exist_ok=True)
    outpath = f'{out_dir}/figure_B_sigma_{sigma_str}.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ {dim_label} σ={sigma_str}, Kc={Kc_user}, {nK} K values → {outpath}')

def main():
    parser = argparse.ArgumentParser(description='图B批量绘制')
    parser.add_argument('--dim', required=True, choices=['1d', '2d'])
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument(
        '--stat-dir',
        type=str,
        default=None,
        help='覆盖 statistics 根目录（1D/2D 均可用）',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='覆盖 figure_B 输出根目录（1D/2D 均可用）',
    )
    parser.add_argument(
        '--1d-only-k-with-raw',
        action='store_true',
        dest='only_k_with_raw',
        help='仅 1D：只使用 raw 下存在 K_xx.xx 目录的数据点（配合 statistics_merged_tail5_6）',
    )
    parser.add_argument(
        '--raw-base',
        type=str,
        default=None,
        help='raw 根目录；默认 1d_lr3/D_2/raw（仅在与 --1d-only-k-with-raw 联用时生效）',
    )
    args = parser.parse_args()

    if args.dim == '1d':
        kc_dict, N_vals, stat_dir, out_dir, sig_fmt = KC_1D, N_1D, STAT_1D, OUT_1D, '.3f'
        dim_label = '1D LR3'
        if args.stat_dir is not None:
            stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir))
        if args.out_dir is not None:
            out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        filter_raw = None
        if args.only_k_with_raw:
            rb = os.path.abspath(os.path.expanduser(args.raw_base or RAW_1D_BASE))
            filter_raw = (rb, None, sig_fmt)

    else:
        kc_dict, N_vals, stat_dir, out_dir, sig_fmt = KC_2D, N_2D, STAT_2D, OUT_2D, '.2f'
        dim_label = '2D LRDG'
        if args.stat_dir is not None:
            stat_dir = os.path.abspath(os.path.expanduser(args.stat_dir))
        if args.out_dir is not None:
            out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        filter_raw = None
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
        fr = filter_raw
        if fr is not None:
            rb, _, sf = fr
            fr = (rb, sigma, sf)
        plot_figureB(
            sigma,
            stat_dir,
            out_dir,
            N_vals,
            kc_dict,
            sig_fmt,
            dim_label,
            spectral_ds(dim_1d, sigma),
            filter_raw=fr if dim_1d else None,
        )

if __name__ == '__main__':
    main()
