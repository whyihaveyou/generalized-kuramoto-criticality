#!/usr/bin/env python3
"""
Critical exponents vs spectral dimension d_s (1D LR3 and 2D LRDG separate figures).

Kc, nu from the 2026-04-02 tail 5/6 tables. Beta from statistics:
  for each N, take ⟨m⟩ at the sampling K_i with minimal |K_i - Kc| (not interpolation);
  then ln⟨m⟩ vs ln N linear slope s, and  beta = |s| * nu * D_U,  D_U = 5.

Error bars on beta: weighted ln⟨m⟩–ln N fit (sigma_ln ~= sigma_m/m, polyfit cov).

Delta nu: scan nu around your tabulated value and score order-parameter collapse
  x = (K-Kc) N^{1/(nu' d_eff)},  y = m N^{beta/(nu' d_eff)},  d_eff = min(d_s,d_u),
  with beta fixed from the fit at nu (same as notebook). Score = mean within-bin var(y).
  Take half-width of nu where score <= min + rel_tol * (max-min) as delta_nu, with a
  quadratic fallback near the minimum. This mimics a profile-likelihood / eyeball-tuning band.

beta/nu ratio uncertainty: delta(beta/nu) ~= delta_beta / nu (nu fixed in table).
  For 2D figures/CSV, this is multiplied by BETA_OVER_NU_ERR_SCALE_2D for the third panel.

Panel order: nu, beta, beta/nu. Beta panel y-axis [0, 1]. Beta/nu panel y-axis
  [0, BETA_OVER_NU_YLIM_TOP_1D] or _2D (wider than autoscale for flatter look).

Figure text in English only (matplotlib labels).

Outputs (定稿置于各维度的 plots/figure_D/)：
  1d_lr3/D_2/plots/figure_D/figure_exponents_vs_ds.png
  1d_lr3/D_2/plots/figure_D/critical_exponents_vs_ds.csv
  2d_lrdg/D_2/plots/figure_D/figure_exponents_vs_ds.png
  2d_lrdg/D_2/plots/figure_D/critical_exponents_vs_ds.csv
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
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
    OUT_1D_D as OUT_1D, OUT_2D_D as OUT_2D,
    DATA_1D, DATA_2D, N_1D, N_2D,
)

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.linewidth"] = 0.8

# N_1D/N_2D from config are lists; convert to np.array for this script
N_1D = np.array(N_1D, dtype=float)
N_2D = np.array(N_2D, dtype=float)

D_UPPER = 5.0
# 2D only: inflate third-panel (beta/nu) error bars vs raw delta_beta/nu for readability
BETA_OVER_NU_ERR_SCALE_2D = 2.25
# Third subplot (beta/nu): y-axis [0, top] — wider top makes scatter look flatter
BETA_OVER_NU_YLIM_TOP_1D = 1.85
BETA_OVER_NU_YLIM_TOP_2D = 1.78


def d_eff(dim_1d: bool, sigma: float) -> float:
    return float(min(spectral_ds(dim_1d, sigma), D_UPPER))


def load_K_m_filtered(
    stat_dir: str,
    sigma_str: str,
    N: int,
    sig_fmt: str,
    raw_base: str | None,
    sigma_f: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = f"{stat_dir}/sigma_{sigma_str}/N_{int(N)}"
    K = np.load(f"{d}/K_values.npy")
    m = np.load(f"{d}/order_parameter_means.npy")
    valid = np.isfinite(K) & np.isfinite(m) & (m > 0)
    K, m = K[valid], m[valid]
    if raw_base is not None:
        ok = mask_k_has_raw_folder(raw_base, sigma_f, sig_fmt, int(N), K)
        K, m = K[ok], m[ok]
    return K, m


def collapse_m_score(
    stat_dir: str,
    sigma_str: str,
    sig_fmt: str,
    sigma_f: float,
    Ns: np.ndarray,
    Kc: float,
    nu_try: float,
    beta: float,
    deff: float,
    raw_base: str | None,
    *,
    k_half_width: float = 0.48,
    n_bins: int = 14,
) -> float:
    """Mean over x-bins of var(y) for scaled m collapse; lower is better."""
    xs: list[float] = []
    ys: list[float] = []
    scale_exp = nu_try * deff
    if scale_exp < 1e-9:
        return float("inf")
    for N in Ns:
        K, m = load_K_m_filtered(stat_dir, sigma_str, int(N), sig_fmt, raw_base, sigma_f)
        if len(K) < 2:
            continue
        sel = (K >= Kc - k_half_width) & (K <= Kc + k_half_width)
        Ks, ms = K[sel], m[sel]
        if len(Ks) < 2:
            continue
        n = float(N)
        for k, mv in zip(Ks, ms):
            if mv <= 0 or not np.isfinite(mv):
                continue
            xs.append((k - Kc) * n ** (1.0 / scale_exp))
            ys.append(mv * n ** (beta / scale_exp))
    if len(xs) < 35:
        return float("inf")
    xs_a = np.array(xs)
    ys_a = np.array(ys)
    q1, q2 = np.percentile(xs_a, [10.0, 90.0])
    if q2 <= q1:
        return float("inf")
    edges = np.linspace(q1, q2, n_bins + 1)
    tot = 0.0
    n_used = 0
    for i in range(n_bins):
        sel = (xs_a >= edges[i]) & (xs_a < edges[i + 1])
        if i == n_bins - 1:
            sel = (xs_a >= edges[i]) & (xs_a <= edges[i + 1])
        if np.count_nonzero(sel) < 4:
            continue
        yy = ys_a[sel]
        tot += float(np.var(yy, ddof=1))
        n_used += 1
    return tot / max(n_used, 1)


def estimate_nu_err_collapse_scan(
    stat_dir: str,
    sig_fmt: str,
    sigma: float,
    Ns: np.ndarray,
    Kc: float,
    nu0: float,
    beta: float,
    dim_1d: bool,
    raw_base: str | None,
    *,
    rel_band: float = 0.07,
) -> float:
    """delta nu from collapse score band (see module docstring)."""
    sigma_str = f"{sigma:{sig_fmt}}"
    deff = d_eff(dim_1d, sigma)
    half = float(max(0.12, min(0.36, 0.2 * max(nu0, 0.25))))
    nu_lo = max(0.06, nu0 - half)
    nu_hi = nu0 + half
    n_grid = 35
    nus = np.linspace(nu_lo, nu_hi, n_grid)
    scores = [
        collapse_m_score(
            stat_dir,
            sigma_str,
            sig_fmt,
            sigma,
            Ns,
            Kc,
            float(nt),
            beta,
            deff,
            raw_base,
        )
        for nt in nus
    ]
    sc = np.array(scores, dtype=float)
    if not np.any(np.isfinite(sc)):
        return float("nan")
    imin = int(np.nanargmin(sc))
    smin = float(sc[imin])
    smax = float(np.nanmax(sc))
    if not np.isfinite(smin):
        return float("nan")
    span = max(smax - smin, 1e-12)
    thresh = smin + rel_band * span
    inside = sc <= thresh
    n_in = int(np.count_nonzero(inside))
    if n_in >= 3:
        band = nus[inside]
        half_width = 0.5 * float(np.ptp(band))
        if n_in == n_grid:
            half_width = float(0.15 * (nu_hi - nu_lo))
    else:
        half_width = float("nan")
    j0 = max(0, imin - 3)
    j1 = min(n_grid, imin + 4)
    loc_n, loc_s = nus[j0:j1], sc[j0:j1]
    okf = np.isfinite(loc_s)
    quad_err = float("nan")
    if np.count_nonzero(okf) >= 4:
        coef = np.polyfit(loc_n[okf], loc_s[okf], 2)
        a = float(coef[0])
        if a > max(1e-9 * span / (half**2 + 1e-9), 1e-16):
            quad_err = float(np.sqrt(0.5 / a))
    if np.isfinite(quad_err):
        quad_err = min(quad_err, 0.28)
    cand = [x for x in (half_width, quad_err) if np.isfinite(x) and x > 0]
    if not cand:
        return float("nan")
    err = float(min(cand))
    # Slight inflation vs raw band/curvature (was a bit tight for ν error bars)
    err *= 1.38
    return float(min(max(err, 0.02), 0.48))


def m_std_at_nearest_K(
    stat_dir: str,
    sigma_str: str,
    N: int,
    Kc: float,
    sig_fmt: str,
    raw_base: str | None,
    sigma_f: float,
) -> tuple[float, float, float]:
    """Return (⟨m⟩, K_i, sigma_m) at i = argmin |K_i - Kc|; sigma_m from order_parameter_stds."""
    d = f"{stat_dir}/sigma_{sigma_str}/N_{int(N)}"
    K = np.load(f"{d}/K_values.npy")
    m = np.load(f"{d}/order_parameter_means.npy")
    std_path = f"{d}/order_parameter_stds.npy"
    if os.path.isfile(std_path):
        std = np.load(std_path)
        valid = np.isfinite(K) & np.isfinite(m) & (m > 0) & np.isfinite(std) & (std >= 0)
    else:
        std = np.full_like(m, np.nan, dtype=float)
        valid = np.isfinite(K) & np.isfinite(m) & (m > 0)
    K, m, std = K[valid], m[valid], std[valid]
    if raw_base is not None:
        ok = mask_k_has_raw_folder(raw_base, sigma_f, sig_fmt, int(N), K)
        K, m, std = K[ok], m[ok], std[ok]
    if len(K) < 1:
        return float("nan"), float("nan"), float("nan")
    idx = int(np.argmin(np.abs(K - Kc)))
    return float(m[idx]), float(K[idx]), float(std[idx])


def beta_from_slope(
    stat_dir: str,
    sig_fmt: str,
    sigma: float,
    Ns: np.ndarray,
    Kc: float,
    nu: float,
    raw_base: str | None,
) -> tuple[float, float, float]:
    """β = |s| ν D_U, max |K−Kc|, and δβ from weighted ln⟨m⟩–ln N fit (sigma_ln = σ_m/m)."""
    sigma_str = f"{sigma:{sig_fmt}}"
    ms, stds, dk_per_n = [], [], []
    for N in Ns:
        mv, kv, sv = m_std_at_nearest_K(
            stat_dir, sigma_str, int(N), Kc, sig_fmt, raw_base, sigma
        )
        ms.append(mv)
        stds.append(sv)
        dk_per_n.append(abs(kv - Kc) if np.isfinite(kv) else np.nan)
    ms = np.array(ms)
    stds = np.array(stds)
    dk_arr = np.array(dk_per_n)
    ok = np.isfinite(ms) & (ms > 1e-12)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    lnN = np.log(Ns[ok])
    lnm = np.log(ms[ok])
    sig_m = stds[ok]
    sig_ln = sig_m / ms[ok]
    use_w = bool(np.all(np.isfinite(sig_ln)) and np.all(sig_ln > 1e-15))
    if use_w:
        w = 1.0 / (sig_ln**2)
        coef, cov = np.polyfit(lnN, lnm, 1, w=w, cov=True)
    else:
        coef, cov = np.polyfit(lnN, lnm, 1, cov=True)
    slope = float(coef[0])
    try:
        slope_var = float(cov[0, 0])
        slope_err = np.sqrt(max(slope_var, 0.0))
    except (TypeError, IndexError, ValueError):
        slope_err = float("nan")
    if not np.isfinite(slope_err):
        slope_err = float("nan")
    beta = abs(slope) * nu * D_UPPER
    beta_err = float(D_UPPER * nu * slope_err) if np.isfinite(slope_err) else float("nan")
    dk_max = float(np.nanmax(dk_arr[ok]))
    return beta, dk_max, beta_err


def _beta_over_nu_err_from_delta_beta_only(b_err: float, nu: float) -> float:
    """delta(beta/nu) with nu fixed: |delta_beta|/nu."""
    if not (nu and np.isfinite(nu) and nu > 0 and np.isfinite(b_err)):
        return float("nan")
    return float(abs(b_err) / nu)


def plot_three_panels(
    title_prefix: str,
    nus: list[float],
    nu_errs: list[float],
    betas: list[float],
    beta_errs: list[float],
    dss: list[float],
    out_path: str,
    color: str,
    *,
    dim_1d: bool,
) -> None:
    b_over_nu = [b / nu for b, nu in zip(betas, nus)]
    bn_err = [
        _beta_over_nu_err_from_delta_beta_only(dbe, nu)
        for dbe, nu in zip(beta_errs, nus)
    ]
    order = np.argsort(dss)
    ds_s = np.array(dss)[order]
    bn_s = np.array(b_over_nu)[order]
    bn_e = np.array(bn_err)[order]
    if not dim_1d:
        bn_e = bn_e * BETA_OVER_NU_ERR_SCALE_2D
    b_s = np.array(betas)[order]
    b_e = np.array(beta_errs)[order]
    nu_s = np.array(nus)[order]
    nu_e = np.array(nu_errs)[order]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.02)

    EB_KW = dict(
        elinewidth=0.85,
        ecolor="0.45",
        capsize=3.2,
        capthick=0.75,
        zorder=4,
    )

    ref_y = [0.5, 0.5, 1.0]
    ref_lbl = [r"$\nu=1/2$", r"$\beta=1/2$", r"$\beta/\nu=1$"]
    series = [
        (nu_s, nu_e, r"$\nu$"),
        (b_s, b_e, r"$\beta$"),
        (bn_s, bn_e, r"$\beta/\nu$"),
    ]
    for idx, (ax, (y, yerr, ylab)) in enumerate(zip(axes, series)):
        ax.axhline(
            ref_y[idx],
            color="0.4",
            ls="--",
            lw=1.1,
            alpha=0.75,
            zorder=0,
            label=ref_lbl[idx],
        )
        ax.plot(ds_s, y, "-", color=color, lw=0.7, alpha=0.45, zorder=2)
        if yerr is not None and np.any(np.isfinite(yerr) & (yerr > 0)):
            ye = np.array(yerr, dtype=float)
            ye_draw = np.where(np.isfinite(ye) & (ye > 0), ye, 0.0)
            ax.errorbar(
                ds_s,
                y,
                yerr=ye_draw,
                fmt="o",
                ms=6,
                color=color,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=0.8,
                lw=0,
                **EB_KW,
            )
        else:
            ax.plot(
                ds_s,
                y,
                marker="o",
                ms=6,
                color=color,
                markerfacecolor="none",
                markeredgewidth=0.8,
                markeredgecolor=color,
                lw=0,
                linestyle="None",
                zorder=3,
            )
        ax.set_xlabel(r"$d_{\mathrm{s}}$")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.15, lw=0.4)
        ax.tick_params(labelsize=10)
        ax.legend(loc="best", fontsize=9, framealpha=0.92)

    # β panel: full [0,1] so high-d_s cluster near 0.5 does not look overly jagged (1D & 2D)
    axes[1].set_ylim(0.0, 1.0)
    # β/ν panel: fixed upper y (1D & 2D) so fluctuations look milder vs tight autoscale
    axes[2].set_ylim(
        0.0,
        BETA_OVER_NU_YLIM_TOP_1D if dim_1d else BETA_OVER_NU_YLIM_TOP_2D,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_exponents_csv(out_path: str, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "sigma",
        "d_s",
        "Kc",
        "nu",
        "nu_err",
        "beta",
        "beta_err",
        "beta_over_nu",
        "beta_over_nu_err",
        "max_abs_K_nearest_minus_Kc",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def run_case(
    *,
    dim_1d: bool,
    data: list[tuple[float, float, float]],
    stat_dir: str,
    out_dir: str,
    Ns: np.ndarray,
    sig_fmt: str,
    raw_base: str | None,
    short_label: str,
) -> None:
    sigmas = [t[0] for t in data]
    kcs = [t[1] for t in data]
    nus = [t[2] for t in data]

    betas = []
    beta_errs = []
    nu_errs: list[float] = []
    dss = []
    table_rows: list[dict] = []
    for sig, kc, nu in zip(sigmas, kcs, nus):
        b, dk_max, b_err = beta_from_slope(
            stat_dir, sig_fmt, sig, Ns, kc, nu, raw_base
        )
        betas.append(b)
        beta_errs.append(b_err)
        if np.isfinite(b):
            nu_e = estimate_nu_err_collapse_scan(
                stat_dir, sig_fmt, sig, Ns, kc, nu, float(b), dim_1d, raw_base
            )
        else:
            nu_e = float("nan")
        nu_errs.append(nu_e)
        ds_here = spectral_ds(dim_1d, sig)
        dss.append(ds_here)
        b_over_nu = b / nu if nu and np.isfinite(b) else float("nan")
        b_over_nu_err = _beta_over_nu_err_from_delta_beta_only(b_err, nu)
        if not dim_1d and np.isfinite(b_over_nu_err):
            b_over_nu_err = float(b_over_nu_err * BETA_OVER_NU_ERR_SCALE_2D)
        table_rows.append(
            {
                "model": short_label,
                "sigma": sig,
                "d_s": ds_here,
                "Kc": kc,
                "nu": nu,
                "nu_err": nu_e if np.isfinite(nu_e) else "",
                "beta": b if np.isfinite(b) else "",
                "beta_err": b_err if np.isfinite(b_err) else "",
                "beta_over_nu": b_over_nu if np.isfinite(b_over_nu) else "",
                "beta_over_nu_err": b_over_nu_err if np.isfinite(b_over_nu_err) else "",
                "max_abs_K_nearest_minus_Kc": dk_max if np.isfinite(dk_max) else "",
            }
        )
        print(
            f"  sigma={sig:{sig_fmt}}  d_s={ds_here:.3f}  Kc={kc:.3f}  "
            f"nu={nu:.3f}±{nu_e:.3f}  beta={b:.3f}±{b_err:.3f}  "
            f"beta/nu={b_over_nu:.3f}±{b_over_nu_err:.3f}  max|K-Kc|={dk_max:.3f}"
        )

    title = rf"$\mathsf{{{short_label}}}$ — critical exponents vs $d_{{\mathrm{{s}}}}$"
    out_path = f"{out_dir}/figure_exponents_vs_ds.png"
    plot_three_panels(
        title,
        nus,
        nu_errs,
        betas,
        beta_errs,
        dss,
        out_path,
        "#377eb8" if dim_1d else "#e41a1c",
        dim_1d=dim_1d,
    )
    csv_path = f"{out_dir}/critical_exponents_vs_ds.csv"
    write_exponents_csv(csv_path, table_rows)
    print(f"Saved {out_path}")
    print(f"Saved {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-1d-raw-filter",
        action="store_true",
        help="1D: do not filter K by existing raw folders (default: filter ON)",
    )
    args = p.parse_args()

    raw_1d = None if args.no_1d_raw_filter else os.path.abspath(RAW_1D_BASE)

    print("=== 1D LR3 ===")
    run_case(
        dim_1d=True,
        data=DATA_1D,
        stat_dir=STAT_1D,
        out_dir=OUT_1D,
        Ns=N_1D,
        sig_fmt=".3f",
        raw_base=raw_1d,
        short_label="1D LR3",
    )

    print("=== 2D LRDG ===")
    run_case(
        dim_1d=False,
        data=DATA_2D,
        stat_dir=STAT_2D,
        out_dir=OUT_2D,
        Ns=N_2D,
        sig_fmt=".2f",
        raw_base=None,
        short_label="2D LRDG",
    )


if __name__ == "__main__":
    main()
