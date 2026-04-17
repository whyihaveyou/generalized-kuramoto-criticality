"""
config.py — 统一路径配置
所有画图脚本通过 import 此文件获取数据/输出路径。

用法:
    from config import STAT_1D, STAT_2D, OUT_1D_A, OUT_2D_A, ...
    from config import PROJECT_ROOT  # 项目根目录

路径约定:
    PROJECT_ROOT/
    ├── 00_README.md
    ├── 01_simulation/
    ├── 02_data/
    │   ├── 1d_lr3/statistics/     ← STAT_1D
    │   └── 2d_lr3/statistics/     ← STAT_2D
    ├── 03_scripts/
    └── 04_figures/
        ├── 1d_lr3/figure_{A,B,C,D,E}/  ← OUT_1D_{A,B,C,D,E}
        └── 2d_lrdg/figure_{A,B,C,D,E}/ ← OUT_2D_{A,B,C,D,E}
"""

import os

# ====== 项目根目录（自动检测）======
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

# ====== 数据目录 ======
DATA_DIR = os.path.join(PROJECT_ROOT, "02_data")
STAT_1D = os.path.join(DATA_DIR, "1d_lr3", "statistics")
STAT_2D = os.path.join(DATA_DIR, "2d_lrdg", "statistics")

# ====== 图片输出目录 ======
FIG_DIR = os.path.join(PROJECT_ROOT, "04_figures")

OUT_1D_A = os.path.join(FIG_DIR, "1d_lr3", "figure_A")
OUT_1D_B = os.path.join(FIG_DIR, "1d_lr3", "figure_B")
OUT_1D_C = os.path.join(FIG_DIR, "1d_lr3", "figure_C")
OUT_1D_D = os.path.join(FIG_DIR, "1d_lr3", "figure_D")
OUT_1D_E = os.path.join(FIG_DIR, "1d_lr3", "figure_E")

OUT_2D_A = os.path.join(FIG_DIR, "2d_lrdg", "figure_A")
OUT_2D_B = os.path.join(FIG_DIR, "2d_lrdg", "figure_B")
OUT_2D_C = os.path.join(FIG_DIR, "2d_lrdg", "figure_C")
OUT_2D_D = os.path.join(FIG_DIR, "2d_lrdg", "figure_D")
OUT_2D_E = os.path.join(FIG_DIR, "2d_lrdg", "figure_E")

# 1D 原始数据目录（用于过滤，可选）
RAW_1D_BASE = os.path.join(DATA_DIR, "1d_lr3", "raw")

# ====== 物理常量 ======

# --- 临界耦合 Kc ---
KC_1D = {
    0.200: 2.0, 0.250: 2.1, 0.300: 2.3, 0.350: 2.6, 0.380: 2.8,
    0.400: 3.0, 0.405: 3.1, 0.410: 3.2, 0.415: 3.3, 0.420: 3.4,
    0.425: 3.5, 0.430: 3.7, 0.435: 3.9, 0.440: 4.2, 0.445: 4.6,
}
KC_2D = {
    0.40: 1.83, 0.50: 1.89, 0.60: 1.97, 0.70: 2.07, 0.75: 2.15,
    0.78: 2.20, 0.80: 2.24, 0.83: 2.32, 0.85: 2.38, 0.90: 2.52,
    0.95: 2.75,
}

# --- 系统尺寸 ---
N_1D = [128, 256, 512, 1024, 2048, 4096]
N_2D = [256, 400, 625, 900, 1296, 2401, 4096]

# --- Figure D 用：每个 (sigma, Kc, beta/nu) 三元组 ---
DATA_1D = [
    (0.200, 2.04, 0.50),
    (0.250, 2.16, 0.50),
    (0.300, 2.32, 0.50),
    (0.350, 2.62, 0.68),
    (0.380, 2.86, 0.77),
    (0.400, 3.06, 0.87),
    (0.405, 3.20, 0.91),
    (0.410, 3.30, 0.98),
    (0.415, 3.40, 1.02),
    (0.420, 3.50, 1.06),
    (0.425, 3.70, 1.16),
    (0.430, 3.90, 1.46),
    (0.435, 4.10, 1.70),
    (0.440, 4.50, 1.95),
    (0.445, 4.90, 2.45),
]
DATA_2D = [
    (0.40, 1.85, 0.49),
    (0.50, 1.91, 0.50),
    (0.60, 1.99, 0.50),
    (0.70, 2.08, 0.50),
    (0.75, 2.17, 0.52),
    (0.78, 2.19, 0.56),
    (0.80, 2.22, 0.60),
    (0.83, 2.34, 0.78),
    (0.85, 2.40, 0.87),
    (0.90, 2.60, 1.23),
    (0.95, 2.90, 1.80),
]

# --- 子图定义（Figure A 用的三面板） ---
PANELS = [
    (r'$\langle m \rangle$ vs $K$', r'$\langle m \rangle$',
     'order_parameter_means', 'order_parameter_stds'),
    (r'$\chi$ vs $K$', r'$\chi$',
     'magnetic_susceptibilities', 'magnetic_susceptibilities_stds'),
    (r'$U_4$ vs $K$', r'$U_4$',
     'binder_cumulants', 'binder_cumulants_stds'),
]


def spectral_ds(dim_1d: bool, sigma: float) -> float:
    """谱维数 d_s：1D LR³ 为 2/σ，2D LRDG 为 4/σ。"""
    return (2.0 if dim_1d else 4.0) / float(sigma)
