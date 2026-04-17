# Generalized Kuramoto Model: Criticality and Universality

> We thank **Prof. Hugues Chaté**, **Prof. Jürgen Kurths**, **Prof. Youjin Deng**, **Prof. Xiaqing Shi**, **Prof. Yueheng Lan**, **Dr. Sarthak Chandra**, and **Dr. Maoxin Liu** for valuable discussions and insights.

**Data repository for:** Z. Qiu, T. Wu, S. Fang, J. Meng, and J. Fan, "Criticality and universality of generalized Kuramoto model," (2026).

This repository contains the simulation code, numerical data, plotting scripts, and publication figures that support the main claims of the manuscript — particularly the determination of the **upper critical dimension $d_u = 5$** via spectral-dimension tests on complex networks with tuneable spectral dimension.

---

## Repository Structure

```
generalized-kuramoto-criticality/
├── 00_README.md                  ← You are here
├── LICENSE                       MIT License
├── 01_simulation/                Simulation source code (Rust)
│   ├── 2d_lrdg/                  2D LRDG simulator
│   │   ├── src/main.rs           Single-file source
│   │   ├── Cargo.toml / .lock    Rust project configuration
│   │   ├── README.md             Build & usage instructions
│   │   └── PARAMS.md             CLI parameter documentation
│   └── 1d_lr3/                   1D long-range network simulator
│       ├── src/main.rs           Single-file source
│       └── Cargo.toml / .lock    Rust project configuration
├── 02_data/                      Statistical data (NumPy .npy)
│   ├── 2d_lrdg/statistics/       539 files, 11 σ × 7 N
│   │   └── sigma_0.XX/N_YYY/     Thermodynamic observables (order param, χ, U₄)
│   └── 1d_lr3/statistics/        630 files, 15 σ × 6 N
│       └── sigma_0.XXX/N_YYY/
├── 03_scripts/                   Plotting scripts (Python)
│   ├── config.py                 Unified path & physical constants (auto-detects project root)
│   ├── plot_figure_A.py          Thermodynamic observables (⟨ρ⟩, χ, U₄) vs coupling K
│   ├── plot_figure_B.py          Data collapse
│   ├── plot_figure_C.py          Refined ν-sweep data collapse
│   ├── plot_figure_D.py          Critical exponents vs spectral dimension dₛ (with CSV export)
│   └── plot_figure_E.py          Susceptibility peak width ΔK vs system size N (Griffiths criterion)
└── 04_figures/                   Publication figures (PNG + CSV)
    ├── 2d_lrdg/                  35 PNG + 1 CSV
    │   ├── figure_A/             11 panels (σ = 0.40 ~ 0.95)
    │   ├── figure_B/             11 panels
    │   ├── figure_C/             11 panels
    │   ├── figure_D/             1 panel + critical_exponents_vs_ds.csv
    │   └── figure_E/             1 panel
    └── 1d_lr3/                   47 PNG + 1 CSV
        ├── figure_A/             15 panels (σ = 0.200 ~ 0.445)
        ├── figure_B/             15 panels
        ├── figure_C/             15 panels
        ├── figure_D/             1 panel + critical_exponents_vs_ds.csv
        └── figure_E/             1 panel
```

---

## Models

Both models follow the framework of Millán et al. [2]: the bond probability between two nodes is proportional to a power law of their distance, $P(r) \sim r^{-\sigma}$, which allows continuous tuning of the spectral dimension $d_s$.

### 2D LRDG (Long-Range Diluted Graph)

Based on Bighin, Enss, and Defenu [1]. Links are randomly placed on a 2D square lattice with $P(r) \sim r^{-\sigma}$, giving $d_s = 4/\sigma$.

| Parameter | Values |
|-----------|--------|
| σ (link decay exponent) | 0.40, 0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.90, 0.95 (11 values) |
| dₛ (spectral dimension) | 10.0, 8.0, 6.67, 5.71, 5.33, 5.13, **5.0**, 4.82, 4.71, 4.44, 4.21 |
| N (system size) | 256, 400, 625, 900, 1296, 2401, 4096 (7 values) |
| Oscillator dimension D | 2 (classical Kuramoto) |

**σ = 0.80 corresponds to $d_s = 5.0$**, the theoretically expected upper critical dimension.

### 1D long-range network

Based on Millán, Gori, Battiston, Enss, and Defenu [2]. Links are placed on a 1D chain with $P(r) \sim r^{-\sigma}$, giving $d_s = 2/\sigma$.

| Parameter | Values |
|-----------|--------|
| σ | 0.200, 0.250, 0.300, 0.350, 0.380, 0.400, 0.405, 0.410, 0.415, 0.420, 0.425, 0.430, 0.435, 0.440, 0.445 (15 values) |
| dₛ | 10.0, 8.0, 6.67, 5.26, 5.26, **5.0**, 4.94, 4.88, 4.82, 4.76, 4.71, 4.65, 4.60, 4.55, 4.49 |
| N | 128, 256, 512, 1024, 2048, 4096 (6 values) |
| Oscillator dimension D | 2 |

**σ = 0.400 corresponds to $d_s = 5.0$**. The 1D model exhibits stronger disorder effects (Griffiths phase), leading to larger data scatter compared to the 2D LRDG [1].

---

## Data Format

Each $(σ, N)$ pair produces a directory under `02_data/{1d_lr3,2d_lrdg}/statistics/` containing 7 NumPy files:

| File | Content | Shape |
|------|---------|-------|
| `K_values.npy` | Coupling strength array | `(n_K,)` |
| `order_parameter_means.npy` | Order parameter $\langle \rho \rangle(K)$ | `(n_K,)` |
| `order_parameter_stds.npy` | Standard deviation of order parameter | `(n_K,)` |
| `magnetic_susceptibilities.npy` | Magnetic susceptibility $\chi(K) = N \cdot \mathrm{Var}(\rho)$ | `(n_K,)` |
| `magnetic_susceptibilities_stds.npy` | Standard deviation of susceptibility | `(n_K,)` |
| `binder_cumulants.npy` | Binder cumulant $U_4(K)$ | `(n_K,)` |
| `binder_cumulants_stds.npy` | Standard deviation of Binder cumulant | `(n_K,)` |

**Quick load in Python:**
```python
import numpy as np, os

sigma, N = 0.83, 2401
d = f"02_data/2d_lrdg/statistics/sigma_{sigma:.2f}/N_{N}"
K = np.load(os.path.join(d, "K_values.npy"))
rho = np.load(os.path.join(d, "order_parameter_means.npy"))
chi = np.load(os.path.join(d, "magnetic_susceptibilities.npy"))
```

The `04_figures/` directory also contains CSV files with extracted critical exponents (`critical_exponents_vs_ds.csv`), including columns for $\beta$, $\nu$, $\gamma$, $\beta/\nu$, and $\gamma/\nu$.

---

## Building & Running the Simulation Code

The simulators are written in **Rust** and require `cargo`. To build:

```bash
cd 01_simulation/2d_lrdg      # or 01_simulation/1d_lr3
cargo build --release
```

CLI parameters (σ, K range, N, ensemble count, etc.) are documented in `PARAMS.md` (2D) and in source code comments (1D). Example:

```bash
# 2D LRDG: simulate σ=0.83, N=2401
./target/release/main --sigma 0.83 --N 2401 --ensembles 2000

# 1D long-range: simulate σ=0.415, N=4096
./target/release/main --sigma 0.415 --N 4096 --ensembles 5000
```

---

## Running the Plotting Scripts

All scripts are in `03_scripts/` and share a unified `config.py` for paths and physical constants. **No path configuration is needed** — `config.py` auto-detects the project root directory.

**Dependencies:** `numpy`, `scipy`, `matplotlib`

```bash
cd 03_scripts/

# Figure A: thermodynamic observables (3 panels × 11 or 15 σ values)
python plot_figure_A.py --dim 2d       # 2D LRDG
python plot_figure_A.py --dim 1d       # 1D long-range

# Figure B: data collapse
python plot_figure_B.py --dim 2d

# Figure C: refined ν-sweep data collapse
python plot_figure_C.py --dim 2d

# Figure D: critical exponents vs dₛ (produces PNG + CSV)
python plot_figure_D.py --dim 2d
python plot_figure_D.py --dim 1d

# Figure E: susceptibility peak width ΔK vs N (Griffiths criterion)
python plot_figure_E.py --dim 2d
python plot_figure_E.py --dim 1d
```

Output figures are saved to the corresponding subdirectory under `04_figures/`.

---

## Citation

If you use this data or code, please cite:

> Z. Qiu, T. Wu, S. Fang, J. Meng, and J. Fan, "Criticality and universality of generalized Kuramoto model," (2026). DOI: [to be added upon publication]

---

## References

[1] G. Bighin, T. Enss, and N. Defenu, "Universal Scaling in Real Dimension," *Nature Communications* **15**, 4207 (2024). DOI: [10.1038/s41467-024-48537-1](https://doi.org/10.1038/s41467-024-48537-1)

[2] A. P. Millán, G. Gori, F. Battiston, T. Enss, and N. Defenu, "Complex Networks with Tuneable Spectral Dimension as a Universality Playground," *Physical Review Research* **3**, 023015 (2021). DOI: [10.1103/PhysRevResearch.3.023015](https://doi.org/10.1103/PhysRevResearch.3.023015)

---

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.
