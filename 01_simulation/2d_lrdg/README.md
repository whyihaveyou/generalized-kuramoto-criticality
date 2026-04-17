# LRDG Kuramoto Batch Simulation - 9 Sigma Values

## Overview
2D LRDG网络Kuramoto动力学大规模并行计算，覆盖9个sigma值。

## Sigma Configurations
| Sigma | K coarse range | Kc (fine scan) |
|-------|---------------|----------------|
| 0.40  | 1.0 - 2.5     | 1.83 ± 0.03   |
| 0.50  | 1.0 - 2.5     | 1.89 ± 0.03   |
| 0.60  | 1.3 - 2.8     | 1.97 ± 0.03   |
| 0.70  | 1.5 - 3.0     | 2.07 ± 0.03   |
| 0.80  | 1.5 - 3.0     | 2.24 ± 0.03   |
| 0.83  | 1.5 - 3.0     | 2.32 ± 0.03   |
| 0.85  | 1.6 - 3.1     | 2.38 ± 0.03   |
| 0.90  | 1.7 - 3.2     | 2.52 ± 0.03   |
| 0.95  | 1.7 - 3.2     | 2.75 ± 0.03   |

## Parameters (same as sigma=0.75)
- D=2, DELTA=1.0, T=200, DT=0.01
- N = [256, 400, 625, 900, 1296, 2401, 4096]
- Coarse: step=0.1, 500 runs
- Fine: step=0.01, 1000 runs

## Directory Structure
```
LRDG_sigma_0XX/D_2_LRDG_sigma0XX/
  sigma_0.XXX/N_YYY/K_Z.ZZ/kuramoto_2d_lrdg_*.npy
  simulation_info.txt
```

## Checking Progress
```bash
# Overall
for s in 040 050 060 070 080 083 085 090 095; do
  dir="/share/home/qiuzhongpu/data_and_program/kuramoto/7_spectral_dimension/LRDG_sigma_0${s}/D_2_LRDG_sigma0${s}"
  files=$(find "$dir" -name '*.npy' 2>/dev/null | wc -l)
  size=$(du -sh "$dir" 2>/dev/null | cut -f1)
  echo "sigma=${s}: ${files} files, ${size}"
done

# Job status
squeue -u qiuzhongpu

# Live log
tail -f slurm_lrdg_batch_*.out
```
