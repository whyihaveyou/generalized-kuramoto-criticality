# LRDG Kuramoto 批量计算 — 参数规格书

> ⚠️ 本文件是权威参数参考，所有后续操作（分析、可视化、重命名）必须以此为准。

## 公共参数
- D = 2（状态维度）
- DELTA = 1.0（频率分布标准差）
- T = 200.0, DT = 0.01, STEPS = 20000
- N = [256, 400, 625, 900, 1296, 2401, 4096]

## 各 Sigma 配置

### Sigma = 0.40
- K coarse: 1.0 → 2.5, step=0.1, **runs=500**
- Kc = 1.83, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.50
- K coarse: 1.0 → 2.5, step=0.1, **runs=500**
- Kc = 1.89, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.60
- K coarse: 1.3 → 2.8, step=0.1, **runs=500**
- Kc = 1.97, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.70
- K coarse: 1.5 → 3.0, step=0.1, **runs=500**
- Kc = 2.07, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.80
- K coarse: 1.5 → 3.0, step=0.1, **runs=500**
- Kc = 2.24, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.83
- K coarse: 1.5 → 3.0, step=0.1, **runs=500**
- Kc = 2.32, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.85
- K coarse: 1.6 → 3.1, step=0.1, **runs=500**
- Kc = 2.38, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.90
- K coarse: 1.7 → 3.2, step=0.1, **runs=500**
- Kc = 2.52, fine: Kc±0.03, step=0.01, **runs=1000**

### Sigma = 0.95
- K coarse: 1.7 → 3.2, step=0.1, **runs=500**
- Kc = 2.75, fine: Kc±0.03, step=0.01, **runs=1000**

## 已知问题
- ⚠️ 目录命名多了前导零（如 LRDG_sigma_0040 而非 LRDG_sigma_040），待任务完成后批量修正
- ✅ Rust 程序参数正确（coarse=500, fine=1000），数据无误
