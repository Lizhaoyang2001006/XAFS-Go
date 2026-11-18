# -*- coding: utf-8 -*-
"""
athena.py — 阶段 1~4：由 μ(E) 到 |χ(R)|
================================================================
对应 Athena 的功能范围。函数名与研究流程里的数学步骤一一对应：

    μ(E)          mu_from_*                     （见 xafs_io）
    能量校准       find_e0_derivative_max / find_e0_half_height
    多扫描合并     align_by_reference / merge_scans
    归一化         fit_linear_pre_edge → compute_edge_step → normalize_absorption
    背景扣除       wave_number → extract_chi_k（内部调 AUTOBK）
    傅里叶变换     ft_to_r_space（内部调 xftf，窗口由其统一处理）

依赖：xafs_core（常数与数据结构）、xafs_io（读写）、larch（AUTOBK 算法）。
**不 import artemis / larch_wt**，可单独运行 `python athena.py` 自测。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from larch import Group
from larch.xafs import autobk
from xraydb import (material_mu, material_mu_components, mu_elam, xray_edge,
                    xray_line)

import xafs_io
from xafs_core import (K2, BackFourierTransform, ChiK, FourierTransform,
                       NormalizedScan, RawScan, XAFSConfigError, XAFSDataError)

# ---- facade: athena_merge.py ----
from athena_merge import (
    align_by_reference,
    merge_scans,
    _copy_raw_scan,
    apply_deadtime_correction,
    deglitch_ranges,
    despike_scan,
    merge_mca_channels,
)

# ---- facade: athena_norm.py ----
from athena_norm import (
    find_e0_derivative_max,
    find_e0_half_height,
    classify_reference_channel,
    determine_e0,
    fit_linear_pre_edge,
    compute_edge_step,
    normalize_absorption,
    normalize_mback,
    _renormalize_flat_plateau,
    _flatten_post_edge,
    fit_whiteline_area,
    extract_xanes_features,
    _pseudo_voigt_area,
    fit_pre_edge_peaks,
)

# ---- facade: athena_selfabs.py ----
from athena_selfabs import (
    _booth_mu_terms,
    booth_self_absorption,
    fluo_xanes_self_absorption,
)

# ---- facade: athena_bkg_ft.py ----
from athena_bkg_ft import (
    wave_number,
    extract_chi_k,
    high_k_snr,
    estimate_single_scan_kmax,
    measure_shell_metrics,
    optimize_rbkg,
    back_ft_to_q_space,
    ft_to_r_space,
)

# ================================================================
# 7. 自测
# ================================================================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from xafs_io import read_scan

    target = (Path(__file__).resolve().parents[3] / "data" / "练手数据" /
              "02_Fe_standards" / "Fe_lepidocrocite.000")
    scan = read_scan(target)

    e0_info = determine_e0(scan)
    print(f"E0 (derivative) : {e0_info['e0_sample']:.2f} eV")
    print(f"E0 (half_height): {find_e0_half_height(scan.energy, scan.mu, e0_info['e0_sample']):.2f} eV")

    norm = normalize_absorption(scan, e0_info["e0_sample"])
    print(f"edge_step       : {norm.edge_step:.4f}")
    print(f"边后平台        : {norm.plateau:.4f}")

    best, table = optimize_rbkg(norm)
    print(f"Rbkg 选中       : {best['rbkg']:.2f} Å  置信度 {best['confidence']}")

    chi_k = extract_chi_k(norm, rbkg=best["rbkg"])
    ft = ft_to_r_space(chi_k.k, chi_k.chi, np.arange(0.0, 4.5, 0.02))
    shell = (ft.r > 1.0) & (ft.r < 2.6)
    print(f"FT 第一峰       : {ft.r[shell][np.argmax(ft.magnitude[shell])]:.3f} Å")
