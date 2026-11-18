# -*- coding: utf-8 -*-
"""
artemis.py — 阶段 5~7：由 FEFF 路径到结构参数
================================================================
对应 Artemis 的功能范围。函数名与 EXAFS 方程里的每一项对应：

    FEFF 输入       build_feff_input
    CIF 读取        read_cif_sites（gemmi 对称展开 → 以吸收原子为原点的近邻坐标）
    路径计算        run_feff6l → load_feff_path
    设参数          set_path_parameters （ΔR、σ²、S₀²、ΔE₀；N 直接改 path.degen）
    理论谱          exafs_model_k （标准 EXAFS 方程求和）
    R 空间拟合      fit_r_space （复数 χ(R) 的实部与虚部同时最小二乘，可选拟合 N）
    统计量          independent_points / r_factor

依赖方向：core ← io ← athena ← **artemis**（拟合需要 athena 的 FT）。
本模块不 import larch_wt。
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

import athena
from xafs_core import (FitResult, PathFit, XAFSConfigError, XAFSDataError)

# ---- facade: artemis_feff.py ----
from artemis_feff import (
    expected_bond_length,
    build_feff_input,
    bcc_lattice_sites,
    octahedron_sites,
    read_cif_sites,
    cif_hetero_cocupancy_branches,
    _set_phase_source,
    phase_source_for_current_thread,
    path_total_phase,
    cif_shell_occupancy_scale,
    run_feff6l,
    load_feff_path,
    describe_paths,
    _ELEMENTS,
    ATOMIC_NUMBER,
    HOLE_INDEX,
    COVALENT_RADII,
    BVS_R0,
    _PHASE_SOURCE_LOCK,
    _PHASE_SOURCE_LOCAL,
)

# ---- facade: artemis_stats.py ----
from artemis_stats import (
    set_path_parameters,
    exafs_model_k,
    independent_points,
    _bond_key,
    _path_ligand,
    _path_n_r,
    bond_valence_sum,
    correlated_debye_sigma2,
    sac_cluster_sites,
    diatomic_m1m2_sites,
    reconstruct_pair_distribution,
    estimate_r_noise,
    shell_f_test,
    r_factor,
    estimate_nanoparticle_size,
    compute_cowley_sro,
    check_shell_resolution_and_z_degeneracy,
    _parse_kweights,
    normalize_constraints,
    _parse_constraint_expression,
    einstein_sigma2,
    profile_chi2_contour,
    C3_LIMIT,
    C4_LIMIT,
    _LATTICE_BULK,
    HBAR2_OVER_2KB,
)

# ---- facade: artemis_fit_single.py ----
from artemis_fit_single import (
    fit_r_space,
)

# ---- facade: artemis_fit_multi.py ----
from artemis_fit_multi import (
    fit_multi_datasets,
)

# ================================================================
# 6. 自测
# ================================================================

if __name__ == "__main__":
    import sys
    import tempfile
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- CIF → sites 与内置 bcc 生成器交叉验证 ----
    bcc_cif = "\n".join([
        "data_fe_bcc",
        "_cell_length_a 2.866",
        "_cell_length_b 2.866",
        "_cell_length_c 2.866",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'I m -3 m'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "Fe1 Fe 0 0 0",
    ])
    with tempfile.TemporaryDirectory() as td:
        cif = Path(td) / "fe_bcc.cif"
        cif.write_text(bcc_cif, encoding="utf-8")
        sites_cif = read_cif_sites(cif, absorber="Fe", rmax=6.0)
    d_cif = sorted(round(float(np.sqrt(s[0]**2 + s[1]**2 + s[2]**2)), 4)
                   for s in sites_cif)
    d_ref = sorted(round(float(np.sqrt(s[0]**2 + s[1]**2 + s[2]**2)), 4)
                   for s in bcc_lattice_sites(2.866, 6.0))
    assert d_cif == d_ref, f"CIF 展开与内置 bcc 不一致: {d_cif[:8]} vs {d_ref[:8]}"
    print(f"CIF→sites 校验: {len(sites_cif)} 个近邻，与 bcc_lattice_sites 一致")

    workdir = Path("./_artemis_selftest")
    print("生成 Fe 箔（bcc）的 feff.inp ...")
    inp = build_feff_input(bcc_lattice_sites(2.866, 6.0), absorber="Fe", rmax=6.0,
                           title="Fe foil bcc")
    files = run_feff6l(inp, workdir)
    print(f"  路径数: {len(files)}")
    for row in describe_paths(files, 2):
        print(f"  {row['file']}: Reff={row['reff']:.3f} Å  "
              f"简并度={row['degeneracy']:.0f}  腿数={row['legs']}")
