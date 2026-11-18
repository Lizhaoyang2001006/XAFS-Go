# -*- coding: utf-8 -*-
"""
xafs_fit.py — 拟合链编排（S6.5 结构识别 + S7 自动/指定结构拟合）
================================================================
归约链（xafs_pipeline，S1–S6）与拟合链在这里分界：

    PipelineResult 进 → S6.5 结构识别
                     → S7 拟合（识别到 confident 结构，或显式给 motif）

关键决策：
  - 识别状态不是 confident（ambiguous / partial / no_match）时，默认只做
    L1 第一壳拟合；长程结构未知。``allow_ambiguous=True`` 才允许用 Top-1
    强制拟合，并在结果里标记。
  - 自动拟合的路径按 Reff + 散射元素分壳层；同一壳层的多条 SS 路径
    合并为一条代表路径，简并度加总，拟合出的 N 是**壳层总配位数**。
    这是 Artemis 里“合并不可分辨壳层”的自动化版本，代表路径的近似
    仍会在结果里保留 warning。
  - S₀² 只从 absorber.py 的标定值取；未标定元素拒绝 CN 定量。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

import artemis
import cn_table
import structure_id
import structure_lib
import xafs_writer
from absorber import get_absorber
from structure_model import CNError, StructureModel, validate_formal
from xafs_core import XAFSConfigError, XAFSDataError, XAFSError

# ---- facade: fit_shell_plan.py ----
from fit_shell_plan import (
    _find_motif,
    _ss_rows,
    _collinear_ms_rows,
    _mixture_rows,
    _hetero_branch_row,
    _hetero_shells_in_window,
    _replace_hetero_rows,
    select_shell_paths,
    _orig_select_shell_paths,
    _normalize_c3_paths,
    _s02_calibrated_flag,
    _t5_s02_hint,
    map_n0_to_plan,
    _num,
    CN_SYSTEMATIC_BUDGET,
    SELF_ABSORPTION_HIGH_EDGE_STEP,
    SELF_ABSORPTION_WARN_EDGE_STEP,
)

# ---- facade: fit_conditional_cn.py ----
from fit_conditional_cn import (
    build_conditional_fit_args,
    compute_n_ratio,
    check_n_ratio,
    parse_defect_model,
    check_defect_model,
    _conditional_fit_qualified,
    run_conditional_cn,
    build_formal_cn_block,
    CN_RATIO_OK_LO,
    CN_RATIO_OK_HI,
    CN_RATIO_WARN_LO,
    CN_RATIO_WARN_HI,
)

# ---- facade: fit_multisite_cn.py ----
from fit_multisite_cn import (
    _site_fractional,
    _site_motif_from_model,
    _first_shell_stats,
    MULTI_SITE_BASE_UNRELIABLE,
    downgrade_multi_site_cn,
    _joint_multi_site_conditional_cn,
    run_multi_site_formal_cn,
)

# ---- facade: fit_single.py ----
from fit_single import (
    classify_fluorescence_scan,
    _self_absorption_diag,
    _max_offdiag_correlation,
    systematic_budget,
    _first_shell_cn,
    xanes_exafs_cross_check,
    rdf_table,
    write_rdf_csv,
    make_s02_evidence,
    _formal_cn_meta,
    _attach_single_site_formal_cn,
    fit_identified,
    run,
    run_operando_exafs_series,
)


# ----------------------------------------------------------------
# 门面兼容层：拆分前所有函数同处一个模块，测试/用户可以用
#   patch.object(xafs_fit, "name", fake)
# 打桩内部函数。拆分后函数体在子模块里，内部调用查的是子模块全局名，
# 门面上的赋值传不进去。这里让门面模块的属性赋值同步到所有持有该名
# 的子模块，保持与拆分前完全一致的 monkeypatch 语义（不改任何逻辑）。
# ----------------------------------------------------------------
import sys as _sys
import types as _types

_FACADE_SUBMODULES = ("fit_shell_plan", "fit_conditional_cn",
                      "fit_multisite_cn", "fit_single")


class _FacadeModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        for _modname in _FACADE_SUBMODULES:
            _mod = _sys.modules.get(_modname)
            if _mod is not None and hasattr(_mod, name):
                setattr(_mod, name, value)


_sys.modules[__name__].__class__ = _FacadeModule
