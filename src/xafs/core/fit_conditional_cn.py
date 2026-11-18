# -*- coding: utf-8 -*-
"""fit_conditional_cn.py — P3 条件 CN、缺陷模型解析、run_conditional_cn、build_formal_cn_block（自 xafs_fit.py 纯拆分）

纯搬运拆分：函数体与原 xafs_fit.py 逐行一致，不含任何逻辑改动。
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

from fit_shell_plan import _num, _s02_calibrated_flag  # noqa: E402,F401


# ================================================================
# P3：条件 CN —— 固定 N → 自由 N（首壳单一比例因子）
# ================================================================
CN_RATIO_OK_LO, CN_RATIO_OK_HI = 0.7, 1.3
CN_RATIO_WARN_LO, CN_RATIO_WARN_HI = 0.6, 1.4


def build_conditional_fit_args(
        plan: dict, n0_mapping: dict, ref_idx: int | None = None) -> dict:
    """构建条件 CN 拟合的 artemis.fit_r_space 参数。

    第一步（固定 N）：vary_n=[], path_degen=n0_per_path，所有壳层 N 固定为 N₀。
    第二步（条件 CN）：vary_n=[ref_idx]，只放开首壳参考路径 N；
    其他首壳路径用 path_links.degen_ratio = N0_j/N0_ref 绑定到参考路径；
    远壳 N 保持固定。

    返回 dict：
        fit_args: dict  透传给 artemis.fit_r_space 的参数
        ref_idx: int    首壳参考路径下标（N₀ 最大的首壳路径）
        n0_total: float 首壳 N₀ 总和
    """
    n0_per_path = n0_mapping["n0_per_path"]
    path_links = list(n0_mapping.get("path_links") or [])
    n_paths = len(n0_per_path)
    plan_shells = plan.get("shells", [])

    # 找出首壳路径（reff 最小的若干条，scatterer 可能相同或不同）
    first_shell_idxs = []
    if plan_shells:
        min_r = min(float(s.get("reff", s.get("reff_mean", 0.0)))
                    for s in plan_shells)
        for i, s in enumerate(plan_shells):
            r = float(s.get("reff", s.get("reff_mean", 0.0)))
            if abs(r - min_r) < 1e-6:
                first_shell_idxs.append(i)

    if not first_shell_idxs:
        first_shell_idxs = [0]

    # 参考路径：首壳中 N₀ 最大的
    ref_idx = ref_idx if ref_idx is not None else \
        max(first_shell_idxs, key=lambda i: n0_per_path[i])
    ref_idx = int(ref_idx)
    if ref_idx < 0 or ref_idx >= n_paths:
        raise XAFSConfigError(
            "FIT", f"ref_idx {ref_idx} 超出路径范围 0–{n_paths-1}")

    n0_total = sum(n0_per_path[i] for i in first_shell_idxs)

    # path_links：非参考首壳路径绑定到参考路径
    # 先清除已有的首壳间链接，重新按 N₀ 比例建立
    links = {}
    for link in path_links:
        links[link["target"]] = link

    for i in first_shell_idxs:
        if i == ref_idx:
            continue
        ratio = n0_per_path[i] / n0_per_path[ref_idx] if n0_per_path[ref_idx] > 0 else 1.0
        links[i] = {
            "target": i, "source": ref_idx,
            "degen_ratio": round(ratio, 6),
            "share_deltar": False,
            "share_sigma2": False,
        }

    fit_args = {
        "vary_n": [ref_idx],           # 只放开首壳参考路径 N
        "path_degen": [round(v, 6) for v in n0_per_path],  # 所有路径 N₀ 初值
        "path_links": list(links.values()),
    }
    return {
        "fit_args": fit_args,
        "ref_idx": ref_idx,
        "n0_total": round(n0_total, 6),
        "first_shell_idxs": first_shell_idxs,
    }


def compute_n_ratio(fit_result: dict, n0_mapping: dict,
                    ref_idx: int | None = None) -> dict:
    """从拟合结果计算 N/N₀。

    N_fit_total = Σ 首壳拟合路径的 N
    N0_total   = Σ 首壳 N₀
    N/N0       = N_fit_total / N0_total

    返回 dict：
        n_ratio: float | None
        n_fit_total: float
        n0_total: float
        n_status: str  "ok" / "warning" / "out_of_range" / "undef"
    """
    paths = fit_result.get("paths", [])
    n0_per_path = n0_mapping.get("n0_per_path", [])
    plan_shells = fit_result.get("shells", []) or \
        n0_mapping.get("plan_shells", [])

    if not paths or not n0_per_path:
        return {"n_ratio": None, "n_fit_total": None, "n0_total": None,
                "n_status": "undef"}

    # 首壳路径：reff 最小的
    first_idxs = []
    if plan_shells:
        min_r = min(float(s.get("reff", s.get("reff_mean", 0.0)))
                    for s in plan_shells)
        for i, s in enumerate(plan_shells):
            r = float(s.get("reff", s.get("reff_mean", 0.0)))
            if abs(r - min_r) < 1e-6:
                first_idxs.append(i)
    if not first_idxs:
        first_idxs = [0] if len(paths) > 0 else []

    n_fit_total = 0.0
    n0_total = 0.0
    for i in first_idxs:
        if i < len(paths):
            n_fit_total += float(paths[i].degen or 0.0)
        if i < len(n0_per_path):
            n0_total += float(n0_per_path[i])

    if n0_total <= 0:
        return {"n_ratio": None, "n_fit_total": round(n_fit_total, 6),
                "n0_total": round(n0_total, 6), "n_status": "undef"}

    n_ratio = n_fit_total / n0_total
    status = check_n_ratio(n_ratio)

    return {
        "n_ratio": round(n_ratio, 6),
        "n_fit_total": round(n_fit_total, 6),
        "n0_total": round(n0_total, 6),
        "n_status": status,
    }


def check_n_ratio(n_ratio: float) -> str:
    """N/N₀ 范围检查（P3 门禁）。

    0.7–1.3: ok（通过）
    0.6–0.7 / 1.3–1.4: warning（警告）
    <0.6 或 >1.4: out_of_range（需要定量缺陷模型，否则拒绝）
    """
    if n_ratio is None:
        return "undef"
    if n_ratio < CN_RATIO_WARN_LO or n_ratio > CN_RATIO_WARN_HI:
        return "out_of_range"
    if n_ratio < CN_RATIO_OK_LO or n_ratio > CN_RATIO_OK_HI:
        return "warning"
    return "ok"


# ================================================================
# P3：缺陷模型解析
# ================================================================
def parse_defect_model(spec: str | dict) -> dict:
    """[Implements: EQ-06] 解析缺陷模型规范，返回标准化 dict。

    支持 JSON 字符串或 dict：
        {"type": "nanoparticle", "diameter_nm": 2.5, "shape": "spherical", "lattice": "fcc"}
        {"type": "vacancy", "site": "O1", "fraction": 0.1}
        {"type": "custom", "expected_n_ratio": 0.75, "uncertainty": 0.10, "justification": "..."}

    返回 dict 包含：type, expected_n_ratio, uncertainty, model_dependent (bool)
    """
    if isinstance(spec, str):
        import json as _json
        try:
            spec = _json.loads(spec)
        except _json.JSONDecodeError as exc:
            raise XAFSConfigError(
                "FIT", f"缺陷模型 JSON 解析失败: {exc}") from exc
    if not isinstance(spec, dict):
        raise XAFSConfigError(
            "FIT", "缺陷模型必须是 JSON 对象或 dict")

    mtype = str(spec.get("type", "")).lower()
    if mtype not in ("nanoparticle", "vacancy", "custom"):
        raise XAFSConfigError(
            "FIT", f"未知缺陷模型类型: {mtype!r}（支持 nanoparticle/vacancy/custom）")

    result = {"type": mtype, "model_dependent": True}

    if mtype == "nanoparticle":
        dia = spec.get("diameter_nm")
        lattice = str(spec.get("lattice", "")).lower()
        if dia is None:
            raise XAFSConfigError(
                "FIT", "nanoparticle 缺陷模型需要 diameter_nm")
        try:
            dia_nm = float(dia)
        except (TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", f"nanoparticle 的 diameter_nm={dia!r} 必须是数值") from exc
        if not math.isfinite(dia_nm) or dia_nm <= 0.0:
            raise XAFSConfigError(
                "FIT", f"nanoparticle 的 diameter_nm={dia!r} 必须 > 0 nm")
        if lattice not in ("fcc", "hcp", "bcc"):
            raise XAFSConfigError(
                "FIT", f"nanoparticle 仅对 fcc/hcp/bcc 开放，收到 {lattice!r}")
        shape = str(spec.get("shape", "spherical")).strip().lower()
        if shape not in ("spherical", "hemispherical"):
            raise XAFSConfigError(
                "FIT", f"nanoparticle shape={shape!r} 只支持 'spherical' / 'hemispherical'")
        r1_raw = spec.get("r1_A", spec.get("r1"))
        if r1_raw is not None:
            try:
                r1_val = float(r1_raw)
            except (TypeError, ValueError) as exc:
                raise XAFSConfigError(
                    "FIT", f"nanoparticle 的 r1_A={r1_raw!r} 必须是数值") from exc
            if not math.isfinite(r1_val) or r1_val <= 0.0:
                raise XAFSConfigError(
                    "FIT", f"nanoparticle 的 r1_A={r1_raw!r} 必须 > 0 Å")
        elif spec.get("absorber") is not None:
            abs_sym = str(spec["absorber"]).strip().capitalize()
            rcov = artemis.COVALENT_RADII.get(abs_sym)
            if rcov is None:
                raise XAFSConfigError(
                    "FIT", f"nanoparticle 未知吸收元素 {spec['absorber']!r}，请显式提供 r1_A")
            r1_val = 2.0 * float(rcov)
        else:
            # 常见催化过渡/贵金属第一壳金属键长标准中位值 (2.55 Å)
            r1_val = 2.55
        # 统一量纲为 Å：d_A = 10 × diameter_nm；半球形基底直径折算回同体积等效球径
        d_a = dia_nm * 10.0
        d_eff_a = d_a / (2.0 ** (1.0 / 3.0)) if shape == "hemispherical" else d_a
        u = r1_val / d_eff_a
        if u >= 1.0:
            raise XAFSConfigError(
                "FIT", f"nanoparticle 粒径 d={dia_nm} nm ({d_eff_a:.2f} Å) ≤ 第一壳键长 "
                       f"r1={r1_val:.2f} Å (u={u:.3f} ≥ 1)，球截断物理模型不成立")
        expected = 1.0 - 1.5 * u + 0.5 * u ** 3
        result["expected_n_ratio"] = round(max(0.0, expected), 6)
        result["uncertainty"] = float(spec.get("uncertainty", 0.10))
        result["diameter_nm"] = dia_nm
        result["lattice"] = lattice
        result["shape"] = shape
        result["r1_A"] = round(r1_val, 4)
        result["u_ratio"] = round(u, 6)

    elif mtype == "vacancy":
        site = spec.get("site")
        fraction = spec.get("fraction")
        if site is None or fraction is None:
            raise XAFSConfigError(
                "FIT", "vacancy 缺陷模型需要 site 与 fraction")
        result["expected_n_ratio"] = round(1.0 - float(fraction), 6)
        result["uncertainty"] = float(spec.get("uncertainty", 0.10))
        result["site"] = str(site)
        result["fraction"] = float(fraction)

    else:  # custom
        ratio = spec.get("expected_n_ratio")
        uncertainty = spec.get("uncertainty")
        if ratio is None or uncertainty is None:
            raise XAFSConfigError(
                "FIT", "custom 缺陷模型需要 expected_n_ratio 与 uncertainty")
        result["expected_n_ratio"] = float(ratio)
        result["uncertainty"] = float(uncertainty)
        result["justification"] = str(spec.get("justification", ""))

    return result


def check_defect_model(n_ratio: float, defect_model: dict) -> dict:
    """检查 N/N₀ 是否在缺陷模型预期的不确定度内。

    返回 dict：
        ok: bool
        status: str  "defect_ok" / "defect_model_mismatch" / "defect_model_missing"
        detail: str
    """
    if defect_model is None:
        return {"ok": False, "status": "defect_model_missing",
                "detail": "N/N₀ 超出范围但未提供定量缺陷模型"}
    expected = defect_model.get("expected_n_ratio")
    uncertainty = defect_model.get("uncertainty", 0.10)
    if expected is None:
        return {"ok": False, "status": "defect_model_missing",
                "detail": "缺陷模型缺少 expected_n_ratio"}
    if abs(n_ratio - expected) <= uncertainty:
        return {"ok": True, "status": "defect_ok",
                "detail": f"N/N₀={n_ratio:.3f} 在模型预期 {expected:.3f}±{uncertainty:.2f} 范围内"}
    return {"ok": False, "status": "defect_model_mismatch",
            "detail": f"N/N₀={n_ratio:.3f} 偏离模型预期 {expected:.3f}±{uncertainty:.2f}"}




# ================================================================
# P3：条件 CN 端到端 —— 固定 N → 自由 N（首壳单一比例因子）→ Hamilton F-test
# ================================================================
def _conditional_fit_qualified(cond_fit: dict, cond_paths: list,
                               first_shell_idxs: list, *,
                               fixed_n: bool = False) -> tuple[bool, list]:
    """Independently qualify each fit's raw metrics; fixed N has no N error gate."""
    stage = "固定 N" if fixed_n else "条件 CN"
    reasons: list[str] = []
    cond_fit = cond_fit or {}
    rf = _num(cond_fit.get("r_factor"))
    de0 = _num(cond_fit.get("delta_e0"))
    nv = _num(cond_fit.get("n_variables"))
    nidp = _num(cond_fit.get("n_independent"))
    if cond_fit.get("rank_deficient") is True:
        reasons.append(f"{stage} 雅可比矩阵秩亏")
    if rf is None or not (0.0 <= rf <= 0.08):
        reasons.append(f"{stage} Rf={rf} 缺失/无效或 > 0.08")
    if de0 is None or not (abs(de0) < 15.0):
        reasons.append(f"{stage} |ΔE0|={de0} 缺失/无效或 ≥ 15 eV")
    if (nv is None or nidp is None or isinstance(cond_fit.get("n_variables"), bool)
            or nv < 0 or not nv.is_integer() or not (nv < nidp)):
        reasons.append(f"{stage} N_var={nv} ≥ N_idp={nidp} 或自由度记录无效（过拟合）")
    first = {int(i) for i in (first_shell_idxs or [])}
    if not cond_paths or not first or any(i < 0 or i >= len(cond_paths) for i in first):
        reasons.append(f"{stage} 路径/首壳原始记录缺失或不完整")
    for i, p in enumerate(cond_paths or []):
        label = p.get("label", i)
        sigma2 = _num(p.get("sigma2"))
        if sigma2 is None or sigma2 <= 5e-4 or p.get("sigma2_at_bound") is True:
            reasons.append(f"{stage} {label} σ² 缺失/无效或撞下界")
        if i in first and not fixed_n:
            ne = p.get("N_err")
            nval = _num(p.get("N"))
            if ne is not None and (_num(ne) is None or nval is None
                                   or float(ne) < 0 or float(ne) > 1.0
                                   or float(ne) > 0.5 * nval):
                reasons.append(f"{stage} {label} N 形式误差过大或无效")
    return (not reasons), reasons


def run_conditional_cn(
    res, plan: dict, n0_mapping: dict,
    absorber_symbol: str, s02: float,
    ref_idx: int | None = None,
    kmin: float = 3.0, kmax: float = 12.0,
    rmin: float | None = None, rmax: float | None = None,
    verbose: bool = True
) -> dict:
    """P3 条件 CN 端到端：固定 N 拟合 → 条件 CN（首壳单一比例因子）
    → 自由 N 对照 → Hamilton F-test 一致性判据。

    参数:
        res: PipelineResult（S1–S6 产物，含 chi_k, ft）
        plan: select_shell_paths 的输出（files, degens, labels, shells, rmin, rmax）
        n0_mapping: map_n0_to_plan 的输出（n0_per_path, path_links, ...）
        absorber_symbol: 吸收元素符号
        s02: 标定的 S₀² 值
        ref_idx: 首壳参考路径下标（None 时自动取 N₀ 最大的首壳路径）
        kmin/kmax: FT 窗口
        rmin/rmax: 拟合窗口（None 时用 plan 中的值）
        verbose: 是否打印进度

    返回 dict:
        conditional_cn: {
            n_ratio, n_fit_total, n0_total, n_status,   # N/N₀ 及状态
            ref_idx, first_shell_idxs,                  # 参考路径与首壳路径
            fixed_n_fit, conditional_n_fit, free_n_fit, # 三种拟合的摘要
            hamilton_f_test,                            # F-test 结果
            consistency_passed,                         # 一致性通过标志
            fixed_n_paths, conditional_n_paths          # 路径级 N 值
        }
    """
    n0_per_path = n0_mapping["n0_per_path"]
    path_links = list(n0_mapping.get("path_links") or [])
    n_paths = len(n0_per_path)
    plan_shells = plan.get("shells", [])

    if rmin is None:
        rmin = float(plan.get("rmin", 1.0))
    if rmax is None:
        rmax = float(plan.get("rmax", 4.0))

    # ---- 找出首壳路径（reff 最小）----
    first_shell_idxs = []
    if plan_shells:
        min_r = min(float(s.get("reff", s.get("reff_mean", 0.0)))
                    for s in plan_shells)
        for i, s in enumerate(plan_shells):
            r = float(s.get("reff", s.get("reff_mean", 0.0)))
            if abs(r - min_r) < 1e-6:
                first_shell_idxs.append(i)
    if not first_shell_idxs:
        first_shell_idxs = [0]

    # 参考路径：首壳中 N₀ 最大的
    if ref_idx is None:
        ref_idx = max(first_shell_idxs, key=lambda i: n0_per_path[i])
    ref_idx = int(ref_idx)
    if ref_idx < 0 or ref_idx >= n_paths:
        raise XAFSConfigError(
            "FIT", f"ref_idx {ref_idx} 超出路径范围 0–{n_paths-1}")

    n0_total = sum(float(n0_per_path[i]) for i in first_shell_idxs)
    if n0_total <= 0:
        raise XAFSConfigError(
            "FIT", f"首壳 N₀ 总和为 {n0_total}，无法计算 N/N₀")

    # ---- 第一步：固定 N 拟合（vary_n=[], path_degen=n0_per_path）----
    if verbose:
        print(f"  [P3-1] 固定 N 拟合：n_paths={n_paths}, "
              f"vary_n=[], N 固定为 N₀")
    _ms_links = plan.get("ms_links") or None
    _iso_dr = bool(plan.get("isotropic_deltar", False))
    fit_fixed = artemis.fit_r_space(
        res.chi_k, res.ft, plan["files"],
        rmin=rmin, rmax=rmax,
        kmin=res.ft.kmin, kmax=res.ft.kmax,
        kweight=res.ft.kweight,
        s02=s02, vary_delta_e0=True, vary_n=[],
        path_degen=[round(v, 6) for v in n0_per_path],
        path_labels=plan["labels"],
        ms_links=_ms_links,
        isotropic_deltar=_iso_dr,
        fit_space="r")

    # ---- 第二步：条件 CN（vary_n=[ref_idx]，首壳非参考路径用 path_links 绑定）----
    links = {}
    for link in path_links:
        links[link["target"]] = link

    for i in first_shell_idxs:
        if i == ref_idx:
            continue
        ratio = n0_per_path[i] / n0_per_path[ref_idx] if n0_per_path[ref_idx] > 0 else 1.0
        links[i] = {
            "target": i, "source": ref_idx,
            "degen_ratio": round(ratio, 6),
            "share_deltar": False,
            "share_sigma2": False,
        }

    if verbose:
        print(f"  [P3-2] 条件 CN 拟合：vary_n=[{ref_idx}], "
              f"path_links={len(links)} 条（首壳非参考路径绑定到参考路径）")
    fit_conditional = artemis.fit_r_space(
        res.chi_k, res.ft, plan["files"],
        rmin=rmin, rmax=rmax,
        kmin=res.ft.kmin, kmax=res.ft.kmax,
        kweight=res.ft.kweight,
        s02=s02, vary_delta_e0=True, vary_n=[ref_idx],
        path_degen=[round(v, 6) for v in n0_per_path],
        path_labels=plan["labels"],
        path_links=list(links.values()),
        ms_links=_ms_links,
        isotropic_deltar=_iso_dr,
        fit_space="r")

    # ---- 第三步：自由 N 对照（所有 N 可变，仅作信息参考；N_idp 不足时不阻断条件 CN）----
    if verbose:
        print(f"  [P3-3] 自由 N 对照拟合：vary_n=True，所有 N 自由")
    try:
        fit_free = artemis.fit_r_space(
            res.chi_k, res.ft, plan["files"],
            rmin=rmin, rmax=rmax,
            kmin=res.ft.kmin, kmax=res.ft.kmax,
            kweight=res.ft.kweight,
            s02=s02, vary_delta_e0=True, vary_n=True,
            path_degen=[round(v, 6) for v in n0_per_path],
            path_labels=plan["labels"],
            ms_links=_ms_links,
            isotropic_deltar=_iso_dr,
            fit_space="r")
    except XAFSConfigError:
        fit_free = None

    # ---- 计算 N/N₀（基于条件 CN 拟合结果）----
    n_fit_total = 0.0
    for i in first_shell_idxs:
        if i < len(fit_conditional.paths):
            n_fit_total += float(fit_conditional.paths[i].degeneracy or 0.0)

    n_ratio = n_fit_total / n0_total
    n_status = check_n_ratio(n_ratio)

    # ---- Hamilton F-test：固定 N vs 条件 CN（嵌套模型比较）----
    try:
        ftest = artemis.shell_f_test(fit_fixed, fit_conditional)
    except XAFSError as exc:
        ftest = {"F": None, "confidence": None, "significant": False,
                 "valid": False, "error": getattr(exc, "message", str(exc))}

    # 一致性判据（每次独立拟合独立判定；F-test 不可执行 ≠ 不显著）：
    # - N/N₀ 在 ok/warning 范围
    # - Hamilton F-test 确实执行（valid）且不显著 → 固定 N 充分
    # - 固定 N 与条件 CN 各自物理资格通过；固定参数 N_err=None 不误拒
    ftest_valid = bool(ftest.get("valid", False))
    ftest_significant = bool(ftest.get("significant", False))
    cond_qualified, cond_reasons = _conditional_fit_qualified(
        {"r_factor": float(fit_conditional.r_factor),
         "delta_e0": float(fit_conditional.delta_e0),
         "n_variables": int(fit_conditional.n_variables),
         "n_independent": float(fit_conditional.n_independent),
         "rank_deficient": bool(getattr(fit_conditional, "rank_deficient", False))},
        [{"label": p.label, "N": float(p.degeneracy),
          "N_err": (float(p.n_err) if p.n_err is not None else None),
          "sigma2": float(p.sigma2),
          "sigma2_at_bound": bool(getattr(p, "sigma2_at_bound", False))}
         for p in fit_conditional.paths],
        first_shell_idxs)
    fixed_qualified, fixed_reasons = _conditional_fit_qualified(
        {"r_factor": float(fit_fixed.r_factor),
         "delta_e0": float(fit_fixed.delta_e0),
         "n_variables": int(fit_fixed.n_variables),
         "n_independent": float(fit_fixed.n_independent),
         "rank_deficient": bool(getattr(fit_fixed, "rank_deficient", False))},
        [{"label": p.label, "sigma2": float(p.sigma2),
          "sigma2_at_bound": bool(getattr(p, "sigma2_at_bound", False))} for p in fit_fixed.paths],
        first_shell_idxs, fixed_n=True)
    consistency_passed = (
        n_status in ("ok", "warning")
        and ftest_valid
        and not ftest_significant
        and cond_qualified and fixed_qualified
    )

    if verbose:
        print(f"  [P3-4] N/N₀ = {n_ratio:.4f} ({n_status}), "
              f"F-test valid={ftest_valid} significant={ftest_significant}, "
              f"fixed_qualified={fixed_qualified}, conditional_qualified={cond_qualified}, "
              f"consistency_passed={consistency_passed}")

    return {
        "n_ratio": round(n_ratio, 6),
        "n_fit_total": round(n_fit_total, 6),
        "n0_total": round(n0_total, 6),
        "n_status": n_status,
        "ref_idx": ref_idx,
        "first_shell_idxs": first_shell_idxs,
        "fixed_n_fit": {
            "r_factor": float(fit_fixed.r_factor),
            "n_variables": int(fit_fixed.n_variables),
            "n_independent": float(fit_fixed.n_independent),
            "delta_e0": float(fit_fixed.delta_e0),
            "overfit": bool(fit_fixed.overfit),
        },
        "conditional_n_fit": {
            "r_factor": float(fit_conditional.r_factor),
            "n_variables": int(fit_conditional.n_variables),
            "n_independent": float(fit_conditional.n_independent),
            "delta_e0": float(fit_conditional.delta_e0),
            "overfit": bool(fit_conditional.overfit),
        },
        "free_n_fit": ({
            "r_factor": float(fit_free.r_factor),
            "n_variables": int(fit_free.n_variables),
            "n_independent": float(fit_free.n_independent),
            "delta_e0": float(fit_free.delta_e0),
            "overfit": bool(fit_free.overfit),
        } if fit_free is not None else {"skipped_due_to_n_idp": True}),
        "hamilton_f_test": {
            "F": ftest.get("F"),
            "confidence": ftest.get("confidence"),
            "significant": ftest_significant,
            "valid": ftest_valid,
            "p_value": (None if ftest.get("confidence") is None
                        else float(max(0.0, 1.0 - float(ftest.get("confidence"))))),
        },
        "consistency_passed": bool(consistency_passed),
        "qualification_reasons": fixed_reasons + cond_reasons,
        "kmin": float(res.ft.kmin),
        "kmax": float(res.ft.kmax),
        "kweight": int(res.ft.kweight),
        "fixed_n_paths": [
            {"label": p.label, "reff": float(p.r),
             "N": float(p.degeneracy),
             "N_err": (float(p.n_err) if p.n_err is not None else None),
             "delta_r": float(p.delta_r),
             "sigma2": float(p.sigma2)}
            for p in fit_fixed.paths
        ],
        "conditional_n_paths": [
            {"label": p.label, "reff": float(p.r),
             "N": float(p.degeneracy),
             "N_err": (float(p.n_err) if p.n_err is not None else None),
             "delta_r": float(p.delta_r),
             "sigma2": float(p.sigma2)}
            for p in fit_conditional.paths
        ],
    }


def build_formal_cn_block(
    *,
    conditional_cn: dict,
    plan: dict,
    n0_mapping: dict,
    structure_model: StructureModel,
    absorber,
    cn_systematics: dict | None = None,
    s02_source: str = "absorber.py",
    data_sha256: str = "",
    software_version: str = cn_table.SOFTWARE_VERSION,
    cn_reliable: bool | None = None,
    reliability_reasons: list[str] | None = None,
    s02_evidence: dict | None = None,
) -> dict:
    """把条件 CN 结果转成正式 CN 块。

    - **硬前置**：主拟合 ``cn_reliable=False``（Rf>0.08 / |ΔE0|≥15 eV /
      自吸收 cn_gate / 参数撞界 / 过拟合）时，无论条件 CN 是否自洽，
      一律拒绝产出正式 CN 行。宪法"物理守恒高于拟合残差"要求门禁在前，
      一致性检验在后；不能用另一次固定 N 拟合把被否决的结果洗白。
    - ``consistency_passed=True``：``status="ok"``，每个拟合路径一行；
      第一壳 ``n_status="fitted"``，远壳 ``n_status="fixed"``。
    - 否则：``status="consistency_failed"``，``rows=[]``，拒绝原因可复核。
    - 每行包含 §4.4 要求的全部 provenance 字段：CIF 来源、sha256、
      吸收位、N₀、模型级别、S₀²、数据哈希与软件版本。
    """
    model = structure_model.to_json()
    if cn_reliable is False:
        detail = "；".join(reliability_reasons or []) or "主拟合可靠性门禁未通过"
        return {
            "status": "consistency_failed",
            "rows": [],
            "reject_reason": ("主拟合 CN 门禁拒绝（" + detail
                              + "）；正式配位数表不产出"),
            "model": model,
            "cn_reliable": False,
        }
    if not _s02_calibrated_flag(absorber, s02_source, s02_evidence):
        return {
            "status": "s02_uncalibrated", "rows": [],
            "reject_reason": ("S₀² 未标定来源或元素/实际值/证书哈希不一致；"
                              "手填覆盖只用于诊断，正式配位数表不产出"),
            "s02_calibrated": False, "model": model,
        }
    cond_fit = conditional_cn.get("conditional_n_fit") or {}
    rf = float(cond_fit.get("r_factor", 1.0))
    overall_n_status = str(conditional_cn.get("n_status") or "")
    ftest = conditional_cn.get("hamilton_f_test") or {}
    paths = conditional_cn.get("conditional_n_paths") or []
    first_shell_idxs = list(conditional_cn.get("first_shell_idxs") or [])
    reasons = []
    if overall_n_status not in ("ok", "warning"):
        reasons.append(
            f"N/N₀={conditional_cn.get('n_ratio')} "
            f"({overall_n_status}) 超出范围，且未提供定量缺陷模型")
    # F-test 不可执行 ≠ 不显著：缺 valid 一律拒绝，不能用未执行的检验洗白
    if not bool(ftest.get("valid", False)):
        reasons.append("Hamilton F-test 不可执行（自由度不足/缺 χ²），"
                       "不能据此宣称固定 N 一致")
    elif bool(ftest.get("significant")):
        p_value = ftest.get("p_value")
        reasons.append(
            "自由 N 显著改善（Hamilton F-test"
            + (f" p={float(p_value):.4g}" if p_value is not None else "")
            + "），固定 N 不一致")
    # 条件 CN 拟合自身物理资格（与常规 motif 门禁同口径，未 round 原值）：
    # Rf≤0.08、|ΔE0|<15、N_var<N_idp、σ² 不撞下界、首壳 N 形式误差不过大
    _qualified, fit_reasons = _conditional_fit_qualified(
        cond_fit, paths, first_shell_idxs)
    reasons.extend(fit_reasons)
    fixed_paths = conditional_cn.get("fixed_n_paths") or []
    _fixed_qualified, fixed_reasons = _conditional_fit_qualified(
        conditional_cn.get("fixed_n_fit") or {}, fixed_paths,
        first_shell_idxs, fixed_n=True)
    reasons.extend(fixed_reasons)
    expected_paths = len(plan.get("shells") or [])
    if len(fixed_paths) != expected_paths:
        reasons.append("固定 N 路径原始记录与壳层计划数量不一致")
    if len(paths) != expected_paths:
        reasons.append("条件 CN 路径原始记录与壳层计划数量不一致")
    # 门禁在前、一致性在后：只要存在拒绝原因即拒产正式行，
    # 即便 consistency_passed=True 也不放行（防止陈旧/不一致记录洗白）
    if not conditional_cn.get("consistency_passed") or reasons:
        return {
            "status": "consistency_failed",
            "rows": [],
            "reject_reason": "；".join(reasons) or "条件 CN 一致性未通过",
            "model": model,
        }

    first_shell = {int(i) for i in first_shell_idxs}
    n0_per_path = list(n0_mapping.get("n0_per_path") or [])
    shells = plan.get("shells") or []
    systematics = (cn_systematics or {}).get("relative_1sigma")
    row_gate = "passed" if overall_n_status == "ok" else "warning"
    row_flags = ("consistency_passed" if overall_n_status == "ok"
                 else "consistency_passed;n_ratio_warning")
    assumptions = (
        "first_shell_single_scale_factor;"
        "higher_shells_fixed_at_N0;"
        "hamilton_f_test_nonsignificant;"
        "data_sha256_over_input_files"
    )

    rows: list[dict] = []
    for i, shell in enumerate(shells):
        if i >= len(paths):
            break
        path = paths[i]
        fitted = i in first_shell
        n_fit = float(path.get("N") or 0.0)
        n0 = (float(n0_per_path[i]) if i < len(n0_per_path)
              else float(shell.get("degen") or 0.0))
        n_ratio = ((n_fit / n0) if fitted and n0 > 0
                   else (1.0 if not fitted else ""))
        row = {column: "" for column in cn_table.CN_TABLE_COLUMNS}
        row.update({
            "phase_id": structure_model.model_id,
            "model_level": structure_model.model_level,
            "source_type": structure_model.source_type,
            "source_ref": structure_model.source_ref,
            "source_cif_sha256": structure_model.source_cif_sha256,
            "absorber_site": structure_model.absorber_site,
            "shell_group": "first_shell" if fitted else "higher_shell",
            "shell_label": shell.get("label", ""),
            "scatterer": shell.get("scatterer", ""),
            "r_cif": float(shell.get("reff")
                           or shell.get("reff_mean") or 0.0),
            "n0": n0,
            "n0_members": shell.get("n_members", ""),
            "n_fit": n_fit,
            "n_status": "fitted" if fitted else "fixed",
            "n_ratio": n_ratio,
            "n_err_formal": (path.get("N_err") if fitted else ""),
            "n_err_systematic": (systematics if fitted else ""),
            "delta_r": path.get("delta_r", ""),
            "sigma2": path.get("sigma2", ""),
            "delta_e0": cond_fit.get("delta_e0", ""),
            "rf": rf,
            "s02": absorber.s02,
            "s02_source": s02_source,
            "s02_calibrated": _s02_calibrated_flag(absorber, s02_source, s02_evidence),
            "kmin": conditional_cn.get("kmin", ""),
            "kmax": conditional_cn.get("kmax", ""),
            "kweight": conditional_cn.get("kweight", ""),
            "rmin": plan.get("rmin", ""),
            "rmax": plan.get("rmax", ""),
            "n_idp": cond_fit.get("n_independent", ""),
            "n_variables": cond_fit.get("n_variables", ""),
            "cn_label": "cn_conditional",
            "gate_status": row_gate,
            "reject_reason": "",
            "flags": row_flags,
            "assumptions": assumptions,
            "data_sha256": data_sha256,
            "software_version": software_version,
        })
        rows.append(row)
    return {"status": "ok", "rows": rows, "reject_reason": None,
            "model": model}
