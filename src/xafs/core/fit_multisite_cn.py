# -*- coding: utf-8 -*-
"""fit_multisite_cn.py — P5-D 多位点正式 CN（--absorber-site all）（自 xafs_fit.py 纯拆分）

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

from fit_conditional_cn import (  # noqa: E402
    _conditional_fit_qualified, build_formal_cn_block, check_n_ratio,
)
from fit_shell_plan import (  # noqa: E402
    CN_SYSTEMATIC_BUDGET, _num, _s02_calibrated_flag, _t5_s02_hint,
    map_n0_to_plan, select_shell_paths,
)


# P5-D：多位点正式 CN（--absorber-site all）
# ================================================================
def _site_fractional(model: StructureModel) -> list[float]:
    """从 StructureModel 取吸收位的分数坐标（asym_sites 缺失时回退解析 source_cif）。"""
    label = str(model.absorber_site or "")
    m = re.match(r"frac:\(([^,]+),([^,]+),([^)]+)\)", label)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    for site in (model.asym_sites or []):
        if str(site.get("label")) == label:
            return [float(site["x"]), float(site["y"]), float(site["z"])]
    if model.source_cif and Path(model.source_cif).is_file():
        import phase_pack
        parsed = phase_pack.parse_cif(model.source_cif)
        for site in parsed.get("sites", []):
            if str(site.get("label")) == label:
                fx, fy, fz = site["fract"]
                return [float(fx), float(fy), float(fz)]
    raise CNError(
        "absorber_site_missing",
        f"{model.model_id}: 找不到吸收位 {label!r} 的分数坐标")


def _site_motif_from_model(model: StructureModel) -> structure_lib.Motif:
    """由 CIF + 吸收位坐标构建该位点的团簇 motif（不写生产结构库）。"""
    if not model.source_cif:
        raise CNError("no_cif", f"{model.model_id}: 缺少 source_cif")
    sites, occs = artemis.read_cif_sites(
        model.source_cif, absorber=model.absorber, rmax=6.0,
        absorber_fract=_site_fractional(model), with_occupancy=True)
    if not sites:
        raise CNError(
            "absorber_missing",
            f"{model.model_id}@{model.absorber_site}: rmax=6.0 Å 内无近邻")
    # motif id 会参与 FEFF 缓存目录名（Windows 不允许 ':'），必须做
    # 文件系统安全化；只影响诊断缓存，不改变 phase/site 溯源字段。
    safe_id = re.sub(r"[^A-Za-z0-9_.@-]+", "_",
                     f"{model.model_id}@{model.absorber_site}")
    return structure_lib.Motif(
        id=safe_id,
        absorber=model.absorber,
        kind="cluster",
        nuclearity="m1",
        source={"type": "cif_site", "ref": model.source_ref or ""},
        sites=sites,
        shell_expect=structure_lib.geometric_shells(sites, occs=occs),
        absorber_site=model.absorber_site,
        source_ref=model.source_ref,
        source_cif_sha256=model.source_cif_sha256,
        model_level=model.model_level,
        provenance_status=model.provenance_status,
    )


def _first_shell_stats(rows: list[dict]) -> dict:
    """把一个位点 formal 行里的第一壳行汇总成 N₀/N_fit/误差/键长。"""
    first = [r for r in rows if r.get("shell_group") == "first_shell"]
    n0 = sum(_num(r.get("n0")) or 0.0 for r in first)
    n_fit = sum(_num(r.get("n_fit")) or 0.0 for r in first)
    errors = [e for e in (_num(r.get("n_err_formal")) for r in first)
              if e is not None]
    r_terms = [(float(r["r_cif"]), _num(r.get("n0")) or 0.0)
               for r in first if _num(r.get("r_cif")) is not None]
    sig_terms = [(_num(r.get("sigma2")), _num(r.get("n0")) or 0.0)
                 for r in first if _num(r.get("sigma2")) is not None]
    dr_terms = [(_num(r.get("delta_r")), _num(r.get("n0")) or 0.0)
                for r in first if _num(r.get("delta_r")) is not None]
    e0_terms = [(_num(r.get("delta_e0")), _num(r.get("n0")) or 0.0)
                for r in first if _num(r.get("delta_e0")) is not None]
    rf = [_num(r.get("rf")) for r in first]
    rf = [v for v in rf if v is not None]
    scatterers = sorted({str(r.get("scatterer") or "") for r in first
                         if r.get("scatterer")})
    return {
        "n0": n0,
        "n_fit": n_fit,
        "n_err": (math.sqrt(sum(e * e for e in errors))
                  if errors else None),
        "r_cif": (sum(r * n for r, n in r_terms) / n0 if n0 > 0
                  and r_terms else None),
        "sigma2": (sum(s * n for s, n in sig_terms) / n0 if n0 > 0
                   and sig_terms else None),
        "delta_r": (sum(v * n for v, n in dr_terms) / n0 if n0 > 0
                    and dr_terms else None),
        "delta_e0": (sum(v * n for v, n in e0_terms) / n0 if n0 > 0
                     and e0_terms else None),
        "rf": (max(rf) if rf else None),
        "scatterer": (scatterers[0] if len(scatterers) == 1 else "mixed"),
        "n_rows": len(first),
    }


MULTI_SITE_BASE_UNRELIABLE = (
    "基础拟合质量未达标 (cn_reliable=False)，拒绝多位点 CN 宣称")


def downgrade_multi_site_cn(multi: dict, *, reason: str | None = None) -> dict:
    """主拟合 ``cn_reliable=False`` 时，把多位点正式 CN 整体如实降级。

    【红线 2 / 异常如实降级】单位点路径由 ``build_formal_cn_block`` 在源头
    以 ``cn_reliable=False`` 硬拒绝；多位点路径必须同步：即使某个位点的
    条件 CN 数值看起来正常，也不能借"逐位点独立"洗白基础拟合的失败。

    本函数**幂等**，供 ``run_multi_site_formal_cn`` 内部与两个 CLI 入口
    （``run_xafs.py`` / ``xafs_report.py``）共同调用，避免三处口径漂移：

    - 整体 ``status`` → ``base_fit_unreliable``，并保留 ``prior_status``；
    - ``per_site`` 中原本 ``ok`` 的记录一并降级；
    - 所有 ``rows`` 打上 ``base_cn_unreliable`` 标记；加权汇总行
      （``ALL_WEIGHTED`` / ``weighted_average``）及任何仍在宣称 CN 的行，
      ``gate_status`` 强制 ``failed``、CN 数值列清空（与
      ``cn_table.refusal_row`` 的"拒绝行不是 CN 行"契约一致）。
    """
    if not isinstance(multi, dict):
        return multi
    detail = reason or MULTI_SITE_BASE_UNRELIABLE
    if multi.get("status") != "base_fit_unreliable":
        multi["prior_status"] = multi.get("status")
        multi["status"] = "base_fit_unreliable"
    multi["cn_reliable"] = False
    multi["base_cn_reliable"] = False
    multi["degraded"] = True
    multi["reject_reason"] = detail
    multi["degradation_reasons"] = [detail]
    for record in multi.get("per_site") or []:
        if not isinstance(record, dict):
            continue
        record["base_cn_reliable"] = False
        if record.get("status") == "ok":
            record["status"] = "base_fit_unreliable"
            record["reject_reason"] = detail
    for row in multi.get("rows") or []:
        if not isinstance(row, dict):
            continue
        flags = str(row.get("flags") or "")
        if "base_cn_unreliable" not in flags.split(";"):
            row["flags"] = (flags + ";base_cn_unreliable").strip(";")
        is_summary = (row.get("n_status") == "weighted_average"
                      or row.get("absorber_site") == "ALL_WEIGHTED"
                      or row.get("shell_group") == "first_shell_weighted")
        claims_cn = (str(row.get("n_fit") or "").strip() != ""
                     and row.get("gate_status") in
                     ("passed", "warning", "", None))
        if is_summary or claims_cn:
            row["gate_status"] = "failed"
            prior = str(row.get("reject_reason") or "")
            if detail not in prior.split(";"):
                row["reject_reason"] = (prior + ";" + detail).strip(";")
            # 拒绝行不是 CN 行：不得保留任何会被误读为通过的数字。
            row["n_fit"] = ""
            row["n_ratio"] = ""
            row["n_status"] = "diagnostic"
            row["cn_label"] = "none"
    return multi


def _joint_multi_site_conditional_cn(
    res, site_plans: list[dict], site_mappings: list[dict],
    site_weights: list[float], site_labels: list[str],
    absorber_symbol: str, s02: float,
    kmin: float = 3.0, kmax: float = 12.0,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """审查 item 7：多位点**联合**条件 CN —— χ_obs = Σ w_i χ_i。

    旧版 ``run_multi_site_formal_cn`` 对每个位点独立拟合**完整** χ_obs，
    再化学计量加权，物理上错误：
      (1) 单位点被迫解释全部观测振幅，首壳 N 被 ~1/w_i 抬高；
      (2) 每个独立拟合各自占用全部 N_idp 预算，却在**同一份数据**上重复
          宣称独立参数，违反全局 Stern 信息量守恒（红线 3）。

    本函数把所有位点的 SS 路径拼成**一个**联合模型，路径有效简并度 =
    w_i × N₀（化学计量权重固定，不拟合权重），一次性联合拟合：
      固定 N → 条件 CN（逐位点首壳参考路径放开 N，单一比例因子/位点），
    Hamilton F-test 在联合模型上做（固定全部 N vs 放开各位点首壳 N）。
    再按位点切片回每位点 conditional_cn：N/N_err 除以 w_i 还原局域配位
    数；Rf/ΔE₀/N_var/N_idp 为联合共享值。自由 N 对照在联合模型上会放开
    全部路径 N、必然超信息量预算，故省略（其本身就是过拟合）。

    返回 (per_site_conditional, joint_meta)。联合拟合被 Stern 门禁拒绝时
    抛 XAFSConfigError，由调用方对全部位点如实降级。
    """
    n_sites = len(site_plans)
    joint_files, joint_labels, joint_shells = [], [], []
    joint_n0, joint_degen = [], []
    offsets, per_site_first, per_site_ref = [], [], []
    joint_links: dict[int, dict] = {}
    joint_ms_links: list[dict] = []
    off = 0
    for si in range(n_sites):
        plan, mapping = site_plans[si], site_mappings[si]
        w = float(site_weights[si])
        shells = plan["shells"]
        n0 = mapping["n0_per_path"]
        degens = plan["degens"]
        np_i = len(plan["files"])
        offsets.append(off)
        for pi in range(np_i):
            joint_files.append(plan["files"][pi])
            lab = f"{site_labels[si]}:{plan['labels'][pi]}"
            joint_labels.append(lab)
            sh = dict(shells[pi])
            sh["label"] = lab
            joint_shells.append(sh)
            joint_n0.append(float(n0[pi]) * w)
            joint_degen.append(float(degens[pi]) * w)
        # 位点内首壳（reff 最小）与参考路径（首壳中 N₀ 最大）
        reffs = [float(s.get("reff", s.get("reff_mean", 0.0)))
                 for s in shells]
        min_r = min(reffs)
        first_local = [pi for pi in range(np_i)
                       if abs(reffs[pi] - min_r) < 1e-6] or [0]
        ref_local = max(first_local, key=lambda pi: float(n0[pi]))
        first_global = [off + pi for pi in first_local]
        ref_global = off + ref_local
        per_site_first.append(first_global)
        per_site_ref.append(ref_global)
        # 子壳绑定（map_n0_to_plan 的 path_links）+ 首壳单一比例因子，
        # 全部平移到位点全局下标；不同位点之间绝不互相绑定。
        for link in (mapping.get("path_links") or []):
            joint_links[off + int(link["target"])] = {
                "target": off + int(link["target"]),
                "source": off + int(link["source"]),
                "degen_ratio": link["degen_ratio"],
                "share_deltar": False, "share_sigma2": False}
        for pi in first_local:
            if pi == ref_local:
                continue
            tgt = off + pi
            ratio = (float(n0[pi]) / float(n0[ref_local])
                     if float(n0[ref_local]) > 0 else 1.0)
            joint_links[tgt] = {
                "target": tgt, "source": ref_global,
                "degen_ratio": round(ratio, 6),
                "share_deltar": False, "share_sigma2": False}
        for mlink in (plan.get("ms_links") or []):
            shifted = dict(mlink)
            shifted["ms_path_index"] = off + int(mlink["ms_path_index"])
            shifted["parent_shell_index"] = off + int(
                mlink["parent_shell_index"])
            joint_ms_links.append(shifted)
        off += np_i

    rmin = min(float(p.get("rmin", 1.0)) for p in site_plans)
    rmax = max(float(p.get("rmax", 4.0)) for p in site_plans)
    degen0 = [round(v, 6) for v in joint_n0]
    links = list(joint_links.values())

    if verbose:
        print(f"  [P5-D joint] 联合模型 {n_sites} 位点 / {len(joint_files)} "
              f"路径，固定化学计量权重，放开各位点首壳 N={per_site_ref}")
    fit_fixed = artemis.fit_r_space(
        res.chi_k, res.ft, joint_files, rmin=rmin, rmax=rmax,
        kmin=res.ft.kmin, kmax=res.ft.kmax, kweight=res.ft.kweight,
        s02=s02, vary_delta_e0=True, vary_n=[],
        path_degen=degen0, path_labels=joint_labels,
        ms_links=(joint_ms_links or None), fit_space="r")
    fit_cond = artemis.fit_r_space(
        res.chi_k, res.ft, joint_files, rmin=rmin, rmax=rmax,
        kmin=res.ft.kmin, kmax=res.ft.kmax, kweight=res.ft.kweight,
        s02=s02, vary_delta_e0=True, vary_n=list(per_site_ref),
        path_degen=degen0, path_labels=joint_labels,
        path_links=links,
        ms_links=(joint_ms_links or None), fit_space="r")

    try:
        ftest = artemis.shell_f_test(fit_fixed, fit_cond)
    except XAFSError as exc:
        ftest = {"F": None, "confidence": None, "significant": False,
                 "valid": False, "error": getattr(exc, "message", str(exc))}
    ftest_valid = bool(ftest.get("valid", False))
    ftest_significant = bool(ftest.get("significant", False))
    joint_ftest = {
        "F": ftest.get("F"), "confidence": ftest.get("confidence"),
        "significant": ftest_significant, "valid": ftest_valid,
        "p_value": (None if ftest.get("confidence") is None
                    else float(max(0.0, 1.0 - float(ftest.get("confidence"))))),
    }
    joint_fit_summary = {
        "r_factor": float(fit_cond.r_factor),
        "n_variables": int(fit_cond.n_variables),
        "n_independent": float(fit_cond.n_independent),
        "delta_e0": float(fit_cond.delta_e0),
        "overfit": bool(fit_cond.overfit),
        "rank_deficient": bool(fit_cond.rank_deficient),
    }

    def _slice_paths(fit, si):
        lo = offsets[si]
        hi = lo + len(site_plans[si]["files"])
        w = float(site_weights[si])
        out = []
        for p in fit.paths[lo:hi]:
            ne = (float(p.n_err) / w if p.n_err is not None and w > 0
                  else None)
            out.append({"label": p.label, "reff": float(p.r),
                        "N": float(p.degeneracy) / w if w > 0 else 0.0,
                        "N_err": ne, "delta_r": float(p.delta_r),
                        "sigma2": float(p.sigma2)})
        return out

    per_site: list[dict] = []
    for si in range(n_sites):
        w = float(site_weights[si])
        n0_local = list(site_mappings[si]["n0_per_path"])
        first_local = [g - offsets[si] for g in per_site_first[si]]
        cond_paths = _slice_paths(fit_cond, si)
        fixed_paths = _slice_paths(fit_fixed, si)
        n_fit_total = sum(cond_paths[pi]["N"] for pi in first_local
                          if pi < len(cond_paths))
        n0_total = sum(float(n0_local[pi]) for pi in first_local
                       if pi < len(n0_local))
        n_ratio = (n_fit_total / n0_total) if n0_total > 0 else None
        n_status = check_n_ratio(n_ratio)
        qualified, q_reasons = _conditional_fit_qualified(
            joint_fit_summary, cond_paths, first_local)
        fixed_qualified, fixed_reasons = _conditional_fit_qualified(
            {"r_factor": float(fit_fixed.r_factor),
             "n_variables": int(fit_fixed.n_variables),
             "n_independent": float(fit_fixed.n_independent),
             "delta_e0": float(fit_fixed.delta_e0)},
            fixed_paths, first_local, fixed_n=True)
        consistency = (n_status in ("ok", "warning") and ftest_valid
                       and not ftest_significant and qualified and fixed_qualified)
        per_site.append({
            "n_ratio": (round(n_ratio, 6) if n_ratio is not None else None),
            "n_fit_total": round(n_fit_total, 6),
            "n0_total": round(n0_total, 6),
            "n_status": n_status,
            "ref_idx": per_site_ref[si] - offsets[si],
            "first_shell_idxs": first_local,
            "fixed_n_fit": {
                "r_factor": float(fit_fixed.r_factor),
                "n_variables": int(fit_fixed.n_variables),
                "n_independent": float(fit_fixed.n_independent),
                "delta_e0": float(fit_fixed.delta_e0),
                "overfit": bool(fit_fixed.overfit)},
            "conditional_n_fit": dict(joint_fit_summary),
            "hamilton_f_test": dict(joint_ftest),
            "consistency_passed": bool(consistency),
            "qualification_reasons": fixed_reasons + q_reasons,
            "kmin": float(res.ft.kmin), "kmax": float(res.ft.kmax),
            "kweight": int(res.ft.kweight),
            "fixed_n_paths": fixed_paths,
            "conditional_n_paths": cond_paths,
            "joint_fit": True,
        })
    joint_meta = {
        "n_paths": len(joint_files), "n_sites": n_sites,
        "r_factor": float(fit_cond.r_factor),
        "n_variables": int(fit_cond.n_variables),
        "n_independent": float(fit_cond.n_independent),
        "hamilton_f_test": joint_ftest,
        "model": "joint_chi_obs_sum_w_i_chi_i",
    }
    return per_site, joint_meta


def run_multi_site_formal_cn(
    res,
    *,
    site_models: list[StructureModel],
    gate_results,
    s02: float | None = None,
    s02_source: str | None = None,
    data_sha256: str = "",
    software_version: str = cn_table.SOFTWARE_VERSION,
    verbose: bool = False,
    site_cn_reliability: list[bool] | None = None,
    s02_evidence: dict | None = None,
) -> dict:
    """P5-D：对 ``--absorber-site all`` 的全部不等价位点做**联合**条件 CN。

    审查 item 7：观测谱是各位点贡献的化学计量加权和 χ_obs=Σ w_i χ_i。
    旧版对每位点**独立**拟合完整 χ_obs 再加权，物理上错误（单位点被迫
    解释全部振幅 → N 虚高 ~1/w_i；各独立拟合在同一份数据上重复占用全部
    N_idp 预算 → 违反全局 Stern 守恒）。本版把所有位点 SS 路径拼成单一
    联合模型，有效简并度 = w_i×N₀（化学计量权重固定），一次性联合拟合，
    全局 Hamilton F-test，再按位点切片（N/N_err 除以 w_i 还原局域 CN）。

    ``site_cn_reliability``：主拟合的 CN 可靠性（逐位点，或由调用方按
    主结果广播）。任一为 ``False`` 则该位点不产出正式 CN——与单位点路径
    的 ``build_formal_cn_block`` 硬前置保持同一门禁口径。默认 ``None``
    表示调用方未提供该信息（保持历史行为）。

    输出契约：
    - 每个位点各自的门禁、条件 CN 行（``flags`` 带 ``multi_site_member``
      与 ``site_weight``）；``joint_fit`` 记录联合模型元数据；
    - 仅当**全部**位点 ``build_formal_cn_block`` 返回 ``status="ok"`` 时，
      追加一行 ``absorber_site="ALL_WEIGHTED"`` 的化学计量加权汇总；
    - 任一位点门禁/N₀ 映射失败，或联合拟合被 Stern 门禁拒绝：联合振幅
      不闭合，**所有**位点写拒绝行（混合谱中无法单独定量任何位点），
      且**不输出**加权汇总行；
    - ``site_cn_reliability`` 含任一 ``False``：整体
      ``status="base_fit_unreliable"``，绝不输出加权汇总行，也不允许
      任何行保留 ``gate_status="passed"`` 的 CN 宣称。
    """
    if not site_models:
        raise XAFSConfigError("FIT", "多位点正式 CN 缺少结构模型")
    if len(site_models) != len(gate_results):
        raise XAFSConfigError(
            "FIT", f"多位点模型数({len(site_models)}) != 门禁结果数"
                   f"({len(gate_results)})")
    absorber_symbol = site_models[0].absorber
    absorber = (get_absorber(absorber_symbol) if s02 is None
                else structure_id.absorber_with_s02(absorber_symbol, s02))
    if absorber.s02 is None:
        raise XAFSConfigError(
            "FIT", f"{absorber.symbol} 未标定 S₀²，拒绝多位点 CN 定量"
                   f"{_t5_s02_hint(absorber.symbol)}")

    if s02_source is None:
        s02_source = "--s02 runtime override" if s02 is not None else "absorber.py"
    n_sites = len(site_models)

    def _refusal_row(si: int, status: str, reason: str) -> dict:
        model = site_models[si]
        return cn_table.refusal_row(
            status, reason,
            phase_id=model.model_id,
            model_level=model.model_level,
            source_type=model.source_type,
            source_ref=model.source_ref,
            source_cif_sha256=model.source_cif_sha256,
            absorber_site=model.absorber_site,
            s02=absorber.s02,
            s02_source=s02_source,
            s02_calibrated=_s02_calibrated_flag(absorber, s02_source, s02_evidence),
        )

    # ---- 阶段 1：逐位点门禁 + 壳层规划 + N₀ 映射（保持位点顺序）----
    # 审查 item 7：联合模型 χ_obs=Σw_iχ_i 需要**全部**位点都可建模。先把
    # 每位点的 plan/mapping 与拒绝原因收集齐，再一次性联合拟合，杜绝旧版
    # "每位点独立拟合完整 χ_obs 再加权" 的振幅虚高与 N_idp 重复占用。
    records: list[dict] = []
    site_plans: list[dict | None] = []
    site_mappings: list[dict | None] = []
    refusals: dict[int, tuple[str, str]] = {}
    valid_sites: list[int] = []
    for si, (model, gate) in enumerate(zip(site_models, gate_results)):
        record = {
            "absorber_site": model.absorber_site,
            "site_weight": getattr(gate, "site_weight", None),
            "site_multiplicity": getattr(gate, "site_multiplicity", None),
            "site_occupancy": getattr(gate, "site_occupancy", None),
            "status": "gate_failed",
            "rows": [],
        }
        records.append(record)
        site_plans.append(None)
        site_mappings.append(None)
        if not getattr(gate, "ok", False):
            record["gate_code"] = getattr(gate, "code", None)
            record["gate_reason"] = getattr(gate, "reason", "")
            refusals[si] = ("gate_failed",
                            getattr(gate, "reason", "") or "CIF gate 拒绝")
            continue
        try:
            motif = _site_motif_from_model(model)
            plan = select_shell_paths(
                motif, edge=absorber.edge,
                kmin=res.ft.kmin, kmax=res.ft.kmax)
            n0_mapping = map_n0_to_plan(
                plan, gate.n0_shells,
                kmin=res.ft.kmin, kmax=res.ft.kmax)
            if not n0_mapping["mapping_ok"]:
                raise CNError("n0_mapping_failed", n0_mapping["mapping_reason"])
            site_plans[si] = plan
            site_mappings[si] = n0_mapping
            valid_sites.append(si)
        except (XAFSConfigError, XAFSDataError, CNError) as exc:
            refusals[si] = ("multi_site_error",
                            getattr(exc, "message", str(exc)))

    # ---- 阶段 2：多位点联合条件 CN（固定化学计量权重，全局 Stern 预算）----
    # 仅当全部位点可建模时联合振幅才闭合；任一位点缺失则混合谱中其贡献
    # 无法扣除，任何单站点 CN 都不可信——对**所有**位点如实拒绝（红线 2）。
    # 联合拟合自身被 Stern 门禁拒绝（N_var≥N_idp）时同样全员降级。
    joint_conditionals: dict[int, dict] = {}
    joint_meta: dict = {}
    if valid_sites and len(valid_sites) == n_sites:
        try:
            per_site_cond, joint_meta = _joint_multi_site_conditional_cn(
                res,
                site_plans=[site_plans[si] for si in valid_sites],
                site_mappings=[site_mappings[si] for si in valid_sites],
                site_weights=[float(getattr(gate_results[si], "site_weight",
                                            0.0) or 0.0)
                              for si in valid_sites],
                site_labels=[site_models[si].absorber_site
                             for si in valid_sites],
                absorber_symbol=absorber.symbol, s02=absorber.s02,
                kmin=res.ft.kmin, kmax=res.ft.kmax, verbose=verbose)
            for k, si in enumerate(valid_sites):
                joint_conditionals[si] = per_site_cond[k]
        except (XAFSConfigError, XAFSDataError, CNError) as exc:
            reason = getattr(exc, "message", str(exc))
            for si in valid_sites:
                refusals[si] = ("multi_site_error", reason)
            joint_conditionals = {}
    elif valid_sites:
        incomplete = ("多位点联合模型不完整：存在门禁/N₀ 映射失败的位点，"
                      "χ_obs=Σw_iχ_i 振幅不闭合，拒绝所有位点 CN 定量")
        for si in valid_sites:
            refusals[si] = ("multi_site_error", incomplete)

    # ---- 阶段 3：按位点顺序产出正式 CN 块或拒绝行 ----
    rows: list[dict] = []
    per_site: list[dict] = []
    all_ok = True
    for si in range(n_sites):
        model = site_models[si]
        gate = gate_results[si]
        record = records[si]
        if si in refusals:
            status, reason = refusals[si]
            all_ok = False
            record["status"] = status
            record["reject_reason"] = reason
            rows.append(_refusal_row(si, status, reason))
            per_site.append(record)
            continue
        try:
            block = build_formal_cn_block(
                conditional_cn=joint_conditionals[si],
                plan=site_plans[si], n0_mapping=site_mappings[si],
                structure_model=model, absorber=absorber,
                s02_source=s02_source, s02_evidence=s02_evidence,
                data_sha256=data_sha256,
                software_version=software_version,
                cn_reliable=(
                    False if site_cn_reliability is not None
                    and (si >= len(site_cn_reliability)
                         or site_cn_reliability[si] is not True)
                    else (True if site_cn_reliability is not None
                          else None)))
        except (XAFSConfigError, XAFSDataError, CNError) as exc:
            all_ok = False
            reason = getattr(exc, "message", str(exc))
            record.update({"status": "error", "reject_reason": reason})
            rows.append(_refusal_row(si, "multi_site_error", reason))
            per_site.append(record)
            continue
        if block["status"] != "ok":
            all_ok = False
            record.update({
                "status": block["status"],
                "reject_reason": block.get("reject_reason"),
            })
            rows.append(_refusal_row(
                si, block["status"], block.get("reject_reason") or ""))
            per_site.append(record)
            continue

        weight = float(getattr(gate, "site_weight", 0.0) or 0.0)
        for row in block["rows"]:
            flags = row.get("flags") or ""
            row["flags"] = (
                flags + ";" if flags else ""
            ) + f"multi_site_member;site_weight={weight:.6f}"
        rows.extend(block["rows"])
        stats = _first_shell_stats(block["rows"])
        record.update({
            "status": "ok",
            "site_weight": weight,
            "n0": stats["n0"],
            "n_fit": stats["n_fit"],
            "n_ratio": (stats["n_fit"] / stats["n0"]
                        if stats["n0"] > 0 else None),
            "rf": stats["rf"],
            "rows": len(block["rows"]),
        })
        per_site.append(record)

    result = {
        "status": "ok" if all_ok else "multi_site_failed",
        "rows": rows,
        "per_site": per_site,
        "multi_site": True,
        "joint_fit": joint_meta,
        "site_weights": {
            str(r["absorber_site"]): r.get("site_weight") for r in per_site},
        "reject_reason": (None if all_ok else
                          "至少一个吸收位点未通过门禁/条件 CN；"
                          "不输出化学计量加权汇总"),
        "model": site_models[0].to_json(),
        "models": [m.to_json() for m in site_models],
    }
    # 【红线 2】主拟合 CN 不可靠时，无论各位点条件 CN 数值多漂亮，都必须
    # 整体降级；此处在底层兜底，两个 CLI 入口不必各自重复实现。
    if site_cn_reliability is not None and (
            len(site_cn_reliability) < n_sites
            or any(v is not True for v in site_cn_reliability)):
        downgrade_multi_site_cn(result)
    # 用 status 判定而非 all_ok：前者同时覆盖了"基础拟合不可靠"与
    # "某位点门禁/CN 失败"两类降级，且对 mock 掉 build_formal_cn_block
    # 的场景同样成立，绝不落到下面的加权汇总分支。
    if result["status"] != "ok":
        return result

    # ---- 全部位点通过：化学计量加权汇总行 ----
    weights = [float(getattr(g, "site_weight", 0.0) or 0.0)
               for g in gate_results]
    site_stats = [_first_shell_stats(block_rows)
                  for block_rows in ([r for r in rows
                                      if r.get("absorber_site") == m.absorber_site]
                                     for m in site_models)]
    n0_total = sum(w * s["n0"] for w, s in zip(weights, site_stats))
    n_fit_total = sum(w * s["n_fit"] for w, s in zip(weights, site_stats))
    r_num = sum(w * s["n0"] * (s["r_cif"] or 0.0)
                for w, s in zip(weights, site_stats))
    sig_num = sum(w * s["n0"] * (s["sigma2"] or 0.0)
                  for w, s in zip(weights, site_stats))
    err_terms = [w * s["n_err"] for w, s in zip(weights, site_stats)
                 if s["n_err"] is not None]
    rf_vals = [s["rf"] for s in site_stats if s["rf"] is not None]
    first_rows = [r for r in rows if r.get("shell_group") == "first_shell"]
    scatterers = sorted({str(r.get("scatterer"))
                         for r in first_rows if r.get("scatterer")})
    dr_num = sum(w * s["n0"] * (s["delta_r"] or 0.0)
                 for w, s in zip(weights, site_stats))
    e0_num = sum(w * s["n0"] * (s["delta_e0"] or 0.0)
                 for w, s in zip(weights, site_stats))

    summary = {column: "" for column in cn_table.CN_TABLE_COLUMNS}
    summary.update({
        "phase_id": site_models[0].model_id,
        "model_level": site_models[0].model_level,
        "source_type": site_models[0].source_type,
        "source_ref": site_models[0].source_ref,
        "source_cif_sha256": site_models[0].source_cif_sha256,
        "absorber_site": "ALL_WEIGHTED",
        "shell_group": "first_shell_weighted",
        "shell_label": "+".join(
            f"{m.absorber_site}({w:.3f})"
            for m, w in zip(site_models, weights)),
        "scatterer": (scatterers[0] if len(scatterers) == 1 else "mixed"),
        "r_cif": (r_num / n0_total if n0_total > 0 else ""),
        "n0": n0_total,
        "n0_members": sum(int(s["n_rows"]) for s in site_stats),
        "n_fit": n_fit_total,
        "n_status": "weighted_average",
        "n_ratio": (n_fit_total / n0_total if n0_total > 0 else ""),
        "n_err_formal": (math.sqrt(sum(e * e for e in err_terms))
                         if err_terms else ""),
        "n_err_systematic": CN_SYSTEMATIC_BUDGET["relative_1sigma"],
        "delta_r": (dr_num / n0_total if n0_total > 0 else ""),
        "sigma2": (sig_num / n0_total if n0_total > 0 else ""),
        "delta_e0": (e0_num / n0_total if n0_total > 0 else ""),
        "rf": (max(rf_vals) if rf_vals else ""),
        "s02": absorber.s02,
        "s02_source": s02_source,
        "s02_calibrated": _s02_calibrated_flag(absorber, s02_source, s02_evidence),
        "kmin": min(_num(r.get("kmin")) or 0.0 for r in first_rows)
                if first_rows else "",
        "kmax": max(_num(r.get("kmax")) or 0.0 for r in first_rows)
                if first_rows else "",
        "kweight": first_rows[0].get("kweight", "") if first_rows else "",
        "rmin": min(_num(r.get("rmin")) or 0.0 for r in first_rows)
                if first_rows else "",
        "rmax": max(_num(r.get("rmax")) or 0.0 for r in first_rows)
                if first_rows else "",
        "n_idp": min(_num(r.get("n_idp")) or 0.0 for r in first_rows)
                 if first_rows else "",
        "n_variables": (int(_num(first_rows[0].get("n_variables")) or 0)
                        if first_rows else 0),
        "cn_label": "cn_conditional_multi_site",
        "gate_status": ("passed"
                        if all(r.get("gate_status") == "passed"
                               for r in first_rows) else "warning"),
        "reject_reason": "",
        "flags": "multi_site_weighted_summary",
        "assumptions": (
            "site_weight=m_i*occ_i_normalized;"
            "joint_chi_obs_sum_w_i_chi_i;"
            "fixed_stoichiometric_weights;"
            "per_site_first_shell_single_scale_factor;"
            "global_hamilton_f_test;"
            "first_shell_weighted_by_stoichiometry;"
            "no_partial_site_release"),
        "data_sha256": data_sha256,
        "software_version": software_version,
    })
    rows.append(summary)
    if verbose:
        print(f"  [P5-D] {len(site_models)} 个吸收位点全部通过；"
              f"加权 N₀={n0_total:.4f}  N_fit={n_fit_total:.4f}")
    return result
