# -*- coding: utf-8 -*-
"""fit_single.py — 自吸收诊断、S02 证据校验、L1 退守与 S7 单谱拟合 fit_identified / run / operando 序列（自 xafs_fit.py 纯拆分）

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
    build_formal_cn_block, run_conditional_cn,
)
from fit_multisite_cn import _site_motif_from_model  # noqa: E402
from fit_shell_plan import (  # noqa: E402
    CN_SYSTEMATIC_BUDGET, SELF_ABSORPTION_HIGH_EDGE_STEP,
    SELF_ABSORPTION_WARN_EDGE_STEP, _find_motif, _normalize_c3_paths,
    _s02_calibrated_flag, _t5_s02_hint, map_n0_to_plan,
    select_shell_paths,
)


def classify_fluorescence_scan(res) -> tuple[bool | None, str]:
    import xafs_io

    merged = getattr(res, "merged", None)
    scans = getattr(res, "scans", None)
    if not scans:
        return xafs_io.measurement_identity(merged)
    rejected = set(((getattr(merged, "meta", None) or {})
                    .get("merge_info") or {}).get("rejected_indices") or [])
    identities = [xafs_io.measurement_identity(scan)
                  for i, scan in enumerate(scans) if i not in rejected]
    modes = {identity[0] for identity in identities}
    basis = "；".join(identity[1] for identity in identities)
    if len(modes) != 1 or None in modes:
        return None, "保留扫描原始测量身份未知或不兼容；" + basis
    return identities[0][0], basis


def _self_absorption_diag(res) -> dict:
    """[Implements: EQ-05] 荧光模式自吸收风险（不校正，只决定能否定量）。

    荧光信号 ∝ 吸收量，但出射荧光会被样品自身再吸收；浓度/厚度越大，
    EXAFS 振幅被压低越多，表现为 N 偏小。可靠校正需要样品组成、密度、
    厚度与几何，本管线不假设这些量，因此在荧光模式下：
      - Δμ ≥ 1.0 → risk=high，cn_gate=True（拒绝 CN/相分数定量）
      - 0.3 ≤ Δμ < 1.0 → risk=possible，只警告
      - Δμ < 0.3 → risk=low（稀样，自吸收通常可忽略）
      - 探测模式无法判别 → risk=unknown，cn_gate=True（保守拒绝）

    只有**明确透射**才不受此门禁影响；文件自带 xmu 且无通道信息时
    不能假定是透射，否则浓样荧光会被静默放行。
    """
    merged = getattr(res, "merged", None)
    meta = ((getattr(merged, "meta", None) or {})
            if merged is not None else {})
    mode = str(meta.get("mode") or "unknown")
    norm = getattr(res, "norm", None)
    edge_step = float(getattr(norm, "edge_step", float("nan")))
    is_fluor, basis = classify_fluorescence_scan(res)
    sa = getattr(res, "self_absorption", None)
    if is_fluor is None:
        return {"mode": mode, "edge_step": edge_step,
                "corrected": bool(sa and sa.get("corrected")),
                "risk": "unknown", "cn_gate": True, "mode_basis": basis,
                "note": (f"无法判定原始探测模式（{basis}）；"
                         "自吸收校正标签不能替代测量身份，拒绝 CN/相分数定量")}
    if sa and sa.get("corrected"):
        gate = not bool(sa.get("reliable", False))
        return {
            "mode": mode, "edge_step": edge_step, "corrected": True,
            "reliable": bool(sa.get("reliable", False)),
            "risk": ("corrected_unreliable" if gate else "corrected"),
            "cn_gate": gate,
            "method": sa.get("method"),
            "formula": sa.get("formula"),
            "thickness_um": sa.get("thickness_um"),
            "density": sa.get("density"),
            "anginp": sa.get("anginp"), "angout": sa.get("angout"),
            "line": sa.get("line"),
            "correction_factor_median": sa.get("correction_factor_median"),
            "note": ("已做 Booth 自吸收校正；可靠性取决于总化学式、厚度、"
                     "有效密度与角度的准确性（" + str(sa.get("note", "")) + "）"
                     + ("；" + str(sa["warning"]) if sa.get("warning") else "")),
        }
    diag = {"mode": mode, "edge_step": edge_step, "corrected": False,
            "risk": "none", "cn_gate": False,
            "note": "非荧光模式，不涉及自吸收"}
    diag["mode_basis"] = basis
    if is_fluor is False:
        if math.isfinite(edge_step):
            if edge_step >= 2.5:
                diag["transmission_thickness_risk"] = "high"
                diag["note"] = (
                    f"非荧光模式，不涉及荧光自吸收；但透射边跳 Δμ={edge_step:.3f} ≥ 2.5，"
                    f"存在显著厚度/针孔效应与高次谐波泄漏风险，可能压低振幅、N 与标定 S₀²"
                )
            elif edge_step >= 1.5:
                diag["transmission_thickness_risk"] = "moderate"
                diag["note"] = (
                    f"非荧光模式，不涉及荧光自吸收；但透射边跳 Δμ={edge_step:.3f} ≥ 1.5，"
                    f"需留意样品过厚或针孔效应引起的振幅压低"
                )
            else:
                diag["transmission_thickness_risk"] = "low"
        else:
            diag["transmission_thickness_risk"] = "unknown"
        return diag
    if not math.isfinite(edge_step):
        diag.update(risk="unknown", cn_gate=True,
                    note="荧光模式但边跳变缺失，无法判断自吸收，拒绝定量")
    elif edge_step >= SELF_ABSORPTION_HIGH_EDGE_STEP:
        diag.update(risk="high", cn_gate=True,
                    note=(f"荧光 Δμ={edge_step:.3f} ≥ "
                          f"{SELF_ABSORPTION_HIGH_EDGE_STEP:.1f}："
                          "浓样/厚样自吸收显著，未校正时振幅与 N 被压低，"
                          "需透射谱或组成/厚度已知的外部校正"))
    elif edge_step >= SELF_ABSORPTION_WARN_EDGE_STEP:
        diag.update(risk="possible",
                    note=(f"荧光 Δμ={edge_step:.3f} 处于可能受自吸收影响的"
                          "区间；若 N 系统性偏低，优先怀疑此项"))
    else:
        diag.update(risk="low",
                    note="荧光稀样，自吸收通常可忽略")
    return diag


def _max_offdiag_correlation(fit) -> tuple[float, str | None, str | None]:
    """返回 (|ρ|max, 变量名 i, 变量名 j)；无相关矩阵时返回 (0, None, None)。"""
    corr = getattr(fit, "correlation", None)
    names = getattr(fit, "parameter_names", None)
    if corr is None or names is None:
        return 0.0, None, None
    c = np.array(corr, dtype=float, copy=True)
    if c.ndim != 2 or c.size == 0:
        return 0.0, None, None
    np.fill_diagonal(c, 0.0)
    ij = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    return abs(float(c[ij])), names[ij[0]], names[ij[1]]


def systematic_budget(fit, flags: dict | None = None,
                      self_abs: dict | None = None,
                      truncated_shells: int = 0,
                      fit_weights: bool = False) -> dict:
    """按**本次拟合的实际条件**生成 CN 系统误差预算（1σ，相对值）。

    基础预算 0.30 是历史实测锚点（见 CN_SYSTEMATIC_BUDGET）。这里不重算
    基础项，只把**额外可归因的系统误差**与基础值按平方和合成：

      - Booth 自吸收已校正且 reliable：+0.10（化学式/厚度/角度误差残留）
      - 荧光自吸收未校正但处于 possible 区间：+0.15（振幅压低方向未知）
      - 最强参数相关 |ρ| ≥ 0.9：+0.10；0.8–0.9：+0.05
      - 因 N_idp 截断远壳层：+0.05
      - σ² 撞下界（形状失配）：+0.10

    宁可将不可归因项显式列出，也不把 0.30 当成放之四海皆准的常数。
    """
    flags = flags or {}
    self_abs = self_abs or {}
    base = float(CN_SYSTEMATIC_BUDGET["relative_1sigma"])
    extras: dict[str, float] = {}
    notes: list[str] = []

    if self_abs.get("corrected") and self_abs.get("reliable"):
        extras["self_absorption_residual"] = 0.10
        notes.append("Booth 校正后残留（化学式/厚度/角度误差）")
    elif (not self_abs.get("corrected")
          and self_abs.get("risk") == "possible"):
        extras["self_absorption_uncorrected"] = 0.15
        notes.append("荧光自吸收可能压低振幅且未校正")

    rho, ni, nj = _max_offdiag_correlation(fit)
    if rho >= 0.9:
        extras["parameter_correlation"] = 0.10
        notes.append(f"参数强相关 |ρ|={rho:.3f}"
                     + (f"（{ni} vs {nj}）" if ni else ""))
    elif rho >= 0.8:
        extras["parameter_correlation"] = 0.05
        notes.append(f"参数相关 |ρ|={rho:.3f}"
                     + (f"（{ni} vs {nj}）" if ni else ""))

    if truncated_shells > 0:
        extras["truncated_shells"] = 0.05
        notes.append(f"{truncated_shells} 条远壳层因 N_idp 截断")
    if flags.get("sigma2_at_bound"):
        extras["sigma2_at_bound"] = 0.10
        notes.append("σ² 撞下界，模型形状失配")

    total = math.hypot(base, *extras.values())
    nidp_sf = getattr(fit, "nidp_scale_factor", None)
    return {
        "relative_1sigma": round(total, 4),
        "base_relative_1sigma": base,
        "extra_components": extras,
        "components": dict(CN_SYSTEMATIC_BUDGET["components"]),
        "max_parameter_correlation": round(rho, 4),
        "nidp_scale_factor": (None if nidp_sf is None else round(float(nidp_sf), 4)),
        "note": ("基础预算 "
                 f"±{100 * base:.0f}% 与本次额外系统项按平方和合成"
                 + ("：" + "；".join(notes) if notes else "（无额外项）")
                 + "。n_err 仍是形式误差，不能当绝对精度。"),
    }


def _first_shell_cn(fit_out) -> dict:
    """从 motif / L1 拟合产物中取第一壳 CN 摘要（只读，不做推断）。"""
    out = {"n_fit": None, "n0": None, "label": None, "reff": None,
           "cn_reliable": None, "mode": None}
    if not isinstance(fit_out, dict):
        return out
    out["mode"] = fit_out.get("mode")
    out["cn_reliable"] = fit_out.get("cn_reliable")
    if fit_out.get("mode") == "motif_weights":
        return out
    l1 = fit_out.get("l1")
    if fit_out.get("mode") == "l1_first_shell" and isinstance(l1, dict):
        out["n_fit"] = (float(l1["N"]) if l1.get("N") is not None else None)
        out["reff"] = (float(l1["R"]) if l1.get("R") is not None else None)
        out["label"] = "L1 first shell"
        return out
    fit_obj = fit_out.get("fit")
    paths = list(getattr(fit_obj, "paths", None) or [])
    if paths:
        p = paths[0]
        degen = getattr(p, "degeneracy", None)
        out["n_fit"] = float(degen) if degen is not None else None
        out["reff"] = float(getattr(p, "r", None)) \
            if getattr(p, "r", None) is not None else None
    shells = fit_out.get("shells") or []
    if shells:
        sh = shells[0]
        out["n0"] = (float(sh["degen"]) if sh.get("degen") is not None
                     else None)
        out["label"] = sh.get("label")
        if out["reff"] is None and sh.get("reff") is not None:
            out["reff"] = float(sh["reff"])
    return out


def xanes_exafs_cross_check(res, fit_out=None, *,
                            white_line=None,
                            white_line_window=(-6.0, 40.0),
                            thresholds: dict | None = None) -> dict:
    """XANES–EXAFS 联合软校验：白线/边步/归一化 vs 第一壳 CN。

    **只产生警告，不改变 ``cn_reliable``、不拒绝 CN、不写正式表。**
    物理上白线强度受氧化态、配位几何、自吸收与多重散射共同影响，
    不能由 CN 单独解释；因此本函数只做**保守极端值筛查**，并把
    ``heuristic=True`` / ``affects_cn_reliable=False`` 写进产物。

    检查项：
      1. 归一化平台是否在 0.85–1.15；
      2. 边步是否非正/过弱/过强（原始 μ 跳变，不是归一化后平台）；
      3. XANES 导数双峰 ``e0_jump_risk``；
      4. 白线拟合是否可靠、白线位置相对 E0 是否越界；
      5. K 边极端对比：高 CN + 极弱白线，或低 CN + 极强白线；
      6. 荧光自吸收风险（若 fit_out 带该诊断）。

    ``white_line`` 可注入预计算结果（测试/批量复用），避免重复拟合。
    """
    th = {
        "plateau_range": (0.85, 1.15),
        "edge_step_min": 0.05,
        "edge_step_max": 5.0,
        "white_line_area_low": 0.5,
        "white_line_area_high": 12.0,
        "high_cn": 8.0,
        "low_cn": 1.5,
        "white_line_offset_range": (-8.0, 40.0),
    }
    if thresholds:
        th.update(thresholds)

    norm = getattr(res, "norm", None)
    xf = dict(getattr(res, "xanes_features", None) or {})
    fit_out = fit_out if fit_out is not None else getattr(res, "fit", None)
    absorber = getattr(res, "absorber_symbol", None)
    edge = "K"
    if absorber:
        try:
            edge = get_absorber(absorber).edge
        except (XAFSConfigError, KeyError):
            edge = str(xf.get("edge") or "K")
    else:
        edge = str(xf.get("edge") or "K")

    wl = white_line
    if wl is None:
        if norm is None:
            wl = {"available": False, "reliable": False,
                  "note": "无 NormalizedScan，无法拟合白线"}
        else:
            try:
                import athena
                wl = athena.fit_whiteline_area(
                    norm, window=white_line_window)
                wl = dict(wl)
                wl["available"] = True
            except Exception as exc:  # noqa: BLE001 - 软校验不得中断主链
                wl = {"available": False, "reliable": False,
                      "note": f"白线拟合失败: {type(exc).__name__}: {exc}"}

    first = _first_shell_cn(fit_out)
    warnings: list[dict] = []
    checks: dict[str, dict] = {}

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    # 1. 归一化平台
    plateau = float(getattr(norm, "plateau", float("nan"))) \
        if norm is not None else float("nan")
    p_lo, p_hi = th["plateau_range"]
    if not math.isfinite(plateau):
        checks["normalization_plateau"] = {
            "status": "unavailable", "value": None}
    elif p_lo <= plateau <= p_hi:
        checks["normalization_plateau"] = {
            "status": "ok", "value": round(plateau, 4)}
    else:
        checks["normalization_plateau"] = {
            "status": "warn", "value": round(plateau, 4)}
        warn("normalization_plateau_out_of_range",
             f"边后平台={plateau:.3f} 不在 {p_lo:.2f}–{p_hi:.2f}；"
             f"EXAFS 振幅/CN 与 XANES 白线均可能被归一化偏差污染")

    # 2. 边步
    estep = float(getattr(norm, "edge_step", float("nan"))) \
        if norm is not None else float("nan")
    if not math.isfinite(estep) or estep <= 0:
        checks["edge_step"] = {"status": "unavailable", "value": None}
    elif estep < th["edge_step_min"]:
        checks["edge_step"] = {"status": "warn", "value": round(estep, 4)}
        warn("edge_step_weak",
             f"原始边步={estep:.3g} 过弱（阈值 {th['edge_step_min']}）："
             f"样品过稀/过薄，EXAFS 信噪比与 CN 可靠性下降")
    elif estep > th["edge_step_max"]:
        checks["edge_step"] = {"status": "warn", "value": round(estep, 4)}
        warn("edge_step_strong",
             f"原始边步={estep:.3g} 过强（阈值 {th['edge_step_max']}）："
             f"厚样/浓样自吸收可能压低 EXAFS 振幅")
    else:
        checks["edge_step"] = {"status": "ok", "value": round(estep, 4)}

    # 3. E0 导数双峰
    if xf.get("e0_jump_risk") is True:
        checks["e0_jump_risk"] = {"status": "warn"}
        warn("e0_jump_risk",
             "XANES 上升沿导数双峰：E0 可能在两个台阶间跳变 3–6 eV，"
             "相位/振幅解释需复核")
    elif "e0_jump_risk" in xf:
        checks["e0_jump_risk"] = {"status": "ok"}
    else:
        checks["e0_jump_risk"] = {"status": "unavailable"}

    # 4. 白线可靠性与位置
    wl_reliable = bool(wl.get("reliable")) if isinstance(wl, dict) else False
    wl_area = (float(wl.get("white_line_area_norm") or 0.0)
               if isinstance(wl, dict) else None)
    wl_energy = (wl.get("white_line_energy")
                 if isinstance(wl, dict) else None)
    if not wl_reliable:
        checks["white_line_fit"] = {
            "status": "unavailable",
            "note": (wl or {}).get("note", "白线拟合不可靠")}
    else:
        checks["white_line_fit"] = {
            "status": "ok", "area_norm": round(wl_area, 4)}
        e0 = float(getattr(norm, "e0", float("nan"))) \
            if norm is not None else float("nan")
        if wl_energy is not None and math.isfinite(e0):
            offset = float(wl_energy) - e0
            lo, hi = th["white_line_offset_range"]
            if not (lo <= offset <= hi):
                checks["white_line_position"] = {
                    "status": "warn", "offset_eV": round(offset, 2)}
                warn("white_line_position_outlier",
                     f"白线位置 E_wl−E0={offset:+.1f} eV 超出 "
                     f"[{lo:+.0f}, {hi:+.0f}] eV；检查 E0 选择/边前背景/"
                     f"价态归属")
            else:
                checks["white_line_position"] = {
                    "status": "ok", "offset_eV": round(offset, 2)}

    # 5. K 边极端 CN–白线对比（L 边白线由 2p→5d 主导，不做此判据）
    n_fit = first.get("n_fit")
    if (str(edge).upper().startswith("K") and wl_reliable
            and n_fit is not None and first.get("cn_reliable") is not False):
        if n_fit >= th["high_cn"] and wl_area <= th["white_line_area_low"]:
            checks["cn_white_line_contrast"] = {
                "status": "warn", "n_fit": round(n_fit, 2),
                "white_line_area_norm": round(wl_area, 4)}
            warn("high_cn_low_white_line",
                 f"第一壳 N≈{n_fit:.1f} 但白线净面积仅 {wl_area:.2f} eV："
                 f"白线极弱与高 CN 同时出现，需排查归一化/自吸收/多相混合")
        elif n_fit <= th["low_cn"] and wl_area >= th["white_line_area_high"]:
            checks["cn_white_line_contrast"] = {
                "status": "warn", "n_fit": round(n_fit, 2),
                "white_line_area_norm": round(wl_area, 4)}
            warn("low_cn_high_white_line",
                 f"第一壳 N≈{n_fit:.1f} 但白线净面积达 {wl_area:.2f} eV："
                 f"强白线通常含氧化态/多重散射贡献，CN 可能被低估")
        else:
            checks["cn_white_line_contrast"] = {
                "status": "ok", "n_fit": round(n_fit, 2),
                "white_line_area_norm": round(wl_area, 4)}
    else:
        checks["cn_white_line_contrast"] = {"status": "unavailable"}

    # 6. 自吸收风险（fit_out 已给出诊断时）
    self_abs = (fit_out.get("self_absorption")
                if isinstance(fit_out, dict) else None)
    if isinstance(self_abs, dict):
        risk = self_abs.get("risk")
        if risk in ("high", "unknown", "corrected_unreliable"):
            checks["self_absorption"] = {"status": "warn", "risk": risk}
            warn("self_absorption_risk",
                 self_abs.get("note") or f"荧光自吸收风险={risk}")
        else:
            checks["self_absorption"] = {"status": "ok", "risk": risk}
    else:
        checks["self_absorption"] = {"status": "unavailable"}

    # 7. CN 可靠性信息（不改它，只记录）
    if isinstance(fit_out, dict) and first.get("cn_reliable") is False:
        checks["cn_reliability"] = {
            "status": "info", "cn_reliable": False}
    elif isinstance(fit_out, dict):
        checks["cn_reliability"] = {
            "status": "ok", "cn_reliable": first.get("cn_reliable")}
    else:
        checks["cn_reliability"] = {"status": "unavailable"}

    status = ("warning" if warnings
              else ("ok" if any(c["status"] == "ok"
                                for c in checks.values()) else "unavailable"))
    return {
        "enabled": True,
        "heuristic": True,
        "affects_cn_reliable": False,
        "status": status,
        "absorber": absorber,
        "edge": edge,
        "first_shell": first,
        "white_line": wl,
        "checks": checks,
        "warnings": warnings,
        "n_warnings": len(warnings),
        "note": ("XANES–EXAFS 软校验只做保守极端值筛查；白线强度不能"
                 "单独证明或否定 CN，本结果不改变 cn_reliable/正式 CN 门禁"),
    }


def rdf_table(fit) -> dict:
    """B9-1：把拟合出的 C3/C4 累积量重构为非高斯 g(R)。

    只对真正放开了 C3 或 C4 的路径生成——常规拟合没有非高斯信息，
    不能凭空画一条不对称分布。返回:
        {"rows": [{"label", "r", "g_r"}, ...],
         "summary": [{"label", "r_mode", "r_mean", "skewness", "area",
                      "c3", "c4"}, ...]}
    """
    rows, summary = [], []
    for p in fit.paths:
        if p.c3 is None and p.c4 is None:
            continue
        r_grid = np.linspace(max(0.5, p.r - 0.8), p.r + 1.0, 401)
        rdf = artemis.reconstruct_pair_distribution(
            r_grid, n=float(p.degeneracy), r0=float(p.r),
            sigma2=float(p.sigma2), c3=float(p.c3 or 0.0),
            c4=float(p.c4 or 0.0))
        rows.append({"label": p.label, "r": rdf["r"], "g_r": rdf["g_r"]})
        summary.append({"label": p.label, "r_mode": rdf["r_mode"],
                        "r_mean": rdf["r_mean"],
                        "r_thermodynamic": rdf["r_thermodynamic"],
                        "skewness": rdf["skewness"],
                        "area": rdf["area"], "c3": p.c3, "c4": p.c4,
                        "gram_charlier_clipped": rdf["gram_charlier_clipped"],
                        "unimodal": rdf["unimodal"]})
    return {"rows": rows, "summary": summary}


def write_rdf_csv(path, rdf: dict):
    """把 rdf_table 的结果写成 long-form CSV（每路径一段）。"""
    import csv
    path = Path(path)
    with xafs_writer.csv_buffer(path, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "r_A", "g_r", "r_mode", "r_mean",
                    "skewness", "area"])
        meta = {m["label"]: m for m in rdf.get("summary", [])}
        for block in rdf.get("rows", []):
            m = meta.get(block["label"], {})
            for riv, g in zip(block["r"], block["g_r"]):
                w.writerow([block["label"], f"{riv:.6f}", f"{g:.6e}",
                            m.get("r_mode"), m.get("r_mean"),
                            m.get("skewness"), m.get("area")])
    return path


def make_s02_evidence(payload: dict, *, certificate_file=None) -> dict:
    """Canonical payload hashes and on-disk certificate byte hashes are distinct."""
    import hashlib
    import json
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    evidence = {"payload": json.loads(serialized),
                "payload_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}
    if certificate_file is not None:
        path = Path(certificate_file).resolve()
        evidence.update(certificate_file=str(path),
                        certificate_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return evidence


def _formal_cn_meta(formal_cn: bool, structure_model: StructureModel | None,
                    status_override: str | None = None,
                    reason_override: str | None = None) -> dict:
    """给拟合结果附加诊断/正式双轨标记。

    P0a 只负责“拒绝或挂起”，绝不产生 ``n_fit`` 点值：
    - ``formal_cn=False``：仅标记 diagnostic_only；
    - 无模型：``no_cif``；
    - L1：``l1_diagnostic_only``；
    - 模型来源合格：``pending_consistency``；P3 条件 CN 成功后由
      ``build_formal_cn_block`` 覆盖为 ``ok``/``consistency_failed``；
    - 模型来源不合格：返回 ``validate_formal`` 的拒绝码。
    """
    if not formal_cn:
        return {"diagnostic_only": True, "cn_status": "diagnostic_only"}
    if status_override:
        status = status_override
        reason = reason_override
    elif structure_model is None:
        status = "no_cif"
        reason = "未提供 StructureModel（无合格结构来源）"
    else:
        verdict = validate_formal(structure_model)
        if verdict["ok"]:
            status = "pending_consistency"
            reason = "P0a 尚未实现固定 N 一致性检验；正式 CN 从 P3 开始"
        else:
            status = verdict["code"] or "unverified_source"
            reason = verdict["reason"]
    return {
        "diagnostic_only": True,
        "cn_status": status,
        "formal_cn": {
            "status": status,
            "rows": [],
            "reject_reason": reason,
            "model": (structure_model.to_json()
                      if structure_model is not None else None),
        },
    }


def _attach_single_site_formal_cn(
    out: dict,
    res,
    *,
    absorber,
    s02_label: str,
    s02_evidence: dict | None = None,
    formal_cn: bool,
    structure_model: StructureModel | None,
    gate_result=None,
    rmin: float | None = None,
    rmax: float | None = None,
    include_ms: bool = False,
    isotropic_deltar: bool = False,
    hetero_cocupancy: bool = False,
    is_l1: bool = False,
    verbose: bool = True,
) -> None:
    """挂载单晶位正式 CN 结果（与诊断轨 _structure_lib 解耦）。

    - 未提供 ``structure_model`` 且诊断轨为 L1 时，保持 #76 的 ``l1_diagnostic_only``；
    - 未提供 ``structure_model`` 且诊断轨为 motif 时，保持 #75 的 ``no_cif``；
    - 提供合格 ``structure_model`` + ``gate_result.ok`` 时，直接由已验证 CIF 构建
      ``_site_motif_from_model(structure_model)`` -> ``select_shell_paths`` ->
      ``map_n0_to_plan`` -> ``run_conditional_cn`` -> ``build_formal_cn_block``。
    """
    out["s02_calibrated"] = _s02_calibrated_flag(absorber, s02_label, s02_evidence)
    if is_l1 and structure_model is None:
        out.update(_formal_cn_meta(
            formal_cn, None,
            status_override=("l1_diagnostic_only" if formal_cn else None),
            reason_override=("L1 合成团簇只做诊断，不进入正式 CN 表"
                             if formal_cn else None)))
        return

    out.update(_formal_cn_meta(formal_cn, structure_model))
    if not (formal_cn and structure_model is not None):
        return
    formal_block = out.get("formal_cn") or {}
    if formal_block.get("status") != "pending_consistency":
        return

    conditional_cn = None
    conditional_cn_error = ""
    n0_mapping = None
    formal_plan = None
    if gate_result is not None and getattr(gate_result, "ok", False):
        try:
            formal_motif = _site_motif_from_model(structure_model)
            formal_plan = select_shell_paths(
                formal_motif, edge=absorber.edge,
                rmin=rmin, rmax=rmax,
                kmin=res.ft.kmin, kmax=res.ft.kmax,
                include_ms=include_ms,
                isotropic_deltar=isotropic_deltar,
                hetero_cocupancy=hetero_cocupancy)
            n0_mapping = map_n0_to_plan(
                formal_plan, gate_result.n0_shells,
                kmin=res.ft.kmin, kmax=res.ft.kmax)
            if n0_mapping["mapping_ok"]:
                conditional_cn = run_conditional_cn(
                    res, formal_plan, n0_mapping,
                    absorber_symbol=absorber.symbol, s02=absorber.s02,
                    kmin=res.ft.kmin, kmax=res.ft.kmax,
                    rmin=formal_plan["rmin"], rmax=formal_plan["rmax"],
                    verbose=verbose)
            else:
                conditional_cn_error = (
                    "N₀ 映射失败: " + n0_mapping["mapping_reason"])
            if verbose and conditional_cn is not None:
                print(f"  [P3] 条件 CN: N/N₀={conditional_cn['n_ratio']} "
                      f"({conditional_cn['n_status']}), "
                      f"consistency_passed={conditional_cn['consistency_passed']}")
        except (CNError, XAFSConfigError, XAFSDataError) as exc:
            conditional_cn_error = (
                "条件 CN 跳过: " + getattr(exc, "message", str(exc)))
            if verbose:
                print(f"  [P3] {conditional_cn_error}")
    else:
        conditional_cn_error = "缺少 CIF gate 结果，未运行条件 CN"
        if verbose:
            print(f"  [P3] {conditional_cn_error}")

    if conditional_cn is not None:
        out["conditional_cn"] = conditional_cn
    if (conditional_cn is not None and n0_mapping is not None
            and formal_plan is not None):
        block = build_formal_cn_block(
            conditional_cn=conditional_cn, plan=formal_plan,
            n0_mapping=n0_mapping,
            structure_model=structure_model,
            absorber=absorber,
            cn_systematics=out.get("cn_systematics"),
            s02_source=s02_label,
            s02_evidence=s02_evidence,
            data_sha256=(getattr(res, "data_sha256", "") or ""),
            cn_reliable=out.get("cn_reliable"),
            reliability_reasons=out.get("degradation_reasons"))
    else:
        block = {
            "status": "consistency_failed",
            "rows": [],
            "reject_reason": (conditional_cn_error or "条件 CN 未运行"),
            "model": structure_model.to_json(),
        }
    out["formal_cn"] = block
    out["cn_status"] = block["status"]


def fit_identified(res, motif_id: str | None = None,
                   absorber_symbol: str | None = None,
                   rmin: float | None = None, rmax: float | None = None,
                   allow_ambiguous: bool = False,
                   l1_ligand: str = "O", l1_geometry: str = "oct",
                   fit_weights: bool = False,
                   shared_sigma2: str | list | None = None,
                   kweights: tuple | list | None = None,
                   include_ms: bool = False,
                   vary_c3: bool | list | str = False,
                   vary_c4: bool | list | str = False,
                   isotropic_deltar: bool = False,
                   fit_space: str = "r",
                   bvs_restraint: dict | None = None,
                   catalytic_model=None,
                   hetero_cocupancy: bool = False,
                   constraints=None,
                   s02: float | None = None,
                   s02_source: str | None = None,
                   verbose: bool = True,
                   formal_cn: bool = False,
                   structure_model: StructureModel | None = None,
                   gate_result=None,
                   s02_evidence: dict | None = None) -> dict:
    """对识别结果做 S7 拟合；未 confident 时返回 L1 第一壳结果。

    ``fit_weights=True`` 仅对 mixture（多位点吸收）生效：固定每条路径的
    晶体学简并度，放开各组分权重（softmax，和=1）与每壳 ΔR/σ²/ΔE₀，
    给出相分数。此时 N 列是**有效简并度 N_crystal·w**，不是配位数。

    ``shared_sigma2`` 对标 Artemis 的路径约束：多条路径共享一个 σ²
    变量，减少自由参数、避免 σ² 与 ΔR 过度相关。
        None        每条路径独立 σ²（默认）
        "all"       所有路径共享一个 σ²
        [[0,1],[2]] 显式分组：路径 0、1 共享 σ²，路径 2 独立
    与 fit_weights 互斥（权重模式下 σ² 已按组分缩放，不再共享）。

    ``kweights``：多 k 权重联合拟合（对标 Artemis），如 ``(1, 2, 3)``。
    None 时沿用单权 ``res.ft.kweight``（默认 2），行为与历史完全一致。

    ``fit_space``（B11-3）：``"r"``（默认）/ ``"q"`` / ``"wt"``，透传给
    ``artemis.fit_r_space``。不同空间的 Rf 口径不同，不可横向比较。
    ``bvs_restraint``（B11-2）：``{"target", "sigma", "r0_table"?}`` 软约束
    键价和；L1 合成团簇不提供 R₀ 语义，遇到 L1 降级时显式拒绝而不是
    静默忽略。
    """
    absorber_symbol = absorber_symbol or getattr(res, "absorber_symbol", None)
    if not absorber_symbol:
        raise XAFSConfigError(
            "FIT", "缺少吸收元素：给 absorber_symbol 或 --absorber")
    absorber = (get_absorber(absorber_symbol) if s02 is None
                else structure_id.absorber_with_s02(absorber_symbol, s02))
    if absorber.s02 is None:
        raise XAFSConfigError(
            "FIT", f"{absorber.symbol} 未标定 S₀²，拒绝 CN 定量；"
                   f"先在 absorber.py 用标准样标定{_t5_s02_hint(absorber.symbol)}")
    s02_label = (s02_source if s02_source is not None else
                 ("--s02 runtime override" if s02 is not None
                  else "absorber.py"))
    if str(s02_label).startswith("s02_calibration:") and s02 is None:
        raise XAFSConfigError(
            "FIT", "T5 S₀² 来源已给出但 s02=None；拒绝在没有数值的情况下"
                   "记录标定溯源")
    self_abs = _self_absorption_diag(res)

    # ---- B13-1 显式催化模型：绕过晶相库搜索，直接进 S7 ----
    catalytic_motif = None
    if catalytic_model is not None:
        catalytic_motif = structure_lib.parse_catalytic_cluster_spec(
            absorber.symbol, catalytic_model)
        if str(catalytic_motif.absorber).capitalize() != absorber.symbol.capitalize():
            raise XAFSConfigError(
                "FIT", f"催化模型的吸收元素 {catalytic_motif.absorber!r} 与 "
                       f"--absorber {absorber.symbol!r} 不一致")

    sid = getattr(res, "structure", None)
    if catalytic_motif is not None:
        if sid is None:
            sid = {
                "status": "catalytic_model",
                "absorber": absorber.symbol,
                "ranked": [{"id": catalytic_motif.id,
                            "score": None, "de0": 0.0,
                            "kind": catalytic_motif.kind,
                            "nuclearity": catalytic_motif.nuclearity}],
                "indistinguishable": [],
                "uncovered_peaks": [],
                "coverage_ok": True,
                "confidence": {
                    "margin": None,
                    "note": "用户显式给定催化模型，未做结构库搜索"},
                "warning": ("使用 --catalytic-model 显式模型；识别状态"
                            "不代表库内排序结果，长程结构由该模型强制假设"),
            }
        if motif_id is None:
            motif_id = catalytic_motif.id
    else:
        if sid is None:
            sid = structure_id.identify_result(
                res, absorber_symbol=absorber_symbol,
                l1_ligand=l1_ligand, l1_geometry=l1_geometry,
                vary_c3=vary_c3, fit_space=fit_space, s02=s02,
                verbose=verbose)

    if catalytic_model is None and motif_id is None:
        status = sid.get("status")
        if status == "confident" or allow_ambiguous:
            motif_id = sid["ranked"][0]["id"]
        else:
            l1 = sid.get("l1_first_shell")
            if l1 is None:
                detail = sid.get("l1_error", "L1 拟合未生成")
                raise XAFSDataError(
                    "FIT", f"识别状态 {status}，不能自动全拟合：{detail}")
            l1_c3_bound = bool(l1.get("c3_at_bound", False))
            if bvs_restraint is not None:
                raise XAFSConfigError(
                    "FIT", "bvs_restraint 需要结构模型给出的配体与参考键长；"
                           "识别状态不是 confident，L1 合成团簇的 BVS 无物理"
                           "意义，拒绝静默忽略该约束")
            norm_obj = getattr(res, "norm", None)
            plateau_val = (float(getattr(norm_obj, "plateau", float("nan")))
                           if norm_obj is not None else float("nan"))
            plateau_bad = bool(
                norm_obj is not None and math.isfinite(plateau_val)
                and not (0.85 <= plateau_val <= 1.15))

            # L1 与常规 motif 同口径：过拟合/Rf/ΔE₀/N 误差/σ² 撞界/
            # C₃ 撞界/自吸收门禁任一不满足即不可靠。资格标志由 structure_id
            # 用未 round 的原始拟合值导出；缺字段时按不可靠 fail-safe。
            l1_reliable = (not l1.get("overfit", False)
                           and not l1.get("rf_bad", True)
                           and not l1.get("delta_e0_bad", True)
                           and not l1.get("n_err_bad", False)
                           and not l1.get("sigma2_at_bound", False)
                           and not self_abs["cn_gate"]
                           and not l1_c3_bound
                           and not plateau_bad)
            l1_flags = {"sigma2_at_bound":
                        bool(l1.get("sigma2_at_bound",
                                    float(l1.get("sigma2", 1.0)) <= 5e-4)),
                        "c3_at_bound": l1_c3_bound,
                        "plateau_bad": plateau_bad}
            out = {"mode": "l1_first_shell", "status": status,
                   "coverage_ok": sid.get("coverage_ok"),
                   "cn_reliable": l1_reliable,
                   "s02": absorber.s02, "s02_source": s02_label,
                   "flags": {"c3_dropped_by_nidp":
                             bool(l1.get("c3_dropped_by_nidp", False)),
                             "c3_at_bound": l1_c3_bound,
                             "plateau_bad": plateau_bad},
                   "structure": sid, "l1": l1,
                   "self_absorption": self_abs,
                   "cn_systematics": systematic_budget(
                       None, flags=l1_flags, self_abs=self_abs),
                   "warning": l1.get(
                       "warning", "仅第一壳可信；长程结构未知")}
            if self_abs["risk"] in ("high", "unknown",
                                    "corrected_unreliable"):
                prefix = ("荧光自吸收校正不可靠" if self_abs["corrected"]
                          else "荧光自吸收未校正")
                out["warning"] += ("；" + prefix + "（"
                                   + self_abs["note"] + "）")
            elif self_abs["risk"] == "possible":
                out["warning"] += ("；荧光自吸收风险（"
                                   + self_abs["note"] + "）")
            if fit_weights:
                out["warning"] += ("；识别状态不是 confident，"
                                   "--fit-weights 未执行（返回 L1 第一壳）")
            _attach_single_site_formal_cn(
                out, res, absorber=absorber, s02_label=s02_label,
                s02_evidence=s02_evidence,
                formal_cn=formal_cn, structure_model=structure_model,
                gate_result=gate_result, rmin=rmin, rmax=rmax,
                include_ms=include_ms, isotropic_deltar=isotropic_deltar,
                hetero_cocupancy=hetero_cocupancy, is_l1=True,
                verbose=verbose)
            return out

    motif = (catalytic_motif if catalytic_motif is not None
             else _find_motif(absorber.symbol, motif_id))
    if fit_weights and motif.kind != "mixture":
        raise XAFSConfigError(
            "FIT", f"{motif.id} 不是混合相（kind={motif.kind}），"
                   f"--fit-weights 只用于多位点吸收的 mixture")
    plan = select_shell_paths(motif, edge=absorber.edge,
                              rmin=rmin, rmax=rmax,
                              kmin=res.ft.kmin, kmax=res.ft.kmax,
                              group_by_component=fit_weights,
                              include_ms=include_ms,
                              isotropic_deltar=isotropic_deltar,
                              hetero_cocupancy=hetero_cocupancy)

    if verbose:
        print(f"  [S7] motif={motif.id}  R∈[{plan['rmin']:.2f},{plan['rmax']:.2f}] Å"
              f"  S₀²={absorber.s02}  N_idp={plan['n_independent']:.1f}")
        for sh in plan["shells"]:
            print(f"       {sh['label']:<10} Reff={sh['reff']:.3f} Å"
                  f"  N0={sh['degen']:.2f}  ({sh['n_members']} 条合并)")
        if plan["truncated"]:
            names = [t["label"] for t in plan["truncated"]]
            print(f"       ⚠ 信息量上限：最多 {plan['max_paths_by_idp']} 条壳层"
                  f"（3n+1 < N_idp），已截断 {names}")

    weight_init = None
    if fit_weights:
        all_ids = list(range(len(motif.components)))
        missing = [i for i in all_ids if i not in plan["component_ids"]]
        if missing:
            names = [motif.components[i]["id"] for i in missing]
            raise XAFSConfigError(
                "FIT", "权重拟合缺少组分路径：" + ", ".join(names)
                       + "；这些组分在 R 窗内没有可用壳层或被 N_idp 截断，"
                         "相分数不可辨识。请扩大 --fit-rmax 或减小壳层数")
        # 初始权重按识别阶段给出的组分比例；仅用于优化起点。
        weight_init = [float(motif.components[i]["weight"])
                       for i in plan["component_ids"]]
        total_w = sum(weight_init)
        if total_w <= 0:
            raise XAFSConfigError("FIT", "mixture 初始权重之和必须为正")
        weight_init = [w / total_w for w in weight_init]

    # ---- σ² 共享组（对标 Artemis 路径约束）----
    sigma2_groups = None
    sigma2_group_names = None
    if shared_sigma2 is not None and not fit_weights:
        n_path = len(plan["files"])
        if shared_sigma2 == "all":
            sigma2_groups = [list(range(n_path))]
            sigma2_group_names = ["all_shells"]
        elif isinstance(shared_sigma2, list):
            # 显式分组：[[0, 1], [2]] 验证下标合法
            for g in shared_sigma2:
                for pi in g:
                    if int(pi) < 0 or int(pi) >= n_path:
                        raise XAFSConfigError(
                            "FIT", f"shared_sigma2 路径下标 {pi} 超出范围 0–{n_path-1}")
            sigma2_groups = [list(map(int, g)) for g in shared_sigma2]
            sigma2_group_names = [f"s2grp{g}" for g in range(len(sigma2_groups))]
        else:
            raise XAFSConfigError(
                "FIT", f"shared_sigma2 必须是 'all' 或路径下标列表，收到 {shared_sigma2!r}")

    requested_c3 = _normalize_c3_paths(vary_c3, len(plan["files"]))
    requested_c4 = _normalize_c3_paths(vary_c4, len(plan["files"]))
    if fit_weights and requested_c3:
        raise XAFSConfigError(
            "FIT", "C3 与 mixture 相分数拟合不能同时开启："
                   "权重模式的有效简并度 N_crystal·w 不是配位数，"
                   "C3 无法解释为近邻分布偏斜")
    if isotropic_deltar and fit_weights:
        raise XAFSConfigError(
            "FIT", "isotropic_deltar 与 mixture 相分数拟合不能同时开启："
                   "各组分晶格膨胀不一致时全局 α 无物理意义")

    def _fit_with_cumulants(c3_arg, c4_arg):
        return artemis.fit_r_space(
            res.chi_k, res.ft, plan["files"],
            rmin=plan["rmin"], rmax=plan["rmax"],
            kmin=res.ft.kmin, kmax=res.ft.kmax,
            kweight=(tuple(kweights) if kweights else res.ft.kweight),
            s02=absorber.s02, vary_delta_e0=True, vary_n=not fit_weights,
            vary_weights=fit_weights,
            vary_c3=(c3_arg if c3_arg is not None else False),
            vary_c4=(c4_arg if c4_arg is not None else False),
            isotropic_deltar=isotropic_deltar,
            path_degen=plan["degens"], path_labels=plan["labels"],
            path_groups=(plan["groups"] if fit_weights else None),
            path_group_names=(plan["group_names"] if fit_weights else None),
            weight_init=weight_init,
            sigma2_groups=sigma2_groups,
            sigma2_group_names=sigma2_group_names,
            ms_links=(plan["ms_links"] or None),
            path_links=(plan.get("path_links") or None),
            constraints=constraints,
            fit_space=fit_space,
            bvs_restraint=bvs_restraint)

    # 自动 N_idp 防护（B9-1：C3 与 C4 完全对称）：
    # 先按用户请求放开；若 n_var ≥ N_idp，降级到仅第一壳；仍超限则
    # 彻底关闭，并分别记录 c3_dropped_by_nidp / c4_dropped_by_nidp。
    # C4 是更高阶项，优先降 C4 以保留用户更常需要的 C3。
    # 底层求解器对 N_var ≥ N_idp 在求解前硬拒绝（Stern 红线）。自动降级
    # 链必须把该拒绝当作“过拟合”信号继续降 C4/C3，而非让异常冒泡——否则
    # 用户请求的累积量在第一次试拟合就中断、来不及降级。
    def _fit_try(c3_arg, c4_arg):
        try:
            return _fit_with_cumulants(c3_arg, c4_arg)
        except XAFSConfigError as exc:
            if "Stern" in str(exc):
                return None
            raise

    def _overfit(f):
        return f is None or f.n_variables >= f.n_independent

    c3_effective = requested_c3
    c4_effective = requested_c4
    c3_dropped_by_nidp = False
    c4_dropped_by_nidp = False
    fit = _fit_try(c3_effective, c4_effective)

    if c4_effective and _overfit(fit):
        c4_dropped_by_nidp = True
        if c4_effective is True or tuple(c4_effective) != (0,):
            c4_effective = [0]
            fit = _fit_try(c3_effective, c4_effective)
        if _overfit(fit):
            c4_effective = None
            fit = _fit_try(c3_effective, c4_effective)

    if c3_effective and _overfit(fit):
        c3_dropped_by_nidp = True
        if c3_effective is True or tuple(c3_effective) != (0,):
            c3_effective = [0]
            fit = _fit_try(c3_effective, c4_effective)
        if _overfit(fit):
            c3_effective = None
            fit = _fit_try(c3_effective, c4_effective)

    # 降尽 C3/C4 后基础拟合仍违反 Stern（如有效 k 窗比请求窗更窄）：
    # 如实拒绝，绝不放宽门禁去求解过拟合模型。
    if fit is None:
        fit = _fit_with_cumulants(c3_effective, c4_effective)

    # ---- B13-1 DAC 金属键存在性：含/不含 M–M 的 Hamilton F-test ----
    # 只看 Rf 无法回答“到底有没有 M–M 键”：多一条路径总会降低残差。
    # 必须在同一 k/R 窗内比较两个嵌套模型，并把 N_MM 与其形式误差
    # 一起纳入判据（N 不显著大于误差时不算检出金属键）。
    mm_bond_ftest = None
    if (catalytic_motif is not None
            and str(catalytic_motif.nuclearity) == "binuclear"
            and not fit_weights):
        partner = None
        for sh in plan["shells"]:
            el = str(sh.get("scatterer") or "")
            if (el and el != absorber.symbol
                    and el not in structure_lib._LIGAND_ELEMENTS):
                partner = el
                break
        mm_idx = [i for i, sh in enumerate(plan["shells"])
                  if partner is not None and sh["scatterer"] == partner]
        mm_set = set(mm_idx)
        keep = [i for i in range(len(plan["files"])) if i not in mm_set]
        if partner is not None and mm_idx and keep:
            try:
                simple_fit = artemis.fit_r_space(
                    res.chi_k, res.ft,
                    [plan["files"][i] for i in keep],
                    rmin=plan["rmin"], rmax=plan["rmax"],
                    kmin=res.ft.kmin, kmax=res.ft.kmax,
                    kweight=(tuple(kweights) if kweights
                             else res.ft.kweight),
                    s02=absorber.s02, vary_delta_e0=True, vary_n=True,
                    path_degen=[plan["degens"][i] for i in keep],
                    path_labels=[plan["labels"][i] for i in keep],
                    fit_space=fit_space)
                ftest = artemis.shell_f_test(simple_fit, fit)
                mm_path = fit.paths[mm_idx[0]]
                conf = ftest.get("confidence")
                p_value = (None if conf is None
                           else float(max(0.0, 1.0 - float(conf))))
                n_err = mm_path.n_err
                significant = bool(
                    p_value is not None and p_value < 0.05
                    and n_err is not None
                    and float(mm_path.degeneracy) > float(n_err))
                mm_bond_ftest = {
                    "partner": partner,
                    "f_stat": (None if ftest.get("F") is None
                               else float(ftest["F"])),
                    "p_value": p_value,
                    "significant": significant,
                    "delta_r_factor": float(simple_fit.r_factor
                                            - fit.r_factor),
                    "n_mm": float(mm_path.degeneracy),
                    "n_mm_err": (None if n_err is None else float(n_err)),
                    "r_mm": float(mm_path.r),
                    "dfn": int(ftest.get("dfn", 0)),
                    "dfd": float(ftest.get("dfd", 0.0)),
                    "method": ftest.get("method"),
                    "simple_r_factor": float(simple_fit.r_factor),
                }
            except XAFSError as exc:
                mm_bond_ftest = {
                    "partner": partner, "f_stat": None, "p_value": None,
                    "significant": False, "delta_r_factor": None,
                    "error": getattr(exc, "message", str(exc))}

    # 形式误差 >1 或 >50% 都说明 N 没被数据约束住（Fe 箔第二壳
    # N=9.45±2.40 是典型：Rf 很好但 N 明显偏高，不能当可靠 CN）。
    n_err_bad = any(
        p.n_err is not None and (p.n_err > 1.0
                                 or p.n_err > 0.5 * p.degeneracy)
        for p in fit.paths)
    # σ² 撞下界不是"振动很小"，而是模型形状与数据失配时 σ² 被反向
    # 压缩的证据（goethite 中距离壳 0.0001–0.0007 Å²）。此时 N 由
    # 振幅补偿撑起来，形式误差可能仍好看，必须独立判不可靠。
    sigma2_bound_bad = any(p.sigma2_at_bound for p in fit.paths)
    c3_bound_bad = any(p.c3_at_bound for p in fit.paths)
    c4_bound_bad = any(p.c4_at_bound for p in fit.paths)
    c3_active_shells = [p.label for p in fit.paths if p.c3 is not None]
    coverage_ok = bool(sid.get("coverage_ok", True))
    norm_obj = getattr(res, "norm", None)
    plateau_val = (float(getattr(norm_obj, "plateau", float("nan")))
                   if norm_obj is not None else float("nan"))
    plateau_bad = bool(
        norm_obj is not None and math.isfinite(plateau_val)
        and not (0.85 <= plateau_val <= 1.15))
    base_reliable = not (fit.overfit or fit.rank_deficient
                         or fit.r_factor > 0.08
                         or abs(fit.delta_e0) >= 15.0 or n_err_bad
                         or sigma2_bound_bad or c3_bound_bad or c4_bound_bad
                         or not coverage_ok
                         or plateau_bad
                         or self_abs["cn_gate"])

    # mixture 权重拟合的可靠性是**相分数**的可靠性，和配位数不是一回事：
    # 权重撞 softmax 边界、误差过大、或两组分在同一壳层不可分辨，都会
    # 让相分数失去物理含义。
    weight_at_bound = False
    weight_err_bad = False
    if fit_weights and fit.mixture_weights:
        weight_at_bound = any(w["weight"] <= 0.01 or w["weight"] >= 0.99
                              for w in fit.mixture_weights)
        weight_err_bad = any(
            (not math.isfinite(w["weight_err"]))
            or w["weight_err"] > 0.15
            or w["weight_err"] > 0.5 * w["weight"]
            for w in fit.mixture_weights)
    weight_degenerate = bool(plan["weight_degenerate_pairs"])
    weights_reliable = None
    if fit_weights:
        weights_reliable = (base_reliable and not weight_at_bound
                            and not weight_err_bad and not weight_degenerate)
    # 权重模式下 N 列是有效简并度 N_crystal·w，不是配位数，故 CN 不成立。
    cn_reliable = base_reliable and not fit_weights
    split_shells = [
        {
            "label": sh.get("label"),
            "sigma2_geom": sh.get("sigma2_geom"),
            "split_span_A": sh.get("split_span_A"),
            "rep_reff_offset_A": sh.get("rep_reff_offset_A"),
            "reff_mean": sh.get("reff_mean"),
        }
        for sh in (plan.get("shells") or [])
        if float(sh.get("split_span_A") or 0.0) >= 0.08
    ]
    flags = {
        "overfit": bool(fit.overfit),
        "rank_deficient": bool(fit.rank_deficient),
        "r_factor_bad": bool(fit.r_factor > 0.08),
        "delta_e0_bad": bool(abs(fit.delta_e0) >= 15.0),
        "n_err_bad": bool(n_err_bad),
        "sigma2_at_bound": bool(sigma2_bound_bad),
        "coverage_bad": bool(not coverage_ok),
        "plateau_bad": bool(plateau_bad),
        # 撞界的壳层标签，报告层可直接列出"哪些壳不可信"
        "sigma2_bound_shells": [p.label for p in fit.paths
                                if p.sigma2_at_bound],
        "weight_at_bound": bool(weight_at_bound),
        "weight_err_bad": bool(weight_err_bad),
        "weight_degenerate": weight_degenerate,
        "weight_degenerate_pairs": plan["weight_degenerate_pairs"],
        "self_absorption_gate": bool(self_abs["cn_gate"]),
        "self_absorption_risk": self_abs["risk"],
        "transmission_thickness_risk": self_abs.get("transmission_thickness_risk"),
        "nidp_scale_factor": getattr(fit, "nidp_scale_factor", None),
        "shell_static_splitting": split_shells,
        "c3_requested": bool(requested_c3),
        "c3_dropped_by_nidp": bool(c3_dropped_by_nidp),
        "c3_at_bound": bool(c3_bound_bad),
        "c3_active_shells": c3_active_shells,
        "c3_bound_shells": [p.label for p in fit.paths if p.c3_at_bound],
        "c4_requested": bool(requested_c4),
        "c4_dropped_by_nidp": bool(c4_dropped_by_nidp),
        "c4_at_bound": bool(c4_bound_bad),
        "c4_active_shells": [p.label for p in fit.paths if p.c4 is not None],
        "c4_bound_shells": [p.label for p in fit.paths if p.c4_at_bound],
    }

    # ---- B17 催化剂精细结构反演：纳米粒径 / 合金短程有序 ----
    nanoparticle_size = None
    alloy_sro = None
    metal_shells = [sh for sh in plan["shells"] if sh.get("scatterer")]
    if (not fit_weights and metal_shells
            and len(metal_shells) == len(plan["shells"])
            and all(sh["scatterer"] == absorber.symbol
                    for sh in metal_shells)):
        lat = ("bcc" if motif.id in ("fe_bcc", "mo_bcc", "cr_bcc")
               else "hcp" if motif.id in ("co_hcp", "zn_hcp") else "fcc")
        n1_bulk = {"fcc": 12.0, "hcp": 12.0, "bcc": 8.0}[lat]
        if float(fit.paths[0].degeneracy) < 0.96 * n1_bulk:
            n2_val = None
            if (len(fit.paths) > 1
                    and plan["shells"][1]["scatterer"] == absorber.symbol):
                n2_val = float(fit.paths[1].degeneracy)
            est = artemis.estimate_nanoparticle_size(
                n1=float(fit.paths[0].degeneracy),
                r1=float(fit.paths[0].r),
                n1_err=(None if fit.paths[0].n_err is None
                        else float(fit.paths[0].n_err)),
                n2=n2_val, lattice=lat)
            if est.get("regime") != "bulk_limit":
                nanoparticle_size = est
    if (str(getattr(motif, "nuclearity", "")) == "alloy"
            and not fit_weights):
        partner = str((motif.source or {}).get("partner") or "")
        aa_paths, ab_paths = [], []
        for sh, p in zip(plan["shells"], fit.paths):
            if sh["scatterer"] == absorber.symbol:
                aa_paths.append(p)
            elif partner and sh["scatterer"] == partner:
                ab_paths.append(p)
        if partner and (aa_paths or ab_paths):
            n_aa = float(sum(p.degeneracy for p in aa_paths))
            n_ab = float(sum(p.degeneracy for p in ab_paths))

            def _quad_err(paths):
                errs = [p.n_err for p in paths if p.n_err is not None]
                if len(errs) != len(paths):
                    return None
                return float(np.sqrt(sum(float(e) ** 2 for e in errs)))

            alloy_sro = artemis.compute_cowley_sro(
                n_aa=n_aa, n_ab=n_ab,
                x_partner=float((motif.source or {}).get("x_partner", 0.5)),
                n_aa_err=_quad_err(aa_paths),
                n_ab_err=_quad_err(ab_paths))
            n_total = n_aa + n_ab
            if n_total < 11.5:
                r_vals = [(float(p.degeneracy), float(p.r))
                          for p in aa_paths + ab_paths]
                r_mean = (sum(n * r for n, r in r_vals) / n_total
                          if n_total > 0 else None)
                if r_mean is not None:
                    est = artemis.estimate_nanoparticle_size(
                        n1=n_total, r1=r_mean,
                        n1_err=_quad_err(aa_paths + ab_paths),
                        lattice=str((motif.source or {}).get(
                            "lattice", "fcc")))
                    if est.get("regime") != "bulk_limit":
                        nanoparticle_size = est

    # ---- B18-1 Rayleigh 分辨极限与轻元素 Z 简并门禁 ----
    res_shells = []
    for pi, p in enumerate(fit.paths):
        sh = plan["shells"][pi] if pi < len(plan["shells"]) else {}
        res_shells.append({"label": p.label,
                           "scatterer": sh.get("scatterer", ""),
                           "r": float(p.r)})
    resolution_check = artemis.check_shell_resolution_and_z_degeneracy(
        res_shells, kmin=res.ft.kmin, kmax=res.ft.kmax)
    flags["sub_rayleigh_z_degeneracy"] = bool(
        not resolution_check["resolvable"])

    # ---- B4-1 增减壳层 F-test：同一 R 窗，仅保留第 1 壳的对照拟合 ----
    # 必须用**同一数据窗口**，否则 χ² 求和的 R 区间不同，F 不成立。
    # 权重拟合与相分数不是配位数问题，不纳入壳层 F-test。
    shell_f_test = None
    if len(plan["shells"]) >= 2 and not fit_weights:
        try:
            simple_fit = artemis.fit_r_space(
                res.chi_k, res.ft, [plan["files"][0]],
                rmin=plan["rmin"], rmax=plan["rmax"],
                kmin=res.ft.kmin, kmax=res.ft.kmax,
                kweight=(tuple(kweights) if kweights else res.ft.kweight),
                s02=absorber.s02, vary_delta_e0=True, vary_n=True,
                path_degen=[plan["degens"][0]],
                path_labels=[plan["labels"][0]],
                fit_space=fit_space)
            shell_f_test = artemis.shell_f_test(simple_fit, fit)
        except XAFSError as exc:
            shell_f_test = {"F": None, "confidence": None,
                            "significant": False,
                            "error": getattr(exc, "message", str(exc))}

    out = {"mode": ("motif_weights" if fit_weights else "motif"),
           "status": sid.get("status"), "motif": motif.id,
           "coverage_ok": sid.get("coverage_ok"), "cn_reliable": cn_reliable,
           "weights_reliable": weights_reliable,
           "mixture_weights": (fit.mixture_weights if fit_weights else None),
           "flags": flags,
           "structure": sid, "rmin": plan["rmin"], "rmax": plan["rmax"],
           "s02": absorber.s02, "s02_source": s02_label,
           "shells": plan["shells"], "fit": fit,
           "fit_space": fit.fit_space,
           "aic": fit.aic, "bic": fit.bic,
           "bvs": fit.bvs, "bvs_target": fit.bvs_target,
           "bvs_chi2": fit.bvs_chi2,
           "truncated_shells": plan["truncated"],
           "ms_links": plan["ms_links"],
           "path_files": [str(f) for f in plan["files"]],
           "path_links": plan.get("path_links") or [],
           "hetero_cocupancy": bool(hetero_cocupancy),
           "hetero_shells": plan.get("hetero_shells") or [],
           "nanoparticle_size": nanoparticle_size,
           "alloy_sro": alloy_sro,
           "resolution_check": resolution_check,
           "mm_bond_ftest": mm_bond_ftest,
           "shell_f_test": shell_f_test,
           "c3_dropped_by_nidp": bool(c3_dropped_by_nidp),
           "c3_at_bound": bool(c3_bound_bad),
           "c4_at_bound": bool(c4_bound_bad),
           "self_absorption": self_abs,
           "cn_systematics": systematic_budget(
               fit, flags=flags, self_abs=self_abs,
               truncated_shells=len(plan["truncated"]),
               fit_weights=fit_weights)}
    if fit_weights and fit.mixture_weights:
        w_max_err = max(float(w["weight_err"])
                        for w in fit.mixture_weights)
        out["weights_systematics"] = {
            "relative_1sigma": round(math.hypot(0.20, w_max_err), 4),
            "components": {"component_model_binding": 0.20,
                           "fitted_weight_error": round(w_max_err, 4)},
            "note": ("相分数误差 = 拟合形式误差与结构模型绑定误差"
                     "（0.20）平方和；组分数由模型选择决定，"
                     "不是独立物相定量。"),
        }
    if sigma2_groups:
        out["sigma2_groups"] = sigma2_groups
        out["sigma2_group_names"] = sigma2_group_names
        out["sigma2_shared"] = True
    else:
        out["sigma2_shared"] = False
    if fit_weights:
        out["warning"] = ("相分数是各组分 χ 的线性组合权重（有效简并度 "
                          "N_crystal·w），N 列不是配位数；组分数与结构模型"
                          "绑定，不是独立物相定量")
    else:
        out["warning"] = ("同壳层路径以代表路径 + 总简并度近似；"
                          "代表路径换一条会改变 F(k)/φ(k)，N 有系统误差")
    if plan["truncated"]:
        out["warning"] += ("；受 N_idp 限制，自动拟合截断了远壳层 "
                           + ", ".join(t["label"] for t in plan["truncated"]))
    if (flags["sub_rayleigh_z_degeneracy"]
            and shared_sigma2 is None and constraints is None):
        out["warning"] += ("；壳层间距低于 Rayleigh 极限且散射元素近简并，"
                           "建议绑定 shared_sigma2/constraints 或用 BVS 消歧")
    if c3_dropped_by_nidp:
        out["warning"] += ("；开启 C3 后自由变量数 ≥ N_idp，"
                           "已按 c3_dropped_by_nidp 自动降级/关闭 C3")
    if c4_bound_bad:
        out["warning"] += ("；C4 达到 ±80% 边界（±1.6e-4 Å⁴），"
                           "非高斯展宽项在吞噬失配，CN 不可靠")
    if isotropic_deltar and fit.r_factor > 0.08:
        out["warning"] += (f"；各向同性晶格约束 Rf={fit.r_factor:.4f} > 0.08："
                           "各壳层位移不服从同一 α·Reff 缩放，"
                           "不要用该约束下的 N 作配位数结论")
    if not sid.get("coverage_ok", True):
        out["warning"] += ("；最优候选未解释部分实验峰，"
                           "结构模型可能不完整，CN 解释需谨慎")
    if self_abs["risk"] in ("high", "unknown", "corrected_unreliable"):
        prefix = ("荧光自吸收校正不可靠" if self_abs["corrected"]
                  else "荧光自吸收未校正")
        out["warning"] += "；" + prefix + "（" + self_abs["note"] + "）"
    elif self_abs["risk"] == "possible":
        out["warning"] += "；荧光自吸收风险（" + self_abs["note"] + "）"
    reasons = []
    if fit.overfit:
        reasons.append("变量数 ≥ 独立点数")
    if fit.rank_deficient:
        reasons.append("参数雅可比矩阵秩亏（不可识别/强共线）")
    if fit.r_factor > 0.08:
        reasons.append(f"Rf={fit.r_factor:.3f}>0.08")
    if abs(fit.delta_e0) >= 15.0:
        reasons.append(f"|ΔE0|={abs(fit.delta_e0):.1f} eV≥15")
    if n_err_bad:
        reasons.append("N 形式误差过大")
    if sigma2_bound_bad:
        reasons.append("σ² 撞下界（形状失配证据"
                       + (f"：{', '.join(flags['sigma2_bound_shells'])}"
                          if flags["sigma2_bound_shells"] else "")
                       + "）")
    if c3_bound_bad:
        reasons.append("C3 撞 ±0.01 Å³ 边界（非谐项在吞噬相位失配"
                       + (f"：{', '.join(flags['c3_bound_shells'])}"
                          if flags["c3_bound_shells"] else "") + "）")
    if c4_bound_bad:
        reasons.append("C4 撞边界（四阶累积量吞噬形状失配）")
    if not coverage_ok:
        reasons.append("覆盖清单未通过/模型不完整")
    if plateau_bad:
        reasons.append(f"边后归一化平台={plateau_val:.3f} 超出 [0.85, 1.15]")
    if self_abs["cn_gate"]:
        if self_abs["corrected"]:
            reasons.append("Booth 自吸收校正不可靠（截断/非有限值）")
        else:
            reasons.append(f"荧光自吸收未校正（Δμ="
                           f"{self_abs['edge_step']:.3g}，"
                           f"risk={self_abs['risk']}）")
    if fit_weights:
        if weight_at_bound:
            reasons.append("组分权重撞 softmax 边界")
        if weight_err_bad:
            reasons.append("组分权重误差过大")
        if weight_degenerate:
            pairs = "; ".join(" vs ".join(p)
                              for p in plan["weight_degenerate_pairs"])
            reasons.append(f"不同组分在同一壳层不可分辨（{pairs}）")
        if not weights_reliable:
            out["warning"] += ("；相分数不可靠（" + "、".join(reasons)
                               + "），不可作为物相定量结论")
    elif not cn_reliable:
        out["warning"] += ("；CN 不可靠（" + "、".join(reasons)
                           + "），不可直接用于配位数结论")
        # 【红线2】拟合指标超出物理范围时唯一合规操作是如实降级。
        # 保留完整多壳层 fit 仅供诊断，但明确：
        #   - 只允许引用第一壳层（退守 L1 语义）；
        #   - 关闭一切 CN 宣称（报表 CN 打码、正式 CN 表拒绝产出）。
        # 绝不通过放宽 Rf / ΔE0 / 评分阈值来"换取"拟合通过。
        out["degraded"] = True
        out["degradation_reasons"] = list(reasons)
        out["quantitative_shells"] = (plan.get("shells") or [])[:1]
        out["cn_claim_allowed"] = False
        out["fit_quantitative"] = False
        out["fit_diagnostic_only"] = True
    if verbose:
        print(f"       ΔE0={fit.delta_e0:+.2f}±{fit.delta_e0_err:.2f} eV  "
              f"Rf={fit.r_factor:.4f}  vars={fit.n_variables}/"
              f"N_idp={fit.n_independent:.1f}"
              + ("" if fit.lattice_strain is None else
                 f"  α={fit.lattice_strain:+.5f}"
                 f"±{fit.lattice_strain_err:.5f}"))
        if fit_weights and fit.mixture_weights:
            print("       相分数（χ 线性组合权重，和=1）：")
            for w in fit.mixture_weights:
                print(f"         {w['group']:<18} w={w['weight']:.4f}"
                      f"±{w['weight_err']:.4f}")
        ntag = "N_eff" if fit_weights else "N"
        for p in fit.paths:
            print(f"       {p.label:<10} {ntag}={p.degeneracy:.2f}"
                  f"±{p.n_err:.2f}" if p.n_err is not None else
                  f"       {p.label:<10} {ntag}={p.degeneracy:.2f}", end="")
            print(f"  R={p.r:.3f} Å  σ²={p.sigma2:.4f}"
                  + (f"  C3={p.c3:+.2e} Å³" if p.c3 is not None else "")
                  + (f"  C4={p.c4:+.2e} Å⁴" if p.c4 is not None else "")
                  + ("  ⚠σ²撞下界" if p.sigma2_at_bound else "")
                  + ("  ⚠C3撞边界" if p.c3_at_bound else "")
                  + ("  ⚠C4撞边界" if p.c4_at_bound else ""))
        if fit_weights and not weights_reliable:
            print(f"       ⚠ 相分数不可靠：{'、'.join(reasons)}")
        elif not cn_reliable:
            print(f"       ⚠ CN 不可靠：{'、'.join(reasons)}")
    if formal_cn and catalytic_model is not None:
        out["s02_calibrated"] = False
        out["diagnostic_only"] = True
        out["formal_cn"] = {
            "status": "hypothesis_model_only",
            "rows": [],
            "reject_reason": (
                "catalytic_model 属于无实验 CIF 的催化位点假设模型，"
                "仅输出诊断 CN，不产出正式 CN"
            ),
            "model": None,
        }
        out["cn_status"] = "hypothesis_model_only"
    else:
        _attach_single_site_formal_cn(
            out, res, absorber=absorber, s02_label=s02_label,
            s02_evidence=s02_evidence,
            formal_cn=formal_cn, structure_model=structure_model,
            gate_result=gate_result, rmin=rmin, rmax=rmax,
            include_ms=include_ms, isotropic_deltar=isotropic_deltar,
            hetero_cocupancy=hetero_cocupancy, is_l1=False,
            verbose=verbose)
    return out


def run(res, identify: bool = True, fit: bool = False,
        absorber_symbol: str | None = None, fit_motif: str | None = None,
        fit_rmin: float | None = None, fit_rmax: float | None = None,
        allow_ambiguous: bool = False, l1_ligand: str = "O",
        l1_geometry: str = "oct", fit_weights: bool = False,
        shared_sigma2: str | list | None = None,
        kweights: tuple | list | None = None,
        include_ms: bool = False,
        vary_c3: bool | list | str = False,
        vary_c4: bool | list | str = False,
        isotropic_deltar: bool = False,
        fit_space: str = "r",
        bvs_restraint: dict | None = None,
        catalytic_model=None,
        hetero_cocupancy: bool = False,
        constraints=None,
        s02: float | None = None,
        s02_source: str | None = None,
        verbose: bool = True,
        formal_cn: bool = False,
        structure_model: StructureModel | None = None,
        gate_result=None,
        xanes_cross_check: bool = False,
        s02_evidence: dict | None = None) -> dict:
    """S6.5（可选 S7）的唯一入口；结果回填 res.structure / res.fit。

    ``s02``（B18-2）为运行时 S₀² 覆盖：仅构造局部 AbsorberConfig 副本，
    不修改 ``absorber.ABSORBERS``；未标定元素需显式给出该值。
    ``xanes_cross_check``（T8）只追加 XANES–EXAFS 软校验警告，
    不改变拟合结果与 ``cn_reliable``。
    """
    out: dict = {}
    if identify or fit:
        # 显式催化模型不需要先跑库搜索：fit_identified 会自建状态。
        if catalytic_model is None:
            out["structure"] = structure_id.identify_result(
                res, absorber_symbol=absorber_symbol,
                l1_ligand=l1_ligand, l1_geometry=l1_geometry,
                vary_c3=vary_c3, fit_space=fit_space, s02=s02,
                verbose=verbose)
            res.structure = out["structure"]
    if fit:
        fit_out = fit_identified(
            res, motif_id=fit_motif, absorber_symbol=absorber_symbol,
            rmin=fit_rmin, rmax=fit_rmax,
            allow_ambiguous=allow_ambiguous,
            l1_ligand=l1_ligand, l1_geometry=l1_geometry,
            fit_weights=fit_weights,
            shared_sigma2=shared_sigma2, kweights=kweights,
            include_ms=include_ms, vary_c3=vary_c3, vary_c4=vary_c4,
            isotropic_deltar=isotropic_deltar,
            fit_space=fit_space, bvs_restraint=bvs_restraint,
            catalytic_model=catalytic_model,
            hetero_cocupancy=hetero_cocupancy,
            constraints=constraints,
            s02=s02,
            s02_source=s02_source,
            s02_evidence=s02_evidence,
            verbose=verbose,
            formal_cn=formal_cn,
            structure_model=structure_model,
            gate_result=gate_result)
        out["fit"] = fit_out
        res.fit = fit_out
        # 催化模型路径绕过库搜索，structure 状态由 fit_identified 自建；
        # 回填外层契约，保证 run 始终导出 structure（报告/落盘依赖该键）。
        if "structure" not in out and isinstance(fit_out, dict):
            out["structure"] = fit_out.get("structure")
            res.structure = out["structure"]
        out["diagnostic_only"] = True
        out["cn_status"] = fit_out.get("cn_status", "diagnostic_only")
        if formal_cn:
            out["formal_cn"] = fit_out.get("formal_cn")
    elif formal_cn:
        out["formal_cn"] = {
            "status": "no_cif",
            "rows": [],
            "reject_reason": "未运行 S7 拟合（fit=False）",
            "model": (structure_model.to_json()
                      if structure_model is not None else None),
        }
        out["diagnostic_only"] = True
        out["cn_status"] = "no_cif"
    if xanes_cross_check:
        cc = xanes_exafs_cross_check(res, fit_out=out.get("fit"))
        out["xanes_cross_check"] = cc
        res.xanes_cross_check = cc
        if isinstance(out.get("fit"), dict):
            out["fit"]["xanes_cross_check"] = cc
        if verbose and cc["warnings"]:
            for w in cc["warnings"]:
                print(f"  [T8] ⚠ {w['code']}: {w['message']}")
    return out


def run_operando_exafs_series(
    results: list,
    absorber_symbol: str,
    motif: str | structure_lib.Motif | None = None,
    catalytic_model=None,
    rmin: float | None = None,
    rmax: float | None = None,
    kmin: float = 3.0,
    kmax: float = 12.0,
    kweight: int = 2,
    s02: float | None = None,
    share_sigma2_across_series: bool = False,
    names: list[str] | None = None,
) -> dict:
    """B19-1：原位序列 EXAFS 联合精修与 N(t)/R(t)/σ²(t) 轨迹。

    全序列共享**单一全局 ΔE₀**（``share_e0=True``），消除逐点 ΔE₀
    漂移经 ΔE₀–ΔR 相关转化成的虚假键长抖动；每个时间点独立放开
    各壳层 N/ΔR。``share_sigma2_across_series=True`` 时同一壳层在
    全序列共享 σ²，显著压低单点 N(t) 的形式误差。
    """
    if not results or len(results) < 3:
        raise XAFSConfigError(
            "OPERANDO", f"run_operando_exafs_series 至少需要 3 个时间点，"
                        f"收到 {0 if not results else len(results)}")
    if names is not None and len(names) != len(results):
        raise XAFSConfigError(
            "OPERANDO", f"names 长度 {len(names)} != 时间点数 {len(results)}")

    absorber = (get_absorber(absorber_symbol) if s02 is None
                else structure_id.absorber_with_s02(absorber_symbol, s02))
    if absorber.s02 is None:
        raise XAFSConfigError(
            "OPERANDO",
            f"{absorber.symbol} 未标定 S₀²；原位 EXAFS 序列联合精修需要 "
            f"显式 s02（0.4–1.3）{_t5_s02_hint(absorber.symbol)}")

    if catalytic_model is not None:
        target = structure_lib.parse_catalytic_cluster_spec(
            absorber.symbol, catalytic_model)
    elif motif is not None:
        if isinstance(motif, structure_lib.Motif):
            target = motif
            if str(target.absorber).capitalize() != absorber.symbol.capitalize():
                raise XAFSConfigError(
                    "OPERANDO", f"motif 吸收元素 {target.absorber!r} 与 "
                                f"absorber_symbol {absorber.symbol!r} 不一致")
        else:
            target = _find_motif(absorber.symbol, str(motif))
    else:
        sid = structure_id.identify_result(
            results[0], absorber_symbol=absorber.symbol, s02=s02,
            verbose=False)
        target = _find_motif(absorber.symbol, sid["ranked"][0]["id"])

    plan = select_shell_paths(
        target, edge=absorber.edge, rmin=rmin, rmax=rmax,
        kmin=kmin, kmax=kmax)
    labels = list(plan["labels"])
    n_points = len(results)
    names_list = (list(names) if names is not None
                  else [f"point{i}" for i in range(n_points)])

    datasets = []
    for di, res in enumerate(results):
        datasets.append({
            "name": names_list[di],
            "chi_k": res.chi_k,
            "ft": res.ft,
            "datfiles": list(plan["files"]),
            "rmin": float(plan["rmin"]),
            "rmax": float(plan["rmax"]),
            "kmin": float(kmin),
            "kmax": float(kmax),
            "kweight": int(kweight),
            "s02": float(absorber.s02),
            "vary_n": True,
            "path_degen": list(plan["degens"]),
            "path_labels": labels,
        })

    shared_bonds = None
    if share_sigma2_across_series:
        shared_bonds = []
        for pi in range(len(plan["files"])):
            for di in range(1, n_points):
                shared_bonds.append({
                    "target": (di, pi),
                    "source": (0, pi),
                    "share_deltar": False,
                    "share_sigma2": True,
                    "degen_ratio": None,
                })

    multi_out = artemis.fit_multi_datasets(
        datasets, shared_bonds=shared_bonds, share_e0=True)

    trajectories = {}
    for pi, label in enumerate(labels):
        trajectories[label] = {
            "reff": float(plan["shells"][pi]["reff"]),
            "N": [float(f.paths[pi].degeneracy)
                  for f in multi_out["datasets"]],
            "N_err": [f.paths[pi].n_err for f in multi_out["datasets"]],
            "R": [float(f.paths[pi].r) for f in multi_out["datasets"]],
            "R_err": [f.paths[pi].delta_r_err
                      for f in multi_out["datasets"]],
            "sigma2": [float(f.paths[pi].sigma2)
                       for f in multi_out["datasets"]],
            "sigma2_err": [f.paths[pi].sigma2_err
                           for f in multi_out["datasets"]],
        }

    return {
        "names": names_list,
        "motif": target.id,
        "shells": labels,
        "global_r_factor": float(multi_out["global_r_factor"]),
        "shared_delta_e0": float(multi_out["datasets"][0].delta_e0),
        "shared_delta_e0_err": multi_out["datasets"][0].delta_e0_err,
        "r_factors": [float(f.r_factor) for f in multi_out["datasets"]],
        "trajectories": trajectories,
        "multi_fit": multi_out,
    }
