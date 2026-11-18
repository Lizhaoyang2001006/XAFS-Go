# -*- coding: utf-8 -*-
"""fit_shell_plan.py — #1 shell 选择 select_shell_paths、SS/MS/异核行构造、P2b N0-shell 映射与信息探测包装；并承载跨子模块共享的常量与纯辅助函数（自 xafs_fit.py 纯拆分）

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


# ---- 共享常量与纯辅助函数（自原文件逐行搬移）----
# 形式误差 ≠ 绝对精度。N 的误差棒只来自最小二乘协方差，
# 不包含 S₀² 传递、代表路径/壳层合并、N–σ²–ΔE₀ 相关和归一化偏差。
# 按 Fe 基准（Fe 箔第二壳 9.45 vs 晶体学 6；lepidocrocite Fe–Fe 7.6 vs 6）
# 实测，CN 的绝对系统误差约 ±20–40%（1σ，量级估计），跨体系比较时
# 不应把 ±0.3 当成“配位数已精确到小数第二位”。
CN_SYSTEMATIC_BUDGET = {
    "relative_1sigma": 0.30,
    "components": {
        "S02_transfer": 0.15,
        "representative_path_and_shell_merge": 0.25,
        "N_sigma2_e0_correlation": 0.15,
        "normalization": 0.10,
    },
    "note": ("n_err 是形式误差；CN 绝对值的系统误差约 ±20–40%，"
             "报告时应注明预算，不能把形式误差当绝对精度。"),
}

# 荧光自吸收：没有样品组成/厚度就无法做可验证的定量校正。
# 这里只做保守筛查：荧光模式下边跳变越大，表观振幅被压低越严重。
# 阈值是**风险门禁**，不是校正因子；高风险时拒绝 CN/相分数结论。
SELF_ABSORPTION_HIGH_EDGE_STEP = 1.0   # Δμ ≥ 1.0：浓样/厚样，必须校正后才定量
SELF_ABSORPTION_WARN_EDGE_STEP = 0.3   # 0.3–1.0：可能受影响，只警告

def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None

def _t5_s02_hint(symbol: str) -> str:
    """若该元素已有 T5 独立标定产物，返回 CLI 提示后缀。"""
    t5_cal = (Path(__file__).resolve().parents[3] / "data"
              / "s02_calibration" / symbol / "calibration.json")
    return ("（该元素已有 T5 独立标定值，CLI 可加 --s02-from-t5 或显式传 --s02）"
            if t5_cal.is_file() else "")


def _find_motif(absorber: str, motif_id: str) -> structure_lib.Motif:
    matches = [motif for motif in structure_lib.load_library(absorber)
               if motif.id == motif_id]
    if len(matches) > 1:
        raise XAFSConfigError(
            "FIT", f"结构库 {absorber} 中存在重复 motif id={motif_id!r}（{len(matches)} 条）")
    if len(matches) == 1:
        return matches[0]
    raise XAFSConfigError(
        "FIT", f"结构库 {absorber} 中不存在 motif {motif_id!r}")


def _ss_rows(motif: structure_lib.Motif, edge: str) -> list:
    """返回 motif 的全部单散射路径摘要（已按 Reff 排序）。"""
    cache = structure_lib.ensure_feff(motif, edge=edge)
    rows = []
    for f in sorted(cache.glob("feff*.dat")):
        path = artemis.load_feff_path(f)
        if int(getattr(path, "nleg", 2)) != 2:
            continue
        geom = getattr(path, "geom", None) or []
        scatterer = geom[-1][0] if geom else "?"
        rows.append({"file": f, "reff": float(path.reff),
                     "degen": float(path.degen), "scatterer": scatterer})
    rows.sort(key=lambda r: r["reff"])
    # ---- B7-2 部分占据：按距离/元素匹配壳层平均占据度缩放简并度 ----
    # 只对确有 occ<0.999 或直接引用外部 CIF 的晶体 motif 做展开，
    # 避免全占据体系每次识别都重复展开晶胞。
    needs_occ = (motif.kind == "crystal"
                 and (motif.cif_path is not None
                      or any(float(s.get("occ", 1.0)) < 0.999
                             for s in (motif.asym_sites or []))))
    if needs_occ:
        try:
            scale = artemis.cif_shell_occupancy_scale(
                structure_lib._cif_path(motif), motif.absorber, rmax=6.0)
        except XAFSError as exc:
            raise CNError(
                "occupancy_scale_failed",
                f"{motif.id}: 无法展开 CIF 占据度；拒绝按满占据继续"
                f"（{getattr(exc, 'message', str(exc))}）") from exc
        if not scale:
            raise CNError(
                "occupancy_scale_failed",
                f"{motif.id}: CIF 占据度展开为空；拒绝按满占据继续")
        for r in rows:
            matches = [occ for (dist, el, occ) in scale
                       if el == r["scatterer"]
                       and abs(dist - float(r["reff"])) <= 0.05]
            if matches:
                occ_mean = float(np.mean(matches))
                if abs(occ_mean - 1.0) > 1e-12:
                    r["degen"] = float(r["degen"]) * occ_mean
                    r["occupancy_scale"] = occ_mean
    return rows


def _collinear_ms_rows(motif: structure_lib.Motif, edge: str,
                       rmax: float, min_abs_cos: float = 0.85,
                       min_reff_ratio: float = 1.5,
                       min_bridge_cos: float | None = None) -> list:
    """[Implements: EQ-07] 筛选近共线多散射（MS）路径：nleg∈{3,4}、reff≤rmax、前向聚焦内角。

    物理扩展（P-04）：
      1. 区分四腿同原子相干往返路径（M → L1 → M → L1 → M，pts[1]==pts[3] 且
         pts[2]==pts[0]）：两次散射在同一配体上，热位移完全相干，
         δR_eff = 2·δr1 ⟹ σ²_MS = 4·σ²_1 = (Reff / r1)² · σ²_1；
         异原子对称链（L1-M-L2）与三腿路径保持线性标度 (Reff / r1) · σ²_1。
      2. 可选 ``min_bridge_cos``（如 0.65 对应桥角 ≥ 130°）：对 nleg=3 桥连
         前向散射路径（如 M1-O-M2），允许桥原子顶点夹角钝角（cos ≤ -min_bridge_cos），
         同时要求回散射腿满足共线聚焦条件。
    """
    cache = structure_lib.ensure_feff(motif, edge=edge)
    rows = []
    for f in sorted(cache.glob("feff*.dat")):
        path = artemis.load_feff_path(f)
        nleg = int(getattr(path, "nleg", 2))
        if nleg not in (3, 4):
            continue
        reff = float(path.reff)
        if reff > rmax:
            continue
        geom = getattr(path, "geom", None) or []
        if len(geom) < nleg:
            continue
        try:
            pts = np.asarray([[float(g[4]), float(g[5]), float(g[6])]
                              for g in geom[:nleg]], dtype=float)
        except (TypeError, ValueError, IndexError):
            continue
        # 闭合路径：末原子 → 吸收原子（GEOM 不重复列吸收原子）。
        closed = np.vstack([pts, pts[0]])
        cosines = []
        for i in range(1, len(closed) - 1):
            v1 = closed[i - 1] - closed[i]
            v2 = closed[i + 1] - closed[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 <= 1e-8 or n2 <= 1e-8:
                cosines = []
                break
            cosines.append(float(np.dot(v1, v2) / (n1 * n2)))
        if not cosines:
            continue
        if nleg == 3 and min_bridge_cos is not None and len(cosines) == 2:
            # 桥连三腿路径 M1->O->M2->M1：桥原子(i=1)钝角前向聚焦 cos <= -min_bridge_cos，
            # 远端散射原子(i=2)满足常规共线条件 |cos| >= min_abs_cos
            if cosines[0] > -float(min_bridge_cos) or abs(cosines[1]) < min_abs_cos:
                continue
        elif min(abs(c) for c in cosines) < min_abs_cos:
            continue
        first_dist = float(np.linalg.norm(pts[1] - pts[0]))
        # 聚焦判据：Reff ≈ 2·R_first（nleg=3 的三体 180° 路径与 nleg=4
        # 的往返路径）。首腿本身等于 Reff 的“长腿”路径（如 Ni–Ni–O）
        # 不是聚焦路径，不纳入受约束 MS。
        if first_dist <= 0 or reff < min_reff_ratio * first_dist:
            continue
        reff_ratio = float(reff / first_dist)
        coherent_rattle = bool(
            nleg == 4
            and float(np.linalg.norm(pts[2] - pts[0])) < 0.05
            and float(np.linalg.norm(pts[1] - pts[3])) < 0.05
        )
        sigma2_ratio = float(reff_ratio ** 2) if coherent_rattle else reff_ratio
        bridge_angle_deg = float(np.degrees(np.arccos(np.clip(cosines[0], -1.0, 1.0))))
        try:
            path_id = int(f.stem.replace("feff", ""))
        except ValueError:
            path_id = 0
        rows.append({
            "file": f, "reff": reff, "degen": float(path.degen),
            "nleg": nleg, "scatterer": str(geom[-1][0]),
            "first_el": str(geom[1][0]), "first_dist": first_dist,
            "sequence": "-".join(str(g[0]) for g in geom[:nleg]),
            "reff_ratio": reff_ratio,
            "sigma2_ratio": sigma2_ratio,
            "coherent_rattle": coherent_rattle,
            "bridge_angle_deg": round(bridge_angle_deg, 2),
            "path_id": path_id,
        })
    rows.sort(key=lambda r: (r["reff"], -r["degen"]))
    return rows


def _mixture_rows(motif: structure_lib.Motif, edge: str, merge_tol: float,
                  apply_weights: bool = True):
    """展开 SS 路径；mixture 则按权重缩放简并度并收紧合并容差。

    返回 (rows, merge_tol_effective)。普通 motif 原样透传；
    mixture 的 χ 是各位点 χ 的加权线性组合，对应到路径层就是
    把每个分量的 degen 乘以该位点权重，再做**同一**窗口拟合。

    apply_weights=False 用于**权重拟合**：保留晶体学简并度、给每条
    路径打上 ``group``（组分下标）与 ``group_name``，由 fit_r_space
    在迭代中按 N_crystal·w 缩放——否则权重会被预先乘死，无法再拟合。
    """
    if motif.kind != "mixture":
        return _ss_rows(motif, edge), merge_tol
    lib = {m.id: m for m in structure_lib.load_library(motif.absorber)}
    rows = []
    for ci, comp in enumerate(motif.components):
        w = float(comp["weight"])
        for r in _ss_rows(lib[comp["id"]], edge):
            d = dict(r)
            d["group"] = ci
            d["group_name"] = comp["id"]
            if apply_weights:
                d["degen"] = float(r["degen"]) * w
            rows.append(d)
    rows.sort(key=lambda r: r["reff"])
    # 两吸收位点的近距壳（磁铁矿 Fe-O 1.886 / 2.060）只差 0.17 Å，
    # 单位点容差 0.20 会把它们错误合并；物理分辨 π/(2Δk) 允许用
    # 更紧的容差分开子壳层。
    return rows, min(merge_tol, 0.10)


def _hetero_branch_row(motif: structure_lib.Motif, edge: str,
                       element: str, distance: float) -> dict:
    """为异核共占位分支生成一条单散射 FEFF 路径（缓存到 _hetero_cache）。

    做法：把共占位位点纯化为该分支元素的单个近邻原子跑一次 FEFF，
    取该距离处的 nleg==2 路径；简并度由占位数在调用方覆盖。
    """
    element = str(element).capitalize()
    distance = float(distance)
    cache = (structure_lib.HETERO_CACHE
             / f"{motif.id}_{element}_{distance:.3f}")
    if not list(cache.glob("feff*.dat")):
        inp = artemis.build_feff_input(
            [(distance, 0.0, 0.0, element)], absorber=motif.absorber,
            rmax=distance + 0.8,
            title=f"{motif.id}_{element}_{distance:.2f}", edge=edge)
        artemis.run_feff6l(inp, cache)
    best = None
    for f in sorted(cache.glob("feff*.dat")):
        p = artemis.load_feff_path(f)
        if int(getattr(p, "nleg", 2)) != 2:
            continue
        if abs(float(p.reff) - distance) > 0.25:
            continue
        if best is None or (abs(float(p.reff) - distance)
                            < abs(float(best[0].reff) - distance)):
            best = (p, f)
    if best is None:
        raise XAFSDataError(
            "FIT", f"异核分支 {motif.absorber}-{element}@{distance:.2f} Å "
                   f"在 FEFF 缓存 {cache} 中无 SS 路径")
    return {"file": best[1], "reff": float(best[0].reff),
            "degen": float(best[0].degen), "scatterer": element}


def _hetero_shells_in_window(motif: structure_lib.Motif,
                             rmin: float, rmax: float) -> list:
    """返回窗口内需要拆分的异核共占位壳层（single_element_approx=False）。"""
    if not motif.cif_path:
        raise XAFSConfigError(
            "FIT", f"{motif.id}: hetero_cocupancy 需要 motif.cif_path "
                   f"（含 _atom_site_occupancy 的共占位 CIF）")
    hetero = artemis.cif_hetero_cocupancy_branches(
        motif.cif_path, motif.absorber, rmax=max(4.0, rmax + 1.5))
    out = []
    for si, sh in enumerate(hetero.get("shells") or []):
        if sh.get("single_element_approx"):
            continue
        if not (rmin - 0.5 <= float(sh["distance"]) <= rmax + 0.5):
            continue
        item = dict(sh)
        item["shell_index"] = si
        out.append(item)
    return out


def _replace_hetero_rows(motif: structure_lib.Motif, edge: str,
                         rows: list, hetero_shells: list) -> list:
    """用每分支 FEFF 路径替换共占位壳层的单元素近似行。"""
    out = list(rows)
    for sh in hetero_shells:
        distance = float(sh["distance"])
        if not math.isfinite(distance) or distance <= 0:
            raise XAFSDataError(
                "FIT", f"{motif.id}: 异核共占位壳层距离 {distance!r} 非物理")
        els = {str(b["element"]).capitalize() for b in sh["branches"]}
        out = [r for r in out
               if not (str(r["scatterer"]).capitalize() in els
                       and abs(float(r["reff"]) - distance) <= 0.25)]
        for bi, branch in enumerate(sh["branches"]):
            degen_val = float(branch["degen"])
            if not math.isfinite(degen_val) or degen_val <= 0:
                raise XAFSDataError(
                    "FIT", f"{motif.id}: 异核共占位分支 {branch.get('element')} "
                           f"简并度 {degen_val!r} 非物理")
            row = _hetero_branch_row(
                motif, edge, str(branch["element"]).capitalize(), distance)
            row["degen"] = degen_val
            row["hetero_group"] = int(sh["shell_index"])
            row["hetero_branch"] = int(bi)
            row["hetero_n_geom"] = int(sh["n_geom"])
            out.append(row)
    return out


def select_shell_paths(motif: structure_lib.Motif, edge: str = "K",
                       rmin: float | None = None, rmax: float | None = None,
                       merge_tol: float = 0.20, max_paths: int = 12,
                       kmin: float = 3.0, kmax: float = 12.0,
                       group_by_component: bool = False,
                       include_ms: bool = False, max_ms: int = 3,
                       isotropic_deltar: bool = False,
                       hetero_cocupancy: bool = False,
                       n_mode: str = "free",
                       min_bridge_cos: float | None = None) -> dict:
    """[Implements: EQ-04] 按 (散射元素, Reff) 分壳层，返回代表路径 + 总简并度。

    ``merge_tol`` 默认 0.20 Å，仍小于 π/(2Δk)（Δk=6 时约 0.26 Å）——
    比 0.15 Å 更符合“不可分辨壳层”的物理判据：针铁矿 Fe-O
    1.95×3 + 2.10×3（ΔR=0.15 Å）应合并成一层 N=6，否则两条 N
    强相关、误差爆炸。代表路径取该壳层中简并度最大的一条，N 初值
    取壳层总简并度。

    自动选壳层还必须服从信息量上限：每条路径放开 N/ΔR/σ² 是 3 个
    变量，加 1 个全局 ΔE0，故 ``3n+1`` 必须小于独立点数 N_idp。
    超出时从**最远壳层**开始截断（近壳信噪比高、相关弱），并在
    ``truncated`` 里返回被丢弃的壳层，避免静默过拟合。

    kind=mixture（多位点吸收）：各分量 SS 路径按权重缩放简并度后
    统一分组，并把合并容差收紧到 ≤0.10 Å——磁铁矿 8a/16d 第一壳
    1.886 与 2.060 Å 仅差 0.17 Å，默认 0.20 容差会错误合并成一层。
    """
    if group_by_component and motif.kind != "mixture":
        raise XAFSConfigError(
            "FIT", f"{motif.id} 不是 mixture，group_by_component 仅用于多位点混合相")
    # 权重拟合必须保留**未乘权重**的晶体学简并度，否则 N·w 被预先乘死。
    rows, merge_tol = _mixture_rows(
        motif, edge, merge_tol, apply_weights=not group_by_component)
    if not rows:
        raise XAFSDataError("FIT", f"{motif.id}: 没有可用单散射路径")
    if rmin is None:
        rmin = max(1.0, min(r["reff"] for r in rows) - 0.6)
    if rmax is None:
        # 默认止步 3.5 Å：覆盖第一壳 + 常见第二壳（Fe 箔 2.87、
        # Fe–Fe 3.06），但不把 3.6–4.4 Å 的弱远壳全部放进自动拟合。
        # 需要第三壳时显式 --fit-rmax 3.9，并自行承担过拟合风险。
        rmax = min(3.5, max(r["reff"] for r in rows) + 0.3)

    # ---- B13-2 异核共占位：把单元素近似拆成每元素一条分支路径 ----
    hetero_shells = []
    if hetero_cocupancy:
        hetero_shells = _hetero_shells_in_window(
            motif, rmin=float(rmin), rmax=float(rmax))
        if hetero_shells:
            rows = _replace_hetero_rows(motif, edge, rows, hetero_shells)

    kept = [r for r in rows if rmin - 0.5 <= r["reff"] <= rmax + 0.5]
    if not kept:
        raise XAFSConfigError(
            "FIT", f"{motif.id}: R∈[{rmin:.2f},{rmax:.2f}] 内无 SS 路径")
    groups = []
    for row in kept:
        row_key = row.get("group") if group_by_component else None
        # 同一 (散射元素, 组分) 内按距离容差合并；不同组分永不合并，
        # 否则磁铁矿 tet/oct 的权重会被并进同一条路径而不可辨识。
        match = None
        for g in groups:
            if g["scatterer"] != row["scatterer"]:
                continue
            if group_by_component and g["group"] != row_key:
                continue
            if abs(row["reff"] - g["reff_mean"]) <= merge_tol:
                if (match is None
                        or abs(row["reff"] - g["reff_mean"])
                        < abs(row["reff"] - match["reff_mean"])):
                    match = g
        if match is not None:
            g = match
            old_n = g["degen"]
            new_n = old_n + row["degen"]
            g["reff_mean"] = ((g["reff_mean"] * old_n
                               + row["reff"] * row["degen"]) / new_n)
            g["degen"] = new_n
            g["members"].append(row)
            if row["degen"] > g["rep"]["degen"]:
                g["rep"] = row
            continue
        groups.append({"scatterer": row["scatterer"], "reff_mean": row["reff"],
                       "degen": row["degen"], "rep": row,
                       "members": [row],
                       "group": row_key,
                       "group_name": (row.get("group_name")
                                      if group_by_component else None)})
    # 逐行插入可能打乱距离顺序；截断策略依赖"从最远壳开始丢"。
    groups.sort(key=lambda g: g["reff_mean"])

    # 只保留中心落在拟合窗口内的壳层；窗口外的弱路径不参与自动拟合。
    groups = [g for g in groups
              if rmin - 0.20 <= g["reff_mean"] <= rmax + 0.15]
    if not groups:
        raise XAFSConfigError(
            "FIT", f"{motif.id}: R∈[{rmin:.2f},{rmax:.2f}] 内无有效壳层")

    n_idp = artemis.independent_points(kmin, kmax, rmin, rmax)
    if isotropic_deltar:
        # ΔR 已压缩为唯一全局 α：每壳只剩 N、σ² 两个变量，加 α、ΔE₀
        # 两个全局变量，信息量上限为 2n + 2 < N_idp。
        max_by_idp = max(1, int((n_idp - 2.0) // 2.0))
    elif n_mode == "fixed_n":
        max_by_idp = max(1, int((n_idp - 1.0) // 2.0))
    elif n_mode == "conditional":
        max_by_idp = max(1, int((n_idp - 2.0) // 2.0))
    else:
        max_by_idp = max(1, int((n_idp - 1.0) // 3.0))
    limit = min(max_paths, max_by_idp)
    truncated = groups[limit:]
    groups = groups[:limit]

    files, degens, labels, shells = [], [], [], []
    for g in groups:
        el = g["scatterer"]
        # 用代表距离做唯一标签，避免 "Fe-O2" 让人误以为是第二壳；
        # 多壳层表里真正重要的是 Reff/N/R，而不是序号。
        base = f"{motif.absorber}-{el}@{g['reff_mean']:.2f}"
        label = (f"{g['group_name']}:{base}" if group_by_component else base)
        files.append(g["rep"]["file"])
        degens.append(g["degen"])
        labels.append(label)
        # P-06：定量计算被合并子壳层的静态几何劈裂方差 σ²_geom、跨度与代表路径重心偏移
        members = g["members"]
        tot_w = sum(float(m["degen"]) for m in members)
        r_mean = float(g["reff_mean"])
        if tot_w > 0 and len(members) > 1:
            s2_geom = sum(float(m["degen"]) * (float(m["reff"]) - r_mean) ** 2
                          for m in members) / tot_w
            r_vals = [float(m["reff"]) for m in members]
            split_span = max(r_vals) - min(r_vals)
        else:
            s2_geom = 0.0
            split_span = 0.0
        rep_offset = float(g["rep"]["reff"]) - r_mean
        shells.append({"label": label, "scatterer": el,
                       "reff": round(g["rep"]["reff"], 4),
                       "reff_mean": round(r_mean, 4),
                       "degen": g["degen"], "n_members": len(members),
                       "sigma2_geom": round(float(s2_geom), 6),
                       "split_span_A": round(float(split_span), 4),
                       "rep_reff_offset_A": round(float(rep_offset), 4),
                       "group": (g["group_name"] if group_by_component
                                 else None)})

    # ---- 权重拟合所需的组分映射 ----
    # fit_r_space 要求 path_groups 是 0..K-1 连续下标，而组分可能因
    # R 窗/信息量截断缺失，故重映射并同时给出原始组分下标。
    path_groups = None
    group_names = None
    component_ids = None
    weight_degenerate_pairs = []
    if group_by_component:
        used = sorted({int(g["group"]) for g in groups})
        remap = {old: i for i, old in enumerate(used)}
        path_groups = [remap[int(g["group"])] for g in groups]
        component_ids = used
        group_names = []
        for old in used:
            name = next(g["group_name"] for g in groups
                        if int(g["group"]) == old)
            group_names.append(name)
        # 不同组分在同一 (散射元素, 距离) 上出现时，只有振幅之和可辨识，
        # 单组分权重严格简并——必须显式标记，不能给出"相分数"。
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ga, gb = groups[a], groups[b]
                if (int(ga["group"]) != int(gb["group"])
                        and ga["scatterer"] == gb["scatterer"]
                        and abs(ga["reff_mean"] - gb["reff_mean"])
                        <= merge_tol):
                    weight_degenerate_pairs.append(
                        [f"{ga['group_name']}@{ga['reff_mean']:.2f}",
                         f"{gb['group_name']}@{gb['reff_mean']:.2f}"])

    # ---- 受约束多散射：追加近共线 nleg>2 路径，不新增自由变量 ----
    ms_links = []
    if include_ms:
        for row in _collinear_ms_rows(motif, edge, rmax + 0.15,
                                      min_bridge_cos=min_bridge_cos):
            parent = None
            for gi, g in enumerate(groups):
                if g["scatterer"] != row["first_el"]:
                    continue
                if abs(g["reff_mean"] - row["first_dist"]) <= 0.05:
                    parent = gi
                    break
            if parent is None:
                continue
            idx = len(files)
            # 标签必须带完整原子序列：同一 Reff 下 Ni–O–O 与
            # Ni–O–Ni–O 是不同路径，只写末原子会产生重名。
            label = (f"{row.get('sequence', motif.absorber + '-' + row['scatterer'])}"
                     f"@{row['reff']:.2f}(MS{row['nleg']}"
                     f"#{row.get('path_id', 0)})")
            files.append(row["file"])
            degens.append(row["degen"])
            labels.append(label)
            reff_ratio = float(row["reff"] / groups[parent]["reff_mean"])
            s2_ratio = (float(reff_ratio ** 2)
                        if row.get("coherent_rattle") else reff_ratio)
            shells.append({"label": label, "scatterer": row["scatterer"],
                           "reff": round(row["reff"], 4),
                           "reff_mean": round(row["reff"], 4),
                           "degen": row["degen"], "n_members": 1,
                           "group": None, "kind": "ms",
                           "parent": labels[parent],
                           "coherent_rattle": bool(row.get("coherent_rattle")),
                           "bridge_angle_deg": row.get("bridge_angle_deg")})
            ms_links.append({
                "ms_path_index": idx,
                "parent_shell_index": parent,
                "reff_ratio": reff_ratio,
                "sigma2_ratio": s2_ratio,
                "degen_ratio": float(row["degen"]
                                     / groups[parent]["degen"]),
                "nleg": int(row["nleg"]),
                "coherent_rattle": bool(row.get("coherent_rattle")),
            })
            if len(ms_links) >= max_ms:
                break

    # ---- B13-2 双分支绑定：同壳层不同元素分支共享 ΔR/σ²，各自放开 N ----
    path_links = []
    if hetero_cocupancy and hetero_shells:
        by_group: dict[int, list[int]] = {}
        for gi, g in enumerate(groups):
            hg = g["rep"].get("hetero_group")
            if hg is None:
                continue
            by_group.setdefault(int(hg), []).append(gi)
        for _hg, idxs in sorted(by_group.items()):
            idxs.sort(key=lambda gi: int(
                groups[gi]["rep"].get("hetero_branch", 0)))
            src = idxs[0]
            for tgt in idxs[1:]:
                path_links.append({"target": int(tgt), "source": int(src),
                                   "share_deltar": True,
                                   "share_sigma2": True,
                                   "degen_ratio": None})

    return {"files": files, "degens": degens, "labels": labels,
            "shells": shells, "rmin": float(rmin), "rmax": float(rmax),
            "merge_tol": merge_tol, "n_independent": round(n_idp, 2),
            "max_paths_by_idp": max_by_idp, "ms_links": ms_links,
            "groups": path_groups, "group_names": group_names,
            "component_ids": component_ids,
            "weight_degenerate_pairs": weight_degenerate_pairs,
            "hetero_shells": [{"distance": float(sh["distance"]),
                               "n_geom": int(sh["n_geom"]),
                               "branches": [dict(b) for b in sh["branches"]]}
                              for sh in hetero_shells],
            "path_links": path_links,
            "truncated": [{"label": (f"{g['group_name']}:{motif.absorber}-"
                                     f"{g['scatterer']}@{g['reff_mean']:.2f}"
                                     if group_by_component else
                                     f"{motif.absorber}-{g['scatterer']}"
                                     f"@{g['reff_mean']:.2f}"),
                           "reff_mean": round(g["reff_mean"], 4),
                           "degen": g["degen"]} for g in truncated]}


def _normalize_c3_paths(vary_c3, n_path: int):
    """把 CLI/API 的 vary_c3 归一成 artemis 接受的 None / True / [idx...]。"""
    if vary_c3 is None or vary_c3 is False:
        return None
    if vary_c3 is True:
        return True
    if isinstance(vary_c3, str):
        key = vary_c3.strip().lower()
        if key in ("", "none", "false", "off"):
            return None
        if key == "first":
            return [0]
        if key == "all":
            return True
        raise XAFSConfigError(
            "FIT", f"vary_c3 字符串只支持 'first'/'all'，收到 {vary_c3!r}")
    if isinstance(vary_c3, (list, tuple, set)):
        idx = sorted({int(i) for i in vary_c3})
        for i in idx:
            if i < 0 or i >= n_path:
                raise XAFSConfigError(
                    "FIT", f"vary_c3 路径下标 {i} 超出范围 0–{n_path - 1}")
        return idx or None
    raise XAFSConfigError(
        "FIT", f"vary_c3 必须是 bool 或路径下标列表，收到 {vary_c3!r}")


def _s02_calibrated_flag(absorber, s02_source: str | None,
                         s02_evidence: dict | None = None) -> bool:
    """Formal provenance only; runtime AbsorberConfig.calibrated stays unchanged."""
    import hashlib
    import json
    from absorber import resolve_s02_spec
    source = "absorber.py" if s02_source is None else str(s02_source)
    value = getattr(absorber, "s02", None)
    if isinstance(value, bool) or _num(value) is None:
        return False
    if source == "absorber.py":
        original = get_absorber(absorber.symbol)
        return (original.calibrated and absorber.edge == original.edge
                and value == original.s02)
    if source.startswith("chemical_state:"):
        try:
            resolved, label = resolve_s02_spec(
                absorber.symbol, source[len("chemical_state:"):])
        except XAFSConfigError:
            return False
        return resolved is not None and value == resolved and source == label
    # Manual/config overrides cannot inherit another source's evidence.
    if source in ("--s02 runtime override", "config:s02"):
        return False
    evidence = (s02_evidence if s02_evidence is not None
                else getattr(s02_source, "s02_evidence", None))
    if not isinstance(evidence, dict) or not isinstance(evidence.get("payload"), dict):
        return False
    payload = evidence["payload"]
    try:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(serialized).hexdigest() != evidence.get("payload_sha256"):
            return False
        if (payload.get("absorber") != absorber.symbol
                or payload.get("edge") != absorber.edge
                or isinstance(payload.get("s02"), bool)
                or _num(payload.get("s02")) is None or payload["s02"] != value):
            return False
        if "certificate_file" in evidence or "certificate_file_sha256" in evidence:
            path = Path(evidence["certificate_file"])
            raw = path.read_bytes()
            if (not path.is_absolute()
                    or hashlib.sha256(raw).hexdigest() != evidence.get("certificate_file_sha256")
                    or json.loads(raw.decode("utf-8")) != payload):
                return False
        # Displayed certificate hashes must describe this same evidence, not a prefix claim.
        for kind, claimed in re.findall(
                r"(?<![A-Za-z0-9_])(sha256|certificate_sha256|payload_sha256):([^;#]+)", source):
            expected = evidence.get("certificate_file_sha256" if kind == "sha256"
                                    else "payload_sha256")
            if (not isinstance(expected, str) or len(claimed) not in (12, 64)
                    or claimed != expected[:len(claimed)]):
                return False
    except (KeyError, TypeError, ValueError, OSError, UnicodeError):
        return False
    import calibrate_s02
    qualified, _reasons = calibrate_s02.certificate_qualifies(payload)
    return qualified


def map_n0_to_plan(plan: dict, n0_shells: list[dict],
                   kmin: float = 3.0, kmax: float = 12.0) -> dict:
    """把 CIFGateResult 的 n0_shells 映射到 plan["files"] 的每条 FEFF 路径。

    匹配规则：用 (scatterer, reff) **就近**匹配 n0_shells 中的壳层；容差
    0.5 Å。CIF 门按 Rayleigh 分辨率 π/(2Δk) 把不可分辨子壳合并成一个 N₀
    壳（N₀=Σ占据度）；当 FEFF 计划把这些子壳拆成多条路径时，**同一 N₀
    壳的总配位数按各路径晶体学简并度比例分摊**（组内 Σn0_per_path = 壳
    N₀，守恒），并用 path_links 按简并度比绑定子壳——绝不把整壳 N₀ 重复
    计给每条子壳路径（旧版就近独占匹配会令子壳映射失败或错配远壳）。
    MS 路径不独立匹配，按 ms_links 从母壳继承 N₀。

    返回 dict：
        mapping_ok: bool          是否成功映射全部路径
        mapping_reason: str       失败原因（ok 时为空）
        n0_per_path: list[float]  每条路径的 N₀（用于 path_degen）
        path_links: list[dict]    同组多条路径的 degen_ratio 绑定
        n_var_fixed_n: int        固定 N 变量数 = 2n + 1
        n_var_free_n_first: int   自由 N（首壳）变量数 = 2n + 2
        n_paths: int              路径总数
    """
    n_paths = len(plan["files"])
    plan_shells = plan.get("shells", [])
    if len(plan_shells) != n_paths:
        return {
            "mapping_ok": False,
            "mapping_reason": (f"plan shells({len(plan_shells)}) != "
                               f"files({n_paths}), 无法一一映射"),
        }

    plan_degens = plan.get("degens", [])
    n0_per_path = [0.0] * n_paths
    shell_idx_per_path = [-1] * n_paths

    # ---- SS 路径就近匹配 N₀ 壳（不再独占：子壳可共享同一 N₀ 壳）----
    ss_idxs = [pi for pi, sh in enumerate(plan_shells)
               if sh.get("kind") != "ms"]
    for pi in ss_idxs:
        sh = plan_shells[pi]
        scatterer = str(sh.get("scatterer", "")).capitalize()
        reff = float(sh.get("reff", sh.get("reff_mean", 0.0)))
        best_si = None
        best_dist = float("inf")
        for si, n0_sh in enumerate(n0_shells):
            if str(n0_sh.get("el", "")).capitalize() != scatterer:
                continue
            dist = abs(float(n0_sh.get("r", 0.0)) - reff)
            if dist < best_dist and dist < 0.5:
                best_dist = dist
                best_si = si
        if best_si is None:
            return {
                "mapping_ok": False,
                "mapping_reason": (f"路径 {pi} ({scatterer}@{reff:.3f} Å) "
                                   f"无匹配 N₀ 壳层"),
            }
        if float(n0_shells[best_si]["n0"]) < 0:
            return {
                "mapping_ok": False,
                "mapping_reason": (f"N₀ 壳层 {best_si} ({scatterer}) 配位数 "
                                   f"{float(n0_shells[best_si]['n0'])} 为负，非物理"),
            }
        shell_idx_per_path[pi] = best_si

    # ---- 同一 N₀ 壳内的子壳：按晶体学简并度比例分摊整壳 N₀（守恒）----
    groups_by_shell: dict[int, list[int]] = {}
    for pi in ss_idxs:
        groups_by_shell.setdefault(shell_idx_per_path[pi], []).append(pi)
    for si, idxs in groups_by_shell.items():
        shell_n0 = float(n0_shells[si]["n0"])
        if len(idxs) == 1:
            n0_per_path[idxs[0]] = shell_n0
            continue
        degen_sum = sum(float(plan_degens[pi]) for pi in idxs)
        if degen_sum <= 0:
            return {
                "mapping_ok": False,
                "mapping_reason": (f"N₀ 壳层 {si} 的子壳简并度总和 "
                                   f"{degen_sum} ≤ 0，无法分摊配位数"),
            }
        for pi in idxs:
            n0_per_path[pi] = shell_n0 * float(plan_degens[pi]) / degen_sum

    # ---- MS 路径：从母壳继承 N₀（× ms_links 简并度比），不独立匹配 ----
    ms_by_index = {int(m["ms_path_index"]): m
                   for m in plan.get("ms_links", [])}
    label_to_idx = {sh.get("label"): pi
                    for pi, sh in enumerate(plan_shells)}
    for pi, sh in enumerate(plan_shells):
        if sh.get("kind") != "ms":
            continue
        link = ms_by_index.get(pi)
        if link is not None:
            pj = int(link["parent_shell_index"])
            ratio = float(link["degen_ratio"])
        else:
            pj = label_to_idx.get(sh.get("parent"), -1)
            ratio = (float(plan_degens[pi]) / float(plan_degens[pj])
                     if 0 <= pj < n_paths and float(plan_degens[pj]) > 0
                     else 1.0)
        if not (0 <= pj < n_paths) or shell_idx_per_path[pj] < 0:
            return {
                "mapping_ok": False,
                "mapping_reason": f"MS 路径 {pi} 的母壳未映射，无法继承 N₀",
            }
        n0_per_path[pi] = n0_per_path[pj] * ratio
        shell_idx_per_path[pi] = shell_idx_per_path[pj]

    # ---- path_links：只绑定同一个 N₀ 壳层内的 SS 子壳 ----
    # 不同 N₀ 壳层是不同配位壳，绝不能互相绑定；否则远壳 N 会跟着
    # 首壳 N 漂移，违反“首壳放开、远壳固定 N₀”的条件 CN 约定。
    # MS 路径由 ms_links 约束，不参与子壳比例绑定。
    path_links: list[dict] = []
    for si, idxs in groups_by_shell.items():
        if len(idxs) <= 1:
            continue
        idxs.sort(key=lambda pi: float(plan_shells[pi].get("reff", 0.0)))
        ref = idxs[0]
        ref_n0 = n0_per_path[ref]
        for tgt in idxs[1:]:
            tgt_n0 = n0_per_path[tgt]
            ratio = (tgt_n0 / ref_n0) if ref_n0 > 0 else 1.0
            path_links.append({
                "target": int(tgt),
                "source": int(ref),
                "degen_ratio": round(ratio, 6),
                "share_deltar": False,
                "share_sigma2": False,
            })

    n_var_fixed_n = 2 * n_paths + 1      # 每壳 ΔR + σ²，加全局 ΔE₀
    n_var_free_n_first = 2 * n_paths + 2 # 首壳多放开一个 N

    return {
        "mapping_ok": True,
        "mapping_reason": "",
        "n0_per_path": [round(v, 6) for v in n0_per_path],
        "path_links": path_links,
        "n_var_fixed_n": n_var_fixed_n,
        "n_var_free_n_first": n_var_free_n_first,
        "n_paths": n_paths,
    }


# ================================================================
# P2b：select_shell_paths 信息量模式扩展
# ================================================================
_orig_select_shell_paths = select_shell_paths


def select_shell_paths(motif, edge="K", rmin=None, rmax=None,
                       merge_tol=0.20, max_paths=12, kmin=3.0, kmax=12.0,
                       group_by_component=False, include_ms=False,
                       max_ms=3, isotropic_deltar=False,
                       hetero_cocupancy=False,
                       n_mode: str = "free",
                       min_bridge_cos: float | None = None):
    """select_shell_paths 的包装，支持 P2b 固定 N 信息量模式。

    n_mode:
        "free"      诊断/自由 N（默认）：3n+1 < N_idp
        "fixed_n"   固定 N：2n+1 < N_idp（每壳 ΔR+σ²，加 ΔE₀）
        "conditional" 条件 CN：2n+2 < N_idp（首壳多放开一个 N）
    """
    if n_mode not in ("free", "fixed_n", "conditional"):
        raise XAFSConfigError(
            "FIT", f"n_mode 必须是 free/fixed_n/conditional，收到 {n_mode!r}")
    plan = _orig_select_shell_paths(
        motif, edge=edge, rmin=rmin, rmax=rmax, merge_tol=merge_tol,
        max_paths=max_paths, kmin=kmin, kmax=kmax,
        group_by_component=group_by_component, include_ms=include_ms,
        max_ms=max_ms, isotropic_deltar=isotropic_deltar,
        hetero_cocupancy=hetero_cocupancy, n_mode=n_mode,
        min_bridge_cos=min_bridge_cos)
    if n_mode == "free":
        return plan
    # 重算信息量上限
    n_idp = plan["n_independent"]
    n_paths = len(plan["files"])
    if n_mode == "fixed_n":
        max_by_idp = max(1, int((n_idp - 1.0) // 2.0))
    else:  # conditional
        max_by_idp = max(1, int((n_idp - 2.0) // 2.0))
    limit = min(max_paths, max_by_idp)
    plan["n_paths_fixed_n"] = n_paths
    plan["n_var_fixed_n"] = 2 * n_paths + 1
    plan["n_var_conditional"] = 2 * n_paths + 2
    # 截截断：按新上限重新截断
    groups_count = len(plan["files"])
    if groups_count > limit:
        plan["truncated"] = plan.get("truncated", []) + [
            {"label": f"truncated_by_{n_mode}", "note":
             f"信息量上限 {n_mode}: {groups_count} > {limit}"}]
        plan["files"] = plan["files"][:limit]
        plan["degens"] = plan["degens"][:limit]
        plan["labels"] = plan["labels"][:limit]
        plan["shells"] = plan["shells"][:limit]
    return plan
