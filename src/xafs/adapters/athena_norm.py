# -*- coding: utf-8 -*-
"""athena_norm.py — # 2 E0 确定 + # 3 归一化、MBACK、XANES 特征与分峰（自 athena.py 纯拆分）

纯搬运拆分：函数体与原 athena.py 逐行一致，不含任何逻辑改动。
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


def find_e0_derivative_max(energy, mu, lo, hi, smooth_eV: float = 2.0) -> float:
    """
    一阶导数最大值处的能量。

    数学:
        E0 = argmax_E  dμ/dE     ,  E ∈ [lo, hi]

    实现要点:
        ① 先重采样到 0.1 eV 再求导，避免原始不等间距导致的偏大导数
        ② 平滑用 uniform_filter1d(mode="nearest")。
           **不能用 np.convolve(mode="same")**——它在边界补零，
           会把窗口起点做成一个虚假的巨大正导数峰（实测会把 E0 判到窗口下沿）。
    """
    from scipy.ndimage import uniform_filter1d

    grid = np.arange(lo, hi, 0.1)
    resampled = np.interp(grid, energy, mu)
    size = max(int(round(smooth_eV / 0.1)), 1)
    if size > 1:
        resampled = uniform_filter1d(resampled, size=size, mode="nearest")
    derivative = np.gradient(resampled, grid)
    return float(grid[int(np.argmax(derivative))])


def find_e0_half_height(energy, mu, e0_guess: float, window_eV: float = 80.0) -> float:
    """
    归一化谱 μ_norm = 0.5 处的能量（half-height 定义）。

    数学:
        μ_norm(E0) = 0.5,   其中 μ_norm = (μ − y_pre) / Δμ

    **必须先归一化再找 0.5**。μ 的绝对值由样品厚度与稀释度决定：
    γ-FeOOH 的 μ 全程在 1.36–2.42，直接判 `mu > 0.5` 会在第一个点就成立，
    插值一路外推到 9593 eV（实测）。
    与归一化的循环依赖用两段式消除：先用导数峰做初值归一化，再取半高点。
    """
    from larch.xafs import pre_edge

    provisional = Group(energy=energy, mu=mu)
    pre_edge(provisional, e0=e0_guess, pre1=-150, pre2=-30,
             norm1=150, norm2=600, nnorm=2)

    mask = ((provisional.energy >= e0_guess - window_eV)
            & (provisional.energy <= e0_guess + window_eV))
    e = np.asarray(provisional.energy)[mask]
    y = np.asarray(provisional.norm)[mask]
    if len(e) < 4:
        return float(e0_guess)

    diff = y - 0.5
    crossings = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    if len(crossings) == 0:
        return float(e0_guess)
    i = int(crossings[0])
    return float(e[i] + (0.5 - y[i]) * (e[i + 1] - e[i]) / (y[i + 1] - y[i]))


def classify_reference_channel(scan) -> dict:
    """
    由采集信息（通道拓扑/角色）判定参考通道能否独立用于能量校准。

    串联电离室拓扑（标准透射 + 参考箔）:
        I0 → 样品 → It → 参考箔 → Ir
        μ_sample(E) = ln(I0/It)
        μ_ref(E)    = ln(It/Ir)     ← 参考箔自身吸收，与样品边无关

    独立性判据全部来自采集信息，绝不靠「样品边位 − 参考边位 > 3 eV」反推：
    旧做法用被样品吸收污染的 ln(I0/Ir)=μ_sample+μ_foil 找边，再拿边位差
    判独立，是循环论证（参考边≈样品边 → 永远判非独立）。改为:
        ① 必须同时有 It 与 Ir：只有 I0/Ir 而无 It 时，Ir 实为样品透射
           探测器，ln(I0/Ir) 就是样品 μ，不存在独立参考箔。
        ② Ir 不能是 It 的常数比例复制：Ir/It 整段恒定 ⟺ 两 chamber 之间
           没有吸收箔 ⟺ 参考信号不独立（采集层信号来源判据，非边位判据）。
        ③ 文件头若显式标注参考与样品同一吸收体，直接判非独立。

    返回 dict: present / topology / independent / mu_ref / reason / ref_element
    """
    meta = scan.meta or {}
    out = {"present": scan.iref is not None, "topology": "absent",
           "independent": False, "mu_ref": None,
           "reason": "无参考通道（Ir）",
           "ref_element": meta.get("reference_element")}
    if scan.iref is None:
        return out
    if scan.it is None:
        out["topology"] = "two_chamber_no_it"
        out["reason"] = ("只有 I0/Ir 而无 It：Ir 为样品透射探测器，"
                         "ln(I0/Ir) 即样品 μ，非独立参考箔")
        return out

    it = np.asarray(scan.it, dtype=float)
    iref = np.asarray(scan.iref, dtype=float)
    out["topology"] = "series_3ch"
    if meta.get("reference_same_absorber"):
        out["reason"] = "文件头标注参考与样品为同一吸收体，非独立参考"
        return out
    if np.any(it <= 0) or np.any(iref <= 0):
        out["reason"] = "It/Ir 含非正值，无法计算参考箔吸收 ln(It/Ir)"
        return out
    ratio = iref / it
    rel_var = float(np.ptp(ratio) / max(abs(float(np.median(ratio))), 1e-12))
    if rel_var < 1e-6:
        out["reason"] = "Ir 与 It 成常数比例（无箔吸收变化），参考通道不独立"
        return out

    out["mu_ref"] = np.log(it / iref)
    out["independent"] = True
    out["reason"] = ("串联三 chamber 拓扑 I0→样品→It→参考箔→Ir，"
                     "μ_ref=ln(It/Ir) 隔离参考箔吸收")
    return out


def determine_e0(scan, method: str = "derivative", manual_e0: float | None = None,
                 guess: float | None = None) -> dict:
    """
    判定样品的吸收边位置，并同时给出参考通道的边位供交叉校验。

    method:
        "derivative"   一阶导数峰（默认，最常用）
        "half_height"  归一化半高点（可把 ΔE0 压低约 3 eV）
        "manual"       手工指定，带范围校验

    返回字典含:
        e0_sample            用于 k 轴零点
        e0_reference         参考通道边位，只用于判断刻度漂移，不参与 k 轴
        e0_derivative_peak   导数法结果（half_height 模式下作为对照保留）
    """
    if scan.mu is None:
        raise XAFSDataError("S2", "缺少 μ(E)，无法判定 E0")
    energy = np.asarray(scan.energy)
    mu = np.asarray(scan.mu)

    nominal = guess or scan.meta.get("e0_nominal")
    if nominal is None:
        raise XAFSConfigError(
            "S2", "无法确定吸收边搜索窗口：文件头无 e0_nominal，也未给吸收元素"
                  "（absorber 查表）或边能量 guess——元素信息是 S2 的必需输入")
    lo, hi = nominal - 30, nominal + 50

    out = {"e0_nominal": scan.meta.get("e0_nominal"), "e0_guess": nominal}
    e0_deriv = find_e0_derivative_max(energy, mu, lo, hi)
    out["e0_derivative_peak"] = e0_deriv

    method = (method or "derivative").lower()
    if method == "manual":
        if manual_e0 is None:
            raise XAFSConfigError("S2", "method='manual' 但未给出 manual_e0")
        if not (energy.min() <= manual_e0 <= energy.max()):
            raise XAFSConfigError(
                "S2", f"manual_e0={manual_e0} 超出数据范围 "
                      f"{energy.min():.0f}–{energy.max():.0f}")
        out["e0_sample"] = float(manual_e0)
        out["e0_method"] = "manual"
    elif method == "half_height":
        out["e0_sample"] = find_e0_half_height(energy, mu, e0_deriv)
        out["e0_method"] = "half_height"
    elif method == "derivative":
        out["e0_sample"] = e0_deriv
        out["e0_method"] = "derivative"
    else:
        raise XAFSConfigError(
            "S2", f"未知 method={method}，可选 derivative / half_height / manual")

    if scan.iref is not None:
        ref = classify_reference_channel(scan)
        out["reference_topology"] = ref["topology"]
        out["reference_independent"] = ref["independent"]
        if ref["ref_element"] is not None:
            out["reference_element"] = ref["ref_element"]
        if ref["independent"]:
            # μ_ref=ln(It/Ir)：参考箔自身吸收，边位与样品边无关
            e0_ref = find_e0_derivative_max(energy, ref["mu_ref"], lo, hi)
            out["e0_reference"] = e0_ref
            out["chemical_shift"] = out["e0_sample"] - e0_ref
        else:
            out["e0_reference"] = None
            out["reference_warning"] = ref["reason"]
    return out


# ================================================================
# 3. 归一化
# ================================================================

def fit_linear_pre_edge(energy, mu, e0: float, pre_window=(-150, -30)):
    """
    边前线性基线。

    数学:
        y_pre(E) = a·E + b,   a,b 由 [E0+pre_lo, E0+pre_hi] 内最小二乘确定

    返回 (a, b, y_pre 数组)
    """
    mask = ((energy >= e0 + pre_window[0]) & (energy <= e0 + pre_window[1]))
    if mask.sum() < 4:
        raise XAFSDataError(
            "S3", f"边前仅 {int(mask.sum())} 点，不足以做线性拟合；"
                  f"请放宽 pre_window 或检查数据起点")
    slope, intercept = np.polyfit(energy[mask], mu[mask], 1)
    return float(slope), float(intercept), slope * energy + intercept


def compute_edge_step(energy, mu, pre_line, e0: float, post_window=(150, 600)) -> float:
    """
    吸收阶跃高度。

    数学:
        Δμ = median[ μ(E) − y_pre(E) ] ,  E ∈ [E0+post_lo, E0+post_hi]

    用中位数而非均值，避免残余 EXAFS 振荡拉偏。
    """
    mask = ((energy >= e0 + post_window[0]) & (energy <= e0 + post_window[1]))
    if mask.sum() < 4:
        raise XAFSDataError(
            "S3", f"边后仅 {int(mask.sum())} 点，无法确定边跳变")
    return float(np.median(mu[mask] - pre_line[mask]))


def normalize_absorption(scan, e0: float, pre_window=(-150, -30),
                         post_window=(150, 600), engine: str = "larch") -> NormalizedScan:
    """
    归一化吸收谱。

    数学:
        μ_norm(E) = (μ(E) − y_pre(E)) / Δμ

    物理:
        扣掉边前基线以消除样品厚度与基底吸收；再以 Δμ 归一，
        使不同浓度样品的边跳变都等于 1，从而可横向比较。
    """
    energy = np.asarray(scan.energy)
    mu = np.asarray(scan.mu)

    if engine == "larch":
        # 与 Athena 一致：边后区域用二次多项式拟合，外推到 E0 得 edge_step。
        # native 模式改用中位数法，边后平台更接近 1（实测 1.002 vs 0.912），
        # 但 edge_step 会小约 10%，与 Athena 的数字对不上。
        from larch.xafs import pre_edge
        g = Group(energy=energy, mu=mu)
        pre_edge(g, e0=e0, pre1=pre_window[0], pre2=pre_window[1],
                 norm1=post_window[0], norm2=post_window[1], nnorm=2)
        if (g.edge_step is None or not np.isfinite(float(g.edge_step))
                or float(g.edge_step) <= 0):
            raise XAFSDataError(
                "S3", f"larch pre_edge 给出 Δμ = {g.edge_step!r} ≤ 0 或非有限，"
                      f"边跳变方向异常，检查数据或改用 engine='native'")
        plateau_mask = ((energy > e0 + 200)
                        & (energy < e0 + min(post_window[1], 600)))
        plateau = (float(np.mean(np.asarray(g.norm)[plateau_mask]))
                   if np.any(plateau_mask) else float("nan"))
        # Larch 的 g.flat 已经扣掉二次边后背景，但外推到远区的残差会让
        # 平台偏离 1（LEPIDO 实测 0.925）。Athena 的 flattened 显示约定
        # 是平台中心≈1，这里按 [E0+100, E0+400] 均值归一。
        flat = np.asarray(getattr(g, "flat", g.norm), dtype=float).copy()
        flat = _renormalize_flat_plateau(energy, flat, e0)
        return NormalizedScan(energy=np.asarray(g.energy), mu=np.asarray(g.mu),
                              mu_norm=np.asarray(g.norm),
                              pre_edge_line=np.asarray(g.pre_edge),
                              edge_step=float(g.edge_step),
                              plateau=plateau, e0=float(e0),
                              flattened=flat)

    _, _, pre_line = fit_linear_pre_edge(energy, mu, e0, pre_window)
    edge_step = compute_edge_step(energy, mu, pre_line, e0, post_window)
    if edge_step <= 0:
        raise XAFSDataError("S3", f"Δμ = {edge_step:.4g} ≤ 0，边跳变方向异常")
    mu_norm = (mu - pre_line) / edge_step

    plateau_mask = (energy > e0 + 200) & (energy < e0 + min(post_window[1], 600))
    plateau = float(np.mean(mu_norm[plateau_mask])) if plateau_mask.any() else np.nan

    flat = _flatten_post_edge(energy, mu_norm, e0, post_window)
    return NormalizedScan(energy=energy, mu=mu, mu_norm=mu_norm,
                          pre_edge_line=pre_line, edge_step=edge_step,
                          plateau=plateau, e0=float(e0), flattened=flat)


def normalize_mback(scan, e0: float, elem: str, edge: str = "K",
                    pre_window=(-200, -30), post_window=(60, 150),
                    fit_order: int = 2) -> NormalizedScan:
    """B10-1 短边后窗口的 Cromer–Liberman（MBACK）归一化。

    传统 ``normalize_absorption`` 在边后仅 60–150 eV 可用时，二次多项式
    外推会把平台拉偏（实测可差 10% 以上）。MBACK 改用一个**理论截面**
    做振幅标尺，只拟合一个比例因子和一条低阶基线：

        μ_meas(E) ≈ a·μ_th(E) + b0 + b1·(E−E0) + …

    其中 μ_th 来自 ``xraydb.mu_elam``（Cromer–Liberman 参数化）。
    归一化:

        μ_norm(E) = (μ_meas(E) − B(E)) / (a·Δμ_th)
        Δμ_th = μ_th(E0+50) − μ_th(E0−50)

    边界条件（不伪造）:
      - 边前/边后窗口各至少 3 点，否则 ``XAFSDataError``；
      - ``a <= 0`` 或 ``a·Δμ_th <= 0``：理论与实测边跳方向矛盾，拒绝。

    ``fit_order`` 是基线多项式阶数（1=常数，2=线性，默认 2）。
    """
    energy = np.asarray(scan.energy, dtype=float)
    mu = np.asarray(scan.mu, dtype=float)
    if energy.ndim != 1 or energy.shape != mu.shape:
        raise XAFSDataError(
            "S3", f"MBACK 输入长度不一致: E={energy.shape}, μ={mu.shape}")
    if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(mu))):
        raise XAFSDataError("S3", "MBACK 输入 energy 或 μ 含非有限值（NaN/Inf）")
    e0 = float(e0)
    elem = str(elem)
    mu_th = np.asarray(mu_elam(elem, energy), dtype=float)
    if energy.size < 8 or not np.all(np.isfinite(mu_th)):
        raise XAFSDataError("S3", f"MBACK 理论截面不可用（元素 {elem}）")

    pre = ((energy >= e0 + float(pre_window[0]))
           & (energy <= e0 + float(pre_window[1])))
    post = ((energy >= e0 + float(post_window[0]))
            & (energy <= e0 + float(post_window[1])))
    if int(pre.sum()) < 3 or int(post.sum()) < 3:
        raise XAFSDataError(
            "S3", f"MBACK 窗口点数不足: 边前 {int(pre.sum())}、"
                  f"边后 {int(post.sum())}（各需 ≥3）")

    mask = pre | post
    x = energy[mask] - e0
    order = max(1, int(fit_order))
    # 设计矩阵：[μ_th, 1, ΔE, ΔE², ...]，基线阶数 = order
    cols = [mu_th[mask]] + [x ** p for p in range(order)]
    design = np.column_stack(cols)
    coef, _, rank, _ = np.linalg.lstsq(design, mu[mask], rcond=None)
    if rank < design.shape[1]:
        raise XAFSDataError(
            "S3", f"MBACK 设计矩阵秩亏（rank={rank}/{design.shape[1]}），"
                  f"窗口内理论截面与多项式共线")
    a = float(coef[0])
    base_coef = np.asarray(coef[1:], dtype=float)
    baseline = np.zeros_like(energy)
    for p, c in enumerate(base_coef):
        baseline += c * (energy - e0) ** p

    if not np.isfinite(a) or a <= 0:
        raise XAFSDataError(
            "S3", f"MBACK 振幅因子 a={a:.4g} ≤ 0：实测谱与 {elem} 理论截面"
                  f"边跳方向不一致，拒绝归一化")

    jump_pair = np.asarray(mu_elam(elem, np.array([e0 + 50.0, e0 - 50.0])),
                           dtype=float)
    jump = float(jump_pair[0] - jump_pair[1])
    edge_step = a * jump
    if not np.isfinite(edge_step) or edge_step <= 0:
        raise XAFSDataError(
            "S3", f"MBACK 边跳 a·Δμ_th={edge_step:.4g} ≤ 0，拒绝归一化")

    mu_norm = (mu - baseline) / edge_step
    plateau_mask = ((energy >= e0 + float(post_window[0]))
                    & (energy <= e0 + float(post_window[1])))
    plateau = (float(np.mean(mu_norm[plateau_mask]))
               if plateau_mask.any() else float("nan"))
    flat = _flatten_post_edge(
        energy, mu_norm, e0,
        (float(post_window[0]), max(float(post_window[1]), 300.0)))
    mback = {
        "a": a,
        "b": float(base_coef[0]) if base_coef.size else 0.0,
        "baseline_coeffs": [float(c) for c in base_coef],
        "theory_edge_step": float(jump),
        "edge_step": float(edge_step),
        "post_window": [float(post_window[0]), float(post_window[1])],
        "pre_window": [float(pre_window[0]), float(pre_window[1])],
        "element": elem,
        "edge": str(edge),
        "fit_order": order,
    }
    return NormalizedScan(energy=energy, mu=mu, mu_norm=mu_norm,
                          pre_edge_line=baseline, edge_step=float(edge_step),
                          plateau=plateau, e0=e0, flattened=flat,
                          engine="mback", mback=mback)


def _renormalize_flat_plateau(energy, flat, e0: float):
    """把 flattened 谱的边后平台中心归一到 1.0（E<E0 不动）。"""
    energy = np.asarray(energy, dtype=float)
    out = np.asarray(flat, dtype=float).copy()
    sel = energy >= e0
    for lo, hi in ((100.0, 400.0), (30.0, 600.0)):
        mask = (energy >= e0 + lo) & (energy <= e0 + hi)
        if int(mask.sum()) >= 5:
            scale = float(np.mean(out[mask]))
            if np.isfinite(scale) and scale > 1e-6:
                out[sel] = out[sel] / scale
            break
    return out


def _flatten_post_edge(energy, mu_norm, e0: float,
                       post_window=(150, 600)):
    """native 分支配平：边后二次背景扣除 + 平台归一。

    E≥E0 时 ``flat = μ_norm − [P(E) − P(E0)]``，P 是拟合到 μ_norm 的
    二次多项式；E<E0 保持 μ_norm。之所以对 μ_norm 而非原始 μ 拟合，
    是因为原始 μ 的二次外推到 E0 会把平台抬到 1.5 左右（实测），
    与 Athena flattened 的“平台≈1”约定不符。
    """
    energy = np.asarray(energy, dtype=float)
    mu_norm = np.asarray(mu_norm, dtype=float)
    out = mu_norm.copy()
    sel = energy >= e0
    mask = ((energy >= e0 + post_window[0])
            & (energy <= e0 + post_window[1]))
    if int(mask.sum()) >= 5:
        coef = np.polyfit(energy[mask], mu_norm[mask], 2)
        post = np.polyval(coef, energy)
        base = float(np.polyval(coef, e0))
        out[sel] = mu_norm[sel] - (post[sel] - base)
    return _renormalize_flat_plateau(energy, out, e0)


def fit_whiteline_area(norm: NormalizedScan,
                       window: tuple[float, float] = (-6.0, 40.0)) -> dict:
    """B9-2 白线净面积：arctan 阶跃 + Pseudo-Voigt。

    K 边白线常与边前峰/主边台阶叠在一起；L2/L3 边没有 1s→3d 四极
    边前峰，主特征就是白线。统一模型:

        μ(E) = b0 + b1(E−E0)
               + h·[1/2 + arctan((E−E0)/w)/π]      （主边阶跃）
               + PV(E; A, E_wl, γ, η)               （白线，A=积分面积）

    拟合后 ``A`` 即扣掉阶跃后的净白线面积（μ_norm 已按边跳=1 归一，
    故单位是 eV；``white_line_area`` 乘回 ``edge_step`` 得到原始 μ·eV）。
    窗口点数 <8 或拟合失败时返回 ``reliable=False`` 与零面积，不伪造。
    """
    from scipy.optimize import least_squares

    energy = np.asarray(norm.energy, dtype=float)
    y_all = (np.asarray(norm.flattened, dtype=float)
             if norm.flattened is not None
             else np.asarray(norm.mu_norm, dtype=float))
    e0 = float(norm.e0)
    emin, emax = e0 + float(window[0]), e0 + float(window[1])
    sel = (energy >= emin) & (energy <= emax)
    E, y = energy[sel], y_all[sel]
    out = {
        "white_line_area": 0.0,
        "white_line_area_norm": 0.0,
        "whiteline_peak_height": 0.0,
        "white_line_energy": None,
        "r_factor": None,
        "reliable": False,
        "window_eV": [float(emin), float(emax)],
        "note": "",
    }
    if E.size < 8:
        out["note"] = f"白线窗口仅 {E.size} 点（<8），安全降级"
        return out

    y_max = float(np.max(y))
    y_min = float(np.min(y))
    span = max(y_max - y_min, 1e-9)
    edge0 = float(np.median(y[:max(3, E.size // 10)]))
    # 初值：阶跃高度 ≈ 远端平台 − 近端基线；白线中心取窗口内最大值。
    tail = float(np.median(y[-max(3, E.size // 10):]))
    h0 = max(tail - edge0, 0.05 * span)
    i_max = int(np.argmax(y))
    A0 = max(0.2 * span * 3.0, 1e-6)
    x0 = np.array([edge0, 0.0, h0, 1.5, A0, float(E[i_max]), 1.5, 0.5])

    def _model(p):
        b0, b1, h, w, A, ec, gam, eta = p
        step = h * (0.5 + np.arctan((E - e0) / max(w, 1e-6)) / np.pi)
        return (b0 + b1 * (E - e0) + step
                + _pseudo_voigt_area(E, A, ec, gam, eta))

    lb = [-np.inf, -np.inf, 0.0, 0.2, 0.0, e0 - 8.0, 0.3, 0.0]
    ub = [np.inf, np.inf, 3.0 * span, 15.0, 50.0 * span * max(emax - emin, 1.0),
          e0 + 20.0, 10.0, 1.0]
    try:
        sol = least_squares(lambda p: _model(p) - y, np.clip(x0, lb, ub),
                            bounds=(lb, ub), max_nfev=40000)
        p = np.asarray(sol.x, dtype=float)
        resid = _model(p) - y
    except Exception as exc:  # noqa: BLE001 - 拟合失败必须可解释地降级
        out["note"] = f"白线拟合失败: {exc}"
        return out

    b0, b1, h, w, A, ec, gam, eta = [float(v) for v in p]
    if not (np.all(np.isfinite(p)) and A >= 0 and h >= 0):
        out["note"] = "白线拟合得到非物理参数（面积/阶跃为负或非有限）"
        return out
    pv_at_center = float(_pseudo_voigt_area(
        np.array([ec]), A, ec, gam, eta)[0])
    denom = float(np.sum((y - np.mean(y)) ** 2))
    out.update({
        "white_line_area": float(A * float(norm.edge_step)),
        "white_line_area_norm": float(A),
        "whiteline_peak_height": pv_at_center,
        "white_line_energy": ec,
        "r_factor": (float(np.sum(resid ** 2) / denom)
                     if denom > 0 else None),
        "reliable": True,
        "note": "",
        "edge_step_fit": h,
        "edge_width_eV": w,
    })
    return out


def extract_xanes_features(norm: NormalizedScan,
                           e0_nominal: float | None = None,
                           edge: str = "K",
                           absorber: str | None = None) -> dict:
    """B2-1 XANES 定量特征 + 颈部肩峰/导数双峰指纹。

    输入优先用 ``norm.flattened``（边后展平谱）；缺失时退回 ``mu_norm``。
    所有能量窗口以本次样品的 E0 为锚点，输出为 JSON 可序列化的标量。

    颈部判据（对标 Athena 的 derivative doublet 预警）：
      1. 在上升沿 E∈[E0−5, E_WL] 且 μ_flat∈[0.15, 0.85] 内算平滑一阶导；
      2. 找 ≥2 个局部极大值（间距 ≥1.5 eV，次峰 ≥0.35×主峰）；
      3. 次峰 ≥0.70×主峰 → ``e0_jump_risk=True``：E0 可能在两个台阶间
         跳变 3–6 eV，后续归一化/FT 相位会随之漂移。
    """
    energy = np.asarray(norm.energy, dtype=float)
    e0 = float(norm.e0)
    mu_flat = norm.flattened if norm.flattened is not None else norm.mu_norm
    mu_flat = np.asarray(mu_flat, dtype=float)
    if energy.shape != mu_flat.shape:
        raise XAFSConfigError(
            "S3", f"XANES 特征提取长度不一致: E={energy.shape}, μ={mu_flat.shape}")

    out = {
        "e_half_height": None,
        "half_height_shift": None,
        "white_line_energy": None,
        "white_line_height": None,
        "pre_edge_peak_energy": None,
        "pre_edge_peak_height": 0.0,
        "pre_edge_area": 0.0,
        "pre_edge_centroid": None,
        "pre_edge_voigt_area": 0.0,
        "pre_edge_n_peaks": 0,
        "pre_edge_voigt_r_factor": None,
        "neck_shoulder_present": False,
        "neck_shoulder_energy": None,
        "derivative_split_eV": 0.0,
        "e0_jump_risk": False,
        "derivative_peaks_eV": [],
        "derivative_peak_heights": [],
    }

    # ---- 白线（flattened 谱）----
    wl_mask = (energy >= e0 - 2.0) & (energy <= e0 + 30.0)
    if wl_mask.any():
        idx = int(np.argmax(mu_flat[wl_mask]))
        out["white_line_energy"] = float(energy[wl_mask][idx])
        out["white_line_height"] = float(mu_flat[wl_mask][idx])

    # ---- 半高能量 E1/2 ----
    hh_mask = (energy >= e0 - 15.0) & (energy <= e0 + 20.0)
    E_h, mu_h = energy[hh_mask], mu_flat[hh_mask]
    above = np.flatnonzero(mu_h >= 0.5)
    if above.size:
        j = int(above[0])
        if j == 0:
            e_half = float(E_h[0])
        else:
            y0, y1 = float(mu_h[j - 1]), float(mu_h[j])
            if abs(y1 - y0) < 1e-12:
                e_half = float(E_h[j])
            else:
                frac = (0.5 - y0) / (y1 - y0)
                e_half = float(E_h[j - 1] + frac * (E_h[j] - E_h[j - 1]))
        out["e_half_height"] = e_half
        if e0_nominal is not None:
            out["half_height_shift"] = float(e_half - e0_nominal)

    edge_type = str(edge or "K").strip().upper()
    out["edge_type"] = edge_type
    if absorber is not None:
        out["absorber"] = str(absorber)
    is_l_edge = edge_type.startswith("L")

    if is_l_edge:
        # B9-2：L2/L3 边没有 1s→3d 四极边前峰，跳过边前峰分峰，
        # 改为拟合主边阶跃 + 白线，输出净面积。字段名与 K 边不混用。
        wl = fit_whiteline_area(norm)
        for key in [k for k in out if k.startswith("pre_edge_")]:
            out.pop(key, None)
        out.update({
            "white_line_area": float(wl.get("white_line_area") or 0.0),
            "white_line_area_norm": float(wl.get("white_line_area_norm") or 0.0),
            "whiteline_peak_height": float(wl.get("whiteline_peak_height") or 0.0),
            "white_line_fit_r_factor": wl.get("r_factor"),
            "white_line_fit_reliable": bool(wl.get("reliable")),
            "white_line_fit_note": wl.get("note", ""),
        })
        if wl.get("white_line_energy") is not None:
            out["white_line_energy"] = float(wl["white_line_energy"])
            out["white_line_height"] = float(
                wl.get("whiteline_peak_height") or out["white_line_height"] or 0.0)
        return out

    # ---- 边前峰（扣两端点连线基线）----
    pre_mask = (energy >= e0 - 12.0) & (energy <= e0 - 2.0)
    if int(pre_mask.sum()) >= 3:
        Ep, mp = energy[pre_mask], mu_flat[pre_mask]
        if Ep[-1] > Ep[0]:
            baseline = mp[0] + (mp[-1] - mp[0]) * (Ep - Ep[0]) / (Ep[-1] - Ep[0])
        else:
            baseline = np.zeros_like(mp)
        resid = mp - baseline
        k = int(np.argmax(resid))
        out["pre_edge_peak_energy"] = float(Ep[k])
        out["pre_edge_peak_height"] = float(max(0.0, resid[k]))
        positive = np.maximum(resid, 0.0)
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        out["pre_edge_area"] = float(max(0.0, trapz(positive, Ep)))

    # ---- B6-3 边前峰分峰（扣主边拖尾 + Pseudo-Voigt）----
    try:
        vor = fit_pre_edge_peaks(norm)
    except (XAFSError, ValueError, FloatingPointError):
        vor = None
    if vor and vor.get("n_peaks", 0) >= 1 \
            and vor.get("centroid_energy") is not None:
        out["pre_edge_centroid"] = float(vor["centroid_energy"])
        out["pre_edge_voigt_area"] = float(vor["total_area"])
        out["pre_edge_n_peaks"] = int(vor["n_peaks"])
        out["pre_edge_voigt_r_factor"] = vor.get("r_factor")
    else:
        # 窗口点数不足/拟合失败：安全降级到两点直线基线的峰位
        out["pre_edge_centroid"] = out["pre_edge_peak_energy"]

    # ---- 颈部肩峰：上升沿一阶导数双峰 ----
    wl_e = out["white_line_energy"]
    rise_hi = float(wl_e) if wl_e is not None else e0 + 30.0
    rise_mask = ((energy >= e0 - 5.0) & (energy <= rise_hi)
                 & (mu_flat >= 0.15) & (mu_flat <= 0.85))
    Er, mr = energy[rise_mask], mu_flat[rise_mask]
    if len(Er) >= 7:
        # 按能量排序（文件偶尔有重复/倒序点）
        order = np.argsort(Er)
        Er, mr = Er[order], mr[order]
        Er, uniq_idx = np.unique(Er, return_index=True)
        mr = mr[uniq_idx]
        dE = float(np.median(np.diff(Er))) if len(Er) > 1 else 0.0
        if np.isfinite(dE) and dE > 0:
            D1 = np.gradient(mr, Er)
            from scipy.signal import find_peaks, savgol_filter
            npts = int(round(1.5 / dE))
            npts = max(5, npts)
            if npts % 2 == 0:
                npts += 1
            if npts > len(Er):
                npts = len(Er) if len(Er) % 2 == 1 else len(Er) - 1
            if npts >= 5 and npts > 3:
                D1 = savgol_filter(D1, npts, 3)
            peaks, _ = find_peaks(D1, distance=max(1, int(1.5 / dE)))
            if len(peaks):
                heights = D1[peaks]
                main_h = float(np.max(heights))
                if main_h > 0:
                    sig = [int(p) for p, h in zip(peaks, heights)
                           if float(h) >= 0.35 * main_h]
                    out["derivative_peaks_eV"] = [
                        float(Er[p]) for p in sorted(sig, key=lambda q: Er[q])]
                    out["derivative_peak_heights"] = [
                        float(D1[p]) for p in sorted(sig, key=lambda q: Er[q])]
                    if len(sig) >= 2:
                        top = sorted(sig, key=lambda p: D1[p],
                                     reverse=True)[:2]
                        e_sorted = sorted(float(Er[p]) for p in top)
                        h_sorted = sorted(float(D1[p]) for p in top)
                        split = e_sorted[1] - e_sorted[0]
                        if split >= 1.5:
                            out["neck_shoulder_present"] = True
                            out["neck_shoulder_energy"] = e_sorted[0]
                            out["derivative_split_eV"] = float(split)
                            hmax = max(h_sorted)
                            out["e0_jump_risk"] = bool(
                                hmax > 0 and min(h_sorted) >= 0.70 * hmax)
    return out


# ================================================================
# 4. 背景扣除
# ================================================================

def _pseudo_voigt_area(E, A: float, E_c: float, gamma: float,
                       eta: float) -> np.ndarray:
    """面积参数化 Pseudo-Voigt（A 为积分面积，单位同 y·eV）。"""
    E = np.asarray(E, dtype=float)
    x = E - float(E_c)
    lor = (float(gamma) / np.pi) / (x ** 2 + float(gamma) ** 2)
    gau = (np.sqrt(np.log(2.0)) / (float(gamma) * np.sqrt(np.pi))
           * np.exp(-np.log(2.0) * (x / float(gamma)) ** 2))
    return float(A) * (float(eta) * lor + (1.0 - float(eta)) * gau)


def fit_pre_edge_peaks(
        norm: NormalizedScan,
        window: tuple[float, float] = (-14.0, -2.0),
        max_peaks: int = 2,
) -> dict:
    """边前峰主边拖尾基线扣除 + 多 Pseudo-Voigt 分峰（Wilke 质心）。

    为什么两点直线基线不够：边前峰常贴近主吸收边上升沿，主边的
    Lorentzian 拖尾在窗口内是弯曲的；直线基线会过度切割边前峰，
    使 ``pre_edge_area`` 系统性偏小，也无法拆分 t2g/eg 双峰。

    基线:
        B(E) = a + b·(E−E0) + c/((E−(E0+2))² + 3²),  c ≥ 0
        仅用窗口两端各 1.5 eV 的边界带拟合，避免把边前峰本身当基线。
    峰形:
        面积参数化 Pseudo-Voigt（A=积分面积，γ=半高半宽，η=洛伦兹权重）。
    双峰判据:
        RSS2 < 0.70·RSS1 且峰间距 ≥0.8 eV 且次峰面积 ≥10% 主峰面积。

    返回:
        centroid_energy / total_area / n_peaks / peaks / r_factor /
        baseline / residual / reliable
        面积加权质心 E_c = Σ A_k E_k / Σ A_k；分母为 0 时回退到
        最大峰中心并置 ``reliable=False``。
    """
    from scipy.optimize import least_squares

    energy = np.asarray(norm.energy, dtype=float)
    y_all = (np.asarray(norm.flattened, dtype=float)
             if norm.flattened is not None
             else np.asarray(norm.mu_norm, dtype=float))
    e0 = float(norm.e0)
    emin, emax = e0 + float(window[0]), e0 + float(window[1])
    if emax <= emin:
        raise XAFSConfigError(
            "S3", f"边前窗口无效: [{window[0]}, {window[1]}] eV")
    sel = (energy >= emin) & (energy <= emax)
    E = energy[sel]
    y = y_all[sel]
    out = {
        "centroid_energy": None, "total_area": 0.0, "n_peaks": 0,
        "peaks": [], "r_factor": None, "baseline": None, "residual": None,
        "reliable": False, "window_eV": [float(emin), float(emax)],
        "note": "",
    }
    if E.size < 6:
        out["note"] = f"边前窗口仅 {E.size} 点（<6），安全降级"
        return out

    # ---- 1) 主边拖尾基线：两端边界带联合拟合 ----
    band = (((E >= emin) & (E <= min(emin + 1.5, emax)))
            | ((E >= max(emax - 1.5, emin)) & (E <= emax)))
    Eb, yb = E[band], y[band]
    e_edge_tail = e0 + 2.0
    gamma0 = 3.0

    def _baseline(p, x):
        return p[0] + p[1] * (x - e0) + p[2] / ((x - e_edge_tail) ** 2
                                                + gamma0 ** 2)

    if Eb.size >= 4:
        x0 = np.array([float(np.median(yb)), 0.0, 0.0])
        try:
            bl = least_squares(
                lambda p: _baseline(p, Eb) - yb, x0,
                bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                max_nfev=20000)
            base = _baseline(bl.x, E)
            if not np.all(np.isfinite(base)):
                raise ValueError("baseline non-finite")
        except Exception:
            coef = np.polyfit(Eb, yb, 1)
            base = np.polyval(coef, E)
    else:
        coef = np.polyfit(E, y, 1)
        base = np.polyval(coef, E)
    y_net = np.maximum(y - base, 0.0)
    out["baseline"] = base
    if float(np.max(y_net)) <= 1e-12:
        out["note"] = "扣基线后无边前峰信号"
        return out

    dE = float(np.median(np.diff(E))) if E.size > 1 else 0.0
    if not np.isfinite(dE) or dE <= 0:
        out["note"] = "能量网格非单调，无法分峰"
        return out

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    ec_lo, ec_hi = emin + 0.5, emax - 0.5
    a_max = max(float(_trapz(y_net, E)) * 5.0, 1e-6)

    def _fit_n(n_peaks, starts):
        best = None
        for x0 in starts:
            def fun(p):
                model = np.zeros_like(E)
                for k in range(n_peaks):
                    off = 4 * k
                    model = model + _pseudo_voigt_area(
                        E, p[off], p[off + 1], p[off + 2], p[off + 3])
                return model - y_net
            lb, ub = [], []
            for _ in range(n_peaks):
                lb.extend([0.0, ec_lo, 0.4, 0.0])
                ub.extend([a_max, ec_hi, 2.5, 1.0])
            try:
                sol = least_squares(fun, np.clip(x0, lb, ub),
                                    bounds=(lb, ub), max_nfev=40000)
            except Exception:
                continue
            rss = float(np.sum(sol.fun ** 2))
            if best is None or rss < best[0]:
                best = (rss, np.asarray(sol.x, dtype=float))
        return best

    # 单峰初值：面积/峰高/中心/宽度粗估
    i_max = int(np.argmax(y_net))
    area_total = float(_trapz(y_net, E))
    gamma_init = 1.0
    x1 = [area_total, float(E[i_max]), gamma_init, 0.5]
    fit1 = _fit_n(1, [x1])
    if fit1 is None:
        out["note"] = "单峰拟合未收敛"
        return out
    rss1, p1 = fit1

    best_n, best_p, best_rss = 1, p1, rss1
    if max_peaks >= 2:
        # 第二峰初值：距主峰 ≥0.8 eV 的最强局部极大（无则取最强点）
        from scipy.signal import find_peaks
        peak_idx, _ = find_peaks(y_net, prominence=1e-12)
        c_main = float(p1[1])
        cand = [int(i) for i in peak_idx
                if abs(float(E[int(i)]) - c_main) >= 0.8]
        if cand:
            c_second = float(E[max(cand, key=lambda i: y_net[i])])
        else:
            far = np.abs(E - c_main) >= 0.8
            c_second = (float(E[far][int(np.argmax(y_net[far]))])
                        if np.any(far) else c_main + 1.0)
        h_main = float(np.interp(c_main, E, y_net))
        h_second = float(np.interp(c_second, E, y_net))
        frac = h_second / (h_main + h_second) if (h_main + h_second) > 0 else 0.5
        area_second = area_total * max(0.05, min(0.95, frac))
        area_main = max(area_total - area_second, 1e-9)
        starts = [
            [area_main, c_main, 1.0, 0.5, area_second, c_second, 1.0, 0.5],
            [area_second, c_second, 1.0, 0.5, area_main, c_main, 1.0, 0.5],
            [area_main * 0.7, c_main, 0.8, 0.5,
             area_second * 1.3, c_second, 1.2, 0.5],
        ]
        fit2 = _fit_n(2, starts)
        if fit2 is not None:
            rss2, p2 = fit2
            centers = [float(p2[1]), float(p2[5])]
            areas = [abs(float(p2[0])), abs(float(p2[4]))]
            sep = abs(centers[0] - centers[1])
            ratio = min(areas) / max(max(areas), 1e-12)
            if rss2 < 0.70 * rss1 and sep >= 0.8 and ratio >= 0.10:
                best_n, best_p, best_rss = 2, p2, rss2

    peaks = []
    for k in range(best_n):
        off = 4 * k
        peaks.append({"area": float(abs(best_p[off])),
                      "center": float(best_p[off + 1]),
                      "hwhm": float(best_p[off + 2]),
                      "eta": float(best_p[off + 3])})
    peaks.sort(key=lambda p: p["center"])
    model = np.zeros_like(E)
    for p in peaks:
        model = model + _pseudo_voigt_area(
            E, p["area"], p["center"], p["hwhm"], p["eta"])
    denom_rss = float(np.sum(y_net ** 2))
    total_area = float(sum(p["area"] for p in peaks))
    if total_area > 0:
        centroid = float(sum(p["area"] * p["center"] for p in peaks)
                         / total_area)
        reliable = True
        note = ""
    else:
        centroid = float(peaks[0]["center"]) if peaks else None
        reliable = False
        note = "分峰总面积为 0，质心回退到最大峰中心"
    out.update({
        "centroid_energy": centroid,
        "total_area": total_area,
        "n_peaks": int(best_n),
        "peaks": peaks,
        "r_factor": (float(best_rss / denom_rss)
                     if denom_rss > 0 else None),
        "model": model,
        "residual": y_net - model,
        "reliable": bool(reliable),
        "note": note,
    })
    return out
