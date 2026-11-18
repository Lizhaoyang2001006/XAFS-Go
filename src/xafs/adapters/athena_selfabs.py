# -*- coding: utf-8 -*-
"""athena_selfabs.py — # 4.5 Booth 自吸收校正与 XANES Tröger 反解（自 athena.py 纯拆分）

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


def _booth_mu_terms(formula: str, elem: str, edge: str, line: str, e0: float,
                    k, density: float):
    """Booth 公式所需的光滑吸收系数（μm⁻¹）与荧光线能量。

    返回 (mu_t, mu_a, mu_f, e_fluor, comp)：
        mu_t  总吸收（photo）
        mu_a  吸收元素贡献（photo，**已扣边前基线**）
        mu_f  荧光能量处的总吸收（total）
    全部已乘密度并换算为 μm⁻¹；d 用 μm 时 α·d 无量纲。

    能量锚点用**表值边能量**（xraydb），不是样品 E0：
        Demeter/Athena 的 Booth 实现同样用 k²/K2 + E_edge + 0.1。
        自吸收会把 E0 拉低（实测合成谱 7111.3 eV < Fe K 7112 eV），
        若用样品 E0，k=0 落在表值边以下，μ_a 扣基线后 ≤0，校正失败。
        以表值边为锚还能避免边能量差造成的 kink。

    为什么 μ_a 要扣边前基线：
        Booth 公式里的 χ 是相对振荡 (μ_a−μ̄_a)/μ̄_a，μ̄_a 是可振荡的
        边跳部分；归一化谱已经把边前吸收扣掉了。Athena/Demeter 的
        Booth 实现同样用 cross_section(E) − cross_section(edge−200 eV)。
        不扣基线会把 μ_a 高估约 15%（Fe K 边），修正因子系统偏大。
    """
    try:
        e_fluor = float(xray_line(elem, line).energy)
        e_edge = float(xray_edge(elem, edge).energy)
        energy = np.asarray(k, dtype=float) ** 2 / K2 + e_edge + 0.1
        comp = material_mu_components(formula, energy, density=density,
                                      kind="photo")
        comp_below = material_mu_components(formula, e_edge - 200.0,
                                            density=density, kind="photo")
        mu_f_cm = float(material_mu(formula, e_fluor, density=density,
                                    kind="total"))
    except Exception as exc:
        raise XAFSConfigError(
            "S4.5", f"无法计算 {formula} / {elem} {edge} {line} 的截面："
                    f"{exc}") from exc
    if elem not in comp["elements"]:
        raise XAFSConfigError(
            "S4.5", f"化学式 {formula} 不含吸收元素 {elem}；"
                    f"校正必须用样品总组成，不能只写活性相")

    mass = float(comp["mass"])
    mu_t_cm = np.zeros_like(energy, dtype=float)
    for el in comp["elements"]:
        stoich, atomic_mass, mu_mass = comp[el]
        mu_t_cm += (density * float(stoich) * float(atomic_mass)
                    * np.asarray(mu_mass, dtype=float) / mass)
    stoich_a, mass_a, mu_a_mass = comp[elem]
    mu_a_cm = (density * float(stoich_a) * float(mass_a)
               * np.asarray(mu_a_mass, dtype=float) / mass)
    stoich_b, mass_b, mu_b_mass = comp_below[elem]
    mu_a_below_cm = (density * float(stoich_b) * float(mass_b)
                     * float(mu_b_mass) / float(comp_below["mass"]))
    mu_a_cm = mu_a_cm - mu_a_below_cm
    if np.any(mu_a_cm <= 0):
        raise XAFSConfigError(
            "S4.5", f"{elem} 扣边前基线后 μ_a ≤ 0；检查 E0/边能量是否匹配")
    return mu_t_cm * 1e-4, mu_a_cm * 1e-4, mu_f_cm * 1e-4, e_fluor, comp


def booth_self_absorption(chi_k: ChiK, formula: str, elem: str, edge: str,
                          line: str, e0: float, thickness_um: float,
                          density: float, anginp: float = 45.0,
                          angout: float = 45.0) -> tuple[ChiK, dict]:
    """[Implements: EQ-05] 在 k 空间做荧光自吸收校正（Booth & Bridges, Phys. Scr. 2005）。

    物理:
        荧光强度 ∝ μ_a(E) / [μ_T(E) + g·μ_f] × [1 − exp(−(μ_T/sinφ
        + μ_f/sinθ)d)]。测量得到的 χ_exp 因此被样品自身的吸收压低，
        且压低程度随 k 变化。Booth & Bridges 把上式对 χ 反解成闭式
        （Eq. 4），得到 χ(k) = f(χ_exp(k), μ_T, μ_a, μ_f, d, φ, θ)。

    输入（缺一不可）:
        formula      样品**总**化学式（含基体/稀释剂），如 Fe3O4、
                     ((NH4)2SO4)0.1(H2O)54.8；不是只写活性相
        elem         吸收元素（Fe/Pt/...）
        edge, line   吸收边与荧光线（K→Ka，L3→La）
        thickness_um 样品厚度（μm）；厚样可取大于信息深度的值
        density      有效密度（g/cm³）；粉末压片用几何密度，不是晶体密度
        anginp/out   入射/出射角（相对样品表面，度）

    假设与边界:
        - 均匀、平表面样品；不均匀/颗粒/孔洞会让有效 d、ρ 失真
        - μ_T/μ_a 用 xraydb 的光滑截面，保留吸收边但没有 EXAFS 振荡；
          μ_a 已扣边前基线（与 Athena/Demeter 的 Booth 实现一致），
          能量锚定表值边 E_edge+0.1（不随样品 E0 漂移）
        - 只改振幅、不改相位；返回新的 ChiK，background/mu_norm 不修改
        - 薄样/厚样由 d 自动落入相应极限；d→0 时 χ_corr→χ_exp
        - 自吸收会压低/移动 E0（合成厚样实测 7111.3 vs 表值 7112 eV）；
          校正不修 E0，拟合前应用参考箔或已知边能量校准
        - 化学式/厚度/密度/角度任一不可靠，校正就不可靠；info 里给出
          correction_factor 与 reliable 标记，报告层不得静默引用
    """
    formula = (formula or "").strip()
    if not formula:
        raise XAFSConfigError("S4.5", "Booth 自吸收校正需要样品总化学式")
    if not (thickness_um > 0):
        raise XAFSConfigError("S4.5", f"厚度必须 >0 μm，收到 {thickness_um}")
    if not (density > 0):
        raise XAFSConfigError("S4.5", f"密度必须 >0 g/cm³，收到 {density}")
    for name, ang in (("anginp", anginp), ("angout", angout)):
        if not (0.0 < ang <= 90.0):
            raise XAFSConfigError(
                "S4.5", f"{name}={ang}° 必须在 (0, 90] 内（相对样品表面；"
                        f"90° = 垂直入射/出射）")

    mu_t, mu_a, mu_f, e_fluor, comp = _booth_mu_terms(
        formula, elem, edge, line, e0, chi_k.k, density)
    sin_phi = np.sin(np.deg2rad(anginp))
    sin_theta = np.sin(np.deg2rad(angout))
    g = sin_phi / sin_theta
    alpha = mu_t + g * mu_f
    if np.any(alpha <= 0):
        raise XAFSDataError("S4.5", "Booth 校正的 α=μ_T+g·μ_f 出现非正值，"
                                    "检查化学式/荧光线/能量范围")
    arg = alpha * float(thickness_um) / sin_phi
    exp_neg = np.exp(-np.clip(arg, 0.0, 700.0))
    gamma = 1.0 - exp_neg
    beta = mu_a * float(thickness_um) * alpha / sin_phi * exp_neg

    chi_exp = np.asarray(chi_k.chi, dtype=float)
    chi_corr = np.array(chi_exp, dtype=float, copy=True)
    thin = arg < 1e-6
    # arg>20 时 exp(-arg)<2e-9，厚样极限与 Eq.4 的差异远小于数据误差；
    # 提前切到厚样分支还能避开 β→0 时二次公式的灾难性相消。
    thick = arg > 20.0
    mid = ~(thin | thick)

    # 中间厚度：Eq. 4（β>0）
    if np.any(mid):
        b = beta[mid]
        a = alpha[mid]
        ma = mu_a[mid]
        gm = gamma[mid]
        ce = chi_exp[mid]
        term = gm * (a - ma * (ce + 1.0)) + b
        disc = term ** 2 + 4.0 * a * b * gm * ce
        disc = np.clip(disc, 0.0, None)
        chi_corr[mid] = (-term + np.sqrt(disc)) / (2.0 * b)

    # 厚样极限（d→∞，β→0）：χ = χ_exp / [1 − (μ_a/α)(χ_exp+1)]
    clipped = 0
    if np.any(thick):
        denom = 1.0 - (mu_a[thick] / alpha[thick]) * (chi_exp[thick] + 1.0)
        ok = denom > 1e-6
        clipped = int(np.count_nonzero(~ok))
        chi_corr[thick] = np.where(ok, chi_exp[thick] / np.where(ok, denom, 1.0),
                                   chi_exp[thick])

    finite = np.all(np.isfinite(chi_corr))
    ratio = np.where(np.abs(chi_exp) > 1e-8, chi_corr / chi_exp, np.nan)
    ratio_finite = ratio[np.isfinite(ratio)]
    e_edge_tab = float(xray_edge(elem, edge).energy)
    e0_offset = float(e0) - e_edge_tab
    e0_warning = None
    if abs(e0_offset) > 3.0:
        e0_warning = (f"E0={e0:.2f} eV 与表值 {edge} 边 {e_edge_tab:.2f} eV "
                      f"相差 {e0_offset:+.2f} eV；自吸收可能移动了 E0，"
                      f"拟合前应用参考箔/已知边能量校准")
    info = {
        "method": "Booth & Bridges 2005 Eq.4 (k-space)",
        "corrected": True,
        "formula": formula, "elem": elem, "edge": edge, "line": line,
        "thickness_um": float(thickness_um), "density": float(density),
        "anginp": float(anginp), "angout": float(angout), "g": float(g),
        "fluor_energy_eV": float(e_fluor),
        "energy_anchor_eV": e_edge_tab + 0.1,
        "e0_offset_eV": e0_offset,
        "mu_a_baseline": "subtracted (E_edge-200 eV)",
        "mu_f_cm": float(mu_f * 1e4),
        "mu_t_cm_range": [float(np.min(mu_t) * 1e4),
                          float(np.max(mu_t) * 1e4)],
        "mu_a_cm_range": [float(np.min(mu_a) * 1e4),
                          float(np.max(mu_a) * 1e4)],
        "correction_factor_median": (float(np.nanmedian(ratio_finite))
                                     if ratio_finite.size else None),
        "correction_factor_range": ([float(np.nanmin(ratio_finite)),
                                     float(np.nanmax(ratio_finite))]
                                    if ratio_finite.size else None),
        "thin_limit_points": int(np.count_nonzero(thin)),
        "thick_limit_points": int(np.count_nonzero(thick)),
        "thick_limit_clipped": clipped,
        "reliable": bool(finite and clipped == 0),
        "warning": e0_warning,
        "note": ("Booth 厚样/任意厚度近似；要求均匀样品与准确的总化学式、"
                 "厚度、有效密度、角度。只改振幅不改相位，也不修 E0"
                 "（自吸收会把 E0 拉低，拟合前需用参考箔校准）；化学式"
                 "偏差会直接进入校正因子，不能把校正后的 N 当无系统误差"
                 "结果。"),
    }
    return ChiK(k=np.asarray(chi_k.k), chi=chi_corr,
                background=chi_k.background, mu_norm=chi_k.mu_norm,
                rbkg=chi_k.rbkg), info


def fluo_xanes_self_absorption(
        norm: NormalizedScan,
        formula: str, elem: str, edge: str, line: str,
        thickness_um: float, density: float,
        anginp: float = 45.0, angout: float = 45.0,
) -> tuple[NormalizedScan, dict]:
    """XANES 能量域荧光自吸收反解（Tröger / FLUO，与 EXAFS Booth 隔离）。

    物理:
        浓样荧光测量的归一化 XANES 满足厚样压缩关系
            μ_exp(E) = (α+1)·μ_true(E) / (α + μ_true(E))
        闭式反解（Tröger 方程）
            μ_true(E) = α·μ_exp(E) / (α + 1 − μ_exp(E))
        其中 α = [μ_bg(E_ref) + g·μ_f] / μ_a(E_ref)，
        E_ref = E0 + 50 eV，g = sinθ_in / sinθ_out。

    为什么不能直接用 Larch fluo_corr 全谱：全谱反解会把高能 EXAFS
    放大数倍，污染 χ(k)。本函数**只改 NormalizedScan 的 XANES**
    （mu_norm / flattened），绝不生成或修改 ChiK；EXAFS 自吸收仍走
    S4.5 的 ``booth_self_absorption``。

    物理护栏:
      1. 仅对 E ≥ E0−15 eV 且 μ_exp > 0 的点应用；
      2. 分母 D = α+1−μ_exp 的下界为 0.25·(α+1)；触界即钳制并置
         ``reliable=False``，防止白线尖峰处分母趋零炸出无穷大；
      3. 返回新对象，不原地修改输入 norm。

    可靠性仍取决于总化学式、厚度、有效密度与角度；这些量不准确时
    反解出的白线高度与 LCF 权重不可当作无系统误差结果。
    """
    formula = (formula or "").strip()
    if not formula:
        raise XAFSConfigError("S4.5", "XANES 荧光自吸收校正需要样品总化学式")
    if not (thickness_um > 0):
        raise XAFSConfigError("S4.5", f"厚度必须 >0 μm，收到 {thickness_um}")
    if not (density > 0):
        raise XAFSConfigError("S4.5", f"密度必须 >0 g/cm³，收到 {density}")
    for name, ang in (("anginp", anginp), ("angout", angout)):
        if not (0.0 < ang <= 90.0):
            raise XAFSConfigError(
                "S4.5", f"{name}={ang}° 必须在 (0, 90] 内")

    energy = np.asarray(norm.energy, dtype=float)
    e0 = float(norm.e0)
    e_edge_tab = float(xray_edge(elem, edge).energy)
    e_ref = e0 + 50.0
    # _booth_mu_terms 的输入是 k；锚点仍是表值边 +0.1 eV（与 Booth 一致）
    k_ref = np.sqrt(max(K2 * (e_ref - (e_edge_tab + 0.1)), 0.0))
    mu_t, mu_a, mu_f, e_fluor, comp = _booth_mu_terms(
        formula, elem, edge, line, e0, np.asarray([k_ref], dtype=float),
        density)
    sin_phi = np.sin(np.deg2rad(anginp))
    sin_theta = np.sin(np.deg2rad(angout))
    g = sin_phi / sin_theta
    mu_bg = float(mu_t[0] - mu_a[0])
    alpha = (mu_bg + g * float(mu_f)) / float(mu_a[0])
    if not np.isfinite(alpha) or alpha <= 0:
        raise XAFSDataError(
            "S4.5", f"Tröger α={alpha!r} 非正/非有限；检查化学式/荧光线/边")

    # 光学厚度 τ = d·μ_a(E_ref)/sinφ（d[μm]·μ_a[μm⁻¹]，无量纲），由真实
    # 厚度/密度/截面决定——绝不拿荧光 edge_step 当光学厚度。β_ref=τ(α+1)
    # 即 Booth 的 arg；薄样 β_ref→0 时自吸收消失，校正→恒等。
    mu_a0 = float(mu_a[0])
    tau = float(thickness_um) * mu_a0 / sin_phi
    beta_ref = tau * (alpha + 1.0)
    c_norm = -np.expm1(-min(beta_ref, 700.0))      # C = 1−exp(−τ(α+1))
    floor = 0.25 * (alpha + 1.0)
    applied = (energy >= e0 - 15.0)
    if beta_ref < 1e-6:
        regime = "thin"
    elif beta_ref > 20.0:
        regime = "thick"
    else:
        regime = "mid"

    def _invert_thick(y):
        # 厚样极限闭式 Tröger（β_ref>20，exp(-β)<2e-9）：与历史逐位一致
        out = np.array(y, dtype=float, copy=True)
        y = np.asarray(y, dtype=float)
        mask = applied & (y > 0)
        if not np.any(mask):
            return out, True, 0
        denom = alpha + 1.0 - y[mask]
        n_clamped = int(np.count_nonzero(denom < floor))
        denom = np.maximum(denom, floor)
        out[mask] = alpha * y[mask] / denom
        return out, n_clamped == 0, n_clamped

    def _finite_x(k_val, brentq):
        # 有限厚度正演 μ_exp=(α+1)x/(α+x)·[1−e^{−τ(α+x)}]/C 的数值反解。
        # 令 w=τ(α+x)、K=μ_exp·C/(α+1)，解 (1−ατ/w)(1−e^{−w})=K，x=w/τ−α。
        # g 单调增，从 w_lo=ατ 倍增括住根后 Brent 求解；τ→0 时 x→μ_exp。
        a_tau = alpha * tau
        w_lo = a_tau

        def g(w):
            return (w - a_tau) / w * (-np.expm1(-w)) - k_val

        w_hi = w_lo * 2.0
        for _ in range(200):
            if g(w_hi) > 0.0:
                break
            w_hi *= 2.0
        else:
            return float("nan")
        w = brentq(g, w_lo, w_hi, xtol=1e-13, rtol=1e-13)
        return w / tau - alpha

    def _invert_finite(y):
        from scipy.optimize import brentq
        out = np.array(y, dtype=float, copy=True)
        y = np.asarray(y, dtype=float)
        mask = applied & (y > 0)
        if not np.any(mask):
            return out, True, 0
        ym = y[mask]
        k_arr = ym * c_norm / (alpha + 1.0)
        # 饱和护栏：K>0.75 等价厚样 denom<0.25(α+1)，跨 β_ref=20 边界连续
        n_clamped = int(np.count_nonzero(k_arr > 0.75))
        k_arr = np.minimum(k_arr, 0.75)
        out[mask] = np.array([_finite_x(float(k), brentq) for k in k_arr])
        return out, n_clamped == 0, n_clamped

    if regime == "thin":
        # 薄样极限：光学厚度可忽略，无自吸收，恒等返回（不校正）
        mu_corr = np.array(norm.mu_norm, dtype=float, copy=True)
        flat_corr = (None if norm.flattened is None
                     else np.array(norm.flattened, dtype=float, copy=True))
        ok_mu = ok_flat = True
        clamp_mu = clamp_flat = 0
    else:
        _invert = _invert_thick if regime == "thick" else _invert_finite
        mu_corr, ok_mu, clamp_mu = _invert(norm.mu_norm)
        if norm.flattened is not None:
            flat_corr, ok_flat, clamp_flat = _invert(norm.flattened)
        else:
            flat_corr, ok_flat, clamp_flat = None, True, 0

    finite = bool(np.all(np.isfinite(mu_corr))
                  and (flat_corr is None or np.all(np.isfinite(flat_corr))))
    reliable = bool(ok_mu and ok_flat and finite
                    and (clamp_mu + clamp_flat) == 0)
    info = {
        "method": ("Tröger/FLUO inverse (XANES energy domain)"
                   if regime == "thick" else
                   "Tröger/FLUO finite-thickness inverse (XANES energy domain)"
                   if regime == "mid" else
                   "Tröger/FLUO thin limit (no self-absorption correction)"),
        "corrected": True,
        "xanes_corrected": True,
        "exafs_modified": False,
        "formula": formula, "elem": elem, "edge": edge, "line": line,
        "thickness_um": float(thickness_um), "density": float(density),
        "anginp": float(anginp), "angout": float(angout), "g": float(g),
        "energy_ref_eV": float(e_ref),
        "fluor_energy_eV": float(e_fluor),
        "alpha": float(alpha),
        "optical_thickness_tau": float(tau),
        "beta_ref": float(beta_ref),
        "thickness_regime": regime,
        "thin_limit_no_correction": bool(regime == "thin"),
        "denominator_floor": float(floor),
        "clamped_points": int(clamp_mu + clamp_flat),
        "applied_from_eV": float(e0 - 15.0),
        "reliable": reliable,
        "warning": (None if reliable else
                    "Tröger 分母触底被钳制：白线尖峰过强或 α 不可靠，"
                    "校正后的白线/LCF 权重不能当无系统误差结果"),
        "note": ("仅修 XANES；EXAFS χ(k) 必须走独立的 Booth 校正。"
                 "校正含厚度适用域：光学厚度 τ=d·μ_a/sinφ 决定薄/中/厚"
                 "（β_ref=τ(α+1)；<1e-6 薄样不校正，>20 厚样闭式 Tröger，"
                 "其间有限厚度数值反解）。τ 由真实厚度/密度/截面算出，"
                 "绝不拿荧光 edge_step 当光学厚度。总化学式/厚度/有效密度/"
                 "角度任一不可靠，α 与反解结果即不可靠。"),
    }
    corrected = NormalizedScan(
        energy=np.array(energy, copy=True),
        mu=np.array(norm.mu, dtype=float, copy=True),
        mu_norm=mu_corr,
        pre_edge_line=np.array(norm.pre_edge_line, dtype=float, copy=True),
        edge_step=float(norm.edge_step),
        plateau=float(norm.plateau),
        e0=e0,
        flattened=(None if flat_corr is None
                   else np.asarray(flat_corr, dtype=float)))
    return corrected, info
