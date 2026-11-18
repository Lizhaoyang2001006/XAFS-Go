# -*- coding: utf-8 -*-
"""athena_bkg_ft.py — # 4 背景扣除 + # 5 Rbkg 寻优 + # 6 傅里叶变换（自 athena.py 纯拆分）

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


def wave_number(energy, e0: float) -> np.ndarray:
    """[Implements: EQ-01]
    光电子波矢。

    数学:
        k = sqrt(2·m_e·(E − E0) / ħ²) = sqrt(K2·(E − E0)) ≈ 0.5123·sqrt(E − E0)

    单位: E 用 eV，k 得 Å⁻¹。E < E0 处返回 0。
    """
    return np.sqrt(np.maximum(K2 * (np.asarray(energy) - e0), 0.0))


def extract_chi_k(norm: NormalizedScan, rbkg: float = 1.2, kweight: int = 2,
                  chi_std=None, k_std=None, clamp_lo: float | None = None,
                  clamp_hi: float | None = None) -> ChiK:
    """
    扣除原子背景，提取 EXAFS 振荡。

    数学:
        μ_norm(k) = χ(k) + μ0(k)
        约束 μ0 在 R 空间不含 R > Rbkg 的成分，
        实现为  χ = F⁻¹{ R > Rbkg } F[μ_norm]

    Rbkg 的物理含义:
        ≈ 0.5–0.7 × 第一配位壳层峰位。Rbkg 越大背景越"软"，
        R<1 Å 的残留越少，但过大会吃掉真实配位峰。

    为什么用 larch 的 autobk 而不是自己写:
        同一份数据，自研 R 空间滤波仅改过渡带宽度（0.06 → 0.25），
        第一峰结果会在 0.86 与 1.47 Å 之间跳；标准实现稳定给出 1.47。

    B10-1 先验与钳制（全部可选，默认不改变历史行为）:
        chi_std/k_std  标准谱 χ(k) 先验，保护低 R 真实信号；
        clamp_lo/hi    抑制样条两端甩尾的权重（0–1）。
        若当前 larch 版本不支持这些关键字，显式降级为无先验 AUTOBK，
        并在 ``ChiK.chi_std_used`` 保留 False（不假装用上了）。
    """
    def _run(extra: dict):
        g = Group(energy=np.asarray(norm.energy, dtype=float),
                  mu=np.asarray(norm.mu_norm, dtype=float))
        g.norm = np.asarray(norm.mu_norm, dtype=float)
        g.e0 = float(norm.e0)
        g.edge_step = 1.0
        autobk(g, rbkg=rbkg, kweight=kweight, e0=norm.e0, **extra)
        return g

    extra: dict = {}
    priors_requested = any(v is not None for v in
                           (chi_std, k_std, clamp_lo, clamp_hi))
    if chi_std is not None or k_std is not None:
        if chi_std is None or k_std is None:
            raise XAFSConfigError(
                "S5", "chi_std 与 k_std 必须成对给出（先验谱需要自己的 k 轴）")
        chi_std_arr = np.asarray(chi_std, dtype=float)
        k_std_arr = np.asarray(k_std, dtype=float)
        if chi_std_arr.ndim != 1 or k_std_arr.ndim != 1:
            raise XAFSConfigError("S5", "chi_std/k_std 必须是一维数组")
        if chi_std_arr.size != k_std_arr.size or chi_std_arr.size < 4:
            raise XAFSConfigError(
                "S5", f"chi_std/k_std 长度不一致或过短: "
                      f"{chi_std_arr.size} vs {k_std_arr.size}")
        if not (np.all(np.isfinite(chi_std_arr))
                and np.all(np.isfinite(k_std_arr))):
            raise XAFSConfigError("S5", "chi_std/k_std 含非有限值")
        extra["chi_std"] = chi_std_arr
        extra["k_std"] = k_std_arr
    if clamp_lo is not None:
        extra["clamp_lo"] = float(clamp_lo)
    if clamp_hi is not None:
        extra["clamp_hi"] = float(clamp_hi)

    degraded = False
    try:
        g = _run(extra)
    except TypeError:
        if not priors_requested:
            raise
        # 当前 larch 版本不接受先验/钳制关键字：显式降级。
        degraded = True
        g = _run({})

    return ChiK(k=np.asarray(g.k), chi=np.asarray(g.chi),
                background=np.asarray(g.bkg), mu_norm=np.asarray(g.norm),
                rbkg=float(rbkg),
                chi_std_used=bool("chi_std" in extra and not degraded),
                clamp_lo=(float(clamp_lo) if clamp_lo is not None
                          and not degraded else None),
                clamp_hi=(float(clamp_hi) if clamp_hi is not None
                          and not degraded else None))


def high_k_snr(chi_k: ChiK, sigma_mu=None, edge_step: float | None = None,
               kweight: int = 2) -> dict:
    """B9-2 由 μ(E) 噪声估计高 k 信噪比与可信 k 上限。

    噪声换算（常数近似，逐点 σ 更细时直接传数组）:

        σ_χ(k) = σ_μ / |edge_step|

    信号包络用 χ(k) 的滚动 RMS（窗口 ≈ 0.8 Å⁻¹），避免单点噪声造成
    假交叉。``snr_k < 2`` 且连续 3 点确认，记为该谱的 ``k_snr2``。

    诚实降级：没有 ``sigma_mu`` 时返回 ``snr_k=None``、``k_snr2=None``，
    并在 ``reason`` 说明，绝不编造一条“看起来很稳”的曲线。
    """
    k = np.asarray(chi_k.k, dtype=float)
    chi = np.asarray(chi_k.chi, dtype=float)
    out = {"snr_k": None, "k_snr2": None, "reliable_kmax": None,
           "sigma_chi": None, "reason": "", "window_A": None}
    if sigma_mu is None:
        out["reason"] = ("未提供 sigma_mu：无法估计 μ(E) 噪声，"
                         "不生成 SNR 判据")
        return out
    if edge_step is None:
        edge_step = 1.0
        out["reason"] = ("未提供 edge_step：按已归一化谱 edge_step=1 解释 "
                         "sigma_mu")
    edge_step = float(edge_step)
    if not np.isfinite(edge_step) or abs(edge_step) < 1e-12:
        raise XAFSConfigError(
            "S5", f"edge_step={edge_step:.4g} 不能用于噪声换算")
    sig = np.asarray(sigma_mu, dtype=float)
    if sig.ndim == 0:
        sigma_chi = np.full_like(k, float(sig) / abs(edge_step))
    elif sig.ndim == 1 and sig.size == k.size:
        sigma_chi = sig / abs(edge_step)
    else:
        raise XAFSConfigError(
            "S5", f"sigma_mu 需为标量或与 k 等长（{k.size}），"
                  f"收到 shape={sig.shape}；不猜测能量网格重采样")
    if not np.all(np.isfinite(sigma_chi)) or np.any(sigma_chi < 0):
        raise XAFSConfigError("S5", "sigma_mu 必须全部为有限非负数")
    # 零噪声（σ_μ≡0，无噪声数据或各扫描完全相同）：SNR=信号/0 无定义。
    # 旧实现把 0 并入"非正数"一律 raise，会在 S5 杀掉整条管线；正确是
    # 诚实降级（与 sigma_mu=None 同口径），不伪造一条"看起来很稳"的曲线。
    if np.all(sigma_chi == 0):
        out["sigma_chi"] = 0.0
        out["reason"] = ("零噪声（σ_μ≡0）：SNR=信号/0 无定义，不生成判据"
                         "（数据无噪声或各扫描完全相同）")
        return out
    out["sigma_chi"] = float(np.median(sigma_chi))

    dk = float(np.median(np.diff(k))) if k.size > 1 else 0.05
    if not np.isfinite(dk) or dk <= 0:
        raise XAFSConfigError("S5", "k 网格非单调，无法计算滚动 SNR")
    nwin = int(round(0.8 / dk))
    nwin = max(3, nwin if nwin % 2 == 1 else nwin + 1)
    nwin = min(nwin, k.size if k.size % 2 == 1 else k.size - 1)
    if nwin < 3:
        raise XAFSConfigError("S5", f"k 点数 {k.size} 太少（需 ≥4）")
    out["window_A"] = float(nwin * dk)
    half = nwin // 2
    padded = np.pad(chi, half, mode="edge")
    local_ms = np.array([
        np.mean(padded[i:i + nwin] ** 2) for i in range(k.size)])
    envelope = np.sqrt(np.maximum(local_ms, 0.0))
    # 个别点 σ_χ=0（零噪声）→ SNR=∞，物理上即"该点无限可信"；用 errstate
    # 抑制除零警告，inf 在 below=snr<2 判据里自然判为可信，不崩溃。
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = envelope / sigma_chi
    out["snr_k"] = snr

    below = snr < 2.0
    k_snr2 = None
    for i in range(max(0, below.size - 2)):
        if below[i] and below[i + 1] and below[i + 2]:
            k_snr2 = float(k[i])
            break
    out["k_snr2"] = k_snr2
    out["reliable_kmax"] = (k_snr2 if k_snr2 is not None
                            else float(k.max()))
    if k_snr2 is None and not out["reason"]:
        out["reason"] = "全 k 区间滚动 SNR 均 ≥2"
    return out


def estimate_single_scan_kmax(chi_k: ChiK, edge_step: float | None = None,
                              kweight: int = 2) -> dict:
    """无多扫描 sigma_mu 时，利用高 k 二阶差分法估计单条谱的白噪声与可信 k 上限。

    物理原理：在 Δk≈0.05 Å⁻¹ 网格上，真实 EXAFS 振荡（R ≤ 6 Å）波长 ≥ 10 点，
    二阶差分 Δ²χ = χ_{i+1} - 2χ_i + χ_{i-1} 对慢变信号高度衰减，而白噪声
    满足 std(Δ²χ) = √6 · σ_χ。据此无偏反推 σ_χ，再复用 high_k_snr 给出 k_snr2。
    """
    k = np.asarray(chi_k.k, dtype=float)
    chi = np.asarray(chi_k.chi, dtype=float)
    out = {
        "method": "single_scan_2nd_diff",
        "sigma_chi_est": None,
        "k_snr2": None,
        "reliable_kmax": None,
        "tail_rise_ratio": None,
        "reason": "",
    }
    if k.size < 20 or chi.size != k.size:
        out["reason"] = "k 点数不足，无法估计单扫高 k 噪声"
        return out
    m_tail = k >= 7.5
    if int(np.sum(m_tail)) < 12:
        m_tail = k >= max(4.0, 0.5 * float(np.nanmax(k)))
    if int(np.sum(m_tail)) < 12:
        out["reason"] = "有效高 k 点数不足 12 点，跳过单扫噪声估计"
        return out

    d2 = np.diff(chi[m_tail], 2)
    med = float(np.median(d2))
    mad = float(np.median(np.abs(d2 - med)))
    sigma_chi_est = mad / (0.67448975 * float(np.sqrt(6.0)))
    out["sigma_chi_est"] = float(sigma_chi_est)
    if sigma_chi_est <= 0.0:
        out["reason"] = "零高频二阶差分噪声（数据高度平滑或无噪声）"
        return out

    step = abs(float(edge_step)) if edge_step is not None and abs(float(edge_step)) > 1e-12 else 1.0
    snr = high_k_snr(chi_k, sigma_mu=sigma_chi_est * step, edge_step=step, kweight=kweight)
    out["k_snr2"] = snr.get("k_snr2")
    out["reliable_kmax"] = snr.get("reliable_kmax")
    out["reason"] = snr.get("reason") or "单扫高 k 二阶差分信噪估计完成"

    m_mid = (k >= 4.5) & (k <= 7.5)
    k_hi = min(float(np.nanmax(k)) - 0.15, 12.5)
    m_end = (k >= max(8.5, k_hi - 2.0)) & (k <= k_hi)
    if np.any(m_mid) and np.any(m_end):
        rms_mid = float(np.sqrt(np.mean(((k[m_mid] ** kweight) * chi[m_mid]) ** 2)))
        rms_end = float(np.sqrt(np.mean(((k[m_end] ** kweight) * chi[m_end]) ** 2)))
        if rms_mid > 0:
            out["tail_rise_ratio"] = float(rms_end / rms_mid)
    return out


def measure_shell_metrics(chi_k: ChiK, kmin=3.0, kmax=12.0, kweight=2):
    """
    用一次 FT 量出「第一壳层」与「低 R 残留」两个指标。

    返回:
        r1         第一峰位置 (Å)
        amplitude  第一峰高度
        leakage    R < 0.8 Å 的最大幅值
        ratio      leakage / amplitude
    """
    ft = ft_to_r_space(chi_k.k, chi_k.chi, None, kmin, kmax, kweight)
    mag = ft.magnitude

    # 第一壳表观峰窗口：上限 2.6 覆盖金属第一壳到重金属（如 Pt–Pt 2.77 →
    # 表观 ~2.45）；元素普适性要求，不再按 Fe 校准
    shell = (ft.r > 1.0) & (ft.r < 2.6)
    r1 = float(ft.r[shell][np.argmax(mag[shell])]) if shell.any() else np.nan
    amplitude = float(mag[shell].max()) if shell.any() else 0.0

    low = ft.r < 0.8
    leakage = float(mag[low].max()) if low.any() else 0.0
    return {"r1": r1, "amplitude": amplitude, "leakage": leakage,
            "ratio": leakage / amplitude if amplitude > 0 else np.inf}


def optimize_rbkg(norm: NormalizedScan, rbkg_range=(0.7, 1.6), n_grid=10,
                  kmin=3.0, kmax=12.0, kweight=2, leak_veto=0.20):
    """
    在 Rbkg 网格上寻优。

    判据（乘积型）:
        score = (1 − 低R残留比) × (主峰强度 / 全网格最大主峰强度)
        veto  : 低R残留比 > 0.20 直接淘汰

    为什么不用「主峰强度最大化」:
        实测 Rbkg 从 0.7 扫到 1.6，主峰强度为
        1.794 / 1.755 / 1.713 / 1.693 / 1.701 / 1.810 / 1.290 / 1.339 / 1.432 / 0.434
        **非单调**。按强度最大会选中 Rbkg=1.2 并否决最干净的 1.5。

    返回: (最优结果字典, 全表)
    """
    table = []
    for rbkg in np.linspace(rbkg_range[0], rbkg_range[1], n_grid):
        chi_k = extract_chi_k(norm, rbkg=rbkg, kweight=kweight)
        metrics = measure_shell_metrics(chi_k, kmin, kmax, kweight)
        table.append({"rbkg": float(rbkg), **metrics})

    amp_max = max(row["amplitude"] for row in table) or 1.0
    for row in table:
        row["score"] = (1.0 - min(row["ratio"], 1.0)) * (row["amplitude"] / amp_max)
        row["veto"] = bool(row["ratio"] > leak_veto)

    valid = [row for row in table if not row["veto"]]
    if valid:
        s_max = max(r["score"] for r in valid)
        # 并列裁决：与最优相差 <5% 时取最小 Rbkg。
        # 单调爬向网格边缘是「样条吃壳层→泄漏比降低→得分虚高」的病理形态
        # （NiO 实测：0.84→0.96 单调爬到 1.60，N(Ni-O) 被吃掉一半），
        # 得分不可分辨时低 Rbkg 对第一壳振幅更安全。
        tied = [r for r in valid if r["score"] >= 0.95 * s_max]
        best = min(tied, key=lambda r: r["rbkg"])
    else:
        best = min(table, key=lambda r: r["ratio"])

    # 置信度：区分「寻优得到的」与「在噪声里挑出来的」
    scores = [row["score"] for row in table]
    valid_fraction = len(valid) / len(table)
    roughness = float(np.median(np.abs(np.diff(scores))) / (max(scores) or 1.0)) if len(scores) > 1 else 0.0
    top2 = sorted((r["score"] for r in valid), reverse=True)[:2]
    margin = float((top2[0] - top2[1]) / top2[0]) if len(top2) > 1 and top2[0] > 0 else 0.0

    if valid_fraction >= 0.6 and roughness < 0.10 and margin > 0.05:
        confidence = "high"
    elif valid_fraction >= 0.4 and roughness < 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    r1_val = float(best.get("r1", np.nan))
    shell1_ratio = (float(best["rbkg"]) / r1_val
                    if (np.isfinite(r1_val) and r1_val > 0.0) else float("nan"))
    safe_hint = None
    safe_row = None
    if valid and np.isfinite(r1_val) and r1_val > 0.0:
        safe_cap = max(0.95, 0.72 * r1_val)
        safe_cands = [r for r in valid if r["rbkg"] <= safe_cap]
        if safe_cands:
            s_safe_max = max(r["score"] for r in safe_cands)
            tied_safe = [r for r in safe_cands if r["score"] >= 0.95 * s_safe_max]
            safe_row = min(tied_safe, key=lambda r: r["rbkg"])
            safe_hint = float(safe_row["rbkg"])

    intrusion_risk = bool(
        safe_row is not None
        and np.isfinite(shell1_ratio)
        and float(best["rbkg"]) >= 1.25
        and shell1_ratio >= 0.78
        and float(best["rbkg"]) - float(safe_row["rbkg"]) >= 0.20
    )
    intrusion_warning = (
        f"Rbkg={best['rbkg']:.2f} Å 超过常规 ≤1.20 Å 上限并逼近第一壳 r1={r1_val:.2f} Å "
        f"({shell1_ratio:.0%})，可能侵入第一壳左脚；建议对照 safe_rbkg={safe_hint:.2f} Å"
        if intrusion_risk else None
    )

    best.update({
        "confidence": confidence,
        "confidence_metrics": {"valid_fraction": round(valid_fraction, 2),
                               "roughness": round(roughness, 3),
                               "margin": round(margin, 3)},
        "shell1_ratio": round(shell1_ratio, 3) if np.isfinite(shell1_ratio) else None,
        "safe_rbkg_hint": safe_hint,
        "intrusion_risk": intrusion_risk,
        "intrusion_warning": intrusion_warning,
    })
    return best, table


# ================================================================
# 6. 傅里叶变换
# ================================================================

def back_ft_to_q_space(r, chi_r, rmin=1.0, rmax=2.3, dr=0.1) -> BackFourierTransform:
    """R 窗反傅里叶变换（Back-FT）：χ(R) → χ(q)。

    对标 Athena 的「反傅里叶滤波」与 Artemis 的 q 空间拟合诊断：只保留
    [rmin, rmax] 内的壳层，反变换回波数空间，得到该壳层的振荡相位与
    包络。q 轴由 larch.xftr 决定（0–30 Å⁻¹，步长 0.05）。

    数学:
        χ(q) = ∫[rmin,rmax] χ(R)·W_R(R)·e^{+2iqR} dR
    与正变换 e^{−2ikR} 互为逆变换；R 窗用 Hanning（dr 为过渡宽度）。
    """
    r = np.asarray(r, dtype=float)
    chi_r = np.asarray(chi_r, dtype=complex)
    if r.shape != chi_r.shape:
        raise XAFSConfigError(
            "S5", f"back-FT 输入长度不一致: r={r.shape}, chi_r={chi_r.shape}")
    mask = (r >= rmin) & (r <= rmax)
    if int(mask.sum()) < 5:
        raise XAFSConfigError(
            "S5", f"back-FT R 区间 [{rmin},{rmax}] 内仅 {int(mask.sum())} 点")
    from larch.xafs import xftr

    g = Group(r=r, chir=chi_r)
    xftr(g, rmin=float(rmin), rmax=float(rmax), dr=float(dr),
         window="hanning")
    q = np.asarray(g.q, dtype=float)
    chi_q = (np.asarray(g.chiq_re, dtype=float)
             + 1j * np.asarray(g.chiq_im, dtype=float))
    if q.shape != chi_q.shape:
        # larch 的 xftr 在粗 R 网格上会输出更长的 q 轴（到 30 Å⁻¹），
        # 但 chiq 只保留 nfft/2 个有效点（实测 1024）。此时 q 的多余
        # 尾部没有对应的 χ(q)，截断而不是插值/猜测。
        if q.size > chi_q.size and q.ndim == 1 and chi_q.ndim == 1:
            q = q[:chi_q.size]
        else:
            raise XAFSConfigError("S5", "back-FT 输出 q 与 χ(q) 长度不一致")
    return BackFourierTransform(
        q=q, chi_q=chi_q, rmin=float(rmin), rmax=float(rmax),
        dr=float(dr), r_window=np.asarray(g.rwin, dtype=float))


def ft_to_r_space(k, chi, r=None, kmin=3.0, kmax=12.0, kweight=2, dk=1.0,
                  rmax_out: float | None = None,
                  phase_correct=None) -> FourierTransform:
    # r 参数保留是为了向后兼容，但**实际被忽略**——R 网格由 xftf 内部决定。
    """[Implements: EQ-03]
    傅里叶变换到 R 空间。

    数学:
        χ(R) = ∫[kmin,kmax] k^n · χ(k) · W(k) · e^{−2ikR} dk

    关键（易错）:
        相位因子必须是 e^{−2ikR}，与 larch.xftf 一致。
        若误写成 e^{−2πikR}，R 轴会被压缩 π 倍——
        Fe–S 的 1.82 Å 会跑到 0.58 Å，拟合窗口随之全部错位
        （实测会把 R-factor 从 0.02 顶到 0.87）。

    B11-1 相移校正:
        ``phase_correct`` 给 FEFF 路径总相移 φ_tot(k)（长度与 k 相同）。
        变换前乘 ``exp(−iφ_tot)``，峰位从“表观 R”（约 Reff−0.3~0.5 Å）
        回到真实 Reff。默认 None，历史数值逐位不变。
        实现用线性性：xftf(re+i·im)=xftf(re)+i·xftf(im)。

    返回 FourierTransform（含复数 χ(R)，实部虚部都能取）。
    """
    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    if k.ndim != 1 or chi.shape != k.shape:
        raise XAFSConfigError(
            "S5", f"FT 输入 k/chi 维度或长度不一致: k={k.shape}, chi={chi.shape}")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(chi))):
        raise XAFSDataError("S5", "FT 输入 k 或 chi 含非有限值（NaN/Inf）")

    mask = (k >= kmin) & (k <= kmax)
    if mask.sum() < 8:
        raise XAFSConfigError("S5", f"k 区间 [{kmin},{kmax}] 内仅 {mask.sum()} 点")

    # 内部调用 larch.xftf 而不是自行搭 FFT。
    # 原因：FT 的归一化因子、窗函数端点处理、相位符号有多种约定，
    # 自研版本与 larch 实测差 0.053 Å（1.420 vs 1.473），且对 dk 不敏感；
    # 而 0.05 Å 会直接传导到拟合出的 R。既然 Athena/Artemis 用的就是这套，
    # 这里保持一致，避免"看起来对、实际错位"。
    from larch import Group
    from larch.xafs import xftf

    xftf_kw = dict(kmin=kmin, kmax=kmax, kweight=kweight, dk=dk,
                   window="hanning")
    # 默认不传 rmax_out，保持历史 R 网格逐位一致（现有 FT/拟合/报告都
    # 消费这条网格）。只有高 R 噪声估计等明确需要远区的调用才扩展。
    if rmax_out is not None:
        xftf_kw["rmax_out"] = float(rmax_out)

    def _one(y):
        g = Group(k=np.asarray(k), chi=np.asarray(y))
        xftf(g, **xftf_kw)
        return (np.asarray(g.chir_re) + 1j * np.asarray(g.chir_im),
                np.asarray(g.r))

    if phase_correct is None:
        chi_r, r_native = _one(chi)
    else:
        phi = np.asarray(phase_correct, dtype=float)
        if phi.ndim != 1 or phi.size != k.size:
            raise XAFSConfigError(
                "S5", f"phase_correct 需与 k 等长（{k.size}），"
                      f"收到 shape={phi.shape}")
        if not np.all(np.isfinite(phi)):
            raise XAFSConfigError("S5", "phase_correct 含非有限值")
        rotated = chi * np.exp(-1j * phi)      # Re + i·Im
        ft_re, r_native = _one(np.real(rotated))
        ft_im, r_im = _one(np.imag(rotated))
        if r_im.shape != r_native.shape:
            raise XAFSConfigError("S5", "相移校正 FT 的 R 网格不一致")
        # FT(re + i·im) = FT(re) + i·FT(im)
        chi_r = ft_re + 1j * ft_im

    return FourierTransform(r=r_native, chi_r=chi_r, kmin=float(kmin),
                            kmax=float(kmax), kweight=int(kweight), dk=float(dk))
