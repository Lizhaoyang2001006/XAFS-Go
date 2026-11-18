# -*- coding: utf-8 -*-
"""athena_merge.py — # 1 多扫描合并、对齐、去毛刺、异常扫描剔除（自 athena.py 纯拆分）

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

# 跨子模块唯一依赖：align_by_reference 调用 determine_e0
from athena_norm import determine_e0  # noqa: E402,F401


def align_by_reference(scans, edge_energy: float, tolerance: float = 0.2):
    """
    用参考通道的边位移把各扫描平移到同一能量刻度。

    数学:
        ΔE_i = E_ref_standard − E_ref_measured,i
        E_corrected = E + ΔE_i          （整体平移，不做插值）

    约束:
        样品与参考**共用同一个 ΔE**，不各自对齐。
        参考通道必须「独立」——由采集拓扑判定（同时有 It 与 Ir 的串联三
        chamber，μ_ref=ln(It/Ir)），绝不靠边位差反推。非独立（缺 It、
        Ir∝It、或文件头标注同一吸收体）时拒绝对齐，而不是硬凑一个平移量。

    返回: (对齐后的 scans, info 字典)
    """
    e0_refs_collected = []
    for i, scan in enumerate(scans):
        # guess=edge_energy：文件头可能无 e0_nominal（SIMPLE 格式），
        # 边窗口必须用调用方已解析的标称边能，绝不回退到任何默认元素
        info = determine_e0(scan, guess=edge_energy)
        if info.get("e0_reference") is None:
            reason = info.get("reference_warning") or "无独立参考通道（Ir）"
            raise XAFSDataError(
                "S2", f"扫描 {i+1} 参考通道不可用于能量对齐：{reason}")
        e0_refs_collected.append(info["e0_reference"])

    e0_ref = np.asarray(e0_refs_collected)
    shifts = edge_energy - e0_ref

    aligned = []
    for scan, dE in zip(scans, shifts):
        moved = RawScan(
            energy=np.asarray(scan.energy) + dE,
            i0=scan.i0, it=scan.it, ifluor=scan.ifluor, iref=scan.iref,
            dtime=scan.dtime, mu=scan.mu, sigma_mu=scan.sigma_mu,
            meta=dict(scan.meta))
        moved._measurement_identity = xafs_io.capture_measurement_identity(scan)
        moved.meta["align_shift_eV"] = float(dE)
        aligned.append(moved)

    e0_after = np.array([determine_e0(s, guess=edge_energy)["e0_sample"]
                         for s in aligned])
    residual = float(np.std(e0_after))
    return aligned, {
        "e0_reference_per_scan": np.round(e0_ref, 3).tolist(),
        "shifts_eV": np.round(shifts, 3).tolist(),
        "e0_sample_after": np.round(e0_after, 3).tolist(),
        "residual_std_eV": round(residual, 3),
        "ok": bool(residual < tolerance),
        "note": f"对齐后样品边位标准差 {residual:.3f} eV",
    }


def merge_scans(scans, reject_outliers: bool = False,
                z_threshold: float = 3.5) -> RawScan:
    """
    多次扫描求算术平均（可选异常扫描剔除 + 逐点标准差）。

    数学:
        μ̄(E) = (1/N) Σ_i μ_i(E)
        σ_μ(E) = std_i μ_i(E)（保留扫描，ddof=1）

    网格一致时直接平均；不一致时线性插值到公共网格。
    **不用三次样条**——吸收边是陡跳变，样条会过冲。

    异常扫描判据（B6-1，``reject_outliers=True`` 且至少 3 条扫描）：
        d_i = RMS_E[μ_i(E) − median_j μ_j(E)]
        d_i > median(d) + z·max(1.4826·MAD(d), 1e-4)
        且 d_i > 0.02·ptp(median spectrum)
    保底护栏：至少保留 2 条最接近中位谱的扫描，绝不全部剔光。
    默认 ``reject_outliers=False`` 时 energy/mu 与原实现逐位一致，
    仅额外填充 ``sigma_mu`` 与 ``meta["merge_info"]``。
    """
    if not scans:
        raise XAFSConfigError("S1.5", "没有可合并的扫描")
    if any(s.mu is None for s in scans):
        raise XAFSDataError("S1.5", "存在缺少 μ(E) 的扫描，无法合并")

    grids_equal = all(
        len(s.energy) == len(scans[0].energy)
        and np.allclose(s.energy, scans[0].energy)
        for s in scans)

    if grids_equal:
        energy = np.asarray(scans[0].energy)
        mu_stack = np.array([s.mu for s in scans])
        interpolated = False
    else:
        lo = max(float(s.energy.min()) for s in scans)
        hi = min(float(s.energy.max()) for s in scans)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 0.2:
            raise XAFSDataError("S1", f"多扫描能量区间无交集或过窄 ([{lo:.2f}, {hi:.2f}] eV)，无法合并")
        energy = np.arange(lo, hi, 0.2)
        mu_stack = np.array([np.interp(energy, s.energy, s.mu) for s in scans])
        interpolated = True

    n_scan = len(scans)
    # 默认路径保持与原实现完全相同的求和顺序与数值。
    mu_mean = mu_stack.mean(axis=0)

    # ---- B6-1 相异度筛查：中位谱 RMS 偏离 + MAD 门禁 ----
    deviations = np.zeros(n_scan, dtype=float)
    mu_median = None
    if n_scan >= 2:
        mu_median = np.median(mu_stack, axis=0)
        deviations = np.sqrt(np.mean((mu_stack - mu_median) ** 2, axis=1))

    kept = np.ones(n_scan, dtype=bool)
    rejected_indices: list[int] = []
    if reject_outliers and n_scan >= 3:
        d_med = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - d_med)))
        robust_sigma = max(1.4826 * mad, 1e-4)
        threshold = d_med + float(z_threshold) * robust_sigma
        ptp = float(np.ptp(mu_median)) if mu_median is not None else 0.0
        for i, d_i in enumerate(deviations):
            if d_i > threshold and d_i > 0.02 * ptp:
                kept[i] = False
        # 保底：至少保留 2 条最接近中位谱的扫描
        if int(kept.sum()) < 2:
            order = np.argsort(deviations)[:2]
            kept[:] = False
            kept[order] = True
        rejected_indices = [int(i) for i in np.flatnonzero(~kept)]
        if rejected_indices:
            mu_mean = mu_stack[kept].mean(axis=0)

    single = n_scan == 1
    rms = [float(np.sqrt(np.mean((m - mu_mean) ** 2))) for m in mu_stack]
    if int(kept.sum()) >= 2:
        sigma_mu = np.std(mu_stack[kept], axis=0, ddof=1)
        sigma_mu = np.asarray(sigma_mu, dtype=float)
    else:
        sigma_mu = None

    meta = dict(scans[int(np.flatnonzero(kept)[0])].meta)
    meta.update({
        "n_scans": n_scan,
        "grids_identical": bool(grids_equal),
        "interpolated": interpolated,
        "scan_rms": None if single else rms,
        "repeatability_note": "单扫描，无法评估重复性" if single else None,
        "merge_info": {
            "n_input": n_scan,
            "n_kept": int(kept.sum()),
            "rejected_indices": rejected_indices,
            "scan_rms_deviations": [float(v) for v in deviations],
            "z_threshold": float(z_threshold),
            "reject_outliers": bool(reject_outliers),
        },
    })
    # ---- 保留测量模式与通道来源（审查 item 12）----
    # 历史缺陷：合并只留 μ，把 I0/It/If/Ir 全丢成 None。荧光谱合并后失去
    # 通道证据（S4.5 自吸收门禁的 _has_if 失效、given 模式荧光被误拒），
    # 参考通道也无法在 S2 交叉校验。这里在公共网格上对 kept 扫描逐通道
    # 算术平均，保留物理身份与溯源。
    # μ 仍取逐扫描 μ 的均值（mu_mean，默认路径逐位不变），绝不改成
    # ln(mean I0 / mean It)——均值对数 ≠ 对数均值，后者会引入偏差；
    # 平均强度通道仅作溯源/参考，下游一律以 merged.mu 为准。
    kept_idx = np.flatnonzero(kept)

    def _avg_channel(attr):
        vals = [getattr(scans[i], attr) for i in kept_idx]
        if any(v is None for v in vals):
            return None                 # 通道不齐：不臆造平均，置 None
        if grids_equal:
            stack = np.array([np.asarray(v, dtype=float) for v in vals])
        else:
            stack = np.array([
                np.interp(energy, np.asarray(scans[i].energy, dtype=float),
                          np.asarray(v, dtype=float))
                for i, v in zip(kept_idx, vals)])
        return stack.mean(axis=0)

    i0_m = _avg_channel("i0")
    it_m = _avg_channel("it")
    if_m = _avg_channel("ifluor")
    iref_m = _avg_channel("iref")

    modes = [str((s.meta or {}).get("mode")) for s in scans]
    mode_consistent = len(set(modes)) == 1
    meta["merge_channels"] = {
        "mode": meta.get("mode"),
        "mode_consistent": mode_consistent,
        "input_modes": modes,
        "averaged": [nm for nm, arr in (("i0", i0_m), ("it", it_m),
                                        ("ifluor", if_m), ("iref", iref_m))
                     if arr is not None],
        "mu_is_mean_of_per_scan_mu": True,
        # 死时间/MCA 坏道校正已在 S1.2 逐扫描完成；合并后 dtime 置 None
        # 防止二次校正，校正状态记入溯源。
        "deadtime_applied_per_scan": any(
            bool((s.meta or {}).get("deadtime_corrected")) for s in scans),
        "note": ("强度通道为 kept 扫描在公共网格上的算术平均，仅作溯源/参考；"
                 "μ 取逐扫描 μ 均值，不得由平均 I0/It 重算"),
    }
    if not mode_consistent:
        meta["merge_channels"]["warning"] = (
            f"输入扫描测量模式不一致 {modes}，合并 μ 的物理意义存疑")

    identities = [xafs_io.measurement_identity(scans[i]) for i in kept_idx]
    modes_used = {identity[0] for identity in identities}
    is_fluorescence = (identities[0][0] if len(modes_used) == 1 else None)
    basis = ("保留扫描原始身份：" + "；".join(identity[1] for identity in identities))
    if len(modes_used) > 1:
        basis = "保留扫描测量身份不兼容；" + basis
    meta["merge_channels"]["retained_indices"] = [int(i) for i in kept_idx]
    meta["measurement_identity"] = {
        "is_fluorescence": is_fluorescence, "basis": basis, "stage": "retained"}
    merged = RawScan(energy=energy, mu=mu_mean, i0=i0_m, it=it_m,
                     ifluor=if_m, iref=iref_m, dtime=None,
                     sigma_mu=sigma_mu, meta=meta)
    merged._measurement_identity = (is_fluorescence, basis)
    return merged


def _copy_raw_scan(scan: RawScan) -> RawScan:
    """深拷贝 RawScan 的逐点数组（不共享可变状态）。"""
    identity = xafs_io.capture_measurement_identity(scan)
    copied = RawScan(
        energy=np.array(scan.energy, dtype=float, copy=True),
        i0=None if scan.i0 is None else np.array(scan.i0, dtype=float, copy=True),
        it=None if scan.it is None else np.array(scan.it, dtype=float, copy=True),
        ifluor=None if scan.ifluor is None else np.array(scan.ifluor,
                                                         dtype=float, copy=True),
        iref=None if scan.iref is None else np.array(scan.iref,
                                                     dtype=float, copy=True),
        dtime=None if scan.dtime is None else np.array(scan.dtime,
                                                       dtype=float, copy=True),
        mu=None if scan.mu is None else np.array(scan.mu, dtype=float, copy=True),
        sigma_mu=(None if scan.sigma_mu is None else
                  np.array(scan.sigma_mu, dtype=float, copy=True)),
        meta=dict(scan.meta or {}),
    )
    copied._measurement_identity = identity
    return copied


def apply_deadtime_correction(scan: RawScan) -> RawScan:
    """B2-3 荧光探测器死时间校正（逐点）。

    约定（对标 Athena 的 dtime 列）：
        dtime >= 1.0   → 倍乘因子：I_corr = I_raw · dtime
        0 < dtime < 1  → 死时间分数：I_corr = I_raw / (1 − dtime)
        dtime <= 0 / 非有限值 → 该点不校正（factor=1）

    只改荧光强度并重算 μ = I_f^corr / I_0；若没有 I_f 但已有 μ，则按
    同一 factor 缩放 μ（等价于 I_f 校正后除以同一 I_0）。透射模式不做
    死时间校正（透射探测器没有荧光死时间语义）。
    """
    out = _copy_raw_scan(scan)
    mode = (scan.meta or {}).get("mode")
    if mode not in (None, "fluor"):
        # 透射/文件自带 xmu 的 dtime 列是积分时间，不是死时间因子；
        # 直接乘/除会把整条谱系统性地缩放。
        out.meta["deadtime_corrected"] = False
        out.meta["deadtime_note"] = f"mode={mode}，非荧光模式，跳过死时间校正"
        return out
    if scan.dtime is None:
        if (scan.meta or {}).get("deadtime_corrected"):
            out.meta["deadtime_corrected"] = True
            out.meta["deadtime_note"] = "MCA 通道级死时间已校正（无逐点 dtime 列）"
        else:
            out.meta["deadtime_corrected"] = False
            out.meta["deadtime_note"] = "无 dtime 列"
        return out
    d = np.asarray(scan.dtime, dtype=float)
    if d.ndim == 0:
        d = np.full(len(scan.energy), float(d))
    if d.shape != np.asarray(scan.energy).shape:
        if d.size == 1:
            d = np.full(len(scan.energy), float(d.ravel()[0]))
        else:
            raise XAFSConfigError(
                "S1.2", f"dtime 长度 {d.shape} 与能量 {np.shape(scan.energy)} 不一致")
    factor = np.ones_like(d, dtype=float)
    mul = np.isfinite(d) & (d >= 1.0)
    frac = np.isfinite(d) & (d > 0.0) & (d < 1.0)
    factor[mul] = d[mul]
    factor[frac] = 1.0 / (1.0 - d[frac])
    n_bad = int(np.sum(~(mul | frac)))
    n_eff = int(np.sum(mul | frac))

    if scan.ifluor is not None:
        out.ifluor = np.asarray(scan.ifluor, dtype=float) * factor
        if scan.i0 is not None:
            i0 = np.asarray(scan.i0, dtype=float)
            out.mu = out.ifluor / np.where(i0 != 0, i0, np.nan)
        elif scan.mu is not None:
            out.mu = np.asarray(scan.mu, dtype=float) * factor
        else:
            raise XAFSConfigError(
                "S1.2", "死时间校正需要 I_0（重算 μ = I_f/I_0）或已有 μ")
    elif scan.mu is not None:
        out.mu = np.asarray(scan.mu, dtype=float) * factor
    else:
        raise XAFSConfigError(
            "S1.2", "死时间校正需要荧光强度 I_f 或已算好的 μ")
    out.meta["deadtime_corrected"] = bool(n_eff > 0)
    out.meta["deadtime_note"] = (f"{n_eff}/{len(d)} 点有效校正"
                                 + (f"，{n_bad} 点无有效因子" if n_bad else ""))
    out.meta["deadtime_factor_range"] = (float(np.min(factor)),
                                         float(np.max(factor)))
    return out


def deglitch_ranges(scan: RawScan, ranges: list[tuple[float, float]]):
    """B2-3 显式能量区间去 glitch：直接删点，不插值伪造数据。

    安全门禁（对标 Athena 的 deglitch 保守原则）：
      - 删除比例 > 15% → 拒绝（多半是区间写错/整段谱被掏空）；
      - 吸收边核心 [E0−10, E0+10] eV 剩余 < 5 点 → 拒绝；
      - E0 未知（meta 无 e0_nominal）时跳过核心区检查并记录。

    返回 ``(new_scan, info)``；``info`` 含 n_total/n_removed/fraction/ranges。
    """
    if not ranges:
        raise XAFSConfigError("S1.2", "deglitch 区间列表为空")
    energy = np.asarray(scan.energy, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise XAFSConfigError("S1.2", "deglitch 需要一维非空能量数组")
    mask = np.zeros(energy.size, dtype=bool)
    norm_ranges = []
    for item in ranges:
        try:
            lo, hi = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise XAFSConfigError(
                "S1.2", f"deglitch 区间格式错误: {item!r}（应 (e_lo, e_hi)）") from exc
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise XAFSConfigError("S1.2", f"deglitch 区间含非有限值: {item!r}")
        if hi < lo:
            lo, hi = hi, lo
        norm_ranges.append((lo, hi))
        mask |= (energy >= lo) & (energy <= hi)

    n_total = int(energy.size)
    n_removed = int(mask.sum())
    fraction = float(n_removed / n_total) if n_total else 0.0
    if fraction > 0.15:
        raise XAFSConfigError(
            "S1.2", f"deglitch 将删除 {n_removed}/{n_total} = {fraction:.1%} "
                    f"数据点（>15% 安全上限）；请缩小区间或检查能量单位")

    e0 = None
    if scan.meta:
        e0 = scan.meta.get("e0_nominal")
    core_remaining = None
    if e0 is not None and np.isfinite(float(e0)):
        core = (energy >= float(e0) - 10.0) & (energy <= float(e0) + 10.0)
        core_remaining = int(np.sum(core & ~mask))
        if int(core.sum()) >= 5 and core_remaining < 5:
            raise XAFSConfigError(
                "S1.2", f"deglitch 把吸收边核心 [{float(e0)-10:.1f},"
                        f"{float(e0)+10:.1f}] eV 掏空（剩余 {core_remaining} 点 <5）；"
                        f"E0 将不可靠，拒绝执行")

    keep = ~mask
    out = _copy_raw_scan(scan)
    out.energy = energy[keep]
    for attr in ("i0", "it", "ifluor", "iref", "dtime", "mu", "sigma_mu"):
        arr = getattr(scan, attr)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.shape == energy.shape:
                setattr(out, attr, arr[keep])
            else:
                setattr(out, attr, None)
    info = {
        "n_total": n_total,
        "n_removed": n_removed,
        "fraction": fraction,
        "ranges": norm_ranges,
        "edge_core_remaining": core_remaining,
        "e0_nominal": None if e0 is None else float(e0),
    }
    out.meta["deglitch"] = info
    return out, info


def despike_scan(scan: RawScan, kernel: int = 5,
                 threshold: float = 3.0) -> tuple[RawScan, dict]:
    """S1.2 单点去毛刺（中值滑窗 + MAD 门禁）。

    注意与 ``deglitch_ranges`` 的分工：
      - ``despike_scan`` 仅适合宽度 1–2 个采样点的孤立电子学毛刺；
      - 宽度 ≥ kernel//2 + 1（如 5–15 eV 的单色器晶体 glitch）会让滑窗中值
        本身被抬高，必须用 ``deglitch_ranges`` 显式删点，严禁指望自动滤波。
    """
    out = _copy_raw_scan(scan)
    mu = np.asarray(out.mu, dtype=float).copy()
    half = max(1, int(kernel) // 2)
    win = 2 * half + 1
    padded = np.pad(mu, half, mode="edge")
    med = np.array([np.median(padded[i:i + win]) for i in range(len(mu))])
    resid = np.abs(mu - med)
    mad = float(np.median(resid)) + 1e-12
    spike_mask = resid > float(threshold) * mad
    n_spike = int(spike_mask.sum())
    if n_spike:
        mu[spike_mask] = med[spike_mask]
        out.mu = mu
    info = {"n_spikes": n_spike, "kernel": win, "threshold_mad": float(threshold)}
    out.meta["despike"] = info
    return out, info


def merge_mca_channels(channels, dtime=None, z_threshold: float = 3.5,
                       min_channels: int = 2):
    """B9-3 多通道荧光探测器（MCA/多电离室）坏道剔除与加权合并。

    步骤:
      1. 逐道死时间校正（与 ``apply_deadtime_correction`` 同一约定）；
      2. 以通道间中位谱为参考，计算每道的稳健 z 分数
         ``max|I−I_med| / (1.4826·MAD)``；Bragg spike 会在少数能量点
         把 max 拉爆，因此比整道 RMS 更敏感；
      3. 按 z 从大到小剔除，但保底 ``min_channels`` 道；
      4. 以各道边阶（用 90 分位−10 分位近似）为权重加权求和——
         边阶高的通道信噪比更好；边阶全为 0 时退化为等权。

    返回 ``(summed, info)``；``info`` 含 ``rejected``、``n_kept``、
    ``weights``、``scores``。``channels`` 形状 ``(n_energy, n_channel)``。
    """
    arr = np.asarray(channels, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise XAFSConfigError(
            "S1", f"MCA channels 需为 (n_energy, n_channel)，收到 {arr.shape}")
    n_energy, n_ch = arr.shape
    if not np.all(np.isfinite(arr)):
        raise XAFSConfigError("S1", "MCA channels 含非有限值")

    corr = arr.copy()
    dt = None
    if dtime is not None:
        dt = np.asarray(dtime, dtype=float)
        if dt.ndim == 1:
            if dt.size != n_ch:
                raise XAFSConfigError(
                    "S1", f"dtime 长度 {dt.size} 与通道数 {n_ch} 不一致")
            factor = np.where(dt >= 1.0, dt,
                              np.where((dt > 0) & (dt < 1), 1.0 / (1.0 - dt), 1.0))
            corr = corr * factor[None, :]
        elif dt.shape == (n_energy, n_ch):
            factor = np.where(dt >= 1.0, dt,
                              np.where((dt > 0) & (dt < 1), 1.0 / (1.0 - dt), 1.0))
            corr = corr * factor
        else:
            raise XAFSConfigError(
                "S1", f"dtime 需为 (n_ch,) 或 (n_energy, n_ch)，收到 {dt.shape}")

    med = np.median(corr, axis=1, keepdims=True)
    diff = corr - med
    mad = 1.4826 * np.median(np.abs(diff), axis=0)
    global_scale = float(np.median(mad[mad > 0])) if np.any(mad > 0) else 0.0
    if global_scale <= 0:
        global_scale = float(np.median(np.abs(diff))) or 1.0
    # 双判据：整道稳健 RMS 超标（增益/死时间异常）或少量极大离群点
    # （Bragg spike）。单用 max|diff| 会被 400 点高斯噪声的多重比较
    # 误伤——干净道 max z 也会到 3.5–4。
    rms_z = np.sqrt(np.mean(diff ** 2, axis=0)) / global_scale
    max_z = np.max(np.abs(diff), axis=0) / global_scale
    n_out = np.sum(np.abs(diff) > float(z_threshold) * global_scale, axis=0)
    bad = (rms_z > float(z_threshold)) | (
        (max_z > 2.0 * float(z_threshold)) & (n_out >= 2))
    scores = rms_z

    order = [int(i) for i in np.argsort(scores)[::-1]]
    rejected: list[int] = []
    keep = list(range(n_ch))
    for idx in order:
        if len(keep) <= max(1, int(min_channels)):
            break
        if bad[idx]:
            rejected.append(int(idx))
            keep.remove(int(idx))
    rejected.sort()

    spans = (np.percentile(corr[:, keep], 90, axis=0)
             - np.percentile(corr[:, keep], 10, axis=0))
    spans = np.maximum(spans, 0.0)
    if float(np.sum(spans)) > 0:
        weights = spans / float(np.sum(spans))
    else:
        weights = np.full(len(keep), 1.0 / len(keep))
    summed = np.sum(corr[:, keep] * weights[None, :], axis=1)

    info = {
        "rejected": rejected,
        "n_kept": len(keep),
        "kept": keep,
        "weights": [float(w) for w in weights],
        "scores": [float(s) for s in scores],
        "max_z": [float(s) for s in max_z],
        "outlier_counts": [int(v) for v in n_out],
        "z_threshold": float(z_threshold),
        "deadtime_corrected": bool(dt is not None),
        "n_channels": int(n_ch),
    }
    return summed, info
