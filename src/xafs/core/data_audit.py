# -*- coding: utf-8 -*-
"""data_audit.py — XAFS 数据预处理与标尺自审计（预防静默低级错误）。

职责（Layer 4 `core/`）：在 S1–S5 归约完成后统一检查 6 类高频隐性故障，
产出结构化审计结论而不篡改历史基准数值（红线 5）：
  1. Athena `.prj` 未应用的 `bkg_eshift` 校准偏移；
  2. 吸收步跳极性反转（`edge_step <= 0`）或极低边跳；
  3. 能量网格卡滞/近重合点（`ΔE < 0.02 eV`）；
  4. 归一化边后平台越界或靠近边界（且后沿可延展提示）；
  5. `optimize_rbkg` 第一壳入侵风险（`rbkg / r1 >= 0.78` 且虚抬振幅）；
  6. 高 k 尾端噪声截断建议（同时覆盖多扫描 `σ_μ` 与单扫描二阶差分估计）。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _check_entry(check_id: str, severity: str, message: str,
                 plot_message: str | None = None, **extra) -> dict[str, Any]:
    item = {
        "id": check_id,
        "severity": severity,
        "message": message,
        "plot_message": plot_message or message,
    }
    if extra:
        item.update(extra)
    return item


def audit_preprocessing(res: Any, kmax_used: float | None = None) -> dict[str, Any]:
    """对 S1–S6 ``PipelineResult`` 执行全链路物理与信号自审计。"""
    checks: list[dict[str, Any]] = []
    scans = list(getattr(res, "scans", None) or [])
    merged = getattr(res, "merged", None)
    norm = getattr(res, "norm", None)
    rbkg = getattr(res, "rbkg", None) or {}
    ft = getattr(res, "ft", None)
    snr = getattr(res, "snr", None) or {}

    if kmax_used is None and ft is not None:
        kmax_used = getattr(ft, "kmax", None)
    kmax_eff = float(kmax_used) if kmax_used is not None and np.isfinite(float(kmax_used)) else None

    # 1. Athena .prj 未应用 eshift 检查
    unapplied_eshifts = []
    for idx, sc in enumerate(scans):
        meta = getattr(sc, "meta", None) or {}
        esh = meta.get("eshift")
        if esh is not None and abs(float(esh)) > 1e-3 and not meta.get("eshift_applied", False):
            lbl = meta.get("label") or meta.get("path") or f"scan_{idx}"
            unapplied_eshifts.append(f"{lbl} ({float(esh):+.3f} eV)")
    if unapplied_eshifts:
        checks.append(_check_entry(
            "prj_eshift_unapplied", "warn",
            "检测到 Athena .prj 记录的能量偏移未平移: " + ", ".join(unapplied_eshifts)
            + "（对照样品/LCF 前请确认能量标尺对齐）",
            plot_message=".prj eshift unapplied: " + ", ".join(unapplied_eshifts),
        ))

    # 2. 吸收边跳极性与量级
    if norm is not None:
        step = float(getattr(norm, "edge_step", np.nan))
        if np.isfinite(step) and step <= 0.0:
            checks.append(_check_entry(
                "step_polarity", "fail",
                f"吸收边跳 Δμ₀={step:.4g} ≤ 0：疑似透射/荧光模式或 I0/It 通道方向颠倒",
                plot_message=f"Edge step dmu0={step:.4g} <= 0 (check polarity)",
            ))
        elif np.isfinite(step) and step < 0.05:
            checks.append(_check_entry(
                "low_edge_step", "warn",
                f"吸收边跳偏低（Δμ₀={step:.4f} < 0.05）：超稀样品，须重点核查高 k 噪声截断",
                plot_message=f"Low edge step dmu0={step:.4f} < 0.05 (dilute)",
            ))

        # 4. 边后平台临界提示
        plat = float(getattr(norm, "plateau", np.nan))
        e0_val = float(getattr(norm, "e0", np.nan))
        e_arr = np.asarray(getattr(norm, "energy", []), dtype=float)
        if np.isfinite(plat):
            if not (0.85 <= plat <= 1.15):
                checks.append(_check_entry(
                    "normalization_plateau", "fail",
                    f"边后归一化平台 {plat:.3f} 超出 [0.85, 1.15]",
                    plot_message=f"Plateau={plat:.3f} outside [0.85, 1.15]",
                ))
            elif plat > 1.09 or plat < 0.90:
                span = float(np.nanmax(e_arr) - e0_val) if (e_arr.size and np.isfinite(e0_val)) else 0.0
                hint = ("；数据后沿覆盖充足，可尝试放宽 post_window=(150, 780)"
                        if span >= 780.0 else "")
                checks.append(_check_entry(
                    "normalization_plateau_edge", "warn",
                    f"边后平台={plat:.3f} 接近 [0.85, 1.15] 门限边界{hint}",
                    plot_message=f"Plateau={plat:.3f} near [0.85, 1.15] edge",
                ))

    # 3. 能量网格卡滞/极小步长检查
    if merged is not None and getattr(merged, "energy", None) is not None:
        e_m = np.asarray(merged.energy, dtype=float)
        if e_m.size >= 5:
            de = np.diff(e_m)
            n_tiny = int(np.sum((de >= 0.0) & (de < 0.02)))
            if n_tiny >= 2:
                checks.append(_check_entry(
                    "energy_tiny_steps", "warn",
                    f"能量轴存在 {n_tiny} 处极小步进 (ΔE<0.02 eV)，可能影响一阶导数 E0/微商平滑",
                    plot_message=f"Energy grid has {n_tiny} tiny steps (<0.02 eV)",
                ))

    # 5. Rbkg 第一壳入侵风险
    safe_rbkg_hint = rbkg.get("safe_rbkg_hint")
    if rbkg.get("intrusion_risk") and rbkg.get("intrusion_warning"):
        hint_s = f" -> try {float(safe_rbkg_hint):.2f}" if safe_rbkg_hint is not None else ""
        r_val = rbkg.get("rbkg")
        checks.append(_check_entry(
            "rbkg_first_shell_intrusion", "warn",
            str(rbkg["intrusion_warning"]),
            plot_message=f"Rbkg={float(r_val):.2f} near shell-1{hint_s}" if r_val is not None else "Rbkg near shell-1",
            safe_rbkg_hint=safe_rbkg_hint,
        ))

    # 6. 高 k 噪声边界检查（多扫描 sigma_mu 优先，缺省回退单扫描二阶差分估计）
    k_snr2 = snr.get("k_snr2")
    snr_source = "multi_scan_sigma_mu"
    single_est = snr.get("single_scan_estimate") or {}
    if k_snr2 is None and single_est.get("k_snr2") is not None:
        k_snr2 = single_est.get("k_snr2")
        snr_source = "single_scan_2nd_diff"

    rec_kmax = None
    if k_snr2 is not None and kmax_eff is not None and float(k_snr2) < kmax_eff - 0.3:
        rec_kmax = float(np.floor(float(k_snr2) * 2.0) / 2.0)
        checks.append(_check_entry(
            "high_k_noise_cutoff", "warn",
            f"高 k 段在 k≈{float(k_snr2):.1f} Å⁻¹ 处信噪比已跌破 2（{snr_source}），"
            f"低于当前拟合/FT 上限 kmax={kmax_eff:.1f}；建议收紧至 kmax≤{rec_kmax:.1f} Å⁻¹",
            plot_message=f"High-k SNR<2 at k={float(k_snr2):.1f} (use kmax<={rec_kmax:.1f})",
            recommended_kmax=rec_kmax,
            k_snr2=float(k_snr2),
            snr_source=snr_source,
        ))
    elif single_est.get("tail_rise_ratio") is not None and float(single_est["tail_rise_ratio"]) >= 0.90:
        tail_hf = single_est.get("tail_hf_ratio")
        rel_kmax = single_est.get("reliable_kmax")
        noisy_hf = (tail_hf is None or float(tail_hf) >= 0.20)
        low_rel = (rel_kmax is None or float(rel_kmax) < 13.0)
        if noisy_hf and low_rel and (kmax_eff is None or kmax_eff > 10.5):
            checks.append(_check_entry(
                "high_k_tail_amplification", "warn",
                f"尾端 k^wχ(k) 振幅不降反升（tail/mid={float(single_est['tail_rise_ratio']):.2f}），"
                "警惕高 k 荧光噪声污染，建议核查 ≤10.5 Å⁻¹ 截断",
                plot_message=f"High-k tail rise ({float(single_est['tail_rise_ratio']):.2f}); check kmax<=10.5",
                recommended_kmax=10.5,
            ))

    fails = [c["message"] for c in checks if c["severity"] == "fail"]
    warnings = [c["message"] for c in checks if c["severity"] == "warn"]
    plot_fails = [c["plot_message"] for c in checks if c["severity"] == "fail"]
    plot_warnings = [c["plot_message"] for c in checks if c["severity"] == "warn"]
    status = "fail" if fails else ("warn" if warnings else "ok")
    return {
        "status": status,
        "checks": checks,
        "fails": fails,
        "warnings": warnings,
        "plot_fails": plot_fails,
        "plot_warnings": plot_warnings,
        "recommended_kmax": rec_kmax,
        "safe_rbkg_hint": float(safe_rbkg_hint) if safe_rbkg_hint is not None else None,
    }
