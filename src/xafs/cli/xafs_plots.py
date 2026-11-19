# -*- coding: utf-8 -*-
"""xafs_plots.py — 催化剂结构解析导向的统一绘图系统。

目标：每一张输出图都直接回答催化局域结构解析中的物理问题：

  1. ``<stem>_structure_main.png`` — 催化剂配位结构剖析总图（6 面板）：
     (a) 归一化 XANES 与白线特征（价态/d带空穴）
     (b) XANES 一阶导数 dμ/dE（边移与前边对称性破缺）
     (c) k^w χ(k) 实验 vs 拟合模型 + 残差（检查高 k 金属/重载体散射）
     (d) 实空间拟合与**逐路径分壳分解**（|χ(R)| + Re χ(R) + 各壳独立贡献）
     (e) Morlet 二维小波 |W(R,k)| + 壳层脊心（轻配体 vs 重原子定性）
     (f) 结构归属、拟合参数与红绿灯物理门禁卡片
  2. ``<stem>_shell_dissection.png`` — 壳层轻重鉴别与真实键长核验图（3 面板）：
     (a) k¹/k²/k³ 多权重归一化 |FT| 对比（第二壳轻配体 vs 金属/载体判别金标准）
     (b) 相位修正前后 |χ(R)| 对比（表观 R → 真实原子间距 R，消除 0.3–0.5 Å 偏差错觉）
     (c) 逐壳层 1D WT k-轮廓与理论元素 k_peak 标尺对比
  3. ``<stem>_candidates_compare.png``（`--plot full` 且存在识别候选排名时生成）：
     实验谱 vs Top-1 及主要竞争候选构型理论谱对比（SAC vs 团簇 vs 体相氧化物）
  4. 按需专场景图：非高斯 g(R)（C3/C4 开启时）、二维 χ² 等高线（`--chi2-contour`）、
     原位演化图（`--operando`）、发表级精简版式（`--plot paper`）
"""
from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import artemis
import athena
import larch_wt
import structure_id
import xafs_plot_special as special
import xafs_writer
from xafs_core import PipelineResult


def hann_window(k, kmin: float, kmax: float, dk: float) -> np.ndarray:
    """与 larch / Athena / xafs_pipeline 约定一致的 Hanning 窗（两端 0、平台 1）。"""
    k = np.asarray(k, dtype=float)
    w = np.zeros_like(k)
    inside = (k >= kmin) & (k <= kmax)
    w[inside] = 1.0
    if dk > 0:
        half = min(dk, max(0.0, 0.5 * (kmax - kmin)))
        left = inside & (k < kmin + half)
        right = inside & (k > kmax - half)
        if half > 0:
            w[left] = np.sin(0.5 * np.pi * (k[left] - kmin) / half) ** 2
            w[right] = np.cos(0.5 * np.pi * (k[right] - (kmax - half)) / half) ** 2
    return np.clip(w, 0.0, 1.0)


def _save_multi(fig, outdir: Path, stem_name: str,
                formats: tuple[str, ...], dpi: int) -> list[Path]:
    saved = []
    for ext in formats:
        ext_clean = str(ext).strip().lstrip(".") or "png"
        out_path = outdir / f"{stem_name}.{ext_clean}"
        xafs_writer.safe_savefig(fig, out_path, dpi=int(dpi), bbox_inches="tight")
        saved.append(out_path)
    return saved


def _white_line_xy(res: PipelineResult) -> tuple[float, float]:
    e0 = float((res.e0 or {}).get("e0_sample", np.nan))
    norm = res.norm
    e = np.asarray(norm.energy, dtype=float)
    y = np.asarray(norm.mu_norm, dtype=float)
    m = (e >= e0 - 10.0) & (e <= e0 + 30.0)
    if not np.any(m):
        return float("nan"), float("nan")
    idx = int(np.argmax(y[m]))
    return float(e[m][idx]), float(y[m][idx])


def _extract_shells_for_plot(res: PipelineResult, fit_out: dict | None,
                             st: dict | None = None) -> list[tuple[str, float, float]]:
    if st and st.get("shells"):
        return list(st["shells"])
    fit = special._fit_result(fit_out)
    payload = special._fit_payload(fit_out)
    if fit is not None and getattr(fit, "paths", None):
        out = []
        for p in fit.paths:
            r0 = float(p.r)
            out.append((str(p.label), max(0.7, r0 - 0.8), r0 + 0.1))
        return out
    l1 = payload.get("l1") or {}
    if l1.get("R"):
        r0 = float(l1["R"])
        return [(f"{getattr(res, 'absorber_symbol', 'M') or 'M'}-{l1.get('ligand', 'O')}",
                 max(0.7, r0 - 0.8), r0 + 0.1)]
    r1 = float(getattr(res, "r1", np.nan))
    if np.isfinite(r1):
        return [("first peak", max(0.8, r1 - 0.45), r1 + 0.45)]
    return [("first peak", 1.0, 2.6)]


def _wt_text_y(r_base: float, placed_ys: list[float], r_top: float) -> float:
    """为小波面板注释错开 y 坐标，避免相邻壳层文本重叠。"""
    y_txt = min(r_base + 0.06, r_top - 0.14)
    while any(abs(y_txt - py) < 0.17 for py in placed_ys):
        y_txt += 0.18
        if y_txt > r_top - 0.10:
            y_txt = max(0.08, r_base - 0.22)
            break
    placed_ys.append(y_txt)
    return y_txt


def _annotate_wt_shell_panel(ax_e, wt, shell_wins: list[tuple[str, float, float]],
                             wt_sanity: dict, r_top: float, *,
                             explicit_shells: bool = False) -> None:
    """在 Panel (e) 标注 WT 壳层脊位；掩蔽或失配轻壳层严禁误画成重原子白圈峰。"""
    kc = np.asarray(getattr(wt, "k_centers", []), dtype=float)
    kc_min = float(np.nanmin(kc)) if kc.size else float("-inf")
    kc_max = float(np.nanmax(kc)) if kc.size else float("inf")
    sanity_shells = list(wt_sanity.get("shells") or [])
    by_norm: dict[str, dict] = {}
    for s in sanity_shells:
        norm_l = str(s["label"]).replace("–", "-")
        by_norm.setdefault(norm_l, s)
        by_norm.setdefault(norm_l.split("@")[0], s)

    items: list[dict] = []
    if sanity_shells and not explicit_shells:
        items = sanity_shells
    else:
        for label, lo, hi in shell_wins:
            norm_l = str(label).replace("–", "-")
            matched = by_norm.get(norm_l) or by_norm.get(norm_l.split("@")[0])
            if matched is not None:
                items.append({**matched, "label": label})
                continue
            k_p, r_p, _ = larch_wt.locate_shell(wt, lo, hi)
            clipped = ("left" if k_p <= kc_min + 0.05
                       else ("right" if k_p >= kc_max - 0.05 else None))
            items.append({
                "label": label,
                "r_anchor": float(r_p),
                "r_peak": float(r_p),
                "k_peak": float(k_p),
                "clipped": clipped,
                "masked_by_heavy": False,
                "light_heavy_mismatch": False,
            })

    drawn_r: list[float] = []
    placed_ys: list[float] = []
    for sh in items:
        slabel = str(sh["label"])
        k_p = float(sh["k_peak"])
        r_p = float(sh.get("r_peak", sh["r_anchor"]))
        r_anc = float(sh["r_anchor"])
        if sh.get("masked_by_heavy"):
            ax_e.axhline(r_anc, color="#ffe082", ls=":", lw=0.9, alpha=0.85)
            kth = sh.get("k_peak_th")
            th_s = f"{float(kth):.1f}" if (kth is not None and np.isfinite(kth)) else "~5.2"
            y_txt = _wt_text_y(r_anc, placed_ys, r_top)
            ax_e.text(
                0.02, y_txt, f"{slabel} [masked by {sh['masked_by']}; th k≈{th_s}]",
                transform=ax_e.get_yaxis_transform(), fontsize=7.6, color="#6b4500",
                bbox=dict(facecolor="#fff8e1", alpha=0.92, edgecolor="#e0a800", lw=0.6, pad=1.0),
            )
            continue

        if not sh.get("light_heavy_mismatch") and any(abs(r_p - prev) < 0.08 for prev in drawn_r):
            continue
        drawn_r.append(r_p)
        is_bad = bool(sh.get("light_heavy_mismatch") or sh.get("clipped"))
        line_col = "#ff6b6b" if sh.get("light_heavy_mismatch") else "w"
        edge_col = "#ff3b30" if sh.get("light_heavy_mismatch") else ("#ffb300" if sh.get("clipped") else "w")
        ax_e.axhline(r_p, color=line_col, ls="--", lw=0.9, alpha=0.85)
        ax_e.plot(k_p, r_p, "o", ms=5.5, mfc="none", mec=edge_col, mew=1.5 if is_bad else 1.3)
        tags = []
        if sh.get("clipped"):
            tags.append("edge!")
        if sh.get("light_heavy_mismatch"):
            tags.append("mismatch!")
        suffix = f" [{' '.join(tags)}]" if tags else ""
        box_fc = "#fdecea" if sh.get("light_heavy_mismatch") else ("#fff8e1" if sh.get("clipped") else "white")
        box_ec = "#d1495b" if sh.get("light_heavy_mismatch") else ("#e0a800" if sh.get("clipped") else "none")
        y_txt = _wt_text_y(r_p, placed_ys, r_top)
        ax_e.text(
            0.02, y_txt, f"{slabel} ({k_p:.1f}, {r_p:.2f}){suffix}",
            transform=ax_e.get_yaxis_transform(), fontsize=7.8,
            color="#8b0000" if sh.get("light_heavy_mismatch") else "black",
            bbox=dict(facecolor=box_fc, alpha=0.88, edgecolor=box_ec, lw=0.6 if is_bad else 0.0, pad=1.0),
        )


def build_structure_main_figure(
    res: PipelineResult,
    fit_out: dict | None,
    outdir: Path,
    stem: str,
    *,
    sample: str | None = None,
    edge_label: str | None = None,
    shells: list[tuple[str, float, float]] | None = None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """生成图 1：催化剂配位结构剖析总图（6 面板，含 R 空间逐路径分壳分解）。"""
    special.apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    e0 = float((res.e0 or {}).get("e0_sample", np.nan))
    norm, chi_k, ft, wt = res.norm, res.chi_k, res.ft, res.wt
    kw = int(getattr(ft, "kweight", 2))
    sample_txt = sample or stem
    abs_sym = getattr(res, "absorber_symbol", None) or "M"
    edge_txt = edge_label or f"{abs_sym}-edge"
    payload = special._fit_payload(fit_out)
    fit = special._fit_result(fit_out)
    shell_wins = shells or _extract_shells_for_plot(res, fit_out)
    path_contribs = special.compute_path_r_contributions(res, fit_out)
    wt_targets = special.select_wt_shell_targets(res, fit_out, path_contribs=path_contribs)
    wt_sanity = special.assess_wt_shell_sanity(res, fit_out, shell_targets=wt_targets)

    fig = plt.figure(figsize=(16.2, 9.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # (a) XANES 归一化与白线特征
    e_rel = np.asarray(norm.energy, dtype=float) - e0
    mu_n = np.asarray(norm.mu_norm, dtype=float)
    m_xanes = (e_rel >= -30.0) & (e_rel <= 90.0)
    if np.any(m_xanes):
        ax_a.plot(e_rel[m_xanes], mu_n[m_xanes], color="#1b6ca8", lw=1.7, label="sample")
    else:
        ax_a.plot(e_rel, mu_n, color="#1b6ca8", lw=1.7, label="sample")
    wl_e, wl_h = _white_line_xy(res)
    if np.isfinite(wl_e):
        ax_a.plot(wl_e - e0, wl_h, "o", ms=5.5, mfc="none", mec="#d1495b", mew=1.4)
        ax_a.annotate(f"WL = {wl_h:.2f}", xy=(wl_e - e0, wl_h),
                      xytext=(min(wl_e - e0 + 18.0, 55.0), wl_h * 0.98),
                      fontsize=8.5, color="#d1495b",
                      arrowprops=dict(arrowstyle="-", color="#d1495b", lw=0.7))
    ax_a.axvline(0.0, ls="--", color="0.55", lw=0.8)
    ax_a.axhline(1.0, ls=":", color="0.7", lw=0.8)
    ax_a.set_xlim(-30.0, 80.0)
    ax_a.set_xlabel(r"$E - E_0$ (eV)")
    ax_a.set_ylabel(r"Normalized $\mu(E)$")
    ax_a.set_title(f"(a) XANES ({edge_txt}, $E_0$={e0:.1f} eV)", fontsize=10.5, loc="left")
    ax_a.legend(frameon=False, fontsize=8.5, loc="lower right")

    # (b) XANES 一阶导数 dμ/dE（氧化态/边前峰指认）
    e_arr = np.asarray(norm.energy, dtype=float)
    dmu = np.gradient(mu_n, e_arr) if e_arr.size > 2 else np.zeros_like(mu_n)
    m_deriv = (e_rel >= -25.0) & (e_rel <= 45.0)
    if np.any(m_deriv):
        ax_b.plot(e_rel[m_deriv], dmu[m_deriv], color="#2a9d8f", lw=1.5, label=r"$d\mu_{norm}/dE$")
    else:
        ax_b.plot(e_rel, dmu, color="#2a9d8f", lw=1.5, label=r"$d\mu_{norm}/dE$")
    ax_b.axvline(0.0, ls="--", color="#d1495b", lw=0.9, label=f"$E_0$={e0:.1f} eV")
    xf = getattr(res, "xanes_features", None) or {}
    if xf.get("neck_shoulder_present") and xf.get("neck_shoulder_energy") is not None:
        sh_rel = float(xf["neck_shoulder_energy"]) - e0
        ax_b.axvline(sh_rel, ls=":", color="#ef6c00", lw=1.0, label="shoulder")
    ax_b.set_xlim(-25.0, 45.0)
    ax_b.set_xlabel(r"$E - E_0$ (eV)")
    ax_b.set_ylabel(r"$d\mu_{\rm norm}/dE$ (eV$^{-1}$)")
    ax_b.set_title("(b) First derivative & edge/pre-edge features", fontsize=10.5, loc="left")
    ax_b.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (c) k^w χ(k) 实验 vs 拟合模型 + 残差
    k_arr = np.asarray(chi_k.k, dtype=float)
    y_k = (k_arr ** kw) * np.asarray(chi_k.chi, dtype=float)
    ax_c.plot(k_arr, y_k, color="#1b6ca8", lw=1.3, label="data")
    if fit is not None and getattr(fit, "model_k", None) is not None and getattr(fit, "model_k_grid", None) is not None:
        mk_grid = np.asarray(fit.model_k_grid, dtype=float)
        mk_val = (mk_grid ** kw) * np.asarray(fit.model_k, dtype=float)
        ax_c.plot(mk_grid, mk_val, color="#d1495b", lw=1.3, ls="--", label="fit")
        d_interp = np.interp(mk_grid, k_arr, y_k)
        resid_k = d_interp - mk_val
        offset_k = -1.25 * float(np.nanmax(np.abs(y_k)) or 1.0)
        ax_c.plot(mk_grid, resid_k + offset_k, color="0.45", lw=0.9, label="residual (offset)")
    ax_c.axvspan(float(ft.kmin), float(ft.kmax), color="#e8f0fe", alpha=0.5, zorder=0)
    ax_c.axhline(0.0, color="0.7", lw=0.7)
    ymax_k = float(np.nanmax(np.abs(y_k))) or 1.0
    ymin_k = (-1.95 * ymax_k) if (fit is not None and getattr(fit, "model_k", None) is not None) else (-1.20 * ymax_k)
    ax_c.set_ylim(ymin_k, 1.38 * ymax_k)
    ax_c.set_xlim(max(0.0, float(ft.kmin) - 1.0), min(float(k_arr.max()), float(ft.kmax) + 1.5))
    ax_c.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax_c.set_ylabel(rf"$k^{{{kw}}}\chi(k)$ ($\AA^{{-{kw}}}$)")
    ax_c.set_title(rf"(c) EXAFS $k^{{{kw}}}\chi(k)$ oscillation & fit", fontsize=10.5, loc="left")
    ax_c.legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none",
                fontsize=8, loc="upper right")

    # (d) R 空间拟合 + 逐路径分壳分解（核心结构判据）
    r_arr = np.asarray(ft.r, dtype=float)
    mag_data = np.asarray(ft.magnitude, dtype=float)
    re_data = np.asarray(ft.real, dtype=float)
    peak_max = float(np.nanmax(mag_data)) or 1.0
    r_hi = max([3.6] + [hi for _, _, hi in shell_wins] + [c["r_fit"] + 0.5 for c in path_contribs])
    r_hi = min(r_hi, float(r_arr.max()))

    ax_d.plot(r_arr, mag_data, color="#1b6ca8", lw=1.6, label="data |χ(R)|")
    has_model_r = fit is not None and getattr(fit, "model_chi_r", None) is not None and getattr(fit, "model_r", None) is not None
    if has_model_r:
        mr = np.asarray(fit.model_r, dtype=float)
        mc = np.asarray(fit.model_chi_r, dtype=complex)
        ax_d.plot(mr, np.abs(mc), color="#d1495b", lw=1.4, ls="--", label="total fit |χ(R)|")
        # 下方绘制实部实验 vs 拟合（或残差），避免压住幅值主峰
        re_scale = 0.55
        re_offset = -0.80 * peak_max
        ax_d.plot(r_arr, re_data * re_scale + re_offset, color="#1b6ca8", lw=0.85, alpha=0.65,
                  label="Re χ(R) data/fit (offset)")
        ax_d.plot(mr, mc.real * re_scale + re_offset, color="#d1495b", lw=0.85, ls="--", alpha=0.8)
        m_interp = np.interp(r_arr, mr, np.abs(mc))
        res_mag = mag_data - m_interp
        ax_d.plot(r_arr, res_mag, color="0.4", lw=0.85, ls=":", label="|χ|_data − |χ|_fit")
        ax_d.axvspan(float(fit.rmin), float(fit.rmax), color="#e8f0fe", alpha=0.55, zorder=0)
    else:
        ax_d.plot(r_arr, re_data * 0.5 - 0.6 * peak_max, color="0.55", lw=0.85, label="Re χ(R) (offset)")

    for j, c in enumerate(path_contribs[:6]):
        col = special.SHELL_COLORS[j % len(special.SHELL_COLORS)]
        ax_d.plot(c["r"], c["magnitude"], lw=1.1, ls="-.", color=col, alpha=0.85,
                  label=f"{c['label']} (R={c['r_fit']:.2f})")

    ax_d.axhline(0.0, color="0.7", lw=0.7)
    ax_d.set_xlim(0.0, r_hi)
    ax_d.set_ylim(-1.45 * peak_max if has_model_r else -1.1 * peak_max,
                  1.52 * peak_max if path_contribs else 1.32 * peak_max)
    ax_d.set_xlabel(r"$R$ ($\AA$, not phase-corrected)")
    ax_d.set_ylabel(r"$|\chi(R)|$ & $\mathrm{Re}\,\chi(R)$")
    ax_d.set_title("(d) R-space fit & path-by-path shell decomposition", fontsize=10.5, loc="left")
    ax_d.legend(frameon=True, facecolor="white", framealpha=0.90, edgecolor="0.88",
                fontsize=6.9, loc="upper right",
                ncol=2 if len(path_contribs) >= 2 else 1)

    # (e) Morlet 二维小波变换 |W(R, k)|
    im = ax_e.contourf(wt.k_centers, wt.r, wt.magnitude.T, levels=21, cmap="jet", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax_e, pad=0.02)
    cb.set_label(r"$|W(R,k)|$ (norm.)", fontsize=9)
    ax_e.set_xlim(max(float(wt.k_centers.min()), float(ft.kmin) - 0.8),
                  min(float(wt.k_centers.max()), float(ft.kmax) + 1.0))
    r_top = min(max(4.0, r_hi + 0.4), float(wt.r.max()))
    ax_e.set_ylim(0.0, r_top)
    _annotate_wt_shell_panel(ax_e, wt, shell_wins, wt_sanity, r_top,
                             explicit_shells=(shells is not None))
    ax_e.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax_e.set_ylabel(r"$R$ ($\AA$, apparent)")
    ax_e.set_title(r"(e) Wavelet $|W(R,k)|$: light vs heavy scatterers", fontsize=10.5, loc="left")

    # (f) 结构定量与门禁汇总卡片
    ax_f.axis("off")
    ax_f.set_title("(f) Fitted structure & physical gates", fontsize=10.5, loc="left")
    sid = getattr(res, "structure", None) or {}
    lines = [
        f"Sample: {sample_txt}  ({edge_txt})",
        f"E₀ = {e0:.2f} eV   |   Δμ₀ = {res.norm.edge_step:.3f}   |   plateau = {res.norm.plateau:.3f}",
        f"ID status: {payload.get('status', sid.get('status', 'not run'))}   |   "
        f"Top-1: {((sid.get('ranked') or [{}])[0].get('id') or 'n/a')}",
        "",
    ]
    if fit is not None and getattr(fit, "paths", None):
        if str(payload.get("mode") or "") == "l1_first_shell" and isinstance(payload.get("l1"), dict):
            l1 = payload["l1"]
            lines.append(f"L1 first-shell fallback: {l1.get('ligand', 'X')} ({l1.get('geometry', '')})")
        lines.append(
            f"Rf = {fit.r_factor:.4f}   ΔE₀ = {fit.delta_e0:+.2f}±{fit.delta_e0_err:.2f} eV   "
            f"Nvar/Nidp = {fit.n_variables}/{fit.n_independent:.1f}"
        )
        lines.append(f"S₀² = {fit.s02:.4f} ({payload.get('s02_source') or 'default'})")
        lines.append("-" * 46)
        for pf in fit.paths:
            n_str = (f"{pf.degeneracy:.2f}±{pf.n_err:.2f}" if pf.n_err is not None
                     else f"{pf.degeneracy:.2f} (fixed)")
            lines.append(f"• {pf.label:<12} N={n_str}  R={pf.r:.3f} Å  σ²={pf.sigma2:.4f}")
    elif payload.get("l1"):
        l1 = payload["l1"]
        lines.append(f"L1 first-shell fallback: {l1.get('ligand', 'X')} ({l1.get('geometry', '')})")
        lines.append(f"N={l1.get('N', np.nan):.2f}  R={l1.get('R', np.nan):.3f} Å  σ²={l1.get('sigma2', np.nan):.4f}  Rf={l1.get('rf', np.nan):.4f}")
    else:
        lines.append("S7 fitting was not executed (--no-fit).")

    flags = payload.get("flags") or {}
    self_abs = payload.get("self_absorption") or getattr(res, "self_absorption", None) or {}
    fit_obj = special._fit_result(fit_out)
    alerts = []
    if payload.get("cn_reliable") is False:
        alerts.append("CN unreliable (gated)")
    elif payload.get("cn_claim_allowed") is False:
        alerts.append("CN claim blocked (cn_claim_allowed=False)")
    if flags.get("overfit") or (
        fit_obj is not None
        and float(getattr(fit_obj, "n_variables", 0)) >= float(getattr(fit_obj, "n_independent", 1.0))
    ):
        alerts.append("Overfit (Nvar >= Nidp)")
    if flags.get("r_factor_bad") or (
        fit_obj is not None and float(getattr(fit_obj, "r_factor", 0.0)) > 0.08
    ):
        alerts.append("R-factor > 0.08")
    if fit_obj is not None and abs(float(getattr(fit_obj, "delta_e0", 0.0))) > 15.0:
        alerts.append("|ΔE0| > 15 eV")
    if flags.get("plateau_bad"):
        alerts.append("Plateau outside [0.85,1.15]")
    if self_abs.get("cn_gate"):
        alerts.append(f"Fluorescence self-abs gate ({self_abs.get('risk')})")
    if self_abs.get("transmission_thickness_risk") in ("moderate", "high"):
        alerts.append(f"Trans. thickness risk: {self_abs['transmission_thickness_risk']}")
    if flags.get("shell_static_splitting"):
        alerts.append("Static subshell splitting >= 0.08 Å")
    if fit_obj is not None:
        corr = getattr(fit_obj, "correlation", None)
        if corr is not None:
            corr_arr = np.asarray(corr, dtype=float)
            if corr_arr.ndim == 2 and corr_arr.shape[0] > 1:
                off = corr_arr - np.diag(np.diag(corr_arr))
                max_rho = float(np.nanmax(np.abs(off)))
                if np.isfinite(max_rho) and max_rho >= 0.90:
                    alerts.append(f"High corr |ρ|={max_rho:.2f}>=0.90")
        for pf in (getattr(fit_obj, "paths", None) or []):
            s2 = float(getattr(pf, "sigma2", np.nan))
            if np.isfinite(s2) and (s2 <= 0.0 or s2 > 0.020):
                alerts.append(f"Unphysical σ² ({pf.label}={s2:.4f})")
                break
    sft = payload.get("shell_f_test")
    if isinstance(sft, dict) and sft.get("confidence") is not None and sft.get("significant") is False:
        alerts.append(f"Outer-shell F-test <95% ({100.0 * float(sft['confidence']):.1f}%)")
    for wt_alert in (wt_sanity.get("alerts") or []):
        alerts.append(wt_alert)

    import data_audit
    audit_res = getattr(res, "audit", None)
    if audit_res is None:
        audit_res = data_audit.audit_preprocessing(res)
    for fail_msg in (audit_res.get("plot_fails") or audit_res.get("fails") or []):
        if fail_msg not in alerts and "平台" not in fail_msg and "Plateau=" not in fail_msg:
            alerts.append(fail_msg)

    cn_status_txt = str(payload.get("cn_status") or "")
    is_l1_mode = str(payload.get("mode") or "") == "l1_first_shell"
    is_diag_only = bool(payload.get("diagnostic_only")) or is_l1_mode or cn_status_txt in (
        "diagnostic_only", "l1_diagnostic_only", "hypothesis_model_only",
        "no_cif", "consistency_failed",
    )
    diag_tag = cn_status_txt or ("l1_diagnostic_only" if is_l1_mode else "diagnostic")
    gate_pass = (
        payload.get("cn_reliable") is True
        and payload.get("cn_claim_allowed") is not False
        and not is_diag_only
        and not alerts
    )
    if gate_pass:
        gate_line = "Gate check: PASS — quantitative CN allowed"
    elif is_diag_only and not alerts:
        gate_line = f"Gate check: DIAGNOSTIC/HYPOTHESIS ONLY ({diag_tag})"
    elif alerts:
        prefix = (f"Gate check: HYPOTHESIS/GATED ({diag_tag}) — "
                  if is_diag_only and diag_tag
                  else "Gate check: FAIL / DOWNGRADE — ")
        gate_line = prefix + "; ".join(alerts)
    else:
        gate_line = f"Gate check: {cn_status_txt or 'diagnostic'}"

    lines.append("")
    lines.extend(textwrap.wrap(gate_line, width=58, subsequent_indent="  ") or [gate_line])
    audit_warns = list(audit_res.get("plot_warnings") or audit_res.get("warnings") or []) + list(wt_sanity.get("warnings") or [])
    if audit_warns:
        audit_line = "Audit: " + "; ".join(audit_warns[:2])
        lines.extend(textwrap.wrap(audit_line, width=58, subsequent_indent="  "))
    gate_fs = 7.4 if len(lines) > 18 else (7.8 if len(lines) > 15 else 8.8)
    ax_f.text(0.02, 0.96, "\n".join(lines), transform=ax_f.transAxes,
              va="top", ha="left", fontsize=gate_fs, family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#cfd8dc"))

    fig.suptitle(f"{sample_txt} ({edge_txt}) — Catalyst Structure Dissection",
                 fontsize=13, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.97, top=0.905, bottom=0.075,
                        hspace=0.40, wspace=0.30)
    saved = _save_multi(fig, outdir, f"{stem}_structure_main", formats, dpi)
    plt.close(fig)
    return saved


def build_shell_dissection_figure(
    res: PipelineResult,
    fit_out: dict | None,
    outdir: Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """生成图 2：壳层轻重鉴别（多 k 权重）+ 相移校正真实键长 + WT 元素 k-轮廓对比（3 面板）。"""
    special.apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chi_k, ft = res.chi_k, res.ft
    k = np.asarray(chi_k.k, dtype=float)
    kmin, kmax = float(ft.kmin), float(ft.kmax)
    dk = float(getattr(ft, "dk", 1.0))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    ax_kw, ax_pc, ax_wt = axes

    # (a) 多 k 权重归一化 |FT(k^1, k^2, k^3 χ)| 对比：轻元素 vs 金属/载体阳离子金标准
    kw_styles = ((1, "#2a9d8f", "-"), (2, "#1b6ca8", "--"), (3, "#d1495b", "-."))
    r_max_plot = 4.2
    for w_val, col, ls in kw_styles:
        try:
            ft_w = athena.ft_to_r_space(k, chi_k.chi, None, kmin, kmax, w_val, dk)
            mag_w = np.asarray(ft_w.magnitude, dtype=float)
            peak_w = float(np.nanmax(mag_w)) or 1.0
            ax_kw.plot(ft_w.r, mag_w / peak_w, color=col, ls=ls, lw=1.4,
                       label=rf"$k^{{{w_val}}}\chi(k)$ (norm.)")
        except Exception:  # noqa: BLE001
            continue
    ax_kw.set_xlim(0.0, r_max_plot)
    ax_kw.set_ylim(0.0, 1.15)
    ax_kw.set_xlabel(r"$R$ ($\AA$, apparent)")
    ax_kw.set_ylabel(r"Normalized $|\mathrm{FT}|$")
    ax_kw.set_title("(a) Multi-$k$-weight FT (heavy shell grows at $k^3$)", fontsize=10, loc="left")
    ax_kw.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (b) 相位修正前后 |χ(R)| 对比（Apparent R -> True R）
    try:
        pc = special.compute_phase_corrected_ft(res, fit_out)
        ax_pc.plot(pc["grid"], pc["mag_raw"], color="#9aa0a6", lw=1.2, label="apparent |χ(R)|")
        ax_pc.plot(pc["grid"], pc["mag_corr"], color="#1b6ca8", lw=1.6, label="phase-corrected |χ(R)|")
        if np.isfinite(pc["r_fit"]):
            ax_pc.axvline(pc["r_fit"], color="#d1495b", ls="--", lw=1.1,
                          label=f"fit $R_1$ = {pc['r_fit']:.3f} Å")
        ax_pc.set_title(
            f"(b) True bond length check ($R_{{app}}$={pc['r_peak_raw']:.2f}→{pc['r_peak_corrected']:.2f} Å)",
            fontsize=10, loc="left")
    except Exception:  # noqa: BLE001
        ax_pc.plot(ft.r, ft.magnitude, color="#1b6ca8", lw=1.5, label="apparent |χ(R)|")
        fit = special._fit_result(fit_out)
        if fit is not None:
            for pf in fit.paths[:3]:
                ax_pc.axvline(float(pf.r), color="#d1495b", ls="--", lw=1.0,
                              label=f"fit {pf.label} = {pf.r:.3f} Å")
        ax_pc.set_title("(b) FT peak vs fitted bond distance $R$", fontsize=10, loc="left")
    ax_pc.set_xlim(0.5, r_max_plot)
    ax_pc.set_xlabel(r"$R$ ($\AA$)")
    ax_pc.set_ylabel(r"$|\chi(R)|$")
    ax_pc.legend(frameon=False, fontsize=8.0, loc="upper right")

    # (c) 逐壳 1D WT k-轮廓与理论散射元素标尺
    wt = res.wt
    fit = special._fit_result(fit_out)
    abs_sym = getattr(res, "absorber_symbol", None)
    path_contribs = special.compute_path_r_contributions(res, fit_out) if fit is not None else []
    wt_targets = special.select_wt_shell_targets(res, fit_out, path_contribs=path_contribs)
    wt_sanity = special.assess_wt_shell_sanity(res, fit_out, shell_targets=wt_targets)
    sanity_by_key = {(s["label"], s["r_anchor"]): s for s in (wt_sanity.get("shells") or [])}

    for idx, t in enumerate(wt_targets):
        slabel, r_anc = str(t["label"]), float(t["r_anchor"])
        prof = larch_wt.extract_k_profile(wt, r_anc, half_width=0.25)
        col = special.SHELL_COLORS[idx % len(special.SHELL_COLORS)]
        rp = np.asarray(prof["ridge_profile"], dtype=float)
        rmax_p = float(np.nanmax(rp)) or 1.0
        sh_info = sanity_by_key.get((slabel, r_anc), {})
        tags = []
        if sh_info.get("clipped"):
            tags.append("edge!")
        if sh_info.get("masked_by_heavy"):
            kth = sh_info.get("k_peak_th")
            tags.append(f"masked; th={float(kth):.1f}" if kth is not None and np.isfinite(kth) else "masked")
        elif sh_info.get("light_heavy_mismatch"):
            tags.append("mismatch!")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        ax_wt.plot(prof["k"], rp / rmax_p, color=col, lw=1.6,
                   label=f"{slabel} ($R_{{app}}$={r_anc:.2f}, $k_{{pk}}$={prof['k_peak']:.1f}{tag_str})")
    if abs_sym and fit is not None:
        try:
            from xafs_report import auto_ref_k_peaks
            ruler = auto_ref_k_peaks(res, fit, abs_sym) or {}
            for el_name, k_ref in sorted(ruler.items(), key=lambda kv: kv[1]):
                if k_ref is not None and np.isfinite(k_ref):
                    ax_wt.axvline(float(k_ref), color="0.55", ls="-.", lw=0.8)
                    ax_wt.text(float(k_ref), 1.12, f"{el_name}",
                               ha="center", va="top", fontsize=8.0, color="0.25",
                               bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                         ec="0.75", alpha=0.9))
        except Exception:  # noqa: BLE001
            pass
    ax_wt.set_xlim(float(wt.k_centers.min()), float(wt.k_centers.max()))
    ax_wt.set_ylim(0.0, 1.18)
    ax_wt.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax_wt.set_ylabel(r"Normalized $|W(k; R_{\rm shell})|$")
    ax_wt.set_title(r"(c) Shell WT $k$-profiles vs reference elements", fontsize=10, loc="left")
    ax_wt.legend(frameon=False, fontsize=7.6, loc="lower right")
    wt_notes = list(wt_sanity.get("alerts") or []) + list(wt_sanity.get("warnings") or [])
    if wt_notes:
        ax_wt.set_ylim(0.0, 1.34)
        badge_raw = ("! " if wt_sanity.get("alerts") else "Note: ") + "; ".join(wt_notes[:2])
        badge_txt = "\n".join(textwrap.wrap(badge_raw, width=54)[:2])
        is_alert = bool(wt_sanity.get("alerts"))
        ax_wt.text(
            0.02, 0.98, badge_txt,
            transform=ax_wt.transAxes, va="top", ha="left", fontsize=7.0,
            color="#8b0000" if is_alert else "#7a4f01",
            bbox=dict(boxstyle="round,pad=0.24",
                      fc="#fdecea" if is_alert else "#fff8e1",
                      ec="#d1495b" if is_alert else "#e0a800", lw=0.8),
        )

    fig.tight_layout()
    saved = _save_multi(fig, outdir, f"{stem}_shell_dissection", formats, dpi)
    plt.close(fig)
    return saved


def build_candidates_compare_figure(
    res: PipelineResult,
    fit_out: dict | None,
    outdir: Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
    max_candidates: int = 4,
) -> list[Path]:
    """生成图 3：实验谱 vs Top-K 竞争候选构型理论 |χ(R)| / k^w χ(k) 对比图。"""
    sid = getattr(res, "structure", None) or {}
    ranked = list(sid.get("ranked") or [])
    if not ranked:
        return []
    from absorber import get_absorber
    import structure_lib

    abs_sym = getattr(res, "absorber_symbol", None)
    if not abs_sym:
        return []
    abs_cfg = get_absorber(abs_sym)
    lib_by_id = {m.id: m for m in structure_lib.load_library(abs_sym)}
    chi_k, ft = res.chi_k, res.ft
    k = np.asarray(chi_k.k, dtype=float)
    kw = int(getattr(ft, "kweight", 2))
    kmin, kmax = float(ft.kmin), float(ft.kmax)
    dk = float(getattr(ft, "dk", 1.0))

    special.apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    mag_exp = np.asarray(ft.magnitude, dtype=float)
    scale_exp = float(np.nanmax(mag_exp)) or 1.0
    ax_r.plot(ft.r, mag_exp / scale_exp, color="#000000", lw=1.9, label="experiment (norm.)")

    y_k_exp = (k ** kw) * np.asarray(chi_k.chi, dtype=float)
    k_scale = float(np.nanmax(np.abs(y_k_exp[(k >= kmin) & (k <= kmax)]))) or 1.0
    ax_k.plot(k, y_k_exp / k_scale, color="#000000", lw=1.7, label="experiment (norm.)")

    plotted = 0
    for idx, cand in enumerate(ranked[:max_candidates]):
        cid = cand.get("id")
        motif = lib_by_id.get(cid)
        if motif is None:
            continue
        try:
            th_k = structure_id.theory_chi(
                motif, k, abs_cfg, sigma2=float(cand.get("sigma2", 0.006)))
            de0_val = float(cand.get("delta_e0", cand.get("de0", 0.0)))
            if de0_val != 0.0:
                th_k = structure_id.shift_e0(k, th_k, de0_val)
            th_ft = athena.ft_to_r_space(k, th_k, None, kmin, kmax, kw, dk)
        except Exception:  # noqa: BLE001
            continue
        col = special.SHELL_COLORS[idx % len(special.SHELL_COLORS)]
        ls = "--" if idx == 0 else ":"
        score = float(cand.get("score", np.nan))
        th_mag = np.asarray(th_ft.magnitude, dtype=float)
        ax_r.plot(th_ft.r, th_mag / (float(np.nanmax(th_mag)) or 1.0),
                  color=col, ls=ls, lw=1.4, label=f"#{idx + 1} {cid} (score={score:.3f})")
        yk_th = (k ** kw) * th_k
        yk_s = float(np.nanmax(np.abs(yk_th[(k >= kmin) & (k <= kmax)]))) or 1.0
        ax_k.plot(k, yk_th / yk_s, color=col, ls=ls, lw=1.2, alpha=0.85, label=f"#{idx + 1} {cid}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []

    ax_r.set_xlim(0.0, 4.2)
    ax_r.set_ylim(0.0, 1.42)
    ax_r.set_xlabel(r"$R$ ($\AA$, apparent)")
    ax_r.set_ylabel(r"Normalized $|\chi(R)|$")
    ax_r.set_title("(a) Candidate fingerprint comparison in R-space", fontsize=10, loc="left")
    ax_r.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="0.88",
                fontsize=7.8, loc="upper right")
    ax_k.set_xlim(kmin, kmax)
    ax_k.set_ylim(-1.25, 1.62)
    ax_k.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax_k.set_ylabel(rf"Normalized $k^{{{kw}}}\chi(k)$")
    ax_k.set_title("(b) Candidate phase/frequency comparison in k-space", fontsize=10, loc="left")
    ax_k.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="0.88",
                fontsize=7.8, loc="upper right", ncol=2 if plotted >= 3 else 1)
    fig.suptitle(f"Structure ID candidates — status: {sid.get('status', 'unknown')}",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    saved = _save_multi(fig, outdir, f"{stem}_candidates_compare", formats, dpi)
    plt.close(fig)
    return saved


def render_plot_suite(
    res: PipelineResult,
    fit_out: dict | None,
    outdir: Path,
    stem: str,
    *,
    mode: str = "structure",
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
    sample: str | None = None,
    edge_label: str | None = None,
    shells: list[tuple[str, float, float]] | None = None,
    chi2_contour: bool = False,
    phase_correct_export: bool = False,
    indicator_rows: list | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """统一出图调度器：`mode in {'none', 'structure', 'full', 'paper'}`。"""
    mode = str(mode or "none").strip().lower()
    if mode not in ("none", "structure", "full", "paper"):
        raise ValueError(f"未知绘图模式 --plot={mode!r}（可选 none/structure/full/paper）")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "mode": mode, "images": [], "tables": [], "extra_images": [],
    }
    if mode == "none":
        return result

    main_imgs = build_structure_main_figure(
        res, fit_out, outdir, stem, sample=sample, edge_label=edge_label,
        shells=shells, formats=formats, dpi=dpi)
    shell_imgs = build_shell_dissection_figure(
        res, fit_out, outdir, stem, formats=formats, dpi=dpi)
    result["images"].extend(main_imgs + shell_imgs)

    summary_csv = special.write_summary_csv(outdir, stem, res, fit_out, indicator_rows)
    result["tables"].append(summary_csv)

    rdf_imgs = special.write_rdf_figure(fit_out, outdir, stem, formats=formats, dpi=dpi)
    if rdf_imgs:
        result["images"].extend(rdf_imgs)
        result["extra_images"].append((rdf_imgs[0], "非高斯径向分布 g(R) 与热力学平均键长"))

    lcf_res = getattr(res, "lcf", None)
    if lcf_res:
        lcf_imgs = special.write_lcf_figure(
            res, lcf_res, outdir, stem, formats=formats, dpi=dpi)
        if lcf_imgs:
            result["images"].extend(lcf_imgs)
            result["extra_images"].append((lcf_imgs[0], "XANES 线性组合拟合 (LCF) 与残差"))

    if mode == "full":
        cand_imgs = build_candidates_compare_figure(
            res, fit_out, outdir, stem, formats=formats, dpi=dpi)
        if cand_imgs:
            result["images"].extend(cand_imgs)
            result["extra_images"].append((cand_imgs[0], "结构识别 Top-K 竞争构型对比"))

    if mode == "paper" and getattr(res, "absorber_symbol", None):
        import paper_figures

        fit_obj = special._fit_result(fit_out)
        payload = special._fit_payload(fit_out)
        fit_arg = None
        if fit_obj is not None and payload.get("path_files"):
            is_diag_paper = (
                bool(payload.get("diagnostic_only"))
                or str(payload.get("mode") or "") == "l1_first_shell"
                or payload.get("cn_reliable") is False
                or payload.get("cn_claim_allowed") is False
            )
            cn_st_paper = payload.get("cn_status") or (
                "l1_diagnostic_only" if str(payload.get("mode") or "") == "l1_first_shell"
                else ("diagnostic_only" if is_diag_paper else None)
            )
            fit_arg = paper_figures.fit_result_to_json(
                fit_obj, path_files=payload["path_files"],
                sample=sample or stem,
                cn_status=cn_st_paper,
                cn_systematics=payload.get("cn_systematics"),
                diagnostic_only=is_diag_paper,
                flags=payload.get("flags"),
                structure=(payload.get("formal_cn") or {}).get("model"))
        paper_out = paper_figures.make_paper_figure(
            res, absorber=res.absorber_symbol, sample_label=sample or stem,
            kmin=float(res.ft.kmin), kmax=float(res.ft.kmax),
            kweight=int(res.ft.kweight), dk=float(getattr(res.ft, "dk", 1.0)),
            fit=fit_arg, out=outdir, stem=f"{stem}_paper", dpi=dpi, verbose=verbose)
        for val in paper_out.values():
            p = Path(val)
            if p.suffix.lower() in (".png", ".pdf"):
                result["images"].append(p)
            else:
                result["tables"].append(p)

    if phase_correct_export:
        try:
            pc = special.write_phase_corrected_diagnostics(res, fit_out, outdir, stem=stem)
            result["images"].append(Path(pc["png"]))
            result["tables"].append(Path(pc["csv"]))
            result["extra_images"].append(
                (Path(pc["png"]),
                 f"相移校正真键长：表观 {pc['r_peak_raw']:.3f} Å → 真实 {pc['r_peak_corrected']:.3f} Å（拟合 R={pc['r_fit']:.3f} Å）"))
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  ⚠ 跳过相移校正单独导出：{exc}")

    if chi2_contour:
        cc = special.write_chi2_contour_diagnostics(res, fit_out, outdir, stem=stem)
        if cc is not None:
            result["images"].append(Path(cc["png"]))
            result["tables"].append(Path(cc["csv"]))
            result["extra_images"].append(
                (Path(cc["png"]),
                 f"二维 χ² 置信椭圆（{cc['x_name']}–{cc['y_name']}，min χ²={cc['min_chi2']:.3f}）"))

    return result
