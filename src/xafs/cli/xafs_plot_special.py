# -*- coding: utf-8 -*-
"""xafs_plot_special.py — 结构解析专用计算、按需扩展图与摘要表导出。

职责划分：
  1. 路径 R 空间分壳拆解（``compute_path_r_contributions``）；
  2. 相移校正 FT/WT（``compute_phase_corrected_ft`` / ``write_phase_corrected_diagnostics``）；
  3. 逐壳层 1D WT k-轮廓与模型残差（``write_wt_shell_profiles``）；
  4. 按需场景图：非高斯 g(R) 分布、二维 χ² 置信椭圆、原位演化（Operando）；
  5. 精简催化剂物理/门禁汇总表（``<stem>_summary.csv``）。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import artemis
import athena
import larch_wt
import xafs_fit
import xafs_writer
from xafs_core import ChiK, PipelineResult, XAFSConfigError

SHELL_COLORS = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22",
)


def apply_plot_style() -> None:
    """统一发表级结构图字体、刻度与线框风格。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 1.0, "font.size": 10,
    })


def _fit_payload(fit_out) -> dict:
    """兼容 xafs_fit.run 外层输出与 run['fit'] 内层字典。"""
    if not isinstance(fit_out, dict):
        return {}
    inner = fit_out.get("fit")
    if isinstance(inner, dict) and ("mode" in inner or "fit" in inner):
        return inner
    return fit_out


def _fit_result(fit_out):
    if fit_out is None:
        return None
    if hasattr(fit_out, "paths") and hasattr(fit_out, "r_factor"):
        return fit_out
    if isinstance(fit_out, dict):
        return _fit_payload(fit_out).get("fit")
    return None


def _first_shell_datfile(fit_payload: dict) -> Path | None:
    """取第一壳 FEFF 路径：优先 S7 path_files，回退 L1 合成团簇缓存。"""
    files = fit_payload.get("path_files") or []
    if files:
        return Path(files[0])
    l1 = fit_payload.get("l1") or {}
    if not l1:
        return None
    import structure_id
    import structure_lib

    motif = structure_id.synthetic_shell_motif(
        str(fit_payload.get("absorber") or l1.get("absorber") or "Fe"),
        ligand=str(l1.get("ligand", "O")),
        geometry=str(l1.get("geometry", "oct")),
        r0=float(l1.get("r0", 2.0)),
    )
    cache = structure_lib.ensure_feff(motif)
    rows = artemis.describe_paths(sorted(cache.glob("feff*.dat")), 40)
    ss = [r for r in rows if r["legs"] == 2]
    if not ss:
        return None
    r0 = float(l1.get("r0", ss[0]["reff"]))
    best = min(ss, key=lambda r: abs(float(r["reff"]) - r0))
    return cache / best["file"]


def _peak_in_window(r: np.ndarray, mag: np.ndarray,
                    lo: float = 0.8, hi: float = 3.2) -> float:
    rr = np.asarray(r, dtype=float)
    mm = np.asarray(mag, dtype=float)
    m = (rr >= lo) & (rr <= hi)
    if not m.any():
        return float(rr[int(np.argmax(mm))])
    return float(rr[m][int(np.argmax(mm[m]))])


def ft_anchor_near_bond(r: np.ndarray, mag: np.ndarray, r_bond: float,
                        path_contrib: dict | None = None) -> float:
    """围绕拟合路径锚定表观 FT 峰位，避免相邻壳层串扰。

    优先依据独立 FEFF 路径的理论表观 FT 峰位 ``r_feff`` 在紧窄窗
    ``[r_feff - 0.25, r_feff + 0.25]`` 内拾取实验主峰；缺单路径贡献时
    回退到表观窗 ``[r_bond - 0.75, r_bond - 0.10]``。
    """
    r = np.asarray(r, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if path_contrib and "r" in path_contrib and "magnitude" in path_contrib:
        pr = np.asarray(path_contrib["r"], dtype=float)
        pmag = np.asarray(path_contrib["magnitude"], dtype=float)
        psel = pr >= 0.5
        if np.any(psel):
            r_feff = float(pr[psel][int(np.argmax(pmag[psel]))])
            sel_exp = (r >= max(0.5, r_feff - 0.25)) & (r <= r_feff + 0.25)
            if np.any(sel_exp):
                return float(r[sel_exp][int(np.argmax(mag[sel_exp]))])
            return r_feff
    for half_lo, half_hi in ((0.75, 0.10), (1.0, 0.05)):
        lo = max(0.5, float(r_bond) - half_lo)
        hi = max(lo + 0.1, float(r_bond) - half_hi)
        sel = (r >= lo) & (r <= hi)
        if sel.any():
            return float(r[sel][int(np.argmax(mag[sel]))])
    return float(max(0.8, float(r_bond) - 0.4))


def _scatterer_element_and_z(label: str, absorber: str | None = None,
                             path_obj=None) -> tuple[str | None, int | None]:
    """提取路径散射元素与原子序数（兼容同核金属-金属路径如 Pt-Pt / Fe-Fe）。"""
    import re

    el = artemis._path_ligand(
        path_obj if path_obj is not None else {"label": str(label)}, absorber)
    if el is None and "-" in str(label):
        tail = str(label).split("-", 1)[1]
        for sym in re.findall(r"[A-Z][a-z]?", tail):
            cap = sym.capitalize()
            if cap in artemis.ATOMIC_NUMBER:
                el = cap
                break
    z = artemis.ATOMIC_NUMBER.get(el) if el else None
    return el, z


def _find_masking_heavy_shell(
    light_sh: dict[str, Any],
    shells_info: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """检查同模型中是否存在与轻散射体 WT 脊重合并主导高 k 信号的相邻重原子壳层。"""
    r_anc = float(light_sh["r_anchor"])
    r_fit = float(light_sh.get("r_fit", np.nan))
    for other in shells_info:
        if other is light_sh:
            continue
        oz = other.get("z")
        if oz is None or int(oz) < 20 or float(other.get("k_peak", 0.0)) < 7.0:
            continue
        close_app = abs(float(other["r_anchor"]) - r_anc) <= 0.80
        other_rf = float(other.get("r_fit", np.nan))
        close_fit = (np.isfinite(r_fit) and np.isfinite(other_rf)
                     and abs(other_rf - r_fit) <= 0.85)
        if close_app or close_fit:
            return other
    return None


def _single_path_wt_kpeak(
    res: PipelineResult,
    k: np.ndarray,
    chi_k_j: np.ndarray,
    r_feff_app: float,
    kw: int,
) -> float | None:
    """在与实验一致的 WT 窗参数下计算单条 FEFF 路径的理论 WT k_peak。"""
    wt_res = getattr(res, "wt", None)
    kc = np.asarray(getattr(wt_res, "k_centers", []), dtype=float)
    if wt_res is None or kc.size < 3:
        return None
    sigma = float(getattr(wt_res, "sigma", 1.5))
    wt_kmin = max(float(np.min(k)), float(np.min(kc)) - sigma)
    wt_kmax = min(float(np.max(k)), float(np.max(kc)) + sigma)
    if wt_kmax - wt_kmin <= 2.0 * sigma + 0.15:
        return None
    chi_j = ChiK(
        k=k, chi=np.asarray(chi_k_j, dtype=float),
        background=np.zeros_like(k), mu_norm=np.ones_like(k), rbkg=0.0,
    )
    wt_j = larch_wt.continuous_wavelet_transform(
        chi_j, sigma=sigma, kmin=wt_kmin, kmax=wt_kmax,
        kweight=kw, rmax=float(np.nanmax(wt_res.r)),
    )
    prof_j = larch_wt.extract_k_profile(wt_j, r_feff_app, half_width=0.30)
    k_th = float(prof_j["k_peak"])
    return k_th if np.isfinite(k_th) else None


def select_wt_shell_targets(res: PipelineResult, fit_out: dict | None,
                            path_contribs: list[dict] | None = None) -> list[dict[str, Any]]:
    """选择用于 1D WT k-轮廓对比与物理一致性核验的代表壳层目标。"""
    fit = _fit_result(fit_out)
    ft = getattr(res, "ft", None)
    abs_sym = getattr(res, "absorber_symbol", None)
    if path_contribs is None and fit is not None:
        try:
            path_contribs = compute_path_r_contributions(res, fit_out)
        except Exception:  # noqa: BLE001
            path_contribs = []
    contrib_by_label: dict[str, dict] = {}
    for pc in (path_contribs or []):
        lbl = str(pc.get("label", ""))
        if lbl and lbl not in contrib_by_label:
            contrib_by_label[lbl] = pc

    targets: list[dict[str, Any]] = []
    if fit is not None and getattr(fit, "paths", None) and ft is not None:
        sorted_paths = sorted(fit.paths, key=lambda q: float(getattr(q, "reff", q.r) or q.r))
        for p in sorted_paths[:6]:
            pc = contrib_by_label.get(str(p.label))
            r_anc = ft_anchor_near_bond(ft.r, ft.magnitude, float(p.r), path_contrib=pc)
            el, z = _scatterer_element_and_z(str(p.label), abs_sym, path_obj=p)
            # 仅跳过“同元素且完全落入同一表观峰”的简并重复路径；保留与重原子近邻的轻散射体以供对称核验
            if any(abs(r_anc - t["r_anchor"]) < 0.05 and t.get("element") == el for t in targets):
                continue
            amp_share = None
            if pc and "r" in pc and "magnitude" in pc:
                pr = np.asarray(pc["r"], dtype=float)
                pm = np.asarray(pc["magnitude"], dtype=float)
                fr = np.asarray(ft.r, dtype=float)
                fm = np.asarray(ft.magnitude, dtype=float)
                psel = (pr >= r_anc - 0.25) & (pr <= r_anc + 0.25)
                fsel = (fr >= r_anc - 0.25) & (fr <= r_anc + 0.25)
                if np.any(psel) and np.any(fsel):
                    fmax = float(np.nanmax(fm[fsel]))
                    if fmax > 1e-12:
                        amp_share = float(np.nanmax(pm[psel])) / fmax
            targets.append({
                "label": str(p.label),
                "r_anchor": float(r_anc),
                "r_fit": float(p.r),
                "sigma2": float(getattr(p, "sigma2", np.nan)),
                "element": el,
                "z": z,
                "k_peak_th": pc.get("k_peak_th") if pc else None,
                "r_feff_app": pc.get("r_feff_app") if pc else None,
                "amp_share": amp_share,
            })
    elif np.isfinite(getattr(res, "r1", np.nan)):
        targets.append({
            "label": "shell 1",
            "r_anchor": float(res.r1),
            "r_fit": float("nan"),
            "sigma2": float("nan"),
            "element": None,
            "z": None,
            "k_peak_th": None,
            "r_feff_app": None,
            "amp_share": None,
        })
    return targets


def assess_wt_shell_sanity(res: PipelineResult, fit_out: dict | None,
                           shell_targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """核验 WT 壳层 k_peak 是否存在边界截断、重轻原子双向错配/掩蔽或多权重矛盾。"""
    wt = getattr(res, "wt", None)
    targets = shell_targets if shell_targets is not None else select_wt_shell_targets(res, fit_out)
    out: dict[str, Any] = {
        "status": "unavailable",
        "alerts": [],
        "warnings": [],
        "shells": [],
        "boundary_clipped": False,
        "wt_k_order_anomaly": False,
        "kw_shrink_for_heavy": False,
        "high_sigma2_shells": [],
    }
    if wt is None or getattr(wt, "k_centers", None) is None:
        return out
    kc = np.asarray(getattr(wt, "k_centers", []), dtype=float)
    if kc.size == 0 or not targets:
        return out
    kc_min = float(np.nanmin(kc))
    kc_max = float(np.nanmax(kc))

    shells_info: list[dict[str, Any]] = []
    for t in targets:
        r_anc = float(t["r_anchor"])
        try:
            prof = larch_wt.extract_k_profile(wt, r_anc, half_width=0.25)
            k_pk = float(prof["k_peak"])
            r_pk = float(prof["r_peak"])
        except Exception:  # noqa: BLE001
            continue
        clipped = ("left" if k_pk <= kc_min + 0.05
                   else ("right" if k_pk >= kc_max - 0.05 else None))
        s2 = float(t.get("sigma2", np.nan))
        entry = {
            **t,
            "k_peak": k_pk,
            "r_peak": r_pk,
            "clipped": clipped,
            "masked_by_heavy": False,
            "masked_by": None,
            "light_heavy_mismatch": False,
        }
        shells_info.append(entry)
        if clipped is not None:
            out["boundary_clipped"] = True
            out["alerts"].append(
                f"WT k_peak {clipped}-edge clipped ({entry['label']}@{k_pk:.1f} Å^-1)")
        if np.isfinite(s2) and s2 > 0.010:
            out["high_sigma2_shells"].append(entry["label"])
            out["warnings"].append(
                f"{entry['label']} σ²={s2:.4f}>0.010 Å² shifts WT k_peak below 0.003 ruler")

    if len(shells_info) >= 2:
        first = shells_info[0]
        z0, k0 = first.get("z"), float(first["k_peak"])
        for sh in shells_info[1:]:
            zj = sh.get("z")
            kj = float(sh["k_peak"])
            is_heavy_outer = (zj is not None and zj >= 20)
            is_light_inner = (z0 is None or z0 < 20)
            if is_heavy_outer and is_light_inner and kj <= k0 + 0.3:
                out["wt_k_order_anomaly"] = True
                out["alerts"].append(
                    f"WT heavy k_peak ({sh['label']} {kj:.1f}) <= shell-1 ({first['label']} {k0:.1f})")
            elif is_heavy_outer and sh["clipped"] == "left":
                out["wt_k_order_anomaly"] = True
                msg = f"WT heavy shell {sh['label']} edge-clipped at k={kj:.1f}"
                if msg not in out["alerts"]:
                    out["alerts"].append(msg)

            # 多 k 权重（k^1 -> k^3）归一化外层幅值趋势交叉校验
            if is_heavy_outer and abs(float(sh["r_anchor"]) - float(first["r_anchor"])) >= 0.5:
                try:
                    chi_k, ft = res.chi_k, res.ft
                    k_arr = np.asarray(chi_k.k, dtype=float)
                    kmin = float(getattr(ft, "kmin", 3.0))
                    kmax = float(getattr(ft, "kmax", 12.0))
                    dk = float(getattr(ft, "dk", 1.0))
                    ft1 = athena.ft_to_r_space(k_arr, chi_k.chi, None, kmin, kmax, 1, dk)
                    ft3 = athena.ft_to_r_space(k_arr, chi_k.chi, None, kmin, kmax, 3, dk)
                    r_arr = np.asarray(ft1.r, dtype=float)
                    m1 = np.asarray(ft1.magnitude, dtype=float)
                    m3 = np.asarray(ft3.magnitude, dtype=float)
                    m1n = m1 / (float(np.nanmax(m1)) or 1.0)
                    m3n = m3 / (float(np.nanmax(m3)) or 1.0)
                    sel = (r_arr >= float(sh["r_anchor"]) - 0.15) & (r_arr <= float(sh["r_anchor"]) + 0.15)
                    if np.any(sel):
                        p1 = float(np.nanmax(m1n[sel]))
                        p3 = float(np.nanmax(m3n[sel]))
                        if p1 > 0.02 and p3 < 0.85 * p1:
                            out["kw_shrink_for_heavy"] = True
                            out["alerts"].append(
                                f"Multi-kw k3 shrinks at {sh['label']} ({p1:.2f}->{p3:.2f})")
                except Exception:  # noqa: BLE001
                    pass

    # 对称物理法则：严禁将轻散射体（Z < 20，如 O/N/C）错标到高 k 重原子 WT 峰脊上
    for sh in shells_info:
        zj = sh.get("z")
        if zj is None or int(zj) >= 20:
            continue
        kj = float(sh["k_peak"])
        k_th = sh.get("k_peak_th")
        high_k_by_th = (k_th is not None and np.isfinite(k_th)
                        and kj - float(k_th) >= 2.0 and kj >= 7.0)
        high_k_by_z = (int(zj) <= 10 and kj >= 7.5) or (10 < int(zj) < 20 and kj >= 8.2)
        if not (high_k_by_th or high_k_by_z):
            continue
        adj_heavy = _find_masking_heavy_shell(sh, shells_info)
        th_txt = f"{float(k_th):.1f}" if (k_th is not None and np.isfinite(k_th)) else "~5.2"
        if adj_heavy is not None:
            sh["masked_by_heavy"] = True
            sh["masked_by"] = adj_heavy["label"]
            out["warnings"].append(
                f"{sh['label']} WT@{kj:.1f} masked by {adj_heavy['label']} (FEFF k_pk={th_txt})")
        else:
            sh["light_heavy_mismatch"] = True
            out["wt_k_order_anomaly"] = True
            out["alerts"].append(
                f"Light shell {sh['label']}@k={kj:.1f} (FEFF {th_txt}) signals unmodeled heavy scatterer")

    out["shells"] = shells_info
    out["status"] = "alert" if out["alerts"] else ("warn" if out["warnings"] else "ok")
    return out


def compute_path_r_contributions(res: PipelineResult, fit_out) -> list[dict]:
    """逐条重建 S7 拟合路径在 R 空间的独立复数与模长贡献，用于分壳拆解图。"""
    payload = _fit_payload(fit_out)
    fit = _fit_result(fit_out)
    files = list(payload.get("path_files") or [])
    if fit is None or not getattr(fit, "paths", None) or not files:
        return []
    k = np.asarray(res.chi_k.k, dtype=float)
    kmin = float(getattr(fit, "kmin", res.ft.kmin))
    kmax = float(getattr(fit, "kmax", res.ft.kmax))
    kweights = tuple(getattr(fit, "kweights", None) or (int(res.ft.kweight),))
    kw = int(getattr(getattr(res, "ft", None), "kweight", None) or kweights[0])
    dk = float(getattr(res.ft, "dk", 1.0))

    contributions: list[dict] = []
    for idx, (pf, fpath) in enumerate(zip(fit.paths, files)):
        pfile = Path(fpath)
        if not pfile.is_file():
            continue
        try:
            path_obj = artemis.load_feff_path(pfile)
            path_obj.degen = float(pf.degeneracy)
            artemis.set_path_parameters(
                path_obj,
                s02=float(fit.s02),
                delta_r=float(pf.delta_r),
                sigma2=float(pf.sigma2),
                delta_e0=float(fit.delta_e0),
                third=getattr(pf, "c3", None),
                fourth=getattr(pf, "c4", None),
            )
            chi_k_j = artemis.exafs_model_k([path_obj], k)
            ft_j = athena.ft_to_r_space(k, chi_k_j, None, kmin, kmax, kw, dk)
        except Exception:  # noqa: BLE001 - 单条路径重建失败不阻塞总图
            continue
        pr = np.asarray(ft_j.r, dtype=float)
        pmag = np.asarray(ft_j.magnitude, dtype=float)
        psel = pr >= 0.5
        r_feff_app = float(pr[psel][int(np.argmax(pmag[psel]))]) if np.any(psel) else float(pf.r)
        try:
            k_peak_th = _single_path_wt_kpeak(res, k, chi_k_j, r_feff_app, kw)
        except Exception:  # noqa: BLE001
            k_peak_th = None
        contributions.append({
            "index": idx,
            "label": str(pf.label),
            "r_fit": float(pf.r),
            "degen": float(pf.degeneracy),
            "r": pr,
            "chi_r": np.asarray(ft_j.chi_r, dtype=complex),
            "magnitude": pmag,
            "real": np.asarray(ft_j.real, dtype=float),
            "r_feff_app": r_feff_app,
            "k_peak_th": k_peak_th,
        })
    return contributions


def compute_phase_corrected_ft(res: PipelineResult, fit_out) -> dict:
    """计算第一壳总相移 φ_tot(k) 校正前后的 |χ(R)| 与二维 WT。"""
    payload = _fit_payload(fit_out)
    fit = payload.get("fit")
    datfile = _first_shell_datfile(payload)
    if datfile is None or not Path(datfile).is_file():
        raise XAFSConfigError(
            "diag", "相移校正需要第一壳 FEFF 路径（S7 path_files 或 L1 缓存）")
    chi_k = res.chi_k
    k = np.asarray(chi_k.k, dtype=float)
    ft = res.ft
    kmin, kmax = float(ft.kmin), float(ft.kmax)
    kw = int(ft.kweight)
    dk = float(getattr(ft, "dk", 1.0))
    r_fit = (float(fit.paths[0].r) if fit is not None and fit.paths
             else float(payload.get("l1", {}).get("R", np.nan)))
    delta_e0 = 0.0
    if fit is not None:
        delta_e0 = float(getattr(fit, "delta_e0", 0.0) or 0.0)
    elif payload.get("l1"):
        delta_e0 = float(payload["l1"].get("delta_e0", 0.0) or 0.0)
    e0_sample = float((res.e0 or {}).get("e0_sample", np.nan))
    e0_edge = None
    _symbol = getattr(res, "absorber_symbol", None) or payload.get("absorber")
    if _symbol:
        try:
            from absorber import get_absorber
            e0_edge = getattr(get_absorber(str(_symbol)), "e0_edge", None)
        except Exception:  # noqa: BLE001
            e0_edge = None
    delta_e = np.nan
    if e0_edge is not None and np.isfinite(e0_sample):
        delta_e = e0_sample + delta_e0 - float(e0_edge)
    phase_source: dict = {}
    if np.isfinite(delta_e):
        k_feff = np.sqrt(np.maximum(k * k - 0.2625 * delta_e, 1e-6))
        phi = artemis.path_total_phase(datfile, k_feff, source_out=phase_source)
    else:
        phi = artemis.path_total_phase(datfile, k, source_out=phase_source)
    raw = athena.ft_to_r_space(k, chi_k.chi, None, kmin, kmax, kw, dk)
    corr = athena.ft_to_r_space(
        k, chi_k.chi, None, kmin, kmax, kw, dk, phase_correct=phi)
    grid = np.asarray(raw.r, dtype=float)
    corr_r = np.interp(grid, np.asarray(corr.r, dtype=float), corr.chi_r.real)
    corr_i = np.interp(grid, np.asarray(corr.r, dtype=float), corr.chi_r.imag)
    mag_raw = np.abs(raw.chi_r)
    mag_corr = np.hypot(corr_r, corr_i)
    lo = max(0.8, r_fit - 0.7) if np.isfinite(r_fit) else 0.8
    hi = r_fit + 0.4 if np.isfinite(r_fit) else 3.2
    r_peak_raw = _peak_in_window(grid, mag_raw, lo, hi)
    r_peak_corrected = _peak_in_window(grid, mag_corr, lo, hi)
    wt_corr = larch_wt.continuous_wavelet_transform(
        chi_k, sigma=1.5, kmin=kmin, kmax=min(12.0, kmax),
        kweight=kw, rmax=3.5, phase_correct=phi)
    return {
        "grid": grid,
        "raw_real": raw.chi_r.real,
        "mag_raw": mag_raw,
        "corr_real": corr_r,
        "mag_corr": mag_corr,
        "r_peak_raw": r_peak_raw,
        "r_peak_corrected": r_peak_corrected,
        "r_fit": r_fit,
        "delta_e": delta_e,
        "phase_source": phase_source.get("source"),
        "wt_corr": wt_corr,
    }


def write_phase_corrected_diagnostics(res: PipelineResult, fit_out, outdir,
                                      stem: str | None = None) -> dict:
    """相移校正真键长导出（FT + 二维 WT），导出 CSV 与 PNG。"""
    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = stem or getattr(res, "stem", "") or "phase_corrected"
    pc = compute_phase_corrected_ft(res, fit_out)
    grid = pc["grid"]
    csv_path = outdir / f"{stem}_ft_phase_corrected.csv"
    with xafs_writer.csv_buffer(csv_path, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["R_A", "mag_raw", "real_raw",
                    "mag_phase_corrected", "real_phase_corrected"])
        for i in range(grid.size):
            w.writerow([
                f"{grid[i]:.8g}", f"{pc['mag_raw'][i]:.10g}",
                f"{pc['raw_real'][i]:.10g}",
                f"{pc['mag_corr'][i]:.10g}", f"{pc['corr_real'][i]:.10g}",
            ])

    png_path = outdir / f"{stem}_phase_corrected.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(grid, pc["mag_raw"], color="#9aa0a6", lw=1.2, label="raw |χ(R)|")
    ax.plot(grid, pc["mag_corr"], color="#1b6ca8", lw=1.5,
            label="phase-corrected |χ(R)|")
    if np.isfinite(pc["r_fit"]):
        ax.axvline(pc["r_fit"], color="#d1495b", ls="--", lw=1.0,
                   label=f"fit R₁ = {pc['r_fit']:.3f} Å")
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("|χ(R)|")
    ax.set_title("Phase-corrected FT: apparent → true bond length")
    ax.legend(fontsize=8, frameon=False)
    wt = pc["wt_corr"]
    ax2 = axes[1]
    mesh = ax2.pcolormesh(wt.r, wt.k_centers, wt.magnitude,
                          shading="auto", cmap="viridis")
    ax2.set_xlim(0.5, 3.5)
    ax2.set_xlabel("R (Å)")
    ax2.set_ylabel("k (Å⁻¹)")
    ax2.set_title("Phase-corrected wavelet |W(R,k)|")
    fig.colorbar(mesh, ax=ax2, label="normalized |W|")
    fig.tight_layout()
    xafs_writer.safe_savefig(fig, png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "csv": csv_path,
        "png": png_path,
        "r_peak_raw": pc["r_peak_raw"],
        "r_peak_corrected": pc["r_peak_corrected"],
        "r_fit": pc["r_fit"],
        "delta_e": pc["delta_e"],
        "phase_source": pc["phase_source"],
    }


def write_wt_shell_profiles(res: PipelineResult, fit_out, outdir,
                            stem: str | None = None,
                            ref_k_peaks: dict[str, float] | None = None) -> dict:
    """壳层一维小波包络 + 实验/模型二维 WT 残差导出。"""
    apply_plot_style()
    fit = _fit_result(fit_out)
    if fit is None or getattr(fit, "model_k", None) is None:
        raise XAFSConfigError(
            "WT", "write_wt_shell_profiles 需要含 model_k 的 S7 FitResult")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = stem or (res.stem or "wt_profiles")
    ss_paths = [p for p in sorted(fit.paths, key=lambda q: q.reff)
                if int(getattr(p, "nleg", 2)) == 2]
    shells, seen = [], []
    for p in ss_paths:
        r_bond = float(p.r)
        if any(abs(r_bond - s) < 0.05 for s in seen):
            continue
        seen.append(r_bond)
        shells.append((p, ft_anchor_near_bond(res.ft.r, res.ft.magnitude, r_bond)))
        if len(shells) >= 2:
            break
    if not shells:
        raise XAFSConfigError("WT", "拟合结果里没有单散射路径，无法生成壳层 WT 切片")

    residual = larch_wt.compute_model_wt_residual(
        res.chi_k, fit.model_k,
        sigma=float(res.wt.sigma),
        kmin=float(fit.kmin),
        kmax=float(min(fit.kmax, np.max(res.chi_k.k))),
        kweight=int(res.ft.kweight),
        rmax=float(res.wt.r.max()),
    )
    data_map = residual["wt_data"]
    model_map = residual["wt_model"]
    scale = residual["scale"] or 1.0

    profiles = []
    for p, r_anchor in shells:
        d = larch_wt.extract_k_profile(data_map, r_anchor, half_width=0.25)
        m = larch_wt.extract_k_profile(model_map, r_anchor, half_width=0.25)
        profiles.append({
            "label": p.label, "r_bond": float(p.r), "r_anchor": float(r_anchor),
            "data": d, "model": m,
            "k_peak_data": float(d["k_peak"]), "k_peak_model": float(m["k_peak"]),
        })

    csv_path = outdir / f"{prefix}_wt_profiles.csv"
    with xafs_writer.csv_buffer(csv_path, encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shell_index", "shell_label", "r_bond_A",
                         "r_anchor_A", "k_invA", "data_ridge", "model_ridge",
                         "data_slice", "model_slice",
                         "data_phase_ridge", "model_phase_ridge"])
        for i, item in enumerate(profiles, start=1):
            d, m = item["data"], item["model"]
            d_phase = (d.get("phase_ridge") if d.get("phase_ridge") is not None
                       else np.zeros_like(d["k"]))
            m_phase = (m.get("phase_ridge") if m.get("phase_ridge") is not None
                       else np.zeros_like(m["k"]))
            for kk, dr, mr, ds, ms, dp, mp in zip(
                    d["k"], d["ridge_profile"] / scale, m["ridge_profile"] / scale,
                    d["slice_profile"] / scale, m["slice_profile"] / scale,
                    d_phase, m_phase):
                writer.writerow([
                    i, item["label"], f"{item['r_bond']:.4f}",
                    f"{item['r_anchor']:.4f}", f"{float(kk):.4f}",
                    f"{float(dr):.8e}", f"{float(mr):.8e}",
                    f"{float(ds):.8e}", f"{float(ms):.8e}",
                    f"{float(dp):.6f}", f"{float(mp):.6f}",
                ])

    n_shells = len(profiles)
    fig, axes = plt.subplots(
        1, n_shells + 1, figsize=(4.1 * (n_shells + 1), 3.6),
        gridspec_kw={"width_ratios": [1.0] * n_shells + [1.25]})
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes[:n_shells], profiles):
        d, m = item["data"], item["model"]
        ax.plot(d["k"], d["ridge_profile"] / scale, color="#1b6ca8", lw=1.7, label="experiment")
        ax.plot(m["k"], m["ridge_profile"] / scale, color="#d1495b", lw=1.5, ls="--", label="fit model")
        ax.axvline(d["k_peak"], color="#1b6ca8", ls=":", lw=0.9)
        ax.axvline(m["k_peak"], color="#d1495b", ls=":", lw=0.9)
        ax.set_xlabel(r"$k$ ($\AA^{-1}$)")
        ax.set_ylabel(r"$|W(k; R_i)|$ (norm.)")
        ax.set_title(f"{item['label']}  R$_i$={item['r_anchor']:.2f} Å\n"
                     f"k$_{{peak}}$: data={d['k_peak']:.2f}, model={m['k_peak']:.2f} Å$^{{-1}}$",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(float(d["k"].min()), float(d["k"].max()))
        if ref_k_peaks:
            _, ymax = ax.get_ylim()
            valid_refs = [(el, float(val)) for el, val in ref_k_peaks.items()
                          if val is not None and np.isfinite(float(val))]
            for el_name, k_ref in sorted(valid_refs, key=lambda kv: kv[1]):
                if not (d["k"].min() <= k_ref <= d["k"].max()):
                    continue
                ax.axvline(float(k_ref), color="#777777", ls="-.", lw=0.8, alpha=0.85)
                ax.text(float(k_ref), ymax * 0.98, str(el_name),
                        rotation=90, ha="right", va="top", fontsize=7, color="#555555")

    ax_res = axes[-1]
    rmap = residual["residual_map"]
    vmax = float(np.max(np.abs(rmap))) or 1.0
    mesh = ax_res.pcolormesh(data_map.k_centers, data_map.r, rmap.T,
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax_res, pad=0.02, label=r"$|W_{data}|-|W_{model}|$ (norm.)")
    ax_res.set_xlabel(r"$k$ ($\AA^{-1}$)")
    ax_res.set_ylabel(r"$R$ ($\AA$)")
    ax_res.set_title(f"WT residual (2D $R_{{WT}}$={residual['r_factor']:.4f}"
                     + ("" if residual.get("complex_r_factor") is None else
                        f", complex={residual['complex_r_factor']:.4f}") + ")",
                     fontsize=9)
    fig.tight_layout()
    png = outdir / f"{prefix}_wt_profiles.png"
    xafs_writer.safe_savefig(fig, png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": png, "csv": csv_path,
        "shells": [{"label": it["label"], "r_bond": it["r_bond"],
                    "r_anchor": it["r_anchor"],
                    "k_peak_data": it["k_peak_data"],
                    "k_peak_model": it["k_peak_model"]} for it in profiles],
        "r_factor": float(residual["r_factor"]),
        "complex_r_factor": residual.get("complex_r_factor"),
        "residual": residual,
    }


def write_chi2_contour_diagnostics(res: PipelineResult, fit_out, outdir,
                                   stem: str | None = None,
                                   n_grid: int = 15) -> dict | None:
    """二维 χ² 置信椭圆（第一壳 N–σ² 参数相关性检查，按需导出）。"""
    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = stem or getattr(res, "stem", "") or "chi2"
    payload = _fit_payload(fit_out)
    fit = payload.get("fit")
    files = payload.get("path_files") or []
    if fit is None or not fit.paths or len(files) != len(fit.paths):
        return None
    paths = [artemis.load_feff_path(f) for f in files]
    for p, pf in zip(paths, fit.paths):
        p.degen = float(pf.degeneracy)
        artemis.set_path_parameters(
            p, s02=float(fit.s02), delta_r=float(pf.delta_r),
            sigma2=float(pf.sigma2), delta_e0=float(fit.delta_e0),
            third=getattr(pf, "c3", None), fourth=getattr(pf, "c4", None))
    p0, pf0 = paths[0], fit.paths[0]
    x_name = "N" if pf0.n_err is not None else "deltar"
    y_name = "sigma2"
    x0 = (float(pf0.degeneracy) if x_name == "N" else float(pf0.delta_r))
    x_err = pf0.n_err if x_name == "N" else pf0.delta_r_err
    y0, y_err = float(pf0.sigma2), pf0.sigma2_err
    x_half = max(3.0 * float(x_err or 0.0), 0.20 * max(abs(x0), 1e-3))
    y_half = max(3.0 * float(y_err or 0.0), 0.20 * max(abs(y0), 1e-4))
    xv = np.linspace(max(0.0, x0 - x_half), x0 + x_half, int(n_grid))
    yv = np.linspace(max(1e-4, y0 - y_half), y0 + y_half, int(n_grid))

    k = np.asarray(res.chi_k.k, dtype=float)
    r_data = np.asarray(res.ft.r, dtype=float)
    data = np.asarray(res.ft.chi_r, dtype=complex)
    mask = (r_data >= float(fit.rmin)) & (r_data <= float(fit.rmax))
    n_pts = int(mask.sum())
    if n_pts < 8:
        return None
    eps = fit.epsilon_r
    denom = float(np.sum(np.abs(data[mask]) ** 2)) or 1.0

    def chi2_at(**params):
        n_val = float(params.get("N", pf0.degeneracy if x_name != "N" else x0))
        dr_val = float(params.get("deltar", pf0.delta_r if x_name != "deltar" else x0))
        s2_val = float(params.get("sigma2", y0))
        p0.degen = n_val
        artemis.set_path_parameters(
            p0, s02=float(fit.s02), delta_r=dr_val,
            sigma2=s2_val, delta_e0=float(fit.delta_e0))
        model_k = artemis.exafs_model_k(paths, k)
        model_ft = athena.ft_to_r_space(
            k, model_k, None, float(fit.kmin), float(fit.kmax),
            int(res.ft.kweight), float(getattr(res.ft, "dk", 1.0)))
        m_re = np.interp(r_data, model_ft.r, model_ft.chi_r.real)
        m_im = np.interp(r_data, model_ft.r, model_ft.chi_r.imag)
        diff2 = float(np.sum((data[mask].real - m_re[mask]) ** 2
                             + (data[mask].imag - m_im[mask]) ** 2))
        if eps is not None and np.isfinite(eps) and eps > 0:
            return float(fit.n_independent * diff2 / (2.0 * n_pts * eps ** 2))
        return float(fit.n_independent * diff2 / (2.0 * n_pts * denom))

    contour = artemis.profile_chi2_contour(chi2_at, x_name, y_name, xv, yv)
    if contour.get("min_chi2") is None:
        return None
    csv_path = outdir / f"{stem}_chi2_contour.csv"
    with xafs_writer.csv_buffer(csv_path, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([x_name, y_name, "chi2"])
        grid = contour["chi2_grid"]
        for iy, y in enumerate(contour["y"]):
            for ix, x in enumerate(contour["x"]):
                w.writerow([f"{x:.8g}", f"{y:.8g}", f"{grid[iy, ix]:.10g}"])

    png_path = outdir / f"{stem}_chi2_contour.png"
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cs = ax.contourf(contour["x"], contour["y"], contour["chi2_grid"],
                     levels=24, cmap="viridis")
    if contour["levels"]:
        try:
            ax.contour(contour["x"], contour["y"], contour["chi2_grid"],
                       levels=contour["levels"], colors="w",
                       linewidths=0.8, linestyles="--")
        except ValueError:
            pass
    ax.plot([x0], [y0], marker="*", color="#d1495b", ms=12, label="optimum")
    ax.set_xlabel("N" if x_name == "N" else r"$\Delta R$ (Å)")
    ax.set_ylabel(r"$\sigma^2$ (Å$^2$)")
    ax.set_title(r"$\chi^2$ confidence contour ($\Delta\chi^2$=2.30/6.18/11.83)")
    ax.legend(fontsize=8, frameon=False)
    fig.colorbar(cs, ax=ax, label=r"$\chi^2$")
    fig.tight_layout()
    xafs_writer.safe_savefig(fig, png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "csv": csv_path, "png": png_path,
        "x_name": x_name, "y_name": y_name,
        "min_chi2": float(contour["min_chi2"]),
        "chi2_grid": contour["chi2_grid"],
        "x": contour["x"], "y": contour["y"],
        "levels": contour["levels"],
    }


def write_rdf_figure(fit_out, outdir, stem: str, *,
                     formats: tuple[str, ...] = ("png",),
                     dpi: int = 300) -> list[Path]:
    """当拟合放开了 C3/C4 时，绘制非高斯配对径向分布函数 g(R)。"""
    fit = _fit_result(fit_out)
    if fit is None or not getattr(fit, "paths", None):
        return []
    entries = []
    for p in fit.paths:
        if getattr(p, "c3", None) is None and getattr(p, "c4", None) is None:
            continue
        r_grid = np.linspace(max(0.5, float(p.r) - 0.8), float(p.r) + 1.0, 401)
        try:
            rdf = artemis.reconstruct_pair_distribution(
                r_grid, n=float(p.degeneracy), r0=float(p.r),
                sigma2=float(p.sigma2),
                c3=float(getattr(p, "c3", None) or 0.0),
                c4=float(getattr(p, "c4", None) or 0.0))
        except Exception:  # noqa: BLE001
            continue
        entries.append((str(p.label), rdf))
    if not entries:
        return []

    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ncols = len(entries)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 3.9), squeeze=False)
    for idx, (label, rdf) in enumerate(entries):
        ax = axes[0, idx]
        r, g_r = np.asarray(rdf["r"]), np.asarray(rdf["g_r"])
        ax.plot(r, g_r, color="#1b6ca8", lw=1.8, label=label)
        ax.fill_between(r, 0.0, g_r, color="#1b6ca8", alpha=0.14)
        ax.axvline(rdf["r_mode"], color="#2a9d8f", ls=":", lw=1.1,
                   label=f"$R_{{mode}}$ = {rdf['r_mode']:.3f} Å")
        ax.axvline(rdf["r_mean"], color="#555555", ls="--", lw=1.0,
                   label=f"$R_{{mean}}$ = {rdf['r_mean']:.3f} Å")
        r_th = rdf.get("r_thermodynamic")
        if r_th is not None and np.isfinite(r_th):
            ax.axvline(r_th, color="#d1495b", ls="-.", lw=1.2,
                       label=f"$\\langle R_{{th}}\\rangle$ = {r_th:.3f} Å")
        warn_parts = []
        if rdf.get("gram_charlier_clipped"):
            warn_parts.append("clipped")
        if rdf.get("unimodal") is False:
            warn_parts.append("non-unimodal side-lobe!")
        warn_txt = ("  [" + ", ".join(warn_parts) + "]") if warn_parts else ""
        ax.set_title(f"{label} (area={rdf['area']:.2f}){warn_txt}", fontsize=9.5)
        ax.set_xlabel(r"$R$ ($\AA$)")
        ax.set_ylabel(r"$g(R)$ ($\AA^{-1}$)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    written = []
    for ext in formats:
        ext_clean = str(ext).strip().lstrip(".") or "png"
        out_path = outdir / f"{stem}_rdf_distribution.{ext_clean}"
        xafs_writer.safe_savefig(fig, out_path, dpi=int(dpi), bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def write_lcf_figure(res, lcf_res: dict | None, outdir, stem: str, *,
                     formats: tuple[str, ...] = ("png",),
                     dpi: int = 300) -> list[Path]:
    """绘制 XANES LCF 拟合对比与残差图（<stem>_xanes_lcf.<ext>，严格 ASCII 标签）。"""
    if not isinstance(lcf_res, dict) or not lcf_res:
        return []
    x = np.asarray(lcf_res.get("x") if lcf_res.get("x") is not None else [], dtype=float)
    y = np.asarray(lcf_res.get("y") if lcf_res.get("y") is not None else [], dtype=float)
    y_fit = np.asarray(lcf_res.get("y_fit") if lcf_res.get("y_fit") is not None else [], dtype=float)
    if x.size < 3 or y.size != x.size or y_fit.size != x.size:
        return []

    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_res) = plt.subplots(
        2, 1, figsize=(6.4, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08},
    )
    ax_top.plot(x, y, color="#1b6ca8", lw=1.7, label="data")
    ax_top.plot(x, y_fit, color="#d1495b", lw=1.5, ls="--", label="LCF fit")
    ax_top.set_ylabel(r"Normalized $\mu(E)$")
    sample_lbl = str(getattr(res, "stem", None) or stem or "sample")
    ax_top.set_title(f"XANES Linear Combination Fit: {sample_lbl}", fontsize=10.5, loc="left")
    ax_top.legend(frameon=False, fontsize=8.5, loc="upper left")

    weights = lcf_res.get("weights") or {}
    w_err = lcf_res.get("weight_err") or {}
    info_lines = []
    for cname, wval in weights.items():
        err = w_err.get(cname) if isinstance(w_err, dict) else None
        if err is not None and np.isfinite(err):
            info_lines.append(f"{cname}: {float(wval):.3f}±{float(err):.3f}")
        else:
            info_lines.append(f"{cname}: {float(wval):.3f}")
    rf_val = float(lcf_res.get("r_factor", float("nan")))
    scale_lbl = str(lcf_res.get("energy_scale") or "relative_e0")
    if np.isfinite(rf_val):
        info_lines.append(f"R-factor = {rf_val:.4f}")
    info_lines.append(f"scale = {scale_lbl}")
    ax_top.text(
        0.97, 0.06, "\n".join(info_lines),
        transform=ax_top.transAxes, ha="right", va="bottom",
        fontsize=8.0, family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc",
              "edgecolor": "#b0bec5", "alpha": 0.92},
    )

    align_chk = lcf_res.get("alignment_check")
    if isinstance(align_chk, dict) and align_chk.get("recommended_shift_eV") is not None:
        shift_ev = float(align_chk["recommended_shift_eV"])
        base_rf = float(align_chk.get("base_r_factor", rf_val))
        sh_rf = float(align_chk.get("shifted_r_factor", rf_val))
        warn_txt = f"ALIGN WARN: shift stds {shift_ev:+.1f} eV reduces Rf {base_rf:.4f} -> {sh_rf:.4f}"
        ax_top.text(
            0.03, 0.78, warn_txt,
            transform=ax_top.transAxes, ha="left", va="top",
            fontsize=7.8, color="#8c2f00",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff3e0",
                  "edgecolor": "#ef6c00", "alpha": 0.95},
        )

    resid = y - y_fit
    ax_res.plot(x, resid, color="#37474f", lw=1.1, label="residual (data - fit)")
    ax_res.axhline(0.0, color="0.65", lw=0.8, ls="--")
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Photon Energy (eV)" if scale_lbl == "absolute" else r"$E - E_0$ (eV)")
    ax_res.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.subplots_adjust(left=0.13, right=0.96, top=0.91, bottom=0.11, hspace=0.08)
    written: list[Path] = []
    for ext in formats:
        ext_clean = str(ext).strip().lstrip(".") or "png"
        out_path = outdir / f"{stem}_xanes_lcf.{ext_clean}"
        xafs_writer.safe_savefig(fig, out_path, dpi=int(dpi), bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def write_operando_diagnostics(operando_out: dict, outdir,
                               stem: str = "operando") -> dict:
    """原位序列（Operando）组分分数/LCF/价态描述符演化导出。"""
    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(operando_out.get("names") or [])
    n = len(names)
    mcr = operando_out.get("mcr_als") or {}
    C = np.asarray(mcr.get("concentrations", np.zeros((0, n))), dtype=float)
    if C.ndim != 2 or C.shape[1] != n:
        C = np.zeros((int(operando_out.get("n_components") or 0), n))
    k = int(C.shape[0])
    S = np.asarray(mcr.get("components", np.zeros((0, k))), dtype=float)
    xg = np.asarray(mcr.get("x", []), dtype=float)
    std_names = list(operando_out.get("accepted_standards") or [])
    lcf_series = list(operando_out.get("lcf_series") or [])
    descriptors = list(operando_out.get("descriptors") or [])

    conc_csv = outdir / f"{stem}_operando_concentrations.csv"
    header = (["index", "name"]
              + [f"mcr_component_{i + 1}" for i in range(k)]
              + [f"lcf_{s}" for s in std_names]
              + ["e0_eV", "e_half_height_eV", "white_line_height", "ft_r1_A"])
    with xafs_writer.csv_buffer(conc_csv, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            row = [i, names[i]]
            row += [f"{C[j, i]:.10g}" if j < C.shape[0] else "" for j in range(k)]
            weights = (lcf_series[i].get("weights") if i < len(lcf_series) else {}) or {}
            row += [f"{float(weights.get(s, np.nan)):.10g}"
                    if weights.get(s) is not None else "" for s in std_names]
            d = descriptors[i] if i < len(descriptors) else {}
            for key in ("e0", "e_half_height", "white_line_height", "ft_r1"):
                val = d.get(key)
                row.append("" if val is None else f"{float(val):.10g}")
            w.writerow(row)

    comp_csv = outdir / f"{stem}_operando_components.csv"
    with xafs_writer.csv_buffer(comp_csv, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["energy_eV"] + [f"mcr_component_{i + 1}" for i in range(k)])
        for i in range(xg.size):
            w.writerow([f"{xg[i]:.10g}"]
                       + [f"{S[i, j]:.10g}" if i < S.shape[0] else "" for j in range(k)])

    png_path = outdir / f"{stem}_operando_evolution.png"
    idx = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ax = axes[0, 0]
    for j in range(k):
        ax.plot(idx, C[j, :], marker="o", lw=1.4, label=f"MCR component {j + 1}")
    ax.set_xlabel("spectrum index")
    ax.set_ylabel("MCR concentration")
    ax.set_title("MCR-ALS species fractions")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    if std_names:
        for s in std_names:
            vals = [float(((lcf_series[i].get("weights") or {}).get(s, np.nan))
                          if i < len(lcf_series) else np.nan) for i in range(n)]
            ax.plot(idx, vals, marker="s", lw=1.2, label=s)
        ax.set_ylabel("LCF weight")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "no accepted standards",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("LCF weight")
    ax.set_xlabel("spectrum index")
    ax.set_title("LCF standard weights (SPOIL-filtered)")

    ax = axes[1, 0]
    e0 = np.asarray([np.nan if (i >= len(descriptors) or descriptors[i].get("e0") is None)
                     else float(descriptors[i]["e0"]) for i in range(n)], dtype=float)
    ehh = np.asarray([np.nan if (i >= len(descriptors) or descriptors[i].get("e_half_height") is None)
                      else float(descriptors[i]["e_half_height"]) for i in range(n)], dtype=float)
    wl = np.asarray([np.nan if (i >= len(descriptors) or descriptors[i].get("white_line_height") is None)
                     else float(descriptors[i]["white_line_height"]) for i in range(n)], dtype=float)
    ax.plot(idx, e0 - np.nanmean(e0) if np.isfinite(e0).any() else e0,
            marker="o", color="#1b6ca8", label="E0 − mean (eV)")
    ax.plot(idx, ehh - np.nanmean(ehh) if np.isfinite(ehh).any() else ehh,
            marker="^", color="#2a9d8f", label="E1/2 − mean (eV)")
    ax.set_xlabel("spectrum index")
    ax.set_ylabel("energy shift (eV)")
    ax2 = ax.twinx()
    ax2.plot(idx, wl, marker="d", color="#d1495b", lw=1.1, label="white-line height")
    ax2.set_ylabel("white-line height", color="#d1495b")
    ax.set_title("Valence / structure descriptors")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="best")

    ax = axes[1, 1]
    for j in range(k):
        ax.plot(xg, S[:, j], lw=1.4, label=f"component {j + 1}")
    if str(operando_out.get("mode") or "") == "exafs":
        ax.set_xlabel(r"$k$ ($\AA^{-1}$)")
    elif (str(operando_out.get("energy_scale") or "") == "absolute"
          or (xg.size > 0 and float(np.nanmin(xg)) > 100.0)):
        ax.set_xlabel("photon energy (eV)")
    else:
        ax.set_xlabel("energy relative to E0 (eV)")
    ax.set_ylabel("normalized pure component")
    ax.set_title("MCR-ALS pure component spectra (energy-sorted)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    xafs_writer.safe_savefig(fig, png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"concentrations_csv": conc_csv, "components_csv": comp_csv, "png": png_path}


def write_operando_exafs_diagnostics(exafs_series_out: dict, outdir,
                                     stem: str = "operando_exafs") -> dict:
    """原位 EXAFS 序列 N(t)/R(t)/σ²(t) 演化导出。"""
    apply_plot_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(exafs_series_out.get("names") or [])
    shells = list(exafs_series_out.get("shells") or [])
    traj = exafs_series_out.get("trajectories") or {}
    r_factors = list(exafs_series_out.get("r_factors") or [])
    n = len(names)
    delta_e0 = exafs_series_out.get("shared_delta_e0")

    csv_path = outdir / f"{stem}_exafs_evolution.csv"
    header = ["index", "name", "r_factor", "delta_e0"]
    for label in shells:
        header += [f"{label}_N", f"{label}_N_err",
                   f"{label}_R", f"{label}_R_err", f"{label}_sigma2"]
    with xafs_writer.csv_buffer(csv_path, encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            rf = r_factors[i] if i < len(r_factors) else None
            row = [i, names[i],
                   "" if rf is None else f"{float(rf):.10g}",
                   "" if delta_e0 is None else f"{float(delta_e0):.10g}"]
            for label in shells:
                t = traj.get(label) or {}

                def _val(key):
                    vals = t.get(key) or []
                    if i >= len(vals) or vals[i] is None:
                        return ""
                    return f"{float(vals[i]):.10g}"

                row += [_val("N"), _val("N_err"), _val("R"),
                        _val("R_err"), _val("sigma2")]
            w.writerow(row)

    png_path = outdir / f"{stem}_exafs_evolution.png"
    idx = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
    colors = ("#1b6ca8", "#d1495b", "#2a9d8f", "#e9c46a", "#6a4c93")
    for j, label in enumerate(shells):
        t = traj.get(label) or {}
        c = colors[j % len(colors)]
        nv = np.asarray([np.nan if v is None else float(v) for v in (t.get("N") or [])], dtype=float)
        ne = np.asarray([0.0 if v is None else float(v) for v in (t.get("N_err") or [])], dtype=float)
        rv = np.asarray([np.nan if v is None else float(v) for v in (t.get("R") or [])], dtype=float)
        re = np.asarray([0.0 if v is None else float(v) for v in (t.get("R_err") or [])], dtype=float)
        axes[0].errorbar(idx, nv, yerr=ne, marker="o", lw=1.3, color=c, capsize=2.5, label=label)
        axes[1].errorbar(idx, rv, yerr=re, marker="s", lw=1.3, color=c, capsize=2.5, label=label)
    axes[0].set_ylabel("N(t)")
    axes[0].set_title("Operando EXAFS: coordination number evolution")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("spectrum index")
    axes[1].set_ylabel("R(t) (Å)")
    axes[1].set_title("Operando EXAFS: bond length evolution")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    xafs_writer.safe_savefig(fig, png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"csv": csv_path, "png": png_path}


def write_summary_csv(outdir: Path, stem: str, res: PipelineResult,
                      fit_out: dict | None,
                      indicator_rows: list | None = None) -> Path:
    """导出紧凑的结构剖析与门禁摘要表 ``<stem>_summary.csv``（表 2）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fit = _fit_result(fit_out)
    payload = _fit_payload(fit_out)
    sid = getattr(res, "structure", None) or {}
    flags = payload.get("flags") or {}
    self_abs = payload.get("self_absorption") or getattr(res, "self_absorption", None) or {}
    rows: list[list[Any]] = [
        ["absorber", getattr(res, "absorber_symbol", "") or "", "", "吸收元素"],
        ["e0_eV", f"{float((res.e0 or {}).get('e0_sample', np.nan)):.2f}", "eV", "实验吸收边能量"],
        ["edge_step", f"{float(res.norm.edge_step):.4f}", "", "边跳高度 Δμ₀"],
        ["plateau", f"{float(res.norm.plateau):.4f}", "", "边后归一化平台（期望 0.85–1.15）"],
        ["structure_status", payload.get("status", sid.get("status", "skipped")), "", "S6.5 识别置信分级"],
        ["structure_top1", ((sid.get("ranked") or [{}])[0].get("id") or ""), "", "结构识别 Top-1 候选"],
        ["fit_mode", payload.get("mode", "skipped"), "", "S7 拟合模式"],
        ["cn_status", payload.get("cn_status", ""), "", "配位数定性/定量地位"],
        ["cn_reliable", payload.get("cn_reliable", ""), "", "CN 是否可用于定量结论"],
        ["s02", payload.get("s02", ""), "", str(payload.get("s02_source", "") or "振幅折减因子")],
        ["self_absorption_risk", self_abs.get("risk", ""), "",
         str(self_abs.get("transmission_thickness_risk") or self_abs.get("note") or "")],
    ]
    if fit is not None:
        rows.extend([
            ["r_factor", f"{float(fit.r_factor):.5f}", "", "全谱拟合 R-factor（门限 ≤0.08）"],
            ["delta_e0_eV", f"{float(fit.delta_e0):+.2f}±{float(fit.delta_e0_err):.2f}", "eV", "全局零点漂移"],
            ["n_variables_vs_nidp", f"{fit.n_variables} / {fit.n_independent:.2f}", "", "自由变量 / Stern 独立点数"],
            ["nidp_scale_factor", getattr(fit, "nidp_scale_factor", ""), "", "过采样误差重标定因子"],
        ])
    elif payload.get("l1"):
        l1 = payload["l1"]
        rows.extend([
            ["r_factor", f"{float(l1.get('rf', np.nan)):.5f}", "", "L1 第一壳拟合 R-factor"],
            ["delta_e0_eV", f"{float(l1.get('delta_e0', np.nan)):+.2f}", "eV", "L1 零点漂移"],
        ])
    if flags.get("shell_static_splitting"):
        rows.append([
            "shell_static_splitting",
            "; ".join(f"{s['label']}:span={s['split_span_A']}" for s in flags["shell_static_splitting"]),
            "Å",
            "被合并子壳层显著静态劈裂告警",
        ])
    wt_sanity = assess_wt_shell_sanity(res, fit_out)
    if wt_sanity.get("shells"):
        sh_txt = ", ".join(
            f"{s['label']}@R={s['r_anchor']:.2f}:k={s['k_peak']:.1f}"
            + ("[edge]" if s["clipped"] else "")
            + (f"[masked:{s['masked_by']}]" if s.get("masked_by_heavy") else "")
            + ("[light-heavy-mismatch]" if s.get("light_heavy_mismatch") else "")
            for s in wt_sanity["shells"]
        )
        note = "; ".join(wt_sanity["alerts"] + wt_sanity["warnings"]) or "WT 壳层 k-空间秩序与多权重趋势一致"
        rows.append([
            "wt_k_sanity",
            f"{wt_sanity['status']} ({sh_txt})",
            "Å⁻¹",
            note,
        ])
    import data_audit
    audit_res = getattr(res, "audit", None)
    if audit_res is None:
        audit_res = data_audit.audit_preprocessing(res)
    if audit_res.get("checks") is not None:
        audit_msgs = list(audit_res.get("fails") or []) + list(audit_res.get("warnings") or [])
        rows.append([
            "data_quality_audit",
            audit_res.get("status", "ok"),
            "",
            "; ".join(audit_msgs) if audit_msgs else "S1–S5 预处理/标尺/高 k 噪声/Rbkg 自审计通过",
        ])
    sft = payload.get("shell_f_test")
    existing_keys = {r[0] for r in rows}
    for item in (indicator_rows or []):
        if len(item) >= 4 and str(item[0]) not in existing_keys:
            rows.append(list(item[:4]))
            existing_keys.add(str(item[0]))
    if isinstance(sft, dict) and sft.get("confidence") is not None and "outer_shells_f_test_confidence" not in existing_keys:
        rows.append([
            "outer_shells_f_test_confidence",
            f"{float(sft['confidence']):.4f}",
            "",
            f"Hamilton F-test significant={bool(sft.get('significant'))} (需≥0.95)",
        ])
    lcf_res = getattr(res, "lcf", None)
    if isinstance(lcf_res, dict) and lcf_res.get("weights") and "xanes_lcf" not in existing_keys:
        w_txt = ", ".join(f"{k}={float(v):.3f}" for k, v in lcf_res["weights"].items())
        rf_txt = (f" (Rf={float(lcf_res['r_factor']):.4g})"
                  if lcf_res.get("r_factor") is not None else "")
        note_parts = [f"energy_scale={lcf_res.get('energy_scale') or 'relative_e0'}"]
        if lcf_res.get("alignment_warning"):
            note_parts.append(str(lcf_res["alignment_warning"]))
        rows.append([
            "xanes_lcf",
            f"{w_txt}{rf_txt}",
            "",
            "; ".join(note_parts),
        ])
    path = outdir / f"{stem}_summary.csv"
    return xafs_writer.safe_write_csv(path, ["item", "value", "unit", "note"], rows)


def write_fit_table_markdown(rows: list, outdir: Path, stem: str,
                             *, weights_mode: bool = False) -> Path:
    """导出拟合参数表 Markdown（``<stem>_fit_table.md``，与论文 S-table 对齐）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    headers = list(xafs_writer.FIT_HEADER)
    if weights_mode:
        headers[2] = "N_eff"
    has_c3 = any(len(r) > 7 and str(r[7]).strip() for r in rows)
    if has_c3:
        headers.append("C3 (A^3)")
    n_cols = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    for r in rows:
        cells = [str(c) for c in list(r)[:n_cols]]
        while len(cells) < n_cols:
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    path = outdir / f"{stem}_fit_table.md"
    return xafs_writer.safe_write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
