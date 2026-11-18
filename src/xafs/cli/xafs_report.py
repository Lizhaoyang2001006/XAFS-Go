# -*- coding: utf-8 -*-
"""
xafs_report.py — 催化剂结构解析成果展示与导出主入口
================================================================
在 athena / artemis / larch_wt / xafs_plots 之上，一次运行产出（xafs_out/<report>/）：

  图（默认 ``--plot structure``，PNG 300 dpi）
    <stem>_structure_main.png    催化剂配位结构剖析 6 面板主图（XANES/dμ/dE/k拟合/R逐路径拆解/WT/门禁卡片）
    <stem>_shell_dissection.png  壳层轻重鉴别与真实键长核验（多 k 权重 |FT|/相位修正 R/逐壳 WT 轮廓）
    <stem>_fit_table.png         CN 拟合参数表（Table S 风格）
    <stem>_candidates_compare.png（仅 ``--plot full`` 且存在识别排名时输出）

  表（可粘贴）
    <stem>_fit_table.csv   拟合表（7 列冻结基线格式，utf-8-sig）
    <stem>_fit_table.md    拟合表 Markdown 版（便于报告/论文直接嵌入）
    <stem>_summary.csv     催化剂结构归属、物理指标与门禁状态精简汇总表
    <stem>_indicators.csv  关键物理指标汇总（指标/值/单位/说明）
    <stem>_report.html     单文件 HTML 报告（图片内联、门禁结论置顶）
    <stem>_meta.json       参数/门禁/决策/结构识别台账
    <stem>_structure_id.json  结构识别候选（全量 ranked + top3）

  逐面板谱数据（与图一一对应）
    <stem>_mu.csv          E, μ(scan1..3), μ(merged)
    <stem>_xanes.csv       E, μ, 边前基线, μ_norm, μ_flat
    <stem>_chik.csv        k, χ(k), k²χ(k)
    <stem>_chir.csv        R, Re χ(R), Im χ(R), |χ(R)|
    <stem>_chiq.csv        第一壳反 FT χ(q)：q, Re, Im, |χ(q)|
    <stem>_wavelet.csv     W(R,k) 矩阵（行=k，列=R）

依赖方向：core ← io ← athena ← {artemis, larch_wt} ← {xafs_plot_special, xafs_plots} ← 本模块。

用法:
    python xafs_report.py <文件...> --absorber Fe [--edge K]
                          [--fit-motif ID] [--no-fit] [--plot {none,structure,full,paper}]
    python xafs_report.py            # 无文件参数 → 内置演示
    python xafs_report.py --demo     # 显式运行内置演示

通用模式不再硬编码 Fe 区、壳层标签或第二个标样：坐标轴按 E0/k 窗口
自动缩放，拟合表来自 S7 的 motif 或 L1 结果。内置演示仅用于检查版式。
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import html
import json
import sys
import threading
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import xafs  # noqa: E402,F401
from xafs.paths import REPO_ROOT as ROOT  # noqa: E402

import artemis          # noqa: E402
import athena           # noqa: E402
import cn_table          # noqa: E402
import larch_wt         # noqa: E402
import xafs_config      # noqa: E402
import xafs_fit         # noqa: E402
import xafs_io          # noqa: E402
import xafs_lcf         # noqa: E402
import xafs_pipeline    # noqa: E402
import xafs_plot_special  # noqa: E402
import xafs_plots       # noqa: E402
import xafs_provenance  # noqa: E402
import xafs_writer      # noqa: E402
import structure_id     # noqa: E402
from absorber import get_absorber, resolve_s02_spec    # noqa: E402
from xafs_core import XAFSConfigError, XAFSDataError  # noqa: E402

# 同 run_xafs：GBK 控制台无法编码 Å/σ²/S₀²/α，--help 会直接崩。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data" / "练手数据"
# 演示用振幅因子：从 absorber.py 标定表查（demo 是 Fe 数据）；
# 通用模式的 s02 已由 xafs_fit / main 从对应元素的表值或 S7 读出。
S02 = get_absorber("Fe").s02
KMIN, KMAX = 3.0, 12.0      # FT 与拟合共用窗口
SCAN_COLORS = ("#a50026", "#f46d43", "#fdae61")
# 壳层标注窗口（R, Å，未相位修正）：WT 定峰用，FT 同用
SHELLS = [("Fe–O", 1.0, 1.8), ("Fe–Fe(1)", 2.2, 3.2), ("Fe–Fe(2)", 3.3, 3.9)]


# ================================================================
# 1. 数据处理（S1–S6，一层封装）
# ================================================================

def process_scans(files, absorber: str = "Fe", strict: bool = False,
                  kmin: float = KMIN, kmax: float = KMAX,
                  kweight: int = 2,
                  e0_method: str = "derivative",
                  manual_e0: float | None = None,
                  rbkg: float | None = None,
                  post_window=(150.0, 600.0),
                  despike: bool = False,
                  deglitch: list[tuple[float, float]] | None = None,
                  deadtime: bool = False,
                  reject_outlier_scans: bool = False,
                  self_absorption_formula: str | None = None,
                  self_absorption_thickness: float | None = None,
                  self_absorption_density: float | None = None,
                  self_absorption_angles: tuple = (45.0, 45.0),
                  self_absorption_line: str | None = None,
                  mback: bool = False, chi_std=None, chi_std_k=None,
                  clamp_lo: float | None = None,
                  clamp_hi: float | None = None) -> dict:
    """读数 → 对齐(可降级) → 合并 → E0 → 归一化 → Rbkg → χ(k) → FT → WT。

    编排在 xafs_pipeline.process_sample（与 run_xafs 同一条链），
    这里只把 PipelineResult 展平成本模块画图用的 dict。
    """
    # 【P1.1/#16】与 run_xafs.process_sample 入口同一条链：先按吸收元素裁出
    # 有效 kmax（Ce L3→10.2 Å⁻¹），再灌进 χ(k)/FT/WT。operando / phase /
    # 单谱 / 演示四条路径都经此函数，共用同一上限，不再用 raw --kmax 把
    # 下一条边的阶跃卷入 EXAFS。
    import run_xafs
    kmax, kmax_clamp = run_xafs.bounded_kmax(kmax, absorber)
    res = xafs_pipeline.process_sample(
        files, align=True, absorber=absorber,
        e0_method=e0_method, manual_e0=manual_e0, rbkg=rbkg,
        kmin=kmin, kmax=kmax, kweight=kweight,
        post_window=post_window,
        strict=strict, despike=despike, deglitch=deglitch,
        deadtime=deadtime,
        reject_outlier_scans=reject_outlier_scans,
        mback=mback, chi_std=chi_std, chi_std_k=chi_std_k,
        clamp_lo=clamp_lo, clamp_hi=clamp_hi,
        self_absorption_formula=self_absorption_formula,
        self_absorption_thickness=self_absorption_thickness,
        self_absorption_density=self_absorption_density,
        self_absorption_angles=self_absorption_angles,
        self_absorption_line=self_absorption_line)
    res.data_sha256 = cn_table.data_sha256(files)
    res.kmax_clamp = kmax_clamp
    return {"scans": res.scans, "merged": res.merged, "e0": res.e0["e0_sample"],
            "norm": res.norm, "rbkg": res.rbkg, "chi_k": res.chi_k,
            "ft": res.ft, "wt": res.wt, "r1": res.r1, "result": res,
            "xanes_features": res.xanes_features}


def white_line(st: dict):
    """白线（归一化谱在 E0±窗口内的最大值）：返回 (能量, 高度)。"""
    e0, norm = st["e0"], st["norm"]
    m = (norm.energy >= e0 - 10) & (norm.energy <= e0 + 30)
    i = int(np.argmax(norm.mu_norm[m]))
    return float(norm.energy[m][i]), float(norm.mu_norm[m][i])


def feff_ss_paths(sites, rmax, workdir, title, ss_max=4.1,
                  absorber: str = "Fe", edge: str = "K"):
    """生成（或复用缓存的）FEFF 路径，返回按 Reff 排序的单散射行。"""
    workdir = Path(workdir)
    if not list(workdir.glob("feff*.dat")):
        inp = artemis.build_feff_input(sites, absorber=absorber, rmax=rmax,
                                       title=title, edge=edge)
        artemis.run_feff6l(inp, workdir)
    rows = artemis.describe_paths(sorted(workdir.glob("feff*.dat")), 40)
    return sorted((r for r in rows if r["legs"] == 2 and r["reff"] <= ss_max),
                  key=lambda r: r["reff"])


def pick_paths(workdir, expected):
    """
    按 (文件名, 期望 Reff) 选定有效壳层的代表路径。

    合并壳层（ΔReff < π/2Δk ≈ 0.26 Å）的**代表路径一经基准验证即固定**：
    同壳层内换一条路径会改变 F(k)/φ(k) 形状，拟合出的 N 与 Rf 随之漂移
    （实测 Fe-Fe(2) 换 3.87→3.93 Å 路径，N 从 1.0 漂到 3.8）。
    因此这里显式校验每条路径的 Reff，缓存重建后若 FEFF 输出变化，
    在选路径阶段就报错，而不是带进拟合。
    """
    workdir = Path(workdir)
    files, rows = [], []
    for fname, reff_want in expected:
        f = workdir / fname
        row = artemis.describe_paths([f], 1)[0]
        if abs(row["reff"] - reff_want) > 0.01:
            raise XAFSDataError(
                "FEFF", f"{fname} 的 Reff={row['reff']:.3f} Å，"
                        f"与基准 {reff_want} Å 不符——FEFF 缓存可能已重建，"
                        f"代表路径需重新验证后再更新 expected 表")
        files.append(f)
        rows.append(row)
    return files, rows


# ================================================================
# 2. 绘图样式与拟合参数表（结构主图由 xafs_plots 统一出图）
# ================================================================

def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 1.0, "font.size": 11,
    })


# ================================================================
# 3. 拟合表（图 + CSV + TSV）
# ================================================================

def _fmt(p):
    n = (f"{p.degeneracy:.1f}±{p.n_err:.1f}" if p.n_err is not None
         else f"{p.degeneracy:.0f}")
    return n, f"{p.r:.3f}±{p.delta_r_err:.3f}"


def fit_table_rows(fits) -> list:
    """fits: [(样品名, [壳层名...], FitResult), ...] → 表体行。"""
    rows = []
    for sample, shells, fit in fits:
        for i, p in enumerate(fit.paths):
            n, r = _fmt(p)
            rows.append([
                sample if i == 0 else "",
                shells[i], n, r, f"{p.sigma2:.4f}",
                f"{fit.delta_e0:+.1f}±{fit.delta_e0_err:.1f}" if i == 0 else "",
                f"{fit.r_factor:.4f}" if i == 0 else "",
                "" if p.c3 is None else f"{p.c3:+.2e}",
            ])
    return rows


def fit_rows_from_output(sample: str, fit_out: dict | None) -> list:
    """S7 输出（motif 或 L1）→ 拟合表行，不要求存在第二个标样。"""
    if not fit_out:
        return []
    gated = (fit_out.get("cn_claim_allowed") is False
             or fit_out.get("cn_reliable") is False)
    if fit_out.get("mode") in ("motif", "motif_weights"):
        labels = [sh["label"] for sh in fit_out["shells"]]
        rows = fit_table_rows([(sample, labels, fit_out["fit"])])
        # 【红线2】门禁否决时绝不在 CSV/PNG 表里留下可被引用的 CN 数值。
        # 表头与列结构保持不变，未触发门禁的基线结果逐位一致。
        if gated:
            for row in rows:
                row[2] = "n/a (gated)"
        return rows
    l1 = fit_out["l1"]
    n = f"{l1['N']:.1f}"
    if l1.get("N_err") is not None:
        n += f"±{l1['N_err']:.1f}"
    r = f"{l1['R']:.3f}"
    if l1.get("R_err") is not None:
        r += f"±{l1['R_err']:.3f}"
    if gated:
        n = "n/a (gated)"
    return [[sample, f"{l1.get('ligand', 'X')} first shell", n, r,
             f"{l1['sigma2']:.4f}", f"{l1['delta_e0']:+.1f}",
             f"{l1['rf']:.4f}",
             "" if l1.get("c3") is None else f"{float(l1['c3']):+.2e}"]]


# 表头定义与 CSV 格式化已抽离到 xafs_writer；此处保留别名以维持公共 API。
FIT_HEADER = xafs_writer.FIT_HEADER


def plot_fit_table(rows: list, out: Path, stem: str, title: str,
                   s02: float = S02, kmin: float = KMIN,
                   kmax: float = KMAX, weights_mode: bool = False,
                   kweights=None):
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=18)
    headers = list(FIT_HEADER)
    if weights_mode:
        headers[2] = "N_eff"
    # C3 只在本次拟合真的放开了 C3 时作为扩展列出现；默认拟合的表格
    # 结构与历史完全一致（7 列），避免零退化核查被格式变化干扰。
    has_c3 = any(len(r) > 7 and str(r[7]).strip() for r in rows)
    if has_c3:
        headers.append("C3 (Å³)")
        table_rows = rows
        widths = [0.19, 0.21, 0.09, 0.13, 0.11, 0.11, 0.08, 0.10]
    else:
        table_rows = [r[:7] for r in rows]
        widths = [0.20, 0.23, 0.10, 0.14, 0.12, 0.12, 0.09]
    max_sample_len = max((len(str(r[0])) for r in rows), default=8)
    tbl = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center",
                   loc="center", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5 if max_sample_len > 22 else 9.5 if max_sample_len > 15 else 10)
    tbl.scale(1, 1.9)
    group_first = {1}
    r = 1
    while r < len(rows):
        if rows[r][0]:
            group_first.add(r + 1)
        r += 1
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linestyle(":")
        if r == 0 or r in group_first:
            cell.set_linestyle("-")
            cell.set_edgecolor("#444444")
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
    if kweights is None:
        kw_txt = "2"
    elif isinstance(kweights, (list, tuple)):
        kw_txt = ", ".join(str(int(w)) for w in kweights) if kweights else "2"
    else:
        kw_txt = str(kweights)
    foot = ("$^a$CN: coordination numbers. $^b$R: bond distance. "
            "$^c\\sigma^2$: Debye-Waller factors. "
            "$^d\\Delta E_0$: the inner potential correction. "
            "R-factor: goodness of fit.\n"
            f"$S_0^2$ = {s02} fixed. Scattering paths from FEFF6; "
            "same-shell paths are represented by the highest-degeneracy "
            "path with summed N.\n"
            f"Fit ranges: $k$ = {kmin:g}–{kmax:g} Å$^{{-1}}$, "
            f"$k_{{\\rm weight}}$ = {kw_txt}; R range is shown per fit. "
            f"CN absolute systematic budget ≈ ±"
            f"{100 * xafs_fit.CN_SYSTEMATIC_BUDGET['relative_1sigma']:.0f}% "
            "(1σ); quoted errors are formal only.")
    if weights_mode:
        foot = ("$^a$N: effective degeneracy $N_{\\rm cryst}w$ "
                "(NOT coordination number). $^b$R: bond distance. "
                "$^c\\sigma^2$: Debye-Waller factors. "
                "$^d\\Delta E_0$: the inner potential correction. "
                "R-factor: goodness of fit.\n"
                f"$S_0^2$ = {s02} fixed. Mixture weights are linear "
                "combination fractions of component χ, not independent "
                "phase quantification. Scattering paths from FEFF6.\n"
                f"Fit ranges: $k$ = {kmin:g}–{kmax:g} Å$^{{-1}}$, "
                f"$k_{{\\rm weight}}$ = {kw_txt}; R range is shown per fit. "
                "Phase-fraction reliability is reported separately "
                "(weights_reliable).")
    fig.text(0.02, 0.02, foot, fontsize=8.5, va="bottom")
    xafs_writer.safe_savefig(fig, out / f"{stem}_fit_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_fit_table(rows: list, out: Path, stem: str):
    """导出拟合参数表 CSV；格式化与防覆盖委托 xafs_writer。"""
    return xafs_writer.write_fit_table(rows, out, stem)


# ================================================================
# 4. 关键物理指标（打消读图猜测）
# ================================================================

def export_indicators(st: dict, fit_l, fit_f, out: Path, stem: str,
                      sample: str = "lepidocrocite (γ-FeOOH)",
                      edge_label: str = "Fe K-edge", s02: float = S02,
                      shells=SHELLS, kmin: float = KMIN,
                      kmax: float = KMAX):
    wt, chi = st["wt"], st["chi_k"]
    wl_e, wl_h = white_line(st)
    shell_peaks = []
    for label, lo, hi in shells:
        k_p, r_p, _ = larch_wt.locate_shell(wt, lo, hi)
        shell_peaks.append((label, k_p, r_p))
    rows = [
        ("sample", sample, "", "样品"),
        ("absorbing_edge", edge_label, "", "吸收边（防元素误判）"),
        ("e0", f"{st['e0']:.2f}", "eV", "E0，一阶导数峰法，合并谱"),
        ("edge_step", f"{st['norm'].edge_step:.4f}", "Δμ", "归一化边跳变"),
        ("white_line", f"{wl_e:.1f} eV, height {wl_h:.2f}", "",
         "白线位置与归一化高度"),
        ("post_edge_plateau", f"{st['norm'].plateau:.3f}", "",
         "边后平台（归一化应≈1）"),
        ("rbkg", f"{st['rbkg']['rbkg']:.2f}", "Å",
         f"AUTOBK 截断，confidence = {st['rbkg']['confidence']}"),
        ("k_data_range",
         f"{chi.k[chi.k > 2.0].min():.2f}–{chi.k.max():.2f}", "Å⁻¹",
         "数据 k 覆盖范围（k>2 起）"),
        ("k_window", f"{kmin:.0f}–{kmax:.0f}", "Å⁻¹",
         f"FT 与拟合窗口（kweight={int(getattr(st.get('ft') or getattr(st.get('result'), 'ft', None), 'kweight', 2))}, dk=1）"),
        ("ft_first_peak", f"{st['r1']:.3f}", "Å",
         "|χ(R)| 第一峰（未相位修正）"),
        ("phase_correction", "not applied", "",
         "未修正 R 比真实键长短约 0.4–0.5 Å"),
        ("s02", f"{s02}", "", "振幅因子，由标样固定"),
        ("delta_e0", f"{fit_l.delta_e0:+.2f}±{fit_l.delta_e0_err:.2f}", "eV",
         "内势校正（验收判据 |ΔE0| < 15 eV）"),
        ("r_factor_primary", f"{fit_l.r_factor:.4f}", "",
         "主样品拟合优度"),
        ("r_factor_calibration", f"{fit_f.r_factor:.4f}", "",
         "标样拟合优度"),
        ("fit_points_vs_vars",
         f"{fit_l.n_independent:.1f} vs {fit_l.n_variables}", "",
         "独立点数 vs 变量数（必须 vars < points）"),
        ("cn_systematic_budget",
         f"±{100 * xafs_fit.CN_SYSTEMATIC_BUDGET['relative_1sigma']:.0f}% (1σ)",
         "", xafs_fit.CN_SYSTEMATIC_BUDGET["note"]),
    ]
    for label, k_p, r_p in shell_peaks:
        key = label.replace("–", "-").replace(" ", "_")
        rows.append((f"wt_{key}_peak", f"k={k_p:.1f}, R={r_p:.2f}",
                     "Å⁻¹, Å", f"小波 {label} 最强响应位置"))
    for i, p in enumerate(fit_l.paths):
        name = (shells[i][0] if i < len(shells) else p.label)
        n, r = _fmt(p)
        rows.append((f"fit_{name}",
                     f"N={n}, R={r} Å, σ²={p.sigma2:.4f} Å²", "",
                     "壳层拟合值（N 为拟合配位数）"))
    if fit_l.correlation is not None and fit_l.parameter_names:
        c = np.array(fit_l.correlation, dtype=float, copy=True)
        np.fill_diagonal(c, 0.0)
        if c.size:
            ij = np.unravel_index(np.argmax(np.abs(c)), c.shape)
            val = abs(float(c[ij]))
            if val > 0:
                rows.append(
                    ("max_parameter_correlation",
                     f"{val:.3f} ({fit_l.parameter_names[ij[0]]} vs "
                     f"{fit_l.parameter_names[ij[1]]})", "",
                     "最强参数相关；>0.9 时各误差棒不能独立解读"))
    xafs_writer.write_csv(out / f"{stem}_indicators.csv",
                          ["indicator", "value", "unit", "说明"], rows)
    return rows


def parse_shell_spec(spec: str | None) -> list:
    """把 ``"Fe-O:1.0:1.8,Fe-Fe:2.2:3.2"`` 解析成壳层窗口列表。"""
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise XAFSDataError(
                "REPORT", f"--shells 项 {item!r} 格式应为 label:lo:hi")
        label, lo, hi = parts[0].strip(), float(parts[1]), float(parts[2])
        if hi <= lo:
            raise XAFSDataError(
                "REPORT", f"--shells 项 {item!r} 的 hi 必须大于 lo")
        out.append((label, lo, hi))
    return out


def _pretty_shell_labels(labels) -> list:
    """把 ``Fe-O@2.04`` 这类机器标签转成图上的 ``Fe–O`` / ``Fe–Fe(1)``。"""
    counts = {}
    out = []
    for raw in labels:
        base = raw.split("@")[0]
        if "-" in base:
            absorber, el = base.split("-", 1)
        else:
            absorber, el = "", base
        counts[el] = counts.get(el, 0) + 1
        idx = counts[el]
        if el == absorber or idx > 1:
            out.append(f"{absorber}–{el}({idx})")
        else:
            out.append(f"{absorber}–{el}")
    return out


# 【并发安全】模块级缓存必须加锁：流水线以进程隔离为主时无碍，但一旦作为
# Web 后端被多线程/协程调用，字典的 check-then-set 会产生竞态（重复计算、
# 半写入的对象被另一个线程读到）。用 double-checked locking，重活（FEFF
# 参考谱计算）在锁外做，只在写入时短暂持锁。
_REF_K_CACHE: dict = {}
_REF_K_CACHE_LOCK = threading.RLock()


def clear_ref_k_cache() -> None:
    """测试/长驻服务显式失效缓存（避免跨样品参数变化后命中旧标尺）。"""
    with _REF_K_CACHE_LOCK:
        _REF_K_CACHE.clear()


def auto_ref_k_peaks(res, fit_obj, absorber_symbol: str):
    """B9-2 自动生成 WT 元素标尺（复用 r_bond=2.2, σ²=0.003 的 FEFF 缓存）。

    元素集合 = 拟合路径散射元素 ∪ {O, 吸收元素}；同一 (吸收元素, 边,
    元素集, k/WT 参数) 只算一次。拟合结果缺失或没有路径时返回 None，
    不生成一条假标尺。
    """
    if fit_obj is None or not getattr(fit_obj, "paths", None):
        return None
    elements = []
    for path in fit_obj.paths:
        el = artemis._path_ligand(path, absorber_symbol)
        if el and el not in elements:
            elements.append(el)
    for el in ("O", absorber_symbol):
        if el and el not in elements:
            elements.append(el)
    wt = getattr(res, "wt", None)
    sigma = float(getattr(wt, "sigma", 1.5))
    key = (str(absorber_symbol), float(fit_obj.kmin), float(fit_obj.kmax),
           int(res.ft.kweight), round(sigma, 6), tuple(elements))
    with _REF_K_CACHE_LOCK:
        cached = _REF_K_CACHE.get(key)
    if cached is None:
        computed = structure_id.reference_element_k_peaks(
            absorber_symbol, elements=tuple(elements), r_bond=2.2,
            sigma2=0.003, kmin=float(fit_obj.kmin),
            kmax=float(fit_obj.kmax), kweight=int(res.ft.kweight),
            wt_sigma=sigma)
        with _REF_K_CACHE_LOCK:
            # setdefault 保证并发下所有调用方拿到同一个对象（回归测试依赖此）
            cached = _REF_K_CACHE.setdefault(key, computed)
    return cached


def shells_from_fit(fit_out: dict, absorber_symbol: str | None = None) -> list:
    """从 S7 结果推 FT 标注窗口（R 未相位修正，约比拟合 R 小 0.4–0.5 Å）。"""
    if not fit_out:
        return []
    if fit_out.get("mode") in ("motif", "motif_weights"):
        labels = _pretty_shell_labels(
            [sh["label"] for sh in fit_out["shells"]])
        shells = []
        for label, p in zip(labels, fit_out["fit"].paths):
            lo = max(0.0, p.r - 0.85)
            hi = max(lo + 0.20, p.r - 0.05)
            shells.append((label, lo, hi))
        return shells
    l1 = fit_out.get("l1") or {}
    if "R" in l1:
        lo = max(0.0, float(l1["R"]) - 0.85)
        hi = max(lo + 0.20, float(l1["R"]) - 0.05)
        ligand = l1.get("ligand", "X")
        label = (f"{absorber_symbol}–{ligand} (L1)"
                 if absorber_symbol else f"{ligand} first shell")
        return [(label, lo, hi)]
    return []


def fit_summary(fit_out: dict | None) -> dict:
    """把含 FitResult 的 S7 结果转成 JSON 可写的摘要。"""
    if not fit_out:
        return {"mode": "skipped"}
    summary = {k: v for k, v in fit_out.items()
               if k not in ("structure", "fit")}
    fit = fit_out.get("fit")
    if fit is not None:
        sf = getattr(fit, "nidp_scale_factor", None)
        summary["fit"] = {
            "delta_e0": fit.delta_e0, "delta_e0_err": fit.delta_e0_err,
            "r_factor": fit.r_factor, "n_independent": fit.n_independent,
            "n_variables": fit.n_variables, "overfit": fit.overfit,
            "nidp_scale_factor": sf,
            "rmin": fit.rmin, "rmax": fit.rmax,
            "kmin": fit.kmin, "kmax": fit.kmax, "s02": fit.s02,
            "fit_space": fit.fit_space,
            "aic": fit.aic, "bic": fit.bic,
            "bvs": fit.bvs, "bvs_target": fit.bvs_target,
            "bvs_chi2": fit.bvs_chi2,
            "kweights": list(fit.kweights or ()),
            "r_factors_by_kweight": (fit.r_factors_by_kweight or {}),
            "epsilon_r": fit.epsilon_r,
            "chi_square": fit.chi_square,
            "reduced_chi_square": fit.reduced_chi_square,
            "lattice_strain": fit.lattice_strain,
            "lattice_strain_err": fit.lattice_strain_err,
            "mixture_weights": fit.mixture_weights,
            "parameter_names": fit.parameter_names,
            "correlation": (np.round(fit.correlation, 5).tolist()
                            if fit.correlation is not None else None),
            "paths": [{"label": p.label, "reff": p.reff,
                       "N": p.degeneracy, "N_err": p.n_err,
                       "N_err_nidp": (None if (p.n_err is None or sf is None)
                                      else float(p.n_err) * float(sf)),
                       "R": p.r, "R_err": p.delta_r_err,
                       "R_err_nidp": (None if (p.delta_r_err is None or sf is None)
                                      else float(p.delta_r_err) * float(sf)),
                       "sigma2": p.sigma2, "sigma2_err": p.sigma2_err,
                       "sigma2_at_bound": p.sigma2_at_bound,
                       "c3": p.c3, "c3_err": p.c3_err,
                       "c3_at_bound": p.c3_at_bound,
                       "c4": p.c4, "c4_err": p.c4_err,
                       "c4_at_bound": p.c4_at_bound}
                      for p in fit.paths],
        }
        rdf_tab = xafs_fit.rdf_table(fit)
        if rdf_tab["summary"]:
            summary["rdf"] = rdf_tab["summary"]
    return summary


def export_indicators_single(st: dict, fit_out: dict | None, out: Path,
                             stem: str, sample: str, edge_label: str,
                             s02: float | None, shells: list,
                             kmin: float = KMIN, kmax: float = KMAX):
    """任意样品的指标表；不假设一定有第二个标样拟合。"""
    chi = st["chi_k"]
    wl_e, wl_h = white_line(st)
    xf = st.get("xanes_features") or {}

    def _opt(key, fmt):
        value = xf.get(key)
        return "unavailable" if value is None else format(value, fmt)

    neck = ("present" if xf.get("neck_shoulder_present") else "absent")
    split = float(xf.get("derivative_split_eV") or 0.0)
    rows = [
        ("sample", sample, "", "样品"),
        ("absorbing_edge", edge_label, "", "吸收边（防元素误判）"),
        ("e0", f"{st['e0']:.2f}", "eV", "E0，一阶导数峰法，合并谱"),
        ("edge_step", f"{st['norm'].edge_step:.4f}", "Δμ", "归一化边跳变"),
        ("white_line", f"{wl_e:.1f} eV, height {wl_h:.2f}", "",
         "白线位置与归一化高度"),
        ("e_half_height", _opt("e_half_height", ".2f"), "eV",
         "flattened 谱半高能量（0.5 交叉，线性插值）"),
        ("white_line_height", _opt("white_line_height", ".4f"), "",
         "flattened 谱白线高度（[E0−2, E0+30] 最大值）"),
        ("pre_edge_area", _opt("pre_edge_area", ".4f"), "eV",
         "边前峰面积（扣两端点连线基线，≥0）"),
        ("pre_edge_centroid", _opt("pre_edge_centroid", ".2f"), "eV",
         "边前峰面积加权质心（Pseudo-Voigt 分峰，B6-3）"),
        ("pre_edge_n_peaks", str(xf.get("pre_edge_n_peaks") or 0), "",
         "边前分峰数（1 或 2；窗口点数不足降级时沿用直线基线峰位）"),
        ("pre_edge_voigt_r_factor", _opt("pre_edge_voigt_r_factor", ".4f"), "",
         "边前 Pseudo-Voigt 分峰残差因子"),
        ("xanes_neck_shoulder", neck, "",
         f"颈部肩峰；导数双峰间距 {split:.2f} eV"
         + ("（E0 跳变风险）" if xf.get("e0_jump_risk") else "")),
        ("e0_jump_risk", str(bool(xf.get("e0_jump_risk"))), "",
         "导数双峰近等高，E0 可能在两台阶间跳变 3–6 eV"),
        ("post_edge_plateau", f"{st['norm'].plateau:.3f}", "",
         "边后平台（归一化应≈1）"),
        ("rbkg", f"{st['rbkg']['rbkg']:.2f}", "Å",
         f"AUTOBK 截断，confidence = {st['rbkg']['confidence']}"),
        ("k_data_range",
         f"{chi.k[chi.k > 2.0].min():.2f}–{chi.k.max():.2f}", "Å⁻¹",
         "数据 k 覆盖范围（k>2 起）"),
        ("k_window", f"{kmin:.0f}–{kmax:.0f}", "Å⁻¹",
         f"FT 与拟合窗口（kweight={int(getattr(st.get('ft') or getattr(st.get('result'), 'ft', None), 'kweight', 2))}, dk=1）"),
        ("ft_first_peak", f"{st['r1']:.3f}", "Å",
         "|χ(R)| 第一峰（未相位修正）"),
        ("phase_correction", "not applied", "",
         "未修正 R 比真实键长短约 0.4–0.5 Å"),
        ("s02", "unavailable" if s02 is None else f"{s02}", "",
         "振幅因子，由标样固定"),
    ]
    kc = np.asarray(getattr(st["wt"], "k_centers", []), dtype=float)
    kc_min = float(np.nanmin(kc)) if kc.size else float("-inf")
    kc_max = float(np.nanmax(kc)) if kc.size else float("inf")
    try:
        wt_sanity = xafs_plot_special.assess_wt_shell_sanity(
            st.get("result") or st, fit_out)
    except Exception:  # noqa: BLE001
        wt_sanity = {}
    sanity_by_norm = {
        str(s["label"]).replace("–", "-"): s
        for s in (wt_sanity.get("shells") or [])
    }
    for label, lo, hi in shells:
        key = label.replace("–", "-").replace(" ", "_")
        sh_info = sanity_by_norm.get(label.replace("–", "-"))
        if sh_info is not None:
            k_p = float(sh_info["k_peak"])
            r_p = float(sh_info.get("r_peak", sh_info["r_anchor"]))
        else:
            k_p, r_p, _ = larch_wt.locate_shell(st["wt"], lo, hi)
        tags = []
        if k_p <= kc_min + 0.05:
            tags.append("edge! 死贴左边界，勿作元素凭据")
        elif k_p >= kc_max - 0.05:
            tags.append("edge! 死贴右边界，警惕高频噪声")
        if sh_info and sh_info.get("masked_by_heavy"):
            kth = sh_info.get("k_peak_th")
            th_s = f"{float(kth):.1f}" if kth is not None and np.isfinite(kth) else "~5.2"
            tags.append(f"masked by {sh_info['masked_by']}; FEFF 理论 k≈{th_s}，勿误认重散射峰")
        elif sh_info and sh_info.get("light_heavy_mismatch"):
            tags.append("light-on-heavy mismatch! 轻壳层落入高 k 区，疑漏重散射原子")
        edge_tag = f" [{'；'.join(tags)}]" if tags else ""
        rows.append((f"wt_{key}_peak", f"k={k_p:.1f}, R={r_p:.2f}",
                     "Å⁻¹, Å", f"小波 {label} 最强响应位置{edge_tag}"))
    if fit_out is None:
        rows.append(("fit", "skipped", "", "未运行 S7（--no-fit）"))
    elif fit_out.get("mode") in ("motif", "motif_weights"):
        fit = fit_out["fit"]
        is_weights = fit_out.get("mode") == "motif_weights"
        rows.extend([
            ("delta_e0", f"{fit.delta_e0:+.2f}±{fit.delta_e0_err:.2f}", "eV",
             "内势校正（验收判据 |ΔE0| < 15 eV）"),
            ("r_factor", f"{fit.r_factor:.4f}", "", "拟合优度"),
            ("fit_space", fit.fit_space, "",
             "拟合空间（r/q/wt；不同空间 R-factor 不可横向比较）"),
            ("aic",
             "unavailable" if fit.aic is None else f"{fit.aic:.3f}", "",
             "AIC = n·ln(RSS/n)+2k（仅同一数据/空间内可比）"),
            ("bic",
             "unavailable" if fit.bic is None else f"{fit.bic:.3f}", "",
             "BIC = n·ln(RSS/n)+k·ln n（对多参数惩罚更重）"),
            ("bvs",
             "unavailable" if fit.bvs is None else f"{fit.bvs:.4f}", "vu",
             "键价和（软约束开启时输出，目标 "
             + ("n/a" if fit.bvs_target is None else f"{fit.bvs_target:.3f}")
             + "）；χ²_BVS="
             + ("n/a" if fit.bvs_chi2 is None else f"{fit.bvs_chi2:.4g}")),
            ("fit_points_vs_vars",
             f"{fit.n_independent:.1f} vs {fit.n_variables}", "",
             "独立点数 vs 变量数（必须 vars < points）"),
            ("epsilon_r",
             "unavailable" if fit.epsilon_r is None
             else f"{fit.epsilon_r:.4g}", "Å⁻²",
             "高 R 区 [15,25] Å 噪声估计 ε_R（IFEFFIT 口径）"),
            ("reduced_chi_square",
             "unavailable" if fit.reduced_chi_square is None
             else f"{fit.reduced_chi_square:.4g}", "",
             "χ²_ν = χ² / max(N_idp − N_var, 1)；与 ε_R 一起解读"),
        ])
        sft = fit_out.get("shell_f_test")
        if sft:
            conf = sft.get("confidence")
            fval = sft.get("F")
            if conf is None:
                rows.append(
                    ("outer_shells_f_test_confidence", "unavailable", "",
                     "增减壳层 F-test 未生成：" + str(sft.get("error", ""))))
            else:
                rows.append(
                    ("outer_shells_f_test_confidence", f"{conf:.4f}", "",
                     "新增壳层 F-test 置信度（≥0.95 且 dfd≥1 判显著）；"
                     f"F={float(fval):.4g}, dfn={sft.get('dfn')}, "
                     f"dfd={float(sft.get('dfd', 0.0)):.2f}, "
                     f"significant={bool(sft.get('significant'))}"))
        if is_weights:
            for w in (fit_out.get("mixture_weights") or []):
                rows.append(
                    (f"phase_fraction_{w['group']}",
                     f"{w['weight']:.4f}±{w['weight_err']:.4f}", "",
                     "相分数（各组分 χ 的线性组合权重，和=1；误差为形式误差）"))
        for p in fit.paths:
            n, r = _fmt(p)
            if is_weights:
                rows.append((f"fit_{p.label}",
                             f"N_eff={n}, R={r} Å, σ²={p.sigma2:.4f} Å²", "",
                             "壳层有效简并度 N_crystal·w（不是配位数）"))
            else:
                c3_txt = ("" if p.c3 is None else
                          f", C3={p.c3:+.2e}±{(p.c3_err or 0.0):.1e} Å³")
                rows.append((f"fit_{p.label}",
                             f"N={n}, R={r} Å, σ²={p.sigma2:.4f} Å²"
                             + c3_txt, "",
                             "壳层拟合值（N 为壳层总配位数）"
                             + ("；C3 为非谐三阶累积量" if p.c3 is not None
                                else "")))
        if fit.correlation is not None and fit.parameter_names:
            c = np.array(fit.correlation, dtype=float, copy=True)
            np.fill_diagonal(c, 0.0)
            if c.size:
                ij = np.unravel_index(np.argmax(np.abs(c)), c.shape)
                val = abs(float(c[ij]))
                if val > 0:
                    rows.append(
                        ("max_parameter_correlation",
                         f"{val:.3f} ({fit.parameter_names[ij[0]]} vs "
                         f"{fit.parameter_names[ij[1]]})", "",
                         "最强参数相关；>0.9 时各误差棒不能独立解读"))
    else:
        l1 = fit_out["l1"]
        rows.extend([
            ("delta_e0", f"{l1['delta_e0']:+.2f}", "eV", "第一壳内势校正"),
            ("r_factor", f"{l1['rf']:.4f}", "", "第一壳拟合优度"),
            ("fit_points_vs_vars",
             f"{l1.get('n_independent', float('nan')):.1f} vs "
             f"{l1.get('n_variables', 0)}", "", "独立点数 vs 变量数"),
            (f"fit_{l1.get('ligand', 'X')}_first_shell",
             f"N={l1['N']}±{l1['N_err']}, R={l1['R']}±{l1['R_err']} Å, "
             f"σ²={l1['sigma2']:.4f} Å²"
             + ("" if l1.get("c3") is None else
                f", C3={float(l1['c3']):+.2e} Å³"), "",
             "L1 第一壳拟合（长程结构未知）"),
        ])
    if fit_out:
        c3_drop = (fit_out.get("flags") or {}).get(
            "c3_dropped_by_nidp", fit_out.get("c3_dropped_by_nidp"))
        if c3_drop is not None:
            rows.append(
                ("c3_dropped_by_nidp", str(bool(c3_drop)), "",
                 "C3 因自由变量数 ≥ N_idp 被自动降级/关闭"))
    if fit_out:
        rows.extend([
            ("structure_status", str(fit_out.get("status", "")), "",
             "S6.5 判定：confident/partial/ambiguous/no_match"),
            ("coverage_ok", str(fit_out.get("coverage_ok")), "",
             "最优候选是否解释全部显著实验峰"),
            ("cn_reliable", str(fit_out.get("cn_reliable")), "",
             "CN 是否可作结论（Rf/过拟合/ΔE0/N 误差综合门槛）"),
        ])
        sa = fit_out.get("self_absorption")
        if sa:
            if sa.get("corrected"):
                sa_note = (f"Booth 校正: {sa.get('formula')} "
                           f"d={sa.get('thickness_um')} μm, "
                           f"ρ={sa.get('density')} g/cm³, "
                           f"φ/θ={sa.get('anginp')}/{sa.get('angout')}°, "
                           f"因子中位数={sa.get('correction_factor_median')}")
            else:
                sa_note = sa.get("note", "荧光自吸收风险筛查（未做定量校正）")
            rows.append(
                ("self_absorption",
                 f"mode={sa.get('mode')}, risk={sa.get('risk')}, "
                 f"corrected={sa.get('corrected')}", "", sa_note))
        if fit_out.get("mode") == "motif_weights":
            rows.append(
                ("weights_reliable", str(fit_out.get("weights_reliable")), "",
                 "相分数是否可作结论（Rf/过拟合/ΔE0/权重撞界/权重误差/组分简并综合门槛）"))
    budget = ((fit_out or {}).get("cn_systematics")
              or xafs_fit.CN_SYSTEMATIC_BUDGET)
    rows.append(
        ("cn_systematic_budget",
         f"±{100 * budget['relative_1sigma']:.0f}% (1σ)",
         "", budget.get("note", "")))
    ws = (fit_out or {}).get("weights_systematics")
    if ws:
        rows.append(
            ("weight_systematic_budget",
             f"±{100 * ws['relative_1sigma']:.0f}% (1σ)", "",
             ws.get("note", "")))
    res_obj = st.get("result")
    if res_obj is not None:
        import data_audit
        audit_res = getattr(res_obj, "audit", None) or data_audit.audit_preprocessing(res_obj, kmax_used=kmax)
        audit_msgs = list(audit_res.get("fails") or []) + list(audit_res.get("warnings") or [])
        rows.append(
            ("data_quality_audit", audit_res.get("status", "ok"), "",
             "; ".join(audit_msgs) if audit_msgs else "S1–S5 预处理/标尺/高 k 噪声/Rbkg 自审计通过"))
    xafs_writer.write_csv(out / f"{stem}_indicators.csv",
                          ["indicator", "value", "unit", "说明"], rows)
    return rows


# ================================================================
# 5. 逐面板 CSV（图-数据一一对应）
# ================================================================

def _write_num_csv(path, header, columns):
    xafs_writer.write_csv(
        path, header,
        [[f"{v:.6g}" for v in row] for row in zip(*columns)])


def export_panel_csvs(st: dict, out: Path, stem: str):
    scans, merged, norm = st["scans"], st["merged"], st["norm"]
    chi_k, ft, wt = st["chi_k"], st["ft"], st["wt"]

    # (a) μ(E)：各扫描插值到合并网格
    E = np.asarray(merged.energy)
    cols = [E] + [np.interp(E, s.energy, s.mu) for s in scans] \
        + [np.asarray(merged.mu)]
    _write_num_csv(out / f"{stem}_mu.csv",
                   ["energy_eV"] + [f"mu_scan{i + 1}" for i in range(len(scans))]
                   + ["mu_merged"], cols)

    # (b) XANES（含 flattened 谱，对标 Athena 的 mu_flat 显示）
    mu_flat = (norm.flattened if norm.flattened is not None
               else norm.mu_norm)
    _write_num_csv(out / f"{stem}_xanes.csv",
                   ["energy_eV", "mu", "pre_edge_line", "mu_norm", "mu_flat"],
                   [norm.energy, norm.mu, norm.pre_edge_line, norm.mu_norm,
                    mu_flat])

    # (c) k²χ(k)
    _write_num_csv(out / f"{stem}_chik.csv",
                   ["k_inv_A", "chi", "k2chi"],
                   [chi_k.k, chi_k.chi, chi_k.k ** 2 * chi_k.chi])

    # (d) χ(R)
    _write_num_csv(out / f"{stem}_chir.csv",
                   ["R_A", "chi_R_re", "chi_R_im", "magnitude"],
                   [ft.r, ft.chi_r.real, ft.chi_r.imag, ft.magnitude])

    # (d2) Back-FT：第一壳 R 窗滤波后的 χ(q)（对标 Athena 反傅里叶）
    r1 = float(st.get("r1", float("nan")))
    rmin = max(1.0, r1 - 0.5) if np.isfinite(r1) else 1.0
    rmax = min(3.4, r1 + 0.5) if np.isfinite(r1) else 2.6
    bft = athena.back_ft_to_q_space(ft.r, ft.chi_r, rmin, rmax)
    _write_num_csv(out / f"{stem}_chiq.csv",
                   ["q_inv_A", "chiq_re", "chiq_im", "chiq_mag"],
                   [bft.q, bft.real, bft.imag, bft.magnitude])

    # (e) 小波矩阵：行 = k，列 = R（WaveletMap.magnitude 的原生形状）
    # 旧版这里写成行=R/列=k，与 run_xafs 的输出互为转置；统一到
    # (len(k_centers), len(r))，避免下游拿两份 CSV 时方向对不上。
    xafs_writer.write_csv(
        out / f"{stem}_wavelet.csv",
        ["k\\R"] + [f"{r:.2f}" for r in wt.r],
        [[f"{kc:.2f}"] + [f"{wt.magnitude[i, j]:.4f}"
                          for j in range(len(wt.r))]
         for i, kc in enumerate(wt.k_centers)])


# ================================================================
# 6. 主流程
# ================================================================

_HTML_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: #f5f6f8; color: #1c2430;
       font: 14px/1.6 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
main { max-width: 1120px; margin: 0 auto; background: #fff; padding: 32px 40px;
       border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { margin: 34px 0 10px; font-size: 17px; border-bottom: 1px solid #e3e7ec;
     padding-bottom: 6px; }
.sub { color: #5a6675; margin: 0 0 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
         gap: 12px; margin: 18px 0 8px; }
.card { border: 1px solid #e3e7ec; border-radius: 8px; padding: 10px 12px;
        background: #fafbfc; }
.card .label { color: #68737f; font-size: 12px; }
.card .value { font-size: 17px; font-weight: 600; margin-top: 2px;
               overflow-wrap: anywhere; }
.card.good .value { color: #10764a; }
.card.warn .value { color: #9a6200; }
.card.bad  .value { color: #b3261e; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 13px; }
th, td { border: 1px solid #e1e5ea; padding: 6px 9px; text-align: left;
         vertical-align: top; }
th { background: #f0f3f6; font-weight: 600; }
tr.bad td { background: #fdecea; }
tr.warn td { background: #fff6e5; }
tr.bad td:first-child { box-shadow: inset 3px 0 #b3261e; }
tr.warn td:first-child { box-shadow: inset 3px 0 #c98a00; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
         font-size: 12px; font-weight: 600; }
.badge.good { background: #e3f4ea; color: #10764a; }
.badge.warn { background: #fff1d6; color: #8a5800; }
.badge.bad  { background: #fde2df; color: #a41d16; }
.note { background: #f7f9fb; border-left: 4px solid #9fb3c8; padding: 10px 14px;
        border-radius: 0 6px 6px 0; margin: 10px 0; color: #34404e; }
.note.bad { border-color: #b3261e; background: #fdf3f2; }
figure { margin: 14px 0 20px; }
figure img { width: 100%; height: auto; border: 1px solid #e3e7ec;
             border-radius: 6px; background: #fff; }
figcaption { color: #5a6675; font-size: 12px; margin-top: 5px; }
details { border: 1px solid #e3e7ec; border-radius: 8px; padding: 8px 14px;
          margin: 10px 0; background: #fbfcfd; }
summary { cursor: pointer; font-weight: 600; color: #34404e; }
ul.files { columns: 2; font-size: 13px; }
ul.files a { color: #1b6ca8; text-decoration: none; }
ul.files a:hover { text-decoration: underline; }
footer { margin-top: 34px; padding-top: 12px; border-top: 1px solid #e3e7ec;
         color: #7a8490; font-size: 12px; }
"""


def _html_badge(text: str, level: str) -> str:
    return f'<span class="badge {level}">{html.escape(str(text))}</span>'


def _embed_image(path: Path, caption: str) -> str:
    """把 PNG 内联为 data URI —— 单个 HTML 文件即可独立分发。"""
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img alt="{html.escape(caption)}" '
            f'src="data:image/png;base64,{b64}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def _indicator_row_class(key: str, value: str) -> str:
    """报告层只做“值得停下来看一眼”的标色，不替代物理门禁。"""
    key = key.lower()
    if key in ("cn_reliable", "weights_reliable"):
        return "good" if value.strip().lower() == "true" else "bad"
    if key == "max_parameter_correlation":
        try:
            rho = float(value.split()[0])
        except (ValueError, IndexError):
            return ""
        if rho >= 0.9:
            return "bad"
        return "warn" if rho >= 0.8 else ""
    if key == "r_factor":
        try:
            rf = float(value)
        except ValueError:
            return ""
        if rf > 0.08:
            return "bad"
        return "warn" if rf > 0.05 else "good"
    if key == "self_absorption":
        low = value.lower()
        if "corrected_unreliable" in low or "risk=high" in low:
            return "bad"
        return "warn" if "risk=possible" in low else ""
    return ""


def export_html_report(out: Path, stem: str, sample: str, edge_label: str,
                       indicator_rows: list, table_rows: list,
                       fit_out: dict | None,
                       extra_images: list | None = None) -> Path:
    """把本次运行的全部结构分析图表与门禁结果汇总成单文件 HTML。

    设计原则：
      - 图片以 base64 内联，HTML 可脱离目录独立发送；
      - 关键指标与门禁结论置顶，强相关（|ρ|≥0.8）与不可靠 CN 标红；
      - 不再附带 S1–S8 中间调试图与交互式调窗链接，聚焦催化剂结构剖析。
    """
    fit = (fit_out or {}).get("fit")
    flags = (fit_out or {}).get("flags") or {}
    self_abs = (fit_out or {}).get("self_absorption") or {}
    ind = {r[0]: (r[1], r[2], r[3]) for r in indicator_rows}

    # ---- 关键指标卡 ----
    status = (fit_out or {}).get("status", "—")
    status_level = ("good" if status == "confident"
                    else "warn" if status in ("partial", "ambiguous")
                    else "" if status == "demo"
                    else "bad" if fit_out else "")
    cn_raw = (fit_out or {}).get("cn_reliable")
    cn_level = ("good" if cn_raw is True
                else "bad" if cn_raw is False else "")
    rf = fit.r_factor if fit is not None else None
    rf_level = ("good" if rf is not None and rf <= 0.05
                else "warn" if rf is not None and rf <= 0.08
                else "bad" if rf is not None else "")
    corr_max, corr_pair = 0.0, ""
    if fit is not None and getattr(fit, "correlation", None) is not None \
            and getattr(fit, "parameter_names", None):
        c = np.array(fit.correlation, dtype=float, copy=True)
        np.fill_diagonal(c, 0.0)
        if c.size:
            ij = np.unravel_index(np.argmax(np.abs(c)), c.shape)
            corr_max = abs(float(c[ij]))
            corr_pair = (f"{fit.parameter_names[ij[0]]} vs "
                         f"{fit.parameter_names[ij[1]]}")
    corr_level = ("bad" if corr_max >= 0.9
                  else "warn" if corr_max >= 0.8 else "good")
    budget = ((fit_out or {}).get("cn_systematics")
              or xafs_fit.CN_SYSTEMATIC_BUDGET)
    budget_txt = f"±{100 * budget['relative_1sigma']:.0f}% (1σ)"

    sa_txt = (self_abs.get("risk", "—") if self_abs else "—")
    sa_level = ("bad" if sa_txt in ("high", "unknown", "corrected_unreliable")
                else "warn" if sa_txt == "possible" else "good")

    def card(label, value, level=""):
        return (f'<div class="card {level}"><div class="label">'
                f'{html.escape(label)}</div><div class="value">'
                f'{html.escape(str(value))}</div></div>')

    cards = "".join([
        card("识别结果", status, status_level),
        card("CN 可作结论", cn_raw, cn_level),
        card("R-factor", "—" if rf is None else f"{rf:.4f}", rf_level),
        card("最大参数相关", (f"{corr_max:.3f}" if corr_max else "—"),
             corr_level),
        card("CN 系统误差预算", budget_txt),
        card("自吸收风险", sa_txt, sa_level),
    ])

    # ---- 结论摘要 ----
    summary_bits = []
    if fit_out:
        if cn_raw is True:
            summary_bits.append("配位数在形式误差与系统预算内可用于结论。")
        elif cn_raw is False:
            summary_bits.append("配位数不可用于定量结论（原因见 flags）。")
        if fit_out.get("mode") == "motif_weights":
            wr = fit_out.get("weights_reliable")
            summary_bits.append(
                "相分数可作结论。" if wr is True else "相分数不可作结论。")
        if corr_max >= 0.8:
            summary_bits.append(
                f"参数强相关 |ρ|={corr_max:.3f}（{corr_pair}），"
                "各误差棒不能当独立精度解读。")
        if self_abs.get("corrected"):
            summary_bits.append(
                f"已做 Booth 自吸收校正（{self_abs.get('formula')}，"
                f"d={self_abs.get('thickness_um')} μm，"
                f"ρ={self_abs.get('density')} g/cm³）。")
        elif self_abs.get("risk") in ("high", "possible", "unknown"):
            summary_bits.append(
                f"荧光自吸收风险 = {self_abs.get('risk')}，"
                "未校正；振幅/N 可能被压低。")
    else:
        summary_bits.append("本次未运行 S7 拟合。")
    summary_html = "".join(
        f"<li>{html.escape(b)}</li>" for b in summary_bits)

    warning = (fit_out or {}).get("warning")
    warning_html = (f'<div class="note bad"><b>警告：</b>'
                    f'{html.escape(str(warning))}</div>' if warning else "")

    # ---- 指标表 ----
    ind_rows = []
    for key, value, unit, note in indicator_rows:
        cls = _indicator_row_class(key, str(value))
        ind_rows.append(
            f"<tr class='{cls}'><td>{html.escape(str(key))}</td>"
            f"<td>{html.escape(str(value))}</td>"
            f"<td>{html.escape(str(unit))}</td>"
            f"<td>{html.escape(str(note))}</td></tr>")
    ind_table = (
        "<table><thead><tr><th>指标</th><th>值</th><th>单位</th>"
        "<th>说明</th></tr></thead><tbody>"
        + "".join(ind_rows) + "</tbody></table>")

    # ---- 拟合表 ----
    if table_rows:
        fit_table = (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(h)}</th>" for h in FIT_HEADER)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(
                f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                for row in table_rows)
            + "</tbody></table>")
    else:
        fit_table = '<div class="note">未运行 S7 拟合，无拟合参数表。</div>'

    # ---- flags ----
    flag_rows = []
    for key, val in flags.items():
        if isinstance(val, bool):
            bad = val
            show = "是" if val else "否"
        elif isinstance(val, (list, tuple)):
            bad = bool(val)
            show = ", ".join(str(v) for v in val) if val else "无"
        else:
            bad = False
            show = str(val)
        cls = "bad" if bad else ""
        flag_rows.append(
            f"<tr class='{cls}'><td>{html.escape(key)}</td>"
            f"<td>{html.escape(show)}</td></tr>")
    flags_html = ("<table><thead><tr><th>门禁项</th><th>值</th></tr></thead>"
                  "<tbody>" + "".join(flag_rows) + "</tbody></table>"
                  if flag_rows else "")

    # ---- 图片 ----
    figs = []
    figs.append(_embed_image(
        out / f"{stem}_structure_main.png",
        "图 1：催化剂配位结构剖析总图（XANES、一阶导、k 空间拟合与残差、R 空间逐路径拆解、Morlet WT 与门禁卡片）"))
    figs.append(_embed_image(
        out / f"{stem}_shell_dissection.png",
        "图 2：壳层轻重鉴别与真实键长核验（k¹/k²/k³ 权重 |FT|、相位修正真实 R、逐壳层 1D WT 轮廓）"))
    figs.append(_embed_image(out / f"{stem}_fit_table.png",
                             "CN 拟合参数表（Table S 风格）"))
    for path, caption in (extra_images or []):
        figs.append(_embed_image(Path(path), caption))
    figs_html = "".join(f for f in figs if f)

    # ---- 数据文件清单 ----
    file_items = []
    if out.exists():
        for f in sorted(out.glob(f"{stem}*")):
            if f.suffix.lower() == ".html":
                continue
            file_items.append(
                f'<li><a href="{html.escape(f.name)}">'
                f'{html.escape(f.name)}</a></li>')
    files_html = (f'<ul class="files">{"".join(file_items)}</ul>'
                  if file_items else '<div class="note">无数据文件。</div>')

    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(sample)} — XAFS 报告</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<main>
<h1>{html.escape(sample)}</h1>
<p class="sub">{html.escape(edge_label)} · 生成于 {generated} · stem = {html.escape(stem)}</p>

<div class="cards">{cards}</div>
{warning_html}

<h2>结论摘要</h2>
<ul>{summary_html}</ul>

<h2>关键指标</h2>
{ind_table}

<h2>拟合参数表</h2>
{fit_table}

<h2>门禁 flags</h2>
{flags_html or '<div class="note">无 flags 记录。</div>'}

<h2>结构剖析图件</h2>
{figs_html or '<div class="note">本次运行未生成图像。</div>'}

<h2>数据文件</h2>
{files_html}

<footer>
XAFS 管线自动报告。形式误差与系统误差分开呈现：n_err 仅为最小二乘
形式误差，请同时阅读 CN 系统误差预算与 flags。未标定元素、无自吸收
参数或未通过门禁时，本报告不会给出可作结论的 CN/相分数。
</footer>
</main>
</body>
</html>
"""
    path = out / f"{stem}_report.html"
    return xafs_writer.safe_write_text(path, doc, encoding="utf-8")


def demo_main():
    """内置演示：lepidocrocite + Fe 箔标定。仅用于展示版式。"""
    xafs_pipeline.configure_console()
    out = ROOT / "xafs_out" / "lepido_report"
    out.mkdir(parents=True, exist_ok=True)
    stem = "Fe_lepidocrocite"
    sample = "lepidocrocite (γ-FeOOH)"
    edge_label = "Fe K-edge"
    s02 = S02
    shells = SHELLS

    print("== 数据处理：lepidocrocite（3 扫描）==")
    st = process_scans([DATA / "02_Fe_standards" / f"Fe_lepidocrocite.{e}"
                        for e in ("000", "100", "200")])
    print(f"  E0={st['e0']:.2f} eV, edge_step={st['norm'].edge_step:.4f}, "
          f"Rbkg={st['rbkg']['rbkg']:.2f}, FT 第一峰={st['r1']:.3f} Å")

    # S6.5 结构识别：指纹匹配排序 + 覆盖清单（只排序候选，不做 CN 定量）
    print("== S6.5 结构识别 ==")
    sid = xafs_fit.run(st["result"])["structure"]
    xafs_io.write_json(out / f"{stem}_structure_id.json", sid)
    print(f"  → {out / f'{stem}_structure_id.json'}")

    print("== FEFF 路径 ==")
    sites = artemis.read_cif_sites(ROOT / "lepidocrocite.cif", absorber="Fe",
                                   rmax=5.0)
    l_ss = feff_ss_paths(sites, 5.0, ROOT / "_feff_cache" / "lepido",
                         "gamma-FeOOH lepidocrocite")
    assert [round(r["reff"], 3) for r in l_ss] == \
        [1.981, 2.072, 3.062, 3.072, 3.655, 3.739, 3.873, 3.934]
    # 3 条有效壳层：Fe-O(合并 1.98×2+2.07×4，代表 2.072)、Fe-Fe(1)(合并
    # 3.06×4+3.07×2 边共享，代表 3.062)、Fe-Fe(2) 角共享(代表 3.873)；
    # 代表路径已经基准验证（见 pick_paths 注释），远 O 壳层背散射弱，省略。
    l_files, _ = pick_paths(ROOT / "_feff_cache" / "lepido", [
        ("feff0002.dat", 2.072), ("feff0003.dat", 3.062),
        ("feff0013.dat", 3.873)])

    print("== 数据处理：Fe 箔（标定，3 扫描）==")
    st_foil = process_scans([DATA / "01_Fe_foil" / f"fe.{e}"
                             for e in ("060", "150", "300")])
    f_ss = feff_ss_paths(artemis.bcc_lattice_sites(2.866, 6.0), 6.0,
                         ROOT / "_feff_cache" / "foil", "Fe foil bcc",
                         ss_max=3.5)
    assert [round(r["reff"], 3) for r in f_ss] == [2.482, 2.866]
    f_files, _ = pick_paths(ROOT / "_feff_cache" / "foil", [
        ("feff0001.dat", 2.482), ("feff0002.dat", 2.866)])

    print("== R 空间拟合 ==")
    fit_l = artemis.fit_r_space(st["chi_k"], st["ft"], l_files, 1.0, 3.9,
                                kmin=KMIN, kmax=KMAX, s02=S02, vary_n=True)
    fit_f = artemis.fit_r_space(st_foil["chi_k"], st_foil["ft"], f_files,
                                1.8, 3.1, kmin=KMIN, kmax=KMAX, s02=S02,
                                vary_n=True)
    for name, fit in (("lepidocrocite", fit_l), ("Fe foil", fit_f)):
        print(f"  {name}: ΔE0={fit.delta_e0:+.2f}±{fit.delta_e0_err:.2f} eV, "
              f"Rf={fit.r_factor:.4f}, vars={fit.n_variables} "
              f"(< N_idp={fit.n_independent:.1f})")
        for p in fit.paths:
            n, r = _fmt(p)
            print(f"    {p.label}: N={n}  R={r} Å  σ²={p.sigma2:.4f}")

    print("== 出图与导出 ==")
    rows = fit_table_rows([
        ("Fe foil", ["Fe-Fe1", "Fe-Fe2"], fit_f),
        ("lepidocrocite", ["Fe-O", "Fe-Fe(1)", "Fe-Fe(2)"], fit_l),
    ])
    plot_fit_table(rows, out, stem,
                   f"Table S1. {edge_label} EXAFS curve-fitting parameters",
                   s02=s02, kmin=KMIN, kmax=KMAX,
                   kweights=getattr(fit_l, "kweights", None))
    export_fit_table(rows, out, stem)
    xafs_plot_special.write_fit_table_markdown(rows, out, stem)
    indicator_rows = export_indicators(st, fit_l, fit_f, out, stem,
                                       sample=sample, edge_label=edge_label,
                                       s02=s02, shells=shells)
    export_panel_csvs(st, out, stem)
    demo_fit_out = {
        "mode": "motif", "status": "demo", "cn_reliable": True,
        "fit": fit_l, "flags": {},
        "cn_systematics": xafs_fit.systematic_budget(fit_l),
        "warning": ("演示模式：表中同时包含 Fe 箔标样与 lepidocrocite 两个"
                    "独立拟合，数值仅用于检查版式。"),
    }
    suite = xafs_plots.render_plot_suite(
        st["result"], demo_fit_out, out, stem,
        mode="structure", formats=("png",), dpi=300,
        sample=sample, edge_label=edge_label, shells=shells,
        indicator_rows=indicator_rows, verbose=True)
    html_path = export_html_report(
        out, stem, sample=sample, edge_label=edge_label,
        indicator_rows=indicator_rows, table_rows=rows,
        fit_out=demo_fit_out, extra_images=suite.get("extra_images"))
    print(f"  → {html_path}")

    print(f"\n输出目录 {out}:")
    for f in sorted(out.glob(f"{stem}*")):
        print(f"  {f.name}")


def main(argv=None):
    """通用报告入口：任意元素/边，S1–S7 + 结构剖析图 + 拟合表 + CSV。

    无文件参数时退化为内置演示（demo_main），方便首次运行检查环境。
    """
    xafs_pipeline.configure_console()
    parser = argparse.ArgumentParser(
        description="XAFS 报告：数据归约 + S6.5 识别 + S7 拟合 + 出图导出")
    parser.add_argument("files", nargs="*", help="原始数据文件（可多扫描）")
    parser.add_argument("--absorber", default=None,
                        help="吸收元素（如 Fe/Co/Ni/Pt）；非演示模式必填")
    parser.add_argument("--edge", default="K", choices=["K", "L1", "L2", "L3"],
                        help="吸收边标签；必须与 absorber.py 的配置一致"
                             "（Pt 默认 L3）")
    parser.add_argument("-o", "--outdir", default=None,
                        help="输出目录（默认 xafs_out/<stem>）")
    parser.add_argument("--stem", default=None, help="输出文件名前缀")
    parser.add_argument("--sample", default=None, help="图上显示的样品名")
    parser.add_argument("--kmin", type=float, default=KMIN)
    parser.add_argument("--kmax", type=float, default=KMAX)
    parser.add_argument("--kweight", type=int, default=2,
                        help="FT 与 Rbkg 的 k 权重（默认 2）")
    parser.add_argument("--rbkg", type=float, default=None,
                        help="显式指定 Rbkg (Å)，跳过自动寻优")
    parser.add_argument("--e0-method", default="derivative",
                        choices=["derivative", "half_height", "manual"],
                        help="S2 E0 判定算法（默认 derivative）")
    parser.add_argument("--manual-e0", type=float, default=None,
                        help="手动指定 E0 (eV)，配合 --e0-method manual")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="边后平台门禁（默认中断；--no-strict 降级）")
    parser.add_argument("--self-absorption-formula", default=None,
                        help="荧光自吸收校正的样品总化学式（如 Fe3O4）；"
                             "仅荧光模式，且必须同时给厚度/密度")
    parser.add_argument("--self-absorption-thickness", type=float, default=None,
                        help="样品厚度（μm）；Booth 校正必填")
    parser.add_argument("--self-absorption-density", type=float, default=None,
                        help="有效密度（g/cm³）；Booth 校正必填")
    parser.add_argument("--self-absorption-angles", default="45,45",
                        help="入射角,出射角（相对样品表面，度；默认 45,45）")
    parser.add_argument("--self-absorption-line", default=None,
                        help="荧光线（默认 K→Ka、L3→La；可显式 Kb/Lb1 等）")
    parser.add_argument("--shells", default=None,
                        help="FT/WT 标注窗口 label:lo:hi,...（默认由 S7 拟合推导）")
    parser.add_argument("--fit-motif", default=None,
                        help="强制指定 motif（跳过识别歧义）")
    parser.add_argument("--catalytic-model", default=None,
                        help="B13-1 显式催化位点模型（SAC/DAC 字符串或 JSON 字典）")
    parser.add_argument("--s02", type=float, default=None,
                        help="B18-2 运行时覆盖 S₀²（0.4–1.3），解锁未标定元素；"
                             "不修改 absorber.py 静态标定表")
    parser.add_argument("--s02-from-t5", action="store_true",
                        help="P5 从 data/s02_calibration/<El>/calibration.json "
                             "加载通过冻结门禁的独立 S₀²（显式 --s02 优先）")
    parser.add_argument("--chemical-state", default=None,
                        help="化学态 S₀² 多态：'metal' 或 'Fe=metal'；"
                             "当前仅 Fe 金属相有交叉标定 0.6566（默认生产值"
                             "仍为 0.803）；显式 --s02 优先")
    parser.add_argument("--xanes-cross-check", action="store_true",
                        help="T8 XANES–EXAFS 联合软校验：白线/边步/归一化"
                             "异常时警告；不改变 cn_reliable/正式 CN")
    parser.add_argument("--post-window", default=None,
                        help="S3 归一化边后窗口（E-E0, eV）：'150,600'；"
                             "默认 150,600")
    parser.add_argument("--allow-unusual-coordination", action="store_true",
                        help="CIF 门禁：允许键长超出 PAIR_BOND_REF±0.08 Å；"
                             "默认拒绝")
    parser.add_argument("--print-gates", action="store_true",
                        help="P6.4 报告完成后在终端打印置信门禁表"
                             "（门禁/阈值/观测/状态/动作）")
    parser.add_argument("--print-params", action="store_true",
                        help="P6.4 报告完成后在终端打印生效参数与来源"
                             "（cli/config/default/auto）")
    parser.add_argument("--phase-cif", action="append", default=None,
                        help="P1b 用户真实 CIF 路径（可重复，最高优先级）")
    parser.add_argument("--phase-id", default=None,
                        help="P1b 精确数据库编号 SOURCE:ID，如 COD:1100108")
    parser.add_argument("--phase", default=None,
                        help="P1b 相 SPEC 字符串")
    parser.add_argument("--phase-index", default=None,
                        help="P1b 本地 phase_index.json 路径")
    parser.add_argument("--phase-pack", default=None,
                        help="P1b phase pack 根目录")
    parser.add_argument("--absorber-site", default=None,
                        help="P1b 吸收位标签或 'all'")
    parser.add_argument("--offline", action="store_true",
                        help="P1b 离线模式")
    parser.add_argument("--hetero-cocupancy", action="store_true",
                        help="B13-2 异核共占位双分支精修（需含占据度 CIF motif）")
    parser.add_argument("--constraints", default=None,
                        help="B14-1 化学计量约束（'N(A)+N(B)=C' 或 JSON 列表）")
    parser.add_argument("--fit-rmin", type=float, default=None)
    parser.add_argument("--fit-rmax", type=float, default=None)
    parser.add_argument("--allow-ambiguous", action="store_true",
                        help="识别 ambiguous/partial 时仍用 Top-1 全拟合")
    parser.add_argument("--fit-weights", action="store_true",
                        help="mixture 多位点模型：放开各组分权重做相分数拟合"
                             "（输出中 N 为有效简并度 N_crystal·w）")
    parser.add_argument("--l1-ligand", default="O")
    parser.add_argument("--l1-geometry", default="oct", choices=["oct", "tet"])
    parser.add_argument("--no-fit", action="store_true",
                        help="只做 S1–S6.5，不跑 S7")
    parser.add_argument("--formal-cn", action="store_true",
                        help="P0a 正式 CN 链：无合格结构来源时写拒绝行到 "
                             "cn_table.csv；诊断拟合照常输出（需拟合）")
    parser.add_argument("--plot", default="structure",
                        choices=["none", "structure", "full", "paper"],
                        help="统一结构解析出图档位：none=仅表/JSON；"
                             "structure=结构总图+壳层鉴别图（默认）；"
                             "full=追加候选结构对比与诊断图；paper=发表级总图")
    parser.add_argument("--plot-formats", default="png",
                        help="图像格式，逗号分隔（默认 png；支持 png,pdf）")
    parser.add_argument("--plot-dpi", type=int, default=300,
                        help="图像分辨率 DPI（默认 300）")
    parser.add_argument("--phase-correct", action="store_true",
                        help="单独导出相移校正真键长对比图（注：structure/full 的 shell_dissection 已内置）")
    parser.add_argument("--chi2-contour", action="store_true",
                        help="按需导出二维 χ² 置信椭圆图并内联到 HTML")
    parser.add_argument("--operando", action="store_true",
                        help="B15-1 原位序列模式：每个文件视为一个时间点"
                             "（至少 3 个），输出演变总表与原位演化图")
    parser.add_argument("--no-html", action="store_true",
                        help="不生成单文件 HTML 报告（默认生成 <stem>_report.html）")
    parser.add_argument("--demo", action="store_true",
                        help="运行内置 lepidocrocite + Fe 箔演示")
    parser.add_argument("--config", default=None,
                        help="配置文件（.yaml/.json），CLI 参数优先级高于配置文件")
    parser.add_argument("--despike", action="store_true",
                        help="启用去毛刺（S1.5–S3 之间，中值滑窗+标准差门禁，默认关闭）")
    parser.add_argument("--deglitch", default=None,
                        help="显式能量区间去 glitch，如 '7250:7262,7410:7415'；"
                             "直接删点，不插值（默认关闭）")
    parser.add_argument("--deadtime", action="store_true",
                        help="按 dtime 列做荧光探测器死时间校正（默认关闭）")
    parser.add_argument("--reject-outlier-scans", action="store_true",
                        help="多扫描合并前按中位谱 RMS+MAD 剔除离群扫描，"
                             "至少保留 2 条（默认关闭；默认均值逐位兼容）")
    parser.add_argument("--lcf-standards", default=None,
                        help="XANES LCF 端元文件（逗号分隔），输出 <stem>_lcf.json")
    parser.add_argument("--shared-sigma2", default=None,
                        help="σ² 共享组（对标 Artemis 路径约束）：'all' 让所有路径共享一个 σ²；"
                             "或 JSON 列表如 '[[0,1],[2]]' 显式分组；默认每条路径独立 σ²")
    parser.add_argument("--fit-kweights", default=None,
                        help="多 k 权重联合拟合（对标 Artemis），如 '1,2,3'；"
                             "默认沿用单权 --kweight（2）")
    parser.add_argument("--include-ms", action="store_true",
                        help="S7 纳入近共线多散射路径（nleg=3/4），参数由母壳层"
                             "几何比例约束，不新增自由变量（默认关闭）")
    parser.add_argument("--vary-c3", default=None,
                        choices=["first", "all"],
                        help="放开三阶累积量 C3（非谐）：first=仅第一壳，"
                             "all=全部壳层；N_idp 不足时自动降级/关闭")
    parser.add_argument("--vary-c4", default=None,
                        choices=["first", "all"],
                        help="放开四阶累积量 C4（非高斯展宽）：first=仅第一壳，"
                             "all=全部壳层；N_idp 不足时自动降级/关闭")
    parser.add_argument("--isotropic-deltar", action="store_true",
                        help="各向同性晶格膨胀约束：多壳层 ΔR 合并为唯一全局 "
                             "α（ΔR_i=α·Reff_i），省 M−1 个自由变量")
    parser.add_argument("--mback", action="store_true",
                        help="理论截面 MBACK 归一化（短边后窗口）")
    parser.add_argument("--chi-std", default=None,
                        help="AUTOBK 先验标准谱文件（两列 k,chi(k)）")
    parser.add_argument("--clamp-lo", type=float, default=None,
                        help="AUTOBK 低 k 端钳制权重（0–1）")
    parser.add_argument("--clamp-hi", type=float, default=None,
                        help="AUTOBK 高 k 端钳制权重（0–1）")
    parser.add_argument("--fit-space", default=None, choices=["r", "q", "wt"],
                        help="S7 拟合空间（默认 r；q/wt 的 Rf 不可横向比较）")
    parser.add_argument("--bvs-target", type=float, default=None,
                        help="BVS 软约束目标价态（如 Fe³⁺=3.0）")
    parser.add_argument("--bvs-sigma", type=float, default=0.1,
                        help="BVS 软约束标准误差（默认 0.1）")
    parser.add_argument("--bvs-b", type=float, default=0.37,
                        help="BVS 键价参数 b（Å，默认 0.37）")
    args = parser.parse_args(argv)

    # ---- 解析 --shared-sigma2（'all' 或 JSON 列表如 '[[0,1],[2]]'）----
    shared_sigma2 = None
    if args.shared_sigma2 is not None:
        s = args.shared_sigma2.strip()
        if s == "all":
            shared_sigma2 = "all"
        else:
            try:
                shared_sigma2 = json.loads(s)
                if not isinstance(shared_sigma2, list):
                    parser.error("--shared-sigma2 必须是 'all' 或 JSON 路径下标列表")
            except json.JSONDecodeError:
                parser.error("--shared-sigma2 格式无效：用 'all' 或 JSON 列表如 '[[0,1],[2]]'")
    try:
        fit_kweights = xafs_config.parse_kweights(args.fit_kweights)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        deglitch = xafs_config.parse_deglitch(args.deglitch)
    except ValueError as exc:
        parser.error(str(exc))
    if isinstance(args.lcf_standards, str) and args.lcf_standards.strip() \
            and not [s.strip() for s in args.lcf_standards.split(",")
                     if s.strip()]:
        parser.error("--lcf-standards 至少需要一个端元文件")

    # ---- 配置文件加载与 CLI 合并 ----（与 run_xafs 同一逻辑）
    config = xafs_config.load_config(args.config)
    _cli_defaults = {
        "kmin": KMIN, "kmax": KMAX, "kweight": 2,
        "rbkg": None, "strict": True,
        "self_absorption_formula": None,
        "self_absorption_thickness": None,
        "self_absorption_density": None,
        "self_absorption_angles": "45,45",
        "self_absorption_line": None,
        "plot": "structure",
        "plot_formats": "png",
        "plot_dpi": 300,
    }
    # CLI 显式给出的 dest（即便其值恰等于代码默认值）优先于配置文件，
    # 与 run_xafs 同一口径：不能用 "值==默认" 反推是否显式给定。
    _cli_explicit = xafs_provenance._cli_dests(
        parser, list(argv) if argv is not None else sys.argv[1:])
    for _key, _default in _cli_defaults.items():
        if _key in _cli_explicit:
            continue
        if getattr(args, _key, None) == _default:
            if _key.startswith("self_absorption_"):
                _sub = _key[len("self_absorption_"):]
                _cfg_val = config.get("self_absorption", {}).get(_sub)
                if _key == "self_absorption_angles" and _cfg_val:
                    _cfg_val = ",".join(str(a) for a in _cfg_val)
            else:
                _cfg_val = config.get(_key)
            if _cfg_val is not None:
                setattr(args, _key, _cfg_val)
    if not args.despike and config.get("despike"):
        args.despike = True
    if not args.formal_cn and config.get("formal_cn"):
        args.formal_cn = True
    if args.catalytic_model is None and config.get("catalytic_model") is not None:
        args.catalytic_model = config["catalytic_model"]
    if args.s02 is None and config.get("s02") is not None:
        args.s02 = float(config["s02"])
        # 来源为配置文件：解析链中降到显式化学态/T5 之后，溯源标 config
        args._s02_is_config = True
    if args.constraints is None and config.get("constraints") is not None:
        args.constraints = config["constraints"]

    # ---- Phase I–III 开关：CLI 显式 > 配置文件 > 默认 ----
    for _key in ("deadtime", "reject_outlier_scans", "include_ms",
                 "isotropic_deltar", "mback", "hetero_cocupancy",
                 "phase_correct", "chi2_contour", "operando"):
        if not getattr(args, _key, False) and config.get(_key):
            setattr(args, _key, True)
    if args.deglitch is None and config.get("deglitch") is not None:
        args.deglitch = config["deglitch"]
    if args.fit_kweights is None and config.get("fit_kweights") is not None:
        args.fit_kweights = config["fit_kweights"]
    if args.shared_sigma2 is None and config.get("shared_sigma2") is not None:
        args.shared_sigma2 = config["shared_sigma2"]
    if args.lcf_standards is None and config.get("lcf_standards") is not None:
        args.lcf_standards = config["lcf_standards"]
    if args.vary_c3 is None and config.get("vary_c3") is not None:
        args.vary_c3 = config["vary_c3"]
    if args.vary_c4 is None and config.get("vary_c4") is not None:
        args.vary_c4 = config["vary_c4"]
    # P1b/P5（与 run_xafs 共用同一套键）
    xafs_config.apply_p1b_config(args, config)
    post_window = (150.0, 600.0)
    if args.post_window is not None:
        try:
            _pw = xafs_config._coerce_config_value(
                "post_window", args.post_window)
        except (TypeError, ValueError):
            parser.error(f"--post-window 需要两个数字，收到 {args.post_window!r}")
        if len(_pw) != 2 or not (0 < _pw[0] < _pw[1]):
            parser.error(
                f"--post-window 必须满足 0 < lo < hi，收到 {args.post_window!r}")
        post_window = (float(_pw[0]), float(_pw[1]))
    try:
        fit_kweights = (xafs_config.parse_kweights(args.fit_kweights)
                        if not isinstance(args.fit_kweights, (list, tuple))
                        else tuple(args.fit_kweights))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        deglitch = (xafs_config.parse_deglitch(args.deglitch)
                    if not isinstance(args.deglitch, (list, tuple))
                    else xafs_config._coerce_config_value("deglitch", args.deglitch))
    except ValueError as exc:
        parser.error(str(exc))
    if isinstance(args.lcf_standards, str):
        lcf_standards = [s.strip() for s in args.lcf_standards.split(",")
                         if s.strip()] or None
    else:
        lcf_standards = args.lcf_standards
    if isinstance(args.shared_sigma2, str) and args.shared_sigma2.strip() == "all":
        shared_sigma2 = "all"
    elif isinstance(args.shared_sigma2, str) and args.shared_sigma2:
        try:
            shared_sigma2 = json.loads(args.shared_sigma2)
        except json.JSONDecodeError:
            parser.error("--shared-sigma2 格式无效：用 'all' 或 JSON 列表如 '[[0,1],[2]]'")
    else:
        shared_sigma2 = args.shared_sigma2
    vary_c3 = xafs_config.normalize_cumulant_switch(args.vary_c3)
    vary_c4 = xafs_config.normalize_cumulant_switch(args.vary_c4)
    if args.fit_space is None:
        args.fit_space = str(config.get("fit_space") or "r").strip().lower()
    if args.fit_space not in ("r", "q", "wt"):
        parser.error(f"--fit-space 只支持 r/q/wt，收到 {args.fit_space!r}")
    if args.chi_std is None and config.get("chi_std") is not None:
        args.chi_std = config["chi_std"]
    if args.clamp_lo is None and config.get("clamp_lo") is not None:
        args.clamp_lo = config["clamp_lo"]
    if args.clamp_hi is None and config.get("clamp_hi") is not None:
        args.clamp_hi = config["clamp_hi"]
    chi_std = chi_std_k = None
    if args.chi_std:
        std_path = Path(args.chi_std)
        if not std_path.is_file():
            parser.error(f"--chi-std 文件不存在: {std_path}")
        try:
            std_arr = np.loadtxt(std_path, comments=("#", ";"), ndmin=2)
        except Exception as exc:  # noqa: BLE001
            parser.error(f"--chi-std 读取失败: {exc}")
        if std_arr.shape[1] < 2:
            parser.error("--chi-std 需要至少两列：k(Å⁻¹), chi(k)")
        chi_std_k = np.asarray(std_arr[:, 0], dtype=float)
        chi_std = np.asarray(std_arr[:, 1], dtype=float)
        order = np.argsort(chi_std_k)
        chi_std_k, chi_std = chi_std_k[order], chi_std[order]
    for _name in ("clamp_lo", "clamp_hi"):
        _val = getattr(args, _name)
        if _val is not None and not (0.0 <= float(_val) <= 1.0):
            parser.error(f"--{_name.replace('_', '-')} 必须在 0–1 之间")
    bvs_restraint = config.get("bvs_restraint")
    if args.bvs_target is not None:
        if not np.isfinite(args.bvs_target):
            parser.error("--bvs-target 必须是有限数")
        if not np.isfinite(args.bvs_sigma) or args.bvs_sigma <= 0:
            parser.error("--bvs-sigma 必须 > 0")
        if not np.isfinite(args.bvs_b) or args.bvs_b <= 0:
            parser.error("--bvs-b 必须 > 0")
        bvs_restraint = {"target": float(args.bvs_target),
                         "sigma": float(args.bvs_sigma),
                         "b": float(args.bvs_b),
                         "absorber": args.absorber or "Fe"}
    if bvs_restraint is not None and not isinstance(bvs_restraint, dict):
        parser.error("配置里的 bvs_restraint 必须是字典")

    if args.fit_weights and args.no_fit:
        parser.error("--fit-weights 需要运行 S7，不能与 --no-fit 同用")
    if args.formal_cn and args.no_fit:
        parser.error("--formal-cn 需要运行 S7，不能与 --no-fit 同用")
    _phase_given = [_lbl for _val, _lbl in (
        (args.phase_cif, "--phase-cif"), (args.phase_id, "--phase-id"),
        (args.phase, "--phase"), (args.phase_index, "--phase-index"))
        if _val is not None]
    if _phase_given and not args.formal_cn:
        parser.error("以下参数只用于 --formal-cn 的结构来源，未启用正式 CN 时"
                     "会被静默忽略；请同时给 --formal-cn: "
                     + ", ".join(_phase_given))
    if args.s02 is not None and not (0.4 <= args.s02 <= 1.3):
        parser.error("--s02 必须在 [0.4, 1.3] 内")
    if args.no_fit:
        for _flag, _label in (("catalytic_model", "--catalytic-model"),
                              ("hetero_cocupancy", "--hetero-cocupancy"),
                              ("constraints", "--constraints"),
                              ("chi2_contour", "--chi2-contour")):
            if getattr(args, _flag, None):
                parser.error(f"{_label} 需要运行 S7（不能与 --no-fit 同用）")
    constraints = args.constraints
    if isinstance(constraints, str):
        _c = constraints.strip()
        if _c.startswith("[") or _c.startswith("{"):
            try:
                constraints = json.loads(_c)
            except json.JSONDecodeError as exc:
                parser.error(f"--constraints JSON 解析失败: {exc}")
        else:
            constraints = _c
    try:
        sa_angles = tuple(float(x) for x in
                          args.self_absorption_angles.replace(" ", "").split(","))
    except ValueError:
        parser.error("--self-absorption-angles 格式为 入射,出射（度），如 45,45")
    if len(sa_angles) != 2:
        parser.error("--self-absorption-angles 需要两个角度：入射,出射")
    if args.self_absorption_formula:
        if (args.self_absorption_thickness is None
                or args.self_absorption_density is None):
            parser.error("Booth 自吸收校正需要 --self-absorption-thickness "
                         "与 --self-absorption-density")
    elif (args.self_absorption_thickness is not None
          or args.self_absorption_density is not None
          or args.self_absorption_line is not None):
        parser.error("厚度/密度/荧光线只在给 --self-absorption-formula 时使用")
    if args.demo or not args.files:
        return demo_main()
    if not args.absorber:
        parser.error("非演示模式需要 --absorber（S6.5 要选 motif 库与 S₀²）")
    abs_cfg = get_absorber(args.absorber)
    if args.edge != abs_cfg.edge:
        parser.error(
            f"--edge {args.edge} 与 absorber.py 中 {abs_cfg.symbol} 的配置 "
            f"{abs_cfg.edge} 不一致；改边请先在 absorber.py 配好再跑，"
            f"避免只改标签不改 FEFF HOLE/边能")

    # P6.4：参数台账快照（只记录，不参与判定）。
    _argv = list(argv) if argv is not None else sys.argv[1:]
    prov = xafs_provenance.Provenance.from_args(
        args, config, _argv, parser=parser, inputs=args.files,
        overrides={
            "post_window": list(post_window),
            "vary_c3": xafs_provenance._jsonable(vary_c3),
            "vary_c4": xafs_provenance._jsonable(vary_c4),
            "fit_kweights": xafs_provenance._jsonable(fit_kweights),
            "deglitch": xafs_provenance._jsonable(deglitch),
            "lcf_standards": xafs_provenance._jsonable(lcf_standards),
            "shared_sigma2": xafs_provenance._jsonable(shared_sigma2),
            "constraints": xafs_provenance._jsonable(constraints),
            "self_absorption_angles": list(sa_angles),
            "bvs_restraint": xafs_provenance._jsonable(bvs_restraint),
        },
        source_overrides={
            "bvs_restraint": ("config" if config.get("bvs_restraint")
                              else "cli" if args.bvs_target is not None
                              else "default"),
        })

    # ---- B15-1 原位序列汇总报告：每个文件=一个时间点，不合并 ----
    if args.operando:
        if args.no_fit:
            parser.error("--operando 需要逐样品跑 S1–S6.5，不能与 --no-fit 同用")
        _incompatible = [name for name, on in (
            ("--formal-cn", args.formal_cn),
            ("--xanes-cross-check", args.xanes_cross_check),
            ("--fit-weights", args.fit_weights),
            ("--phase-cif", args.phase_cif is not None),
            ("--phase-id", args.phase_id is not None),
            ("--phase", args.phase is not None),
            ("--phase-index", args.phase_index is not None),
        ) if on]
        if _incompatible:
            parser.error(
                "--operando 只执行 S1–S6 + PCA/MCR/LCF，不执行 S6.5/S7；"
                "以下参数不适用: " + ", ".join(_incompatible))
        if len(args.files) < 3:
            parser.error("--operando 至少需要 3 个样品文件")
        op_root = (Path(args.outdir) if args.outdir
                   else ROOT / "xafs_out" / "operando_series")
        op_root.mkdir(parents=True, exist_ok=True)
        results = []
        for i, fpath in enumerate(args.files, 1):
            print(f"[operando {i}/{len(args.files)}] {Path(fpath).name}")
            st = process_scans(
                [fpath], absorber=args.absorber, strict=args.strict,
                kmin=args.kmin, kmax=args.kmax,
                kweight=args.kweight, e0_method=args.e0_method,
                manual_e0=args.manual_e0, rbkg=args.rbkg,
                post_window=post_window,
                despike=args.despike,
                deglitch=deglitch, deadtime=args.deadtime,
                reject_outlier_scans=args.reject_outlier_scans,
                mback=args.mback, chi_std=chi_std, chi_std_k=chi_std_k,
                clamp_lo=args.clamp_lo, clamp_hi=args.clamp_hi,
                self_absorption_formula=args.self_absorption_formula,
                self_absorption_thickness=args.self_absorption_thickness,
                self_absorption_density=args.self_absorption_density,
                self_absorption_angles=sa_angles,
                self_absorption_line=args.self_absorption_line)
            results.append(st["result"])
        std_scans = None
        if lcf_standards:
            std_scans = {}
            for p in lcf_standards:
                std = xafs_pipeline.process_sample(
                    [p], absorber=args.absorber, strict=False,
                    e0_method=args.e0_method, post_window=post_window,
                    mback=args.mback, despike=args.despike,
                    deadtime=args.deadtime, verbose=False)
                std_scans[Path(p).stem] = std.norm
        op = xafs_lcf.run_operando_series(
            results, standards=std_scans,
            **xafs_config.operando_kwargs_from_config(config))
        made = xafs_plot_special.write_operando_diagnostics(
            op, op_root / "operando", stem="operando")
        op_meta = xafs_provenance.build_operando_meta(
            prov, op, overrides={"operando": True})
        xafs_io.write_json(op_root / "operando_meta.json", op_meta)
        print(f"  → {op_root / 'operando_meta.json'}")
        if args.print_params:
            print(xafs_provenance.format_params(op_meta["params"]))
        if args.print_gates:
            print(xafs_provenance.format_gates(op_meta["gates"]))
        indicator_rows = [
            ("n_spectra", len(args.files), "条", "原位时间序列"),
            ("n_components", op["n_components"], "个",
             "PCA 99% 方差分量，至少 2"),
            ("accepted_standards", ", ".join(op["accepted_standards"]) or "—",
             "", "SPOIL 过滤后保留"),
            ("rejected_standards", ", ".join(op["rejected_standards"]) or "—",
             "", "SPOIL > 2×poor 的伪标样"),
            ("mcr_r_factor", f"{op['mcr_als']['r_factor']:.4g}", "",
             "MCR-ALS 盲解谱重构 Rf"),
        ]
        html_path = export_html_report(
            op_root, "operando", sample="Operando series",
            edge_label=f"{args.absorber} {args.edge}",
            indicator_rows=indicator_rows, table_rows=[], fit_out=None,
            extra_images=[(
                made["png"],
                "原位演化四宫格：MCR 组分分数 / LCF 标样权重 / "
                "E0–白线描述符 / 按特征能量排序的纯组分谱")])
        print(f"原位序列：{len(args.files)} 条谱，"
              f"n_components={op['n_components']}")
        for key, path in made.items():
            print(f"  → {path}")
        print(f"  → {html_path}")
        return

    stem = args.stem or Path(args.files[0]).stem
    sample = args.sample or stem
    out = (Path(args.outdir) if args.outdir
           else ROOT / "xafs_out" / stem)
    out.mkdir(parents=True, exist_ok=True)
    _artifact_baseline = xafs_provenance.snapshot_artifacts(out)

    print(f"== 数据处理：{sample}（{len(args.files)} 个文件）==")
    st = process_scans(args.files, absorber=args.absorber,
                       strict=args.strict, kmin=args.kmin, kmax=args.kmax,
                       kweight=args.kweight, e0_method=args.e0_method,
                       manual_e0=args.manual_e0, rbkg=args.rbkg,
                       post_window=post_window,
                       despike=args.despike, deglitch=deglitch,
                       deadtime=args.deadtime,
                       reject_outlier_scans=args.reject_outlier_scans,
                       mback=args.mback, chi_std=chi_std,
                       chi_std_k=chi_std_k,
                       clamp_lo=args.clamp_lo, clamp_hi=args.clamp_hi,
                       self_absorption_formula=args.self_absorption_formula,
                       self_absorption_thickness=args.self_absorption_thickness,
                       self_absorption_density=args.self_absorption_density,
                       self_absorption_angles=sa_angles,
                       self_absorption_line=args.self_absorption_line)
    print(f"  E0={st['e0']:.2f} eV, edge_step={st['norm'].edge_step:.4f}, "
          f"Rbkg={st['rbkg']['rbkg']:.2f}, FT 第一峰={st['r1']:.3f} Å")

    if lcf_standards:
        std_norms = {}
        for std_path in lcf_standards:
            std_res = xafs_pipeline.process_sample(
                [std_path], absorber=args.absorber, strict=False,
                e0_method=args.e0_method, post_window=post_window,
                mback=args.mback, despike=args.despike,
                deadtime=args.deadtime, verbose=False)
            std_norms[Path(std_path).stem] = std_res.norm
        lcf_res = xafs_lcf.lcf_from_scans(st["norm"], std_norms,
                                          mode="xanes")
        xafs_io.write_json(out / f"{stem}_lcf.json",
                           xafs_lcf.lcf_summary(lcf_res))
        w = ", ".join(f"{k}={v:.3f}" for k, v in lcf_res["weights"].items())
        print(f"  [B2-2] XANES LCF: {w}   Rf={lcf_res['r_factor']:.2e}"
              f"  → {out / f'{stem}_lcf.json'}")

    eff_absorber = args.absorber or getattr(st["result"], "absorber_symbol", None)
    if not eff_absorber and args.files:
        import run_xafs
        eff_absorber, _ = run_xafs.detect_absorber(args.files[0])
    _s02_is_config = bool(getattr(args, "_s02_is_config", False))
    # CLI 显式 --s02 最高优先；config 来源降到化学态/T5 之后并标 config:s02
    s02_val = args.s02 if not _s02_is_config else None
    s02_source = "--s02 runtime override" if s02_val is not None else None
    if s02_val is None and args.chemical_state:
        if not eff_absorber:
            raise XAFSConfigError(
                "FIT", "启用 --chemical-state 但无法判定吸收元素；"
                       "请显式给 --absorber")
        s02_val, s02_source = resolve_s02_spec(eff_absorber,
                                               args.chemical_state)
    if s02_val is None and args.s02_from_t5:
        import run_xafs
        if not eff_absorber:
            raise XAFSConfigError(
                "T5", "启用 --s02-from-t5 但无法判定吸收元素；请显式给 --absorber")
        s02_val, s02_source = run_xafs.load_t5_s02(eff_absorber)
    if s02_val is None and _s02_is_config and args.s02 is not None:
        s02_val = args.s02
        s02_source = "config:s02"

    structure_model = None
    structure_models: list = []
    gate_result = None
    gate_results: list = []
    if args.formal_cn and (args.phase_cif or args.phase_id
                           or args.phase or args.phase_index):
        if not eff_absorber:
            raise XAFSConfigError(
                "FIT", "--formal-cn + --phase-* requires --absorber to specify absorber element")
        import cif_gate
        import phase_resolver
        resolution = phase_resolver.resolve_phase(
            eff_absorber,
            phase_cif=args.phase_cif,
            phase_id=args.phase_id,
            phase_spec=args.phase,
            phase_index_path=args.phase_index,
            phase_pack_path=args.phase_pack,
            absorber_site=args.absorber_site,
            offline=args.offline)
        if resolution.status != "resolved":
            raise XAFSConfigError(
                "FIT", f"--phase-* 解析失败 ({resolution.status}): "
                       f"{resolution.reason}")
        structure_model = resolution.structure_model
        if structure_model is None:
            raise XAFSConfigError(
                "FIT", f"Phase resolution did not produce a usable structure model: "
                       f"{resolution.reason}")
        structure_models = list(resolution.site_models)
        multi_site = len(structure_models) > 1
        gate_kwargs = dict(
            kmin=args.kmin, kmax=args.kmax, rmax=6.0,
            allow_computed=False,
            allow_unusual_coordination=args.allow_unusual_coordination,
            is_computed=not resolution.selected.experimental)
        if multi_site:
            gate_results = cif_gate.gate_cif_all_sites(
                resolution.selected.cif_path, eff_absorber, **gate_kwargs)
            failed = [g for g in gate_results if not g.ok]
            if failed:
                raise XAFSConfigError(
                    "FIT", f"CIF gate rejected multi-site: {failed[0].reason}")
            gate_result = gate_results[0]
            for m, g in zip(structure_models, gate_results):
                m.occupancy_policy = g.occupancy_source
                m.n0_shells = list(g.n0_shells)
                m.warnings = list(g.warnings)
        else:
            gate_result = cif_gate.gate_cif(
                resolution.selected.cif_path, eff_absorber,
                absorber_site=resolution.absorber_site, **gate_kwargs)
            if not gate_result.ok:
                raise XAFSConfigError(
                    "FIT", f"CIF gate rejected: {gate_result.code} — "
                           f"{gate_result.reason}")
            if structure_model is not None:
                structure_model.occupancy_policy = gate_result.occupancy_source
                structure_model.n0_shells = list(gate_result.n0_shells)
                structure_model.warnings = list(gate_result.warnings)

    multi_site_formal = bool(args.formal_cn and len(structure_models) > 1)
    print("== S6.5 结构识别 + S7 拟合 ==")
    out_fit = xafs_fit.run(
        st["result"], identify=True, fit=not args.no_fit,
        absorber_symbol=eff_absorber, fit_motif=args.fit_motif,
        fit_rmin=args.fit_rmin, fit_rmax=args.fit_rmax,
        allow_ambiguous=args.allow_ambiguous,
        l1_ligand=args.l1_ligand, l1_geometry=args.l1_geometry,
        fit_weights=args.fit_weights,
        shared_sigma2=shared_sigma2, kweights=fit_kweights,
        include_ms=args.include_ms, vary_c3=vary_c3, vary_c4=vary_c4,
        isotropic_deltar=args.isotropic_deltar,
        fit_space=args.fit_space, bvs_restraint=bvs_restraint,
        catalytic_model=args.catalytic_model,
        hetero_cocupancy=args.hetero_cocupancy,
        constraints=constraints,
        s02=s02_val,
        s02_source=s02_source,
        s02_evidence=getattr(s02_source, "s02_evidence", None),
        formal_cn=args.formal_cn and not multi_site_formal,
        structure_model=structure_model,
        gate_result=gate_result,
        xanes_cross_check=args.xanes_cross_check,
        verbose=True)
    if multi_site_formal and out_fit.get("fit") is not None:
        # 【红线 2 / 异常如实降级】xafs_fit.run 的返回契约是
        # {"fit": fit_out, ...}，主拟合的 cn_reliable 位于 fit_out 内；
        # 不能写成 out_fit.get("cn_reliable")（该键不在外层，会恒为 False）。
        _fit_out = out_fit.get("fit") or {}
        base_cn_reliable = _fit_out.get("cn_reliable")
        if base_cn_reliable is False:
            base_reject = xafs_fit.MULTI_SITE_BASE_UNRELIABLE
        elif base_cn_reliable is None:
            base_reject = ("主拟合未提供 cn_reliable 状态，按保守原则"
                           "拒绝多位点 CN 宣称")
        elif (_fit_out.get("cn_claim_allowed") is False
              or bool(_fit_out.get("degraded"))):
            base_reject = ("主拟合已标记 degraded / "
                           "cn_claim_allowed=False，拒绝多位点 CN 宣称")
        else:
            base_reject = None
        base_unreliable = base_reject is not None
        multi = xafs_fit.run_multi_site_formal_cn(
            st["result"],
            site_models=structure_models,
            gate_results=gate_results,
            s02=s02_val,
            s02_source=s02_source,
            s02_evidence=getattr(s02_source, "s02_evidence", None),
            data_sha256=(getattr(st["result"], "data_sha256", "") or ""),
            # 透传给每个位点 build_formal_cn_block 的硬前置：主拟合
            # 不可靠时，任何位点都不得产出正式 CN。
            site_cn_reliability=(
                [False] * len(structure_models)
                if base_unreliable else None),
            verbose=True)
        # 调用方再显式兜底一次（幂等）：即使底层实现被 mock/替换，
        # 汇总行也绝不会在基础拟合不可靠时保留"通过"的假象。
        if base_unreliable:
            xafs_fit.downgrade_multi_site_cn(multi, reason=base_reject)
        out_fit["fit"]["formal_cn"] = multi
        out_fit["fit"]["cn_status"] = multi.get("status", "multi_site_failed")
    sid = out_fit.get("structure")
    fout = out_fit.get("fit")
    xafs_io.write_json(out / f"{stem}_structure_id.json", sid)
    xafs_io.write_json(out / f"{stem}_fit.json", fit_summary(fout))
    if fout and fout.get("fit") is not None:
        rdf_tab = xafs_fit.rdf_table(fout["fit"])
        if rdf_tab["rows"]:
            xafs_fit.write_rdf_csv(out / f"{stem}_rdf.csv", rdf_tab)
            print(f"  → {out / f'{stem}_rdf.csv'} "
                  f"（{len(rdf_tab['rows'])} 条非高斯 g(R)）")
    print(f"  → {out / f'{stem}_structure_id.json'}")
    print(f"  → {out / f'{stem}_fit.json'}")

    s02 = (fout or {}).get(
        "s02", s02_val if s02_val is not None else abs_cfg.s02)
    formal_block = None
    if args.formal_cn:
        formal = (fout or {}).get("formal_cn") or {}
        formal_block = formal
        rows = list(formal.get("rows") or [])
        model = formal.get("model") or {}
        if not rows:
            reason = formal.get("reject_reason") or "未生成正式 CN"
            rows = [cn_table.refusal_row(
                formal.get("status", "no_cif"), reason,
                phase_id=model.get("model_id", ""),
                model_level=model.get("model_level", ""),
                source_type=model.get("source_type", ""),
                source_ref=model.get("source_ref", ""),
                source_cif_sha256=model.get("source_cif_sha256", ""),
                absorber_site=model.get("absorber_site", ""),
                s02=s02 if s02 is not None else "",
                s02_source=(fout or {}).get("s02_source", s02_source),
                s02_calibrated=(fout or {}).get("s02_calibrated", False),
                kmin=args.kmin, kmax=args.kmax,
                kweight=int(st["result"].ft.kweight),
                rmin=(fout or {}).get("rmin", ""),
                rmax=(fout or {}).get("rmax", ""),
            )]
        cn_table.write_cn_table(out / "cn_table.csv", rows)
        print(f"  → {out / 'cn_table.csv'} "
              f"（formal CN status={formal.get('status', 'no_cif')}）")

    # P6.4：报告入口也写同一套参数/门禁/决策/摘要台账。
    resolved_model = structure_model
    overrides = {
        "absorber": eff_absorber,
        "e0": st["e0"],
        "rbkg": st["rbkg"]["rbkg"],
        "rbkg_confidence": st["rbkg"].get("confidence"),
        "rbkg_manual": bool(getattr(args, "rbkg", None) is not None),
        "post_window": list(post_window),
        "s02": (fout or {}).get("s02", s02_val),
        "s02_source": (fout or {}).get("s02_source", s02_source),
        "formal_cn": bool(args.formal_cn),
        "identify": True,
        "fit": not bool(args.no_fit),
        "resolved_phase_id": (resolved_model.model_id
                              if resolved_model is not None else None),
        "resolved_phase_source": (resolved_model.source_type
                                  if resolved_model is not None else None),
        "resolved_absorber_site": (resolved_model.absorber_site
                                   if resolved_model is not None else None),
        "model_level": (resolved_model.model_level
                        if resolved_model is not None else None),
        "fit_mode": (fout or {}).get("mode"),
        "cn_status": (fout or {}).get("cn_status"),
        "cn_reliable": (fout or {}).get("cn_reliable"),
    }
    source_overrides = {
        "e0": "cli" if args.manual_e0 is not None else "auto",
        "rbkg": ("cli" if getattr(args, "rbkg", None) is not None
                 else "auto"),
    }
    if s02_source:
        source_absorber = (get_absorber(eff_absorber) if s02_val is None else
                           structure_id.absorber_with_s02(eff_absorber, s02_val))
        source_overrides["s02"] = (
            "calibrated" if xafs_fit._s02_calibrated_flag(
                source_absorber, s02_source,
                getattr(s02_source, "s02_evidence", None)) else "runtime_override")
    eff_kmax = float(st["result"].ft.kmax)
    decision_context = {
        "n_scans": len(st["scans"]) if st.get("scans") else None,
        "absorber": eff_absorber,
        "e0_method": args.e0_method,
        "e0": st["e0"],
        "plateau": st["norm"].plateau,
        "strict": bool(args.strict),
        "post_window": list(post_window),
        "rbkg": st["rbkg"]["rbkg"],
        "rbkg_confidence": st["rbkg"].get("confidence"),
        "rbkg_manual": bool(getattr(args, "rbkg", None) is not None),
        "kmin": args.kmin, "kmax": eff_kmax,
        "kweight": int(st["result"].ft.kweight),
        "r_first_peak": st["r1"],
        "wt_sigma": st["wt"].sigma,
        "r_wt_peak": getattr(st["result"], "r_wt", None),
        "k_wt_peak": getattr(st["result"], "k_wt", None),
        "structure_status": (sid or {}).get("status"),
        "structure_top1": (((sid or {}).get("ranked") or [{}])[0].get("id")),
        "structure_margin": (((sid or {}).get("confidence")
                              or {}).get("margin")),
        "structure_coverage_ok": (sid or {}).get("coverage_ok"),
        "phase_id": (resolved_model.model_id
                     if resolved_model is not None else args.phase_id),
        "phase_source": (resolved_model.source_type
                         if resolved_model is not None else None),
        "model_level": (resolved_model.model_level
                        if resolved_model is not None else None),
        "absorber_site": (resolved_model.absorber_site
                          if resolved_model is not None else args.absorber_site),
        "catalytic_model": args.catalytic_model,
        "fit_mode": (fout or {}).get("mode"),
        "fit_motif": args.fit_motif,
        "fit_rmin": (fout or {}).get("rmin", args.fit_rmin),
        "fit_rmax": (fout or {}).get("rmax", args.fit_rmax),
        "s02": (fout or {}).get("s02", s02_val),
        "s02_source": (fout or {}).get("s02_source", s02_source),
        "cn_reliable": (fout or {}).get("cn_reliable"),
        "cn_status": (fout or {}).get("cn_status"),
        "fit_flags": (fout or {}).get("flags"),
        "formal_status": (formal_block or {}).get("status"),
        "formal_reject_reason": (formal_block or {}).get("reject_reason"),
        "n_ratio": (((fout or {}).get("conditional_cn")
                     or {}).get("n_ratio")),
        "xanes_cross_check_warnings": (
            len((getattr(st["result"], "xanes_cross_check", None)
                 or {}).get("warnings") or [])
            if getattr(st["result"], "xanes_cross_check", None) else None),
    }
    merged_meta = dict(getattr(st["merged"], "meta", None) or {})
    l1_block = (fout or {}).get("l1") or {}
    fit_obj_meta = (fout or {}).get("fit")
    report_meta = {
        "source": {
            **merged_meta,
            "files": [str(f) for f in args.files],
            "stem": stem,
            "mode": "report",
            "report_mode": merged_meta.get("mode"),
            "data_sha256": getattr(st["result"], "data_sha256", ""),
        },
        "absorber": eff_absorber,
        "edge": abs_cfg.edge,
        "e0": st["e0"],
        "normalization": {
            "edge_step": st["norm"].edge_step,
            "plateau": st["norm"].plateau,
            "plateau_ok": bool(0.85 <= st["norm"].plateau <= 1.15),
            "strict": bool(args.strict),
        },
        "rbkg": {
            "selected": st["rbkg"]["rbkg"],
            "confidence": st["rbkg"].get("confidence"),
            "table": getattr(st["result"], "rbkg_table", []) or [],
        },
        "ft": {
            "kmin": args.kmin, "kmax": st["result"].ft.kmax,
            "kweight": int(st["result"].ft.kweight),
            "r_first_peak": st["r1"],
        },
        "wt": {
            "sigma": st["wt"].sigma,
            "r_peak": getattr(st["result"], "r_wt", None),
            "k_peak": getattr(st["result"], "k_wt", None),
        },
        "self_absorption": ((fout or {}).get("self_absorption")
                            or st["result"].self_absorption),
        "structure_id": sid,
        "fit": {
            "mode": (fout or {}).get("mode"),
            "status": (fout or {}).get("status"),
            "cn_reliable": (fout or {}).get("cn_reliable"),
            "cn_status": (fout or {}).get("cn_status"),
            # L1 第一壳降级时 fit=None，真实 Rf/ΔE0 在 l1 块里，不能写 None。
            "r_factor": (fit_obj_meta.r_factor
                         if fit_obj_meta is not None
                         else l1_block.get("rf")),
            "delta_e0": (fit_obj_meta.delta_e0
                         if fit_obj_meta is not None
                         else l1_block.get("delta_e0")),
        },
        "formal_cn": formal_block,
    }
    # 【P1.1/#16】只在真的裁剪时写入，未裁剪样品台账逐位一致（红线5）。
    if st["result"].kmax_clamp is not None:
        report_meta["ft"]["kmax_clamp"] = st["result"].kmax_clamp
    report_meta.update(xafs_provenance.build_output_meta(
        prov,
        norm=st["norm"], strict=args.strict,
        self_absorption=((fout or {}).get("self_absorption")
                         or st["result"].self_absorption),
        structure=sid,
        gate_result=gate_result, gate_results=gate_results,
        fit_out=fout, formal_block=formal_block,
        xanes_cross_check=getattr(st["result"], "xanes_cross_check", None),
        overrides=overrides, source_overrides=source_overrides,
        decision_context=decision_context))
    # 台账先落盘：下游绘图与报表需读取 meta.json，用户旧记录在此按防覆盖
    # 机制备份。run_manifest 待全部产物（含图件/HTML）落盘后在末尾补全。
    xafs_io.write_json(out / f"{stem}_meta.json", report_meta)
    print(f"  → {out / f'{stem}_meta.json'} "
          f"（params/gates/decisions/summary）")
    if args.print_params:
        print(xafs_provenance.format_params(report_meta["params"]))
    if args.print_gates:
        print(xafs_provenance.format_gates(report_meta["gates"]))
    edge_label = f"{args.absorber} {abs_cfg.edge}-edge"
    explicit_shells = parse_shell_spec(args.shells)
    shells = explicit_shells or shells_from_fit(fout, absorber_symbol=args.absorber)
    if not shells:
        shells = [("first shell", 1.0, min(2.6, st["r1"] + 0.5))]

    print("== 出图与导出 ==")
    rows = fit_rows_from_output(sample, fout)
    weights_mode = bool(fout and fout.get("mode") == "motif_weights")
    if rows:
        title = f"Table S1. {sample}, {edge_label} EXAFS fit"
        if weights_mode:
            title += " — N is effective degeneracy $N_{cryst}w$"
        elif fout and not fout.get("cn_reliable", True):
            title += " — CN unreliable (see indicators)"
        _fit_obj = xafs_plot_special._fit_result(fout)
        plot_fit_table(
            rows, out, stem, title,
            s02=s02 if s02 is not None else float("nan"),
            kmin=args.kmin, kmax=eff_kmax,
            weights_mode=weights_mode,
            kweights=getattr(_fit_obj, "kweights", None))
        export_fit_table(rows, out, stem)
        xafs_plot_special.write_fit_table_markdown(
            rows, out, stem, weights_mode=weights_mode)
    else:
        print("  （未运行 S7 或无拟合结果，跳过拟合表）")
    indicator_rows = export_indicators_single(
        st, fout, out, stem, sample=sample,
        edge_label=edge_label, s02=s02,
        shells=shells, kmin=args.kmin, kmax=eff_kmax)
    export_panel_csvs(st, out, stem)

    plot_formats = tuple(x.strip() for x in str(args.plot_formats).split(",")
                         if x.strip()) or ("png",)
    extra_images = []
    if str(args.plot).lower() != "none":
        try:
            suite = xafs_plots.render_plot_suite(
                st["result"], fout, out, stem,
                mode=args.plot, formats=plot_formats, dpi=int(args.plot_dpi),
                sample=sample, edge_label=edge_label, shells=explicit_shells,
                chi2_contour=bool(args.chi2_contour),
                phase_correct_export=bool(args.phase_correct),
                indicator_rows=indicator_rows, verbose=True)
            extra_images.extend(suite.get("extra_images") or [])
            for p in suite.get("images") or []:
                print(f"  → {p}")
            for p in suite.get("tables") or []:
                print(f"  → {p}")
        except Exception as exc:  # noqa: BLE001 - 绘图异常不得覆盖已落盘拟合结果
            report_meta["plot_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ⚠ 绘图失败：{exc}", file=sys.stderr)
    else:
        b14_payload = fout
        if b14_payload is None and sid:
            b14_payload = {"l1": sid.get("l1_first_shell"),
                           "absorber": args.absorber}
        if args.phase_correct:
            try:
                pc = xafs_plot_special.write_phase_corrected_diagnostics(
                    st["result"], b14_payload, out, stem=stem)
                extra_images.append(
                    (pc["png"], f"相移校正真键长（表观 {pc['r_peak_raw']:.3f} Å → "
                                f"真实 {pc['r_peak_corrected']:.3f} Å）"))
            except XAFSConfigError as exc:
                print(f"  ⚠ 跳过相移校正：{exc.message}")
        if args.chi2_contour:
            cc = xafs_plot_special.write_chi2_contour_diagnostics(
                st["result"], b14_payload, out, stem=stem)
            if cc is not None:
                extra_images.append(
                    (cc["png"], f"二维 χ² 置信椭圆（{cc['x_name']}–{cc['y_name']}，"
                                f"min χ²={cc['min_chi2']:.3f}）"))

    if not args.no_html:
        html_path = export_html_report(
            out, stem, sample=sample, edge_label=edge_label,
            indicator_rows=indicator_rows, table_rows=rows,
            fit_out=fout, extra_images=extra_images)
        print(f"  → {html_path}")

    # run_manifest 须在全部产物（含图件/HTML）落盘后构建，否则晚写图件漏列。
    # protect=False：覆盖本次运行刚写的中间台账（用户旧记录已在早写时备份）。
    report_meta["run_manifest"] = xafs_provenance.build_run_manifest(
        out, _artifact_baseline, meta_name=f"{stem}_meta.json")
    xafs_writer.safe_write_json(out / f"{stem}_meta.json", report_meta,
                                default=xafs_writer.json_default, protect=False)

    print(f"\n输出目录 {out}:")
    for f in sorted(out.glob(f"{stem}*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
