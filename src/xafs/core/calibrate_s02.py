# -*- coding: utf-8 -*-
"""
calibrate_s02.py — 用金属箔标准样标定 S₀²（新元素接入第 2 步）
================================================================
物理依据：χ(k) 振幅 = S₀²·N·F(k)·…。对完全已知结构的标样
（金属箔：第一壳 N 与 R 晶体学确定），把 N 固定为晶体学简并度、
ΔR/σ²/ΔE₀/ S₀² 中只放开 S₀²（+ΔE₀），拟合出的 S₀² 即标定值。

注意（诚实边界）：
  - S₀² 是**本管线归一化约定下的相对值**（协议="生产等效拟合"：
    N 固定晶体学简并度、S₀² + 每路径 ΔR/σ² + 全局 ΔE₀ 自由，与
    生产拟合变 N 同构），不是普适常数——它吸收了归一化/边跳定义/
    有效路径集/数据-FEFF 形状失配的实现差异；跨管线不可直接移植。
  - 数学上"定 N 变 S₀²"≡"定 S₀² 变 N"（振幅只以乘积出现），但
    数值上必须与生产拟合同构：标定值写入后，生产拟合的 CN 直接
    复现晶体学简并度，这才是标定的目的。
  - 门禁：S₀²∈[0.4,1.15]；s02_err<0.06；Rf≤0.12；|ΔE₀|<10 eV；
    vars<N_idp（每窗口分别核验）；**跨 3 个窗口宽的 s02 极差
    ≤0.05**（过参数化窗口拟合曾给出跨配置 8%–崩塌级的抖动，
    单一"恰好稳定"的配置不可作为生产值）。
  - kw 敏感性（kw=1/3 拟合）只做诊断记录、不门禁：kw=2 才是
    本管线的生产配置；kw≠2 的失真是形状失配的放大镜，不携带
    生产语义。
  - 箔相（hcp/fcc/bcc）是假设，由 --motif 显式覆盖；相错了第一壳
    壳层总数（N=12）基本不变，标定值仍可用，但要在结果里记录假设。
  - 若 absorber.py 已有标定值，拒绝覆盖，除非 --force（重标定）。

用法:
    python calibrate_s02.py <箔数据文件...> --absorber Co [--motif co_hcp]
                            [--kmin 3 --kmax auto] [--apply] [--force]
                            [--strict/--no-strict] [-o 输出目录]

默认 motif 映射（absorber → 第一壳 N=12 的种子 motif）：
    Fe→fe_bcc  Co→co_hcp  Ni→ni_fcc  Cu→cu_fcc  Pt→pt_fcc（未建则报错说明）
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

import absorber as absorber_mod
import artemis
import structure_lib
import xafs_fit
import xafs_io
import xafs_pipeline
from xafs_core import XAFSError, XAFSConfigError, XAFSDataError

# 通过门禁的上下界（模型硬约束 [0.3,1.3] 在 artemis 里）
S02_PASS = (0.4, 1.15)
S02_ERR_MAX = 0.06
RF_MAX = 0.12          # 生产等效拟合（σ² 自由、ΔR±0.03）的真实 Rf 量级
E0_MAX = 10.0
# 鲁棒性：第一壳窗口上沿外推宽度社区；最终值 = 中位数
WINDOW_FRACS = (0.12, 0.18, 0.26)
WINDOW_SPREAD_MAX = 0.05   # 跨窗口 s02 最大-最小 ≤ 此值才算稳定
DELTAR_LIM_CAL = (-0.03, 0.03)   # 箔 RT 热膨胀 << 0.03 Å；防 ΔR–ΔE₀ 相 stray

DEFAULT_MOTIF = {
    "Fe": "fe_bcc", "Co": "co_hcp", "Ni": "ni_fcc", "Cu": "cu_fcc",
    "Pt": "pt_fcc", "Au": "au_fcc", "Mo": "mo_bcc", "Cr": "cr_bcc",
    "Zn": "zn_hcp",
}

# #60 契约：这 12 个元素的 S₀² 静态表必须保持 None。T5 标定结果只能
# 落到独立标定产物，运行时通过 run_xafs.py --s02 显式覆盖。
# --apply 对它们禁用（不是“标不出来”，而是“不得改静态表”）。
T5_STATIC_LOCKED = frozenset({
    "Cr", "Mn", "Zn", "Mo", "Ru", "Rh", "Pd", "Ag", "Ir", "Au", "Ce", "U",
})


def load_motif_json(path, absorber_symbol: str) -> structure_lib.Motif:
    """加载标定专用 motif JSON（**不写入** _structure_lib）。

    JSON 必须带展开后的 ``sites``：这样 ``ensure_feff`` 直接使用近邻
    坐标，不会调用 ``_cif_path`` 往生产 motif 库写生成 CIF；标定专用
    结构因而与 43 条生产 motif 的覆盖率/溯源断言完全隔离。

    ``shell_expect`` 可选但建议保留，用于人工复核第一壳几何。
    """
    import json
    p = Path(path)
    if not p.is_file():
        raise XAFSDataError("CAL", f"标定 motif 文件不存在: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise XAFSDataError("CAL", f"{p}: JSON 解析失败: {exc}") from exc
    motif = structure_lib.Motif.from_json(payload)
    if motif.absorber.capitalize() != absorber_symbol.capitalize():
        raise XAFSDataError(
            "CAL", f"{p}: motif absorber={motif.absorber!r} 与 "
                   f"--absorber={absorber_symbol!r} 不一致")
    if not motif.sites:
        raise XAFSDataError(
            "CAL", f"{p}: 标定专用 motif 必须带展开后的 sites；"
                   f"不能在标定路径里调用 expand_full（会写生产 _cif 目录）")
    return motif


def _resolve_motif(cfg, motif_id: str | None, motif_file=None):
    """解析标定 motif：motif_file > 显式 --motif > DEFAULT_MOTIF > 库内查找。

    两条失败分支分开报错：库为空（先建库）与库内缺该条目（--motif
    或建库）——新元素 on-ramp 的第一步经常撞在这里，错误信息必须
    可直接指导下一步。
    """
    if motif_file is not None:
        motif = load_motif_json(motif_file, cfg.symbol)
        if motif_id is not None and motif.id != motif_id:
            raise XAFSDataError(
                "CAL", f"--motif={motif_id!r} 与 --motif-file 内的 "
                       f"id={motif.id!r} 不一致")
        return motif
    mid = motif_id or DEFAULT_MOTIF.get(cfg.symbol)
    library = structure_lib.load_library(cfg.symbol)
    if not library:
        raise XAFSDataError(
            "CAL", f"{cfg.symbol} 的 motif 库为空（_structure_lib/"
                   f"{cfg.symbol}/ 不存在）——先注册金属箔种子 motif"
                   f"（第一壳 N=12, fcc/hcp/bcc），参考 "
                   f"_structure_lib/build_seed.py 的 Fe/Co/Ni/Cu 写法")
    motif = next((m for m in library if m.id == mid), None)
    if motif is None:
        raise XAFSDataError(
            "CAL", f"{cfg.symbol} 没有 motif {mid!r}（现有库: "
                   f"{[m.id for m in library]}）——金属箔标定需要该相的"
                   f"种子 motif（第一壳 N=12, fcc/hcp/bcc），或用 "
                   f"--motif 显式指定")
    return motif


def calibrate_s02(files, absorber_symbol: str, motif_id: str | None = None,
                  kmin: float | None = None, kmax: float | None = None,
                  s02_start: float = 0.85, strict: bool = True,
                  verbose: bool = True,
                  allow_recalibrate: bool = False,
                  motif_file=None, rmin: float | None = None,
                  e0_method: str = "derivative",
                  manual_e0: float | None = None,
                  fit_shells: int = 1) -> dict:
    """标样箔 → 标定 S₀²。返回 dict；不改 absorber.py（写入见 apply_s02）。

    ``motif_file``：标定专用 motif JSON（必须带展开 sites），用于
    氧化物/化合物标样——不写生产 _structure_lib，不影响 #119 覆盖率。
    ``rmin``：显式 R 窗下限，用于第一壳很近（如 CeO₂ Ce–O 2.34 Å +
    N_idp 不足）时扩大窗口、让 4 变量拟合有足够独立点数。
    ``e0_method`` / ``manual_e0``：E₀ 判定方法，透传 S2。bcc 金属箔的
    导数峰可能落在近边肩峰上（Mo 箔导数峰 20018.8 eV、半高点
    20012.9 eV），导致归一化平台偏低或 ΔE₀ 偏大；改用 half_height
    是数据物理判定，不是放宽门禁。
    ``fit_shells``：一起拟合的壳层数（默认 1）。bcc 金属第一壳
    （8×2.73 Å）与第二壳（6×3.15 Å）同属 Mo–Mo，单壳模型会把
    第二壳尾巴折进第一壳振幅，S₀² 撞上界；双壳模型让两壳各自
    拥有 ΔR/σ²、共享 S₀²/ΔE₀，才能得到稳定的 S₀²。
    """
    cfg = absorber_mod.get_absorber(absorber_symbol)
    if cfg.s02 is not None and not allow_recalibrate:
        raise XAFSConfigError(
            "CAL", f"{cfg.symbol} 已有标定值 {cfg.s02}——重标定是显式决定，"
                   f"调用方需要 --force 或改 absorber.py 后再跑")
    res = xafs_pipeline.process_sample(list(files), absorber=cfg.symbol,
                                       strict=strict, verbose=verbose,
                                       e0_method=e0_method,
                                       manual_e0=manual_e0)
    kmin = res.ft.kmin if kmin is None else kmin
    kmax_limit = float(absorber_mod.effective_kmax_limit(cfg.symbol))
    kmax_req = kmax_limit if kmax is None else float(kmax)
    kmax = min(float(res.ft.kmax), kmax_req, kmax_limit)

    motif = _resolve_motif(cfg, motif_id, motif_file=motif_file)

    plan = xafs_fit.select_shell_paths(
        motif, edge=cfg.edge, kmin=kmin, kmax=kmax, rmin=rmin)
    if len(plan["shells"]) < 1:
        raise XAFSDataError("CAL", f"{motif.id}: 无第一壳路径")
    n_shells = max(1, int(fit_shells))
    if n_shells > len(plan["files"]):
        raise XAFSDataError(
            "CAL", f"{motif.id}: 请求 {n_shells} 壳，但信息量/路径只允许 "
                   f"{len(plan['files'])} 壳")
    first = plan["shells"][0]
    fit_files = plan["files"][:n_shells]
    fit_degens = plan["degens"][:n_shells]
    fit_labels = plan["labels"][:n_shells]
    upper_reff = plan["shells"][n_shells - 1]["reff"]

    # ---- 生产等效拟合（窗口鲁棒性 × 3）----
    # 数学上 "定 N 变 S₀²" ≡ "定 S₀²=1 变 N"（振幅都是 s02·N 乘积），
    # 但自由结构与生产拟合（变 N、σ²/ΔR/ΔE0 自由、kw=2、k∈[3,12]、
    # 同一 FT 约定）完全同构：标定值使生产拟合的 CN 直接落在
    # 晶体学简并度上，管线的幅度-形状失配都被 S₀² 相对值吸收。
    runs = []
    for frac in WINDOW_FRACS:
        r_fit = (plan["rmin"], upper_reff + frac)
        fit = artemis.fit_r_space(
            res.chi_k, res.ft, fit_files,
            rmin=r_fit[0], rmax=r_fit[1], kmin=kmin, kmax=kmax,
            kweight=res.ft.kweight,
            s02=s02_start, vary_delta_e0=True, vary_n=False, vary_s02=True,
            deltar_lim=DELTAR_LIM_CAL,
            path_degen=fit_degens, path_labels=fit_labels)
        p0 = fit.paths[0]
        s2_at_bound = any(bool(getattr(p, "sigma2_at_bound", False))
                          for p in fit.paths)
        run = {"r_window": [round(r_fit[0], 3), round(r_fit[1], 3)],
               "s02": round(float(fit.s02), 4),
               "s02_err": round(float(fit.s02_err), 4),
               "rf": round(float(fit.r_factor), 4),
               "delta_e0": round(float(fit.delta_e0), 2),
               "sigma2": round(float(p0.sigma2), 4),
               "delta_r": round(float(p0.delta_r), 4),
               "sigma2_at_bound": s2_at_bound,
               "paths": [{"label": p.label,
                          "degen": round(float(p.degeneracy), 1),
                          "sigma2": round(float(p.sigma2), 4),
                          "delta_r": round(float(p.delta_r), 4),
                          "sigma2_at_bound": bool(
                              getattr(p, "sigma2_at_bound", False))}
                         for p in fit.paths],
               "interior": (fit.s02_err is not None and
                            S02_PASS[0] < float(fit.s02) < S02_PASS[1] and
                            float(fit.s02_err) < S02_ERR_MAX and
                            float(fit.r_factor) <= RF_MAX and
                            abs(float(fit.delta_e0)) < E0_MAX and
                            not s2_at_bound and
                            fit.n_variables < fit.n_independent)}
        runs.append(run)

    good = [r for r in runs if r["interior"]]
    s02_values = [r["s02"] for r in good]
    spread = (max(s02_values) - min(s02_values)) if len(s02_values) >= 2 \
        else float("nan")
    s02_final = (float(np.median(s02_values)) if s02_values else None)
    s02_err_final = max(r["s02_err"] for r in good) if good else None
    rf_final = max(r["rf"] for r in good) if good else None

    # ---- kw 敏感性（诊断，不门禁）：kw=1/3 不是生产配置，其失真
    # 反映数据-FEFF 形状失配在非常规加权下的表现，仅供迁移性参考。----
    kw_diag = {}
    for kw_probe in (1, 3):
        try:
            r_fit = (plan["rmin"], upper_reff + 0.18)
            f_kw = artemis.fit_r_space(
                res.chi_k, res.ft, fit_files,
                rmin=r_fit[0], rmax=r_fit[1], kmin=kmin, kmax=kmax,
                kweight=kw_probe, s02=s02_start,
                vary_delta_e0=True, vary_n=False, vary_s02=True,
                deltar_lim=DELTAR_LIM_CAL,
                path_degen=fit_degens, path_labels=fit_labels)
            kw_diag[str(kw_probe)] = {
                "s02": round(float(f_kw.s02), 4),
                "rf": round(float(f_kw.r_factor), 4),
                "delta_e0": round(float(f_kw.delta_e0), 2)}
        except XAFSError as exc:
            kw_diag[str(kw_probe)] = {"error": str(exc)}

    # n_independent 按实际 res.chi_k.k 支撑域截断后的 (kmin_eff,kmax_eff) 计算
    k_arr = np.asarray(getattr(res.chi_k, "k", []), dtype=float)
    k_arr = k_arr[np.isfinite(k_arr)]
    if k_arr.size:
        k_support_min = float(np.min(k_arr))
        k_support_max = float(np.max(k_arr))
        kmin_eff = max(float(kmin), k_support_min)
        kmax_eff = min(float(kmax), k_support_max)
    else:
        kmin_eff, kmax_eff = float(kmin), float(kmax)
    n_idp_ref = artemis.independent_points(kmin_eff, kmax_eff, plan["rmin"],
                                           upper_reff + 0.18)

    out = {
        "absorber": cfg.symbol, "edge": cfg.edge, "motif": motif.id,
        "motif_file": (str(motif_file) if motif_file is not None else None),
        "fixed_CN": first["degen"], "R_first": first["reff"],
        "n_shells": n_shells,
        "shell_paths": [{"label": s["label"], "scatterer": s["scatterer"],
                         "reff": s["reff"], "degen": s["degen"]}
                        for s in plan["shells"][:n_shells]],
        "e0_method": res.e0.get("e0_method"),
        "e0": round(float(res.e0.get("e0_sample")), 4),
        "k_window": [round(kmin, 3), round(kmax, 3)],
        "protocol": (f"production_equivalent: {n_shells}-shell, N fixed to "
                     f"crystallographic degeneracy, vary s02 + "
                     f"(dR, sigma2) per path + dE0; kw=2; robustness x3"),
        "runs": runs,
        "s02": (round(s02_final, 4) if s02_final is not None else None),
        "s02_err": (round(s02_err_final, 4) if s02_err_final is not None
                    else None),
        "rf": (round(rf_final, 4) if rf_final is not None else None),
        "spread": round(spread, 4) if spread == spread else None,
        "delta_e0": good[-1]["delta_e0"] if good else None,
        "sigma2": good[-1]["sigma2"] if good else None,
        "n_variables": 2 + 2 * n_shells, "n_independent": round(float(n_idp_ref), 2),
        "plateau": round(float(res.norm.plateau), 4),
        "rbkg_choice": res.rbkg.get("rbkg"),
        "kw_sensitivity": kw_diag,
    }
    checks = {
        "window_runs_interior": len(good) == len(WINDOW_FRACS),
        "s02_in_range": out["s02"] is not None
                        and S02_PASS[0] <= out["s02"] <= S02_PASS[1],
        "s02_err_ok": out["s02_err"] is not None
                      and out["s02_err"] < S02_ERR_MAX,
        "window_spread_ok": (out["spread"] is not None
                             and out["spread"] <= WINDOW_SPREAD_MAX),
        "sigma2_not_at_bound": not any(r["sigma2_at_bound"] for r in runs),
    }
    out["checks"] = checks
    out["pass"] = all(checks.values())
    if verbose:
        print(f"  [CAL] {motif.id} 第一壳 N={first['degen']:.0f} 固定，"
              f"生产等效拟合 ×{len(WINDOW_FRACS)} 窗口，k∈[{kmin:.1f},{kmax:.1f}]")
        for r in runs:
            tag = "✓" if r["interior"] else "✗"
            print(f"        {tag} rmax={r['r_window'][1]:.2f}: "
                  f"s02={r['s02']:.4f}±{r['s02_err']:.4f} rf={r['rf']:.4f} "
                  f"ΔE₀={r['delta_e0']:+.2f} σ²={r['sigma2']:.4f} ΔR={r['delta_r']:+.4f}")
        print(f"        中位 S₀² = {out['s02']} ± 本次 {out['s02_err']}（取最大）"
              f"  窗口极差 {out['spread']}")
        for kprobe, d in kw_diag.items():
            if "error" in d:
                print(f"        kw={kprobe}: {d['error']}")
            else:
                print(f"        kw={kprobe}: s02={d['s02']} rf={d['rf']} "
                      f"ΔE₀={d['delta_e0']:+.1f}（诊断项）")
        for k, v in checks.items():
            print(f"        {'✓' if v else '✗'} {k}")
        print("       ", "标定通过" if out["pass"] else "标定未通过——不写入 absorber.py")
    return out


def certificate_qualifies(payload: dict) -> tuple[bool, list[str]]:
    """独立复核 T5 证书是否满足*现行*标定协议门禁（审查 item 9）。

    不信任证书里存的 ``status``/``pass``/``checks``/``interior`` 布尔——
    那些只是生成当时的快照：协议常量日后收紧、或产物被手改/损坏时，
    快照会与真实指标脱节。这里只用证书保存的*原始指标*（每 run 的
    s02/s02_err/rf/delta_e0/sigma2_at_bound、k_window/r_window、
    n_variables）按 ``calibrate_s02`` 现行常量重算逐 run interior 与五项
    checks，并核对聚合值确由这些 run 推出（防手改聚合 s02 绕过逐 run 门）。

    标定门禁用*标定*常量（RF_MAX=0.12 / E0_MAX=10.0 /
    S02_PASS=(0.4,1.15) / WINDOW_SPREAD_MAX=0.05），不是生产拟合门禁
    （Rf≤0.08 / |ΔE0|≤15）——两者物理含义不同，不可混用，也不得放宽。

    返回 ``(是否合格, 不合格原因列表)``；合格时原因为空。
    """
    from math import isfinite

    def finite(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return isfinite(value)
        except OverflowError:
            return False

    def window(value):
        return (isinstance(value, list) and len(value) == 2
                and all(finite(x) for x in value) and 0 <= value[0] < value[1])

    reasons: list[str] = []
    if not isinstance(payload, dict):
        return False, ["证书非字典"]
    if not isinstance(payload.get("absorber"), str) or not payload["absorber"].strip():
        return False, ["证书缺少合法 absorber 元素标识"]
    if not isinstance(payload.get("edge"), str) or not payload["edge"].strip():
        return False, ["证书缺少合法 edge 吸收边标识"]
    if not isinstance(payload.get("protocol"), str) or not payload["protocol"].strip():
        return False, ["证书缺少 protocol 协议说明"]
    e0_method = payload.get("e0_method")
    if e0_method is not None and e0_method not in ("derivative", "half_height", "manual"):
        return False, [f"证书 e0_method={e0_method!r} 非法"]
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != len(WINDOW_FRACS):
        n = len(runs) if isinstance(runs, list) else "非列表"
        return False, [f"runs 数={n} ≠ 协议窗口数 {len(WINDOW_FRACS)}"]
    if not all(isinstance(r, dict) for r in runs):
        return False, ["run 原始记录非字典"]
    kw = payload.get("k_window")
    if not window(kw):
        return False, ["k_window 缺失/非有限/非递增"]
    n_shells = payload.get("n_shells")
    if isinstance(n_shells, bool) or not isinstance(n_shells, int) or n_shells <= 0:
        return False, ["n_shells 必须是正整数"]
    nvar = 2 * n_shells + 2  # N 固定；每壳 dR/σ²；全局 S02/ΔE0。
    recorded_nvar = payload.get("n_variables")
    if (not finite(recorded_nvar) or recorded_nvar <= 0
            or recorded_nvar != int(recorded_nvar) or recorded_nvar != nvar):
        return False, [f"n_variables 与协议 2*n_shells+2={nvar} 不一致"]
    shells = payload.get("shell_paths")
    if not isinstance(shells, list) or len(shells) != n_shells:
        return False, ["shell_paths 数量与 n_shells 不一致"]
    labels = set()
    previous_reff = 0.0
    for i, shell in enumerate(shells):
        if (not isinstance(shell, dict)
                or not isinstance(shell.get("label"), str) or not shell["label"].strip()
                or not isinstance(shell.get("scatterer"), str) or not shell["scatterer"].strip()
                or not finite(shell.get("reff")) or shell["reff"] <= previous_reff
                or not finite(shell.get("degen")) or shell["degen"] <= 0
                or shell["label"] in labels):
            return False, [f"shell_paths[{i}] 身份/几何/简并度非法"]
        labels.add(shell["label"])
        previous_reff = shell["reff"]
    if not finite(payload.get("n_independent")) or payload["n_independent"] <= 0:
        reasons.append("n_independent 缺失/非有限/非正值")
    first_rw = runs[0].get("r_window")
    if not window(first_rw):
        return False, ["run0 r_window 非法"]
    r_lower = round(first_rw[0], 3)

    # 逐 run 重算 interior（镜像 calibrate_s02 生产等效门）；n_idp 按
    # IFEFFIT 口径 2ΔkΔR/π 由该 run 自己的窗口重算（红线3，不加 2）。
    good = []
    for i, r in enumerate(runs):
        if (not all(finite(r.get(key)) for key in
                    ("s02", "s02_err", "rf", "delta_e0", "sigma2", "delta_r"))
                or any(r[key] < 0 for key in ("s02_err", "rf", "sigma2"))
                or not isinstance(r.get("sigma2_at_bound"), bool)):
            reasons.append(f"run{i} 原始指标缺失/非有限/非法负值")
            continue
        s02, err, rf, de0 = (float(r[key]) for key in
                             ("s02", "s02_err", "rf", "delta_e0"))
        rw = r.get("r_window")
        if not window(rw):
            reasons.append(f"run{i} r_window 非法")
            continue
        # 与生成端 round(..., 3) 同口径；不加物理门限容差。
        expected_rw = [r_lower, round(shells[-1]["reff"] + WINDOW_FRACS[i], 3)]
        if [round(x, 3) for x in rw] != expected_rw:
            reasons.append(f"run{i} r_window={rw} 与协议窗口 {expected_rw} 不一致")
            continue
        paths = r.get("paths")
        if not isinstance(paths, list) or len(paths) != n_shells:
            reasons.append(f"run{i} paths 数量与 n_shells 不一致")
            continue
        valid_paths = True
        for j, (path, shell) in enumerate(zip(paths, shells)):
            if (not isinstance(path, dict) or path.get("label") != shell["label"]
                    or not finite(path.get("degen"))
                    or path["degen"] != round(float(shell["degen"]), 1)
                    or not finite(path.get("sigma2")) or path["sigma2"] <= 5e-4
                    or not finite(path.get("delta_r"))
                    or not isinstance(path.get("sigma2_at_bound"), bool)):
                reasons.append(f"run{i} paths[{j}] 身份/简并度/原始指标非法")
                valid_paths = False
        if not valid_paths:
            continue
        if any(r[key] != paths[0][key] for key in ("sigma2", "delta_r")):
            reasons.append(f"run{i} 首壳原始指标与 paths[0] 不一致")
            continue
        s2b = r["sigma2_at_bound"] or any(p["sigma2_at_bound"] for p in paths)
        nidp = artemis.independent_points(kw[0], kw[1], rw[0], rw[1])
        if (S02_PASS[0] < s02 < S02_PASS[1] and err < S02_ERR_MAX
                and rf <= RF_MAX and abs(de0) < E0_MAX and not s2b
                and finite(nidp) and nvar < nidp):
            good.append(r)
        else:
            reasons.append(
                f"run{i} 未过现行 interior 门(s02={s02}, err={err}, "
                f"rf={rf}, ΔE0={de0}, σ²贴界={s2b}, "
                f"Nvar={nvar}, Nidp={nidp:.2f})")

    # 由 good runs 重算聚合值（与 calibrate_s02 同口径：中位 s02、
    # 最大 err/rf、极差 spread、末个 good 的 ΔE0）。
    if good:
        vals = [float(r["s02"]) for r in good]
        s02_med = round(float(np.median(vals)), 4)
        err_max = round(max(float(r["s02_err"]) for r in good), 4)
        rf_max = round(max(float(r["rf"]) for r in good), 4)
        spread = round(max(vals) - min(vals), 4) if len(vals) >= 2 else None
        de0_last = good[-1]["delta_e0"]
        sigma2_last = good[-1]["sigma2"]
    else:
        s02_med = err_max = rf_max = spread = de0_last = sigma2_last = None

    checks = {
        "window_runs_interior": len(good) == len(WINDOW_FRACS),
        "s02_in_range": (s02_med is not None
                         and S02_PASS[0] <= s02_med <= S02_PASS[1]),
        "s02_err_ok": err_max is not None and err_max < S02_ERR_MAX,
        "window_spread_ok": (spread is not None
                             and spread <= WINDOW_SPREAD_MAX),
        "sigma2_not_at_bound": not any(bool(r.get("sigma2_at_bound"))
                                       for r in runs),
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"现行 check {name} 未过")

    # 聚合值必须确由这些 run 推出：loader 注入的是证书聚合 s02，
    # 若它与逐 run 重算不符（手改/损坏），即使落在门禁内也拒绝。
    for key, redis in (("s02", s02_med), ("s02_err", err_max),
                       ("rf", rf_max), ("spread", spread),
                       ("delta_e0", de0_last), ("sigma2", sigma2_last)):
        stored = payload.get(key)
        if redis is None:
            if stored is not None:
                reasons.append(f"聚合 {key} 应为 None 却={stored!r}")
            continue
        if (not finite(stored)
                or (key in ("s02_err", "rf", "spread", "sigma2") and stored < 0)):
            reasons.append(f"聚合 {key} 缺失/非有限/非法负值")
            continue
        sv = float(stored)
        if round(sv, 4) != round(float(redis), 4):
            reasons.append(
                f"聚合 {key}={sv} 与逐 run 重算 {redis} 不符（疑似手改/损坏）")

    return (not reasons), reasons


def apply_s02(absorber_symbol: str, value: float, *, note: str = "",
              override: bool = False) -> Path:
    """把标定值写进 absorber.py 的 ABSORBERS 表（该元素那一行）。

    正常路径只替换该行的 ``None`` 占位（s02 槽），并在条目后追加标定
    依据注释；``override=True``（CLI --force）才允许覆盖已有数值。
    行按 AbsorberConfig 位置参数格式解析（symbol, z, edge, e0_edge,
    s02, ...），s02 槽以锚定正则定位。命中行数≠1 或格式不符时不写——
    标定值属于生产配置，宁可手工不可赌。
    """
    import re
    cap = absorber_symbol.capitalize()
    # __spec__.origin 是永久真实路径（reload 后 __file__ 会被 loader 重置，
    # 依赖它取路径对连写场景不安全）
    src = (absorber_mod.__spec__.origin if absorber_mod.__spec__ is not None
           else absorber_mod.__file__)
    path = Path(src)
    key = f'"{cap}": AbsorberConfig('
    slot = re.compile(
        rf'^(\s*"{cap}": AbsorberConfig\("[^"]*",\s*[-+]?\d+,\s*"[^"]*",'
        rf'\s*[-+]?[\d.]+(?:[eE][-+]?\d+)?,\s*)'
        rf'(None|[-+]?[\d.]+(?:[eE][-+]?\d+)?)(,.*)$')
    comment = (f"    # 标定: {note}\n" if note else "")
    hit = 0
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip().startswith(key):
            continue
        m = slot.match(line)
        if m is not None and (m.group(2) == "None" or override):
            # 值只含数字/小数点/负号，不会引入正则特殊字符，
            # \g<1>/\g<3> 原样保留前后缀。
            lines[i] = slot.sub(rf"\g<1>{value:.4f}\g<3>", line)
        else:
            raise XAFSConfigError(
                "CAL", f"absorber.py 中 {absorber_symbol} 已有标定值 "
                       f"{m.group(2) if m else '?'}——覆盖需显式 override"
                       f"（CLI --force），且应先复核旧值是否为错误标定")
        if comment:
            lines.insert(i + 1, comment)
        hit += 1
    if hit != 1:
        raise XAFSConfigError(
            "CAL", f"absorber.py 中元素 {absorber_symbol} 命中 {hit} 行（期望 1）"
                   f"——不写入，请手工编辑")
    path.write_text("".join(lines), encoding="utf-8")
    import importlib
    importlib.reload(absorber_mod)
    absorber_mod.get_absorber(absorber_symbol)  # 导入期自检
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="金属箔标样标定 S₀²：N 固定、放开 S₀² 与 ΔE₀ 的 R 空间拟合")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--absorber", required=True)
    ap.add_argument("--motif", default=None,
                    help="标样相的种子 motif（默认 fcc/hcp/bcc 金属映射）")
    ap.add_argument("--motif-file", default=None,
                    help="标定专用 motif JSON（必须带展开 sites；"
                         "不写入生产 _structure_lib）")
    ap.add_argument("--kmin", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=None)
    ap.add_argument("--rmin", type=float, default=None,
                    help="拟合 R 窗下限（默认按第一壳自动；"
                         "第一壳很近时显式放宽以增加 N_idp）")
    ap.add_argument("--e0-method", default="derivative",
                    choices=("derivative", "half_height", "manual"),
                    help="E₀ 判定方法；厚样/近边肩峰导致平台偏低时可用 "
                         "half_height（默认 derivative）")
    ap.add_argument("--manual-e0", type=float, default=None,
                    help="--e0-method manual 时的 E₀ (eV)")
    ap.add_argument("--fit-shells", type=int, default=1,
                    help="一起拟合的壳层数（默认 1；bcc 金属可用 2 让"
                         "第一/第二壳同属 Mo–Mo 并共享 S₀²/ΔE₀）")
    ap.add_argument("--s02-start", type=float, default=0.85)
    ap.add_argument("--strict", dest="strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--apply", action="store_true",
                    help="通过门禁后写入 absorber.py（默认只打印建议值）")
    ap.add_argument("--force", action="store_true",
                    help="允许重标定/覆盖 absorber.py 中已有标定值（显式决定）")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args(argv)

    res = calibrate_s02(args.files, args.absorber, motif_id=args.motif,
                        kmin=args.kmin, kmax=args.kmax,
                        s02_start=args.s02_start, strict=args.strict,
                        allow_recalibrate=args.force,
                        motif_file=args.motif_file, rmin=args.rmin,
                        e0_method=args.e0_method,
                        manual_e0=args.manual_e0,
                        fit_shells=args.fit_shells)
    if args.outdir:
        xafs_io.write_json(Path(args.outdir) / "calibration.json", res)
        print(f"  → {Path(args.outdir) / 'calibration.json'}")
    if not res["pass"]:
        raise SystemExit(1)
    if args.apply:
        cap = args.absorber.capitalize()
        if cap in T5_STATIC_LOCKED:
            raise SystemExit(
                f"  --apply 拒绝：{cap} 属于 #60 锁定的 12 个元素，"
                f"静态 s02 必须保持 None。\n"
                f"  标定结果已写入独立产物；运行时用 "
                f"run_xafs.py --s02 {res['s02']} 或 --s02-from-t5 "
                f"显式覆盖。")
        note = (f"{args.absorber} 标样 {res['motif']} N={res['fixed_CN']:.0f}, "
                f"Rf={res['rf']}, {_dt.date.today().isoformat()}")
        p = apply_s02(args.absorber, res["s02"], note=note,
                      override=args.force)
        print(f"  absorber.py 已更新 → {p}")
    else:
        print(f"  建议: absorber.py 中 {args.absorber} 的 s02 填 "
              f"{res['s02']}（--apply 可自动写入；#60 锁定的 12 个"
              f"元素只能走独立标定产物/运行时 --s02）")


if __name__ == "__main__":
    main()
