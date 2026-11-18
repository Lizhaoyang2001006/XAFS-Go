# -*- coding: utf-8 -*-
"""artemis_fit_single.py — #5 R/q/WT 单谱最小二乘拟合（自 artemis.py 纯拆分）

纯搬运拆分：代码块与原 artemis.py 逐行一致，不含任何逻辑改动。
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

import athena
from xafs_core import (FitResult, PathFit, XAFSConfigError, XAFSDataError)

from artemis_feff import load_feff_path  # noqa: E402
from artemis_stats import (  # noqa: E402
    C3_LIMIT, C4_LIMIT, _parse_kweights, _path_ligand, bond_valence_sum,
    estimate_r_noise, exafs_model_k, independent_points,
    normalize_constraints, r_factor, set_path_parameters,
)


def fit_r_space(chi_k, ft, datfiles, rmin: float, rmax: float,
                kmin: float = 3.0, kmax: float = 12.0, kweight=2,
                s02: float | None = None, vary_delta_e0: bool = True,
                vary_n=False, vary_c3=False, vary_s02=False,
                vary_sigma2=True, sigma2_init: float = 0.005,
                deltar_lim: tuple = (-0.3, 0.3),
                path_degen: list | None = None,
                path_labels: list | None = None,
                path_groups: list | None = None,
                path_group_names: list | None = None,
                vary_weights: bool = False,
                weight_init: list | None = None,
                sigma2_groups: list | None = None,
                sigma2_group_names: list | None = None,
                ms_links: list | None = None,
                isotropic_deltar: bool = False,
                vary_c4=False,
                bvs_restraint: dict | None = None,
                path_links: list | None = None,
                constraints=None,
                fit_space: str = "r",
                wt_sigma: float = 1.5,
                wt_rmin: float | None = None,
                wt_rmax: float | None = None,
                wt_kmin: float | None = None,
                wt_kmax: float | None = None):
    """
    在 R 空间对 χ(R) 的实部与虚部同时最小二乘。

    数学:
        min Σ_j [ (Re χ_data(R_j) − Re χ_th(R_j))²
                + (Im χ_data(R_j) − Im χ_th(R_j))² ]

    拟合变量: 每条路径的 ΔR、σ²；全局共享的 ΔE₀；
    可选每条路径的 N（配位数）。
    **只拟合 |χ(R)| 会丢相位**：实测用幅度拟合时 ΔE₀ 不可识别，
    Jacobian 直接奇异（Singular matrix）。

    vary_n:
        False / None   N 用 FEFF 晶体学简并度（固定）
        True           所有路径的 N 都作为变量
        [0, 2, ...]    只把指定下标的路径 N 设为变量（如只放开第一壳层）
    vary_c3: 同 vary_n 的开关语义，放开各路径三阶累积量 C₃（Å³）。
        仅用于近邻分布显著偏斜的体系（纳米颗粒表面张力、强非谐）；
        C₃ 与 ΔR 部分相关，常规体系不要开（会吃掉 R 的精度）。
    vary_sigma2: True / False / 路径下标列表。False 时 σ² 全部固定在
        sigma2_init——标定（calibrate_s02.py）用它把问题降成
        (S₀², ΔE₀) 两变量近线性拟合，否则 4 变量在窄窗内过参数化
        （实测 Fe 箔 kw=1/3 直接塌到参数边界、Rf 0.63，Kw 敏感）。
    sigma2_groups: 可选，列表的列表，如 [[0, 1], [2]] 表示路径 0、1
        共享一个 σ² 变量，路径 2 独立。用于“多壳层共享 σ²”的简化模型：
        减少自由参数、避免 σ² 与 ΔR 过度相关。组内路径输出相同的
        σ² 和 σ²_err。与 vary_sigma2 互斥（提供其一即可）。
    sigma2_init: 各路径 σ² 初值（默认 0.005）；σ² 被固定时即锁定值。
    deltar_lim: ΔR 的拟合边界 (下, 上)，默认 ±0.3。标定时收紧到
        ±0.02（RT 金属箔热膨胀位移量级），防止 ΔR 与 ΔE₀ 相 stray。
    vary_s02: **标定模式专用**（calibrate_s02.py）——N 固定为晶体学
        简并度、S₀² 作为唯一振幅变量放下。生产拟合严禁用它
        （S₀²·N 简并，会同时抬高两者）；s02=None 且 vary_s02 时
        用 1.0 作初值（标定不需要先验标定值）。
    path_degen: 可选，与 datfiles 等长；拟合前覆盖各路径的初始简并度。
        用于“合并壳层”的代表路径：把同壳多条 SS 路径的简并度加总到
        代表路径上，拟合出的 N 才是壳层总配位数，而不是单条路径的 N。
    path_labels: 可选，与 datfiles 等长；写入 PathFit.label 的壳层名。
    path_groups / path_group_names / vary_weights / weight_init:
        **mixture 相分数拟合**。path_groups 是每条路径所属组分下标
        （0..K-1，连续），vary_weights=True 时用 softmax 参数化放开
        K 个组分权重（恒正、和=1），振幅按 ``N_crystal·w_k`` 缩放。
        此时禁止同时 vary_n（N 与 w 只以乘积出现，严格简并）。
        输出 FitResult.mixture_weights = [{group, weight, weight_err}]；
        PathFit.degeneracy 是**有效简并度** N_crystal·w（不是配位数），
        n_err 由权重误差传播得到。权重误差来自协方差 + softmax 雅可比。
        weight_init 为各组分初始权重（默认均匀）。
    S₀² 与 N 在振幅上只以乘积 S₀²·N 出现，强关联不可同时放开——
    这里固定 S₀²（由标样定）、放开 N，是 Artemis 的标准做法。
    N 的实现：larch 的 _calc_chi 直接读 path.degen 属性乘进振幅
    （与 ΔR/σ² 走 params 字典不同），故每轮迭代直接改属性。

    ``isotropic_deltar``：True 时所有单散射路径共用唯一全局晶格膨胀
    系数 α（槽位 ``(-5, "alpha")``），ΔR_i = α·Reff_i，边界 ±0.08
    （±8% 晶格应变）。多壳层立方/高对称体系可把 M 个 ΔR 变量压缩成
    1 个，避免远壳层因 N_idp 不足被截断。
    ``vary_c4``：同 vary_c3 的开关语义，放开各路径四阶累积量 C₄
    （槽位 ``(i, "c4")``，边界 ±C4_LIMIT = ±2e-4 Å⁴）。

    ``bvs_restraint``（B11-2）：``{"target", "sigma", "r0_table"}``，
    把 ``(BVS−target)/sigma`` 追加到残差；只增加一行约束、不增加自由
    变量。输出 FitResult.bvs/bvs_target/bvs_chi2。
    ``fit_space``（B11-3）：``"r"``（默认，历史逐位一致）/ ``"q"``
    （R 窗反傅里叶）/ ``"wt"``（二维复数小波）。不同空间的 R-factor
    口径不同，不可横向比较。

    返回 FitResult。
    """
    from scipy.optimize import least_squares

    fit_space = str(fit_space or "r").strip().lower()
    if fit_space not in ("r", "q", "wt"):
        raise XAFSConfigError(
            "FIT", f"fit_space={fit_space!r} 只支持 'r' / 'q' / 'wt'")

    if s02 is None and not vary_s02:
        raise XAFSConfigError(
            "FIT", "s02 未提供——S₀² 必须显式传入（absorber.py 标定值，"
                   "Fe=0.803 箔标定）；无标定元素的 CN 定量一律拒绝")
    kw_tuple = _parse_kweights(kweight)
    multi_kw = len(kw_tuple) > 1
    # 主权重：生产约定 kw=2；不在元组中时取中位/第一个。
    primary_kw = 2 if 2 in kw_tuple else kw_tuple[len(kw_tuple) // 2]
    paths = [load_feff_path(f) for f in datfiles]
    n_path = len(paths)
    if path_degen is not None:
        if len(path_degen) != n_path:
            raise XAFSConfigError(
                "FIT", f"path_degen 长度 {len(path_degen)} != 路径数 {n_path}")
        for path, degen0 in zip(paths, path_degen):
            path.degen = float(degen0)
    if path_labels is not None and len(path_labels) != n_path:
        raise XAFSConfigError(
            "FIT", f"path_labels 长度 {len(path_labels)} != 路径数 {n_path}")
    # 晶体学简并度快照：权重模式下每轮迭代按 N_crystal·w 重新写 degen。
    crystal_degen = np.asarray([float(p.degen) for p in paths], dtype=float)

    # ---- 受约束多散射（MS）：路径参数由母单散射壳层按几何比例驱动 ----
    # 关键：MS 路径**不建立任何自由变量槽位**，n_var 不变，因此不会因
    # 加 MS 而触发 N_idp 过拟合截断。
    ms_by_path: dict[int, tuple] = {}
    if ms_links:
        for link in ms_links:
            idx = int(link["ms_path_index"])
            parent = int(link["parent_shell_index"])
            if not (0 <= idx < n_path and 0 <= parent < n_path):
                raise XAFSConfigError(
                    "FIT", f"ms_links 下标越界: ms={idx}, parent={parent}")
            if idx == parent or idx in ms_by_path:
                raise XAFSConfigError(
                    "FIT", f"ms_links 路径 {idx} 重复或指向自身")
            if parent in ms_by_path:
                raise XAFSConfigError(
                    "FIT", f"MS 母壳层 {parent} 不能又是受约束 MS 路径")
            ms_by_path[idx] = (parent, float(link["reff_ratio"]),
                               float(link["degen_ratio"]),
                               float(link.get("sigma2_ratio", link["reff_ratio"])))
    ms_path_set = set(ms_by_path)

    # ---- σ² 共享组：多条路径共享一个 σ² 变量，减少自由参数 ----
    sigma2_group_of_path = {}   # path_idx -> group_idx
    n_s2_group = 0
    if sigma2_groups and ms_path_set:
        # MS 路径的 σ² 由母壳层驱动，不参与共享组变量。名称必须与
        # 保留下来的组同步过滤——按原下标截断会在中间组被删空时错位。
        filtered = []
        filtered_names = []
        for gi, group in enumerate(sigma2_groups):
            keep = [int(pi) for pi in group if int(pi) not in ms_path_set]
            if keep:
                filtered.append(keep)
                if sigma2_group_names is not None \
                        and gi < len(sigma2_group_names):
                    filtered_names.append(sigma2_group_names[gi])
        sigma2_groups = filtered
        if sigma2_group_names is not None:
            sigma2_group_names = filtered_names
    if sigma2_groups:
        n_s2_group = len(sigma2_groups)
        if sigma2_group_names is not None \
                and len(sigma2_group_names) != n_s2_group:
            raise XAFSConfigError(
                "FIT", f"sigma2_group_names 长度 {len(sigma2_group_names)} "
                       f"!= 组数 {n_s2_group}")
        for gi, group in enumerate(sigma2_groups):
            for pi in group:
                pi = int(pi)
                if pi < 0 or pi >= n_path:
                    raise XAFSConfigError(
                        "FIT", f"sigma2_groups 路径下标 {pi} 超出范围 0–{n_path-1}")
                if pi in sigma2_group_of_path:
                    raise XAFSConfigError(
                        "FIT", f"路径 {pi} 出现在多个 σ² 共享组中，"
                               f"每条路径只能属于一个组")
                sigma2_group_of_path[pi] = gi
    sigma2_group_names = (list(sigma2_group_names)
                          if sigma2_group_names is not None
                          else [f"s2grp{g}" for g in range(n_s2_group)])

    # ---- B13-2 单数据集内路径绑定（异核共占位双分支）----
    link_deltar: dict[int, int] = {}
    link_sigma2: dict[int, int] = {}
    link_n: dict[int, tuple[int, float]] = {}
    for li, link in enumerate(path_links or []):
        if not isinstance(link, dict):
            raise XAFSConfigError("FIT", f"path_links[{li}] 必须是字典")
        try:
            tgt = int(link["target"])
            src = int(link["source"])
        except (KeyError, TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", f"path_links[{li}] 需要整数 target/source") from exc
        for val, what in ((tgt, "target"), (src, "source")):
            if val < 0 or val >= n_path:
                raise XAFSConfigError(
                    "FIT", f"path_links[{li}].{what}={val} 超出路径范围 "
                           f"0–{n_path - 1}")
        if tgt == src:
            raise XAFSConfigError(
                "FIT", f"path_links[{li}] 的 target==source，自绑定无约束")
        if tgt in ms_path_set or src in ms_path_set:
            raise XAFSConfigError(
                "FIT", f"path_links[{li}] 不能以受约束 MS 路径作为 target 或 source")
        if link.get("share_deltar", True):
            if tgt in link_deltar:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多条 share_deltar 绑定")
            link_deltar[tgt] = src
        if link.get("share_sigma2", True):
            if tgt in link_sigma2:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多条 share_sigma2 绑定")
            link_sigma2[tgt] = src
        ratio = link.get("degen_ratio")
        if ratio is not None:
            ratio = float(ratio)
            if not np.isfinite(ratio) or ratio <= 0:
                raise XAFSConfigError(
                    "FIT", f"path_links[{li}].degen_ratio={ratio!r} 必须 > 0")
            if tgt in link_n:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多条 degen_ratio 绑定")
            link_n[tgt] = (src, ratio)
    if link_deltar and isotropic_deltar:
        raise XAFSConfigError(
            "FIT", "path_links 的 share_deltar 与 isotropic_deltar 互斥："
                   "全局 α 已经把所有 ΔR 绑在一起")
    if link_sigma2 and sigma2_groups:
        for tgt in link_sigma2:
            if tgt in sigma2_group_of_path:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 同时被 share_sigma2 与 σ² 共享组约束")

    # ---- B14-1 催化化学计量约束（Mini-GDS）----
    fit_labels = [str(path_labels[i]) if path_labels is not None
                  else Path(datfiles[i]).name for i in range(n_path)]
    con_list = normalize_constraints(constraints, fit_labels)
    con_n: dict[int, dict] = {}
    con_dr: dict[int, dict] = {}
    for rec in con_list:
        if rec["kind"] in ("n_sum", "n_ratio"):
            con_n[int(rec["target_index"])] = rec
        elif rec["kind"] in ("deltar_offset", "deltar_ratio"):
            con_dr[int(rec["target_index"])] = rec
    if con_dr and isotropic_deltar:
        raise XAFSConfigError(
            "FIT", "deltar_offset / deltar_ratio 约束与 isotropic_deltar 互斥"
            "（后者把所有 ΔR 压成唯一全局 α，不能再逐路径绑定）")
    for tgt in set(con_n) & set(link_n):
        raise XAFSConfigError("FIT", f"路径 {tgt} 的 N 同时被 path_links 与 constraints 绑定")
    for tgt in set(con_dr) & set(link_deltar):
        raise XAFSConfigError("FIT", f"路径 {tgt} 的 ΔR 同时被 path_links 与 constraints 绑定")

    for rec in con_list:
        if rec["kind"] != "bounds" and (
                int(rec["target_index"]) in ms_path_set
                or int(rec["source_index"]) in ms_path_set):
            raise XAFSConfigError(
                "FIT", f"约束 {rec['target']!r}←{rec['source']!r} 引用了受约束 MS 路径，"
                       f"其 N/ΔR 由母壳层驱动，不能再参与化学计量约束")

    # ---- 哪些路径的 N 作为变量 ----
    if vary_n is True:
        n_paths = set(range(n_path))
    elif vary_n:
        n_paths = {int(i) for i in vary_n}
        bad = n_paths - set(range(n_path))
        if bad:
            raise XAFSConfigError(
                "FIT", f"vary_n 索引 {sorted(bad)} 超出路径范围 0–{n_path - 1}")
    else:
        n_paths = set()
    # MS 路径的 N 由母壳层驱动；即使用户传 True 也不给它独立变量。
    n_paths -= ms_path_set
    # B13-2/B14-1：被比例/守恒约束绑定的 N 不是自由变量。
    n_paths -= set(link_n) | set(con_n)
    # S₀²·N 严格乘积简并：标定固定 N 放 S₀²，生产固定 S₀² 放 N，
    # 二者绝不可同时作为自由变量（底层公共 API 同样拦截）。
    if vary_s02 and n_paths:
        raise XAFSConfigError(
            "FIT", "vary_s02 与自由 N 不能同时放开：振幅只以 S₀²·N "
                   "乘积出现，两者同时自由严格简并、不可辨识；"
                   "标定请固定 N，生产 CN 请固定 S₀²")
    c3_paths = set(range(n_path)) if vary_c3 is True else \
        ({int(i) for i in vary_c3} if vary_c3 else set())
    c3_paths -= ms_path_set
    bad_c3 = c3_paths - set(range(n_path))
    if bad_c3:
        raise XAFSConfigError(
            "FIT", f"vary_c3 索引 {sorted(bad_c3)} 超出路径范围 0–{n_path - 1}")
    c4_paths = set(range(n_path)) if vary_c4 is True else \
        ({int(i) for i in vary_c4} if vary_c4 else set())
    c4_paths -= ms_path_set
    bad = c4_paths - set(range(n_path))
    if bad:
        raise XAFSConfigError(
            "FIT", f"vary_c4 索引 {sorted(bad)} 超出路径范围 0–{n_path - 1}")

    # ---- mixture 权重（vary_weights）----
    weight_slot_ids = []
    weight_groups = None
    weight_names = None
    n_group = 0
    w0 = None
    if vary_weights:
        if n_paths:
            raise XAFSConfigError(
                "FIT", "vary_weights 与 vary_n 不能同时放开：振幅只以 "
                       "N·w 的乘积出现，两者同时自由会严格简并")
        if path_groups is None:
            raise XAFSConfigError(
                "FIT", "vary_weights 需要 path_groups（每条路径的组分下标）")
        if len(path_groups) != n_path:
            raise XAFSConfigError(
                "FIT", f"path_groups 长度 {len(path_groups)} != 路径数 {n_path}")
        weight_groups = np.asarray([int(g) for g in path_groups], dtype=int)
        if weight_groups.min() < 0:
            raise XAFSConfigError("FIT", "path_groups 不能含负数")
        group_ids = sorted(set(weight_groups.tolist()))
        if group_ids != list(range(len(group_ids))):
            raise XAFSConfigError(
                "FIT", f"path_groups 必须是 0..K-1 连续下标，收到 {group_ids}")
        n_group = len(group_ids)
        if n_group < 2:
            raise XAFSConfigError(
                "FIT", "vary_weights 至少需要 2 个组分（单组分权重恒为 1）")
        if weight_init is None:
            w0 = np.full(n_group, 1.0 / n_group)
        else:
            w0 = np.asarray(weight_init, dtype=float)
            if w0.shape != (n_group,):
                raise XAFSConfigError(
                    "FIT", f"weight_init 长度 {len(w0)} != 组分数 {n_group}")
            if np.any(w0 <= 0) or abs(float(w0.sum()) - 1.0) > 1e-6:
                raise XAFSConfigError("FIT", "weight_init 必须为正且和为 1")
        weight_names = (list(path_group_names) if path_group_names is not None
                        else [f"group{k}" for k in range(n_group)])
        if len(weight_names) != n_group:
            raise XAFSConfigError(
                "FIT", f"path_group_names 长度 {len(weight_names)} != 组分数 {n_group}")

    # ---- 变量槽位: 每条路径 (ΔR[, σ²][, N][, C₃])，末尾可选 ΔE₀ ----
    slots = []                       # [(path_idx, 'deltar'|'sigma2'|'n'|'c3'|'e0'), ...]
    x0, lower, upper = [], [], []
    if vary_sigma2 is True:
        s2_paths = set(range(n_path))
    else:
        s2_paths = ({int(i) for i in vary_sigma2} if vary_sigma2 else set())
        bad = s2_paths - set(range(n_path))
        if bad:
            raise XAFSConfigError(
                "FIT", f"vary_sigma2 索引 {sorted(bad)} 超出路径范围 0–{n_path - 1}")
    # 与 σ² 共享组互斥
    if sigma2_groups and vary_sigma2 is not True:
        raise XAFSConfigError(
            "FIT", "sigma2_groups 与 vary_sigma2 列表模式不可同时使用；"
                   "请只提供 sigma2_groups，或只提供 vary_sigma2")

    # 1) 每条 SS 路径的 ΔR（MS 路径由母壳层比例驱动，不建槽位）
    # B7-1：isotropic_deltar 时不再逐路径建 ΔR，只建唯一全局 α。
    use_iso_deltar = bool(isotropic_deltar) and n_path > 1
    if use_iso_deltar:
        slots.append((-5, "alpha"))
        x0.append(0.0)
        lower.append(-0.08); upper.append(0.08)
    else:
        for i, path in enumerate(paths):
            if i in ms_path_set or i in link_deltar or i in con_dr:
                continue
            slots.append((i, "deltar"))
            x0.append(0.0); lower.append(deltar_lim[0]); upper.append(deltar_lim[1])

    # 2) σ²：共享组（每组一个变量）或独立
    s2_independent = set()           # 拥有独立 σ² 变量的路径下标
    if sigma2_groups:
        for gi, group in enumerate(sigma2_groups):
            slots.append((-4, gi))   # -4 标记 σ² 共享组，gi=组索引
            x0.append(float(sigma2_init)); lower.append(1e-4); upper.append(0.04)
    else:
        for i in sorted(s2_paths - ms_path_set - set(link_sigma2)):
            slots.append((i, "sigma2"))
            x0.append(float(sigma2_init)); lower.append(1e-4); upper.append(0.04)
            s2_independent.add(i)

    # 3) N
    for i in sorted(n_paths):
        # 必须按**该路径自己的**晶体学简并度设初值与边界；旧版误用
        # 循环残留变量（上一条路径的 degen），多壳层时会把近壳 N 的
        # 上界错误压到 3×远壳简并度，直接截断真实配位数。
        degen = float(crystal_degen[i])
        # 审查 item 8：零/负占据路径放开 N 会给出 lower=0.5 > upper=3·degen≤0
        # 的不可行边界（N 也不可辨识）；配位数定量必须显式拒绝而非静默崩溃。
        if degen <= 0:
            raise XAFSConfigError(
                "FIT", f"路径 {i} 晶体学简并度={degen} ≤ 0，无法放开其 N "
                       f"做配位数定量（零/负占据壳无 CN 可言）")
        slots.append((i, "n"))
        x0.append(degen)
        lower.append(max(0.5, 0.1 * degen))
        upper.append(3.0 * degen)
    # 4) C3
    for i in sorted(c3_paths):
        slots.append((i, "c3"))
        x0.append(0.0); lower.append(-C3_LIMIT); upper.append(C3_LIMIT)
    # 4.5) C4（B7-3）
    for i in sorted(c4_paths):
        slots.append((i, "c4"))
        x0.append(0.0); lower.append(-C4_LIMIT); upper.append(C4_LIMIT)
    # 5) S0²（标定模式）
    if vary_s02:
        slots.append((-2, "s02"))
        x0.append(1.0 if s02 is None else float(s02))
        lower.append(0.3); upper.append(1.3)
    # 6) mixture 权重 softmax logit
    if vary_weights:
        for k in range(n_group - 1):
            p0 = np.log(max(float(w0[k]), 1e-6) / max(float(w0[-1]), 1e-6))
            slots.append((-3, k))
            x0.append(float(np.clip(p0, -8.0, 8.0)))
            lower.append(-8.0); upper.append(8.0)
            weight_slot_ids.append(len(slots) - 1)
    # 7) 全局 ΔE₀
    if vary_delta_e0:
        slots.append((-1, "e0"))
        x0.append(0.0); lower.append(-15.0); upper.append(15.0)
    n_var = len(slots)
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    slot_index = {s: idx for idx, s in enumerate(slots)}

    # ---- Stern 硬门禁：求解前校验 N_var < N_idp（IFEFFIT 口径，不加 2）----
    # N_idp 按实际有效 k 窗计算：请求窗超出 χ(k) 数据支持域的部分不含
    # 独立信息，不能用来虚增自由度。窗口非法/无交集直接拒绝。
    if not (np.isfinite(kmin) and np.isfinite(kmax) and kmin < kmax):
        raise XAFSConfigError("FIT", f"k 窗非法: [{kmin}, {kmax}]")
    if not (np.isfinite(rmin) and np.isfinite(rmax) and rmin < rmax):
        raise XAFSConfigError("FIT", f"R 窗非法: [{rmin}, {rmax}]")
    _k_data = np.asarray(chi_k.k, dtype=float)
    _k_lo = float(max(kmin, _k_data.min()))
    _k_hi = float(min(kmax, _k_data.max()))
    if _k_hi <= _k_lo:
        raise XAFSConfigError(
            "FIT", f"有效 k 窗为空: 请求 [{kmin},{kmax}] 与数据支持域 "
                   f"[{_k_data.min():.2f},{_k_data.max():.2f}] 无交集")
    n_idp_eff = independent_points(_k_lo, _k_hi, rmin, rmax)
    if n_var >= n_idp_eff:
        raise XAFSConfigError(
            "FIT", f"自由变量数 N_var={n_var} ≥ N_idp={n_idp_eff:.2f}"
                   f"（有效 k∈[{_k_lo:.2f},{_k_hi:.2f}], "
                   f"R∈[{rmin},{rmax}]）；违反 Stern 规则，拒绝求解"
                   f"（请减少自由参数或扩大有效窗口）")

    # ---- B14-1 约束边界：bounds 覆盖 + n_sum 的物理非负性 ----
    for rec in con_list:
        if rec["kind"] != "bounds":
            continue
        key = (int(rec["path_index"]), rec["param"])
        if key not in slot_index:
            raise XAFSConfigError(
                "FIT", f"bounds 目标 {rec['path']}:{rec['param']} 不是自由变量"
                       f"（可能被约束/固定），无法施加边界")
        j = slot_index[key]
        lower[j] = max(float(lower[j]), float(rec["min"]))
        upper[j] = min(float(upper[j]), float(rec["max"]))
        if lower[j] >= upper[j]:
            raise XAFSConfigError(
                "FIT", f"bounds 与现有边界不相容: {rec['path']}:{rec['param']} "
                       f"[{lower[j]}, {upper[j]}]")
    for rec in con_list:
        if rec["kind"] != "n_sum":
            continue
        key = (int(rec["source_index"]), "n")
        if key not in slot_index:
            continue
        j = slot_index[key]
        ub = float(rec["total"]) / float(rec["coeff"])
        if ub <= float(lower[j]):
            raise XAFSConfigError(
                "FIT", f"n_sum 约束不可行：total/coeff={ub:.3f} ≤ "
                       f"{rec['source']} 的 N 下界 {lower[j]:.3f}")
        upper[j] = min(float(upper[j]), ub)

    # 自由变量名：与 covariance/correlation 行列一一对应，报告层据此解释
    # 强相关（N–σ²–ΔE0）而不是把形式误差当独立精度。
    _kind_names = {"deltar": "ΔR", "sigma2": "σ²", "n": "N",
                   "c3": "C3", "c4": "C4", "s02": "S0²", "e0": "ΔE0",
                   "alpha": "lattice_alpha"}
    parameter_names = []
    for (pi, pkind) in slots:
        if pi == -4:
            parameter_names.append(f"s2grp:{sigma2_group_names[pkind]}")
        elif pi >= 0:
            plabel = (path_labels[pi] if path_labels is not None
                      else Path(datfiles[pi]).name)
            parameter_names.append(f"{plabel}:{_kind_names[pkind]}")
        elif pi == -3:
            parameter_names.append(f"w:{weight_names[pkind]}")
        else:
            parameter_names.append(_kind_names[pkind])

    def _link_root(mapping, key, what):
        """沿 target→source 链走到根；成环时显式报错。"""
        seen = set()
        k = int(key)
        while k in mapping:
            if k in seen:
                raise XAFSConfigError("FIT", f"{what} 绑定成环: 路径 {k}")
            seen.add(k)
            k = int(mapping[k])
        return k

    def _n_ratio_chain(key):
        """N_key = prod·N_root，返回 (root, prod)。"""
        prod = 1.0
        k = int(key)
        seen = set()
        while k in link_n:
            if k in seen:
                raise XAFSConfigError("FIT", f"degen_ratio 绑定成环: 路径 {k}")
            seen.add(k)
            k, ratio = link_n[k]
            k = int(k)
            prod *= float(ratio)
        return k, prod

    def _n_constraint_chain(key):
        """返回 (root, coeff, const)，满足 N_key = const + coeff·N_root。"""
        if key in con_n:
            rec = con_n[key]
            root, coeff, const = _n_constraint_chain(int(rec["source_index"]))
            if rec["kind"] == "n_sum":
                return root, -float(rec["coeff"]) * coeff, \
                    float(rec["total"]) - float(rec["coeff"]) * const
            return root, float(rec["ratio"]) * coeff, float(rec["ratio"]) * const
        return int(key), 1.0, 0.0

    def _dr_constraint_chain(key):
        """返回 (root, coeff, const)，满足 ΔR_key = coeff·ΔR_root + const。

        与 ``_n_constraint_chain`` 同构；deltar_offset 累加常数项，
        deltar_ratio 累乘系数并把已有常数项一并缩放。
        """
        coeff, const = 1.0, 0.0
        k = int(key)
        seen = set()
        while k in con_dr:
            if k in seen:
                raise XAFSConfigError("FIT", f"ΔR 约束成环: 路径 {k}")
            seen.add(k)
            rec = con_dr[k]
            if rec["kind"] == "deltar_ratio":
                ratio = float(rec["ratio"])
                coeff *= ratio
                const *= ratio
            else:
                const += float(rec["offset"])
            k = int(rec["source_index"])
        return k, coeff, const

    def _n_affine(key, seen=None):
        """N_key = const + coeff·N_root（交替展开 link_n/con_n，与 _n_of 同构）。"""
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"N 约束成环: 路径 {key}")
        seen = seen | {key}
        if key in link_n:
            src, ratio = link_n[key]
            r, c, b = _n_affine(int(src), seen)
            return r, float(ratio) * c, float(ratio) * b
        if key in con_n:
            rec = con_n[key]
            r, c, b = _n_affine(int(rec["source_index"]), seen)
            if rec["kind"] == "n_sum":
                return r, -float(rec["coeff"]) * c, \
                    float(rec["total"]) - float(rec["coeff"]) * b
            return r, float(rec["ratio"]) * c, float(rec["ratio"]) * b
        return int(key), 1.0, 0.0

    # 审查 #18②：固定源/链式约束仍可能解析出负 CN，按仿射式在根可行区间取最小值拒绝。
    for _tgt in sorted(con_n):
        _root, _coeff, _const = _n_affine(_tgt)
        _rkey = (int(_root), "n")
        if _rkey in slot_index:
            _j = slot_index[_rkey]
            _lo, _hi = float(lower[_j]), float(upper[_j])
        else:
            _lo = _hi = float(crystal_degen[_root])
        _n_min = _const + _coeff * (_lo if _coeff >= 0 else _hi)
        if _n_min < -1e-9:
            raise XAFSConfigError(
                "FIT", f"约束可行域检查失败：{fit_labels[_tgt]} 的 N 最小可能值"
                       f"={_n_min:.3f} < 0（根 {fit_labels[_root]} 的 "
                       f"N∈[{_lo:.3f}, {_hi:.3f}]）；禁止负配位数，"
                       f"请修正化学计量约束或放开相应源 N")

    def softmax_weights(x):
        """当前 logit → 组分权重（最后一个组分为参考，logit=0）。"""
        p = np.asarray([x[slot_index[(-3, k)]] for k in range(n_group - 1)]
                       + [0.0], dtype=float)
        p = p - p.max()
        e = np.exp(p)
        return e / e.sum()

    # 模型必须在**数据的 k 网格**上计算，否则 FT 后相位对不齐。
    ft_k = np.asarray(chi_k.k)
    data_r = ft.chi_r
    r_grid = ft.r
    mask = (r_grid >= rmin) & (r_grid <= rmax)
    if mask.sum() < 8:
        raise XAFSConfigError("FIT", f"R 区间 [{rmin},{rmax}] 内仅 {mask.sum()} 点")

    # ---- 多 kweight 联合拟合：逐权重准备实验谱与尺度因子 ----
    # 单权模式直接用传入 ft 的复数谱，保证与历史结果逐位一致。
    # 多权模式对同一 χ(k) 用相同 (kmin,kmax,dk) 重新 FT；分支残差必须
    # 除以该分支实验谱的 RMS 范数 s_w，否则 kw=3 的幅度（~k² 倍）会
    # 完全统治最小二乘，退化成单权 kw=3 拟合。
    data_r_by_kw: dict[int, np.ndarray] = {}
    scale_by_kw: dict[int, float] = {}
    for w in kw_tuple:
        if not multi_kw and w == int(getattr(ft, "kweight", primary_kw)):
            data_w = data_r
        else:
            data_w = athena.ft_to_r_space(
                ft_k, np.asarray(chi_k.chi), None, kmin, kmax, w, ft.dk).chi_r
        data_r_by_kw[w] = data_w
        if multi_kw:
            scale_by_kw[w] = float(np.sqrt(np.mean(np.abs(data_w[mask]) ** 2)))
            if not np.isfinite(scale_by_kw[w]) or scale_by_kw[w] <= 0:
                scale_by_kw[w] = 1.0
        else:
            scale_by_kw[w] = 1.0

    # ---- B11-3 多空间：q（反 FT）或 wt（二维复数小波）的数据侧 ----
    data_space_by_kw: dict = {}
    space_mask_by_kw: dict = {}
    space_scale_by_kw: dict = {}
    wt_r_window = None
    if fit_space == "q":
        for w in kw_tuple:
            bt = athena.back_ft_to_q_space(
                r_grid, data_r_by_kw[w], rmin=float(rmin), rmax=float(rmax),
                dr=float(getattr(ft, "dk", 0.1)))
            q = np.asarray(bt.q, dtype=float)
            mq = (q >= float(kmin)) & (q <= float(kmax))
            if int(mq.sum()) < 8:
                raise XAFSConfigError(
                    "FIT", f"q 空间 [{kmin},{kmax}] 内仅 {int(mq.sum())} 点")
            data_space_by_kw[w] = (q, np.asarray(bt.chi_q, dtype=complex))
            space_mask_by_kw[w] = mq
            s = float(np.sqrt(np.mean(np.abs(bt.chi_q[mq]) ** 2)))
            space_scale_by_kw[w] = s if np.isfinite(s) and s > 0 else 1.0
    elif fit_space == "wt":
        import larch_wt

        rlo = float(wt_rmin if wt_rmin is not None else rmin)
        rhi = float(wt_rmax if wt_rmax is not None else rmax)
        if rhi <= rlo:
            raise XAFSConfigError("FIT", f"wt R 窗无效: [{rlo},{rhi}]")
        wt_r_window = (rlo, rhi)
        wt_rmax_eff = max(rhi, float(rmax))
        for w in kw_tuple:
            wm = larch_wt.continuous_wavelet_transform(
                chi_k, sigma=float(wt_sigma), kmin=float(kmin),
                kmax=min(float(kmax), float(np.max(chi_k.k))),
                kweight=int(w), rmax=wt_rmax_eff, normalize=False)
            kc = np.asarray(wm.k_centers, dtype=float)
            klo = float(wt_kmin if wt_kmin is not None else kc.min())
            khi = float(wt_kmax if wt_kmax is not None else kc.max())
            mk = (kc >= klo) & (kc <= khi)
            mr = (np.asarray(wm.r) >= rlo) & (np.asarray(wm.r) <= rhi)
            if int(mk.sum()) < 3 or int(mr.sum()) < 3:
                raise XAFSConfigError(
                    "FIT", f"WT 窗口过小: k={int(mk.sum())}, R={int(mr.sum())}")
            data_space_by_kw[w] = (kc, np.asarray(wm.r)[mr],
                                   np.asarray(wm.complex_map)[np.ix_(mk, mr)])
            space_mask_by_kw[w] = (mk, mr)
            block = data_space_by_kw[w][2]
            s = float(np.sqrt(np.mean(np.abs(block) ** 2)))
            space_scale_by_kw[w] = s if np.isfinite(s) and s > 0 else 1.0

    def _model_in_space(model_k, w):
        """把模型 χ(k) 变到当前 fit_space 的数据网格（复数）。"""
        if fit_space == "r":
            return _ft_model(model_k, w)
        if fit_space == "q":
            model_r = _ft_model(model_k, w)
            bt = athena.back_ft_to_q_space(
                r_grid, model_r, rmin=float(rmin), rmax=float(rmax),
                dr=float(getattr(ft, "dk", 0.1)))
            q_data = data_space_by_kw[w][0]
            return (np.interp(q_data, bt.q, np.asarray(bt.chi_q).real)
                    + 1j * np.interp(q_data, bt.q, np.asarray(bt.chi_q).imag))
        import larch_wt
        wm = larch_wt.continuous_wavelet_transform(
            _chi_k_like(model_k), sigma=float(wt_sigma), kmin=float(kmin),
            kmax=min(float(kmax), float(np.max(_chi_k_like(model_k).k))),
            kweight=int(w), rmax=max(wt_r_window[1], float(rmax)),
            normalize=False)
        return np.asarray(wm.complex_map)

    def _chi_k_like(model_k):
        from types import SimpleNamespace
        return SimpleNamespace(k=np.asarray(ft_k, dtype=float),
                               chi=np.asarray(model_k, dtype=float))

    # ---- B13-2/B14-1 派生参数求值：沿 link/constraint 链回代到自由变量 ----
    def _dr_of(x, key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"ΔR 绑定成环: 路径 {key}")
        seen.add(key)
        if key in link_deltar:
            return _dr_of(x, int(link_deltar[key]), seen)
        if key in con_dr:
            rec = con_dr[key]
            src = _dr_of(x, int(rec["source_index"]), seen)
            if rec["kind"] == "deltar_ratio":
                return float(rec["ratio"]) * src
            return src + float(rec["offset"])
        sk = (int(key), "deltar")
        return float(x[slot_index[sk]]) if sk in slot_index else 0.0

    def _s2_of(x, key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"σ² 绑定成环: 路径 {key}")
        seen.add(key)
        if key in link_sigma2:
            return _s2_of(x, int(link_sigma2[key]), seen)
        if key in sigma2_group_of_path:
            gi = sigma2_group_of_path[key]
            sk = (-4, gi)
            return float(x[slot_index[sk]]) if sk in slot_index \
                else float(sigma2_init)
        sk = (int(key), "sigma2")
        return float(x[slot_index[sk]]) if sk in slot_index \
            else float(sigma2_init)

    def _n_of(x, key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"N 绑定成环: 路径 {key}")
        seen.add(key)
        if key in link_n:
            src, ratio = link_n[key]
            return float(ratio) * _n_of(x, int(src), seen)
        if key in con_n:
            rec = con_n[key]
            src_val = _n_of(x, int(rec["source_index"]), seen)
            if rec["kind"] == "n_sum":
                return float(rec["total"]) - float(rec["coeff"]) * src_val
            return float(rec["ratio"]) * src_val
        sk = (int(key), "n")
        return (float(x[slot_index[sk]]) if sk in slot_index
                else float(crystal_degen[key]))

    def apply_params(x):
        """把变量向量写回路径：ΔR/σ²/E0 走 params 通道，N 改 degen 属性。"""
        by_path = {}
        for (i, kind), v in zip(slots, x):
            v = float(v)
            if i == -4:
                # σ² 共享组：kind 是组索引，写到组内所有路径
                for pi in sigma2_groups[kind]:
                    by_path.setdefault(pi, {})["sigma2"] = v
            elif i >= 0:
                by_path.setdefault(i, {})[kind] = v
            elif i == -2:
                by_path.setdefault(-2, {})["s02"] = v
            elif i == -1:
                by_path.setdefault(-1, {})["e0"] = v
        # 派生参数注入 by_path，后续统一按“自由变量”写回路径。
        for tgt in set(link_deltar) | set(con_dr):
            by_path.setdefault(tgt, {})["deltar"] = _dr_of(x, tgt)
        for tgt in link_sigma2:
            by_path.setdefault(tgt, {})["sigma2"] = _s2_of(x, tgt)
        for tgt in set(link_n) | set(con_n):
            by_path.setdefault(tgt, {})["n"] = _n_of(x, tgt)
        s02_val = by_path.get(-2, {}).get("s02", s02)
        alpha_val = (float(x[slot_index[(-5, "alpha")]])
                     if use_iso_deltar else None)
        w = softmax_weights(x) if vary_weights else None
        for i, path in enumerate(paths):
            v = by_path.get(i, {})
            if w is not None:
                # 相分数拟合：有效简并度 = 晶体学 N × 组分权重。
                path.degen = float(crystal_degen[i]
                                   * w[int(weight_groups[i])])
            elif "n" in v:
                path.degen = v["n"]
            if alpha_val is not None and i not in ms_path_set:
                dr_value = alpha_val * float(path.reff)
            else:
                dr_value = v.get("deltar", 0.0)
            set_path_parameters(
                path, s02=s02_val,
                delta_r=dr_value,
                sigma2=v.get("sigma2", sigma2_init),
                delta_e0=by_path.get(-1, {}).get("e0", 0.0),
                third=v.get("c3"), fourth=v.get("c4"))
        # 受约束 MS：ΔR 按路径长度比例、σ² 按 sigma2_ratio（同原子往返四腿路径≈4×，其余≈2×）、N 按母壳层有效简并度缩放。
        for idx, (parent, reff_ratio, degen_ratio, s2_ratio) in ms_by_path.items():
            if use_iso_deltar:
                dr_p = float(alpha_val) * float(paths[parent].reff)
            else:
                dr_p = _dr_of(x, parent)
            if parent in sigma2_group_of_path:
                s2_key = (-4, sigma2_group_of_path[parent])
                s2_p = (float(x[slot_index[s2_key]])
                        if s2_key in slot_index else float(sigma2_init))
            elif (parent, "sigma2") in slot_index:
                s2_p = float(x[slot_index[(parent, "sigma2")]])
            else:
                s2_p = float(sigma2_init)
            if vary_weights:
                n_parent = float(crystal_degen[parent]
                                 * w[int(weight_groups[parent])])
            else:
                n_parent = _n_of(x, parent)
            paths[idx].degen = float(degen_ratio * n_parent)
            set_path_parameters(
                paths[idx], s02=s02_val, delta_r=dr_p * reff_ratio,
                sigma2=s2_p * s2_ratio,
                delta_e0=by_path.get(-1, {}).get("e0", 0.0), third=None)

    def _ft_model(model_k, w):
        """把模型 χ(k) 用权重 w 变到 R 网格并插值到数据 R 轴。"""
        model_ft = athena.ft_to_r_space(ft_k, model_k, None, kmin, kmax,
                                        w, ft.dk)
        return np.interp(r_grid, model_ft.r, model_ft.chi_r.real) \
            + 1j * np.interp(r_grid, model_ft.r, model_ft.chi_r.imag)

    def model_chi_r(w=None):
        """单权兼容入口：返回主权重的模型 χ(R)。"""
        return _ft_model(exafs_model_k(paths, ft_k),
                         primary_kw if w is None else w)

    # ---- B11-2 BVS 软约束配置 ----
    bvs_cfg = None
    if bvs_restraint is not None:
        if not isinstance(bvs_restraint, dict):
            raise XAFSConfigError("FIT", "bvs_restraint 必须是字典")
        try:
            bvs_target_val = float(bvs_restraint["target"])
            bvs_sigma_val = float(bvs_restraint.get("sigma", 0.1))
        except (KeyError, TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", "bvs_restraint 需要 target 与 sigma（正数）") from exc
        if not np.isfinite(bvs_target_val):
            raise XAFSConfigError("FIT", "bvs_restraint.target 非有限")
        if not np.isfinite(bvs_sigma_val) or bvs_sigma_val <= 0:
            raise XAFSConfigError("FIT", "bvs_restraint.sigma 必须 > 0")
        bvs_cfg = {
            "target": bvs_target_val,
            "sigma": bvs_sigma_val,
            "r0_table": bvs_restraint.get("r0_table"),
            "absorber": bvs_restraint.get("absorber"),
            "b": float(bvs_restraint.get("b", 0.37)),
        }

    def _current_bvs():
        rows = []
        for i, path in enumerate(paths):
            label = (path_labels[i] if path_labels is not None
                     else Path(datfiles[i]).name)
            # ΔR 必须读 params 里的当前值：larch 的 set_path_parameters
            # 只改 params、不同步 path.deltar 属性（实测恒为 0）。
            # 若用属性，BVS 会在 Reff 处计算，约束与最终 R 脱节。
            try:
                dr_now = float(
                    path.params[path.pathpar_name("deltar")].value)
            except (AttributeError, KeyError):
                dr_now = float(getattr(path, "deltar", 0.0) or 0.0)
            rows.append({
                "label": label,
                "N": float(path.degen),
                "R": float(path.reff) + dr_now,
                "ligand": _path_ligand(label, bvs_cfg["absorber"]),
            })
        return bond_valence_sum(
            rows, r0_table=bvs_cfg["r0_table"], b=bvs_cfg["b"],
            absorber=bvs_cfg["absorber"])

    def residual(x):
        apply_params(x)
        model_k = exafs_model_k(paths, ft_k)
        parts = []
        for w in kw_tuple:
            if fit_space == "r":
                model_w = _ft_model(model_k, w)
                data_w = data_r_by_kw[w]
                dr = (model_w.real - data_w.real)[mask]
                di = (model_w.imag - data_w.imag)[mask]
                if multi_kw:
                    dr = dr / scale_by_kw[w]
                    di = di / scale_by_kw[w]
            elif fit_space == "q":
                model_w = _model_in_space(model_k, w)
                data_w = data_space_by_kw[w][1]
                mq = space_mask_by_kw[w]
                dr = (model_w.real - data_w.real)[mq] / space_scale_by_kw[w]
                di = (model_w.imag - data_w.imag)[mq] / space_scale_by_kw[w]
            else:
                mk, mr = space_mask_by_kw[w]
                model_full = _model_in_space(model_k, w)
                model_w = model_full[np.ix_(mk, mr)]
                data_w = data_space_by_kw[w][2]
                dr = ((model_w.real - data_w.real)
                      / space_scale_by_kw[w]).ravel()
                di = ((model_w.imag - data_w.imag)
                      / space_scale_by_kw[w]).ravel()
            parts.append(dr)
            parts.append(di)
        if bvs_cfg is not None:
            bvs_now = _current_bvs()
            parts.append(np.asarray(
                [(bvs_now - bvs_cfg["target"]) / bvs_cfg["sigma"]]))
        return np.concatenate(parts)

    solution = least_squares(residual, x0, bounds=(lower, upper))

    dof = max(len(solution.fun) - n_var, 1)
    # 直接对 J 做 SVD 求逆，避免 J^T J 将条件数平方引起病态误差放大。
    _, s_jac, vt_jac = np.linalg.svd(solution.jac, full_matrices=False)
    sv_tol = (float(s_jac.max()) * max(solution.jac.shape) * np.finfo(float).eps
              if s_jac.size else 0.0)
    keep_sv = s_jac > sv_tol
    if np.any(keep_sv):
        j_inv_t = vt_jac[keep_sv].T / s_jac[keep_sv]
        covariance = (j_inv_t @ j_inv_t.T) * (2 * solution.cost / dof)
    else:
        covariance = np.zeros((n_var, n_var), dtype=float)
    errors = np.sqrt(np.abs(np.diag(covariance)))
    # 相关矩阵：cov_ij / (σ_i σ_j)。变量数可到几十，JSON 里直接存完整
    # 矩阵（报告层只摘最强相关对）；零误差槽位不产生 NaN。
    denom = np.outer(errors, errors)
    correlation = np.divide(covariance, denom,
                            out=np.zeros_like(covariance), where=denom > 0)
    np.fill_diagonal(correlation, 1.0)
    # 审查 item 5：检测雅可比秩亏。满秩时 covariance/errors/correlation
    # 逐位不变；秩亏（N–σ²–ΔE0 精确共线、简并路径等）时 pinv 把零空间
    # 方向方差静默置 0 → 虚假小误差。此处只标记，供 base_reliable 降级。
    jac_rank = int(np.sum(keep_sv))

    # ---- mixture 权重与误差（delta method，softmax 雅可比）----
    mixture_weights = None
    w_final = None
    w_err_final = None
    if vary_weights:
        w_final = softmax_weights(solution.x)
        cov_pp = covariance[np.ix_(weight_slot_ids, weight_slot_ids)]
        # G[k, j] = ∂w_k/∂p_j；参考组分 logit 固定为 0（第 K 行无自由列）
        G = np.zeros((n_group, n_group - 1), dtype=float)
        for k in range(n_group):
            for j in range(n_group - 1):
                G[k, j] = w_final[k] * ((1.0 if k == j else 0.0) - w_final[j])
        cov_w = G @ cov_pp @ G.T
        w_err_final = np.sqrt(np.abs(np.diag(cov_w)))
        mixture_weights = [
            {"group": weight_names[k], "weight": float(w_final[k]),
             "weight_err": float(w_err_final[k])}
            for k in range(n_group)]

    # ---- 预计算 σ² 共享组的最优值与误差 ----
    s2_group_values = {}
    s2_group_errors = {}
    if sigma2_groups:
        for gi in range(n_s2_group):
            slot_key = (-4, gi)
            if slot_key in slot_index:
                s2_group_values[gi] = float(
                    solution.x[slot_index[slot_key]])
                s2_group_errors[gi] = float(
                    errors[slot_index[slot_key]])

    # 先写回最优解，MS 路径的缩放值也才是最终值。
    apply_params(solution.x)
    alpha_val = (float(solution.x[slot_index[(-5, "alpha")]])
                 if use_iso_deltar else None)
    alpha_err = (float(errors[slot_index[(-5, "alpha")]])
                 if use_iso_deltar else None)
    # ---- B13-2/B14-1 输出层：派生参数误差沿同一链传播 ----
    def _dr_err_of(key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"ΔR 误差链成环: 路径 {key}")
        seen.add(key)
        if key in link_deltar:
            return _dr_err_of(int(link_deltar[key]), seen)
        if key in con_dr:
            rec = con_dr[key]
            src_err = _dr_err_of(int(rec["source_index"]), seen)
            if src_err is None:
                return None
            # 比例约束的误差按 |ratio| 线性传播；offset 约束不改变标度。
            scale = (abs(float(rec["ratio"]))
                     if rec["kind"] == "deltar_ratio" else 1.0)
            return float(scale * src_err)
        sk = (int(key), "deltar")
        return float(errors[slot_index[sk]]) if sk in slot_index else None

    def _s2_err_of(key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"σ² 误差链成环: 路径 {key}")
        seen.add(key)
        if key in link_sigma2:
            return _s2_err_of(int(link_sigma2[key]), seen)
        if key in sigma2_group_of_path:
            gi = sigma2_group_of_path[key]
            return s2_group_errors.get(gi)
        sk = (int(key), "sigma2")
        return float(errors[slot_index[sk]]) if sk in slot_index else None

    def _n_err_of(key, seen=None):
        seen = set() if seen is None else seen
        if key in seen:
            raise XAFSConfigError("FIT", f"N 误差链成环: 路径 {key}")
        seen.add(key)
        if vary_weights:
            return float(crystal_degen[key]
                         * w_err_final[int(weight_groups[key])])
        if key in link_n:
            src, ratio = link_n[key]
            e = _n_err_of(int(src), seen)
            return None if e is None else float(ratio) * e
        if key in con_n:
            rec = con_n[key]
            e = _n_err_of(int(rec["source_index"]), seen)
            if e is None:
                return None
            coeff = (float(rec["coeff"]) if rec["kind"] == "n_sum"
                     else float(rec["ratio"]))
            return abs(coeff) * e
        sk = (int(key), "n")
        return float(errors[slot_index[sk]]) if sk in slot_index else None

    fit_paths_list = []
    for i, path in enumerate(paths):
        if i in ms_by_path:
            parent, reff_ratio, degen_ratio, s2_ratio = ms_by_path[i]
            if use_iso_deltar:
                dr_p = float(alpha_val) * float(paths[parent].reff)
                dr_err_p = float(alpha_err) * float(paths[parent].reff)
            else:
                dr_p = _dr_of(solution.x, parent)
                dr_err_p = _dr_err_of(parent)
            s2_p = _s2_of(solution.x, parent)
            s2_err_p = _s2_err_of(parent)
            n_err_p = _n_err_of(parent)
            fit_paths_list.append(PathFit(
                label=(path_labels[i] if path_labels is not None
                       else Path(datfiles[i]).name),
                reff=float(path.reff),
                degeneracy=float(path.degen),
                delta_r=float(dr_p * reff_ratio),
                delta_r_err=(0.0 if dr_err_p is None
                             else float(dr_err_p * reff_ratio)),
                sigma2=float(s2_p * s2_ratio),
                sigma2_err=(None if s2_err_p is None
                            else float(s2_err_p * s2_ratio)),
                n_err=(None if n_err_p is None
                       else float(n_err_p * degen_ratio)),
                sigma2_at_bound=bool(s2_p * s2_ratio <= 5e-4),
                nleg=int(getattr(path, "nleg", 2)),
                ms_parent=(path_labels[parent]
                           if path_labels is not None else str(parent))))
            continue
        if i in sigma2_group_of_path or i in link_sigma2 \
                or (i, "sigma2") in slot_index:
            s2_val = _s2_of(solution.x, i)
            s2_err = _s2_err_of(i)
            s2_at_bound = s2_val <= 5e-4
        else:
            s2_val = float(sigma2_init)
            s2_err = None
            s2_at_bound = False
        c3_val = (float(solution.x[slot_index[(i, "c3")]])
                  if i in c3_paths else None)
        c3_err_val = (float(errors[slot_index[(i, "c3")]])
                      if i in c3_paths else None)
        c4_val = (float(solution.x[slot_index[(i, "c4")]])
                  if i in c4_paths else None)
        c4_err_val = (float(errors[slot_index[(i, "c4")]])
                      if i in c4_paths else None)
        if use_iso_deltar:
            dr_val = float(alpha_val) * float(paths[i].reff)
            dr_err_val = float(alpha_err) * float(paths[i].reff)
        else:
            dr_val = _dr_of(solution.x, i)
            dr_err_val = _dr_err_of(i)
        fit_paths_list.append(PathFit(
            label=(path_labels[i] if path_labels is not None
                   else Path(datfiles[i]).name),
            reff=float(path.reff),
            # 权重模式：有效简并度 N_crystal·w（不是配位数）；普通模式：拟合 N
            degeneracy=(float(crystal_degen[i] * w_final[int(weight_groups[i])])
                        if vary_weights else float(path.degen)),
            delta_r=dr_val,
            delta_r_err=dr_err_val,
            sigma2=float(s2_val),
            sigma2_err=s2_err,
            n_err=((float(crystal_degen[i]
                          * w_err_final[int(weight_groups[i])]))
                   if vary_weights else _n_err_of(i)),
            c3=c3_val,
            c3_err=c3_err_val,
            c3_at_bound=bool(c3_val is not None
                             and abs(c3_val) >= 0.8 * C3_LIMIT),
            c4=c4_val,
            c4_err=c4_err_val,
            c4_at_bound=bool(c4_val is not None
                             and abs(c4_val) >= 0.8 * C4_LIMIT),
            sigma2_at_bound=s2_at_bound))

    # R-factor 在最优解处重算：必须给**所有**路径都设上最优参数。
    # 旧版只设了 paths[0]，多路径时其余路径仍是残差函数最后一次迭代的值，
    # 导致双壳层拟合的 R-factor 不可信。
    apply_params(solution.x)
    model_k_final = exafs_model_k(paths, ft_k)

    def _space_rfactor(w):
        if fit_space == "r":
            return float(r_factor(data_r_by_kw[w],
                                  _ft_model(model_k_final, w), mask))
        if fit_space == "q":
            mq = space_mask_by_kw[w]
            data_w = data_space_by_kw[w][1][mq]
            model_w = _model_in_space(model_k_final, w)[mq]
            ones = np.ones(mq.sum(), dtype=bool)
            return float(r_factor(data_w, model_w, ones))
        mk, mr = space_mask_by_kw[w]
        data_w = data_space_by_kw[w][2].ravel()
        model_w = _model_in_space(model_k_final, w)[np.ix_(mk, mr)].ravel()
        ones = np.ones(data_w.size, dtype=bool)
        return float(r_factor(data_w, model_w, ones))

    r_factors_by_kweight = {
        int(w): _space_rfactor(w) for w in kw_tuple}

    # ---- B11-2 BVS 最终值与约束 χ² ----
    bvs_value = None
    bvs_target_out = None
    bvs_chi2_out = None
    if bvs_cfg is not None:
        bvs_value = float(_current_bvs())
        bvs_target_out = float(bvs_cfg["target"])
        bvs_chi2_out = float(
            ((bvs_value - bvs_target_out) / bvs_cfg["sigma"]) ** 2)

    # ---- B12-1 AIC / BIC（以数据残差 RSS 计，不含 BVS 约束行） ----
    data_fun = (solution.fun[:-1] if bvs_cfg is not None
                else solution.fun)
    rss = float(np.sum(data_fun ** 2))
    n_obs = int(data_fun.size)
    aic = bic = None
    if n_obs > 0 and rss > 0:
        aic = float(n_obs * np.log(rss / n_obs) + 2.0 * n_var)
        bic = float(n_obs * np.log(rss / n_obs) + n_var * np.log(n_obs))

    # ---- B4-1 统计量：高 R 噪声 ε_R、χ²、χ²_ν ----
    # 用主 kweight 分支计算，与 r_factor 的汇报口径一致；多权时各分支
    # 共用 N_idp（权重不增加独立信息量，见 B1-1 契约）。
    # N_idp 与求解前 Stern 门禁同口径：按有效 k 窗，不用超出数据支持域的请求窗。
    n_idp = n_idp_eff
    epsilon_r = None
    chi_square = None
    reduced_chi_square = None
    try:
        epsilon_r = estimate_r_noise(
            chi_k, kmin, kmax, kweight=int(primary_kw), dk=float(ft.dk))
    except XAFSConfigError:
        epsilon_r = None
    if epsilon_r is not None and epsilon_r > 0:
        data_main = data_r_by_kw[int(primary_kw)]
        model_main = _ft_model(model_k_final, int(primary_kw))
        n_grid = int(mask.sum())
        diff2 = float(np.sum(np.abs(data_main[mask] - model_main[mask]) ** 2))
        chi_square = (n_idp * diff2
                      / (2.0 * max(n_grid, 1) * epsilon_r ** 2))
        reduced_chi_square = chi_square / max(n_idp - n_var, 1.0)

    nidp_scale_factor = float(np.sqrt(
        max(len(solution.fun) - n_var, 1) / max(n_idp_eff - n_var, 1)))
    return FitResult(
        paths=fit_paths_list,
        s02=(float(solution.x[slot_index[(-2, "s02")]])
             if vary_s02 else float(s02)),
        s02_err=(float(errors[slot_index[(-2, "s02")]]) if vary_s02 else None),
        delta_e0=float(solution.x[slot_index[(-1, "e0")]]) if vary_delta_e0 else 0.0,
        delta_e0_err=float(errors[slot_index[(-1, "e0")]]) if vary_delta_e0 else 0.0,
        r_factor=r_factors_by_kweight[int(primary_kw)],
        n_independent=float(n_idp),
        n_variables=n_var,
        epsilon_r=epsilon_r,
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        lattice_strain=alpha_val,
        lattice_strain_err=alpha_err,
        kmin=float(kmin), kmax=float(kmax),
        rmin=float(rmin), rmax=float(rmax),
        mixture_weights=mixture_weights,
        parameter_names=parameter_names,
        correlation=correlation,
        sigma2_groups=sigma2_groups,
        nidp_scale_factor=nidp_scale_factor,
        sigma2_group_names=sigma2_group_names,
        kweights=kw_tuple,
        r_factors_by_kweight=r_factors_by_kweight,
        model_r=np.asarray(r_grid, dtype=float),
        model_chi_r=np.asarray(
            _ft_model(model_k_final, primary_kw), dtype=complex),
        model_k=np.asarray(model_k_final, dtype=float),
        model_k_grid=np.asarray(ft_k, dtype=float),
        aic=aic, bic=bic,
        bvs=bvs_value, bvs_target=bvs_target_out,
        bvs_chi2=bvs_chi2_out,
        fit_space=fit_space,
        jac_rank=jac_rank, rank_deficient=(jac_rank < n_var))
