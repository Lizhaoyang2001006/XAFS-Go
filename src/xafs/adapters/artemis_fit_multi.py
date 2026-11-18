# -*- coding: utf-8 -*-
"""artemis_fit_multi.py — #5b 多数据集全局受约束联合拟合（自 artemis.py 纯拆分）

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
    HBAR2_OVER_2KB, correlated_debye_sigma2, einstein_sigma2,
    exafs_model_k, independent_points, r_factor, set_path_parameters,
)


def fit_multi_datasets(
    datasets: list[dict],
    shared_bonds: list[dict] | None = None,
    einstein_models: list[dict] | None = None,
    share_e0="auto",
    debye_models: list[dict] | None = None,
) -> dict:
    """多数据集全局受约束联合拟合（Co-refinement）。

    支持三类痛点:
      1. 双金属双吸收边（如 Pt L3 + Co K）：用 ``shared_bonds`` 强制
         异核键满足 R_A-B = R_B-A、σ²_A-B = σ²_B-A；
      2. 变温多谱（如 80 K + 350 K）：用 ``einstein_models`` 以
         σ²_static 与 Θ_E 两个全局参数驱动每个温度下的 σ²(T)；
      3. 关联 Debye 模型（B14-3）：用 ``debye_models`` 以
         (σ²_static, Θ_D) 驱动同一温度下**随 R_eff 变化**的 σ²(T,R)，
         近壳运动相关压低、远壳回到非关联值。

    数学:
        每个数据集 d 在自身 mask_d 内的实/虚部残差除以该数据集实验谱
        的 RMS 范数 s_d = sqrt(mean(|χ_data,d(R)|²))，再拼接成全局残差：
            min Σ_d ||(χ_model,d − χ_data,d)/s_d||²
        这样 80 K 与 350 K 的振幅差异不会让某一个数据集统治最小二乘。

    参数:
        datasets : 每项
            {"name", "chi_k", "ft", "datfiles", "rmin", "rmax",
             "kmin", "kmax", "kweight", "s02", "vary_n",
             "path_degen", "path_labels", "temperature_K"}
        shared_bonds : 每项
            {"target": (ds_b, path_b), "source": (ds_a, path_a),
             "share_deltar": bool, "share_sigma2": bool,
             "degen_ratio": float | None}
            目标路径不建独立 ΔR/σ²/N 槽位，复用源路径；N 按
            degen_ratio 缩放（None 表示不共享 N）。
        einstein_models : 每项
            {"paths": [(ds, path), ...], "reduced_mass_amu": float,
             "theta_e_init": float = 350.0}
            组内路径的 σ² 全部由 (σ²_static, Θ_E) 两个全局变量按
            各自数据集温度计算。
        debye_models : 每项
            {"paths": [(ds, path), ...], "mass_absorber": float,
             "mass_scatterer": float, "theta_d_init": float = 400.0,
             "k_debye": float | None}
            组内路径的 σ² 由 (σ²_static, Θ_D) 按
            ``correlated_debye_sigma2(T, Θ_D, R_eff, m_a, m_s)`` 实时
            计算；与 einstein/share_sigma2 对同一路径互斥。

    返回:
        {"datasets": [FitResult, ...], "n_independent_total": float,
         "n_variables_total": int, "global_r_factor": float,
         "einstein_results": [{"theta_e", "theta_e_err",
                               "sigma2_static", "sigma2_static_err"}, ...],
         "debye_results": [{"theta_d", "theta_d_err",
                            "sigma2_static", "sigma2_static_err",
                            "paths", ...}, ...],
         "parameter_names": list, "correlation": np.ndarray}

    注意:
        ``share_e0`` 控制 ΔE₀ 的分组（B12-1）：
        True 所有数据集共享一个 ΔE₀（历史行为）；False 每个数据集独立；
        "auto"（默认）按 ``datasets[di]["edge_group"]`` 分组，缺省用
        ``edge`` 字符串，再缺省全部共享。同组共享一个 ΔE₀。
        双吸收边场景应让不同边落在不同 edge_group，而不是硬凑一个
        无物理意义的共同值。
    """
    from scipy.optimize import least_squares

    if not datasets:
        raise XAFSConfigError("FIT", "datasets 不能为空")
    n_ds = len(datasets)
    required = ("chi_k", "ft", "datfiles", "rmin", "rmax",
                "kmin", "kmax", "kweight")
    for di, d in enumerate(datasets):
        for key in required:
            if d.get(key) is None:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 缺少字段 {key!r}")
        if not d["datfiles"]:
            raise XAFSConfigError("FIT", f"数据集 {di} 的 datfiles 为空")
        if d.get("s02") is None:
            raise XAFSConfigError(
                "FIT", f"数据集 {di} 未提供 s02；S₀² 必须显式给定")

    # ---- 载入路径并冻结晶体学简并度 ----
    ds_paths: list[list] = []
    crystal_degen: list[list[float]] = []
    for di, d in enumerate(datasets):
        paths = [load_feff_path(f) for f in d["datfiles"]]
        degs = d.get("path_degen")
        if degs is not None:
            if len(degs) != len(paths):
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 的 path_degen 长度 {len(degs)} "
                           f"!= 路径数 {len(paths)}")
            for p, deg in zip(paths, degs):
                p.degen = float(deg)
        labels = d.get("path_labels")
        if labels is not None and len(labels) != len(paths):
            raise XAFSConfigError(
                "FIT", f"数据集 {di} 的 path_labels 长度 {len(labels)} "
                       f"!= 路径数 {len(paths)}")
        ds_paths.append(paths)
        crystal_degen.append([float(p.degen) for p in paths])

    def _check_path(di, pi, what):
        if di < 0 or di >= n_ds:
            raise XAFSConfigError(
                "FIT", f"{what} 的数据集下标 {di} 超出 0–{n_ds - 1}")
        if pi < 0 or pi >= len(ds_paths[di]):
            raise XAFSConfigError(
                "FIT", f"{what} 的路径下标 {pi} 超出数据集 {di} 的 "
                       f"0–{len(ds_paths[di]) - 1}")

    # ---- 跨数据集路径绑定 ----
    bound_deltar: dict[tuple, tuple] = {}
    bound_sigma2: dict[tuple, tuple] = {}
    bound_n: dict[tuple, tuple] = {}
    for bi, bond in enumerate(shared_bonds or []):
        try:
            tgt = tuple(int(v) for v in bond["target"])
            src = tuple(int(v) for v in bond["source"])
        except (KeyError, TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", f"shared_bonds[{bi}] 的 target/source 必须是 "
                       f"(数据集下标, 路径下标)") from exc
        if len(tgt) != 2 or len(src) != 2:
            raise XAFSConfigError(
                "FIT", f"shared_bonds[{bi}] 的 target/source 必须各含 2 个整数")
        _check_path(tgt[0], tgt[1], f"shared_bonds[{bi}].target")
        _check_path(src[0], src[1], f"shared_bonds[{bi}].source")
        if tgt == src:
            raise XAFSConfigError(
                "FIT", f"shared_bonds[{bi}] 的 target 与 source 相同；"
                       f"自绑定不产生约束")
        if bond.get("share_deltar"):
            if tgt in bound_deltar:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多个 share_deltar 绑定")
            bound_deltar[tgt] = src
        if bond.get("share_sigma2"):
            if tgt in bound_sigma2:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多个 share_sigma2 绑定")
            bound_sigma2[tgt] = src
        ratio = bond.get("degen_ratio")
        if ratio is not None:
            ratio = float(ratio)
            if not np.isfinite(ratio) or ratio <= 0:
                raise XAFSConfigError(
                    "FIT", f"shared_bonds[{bi}].degen_ratio={ratio!r} "
                           f"必须为正")
            if tgt in bound_n:
                raise XAFSConfigError(
                    "FIT", f"路径 {tgt} 被多个 degen_ratio 绑定")
            bound_n[tgt] = (src, ratio)

    # ---- 爱因斯坦变温组 ----
    einstein_groups = list(einstein_models or [])
    einstein_of: dict[tuple, int] = {}
    for gi, model in enumerate(einstein_groups):
        try:
            mu = float(model["reduced_mass_amu"])
            plist = model["paths"]
        except (KeyError, TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", f"einstein_models[{gi}] 缺少 reduced_mass_amu/paths") from exc
        if not np.isfinite(mu) or mu <= 0:
            raise XAFSConfigError(
                "FIT", f"einstein_models[{gi}].reduced_mass_amu={mu!r} "
                       f"必须 > 0")
        if not plist:
            raise XAFSConfigError(
                "FIT", f"einstein_models[{gi}].paths 不能为空")
        for item in plist:
            if len(item) != 2:
                raise XAFSConfigError(
                    "FIT", f"einstein_models[{gi}].paths 每项必须是 "
                           f"(数据集下标, 路径下标)")
            di, pi = int(item[0]), int(item[1])
            _check_path(di, pi, f"einstein_models[{gi}].paths")
            key = (di, pi)
            if key in einstein_of:
                raise XAFSConfigError(
                    "FIT", f"路径 {key} 出现在多个爱因斯坦组中")
            if key in bound_sigma2:
                raise XAFSConfigError(
                    "FIT", f"路径 {key} 同时被 share_sigma2 与爱因斯坦模型约束")
            temp = datasets[di].get("temperature_K")
            if temp is None:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 参与爱因斯坦模型但未给 temperature_K")
            if float(temp) <= 0:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 的 temperature_K={temp!r} 必须 > 0")
            einstein_of[key] = gi

    # ---- 关联 Debye 变温组（B14-3）----
    # 与 einstein_models 的区别：爱因斯坦模型只按温度给出 σ²(T)，
    # 不区分键长；Debye 模型用关联因子 C(R)=1−sin(2k_D R)/(2k_D R)
    # 让近壳运动强相关（σ² 小）、远壳回到非关联值（σ² 大）。六方
    # 密堆/岩盐等固体中同一 Θ_D 必须同时约束近壳与远壳。
    debye_groups = list(debye_models or [])
    debye_of: dict[tuple, int] = {}
    for gi, model in enumerate(debye_groups):
        try:
            m_abs = float(model["mass_absorber"])
            m_sc = float(model["mass_scatterer"])
            plist = model["paths"]
        except (KeyError, TypeError, ValueError) as exc:
            raise XAFSConfigError(
                "FIT", f"debye_models[{gi}] 缺少 "
                       f"mass_absorber/mass_scatterer/paths") from exc
        if not (np.isfinite(m_abs) and m_abs > 0):
            raise XAFSConfigError(
                "FIT", f"debye_models[{gi}].mass_absorber={m_abs!r} 必须 > 0")
        if not (np.isfinite(m_sc) and m_sc > 0):
            raise XAFSConfigError(
                "FIT", f"debye_models[{gi}].mass_scatterer={m_sc!r} 必须 > 0")
        if not plist:
            raise XAFSConfigError(
                "FIT", f"debye_models[{gi}].paths 不能为空")
        for item in plist:
            if len(item) != 2:
                raise XAFSConfigError(
                    "FIT", f"debye_models[{gi}].paths 每项必须是 "
                           f"(数据集下标, 路径下标)")
            di, pi = int(item[0]), int(item[1])
            _check_path(di, pi, f"debye_models[{gi}].paths")
            key = (di, pi)
            if key in debye_of:
                raise XAFSConfigError(
                    "FIT", f"路径 {key} 出现在多个 Debye 组中")
            if key in einstein_of:
                raise XAFSConfigError(
                    "FIT", f"路径 {key} 同时被 Debye 与爱因斯坦模型约束")
            if key in bound_sigma2:
                raise XAFSConfigError(
                    "FIT", f"路径 {key} 同时被 share_sigma2 与 Debye 模型约束")
            temp = datasets[di].get("temperature_K")
            if temp is None:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 参与 Debye 模型但未给 temperature_K")
            if float(temp) <= 0:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 的 temperature_K={temp!r} 必须 > 0")
            debye_of[key] = gi

    # ---- 每数据集的 N 变量集合 ----
    ds_n_paths: list[set[int]] = []
    for di, d in enumerate(datasets):
        vn = d.get("vary_n", False)
        n_path = len(ds_paths[di])
        if vn is True:
            nset = set(range(n_path))
        elif vn:
            nset = {int(i) for i in vn}
            bad = nset - set(range(n_path))
            if bad:
                raise XAFSConfigError(
                    "FIT", f"数据集 {di} 的 vary_n 索引 {sorted(bad)} "
                           f"超出 0–{n_path - 1}")
        else:
            nset = set()
        ds_n_paths.append(nset)

    # ---- B12-1 ΔE₀ 分组（share_e0）----
    if share_e0 is True:
        e0_raw_groups = ["global"] * n_ds
    elif share_e0 is False:
        e0_raw_groups = [f"ds{di}" for di in range(n_ds)]
    elif str(share_e0).lower() == "auto":
        e0_raw_groups = []
        for di, d in enumerate(datasets):
            key = d.get("edge_group")
            if key is None:
                key = d.get("edge")
            e0_raw_groups.append("global" if key is None else str(key))
        if all(k == "global" for k in e0_raw_groups):
            e0_raw_groups = ["global"] * n_ds
    else:
        raise XAFSConfigError(
            "FIT", f"share_e0={share_e0!r} 只支持 True / False / 'auto'")
    e0_group_order = []
    for key in e0_raw_groups:
        if key not in e0_group_order:
            e0_group_order.append(key)
    e0_group_index = {key: gi for gi, key in enumerate(e0_group_order)}
    e0_group_of_ds = [e0_group_index[key] for key in e0_raw_groups]
    n_e0_groups = len(e0_group_order)

    def e0_slot_key(gi):
        # 单组时沿用历史键，保证旧回归的参数名/数值逐位不变。
        return ("global", "e0") if n_e0_groups == 1 else ("e0grp", gi)

    # ---- 全局变量槽位 ----
    slots: list[tuple] = []
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_slot(key, init, lo, hi):
        slots.append(key)
        x0.append(float(init))
        lower.append(float(lo))
        upper.append(float(hi))

    for di in range(n_ds):
        for pi in range(len(ds_paths[di])):
            key = (di, pi)
            if key not in bound_deltar:
                add_slot((di, pi, "deltar"), 0.0, -0.3, 0.3)
            if key not in bound_sigma2 and key not in einstein_of \
                    and key not in debye_of:
                add_slot((di, pi, "sigma2"), 0.005, 1e-4, 0.04)
            if pi in ds_n_paths[di] and key not in bound_n:
                deg = crystal_degen[di][pi]
                if deg <= 0:
                    raise XAFSConfigError(
                        "FIT", f"数据集 {di} 路径 {pi} 晶体学简并度={deg} ≤ 0，"
                               f"无法放开其 N 做配位数定量（零/负占据壳无 CN）")
                add_slot((di, pi, "n"), deg,
                         max(0.5, 0.1 * deg), 3.0 * deg)
    for gi, model in enumerate(einstein_groups):
        add_slot(("einstein", gi, "sigma2_static"), 0.001, 0.0, 0.015)
        te0 = float(model.get("theta_e_init", 350.0))
        if not np.isfinite(te0) or not (80.0 <= te0 <= 1200.0):
            raise XAFSConfigError(
                "FIT", f"einstein_models[{gi}].theta_e_init={te0!r} "
                       f"必须在 [80.0, 1200.0] K 内")
        add_slot(("einstein", gi, "theta_e"), te0, 80.0, 1200.0)
    for gi, model in enumerate(debye_groups):
        add_slot(("debye", gi, "sigma2_static"), 0.001, 0.0, 0.015)
        td0 = float(model.get("theta_d_init", 400.0))
        if not np.isfinite(td0) or not (80.0 <= td0 <= 1200.0):
            raise XAFSConfigError(
                "FIT", f"debye_models[{gi}].theta_d_init={td0!r} "
                       f"必须在 [80.0, 1200.0] K 内")
        add_slot(("debye", gi, "theta_d"), td0, 80.0, 1200.0)
    for gi in range(n_e0_groups):
        add_slot(e0_slot_key(gi), 0.0, -15.0, 15.0)

    x0_a = np.asarray(x0, dtype=float)
    lower_a = np.asarray(lower, dtype=float)
    upper_a = np.asarray(upper, dtype=float)
    slot_index = {s: i for i, s in enumerate(slots)}
    n_var = len(slots)

    # ---- Stern 硬门禁：求解前校验全局 N_var < Σ N_idp（按各数据集有效 k 窗）----
    _ds_n_idp_eff = []
    for di, d in enumerate(datasets):
        kmin_d, kmax_d = float(d["kmin"]), float(d["kmax"])
        rmin_d, rmax_d = float(d["rmin"]), float(d["rmax"])
        if not (np.isfinite(kmin_d) and np.isfinite(kmax_d) and kmin_d < kmax_d):
            raise XAFSConfigError("FIT", f"数据集 {di} 的 k 窗非法: [{kmin_d}, {kmax_d}]")
        if not (np.isfinite(rmin_d) and np.isfinite(rmax_d) and rmin_d < rmax_d):
            raise XAFSConfigError("FIT", f"数据集 {di} 的 R 窗非法: [{rmin_d}, {rmax_d}]")
        kw_d = d["kweight"]
        if isinstance(kw_d, (list, tuple, np.ndarray)) or int(kw_d) < 0:
            raise XAFSConfigError("FIT", f"数据集 {di} 的 kweight={kw_d!r} 必须为非负整数")
        _kd = np.asarray(d["chi_k"].k, dtype=float)
        _klo = float(max(float(d["kmin"]), _kd.min()))
        _khi = float(min(float(d["kmax"]), _kd.max()))
        if not (_khi > _klo):
            raise XAFSConfigError(
                "FIT", f"数据集 {di} 有效 k 窗为空: 请求 "
                       f"[{d['kmin']},{d['kmax']}] 与数据支持域无交集")
        _ds_n_idp_eff.append(independent_points(
            _klo, _khi, float(d["rmin"]), float(d["rmax"])))
    _n_idp_total_eff = float(sum(_ds_n_idp_eff))
    if n_var >= _n_idp_total_eff:
        raise XAFSConfigError(
            "FIT", f"联合拟合 N_var={n_var} ≥ N_idp_total="
                   f"{_n_idp_total_eff:.2f}；违反 Stern 规则，拒绝求解")

    # ---- 槽位解析（支持绑定链） ----
    def _slot_value(x, key):
        return float(x[slot_index[key]])

    def resolved_dr(x, key):
        seen = set()
        k = key
        while k in bound_deltar:
            if k in seen:
                raise XAFSConfigError(
                    "FIT", f"share_deltar 绑定成环: {k}")
            seen.add(k)
            k = bound_deltar[k]
        return _slot_value(x, (k[0], k[1], "deltar"))

    def resolved_sigma2(x, key):
        seen = set()
        k = key
        while True:
            if k in einstein_of:
                gi = einstein_of[k]
                T = float(datasets[k[0]]["temperature_K"])
                mu = float(einstein_groups[gi]["reduced_mass_amu"])
                return einstein_sigma2(
                    T, _slot_value(x, ("einstein", gi, "theta_e")), mu,
                    _slot_value(x, ("einstein", gi, "sigma2_static")))
            if k in debye_of:
                gi = debye_of[k]
                model = debye_groups[gi]
                T = float(datasets[k[0]]["temperature_K"])
                reff = float(ds_paths[k[0]][k[1]].reff)
                return correlated_debye_sigma2(
                    T, _slot_value(x, ("debye", gi, "theta_d")), reff,
                    float(model["mass_absorber"]),
                    float(model["mass_scatterer"]),
                    k_debye=model.get("k_debye"),
                    sigma2_static=_slot_value(
                        x, ("debye", gi, "sigma2_static")))
            if k in bound_sigma2:
                if k in seen:
                    raise XAFSConfigError(
                        "FIT", f"share_sigma2 绑定成环: {k}")
                seen.add(k)
                k = bound_sigma2[k]
                continue
            return _slot_value(x, (k[0], k[1], "sigma2"))

    def resolved_n(x, key):
        seen = set()
        k = key
        factor = 1.0
        while k in bound_n:
            if k in seen:
                raise XAFSConfigError("FIT", f"degen_ratio 绑定成环: {k}")
            seen.add(k)
            k, ratio = bound_n[k][0], bound_n[k][1]
            factor *= ratio
        slot = (k[0], k[1], "n")
        base = (_slot_value(x, slot) if slot in slot_index
                else crystal_degen[k[0]][k[1]])
        return factor * base

    def apply_params(x):
        for di, paths in enumerate(ds_paths):
            e0 = _slot_value(x, e0_slot_key(e0_group_of_ds[di]))
            s02 = float(datasets[di]["s02"])
            for pi, path in enumerate(paths):
                key = (di, pi)
                path.degen = resolved_n(x, key)
                set_path_parameters(
                    path, s02=s02,
                    delta_r=resolved_dr(x, key),
                    sigma2=resolved_sigma2(x, key),
                    delta_e0=e0, third=None)

    # ---- 各数据集的实验谱、mask 与 RMS 尺度 ----
    ds_ft_k = [np.asarray(d["chi_k"].k, dtype=float) for d in datasets]
    ds_data_r = [np.asarray(d["ft"].chi_r, dtype=complex) for d in datasets]
    ds_r_grid = [np.asarray(d["ft"].r, dtype=float) for d in datasets]
    ds_mask = []
    ds_scale = []
    for di, d in enumerate(datasets):
        m = (ds_r_grid[di] >= float(d["rmin"])) & \
            (ds_r_grid[di] <= float(d["rmax"]))
        if m.sum() < 8:
            raise XAFSConfigError(
                "FIT", f"数据集 {di} 的 R 区间 "
                       f"[{d['rmin']},{d['rmax']}] 内仅 {m.sum()} 点")
        s = float(np.sqrt(np.mean(np.abs(ds_data_r[di][m]) ** 2)))
        ds_mask.append(m)
        ds_scale.append(s if np.isfinite(s) and s > 0 else 1.0)

    def model_chi_r(di):
        d = datasets[di]
        model_k = exafs_model_k(ds_paths[di], ds_ft_k[di])
        model_ft = athena.ft_to_r_space(
            ds_ft_k[di], model_k, None,
            float(d["kmin"]), float(d["kmax"]),
            int(d["kweight"]), float(getattr(d["ft"], "dk", 1.0)))
        return (np.interp(ds_r_grid[di], model_ft.r, model_ft.chi_r.real)
                + 1j * np.interp(ds_r_grid[di], model_ft.r,
                                 model_ft.chi_r.imag))

    def residual(x):
        apply_params(x)
        parts = []
        for di in range(n_ds):
            model = model_chi_r(di)
            data = ds_data_r[di]
            m = ds_mask[di]
            s = ds_scale[di]
            parts.append((model.real - data.real)[m] / s)
            parts.append((model.imag - data.imag)[m] / s)
        return np.concatenate(parts)

    solution = least_squares(residual, x0_a, bounds=(lower_a, upper_a))

    dof = max(len(solution.fun) - n_var, 1)
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
    denom = np.outer(errors, errors)
    correlation = np.divide(
        covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
    np.fill_diagonal(correlation, 1.0)
    # 审查 item 5：全局联合拟合的雅可比秩亏检测（同单数据集口径）。
    jac_rank = int(np.sum(keep_sv))

    # ---- 最优参数写回并计算各数据集 R-factor ----
    apply_params(solution.x)
    ds_rf = []
    for di in range(n_ds):
        model = model_chi_r(di)
        ds_rf.append(float(r_factor(ds_data_r[di], model, ds_mask[di])))
    scaled_res = residual(solution.x)
    den = 0.0
    for di in range(n_ds):
        m = ds_mask[di]
        den += float(np.sum(np.abs(ds_data_r[di][m]) ** 2)
                     / ds_scale[di] ** 2)
    global_rf = float(np.sum(scaled_res ** 2) / den) if den > 0 else np.inf

    # ---- 误差传播 ----
    def _slot_err(key):
        idx = slot_index.get(key)
        return None if idx is None else float(errors[idx])

    def resolved_dr_err(key):
        seen = set()
        k = key
        while k in bound_deltar:
            if k in seen:
                raise XAFSConfigError(
                    "FIT", f"share_deltar 绑定成环: {k}")
            seen.add(k)
            k = bound_deltar[k]
        return _slot_err((k[0], k[1], "deltar"))

    def resolved_sigma2_err(key):
        seen = set()
        k = key
        while True:
            if k in einstein_of:
                gi = einstein_of[k]
                i_static = slot_index[("einstein", gi, "sigma2_static")]
                i_theta = slot_index[("einstein", gi, "theta_e")]
                T = float(datasets[k[0]]["temperature_K"])
                mu = float(einstein_groups[gi]["reduced_mass_amu"])
                theta = float(solution.x[i_theta])
                xval = theta / (2.0 * T)
                coth = 1.0 / np.tanh(xval)
                csch2 = (1.0 / np.sinh(xval) ** 2
                         if xval < 350.0 else 0.0)
                a = HBAR2_OVER_2KB / mu
                d_theta = -a / theta ** 2 * (coth + xval * csch2)
                var = (errors[i_static] ** 2
                       + (d_theta * errors[i_theta]) ** 2
                       + 2.0 * d_theta * covariance[i_static, i_theta])
                return float(np.sqrt(abs(var)))
            if k in debye_of:
                gi = debye_of[k]
                model = debye_groups[gi]
                i_static = slot_index[("debye", gi, "sigma2_static")]
                i_theta = slot_index[("debye", gi, "theta_d")]
                T = float(datasets[k[0]]["temperature_K"])
                reff = float(ds_paths[k[0]][k[1]].reff)
                m_abs = float(model["mass_absorber"])
                m_sc = float(model["mass_scatterer"])
                kd = model.get("k_debye")
                theta = float(solution.x[i_theta])
                static = float(solution.x[i_static])
                f0 = correlated_debye_sigma2(
                    T, theta, reff, m_abs, m_sc, k_debye=kd,
                    sigma2_static=static)
                # 规范要求的有限差分步长：σ²_static 1e-4 Å²、Θ_D 0.5 K。
                d_s = 1e-4
                f_s = correlated_debye_sigma2(
                    T, theta, reff, m_abs, m_sc, k_debye=kd,
                    sigma2_static=static + d_s)
                d_static = (f_s - f0) / d_s
                d_t = 0.5
                f_t = correlated_debye_sigma2(
                    T, theta + d_t, reff, m_abs, m_sc, k_debye=kd,
                    sigma2_static=static)
                d_theta = (f_t - f0) / d_t
                var = (d_static ** 2 * errors[i_static] ** 2
                       + d_theta ** 2 * errors[i_theta] ** 2
                       + 2.0 * d_static * d_theta
                       * covariance[i_static, i_theta])
                return float(np.sqrt(abs(var)))
            if k in bound_sigma2:
                if k in seen:
                    raise XAFSConfigError(
                        "FIT", f"share_sigma2 绑定成环: {k}")
                seen.add(k)
                k = bound_sigma2[k]
                continue
            return _slot_err((k[0], k[1], "sigma2"))

    def resolved_n_err(key):
        seen = set()
        k = key
        factor = 1.0
        while k in bound_n:
            if k in seen:
                raise XAFSConfigError(
                    "FIT", f"degen_ratio 绑定成环: {k}")
            seen.add(k)
            src, ratio = bound_n[k]
            k = src
            factor *= ratio
        e = _slot_err((k[0], k[1], "n"))
        return None if e is None else factor * e

    # ---- 全局变量名 ----
    _kind_names = {"deltar": "ΔR", "sigma2": "σ²", "n": "N"}
    parameter_names = []
    for slot in slots:
        if slot[0] == "global":
            parameter_names.append("ΔE0")
        elif slot[0] == "e0grp":
            parameter_names.append(f"ΔE0:{e0_group_order[slot[1]]}")
        elif slot[0] == "einstein":
            parameter_names.append(
                f"einstein{slot[1]}:"
                + ("sigma2_static" if slot[2] == "sigma2_static"
                   else "theta_e"))
        elif slot[0] == "debye":
            parameter_names.append(
                f"debye{slot[1]}:"
                + ("sigma2_static" if slot[2] == "sigma2_static"
                   else "theta_d"))
        else:
            di, pi, kind = slot
            labels = datasets[di].get("path_labels")
            label = (labels[pi] if labels is not None
                     else Path(datasets[di]["datfiles"][pi]).name)
            name = datasets[di].get("name", f"ds{di}")
            parameter_names.append(f"{name}:{label}:{_kind_names[kind]}")

    def variables_for_dataset(di):
        """该数据集实际受影响的全局变量数（含被绑定源变量）。"""
        keys = set()

        def root_dr(key):
            k, seen = key, set()
            while k in bound_deltar:
                if k in seen:
                    break
                seen.add(k)
                k = bound_deltar[k]
            keys.add((k[0], k[1], "deltar"))

        def root_s2(key):
            k, seen = key, set()
            while k in bound_sigma2:
                if k in seen:
                    break
                seen.add(k)
                k = bound_sigma2[k]
            if k in einstein_of:
                gi = einstein_of[k]
                keys.add(("einstein", gi, "sigma2_static"))
                keys.add(("einstein", gi, "theta_e"))
            elif k in debye_of:
                gi = debye_of[k]
                keys.add(("debye", gi, "sigma2_static"))
                keys.add(("debye", gi, "theta_d"))
            else:
                keys.add((k[0], k[1], "sigma2"))

        def root_n(key):
            k, seen = key, set()
            while k in bound_n:
                if k in seen:
                    break
                seen.add(k)
                k = bound_n[k][0]
            if (k[0], k[1], "n") in slot_index:
                keys.add((k[0], k[1], "n"))

        for pi in range(len(ds_paths[di])):
            root_dr((di, pi))
            root_s2((di, pi))
            root_n((di, pi))
        e0_key = e0_slot_key(e0_group_of_ds[di])
        if e0_key in slot_index:
            keys.add(e0_key)
        return len(keys)

    # ---- B12-1 AIC / BIC ----
    # 必须在 variables_for_dataset 定义之后计算：每个数据集的 k 是
    # 该数据集实际受影响的全局变量数（含被 shared_bonds 绑定的源变量），
    # 不能用全局 n_var 冒充。
    def _aic_bic(rss, n_obs, k):
        if n_obs <= 0 or rss <= 0:
            return None, None
        base = n_obs * np.log(rss / n_obs)
        return float(base + 2.0 * k), float(base + k * np.log(n_obs))

    ds_aic, ds_bic = [], []
    for di in range(n_ds):
        model_d = model_chi_r(di)
        m = ds_mask[di]
        rss_d = float(np.sum(np.abs(ds_data_r[di][m] - model_d[m]) ** 2))
        n_d = 2 * int(m.sum())
        k_d = int(variables_for_dataset(di))
        a_d, b_d = _aic_bic(rss_d, n_d, k_d)
        ds_aic.append(a_d)
        ds_bic.append(b_d)
    global_aic, global_bic = _aic_bic(
        float(np.sum(scaled_res ** 2)), int(scaled_res.size), int(n_var))

    # ---- 每个数据集的 FitResult ----
    n_idp_total = 0.0
    fit_results = []
    for di, d in enumerate(datasets):
        paths = ds_paths[di]
        labels = d.get("path_labels")
        model_r = model_chi_r(di)
        model_k = exafs_model_k(paths, ds_ft_k[di])
        fit_paths = []
        for pi, path in enumerate(paths):
            key = (di, pi)
            dr = resolved_dr(solution.x, key)
            dr_err = resolved_dr_err(key)
            s2 = resolved_sigma2(solution.x, key)
            s2_err = resolved_sigma2_err(key)
            n_val = resolved_n(solution.x, key)
            n_err = resolved_n_err(key)
            fit_paths.append(PathFit(
                label=(labels[pi] if labels is not None
                       else Path(d["datfiles"][pi]).name),
                reff=float(path.reff),
                degeneracy=float(n_val),
                delta_r=float(dr),
                delta_r_err=float(dr_err if dr_err is not None else 0.0),
                sigma2=float(s2),
                sigma2_err=(None if s2_err is None else float(s2_err)),
                n_err=(None if n_err is None else float(n_err)),
                sigma2_at_bound=bool(s2 <= 5e-4),
                nleg=int(getattr(path, "nleg", 2))))
        n_idp = _ds_n_idp_eff[di]
        n_idp_total += n_idp
        n_var_di = int(variables_for_dataset(di))
        nidp_sf_di = float(np.sqrt(
            max(len(solution.fun) - n_var_di, 1) /
            max(float(n_idp) - n_var_di, 1)))
        fit_results.append(FitResult(
            paths=fit_paths,
            s02=float(d["s02"]),
            delta_e0=float(solution.x[slot_index[e0_slot_key(
                e0_group_of_ds[di])]]),
            delta_e0_err=float(errors[slot_index[e0_slot_key(
                e0_group_of_ds[di])]]),
            r_factor=float(ds_rf[di]),
            n_independent=float(n_idp),
            n_variables=n_var_di,
            kmin=float(d["kmin"]), kmax=float(d["kmax"]),
            rmin=float(d["rmin"]), rmax=float(d["rmax"]),
            parameter_names=list(parameter_names),
            correlation=correlation,
            model_r=np.asarray(ds_r_grid[di], dtype=float),
            model_chi_r=np.asarray(model_r, dtype=complex),
            model_k=np.asarray(model_k, dtype=float),
            model_k_grid=np.asarray(ds_ft_k[di], dtype=float),
            aic=ds_aic[di], bic=ds_bic[di],
            jac_rank=jac_rank, rank_deficient=(jac_rank < n_var),
            nidp_scale_factor=nidp_sf_di))

    # ---- 爱因斯坦结果 ----
    einstein_results = []
    for gi in range(len(einstein_groups)):
        i_static = slot_index[("einstein", gi, "sigma2_static")]
        i_theta = slot_index[("einstein", gi, "theta_e")]
        einstein_results.append({
            "theta_e": float(solution.x[i_theta]),
            "theta_e_err": float(errors[i_theta]),
            "sigma2_static": float(solution.x[i_static]),
            "sigma2_static_err": float(errors[i_static])})

    # ---- 关联 Debye 结果 ----
    debye_results = []
    for gi, model in enumerate(debye_groups):
        i_static = slot_index[("debye", gi, "sigma2_static")]
        i_theta = slot_index[("debye", gi, "theta_d")]
        debye_results.append({
            "theta_d": float(solution.x[i_theta]),
            "theta_d_err": float(errors[i_theta]),
            "sigma2_static": float(solution.x[i_static]),
            "sigma2_static_err": float(errors[i_static]),
            "paths": [(int(item[0]), int(item[1]))
                      for item in model["paths"]],
            "mass_absorber": float(model["mass_absorber"]),
            "mass_scatterer": float(model["mass_scatterer"]),
            "k_debye": model.get("k_debye"),
        })

    return {
        "datasets": fit_results,
        "n_independent_total": float(n_idp_total),
        "n_variables_total": int(n_var),
        "global_r_factor": global_rf,
        "einstein_results": einstein_results,
        "debye_results": debye_results,
        "parameter_names": parameter_names,
        "correlation": correlation,
        "aic": global_aic,
        "bic": global_bic,
        "global_aic": global_aic,
        "global_bic": global_bic,
        "e0_groups": [e0_group_order[gi] for gi in range(n_e0_groups)],
        "e0_group_of_dataset": list(e0_group_of_ds),
    }
