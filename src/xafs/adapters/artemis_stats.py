# -*- coding: utf-8 -*-
"""artemis_stats.py — #3 参数设置与理论谱 + #4 统计量/F-test + #4b 催化剂精细结构反演（自 artemis.py 纯拆分）

纯搬运拆分：代码块与原 artemis.py 逐行一致，不含任何逻辑改动。
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

import athena
from xafs_core import (FitResult, PathFit, XAFSConfigError, XAFSDataError)

from artemis_feff import ATOMIC_NUMBER, BVS_R0  # noqa: E402,F401


# ================================================================
# 3. 参数设置与理论谱
# ================================================================

# C₃ 的物理量级：1e-4–1e-3 Å³ 是常见非谐；±0.01 Å³ 以上属于
# 相位被 C₃ 吞噬的失配补偿，不能再解释为真实分布偏斜。
C3_LIMIT = 0.01

# C₄ 的物理量级：非高斯展宽的四阶累积量。Larch 参数名 "fourth"，
# 振幅项 exp(+2/3·C₄·k⁴)。|C₄| ≥ 2e-4 Å⁴ 时 k⁴ 项在高 k 端会
# 指数爆炸，只能作为失配补偿的边界信号，不能当真实分布峰度。
C4_LIMIT = 2e-4


def set_path_parameters(path, s02=None, delta_r=None, sigma2=None,
                        delta_e0=None, third=None, fourth=None):
    """
    把拟合参数写进 path.params 并重算 χ(k)。

    **关键（踩过的坑）**:
        larch 的 __path_params 从 self.params[parname] 取值，
        而 params 一旦创建就不再跟随 self.deltar / self.sigma2 属性。
        所以必须走 pathpar_name → params[...].value → _calc_chi()，
        直接改属性或用 calc_chi_from_params() 都不会生效（实测参数逐点不变）。

    third 是三阶累积量 C₃：近邻距离分布偏斜（非高斯，如纳米颗粒表面
    张力导致的 Fe-Fe 分布不对称）时的修正项，进入相位 4/3·C₃·k³。
    fourth 是四阶累积量 C₄：振幅项 exp(+2/3·C₄·k⁴)，描述非高斯展宽。
    """
    mapping = {"s02": s02, "deltar": delta_r, "sigma2": sigma2,
               "e0": delta_e0, "third": third, "fourth": fourth}
    for name, value in mapping.items():
        if value is None:
            continue
        key = path.pathpar_name(name)
        if key in path.params:
            path.params[key].value = float(value)
    path._calc_chi()


def exafs_model_k(paths, k) -> np.ndarray:
    """[Implements: EQ-02]
    多条路径叠加得到理论 χ(k)。

    数学（标准 EXAFS 方程）:
        χ(k) = Σ_j  S₀²·N_j / (k·R_j²) · F_j(k)
                    · exp(−2R_j/λ(k)) · exp(−2σ_j²k²)
                    · sin(2kR_j + φ_j(k))

    其中 F_j、φ_j、λ 由 FEFF 算出，R_j = Reff_j + ΔR_j。
    """
    k = np.asarray(k)
    total = np.zeros_like(k)
    for path in paths:
        total += np.interp(k, np.asarray(path.k), np.asarray(path.chi))
    return total


# ================================================================
# 4. 统计量
# ================================================================

def independent_points(kmin: float, kmax: float, rmin: float, rmax: float) -> float:
    """[Implements: EQ-04]
    独立数据点数（Nyquist 判据）。

    数学:
        N_idp = 2·Δk·ΔR / π

    注意：**不加 2**。部分教材写成 2ΔkΔR/π + 2，属保守修正；
    larch / Artemis / IFeffit 均不加，加了会高估自由度、让过拟合更容易通过。
    """
    return float(2.0 * (kmax - kmin) * (rmax - rmin) / np.pi)


def _bond_key(absorber: str, ligand: str):
    return f"{str(absorber).capitalize()}-{str(ligand).capitalize()}"


def _path_ligand(path, absorber: str | None = None):
    """从 PathFit / dict / 标签中提取散射元素，取不到返回 None。"""
    import re

    if isinstance(path, dict):
        for key in ("ligand", "element", "scatterer", "symbol"):
            if path.get(key):
                return str(path[key]).capitalize()
        label = str(path.get("label", ""))
    else:
        label = str(getattr(path, "label", ""))
    symbols = [s.capitalize() for s in re.findall(r"[A-Z][a-z]?", label)]
    symbols = [s for s in symbols if s in ATOMIC_NUMBER]
    if absorber:
        symbols = [s for s in symbols if s != str(absorber).capitalize()]
    return symbols[0] if symbols else None


def _path_n_r(path):
    if isinstance(path, dict):
        n = path.get("N", path.get("degeneracy", path.get("n")))
        r = path.get("R", path.get("r"))
        if r is None:
            r = float(path.get("reff", 0.0)) + float(path.get("delta_r", 0.0))
    else:
        n = getattr(path, "degeneracy", None)
        r = getattr(path, "r", None)
        if r is None:
            r = float(getattr(path, "reff", 0.0)) + float(
                getattr(path, "delta_r", 0.0))
    return float(n), float(r)


def bond_valence_sum(paths, r0_table=None, b: float = 0.37,
                     target: float | None = None,
                     sigma: float | None = None,
                     absorber: str | None = None) -> float:
    """B11-2 键价和（BVS）:

        BVS = Σ_i N_i · exp[(R₀,i − R_i)/b]

    ``paths`` 每项可为 ``PathFit`` 或 dict（N/degeneracy、R/r、
    ligand/element）。R₀ 从内置 ``BVS_R0`` 取，可用 ``r0_table`` 覆盖
    （键写成 "Fe-O" 或 ("Fe","O")）。缺参考键长时抛
    ``XAFSConfigError``——不静默丢掉一条键。

    Fe–O N=6, R=2.0 → 6·exp((1.759−2.0)/0.37) ≈ 3.13（Fe³⁺ 合理）。
    """
    table = dict(BVS_R0)
    if r0_table:
        for key, val in r0_table.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                table[_bond_key(key[0], key[1])] = float(val)
            else:
                table[str(key)] = float(val)
    if absorber is None:
        absorber = "Fe"
    if not np.isfinite(b) or b <= 0:
        raise XAFSConfigError("BVS", f"b={b!r} 必须为正")
    total = 0.0
    for path in paths:
        n, r = _path_n_r(path)
        ligand = _path_ligand(path, absorber)
        if ligand is None:
            raise XAFSConfigError(
                "BVS", f"无法确定路径 {path!r} 的散射元素；请给 "
                       f"ligand/element 字段或可解析的标签")
        key = _bond_key(absorber, ligand)
        if key not in table:
            raise XAFSConfigError(
                "BVS", f"缺少参考键长 {key}；在 r0_table 中显式提供，"
                       f"不猜测未标定的 R₀")
        r0 = float(table[key])
        if not np.isfinite(n) or n <= 0 or not np.isfinite(r) or r <= 0:
            raise XAFSConfigError(
                "BVS", f"路径 N={n!r}, R={r!r} 非物理")
        total += n * np.exp((r0 - r) / float(b))
    return float(total)


def correlated_debye_sigma2(temperature_K: float, theta_d: float, reff: float,
                            mass_absorber: float, mass_scatterer: float,
                            k_debye: float | None = None,
                            sigma2_static: float = 0.0) -> float:
    """B11-2 关联 Debye 模型的键长无序 σ²(T, R)。

    非关联项（Debye 固体）:

        σ²_un = (3ħ²/(2 μ k_B Θ_D)) · [1/4
                + (T/Θ_D)² ∫₀^{Θ_D/T} x/(eˣ−1) dx]

    关联因子:

        C(R) = 1 − sin(2·k_D·R)/(2·k_D·R)
        σ²   = σ²_static + C(R)·σ²_un

    近邻（R 小）C→0，运动强相关、σ² 被压低；远壳 C→1，回到非关联值。
    ``k_debye`` 缺省由最近邻密度估计 ``k_D ≈ (36π/R³)^(1/3)`` 并在
    返回值的 ``k_debye_source`` 中显式记录（见 ``last_details``）。
    """
    from scipy.integrate import quad

    T = float(temperature_K)
    theta = float(theta_d)
    r = float(reff)
    m1 = float(mass_absorber)
    m2 = float(mass_scatterer)
    static = float(sigma2_static)
    if not np.isfinite(T) or T <= 0:
        raise XAFSConfigError("DEBYE", f"温度 T={temperature_K!r} 必须 > 0 K")
    if not np.isfinite(theta) or theta <= 0:
        raise XAFSConfigError("DEBYE", f"Θ_D={theta_d!r} 必须 > 0 K")
    if not np.isfinite(r) or r <= 0:
        raise XAFSConfigError("DEBYE", f"Reff={reff!r} 必须 > 0 Å")
    if not (np.isfinite(m1) and m1 > 0 and np.isfinite(m2) and m2 > 0):
        raise XAFSConfigError(
            "DEBYE", f"质量必须为正: m1={mass_absorber!r}, m2={mass_scatterer!r}")
    if not np.isfinite(static) or static < 0:
        raise XAFSConfigError("DEBYE", f"σ²_static={sigma2_static!r} 不能为负")
    mu = m1 * m2 / (m1 + m2)

    upper = theta / T
    integral, _ = quad(lambda x: x / np.expm1(x) if x > 0 else 1.0,
                       0.0, upper, limit=200)
    bracket = 0.25 + (T / theta) ** 2 * float(integral)
    sigma2_un = (3.0 * HBAR2_OVER_2KB / (mu * theta)) * bracket

    if k_debye is None:
        # 保守默认：k_D=0.35 Å⁻¹ 对应低密度固体（最近邻密度量级
        # ~3×10⁻³ Å⁻³）。关键是它**不随 reff 缩放**——若用
        # k_D∝1/reff，2k_D·R 变成常数，近壳与远壳的关联因子相同，
        # 远壳 σ² 不会按物理预期增大。需要精确值时显式传 k_debye。
        kd = 0.35
        source = "conservative_default_0.35A^-1"
    else:
        kd = float(k_debye)
        source = "user"
        if not np.isfinite(kd) or kd <= 0:
            raise XAFSConfigError("DEBYE", f"k_D={k_debye!r} 必须 > 0 Å⁻¹")
    x = 2.0 * kd * r
    corr = 1.0 - (np.sin(x) / x if abs(x) > 1e-12 else 1.0)
    sigma2 = static + corr * sigma2_un
    correlated_debye_sigma2.last_details = {
        "reduced_mass_amu": float(mu),
        "sigma2_un_correlated": float(sigma2_un),
        "correlation_factor": float(corr),
        "k_debye": float(kd),
        "k_debye_source": source,
        "integral": float(integral),
    }
    return float(sigma2)


correlated_debye_sigma2.last_details = None


def sac_cluster_sites(bond_m_ligand: float = 2.00, ligand: str = "N",
                      n_coord: int = 4, axial: str | None = None,
                      axial_bond: float = 2.10, carbon_shell: bool = True,
                      ring_radius: float = 2.90) -> list:
    """B12-2 单原子催化剂（SAC）M–N₄ 平面团簇生成器。

    返回 ``[(x, y, z, element), ...]``，中心吸收原子固定在原点（不包含
    在列表里），可直接送 ``build_feff_input``。

    ``n_coord=4`` 生成正方形 M–N₄；``axial`` 给 "O"/"Cl" 时沿 ±z 加
    轴向配位；``carbon_shell=True`` 时在 R≈ring_radius 的平面内加 8 个
    C（模拟石墨烯/碳氮载体第一壳）。
    """
    d = float(bond_m_ligand)
    n = int(n_coord)
    if n not in (2, 3, 4, 5, 6):
        raise XAFSConfigError(
            "SAC", f"n_coord={n_coord} 只支持 2/3/4/5/6")
    if not np.isfinite(d) or d <= 0:
        raise XAFSConfigError("SAC", f"bond_m_ligand={bond_m_ligand!r} 必须 > 0")
    lig = str(ligand).capitalize()
    if lig not in ATOMIC_NUMBER:
        raise XAFSConfigError("SAC", f"未知配体元素 {ligand!r}")

    sites = []
    if n == 2:
        sites += [(d, 0.0, 0.0, lig), (-d, 0.0, 0.0, lig)]
    elif n == 3:
        for k in range(3):
            ang = 2.0 * np.pi * k / 3.0
            sites.append((float(d * np.cos(ang)), float(d * np.sin(ang)), 0.0, lig))
    else:
        half = d / np.sqrt(2.0)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                sites.append((sx * half, sy * half, 0.0, lig))
        if n == 5:
            sites.append((0.0, 0.0, d, lig))
        elif n == 6:
            for sz in (-1.0, 1.0):
                sites.append((0.0, 0.0, sz * d, lig))

    if axial:
        if n == 6:
            raise XAFSConfigError("SAC", "n_coord=6 已含 ±z 轴向配体，不可再叠加 axial 配体")
        ax = str(axial).capitalize()
        if ax not in ATOMIC_NUMBER:
            raise XAFSConfigError("SAC", f"未知轴向配体元素 {axial!r}")
        if not np.isfinite(axial_bond) or axial_bond <= 0:
            raise XAFSConfigError("SAC", "axial_bond 必须 > 0 Å")
        sites.append((0.0, 0.0, float(axial_bond), ax))
        sites.append((0.0, 0.0, -float(axial_bond), ax))

    if carbon_shell:
        if not np.isfinite(ring_radius) or ring_radius <= 0:
            raise XAFSConfigError("SAC", "ring_radius 必须 > 0 Å")
        for k in range(8):
            ang = 2.0 * np.pi * k / 8.0 + np.pi / 8.0
            sites.append((float(ring_radius * np.cos(ang)),
                          float(ring_radius * np.sin(ang)), 0.0, "C"))
    return sites


def diatomic_m1m2_sites(m1: str = "Fe", m2: str = "Co",
                        bond_length: float = 2.50, ligand: str = "N",
                        n_coord: int = 4, ligand_bond: float = 2.00) -> list:
    """B12-2 双原子催化剂（DAC）M₁–M₂ 位点生成器。

    M₁ 在原点（吸收中心），M₂ 沿 +z 距离 ``bond_length``；两者各带
    ``n_coord`` 个平面配体（默认 N₄）。返回包含 M₂ 与所有配体的
    ``[(x, y, z, element), ...]``，M₁ 不重复列出。
    """
    m1 = str(m1).capitalize()
    m2 = str(m2).capitalize()
    lig = str(ligand).capitalize()
    for sym in (m1, m2, lig):
        if sym not in ATOMIC_NUMBER:
            raise XAFSConfigError("DAC", f"未知元素 {sym!r}")
    if not (np.isfinite(bond_length) and bond_length > 0):
        raise XAFSConfigError("DAC", "bond_length 必须 > 0 Å")
    if not (np.isfinite(ligand_bond) and ligand_bond > 0):
        raise XAFSConfigError("DAC", "ligand_bond 必须 > 0 Å")
    n = int(n_coord)
    if n not in (2, 3, 4, 5, 6):
        raise XAFSConfigError("DAC", "n_coord 只支持 2/3/4/5/6")

    z2 = float(bond_length)
    sites = [(0.0, 0.0, z2, m2)]
    half = float(ligand_bond) / np.sqrt(2.0)

    def _add_ligands(z_center, is_upper: bool):
        if n == 2:
            pts = [(ligand_bond, 0.0), (-ligand_bond, 0.0)]
        elif n == 3:
            pts = [(float(ligand_bond * np.cos(2.0 * np.pi * k / 3.0)),
                    float(ligand_bond * np.sin(2.0 * np.pi * k / 3.0)))
                   for k in range(3)]
        else:
            pts = [(sx * half, sy * half)
                   for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
            if n in (5, 6):
                # 仅在外侧加轴向配体（M1 向 -z，M2 向 +z），避免 M1-M2 键内碰撞
                z_ax = (z_center + ligand_bond) if is_upper else (z_center - ligand_bond)
                sites.append((0.0, 0.0, float(z_ax), lig))
        for (x, y) in pts:
            sites.append((float(x), float(y), float(z_center), lig))

    _add_ligands(0.0, False)
    _add_ligands(z2, True)
    return sites


def reconstruct_pair_distribution(
        r_grid: np.ndarray,
        n: float,
        r0: float,
        sigma2: float,
        c3: float = 0.0,
        c4: float = 0.0,
        lambda_eff: float = 8.0,
) -> dict:
    """[Implements: EQ-07] 由累积量展开重构非高斯近邻径向分布函数 g(R)（纯数学，无 FEFF）。

    数学:
        z = (R − r0)/σ,  σ = sqrt(max(σ², 1e-6))
        γ1 = C3/σ³,  γ2 = C4/σ⁴
        g_raw(R) = N·φ(z)·max(0, 1 + γ1/6·H3(z) + γ2/24·H4(z))
        H3 = z³ − 3z,  H4 = z⁴ − 6z² + 3
    再按梯形积分把总面积归一回 N（非负截断会改变面积，必须重归一）。
    同时依据 Bunker (1983) / Tranquada 有效径向分布加权给出热力学平均键长近似：
        r_thermodynamic ≈ r0 + (2σ²/r0)·(1 + r0/λ_eff)

    物理:
        C3 > 0 表示右偏（长尾在高 R 侧）——正偏态分布的峰位
        r_mode 低于平均键长 r_mean。C4 描述超额峰度（展宽/尖峰）。
        当 |γ1| 或 |γ2| 较大时，Gram-Charlier 级数可能出现负值截断并伴随非物理双峰/旁瓣，
        `gram_charlier_clipped` 与 `unimodal` 会显式给出诊断告警。
        该函数只回答“给定累积量时 g(R) 长什么样”，不替代拟合；
        C3/C4 必须来自数据约束，不能用它反推不可辨识的高阶矩。

    返回:
        {"r", "g_r", "r_mode", "r_mean", "r_thermodynamic",
         "cumulant_c1_shift_A", "skewness", "area",
         "gram_charlier_clipped", "unimodal"}
    """
    r = np.asarray(r_grid, dtype=float)
    if r.ndim != 1 or r.size < 3:
        raise XAFSConfigError(
            "RDF", f"r_grid 必须是一维且至少 3 点，收到形状 {r.shape}")
    if np.any(np.diff(r) <= 0):
        raise XAFSConfigError("RDF", "r_grid 必须严格单调递增")
    if not np.isfinite(n) or n <= 0:
        raise XAFSConfigError("RDF", f"配位数 N={n!r} 必须为正")
    if not np.isfinite(sigma2) or float(sigma2) <= 0:
        raise XAFSConfigError("RDF", f"德拜-瓦勒因子 σ²={sigma2!r} 必须为正")
    if not np.isfinite(lambda_eff) or float(lambda_eff) <= 0:
        raise XAFSConfigError(
            "RDF", f"有效平均自由程 λ_eff={lambda_eff!r} 必须为正有限值")
    sigma = float(np.sqrt(max(float(sigma2), 1e-6)))
    if not np.isfinite(r0):
        raise XAFSConfigError("RDF", f"r0={r0!r} 非有限")
    gamma1 = float(c3) / sigma ** 3
    gamma2 = float(c4) / sigma ** 4
    z = (r - float(r0)) / sigma
    phi = np.exp(-0.5 * z ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    h3 = z ** 3 - 3.0 * z
    h4 = z ** 4 - 6.0 * z ** 2 + 3.0
    raw_poly = 1.0 + gamma1 / 6.0 * h3 + gamma2 / 24.0 * h4
    gram_charlier_clipped = bool(np.any(raw_poly < 0.0))
    shape = np.maximum(0.0, raw_poly)
    g_raw = float(n) * phi * shape
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area_raw = float(trapz(g_raw, r))
    if area_raw <= 0 or not np.isfinite(area_raw):
        raise XAFSConfigError(
            "RDF", f"累积量给不出非负 g(R)（截断后面积={area_raw!r}）；"
                   f"C3/C4 相对 σ 过大")
    g_norm = g_raw * (float(n) / area_raw)
    area = float(trapz(g_norm, r))
    c1_shift = (
        (2.0 * float(sigma2) / float(r0)) * (1.0 + float(r0) / float(lambda_eff))
        if float(r0) > 0.0
        else 0.0
    )
    g_max = float(np.max(g_norm)) if g_norm.size else 0.0
    interior_peaks = (
        (g_norm[1:-1] > g_norm[:-2])
        & (g_norm[1:-1] >= g_norm[2:])
        & (g_norm[1:-1] > 1e-4 * g_max)
    )
    unimodal = bool(int(np.count_nonzero(interior_peaks)) <= 1)
    return {
        "r": r,
        "g_r": g_norm,
        "r_mode": float(r[int(np.argmax(g_norm))]),
        "r_mean": float(r0),
        "r_thermodynamic": float(float(r0) + c1_shift),
        "cumulant_c1_shift_A": float(c1_shift),
        "skewness": gamma1,
        "area": area,
        "gram_charlier_clipped": gram_charlier_clipped,
        "unimodal": unimodal,
    }


def estimate_r_noise(chi_k, kmin: float, kmax: float, kweight: int = 2,
                     dk: float = 1.0,
                     r_noise_range=(15.0, 25.0)) -> float:
    """高 R 区噪声估计 ε_R（对标 IFEFFIT 的 epsilon_k / epsilon_r）。

    物理依据：EXAFS 的物理信号在 R>15 Å 早已衰减到噪声量级；该区间
    的 |χ(R)| 只剩下数据噪声与 FT 泄漏。对实部、虚部同时取均方根：

        ε_R = sqrt( 1/(2N) · Σ_{R∈[15,25]} (Re² + Im²) )

    为什么必须用 k^w χ(k) 做 FT：IFEFFIT 的 ε 与拟合时使用同一 k 加权；
    用未加权 χ(k) 会把噪声换算错一个 k^{2w} 的尺度，χ² 随之系统偏移。
    """
    lo, hi = float(r_noise_range[0]), float(r_noise_range[1])
    if hi <= lo:
        raise XAFSConfigError(
            "FIT", f"r_noise_range 上界 {hi} 必须大于下界 {lo}")
    # larch 的 xftf 默认 R 网格只到 ~kmax（本环境实测 9.97 Å），
    # 必须显式要求扩展到噪声区上界；否则 [15,25] 内一个点都没有。
    ft_full = athena.ft_to_r_space(
        np.asarray(chi_k.k), np.asarray(chi_k.chi), None,
        kmin=float(kmin), kmax=float(kmax), kweight=int(kweight),
        dk=float(dk), rmax_out=hi)
    r = np.asarray(ft_full.r, dtype=float)
    chi_r = np.asarray(ft_full.chi_r, dtype=complex)
    mask = (r >= lo) & (r <= hi)
    n = int(mask.sum())
    if n < 8:
        raise XAFSConfigError(
            "FIT", f"高 R 噪声区 [{lo},{hi}] Å 内仅 {n} 个 FT 点"
                   f"（FT R 网格上限 {float(r.max()):.2f} Å）；"
                   f"无法估计 ε_R")
    re = chi_r[mask].real
    im = chi_r[mask].imag
    eps = float(np.sqrt(np.mean(0.5 * (re ** 2 + im ** 2))))
    if not np.isfinite(eps) or eps < 0:
        raise XAFSConfigError("FIT", f"ε_R 计算得到非法值 {eps!r}")
    return eps


def shell_f_test(fit_simple, fit_complex) -> dict:
    """增减壳层的 Hamilton / F-test（对标 Artemis 的 F-test）。

    少壳层模型（p1 个变量）与多壳层模型（p2 > p1）必须使用**同一
    R 窗与同一数据**。IFEFFIT 定义：

        F = [(χ²₁ − χ²₂)/(p₂ − p₁)] / [χ²₂/(N_idp − p₂)]

    置信度 = F 分布的 CDF(F; dfn=p₂−p₁, dfd=N_idp−p₂)。默认
    ``confidence ≥ 0.95`` 且 ``dfd ≥ 1`` 才判显著；否则新增壳层
    只是拟合噪声。
    """
    from scipy.stats import f as f_dist

    p1 = int(getattr(fit_simple, "n_variables", 0))
    p2 = int(getattr(fit_complex, "n_variables", 0))
    n_idp = float(getattr(fit_complex, "n_independent",
                          getattr(fit_simple, "n_independent", 0.0)))
    dfn = p2 - p1
    dfd = float(n_idp - p2)

    out = {
        "F": 0.0, "dfn": int(dfn), "dfd": float(dfd),
        "confidence": 0.0, "significant": False, "valid": False,
        "p1": p1, "p2": p2, "n_independent": float(n_idp),
        "delta_chi_square": None, "method": None,
    }

    # 窗口/拟合空间一致性校验：不一致或 p2<=p1 时禁止给出虚假 F 值
    keys = ("kmin", "kmax", "rmin", "rmax", "fit_space", "kweights")
    for key in keys:
        v1 = getattr(fit_simple, key, None)
        v2 = getattr(fit_complex, key, None)
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, (tuple, list)) or isinstance(v2, (tuple, list)):
            mismatch = tuple(v1) != tuple(v2)
        elif isinstance(v1, str) or isinstance(v2, str):
            mismatch = str(v1) != str(v2)
        else:
            try:
                mismatch = abs(float(v1) - float(v2)) > 1e-12
            except Exception:
                mismatch = v1 != v2
        if mismatch:
            out["note"] = f"F-test 无效：{key} 不一致 ({v1!r} vs {v2!r})"
            return out

    if p2 <= p1:
        out["note"] = "多壳层模型变量数未增加或不合法（p2 <= p1），F-test 不适用"
        return out
    if dfn <= 0:
        out["note"] = "多壳层模型变量数未增加，F-test 不适用"
        return out
    if dfd < 1.0:
        out["note"] = f"dfd = N_idp − p2 = {dfd:.2f} < 1，无自由度"
        return out

    chi1 = getattr(fit_simple, "chi_square", None)
    chi2 = getattr(fit_complex, "chi_square", None)
    if chi1 is not None and chi2 is not None and chi2 > 0:
        delta = float(chi1 - chi2)
        F = (delta / dfn) / (float(chi2) / dfd)
        out["method"] = "chi_square"
        out["delta_chi_square"] = delta
    else:
        # 回退：R-factor 与 χ² 只差同一常数因子，比值形式仍成立。
        r1 = float(getattr(fit_simple, "r_factor", np.nan))
        r2 = float(getattr(fit_complex, "r_factor", np.nan))
        if not (np.isfinite(r1) and np.isfinite(r2) and r2 > 0):
            out["note"] = "两个模型都缺少 χ²，R-factor 也非法"
            return out
        F = ((r1 - r2) / dfn) / (r2 / dfd)
        out["method"] = "r_factor"

    if not np.isfinite(F):
        out["note"] = "F 非有限值"
        return out
    out["F"] = float(F)
    out["valid"] = True
    if F > 0:
        conf = float(f_dist.cdf(F, dfn, dfd))
        out["confidence"] = conf
        out["significant"] = bool(conf >= 0.95 and dfd >= 1.0)
    else:
        out["note"] = "多壳层模型未降低残差（F ≤ 0）"
    return out


def r_factor(data_chi_r, model_chi_r, mask) -> float:
    """
    绝对拟合优度。

    数学:
        R-factor = Σ|χ_data(R) − χ_th(R)|² / Σ|χ_data(R)|²

    判据: < 0.02 优秀，< 0.05 良好，> 0.10 模型不可用。
    """
    num = float(np.sum(np.abs(data_chi_r[mask] - model_chi_r[mask]) ** 2))
    den = float(np.sum(np.abs(data_chi_r[mask]) ** 2))
    return num / den if den > 0 else np.inf


# ================================================================
# 4b. 催化剂精细结构反演（B17）
# ================================================================

# fcc/hcp 与 bcc 的体相第一壳配位数、第二壳几何与堆积因子。
_LATTICE_BULK = {
    "fcc": {"n1": 12.0, "n2": 6.0, "r2_over_r1": float(np.sqrt(2.0)),
            "eta": float(np.pi / (3.0 * np.sqrt(2.0)))},
    "hcp": {"n1": 12.0, "n2": 6.0, "r2_over_r1": float(np.sqrt(2.0)),
            "eta": float(np.pi / (3.0 * np.sqrt(2.0)))},
    "bcc": {"n1": 8.0, "n2": 6.0,
            "r2_over_r1": float(2.0 / np.sqrt(3.0)),
            "eta": float(np.sqrt(3.0) * np.pi / 8.0)},
}


def estimate_nanoparticle_size(
    n1: float,
    r1: float,
    n1_err: float | None = None,
    n2: float | None = None,
    lattice: str = "fcc",
    shape: str = "spherical",
) -> dict:
    """[Implements: EQ-06] B17-1：由第一壳金属配位数反演纳米颗粒粒径、原子数与分散度。

    球截断（Calvin / Beale）平均配位数：

        N(r, d) = N_bulk · [1 − 1.5·u + 0.5·u³],  u = r/d ∈ (0, 1]

    由拟合 N₁ 反解 u 后 d = r₁/u（Å）。``shape="hemispherical"`` 时
    半球/筏状基底直径乘 2^(1/3)（同体积球等效）。
    """
    lat = str(lattice).strip().lower()
    if lat not in _LATTICE_BULK:
        raise XAFSConfigError(
            "NP", f"lattice={lattice!r} 只支持 'fcc' / 'hcp' / 'bcc'")
    shp = str(shape).strip().lower()
    if shp not in ("spherical", "hemispherical"):
        raise XAFSConfigError(
            "NP", f"shape={shape!r} 只支持 'spherical' / 'hemispherical'")
    try:
        n1v = float(n1)
        r1v = float(r1)
    except (TypeError, ValueError) as exc:
        raise XAFSConfigError("NP", "n1 / r1 必须是数值") from exc
    if not np.isfinite(n1v) or n1v <= 0.0:
        raise XAFSConfigError("NP", f"n1={n1!r} 必须是有限正数")
    if not np.isfinite(r1v) or r1v <= 0.0:
        raise XAFSConfigError("NP", f"r1={r1!r} 必须是有限正数")
    err1 = None
    if n1_err is not None:
        err1 = float(n1_err)
        if not np.isfinite(err1) or err1 < 0.0:
            raise XAFSConfigError(
                "NP", f"n1_err={n1_err!r} 必须是非负有限数")

    bulk = _LATTICE_BULK[lat]
    n1_bulk = bulk["n1"]
    n2_bulk = bulk["n2"]
    r2 = bulk["r2_over_r1"] * r1v
    eta = bulk["eta"]
    scale = 2.0 ** (1.0 / 3.0) if shp == "hemispherical" else 1.0

    def _shape_hint(diameter_A: float, n2_sphere: float) -> str:
        if n2 is None:
            return "single_shell_only"
        if n2_sphere > 0.5 and float(n2) < 0.65 * n2_sphere:
            return "raft_or_flattened"
        return "spherical_consistent"

    # ---- 体相极限：不再有尺寸信息 ----
    if n1v >= 0.99 * n1_bulk:
        n2_sphere = float(n2_bulk)
        return {
            "diameter_nm": float("inf"),
            "diameter_A": float("inf"),
            "diameter_nm_err": None,
            "n_atoms": float("inf"),
            "dispersion": 0.0,
            "lattice": lat,
            "n1_bulk": float(n1_bulk),
            "n2_spherical_expected": n2_sphere,
            "shape_hint": _shape_hint(float("inf"), n2_sphere),
            "regime": "bulk_limit",
        }

    # ---- 球截断方程反解 u ----
    from scipy.optimize import brentq

    f = float(np.clip(n1v / n1_bulk, 0.05, 0.99))

    def _trunc(u):
        return 1.0 - 1.5 * u + 0.5 * u ** 3 - f

    u = float(brentq(_trunc, 1e-9, 1.0, xtol=1e-14, rtol=1e-14))
    d_A = float(r1v / u) * scale
    diameter_nm = d_A / 10.0

    # 隐函数求导：dd/dN = 2·r1 / [3·u²·(1−u²)·N_bulk]
    dd_dn = (2.0 * r1v
             / (3.0 * u ** 2 * (1.0 - u ** 2) * n1_bulk)) * scale
    diameter_nm_err = None if err1 is None else abs(dd_dn) * err1 / 10.0

    n_atoms = max(2.0, float(eta * (d_A / r1v) ** 3))
    dispersion = float(np.clip(4.0 / n_atoms ** (1.0 / 3.0), 0.0, 1.0))

    # ---- 第二壳形貌自洽检验 ----
    if d_A <= r2:
        n2_sphere = 0.0
    else:
        v = r2 / d_A
        n2_sphere = float(n2_bulk * (1.0 - 1.5 * v + 0.5 * v ** 3))

    return {
        "diameter_nm": diameter_nm,
        "diameter_A": d_A,
        "diameter_nm_err": (None if diameter_nm_err is None
                            else float(diameter_nm_err)),
        "n_atoms": n_atoms,
        "dispersion": dispersion,
        "lattice": lat,
        "n1_bulk": float(n1_bulk),
        "n2_spherical_expected": n2_sphere,
        "shape_hint": _shape_hint(d_A, n2_sphere),
        "regime": "nanoparticle",
    }


def compute_cowley_sro(
    n_aa: float,
    n_ab: float,
    x_partner: float,
    n_aa_err: float | None = None,
    n_ab_err: float | None = None,
) -> dict:
    """B17-2：由同核/异核配位数计算 Cowley 短程有序参数。

        α_Cowley = 1 − p_AB / x_B,  p_AB = N_AB / (N_AA + N_AB)

    α≈0 随机固溶体；α<0 异核有序（金属间化合物）；α>0 同核团聚/偏析。
    """
    try:
        aa = float(n_aa)
        ab = float(n_ab)
        xb = float(x_partner)
    except (TypeError, ValueError) as exc:
        raise XAFSConfigError("SRO", "n_aa / n_ab / x_partner 必须是数值") from exc
    if not (np.isfinite(xb) and 0.0 < xb < 1.0):
        raise XAFSConfigError(
            "SRO", f"x_partner={x_partner!r} 必须在 (0, 1) 内")
    if not (np.isfinite(aa) and np.isfinite(ab)) or aa < 0.0 or ab < 0.0:
        raise XAFSConfigError(
            "SRO", f"N_AA={n_aa!r} / N_AB={n_ab!r} 必须是非负有限数")
    total = aa + ab
    if total <= 0.0:
        raise XAFSConfigError("SRO", "N_AA + N_AB 必须 > 0")

    p_ab = ab / total
    alpha = 1.0 - p_ab / xb
    alloy_pct = p_ab / xb * 100.0

    alpha_err = None
    if n_aa_err is not None and n_ab_err is not None:
        ea = float(n_aa_err)
        eb = float(n_ab_err)
        if not (np.isfinite(ea) and np.isfinite(eb)
                and ea >= 0.0 and eb >= 0.0):
            raise XAFSConfigError(
                "SRO", "n_aa_err / n_ab_err 必须是非负有限数")
        alpha_err = float(
            np.sqrt((ab * ea) ** 2 + (aa * eb) ** 2)
            / (xb * total ** 2))

    if alpha < -0.15:
        tendency = "heteroatomic_ordering"
    elif alpha > 0.15:
        tendency = "homoatomic_clustering_or_segregation"
    else:
        tendency = "random_solid_solution"

    return {
        "n_aa": aa,
        "n_ab": ab,
        "n_total": float(total),
        "p_ab": float(p_ab),
        "x_partner": xb,
        "alpha_cowley": float(alpha),
        "alpha_err": alpha_err,
        "alloying_extent_pct": float(alloy_pct),
        "tendency": tendency,
    }


def check_shell_resolution_and_z_degeneracy(
    shells: list[dict],
    kmin: float = 3.0,
    kmax: float = 12.0,
) -> dict:
    """B18-1：Rayleigh 径向分辨极限与轻元素 Z 简并门禁。

    有限 k 窗口能分辨的最小壳层间距：

        ΔR_Rayleigh = π / (2·(k_max − k_min))

    当两壳 ΔZ ≤ 2（C/N/O 等轻元素相移/振幅几乎重合）且
    ΔR < ΔR_Rayleigh 时，同时放开两条路径的 N/ΔR/σ² 会发生反相
    干涉抵消，配位数无物理意义。这里只诊断并给出建议，不擅自删壳。
    """
    try:
        lo = float(kmin)
        hi = float(kmax)
    except (TypeError, ValueError) as exc:
        raise XAFSConfigError("RES", "kmin / kmax 必须是数值") from exc
    delta_k = max(hi - lo, 1e-6)
    rayleigh = float(np.pi / (2.0 * delta_k))

    rows = []
    for i, sh in enumerate(shells or []):
        label = str(sh.get("label") or f"shell{i}")
        el = str(sh.get("scatterer") or "").capitalize()
        if el not in ATOMIC_NUMBER:
            raise XAFSConfigError(
                "RES", f"{label}: 未知散射元素 {sh.get('scatterer')!r}")
        r_val = sh.get("r", sh.get("reff"))
        if r_val is None:
            raise XAFSConfigError("RES", f"{label}: 缺少 r / reff")
        rv = float(r_val)
        if not np.isfinite(rv):
            raise XAFSConfigError("RES", f"{label}: r 非有限值")
        rows.append({"label": label, "el": el,
                     "z": int(ATOMIC_NUMBER[el]), "r": rv})

    warnings = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            dz = abs(a["z"] - b["z"])
            dr = abs(a["r"] - b["r"])
            if dz <= 2 and dr < rayleigh:
                warnings.append({
                    "path_a": a["label"],
                    "path_b": b["label"],
                    "scatterer_a": a["el"],
                    "scatterer_b": b["el"],
                    "delta_z": int(dz),
                    "delta_r": float(dr),
                    "rayleigh_limit_A": float(rayleigh),
                    "kind": ("light_element_z_degeneracy"
                             if max(a["z"], b["z"]) <= 10
                             else "sub_rayleigh_split"),
                })

    resolvable = len(warnings) == 0
    if resolvable:
        recommendation = ("所有壳层间距 ≥ Rayleigh 极限或散射元素 Z 差异"
                          "足够大，可独立拟合。")
    else:
        recommendation = ("检测到亚 Rayleigh 近简并壳层：建议绑定 "
                          "shared_sigma2/constraints，或用 BVS/化学计量"
                          "消歧；不要同时放开近简并拆壳的 N/ΔR/σ²。")
    return {
        "rayleigh_limit_A": rayleigh,
        "delta_k": float(delta_k),
        "resolvable": resolvable,
        "pair_warnings": warnings,
        "recommendation": recommendation,
    }


# ================================================================
# 5. R 空间最小二乘拟合
# ================================================================

def _parse_kweights(kweight) -> tuple:
    """把单权 int 或序列统一成去重后保序的正整数元组。"""
    if isinstance(kweight, (int, np.integer)):
        vals = [int(kweight)]
    else:
        vals = [int(x) for x in kweight]
    if not vals:
        raise XAFSConfigError("FIT", "kweight 不能为空")
    if any(v < 0 for v in vals):
        raise XAFSConfigError("FIT", f"kweight 不能为负: {vals}")
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return tuple(out)


def normalize_constraints(constraints, path_labels) -> list:
    """B14-1 催化配位化学计量约束（Mini-GDS）解析与校验。

    支持两种等价输入（**绝不使用 eval/exec**）：

    1. 结构化字典列表：
       - ``{"kind": "n_sum", "target": "Fe-Cl", "source": "Fe-O",
           "total": 6.0, "coeff": 1.0}``
         ⟹ N_target = total − coeff·N_source
       - ``{"kind": "n_ratio", "target": "Fe-Cl", "source": "Fe-O",
           "ratio": 1.5}`` ⟹ N_target = ratio·N_source
       - ``{"kind": "deltar_offset", "target": "Fe-Cl", "source": "Fe-O",
           "offset": 0.02}`` ⟹ ΔR_target = ΔR_source + offset
       - ``{"kind": "deltar_ratio", "target": "Fe-Cl", "source": "Fe-O",
           "ratio": 1.5}`` ⟹ ΔR_target = 1.5·ΔR_source
           （键长变化按 Reff 比例缩放，用于同一结构膨胀/压缩的远壳）
       - ``{"kind": "bounds", "path": "Fe-O", "param": "sigma2",
           "min": 0.002, "max": 0.010}``
    2. 字符串表达式（分号或逗号分隔）：
       - ``"N(Fe-O)+N(Fe-Cl)=6.0"``
       - ``"N(Fe-Cl)=1.5*N(Fe-O)"``
       - ``"dR(Fe-Cl)=dR(Fe-O)+0.02"``
       - ``"dR(Fe-Cl)=1.5*dR(Fe-O)"``

    返回带 ``target_index`` / ``source_index`` 的规范化列表；未知路径
    标签、重复目标、非法数值或成环依赖一律抛 ``XAFSConfigError``。
    """
    import re

    labels = [str(x) for x in (path_labels or [])]
    if constraints is None or constraints == "" or constraints == []:
        return []
    if not labels:
        raise XAFSConfigError(
            "FIT", "constraints 需要 path_labels 才能按壳层名绑定")
    index_of: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        index_of.setdefault(lab, []).append(i)

    def _resolve(label, what):
        key = str(label).strip()
        hits = index_of.get(key) or []
        if not hits:
            raise XAFSConfigError(
                "FIT", f"约束 {what} 引用了未知路径标签 {key!r}；"
                       f"可选 {labels}")
        if len(hits) > 1:
            raise XAFSConfigError(
                "FIT", f"约束 {what} 的标签 {key!r} 在路径中不唯一")
        return hits[0]

    if isinstance(constraints, str):
        raw = [c.strip() for c in re.split(r"[;,]", constraints)
               if c.strip()]
    elif isinstance(constraints, (list, tuple)):
        raw = list(constraints)
    else:
        raise XAFSConfigError(
            "FIT", f"constraints 必须是字符串或列表，收到 {type(constraints).__name__}")

    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            spec = _parse_constraint_expression(item)
        elif isinstance(item, dict):
            spec = dict(item)
        else:
            raise XAFSConfigError(
                "FIT", f"约束项必须是字符串或字典，收到 {item!r}")
        kind = str(spec.get("kind", "")).strip().lower()
        if kind == "bounds":
            try:
                path = str(spec["path"]).strip()
                param = str(spec.get("param", "sigma2")).strip().lower()
                lo = float(spec["min"])
                hi = float(spec["max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise XAFSConfigError(
                    "FIT", f"bounds 约束需要 path/param/min/max: {spec!r}") from exc
            idx = _resolve(path, "bounds.path")
            if param not in ("n", "deltar", "sigma2"):
                raise XAFSConfigError(
                    "FIT", f"bounds.param={param!r} 只支持 n / deltar / sigma2")
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                raise XAFSConfigError(
                    "FIT", f"bounds 区间非法: min={lo!r}, max={hi!r}")
            out.append({"kind": "bounds", "path": path, "path_index": idx,
                        "param": param, "min": lo, "max": hi})
            continue
        if kind not in ("n_sum", "n_ratio", "deltar_offset", "deltar_ratio"):
            raise XAFSConfigError(
                "FIT", f"未知约束 kind={kind!r}；只支持 n_sum / n_ratio / "
                       f"deltar_offset / deltar_ratio / bounds")
        try:
            target = str(spec["target"]).strip()
            source = str(spec["source"]).strip()
        except (KeyError, TypeError) as exc:
            raise XAFSConfigError(
                "FIT", f"{kind} 约束需要 target 与 source: {spec!r}") from exc
        ti = _resolve(target, f"{kind}.target")
        si = _resolve(source, f"{kind}.source")
        if ti == si:
            raise XAFSConfigError(
                "FIT", f"约束 target 与 source 同为 {target!r}，自约束无意义")
        rec = {"kind": kind, "target": target, "source": source,
               "target_index": ti, "source_index": si}
        try:
            if kind == "n_sum":
                total = float(spec["total"])
                coeff = float(spec.get("coeff", 1.0))
                if not np.isfinite(total) or total <= 0:
                    raise XAFSConfigError(
                        "FIT", f"n_sum.total={total!r} 必须 > 0")
                if not np.isfinite(coeff) or coeff <= 0:
                    raise XAFSConfigError(
                        "FIT", f"n_sum.coeff={coeff!r} 必须 > 0")
                rec.update(total=total, coeff=coeff)
            elif kind == "n_ratio":
                ratio = float(spec["ratio"])
                if not np.isfinite(ratio) or ratio < 0:
                    raise XAFSConfigError(
                        "FIT", f"n_ratio.ratio={ratio!r} 必须 ≥ 0")
                rec.update(ratio=ratio)
            elif kind == "deltar_ratio":
                ratio = float(spec["ratio"])
                # 负比例意味着远壳与第一壳反向弛豫，物理上可以成立，
                # 但必须有限且不能是 0（否则该壳 ΔR 被硬钉死、误差被抹掉）。
                if not np.isfinite(ratio) or abs(ratio) < 1e-12:
                    raise XAFSConfigError(
                        "FIT", f"deltar_ratio.ratio={ratio!r} 必须有限且非零")
                rec.update(ratio=ratio)
            else:
                offset = float(spec.get("offset", 0.0))
                if not np.isfinite(offset):
                    raise XAFSConfigError(
                        "FIT", f"deltar_offset.offset={offset!r} 非有限")
                rec.update(offset=offset)
        except KeyError as exc:
            raise XAFSConfigError(
                "FIT", f"{kind} 约束缺少字段 {exc.args[0]!r}") from exc
        out.append(rec)

    # ---- 同一参数被多条约束指向同一 target → 过约束，拒绝 ----
    seen_n: dict[int, int] = {}
    seen_dr: dict[int, int] = {}
    for rec in out:
        if rec["kind"] in ("n_sum", "n_ratio"):
            if rec["target_index"] in seen_n:
                raise XAFSConfigError(
                    "FIT", f"路径 {rec['target']!r} 的 N 被多条约束重复绑定")
            seen_n[rec["target_index"]] = rec["source_index"]
        elif rec["kind"] in ("deltar_offset", "deltar_ratio"):
            if rec["target_index"] in seen_dr:
                raise XAFSConfigError(
                    "FIT", f"路径 {rec['target']!r} 的 ΔR 被多条约束重复绑定")
            seen_dr[rec["target_index"]] = rec["source_index"]
    for mapping, what in ((seen_n, "N"), (seen_dr, "ΔR")):
        for start in mapping:
            seen = set()
            k = start
            while k in mapping:
                if k in seen:
                    raise XAFSConfigError(
                        "FIT", f"{what} 约束成环（存在自我依赖），拒绝执行")
                seen.add(k)
                k = mapping[k]
    return out


def _parse_constraint_expression(text: str) -> dict:
    """解析一条约束字符串表达式（正则，不使用 eval）。"""
    import re

    s = str(text).strip()
    m = re.fullmatch(
        r"N\(\s*(?P<a>[^()]+?)\s*\)\s*\+\s*N\(\s*(?P<b>[^()]+?)\s*\)"
        r"\s*=\s*(?P<c>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", s)
    if m:
        return {"kind": "n_sum", "source": m.group("a"),
                "target": m.group("b"), "total": float(m.group("c")),
                "coeff": 1.0}
    m = re.fullmatch(
        r"N\(\s*(?P<t>[^()]+?)\s*\)\s*=\s*"
        r"(?P<r>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\*\s*"
        r"N\(\s*(?P<s>[^()]+?)\s*\)", s)
    if m:
        return {"kind": "n_ratio", "target": m.group("t"),
                "source": m.group("s"), "ratio": float(m.group("r"))}
    # 【约束解析补齐】比例键长约束 dR(A) = r * dR(B)（比例在前/在后都支持）。
    # 必须放在 deltar_offset 正则之前：后者遇到 '*' 不会匹配，但顺序显式
    # 化可避免将来有人放宽 offset 正则时把比例式误解析成 offset=0。
    _num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    m = re.fullmatch(
        r"dR\(\s*(?P<t>[^()]+?)\s*\)\s*=\s*"
        rf"(?P<r>{_num})\s*\*\s*dR\(\s*(?P<s>[^()]+?)\s*\)", s, re.I)
    if m:
        return {"kind": "deltar_ratio", "target": m.group("t"),
                "source": m.group("s"), "ratio": float(m.group("r"))}
    m = re.fullmatch(
        r"dR\(\s*(?P<t>[^()]+?)\s*\)\s*=\s*"
        r"dR\(\s*(?P<s>[^()]+?)\s*\)\s*\*\s*"
        rf"(?P<r>{_num})", s, re.I)
    if m:
        return {"kind": "deltar_ratio", "target": m.group("t"),
                "source": m.group("s"), "ratio": float(m.group("r"))}
    m = re.fullmatch(
        r"dR\(\s*(?P<t>[^()]+?)\s*\)\s*=\s*dR\(\s*(?P<s>[^()]+?)\s*\)"
        r"(?:\s*(?P<sign>[+-])\s*(?P<off>(?:\d+\.?\d*|\.\d+)"
        r"(?:[eE][-+]?\d+)?))?", s, re.I)
    if m:
        off = float(m.group("off") or 0.0)
        if m.group("sign") == "-":
            off = -off
        return {"kind": "deltar_offset", "target": m.group("t"),
                "source": m.group("s"), "offset": off}
    raise XAFSConfigError(
        "FIT", f"无法解析约束表达式 {text!r}；支持 "
               f"'N(A)+N(B)=C' / 'N(B)=r*N(A)' / 'dR(B)=dR(A)+off' / "
               f"'dR(B)=r*dR(A)'")



#   → 24.254316 amu·Å²·K
HBAR2_OVER_2KB = 24.254316


def einstein_sigma2(temperature_K: float, theta_e: float,
                    reduced_mass_amu: float,
                    sigma2_static: float = 0.0) -> float:
    """爱因斯坦模型的热无序 σ²(T)（量子谐振子精确式）。

    数学:
        σ²(T) = σ²_static + (ħ² / (2 μ k_B Θ_E)) · coth(Θ_E / (2T))
        ħ²/(2 k_B) = 24.254316 amu·Å²·K

    物理:
        T → 0 时 coth → 1，保留零点振动；T ≫ Θ_E 时
        σ² ≈ σ²_static + 2·(24.254316/(μΘ_E))·(T/Θ_E)，线性于 T。
        高温近似不能替代本式——变温联合拟合的全部信息来自
        coth 的非线性温度依赖。

    参数:
        temperature_K   : 样品温度 (K)，必须 > 0
        theta_e         : 爱因斯坦温度 (K)，必须 > 0
        reduced_mass_amu: 键折合质量 (amu)，必须 > 0
        sigma2_static   : 静态无序 (Å²)，必须 ≥ 0

    返回:
        σ²(T) (Å²)
    """
    T = float(temperature_K)
    theta = float(theta_e)
    mu = float(reduced_mass_amu)
    static = float(sigma2_static)
    if not np.isfinite(T) or T <= 0:
        raise XAFSConfigError("FIT", f"温度 T={temperature_K!r} 必须 > 0 K")
    if not np.isfinite(theta) or theta <= 0:
        raise XAFSConfigError(
            "FIT", f"爱因斯坦温度 Θ_E={theta_e!r} 必须 > 0 K")
    if not np.isfinite(mu) or mu <= 0:
        raise XAFSConfigError(
            "FIT", f"折合质量 μ={reduced_mass_amu!r} 必须 > 0 amu")
    if not np.isfinite(static) or static < 0:
        raise XAFSConfigError(
            "FIT", f"σ²_static={sigma2_static!r} 不能为负")
    x = theta / (2.0 * T)
    thermal = (HBAR2_OVER_2KB / (mu * theta)) / np.tanh(x)
    return float(static + thermal)


def profile_chi2_contour(fit_func, x_name: str, y_name: str,
                         x_values, y_values, fixed: dict | None = None,
                         sigma_levels=(2.30, 6.18, 11.83)) -> dict:
    """B12-1 二维 χ² 剖面扫描（供诊断图等高线）。

    对 ``x_name × y_name`` 的每个网格点调用 ``fit_func(**params)``，
    其余参数由 ``fixed`` 固定。``fit_func`` 可返回标量 χ²、``FitResult``
    或 ``(chi2, ...)`` 元组。``levels`` 是相对最小值的 Δχ² 阈值
    （二维 1σ/2σ/3σ 分别 ≈2.30/6.18/11.83）。
    """
    xv = np.asarray(x_values, dtype=float)
    yv = np.asarray(y_values, dtype=float)
    if xv.ndim != 1 or yv.ndim != 1 or xv.size < 2 or yv.size < 2:
        raise XAFSConfigError("PROFILE", "x_values/y_values 必须是一维且 ≥2 点")
    fixed = dict(fixed or {})
    grid = np.full((yv.size, xv.size), np.nan, dtype=float)
    for iy, yv_i in enumerate(yv):
        for ix, xv_i in enumerate(xv):
            params = dict(fixed)
            params[x_name] = float(xv_i)
            params[y_name] = float(yv_i)
            try:
                res = fit_func(**params)
            except Exception:  # noqa: BLE001 - 失败网格保留 NaN，不伪造
                continue
            if hasattr(res, "chi_square"):
                val = res.chi_square
            elif isinstance(res, (tuple, list)):
                val = res[0]
            else:
                val = res
            try:
                grid[iy, ix] = float(val)
            except (TypeError, ValueError):
                grid[iy, ix] = np.nan
    finite = grid[np.isfinite(grid)]
    levels = ([float(np.min(finite) + d) for d in sigma_levels]
              if finite.size else [])
    return {"x": xv, "y": yv, "chi2_grid": grid,
            "min_chi2": float(np.min(finite)) if finite.size else None,
            "levels": levels, "delta_chi2": list(sigma_levels)}
