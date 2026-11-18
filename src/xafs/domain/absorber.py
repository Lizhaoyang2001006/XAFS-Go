# -*- coding: utf-8 -*-
"""
absorber.py — 吸收元素配置（元素泛化的唯一入口）
================================================================
物理计算层（athena/artemis/larch_wt）本身元素无关；元素特异的全在这里：

    symbol        元素符号
    edge          吸收边（"K" / "L3"）
    e0_edge       边的标称能量 (eV)，E0 初值与对齐基准
    s02           S₀² 标定值。**None = 未标定**——此时自动拟合必须拒绝
                  启动（S₀² 与 N 全相关，没有标定就没有 CN 定量）
    kmax_ceiling  k 上限 (Å⁻¹)。L 边被下一条边硬截断
                  （如 Pt L₃ 受 L₂ 边限制），K 边取实用上限
    next_edge     下一条边的 (名称, 能量)，供诊断展示

新元素接入流程：
    1. 查边能量填表（s02=None）
    2. 测该元素标准样（金属箔或已知结构），标定 s02 填入
    3. 向 _structure_lib/<symbol>/ 挂 motif
    代码本身零改动——这是泛化设计是否成立的验收标准。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from xafs_core import K2, XAFSConfigError


@dataclass(frozen=True)
class AbsorberConfig:
    symbol: str
    z: int
    edge: str
    e0_edge: float
    s02: float | None
    kmax_ceiling: float
    next_edge: tuple | None = None

    @property
    def calibrated(self) -> bool:
        return self.s02 is not None


# K 边 kmax_ceiling 取 15（数据质量通常先于物理极限恶化）；
# L 边按下一吸收边估算后取保守值。
ABSORBERS = {
    # 生产值 0.803：氧化物价态验收锚定的跨材料折中值（原始流程不可复原）。
    # 金属 Fe 交叉标定（2026-11-10）: BMM Fe-K-IronFoil.xdi, fe_bcc 单壳
    # N=8, E0=half_height → 0.6566（3 窗口 0.6471/0.6566/0.6676，极差
    # 0.0205，Rf≤0.0422；IIT-Lytle 5 µm 箔交叉 0.6335）。该值只适用于
    # 金属 Fe，不替换 0.803——否则 Fe 氧化物 CN 会高约 22%。
    "Fe": AbsorberConfig("Fe", 26, "K", 7112.0, 0.803, 15.0),
    "Co": AbsorberConfig("Co", 27, "K", 7709.0, 0.7308, 15.0),
    # 标定: Co 箔 co_hcp N=12, Rf=0.0001, 2026-11-05
    "Ni": AbsorberConfig("Ni", 28, "K", 8333.0, 0.7795, 15.0),
    # 标定: Ni 箔 ni_fcc N=12, Rf=0.0002, 2026-11-05
    "Cu": AbsorberConfig("Cu", 29, "K", 8979.0, 0.8670, 15.0),
    # 标定: Cu 箔 cu_fcc N=12, Rf=0.0006, 2026-11-05
    "Pt": AbsorberConfig("Pt", 78, "L3", 11564.0, 0.8075, 16.0,
    # 标定: Pt 箔 pt_fcc N=12, Rf=0.0004, 2026-11-05
                         next_edge=("L2", 13273.0)),
    # ---- Phase III：常用催化/环境元素。全部 s02=None（未标定），
    # 自动 CN 定量必须拒绝；标定后逐元素填入实测值。
    "Cr": AbsorberConfig("Cr", 24, "K", 5989.0, None, 15.0),
    "Mn": AbsorberConfig("Mn", 25, "K", 6539.0, None, 15.0),
    "Zn": AbsorberConfig("Zn", 30, "K", 9659.0, None, 15.0),
    "Mo": AbsorberConfig("Mo", 42, "K", 20000.0, None, 16.0),
    "Ru": AbsorberConfig("Ru", 44, "K", 22117.0, None, 16.0),
    "Rh": AbsorberConfig("Rh", 45, "K", 23220.0, None, 16.0),
    "Pd": AbsorberConfig("Pd", 46, "K", 24350.0, None, 16.0),
    "Ag": AbsorberConfig("Ag", 47, "K", 25514.0, None, 16.0),
    "Ir": AbsorberConfig("Ir", 77, "L3", 11215.0, None, 16.0,
                         next_edge=("L2", 12824.0)),
    "Au": AbsorberConfig("Au", 79, "L3", 11919.0, None, 16.0,
                         next_edge=("L2", 13734.0)),
    # 【P1.1】Ce L3(5723 eV) → L2(6164 eV) 仅隔 441 eV，
    # k_max = sqrt(K2·ΔE) = sqrt(0.2624685×441) = 10.76 Å⁻¹。
    # 旧值 12.0 会把 L2 边的强阶跃卷入 EXAFS 振荡区，产生不可扣除的
    # 低频背景。取 10.2 留约 45 eV 边前缓冲（边前白线/预边结构）。
    "Ce": AbsorberConfig("Ce", 58, "L3", 5723.0, None, 10.2,
                         next_edge=("L2", 6164.0)),
    "U": AbsorberConfig("U", 92, "L3", 17166.0, None, 16.0,
                        next_edge=("L2", 20948.0)),
}


# 荧光自吸收校正默认荧光线：K→Kα，L3→Lα，L2→Lβ1，L1→Lβ3。
# 用户可用 --self-absorption-line 覆盖（例如探测器只收 Kβ）。
_FLUORESCENCE_LINES = {"K": "Ka", "L3": "La", "L2": "Lb1", "L1": "Lb3"}


def default_fluorescence_line(edge: str) -> str:
    try:
        return _FLUORESCENCE_LINES[edge.upper()]
    except KeyError:
        raise XAFSConfigError(
            "S4.5", f"边 {edge} 没有默认荧光线；用 --self-absorption-line "
                    f"显式指定（K→Ka, L3→La, L2→Lb1, L1→Lb3）") from None


def physical_kmax_limit(symbol: str, *, safety_eV: float = 0.0) -> float | None:
    """由下一条吸收边推导光电子波矢的物理上限 (Å⁻¹)。

    光电子动能 ``E - E0 = ħ²k²/(2m_e)``，即 ``k = sqrt(K2·(E − E0))``，
    ``K2 = 2m_e/ħ² = 0.2624685 Å⁻²·eV⁻¹``。L3 边被紧邻的 L2 边硬截断：
    超过 ``E(L2)`` 的数据已不是该边的 EXAFS，而是下一条边的阶跃与白线，
    会在 χ(k) 里留下无法用 AUTOBK 扣除的低频背景。

    例：Ce L3 5723 eV → L2 6164 eV，ΔE=441 eV，
    ``k_max = sqrt(0.2624685 × 441) = 10.76 Å⁻¹``。

    无 ``next_edge`` 的 K 边返回 ``None``——其上限由数据质量而非边的
    物理截断决定，仍用静态 ``kmax_ceiling``。
    """
    cfg = get_absorber(symbol)
    if cfg.next_edge is None:
        return None
    delta = float(cfg.next_edge[1]) - float(cfg.e0_edge) - float(safety_eV)
    if delta <= 0.0:
        return 0.0
    return float(math.sqrt(K2 * delta))


def effective_kmax_limit(symbol: str) -> float:
    """生产链实际允许的最大 k：静态 ceiling 与物理上限取较小者。

    静态表里的 ``kmax_ceiling`` 是"实用"上限（数据信噪比先恶化），物理上限
    是"硬"上限（下一条边）。二者必须同时满足，任一被突破都会污染 χ(k)。
    """
    cfg = get_absorber(symbol)
    phys = physical_kmax_limit(symbol)
    return float(min(cfg.kmax_ceiling, phys)) if phys is not None \
        else float(cfg.kmax_ceiling)


def get_absorber(symbol: str) -> AbsorberConfig:
    """按元素符号取配置；未收录元素显式报错而不是静默猜参数。"""
    key = symbol.strip().capitalize()
    if key not in ABSORBERS:
        raise XAFSConfigError(
            "absorber", f"元素 {symbol} 未收录。已收录: {sorted(ABSORBERS)}。"
                        f"新元素需按 absorber.py 头注的三步流程接入")
    return ABSORBERS[key]


# 化学态多态：静态表 ``s02`` 仍是默认生产值；只有显式选择化学态时才
# 返回态特异值。当前仅 Fe 金属相有真实箔交叉标定（0.6566），氧化物/
# 羟基氧化物继续沿用默认 0.803。新增态必须通过 calibrate_s02.py 门禁。
S02_CHEMICAL_STATES: dict[str, dict[str, float]] = {
    "Fe": {"metal": 0.6566},
}


def resolve_s02(symbol: str, chemical_state: str | None = None) -> float | None:
    """解析元素 S₀²；``chemical_state=None`` 等同 ``get_absorber().s02``。

    化学态仅作显式 opt-in，不修改静态表（零退化）。未知态、或该元素
    没有对应态标定时抛 ``XAFSConfigError``，不静默退回默认值。
    """
    cfg = get_absorber(symbol)
    if chemical_state is None:
        return cfg.s02
    state = str(chemical_state).strip().lower()
    states = S02_CHEMICAL_STATES.get(cfg.symbol, {})
    if state not in states:
        available = ", ".join(sorted(states)) or "无"
        raise XAFSConfigError(
            "absorber",
            f"{cfg.symbol} 没有化学态 {chemical_state!r} 的 S₀² 标定；"
            f"可用化学态: {available}。未标定元素请显式传 --s02")
    return states[state]


def resolve_s02_spec(symbol: str, spec: str | None
                     ) -> tuple[float | None, str | None]:
    """解析 CLI 化学态表达式，返回 ``(s02, source)``。

    支持两种写法：
    - ``metal``：全局态，应用到当前元素；无该态标定则报错；
    - ``Fe=metal,Co=metal``：逐元素映射，当前元素未列出时返回
      ``(None, None)``，由调用方继续走 T5/静态表优先级。
    """
    if spec is None or not str(spec).strip():
        return None, None
    text = str(spec).strip()
    cfg = get_absorber(symbol)
    if "=" not in text:
        return (resolve_s02(cfg.symbol, text),
                f"chemical_state:{cfg.symbol}={text.strip().lower()}")
    mapping: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise XAFSConfigError(
                "absorber",
                f"--chemical-state 混用格式: {text!r}；请统一用 "
                f"'metal' 或 'Fe=metal,Co=metal'")
        key, state = part.split("=", 1)
        try:
            el = get_absorber(key).symbol
        except XAFSConfigError:
            raise XAFSConfigError(
                "absorber", f"--chemical-state 未知元素: {key!r}") from None
        mapping[el] = state.strip().lower()
    state = mapping.get(cfg.symbol)
    if state is None:
        return None, None
    return (resolve_s02(cfg.symbol, state),
            f"chemical_state:{cfg.symbol}={state}")
