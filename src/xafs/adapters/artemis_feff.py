# -*- coding: utf-8 -*-
"""artemis_feff.py — #1 FEFF 输入生成（含 CIF 读取、SAC/DAC 团簇）+ #2 FEFF 运行与路径读取（自 artemis.py 纯拆分）

纯搬运拆分：代码块与原 artemis.py 逐行一致，不含任何逻辑改动。
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

import athena
from xafs_core import (FitResult, PathFit, XAFSConfigError, XAFSDataError)


# ================================================================
# 1. FEFF 输入生成
# ================================================================
#
# 格式必须以 Artemis 导出的 feff.inp 为基准，以下是硬约束：
#   - 段名是 POTENTIALS（复数），三列表格：ipot  Z  element
#   - ATOMS 行 6 字段：x  y  z  ipot  tag  distance
#   - CONTROL / PRINT 各 4 个参数（不是 6 个）
#   - 无 EDGE 关键字，吸收边由 HOLE 卡指定；吸收原子 ipot=0
#     HOLE 1=K, 2=L1, 3=L2, 4=L3（FEFF8 约定）
# 曾经写成 POTENTIAL（单数）+ 两行 "26 Fe 2.0"，FEFF 把 26 当成势编号，
# 报 "Unique potential 26 not allowed"，连续 6 种写法全被拒。

_ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(_ELEMENTS, start=1)}
HOLE_INDEX = {"K": 1, "L1": 2, "L2": 3, "L3": 4}

# 共价半径 (Å)，Cordero et al., Dalton Trans. 2008。只收 XAFS 常见元素；
# 缺失的键返回 None，调用方必须显式降级而不是猜。
# 用途：L1 第一壳的**峰值选择**（不是最终键长）——预期 M–L 键长
# = r(absorber)+r(ligand)，再减相位偏移，用来在多个显著峰中选最像
# 第一壳的那个。实测：Fe-O 1.98（hematite 1.35 峰而非 Fe-Fe 2.55）、
# Pt-Cl 2.38（1.90 峰而非低 R 白线伪峰 1.44）。
COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20,
    "I": 1.39,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46,
    "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "La": 2.07, "Ce": 2.04, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "U": 1.96,
}

# BVS 参考键长 R₀ (Å)，Brown & Altermatt, Acta Cryst. B41 (1985) 244。
# 键写法同时接受 "Fe-O" 与 ("Fe", "O")；b=0.37 Å 是同一文献的通用值。
BVS_R0 = {
    "Fe-O": 1.759, "Fe-N": 1.90, "Fe-S": 2.20, "Fe-Cl": 2.29,
    "Co-O": 1.692, "Co-N": 1.85, "Co-S": 2.16,
    "Ni-O": 1.654, "Ni-N": 1.83, "Ni-S": 2.13,
    "Cu-O": 1.679, "Cu-N": 1.85, "Cu-S": 2.15, "Cu-Cl": 2.24,
    "Mn-O": 1.790, "Mn-N": 1.95, "Cr-O": 1.720, "Zn-O": 1.704,
    "Mo-O": 1.860, "Mo-S": 2.32, "Ru-O": 1.834, "Rh-O": 1.820,
    "Pd-O": 1.792, "Ag-O": 1.842, "Pt-Cl": 2.190, "Pt-O": 1.820,
    "Ir-O": 1.820, "Au-Cl": 2.220, "Ce-O": 2.340, "U-O": 2.300,
}


def expected_bond_length(absorber: str, ligand: str) -> float | None:
    """共价半径和作为 M–L 键长初估；任一元素缺半径返回 None。"""
    ra = COVALENT_RADII.get(absorber)
    rl = COVALENT_RADII.get(ligand)
    if ra is None or rl is None:
        return None
    return ra + rl


def build_feff_input(sites, absorber: str | None = None, rmax: float = 6.0,
                     title: str = "auto", edge: str = "K") -> str:
    """
    由中心原子 + 近邻坐标生成 feff.inp 文本。

    参数:
        sites    : [(x, y, z, 元素符号), ...] 笛卡尔坐标（Å），中心固定在原点
        absorber : 中心原子元素符号
        rmax     : 团簇半径（Å），FEFF 据此扩展周期
        edge     : 吸收边 "K" / "L1" / "L2" / "L3"（决定 HOLE 卡）

    返回: feff.inp 的完整文本
    """
    if not absorber:
        raise XAFSConfigError("FEFF", "未指定吸收元素（absorber）——生成 FEFF 输入必需")
    if absorber not in ATOMIC_NUMBER:
        raise XAFSConfigError("FEFF", f"未知元素 {absorber}，请补进 ATOMIC_NUMBER")
    if edge not in HOLE_INDEX:
        raise XAFSConfigError(
            "FEFF", f"未知吸收边 {edge!r}；可选 {sorted(HOLE_INDEX)}")
    if not sites:
        raise XAFSConfigError("FEFF", "sites 为空")

    valid_sites = []
    for site in sorted(sites, key=lambda s: sum(v * v for v in s[:3])):
        x, y, z = site[0], site[1], site[2]
        distance = float(np.sqrt(x * x + y * y + z * z))
        if distance < 0.1 or distance > rmax:
            continue
        symbol = str(site[3]).strip().capitalize() if len(site) > 3 else absorber
        valid_sites.append((x, y, z, symbol, distance))
    if not valid_sites:
        raise XAFSConfigError("FEFF", f"0.1 <= r <= {rmax} Å 内无有效散射原子")

    scatterers = []
    for _, _, _, symbol, _ in valid_sites:
        if symbol not in scatterers:
            scatterers.append(symbol)

    potential_lines = [f"     0    {ATOMIC_NUMBER[absorber]:3d}   {absorber}"]
    for index, symbol in enumerate(scatterers, start=1):
        if symbol not in ATOMIC_NUMBER:
            raise XAFSConfigError("FEFF", f"未知元素 {symbol}")
        potential_lines.append(f"    {index:3d}    {ATOMIC_NUMBER[symbol]:3d}   {symbol}")

    atom_lines = [f"   0.00000    0.00000    0.00000  0    {absorber:<4s}   0.00000"]
    for x, y, z, symbol, distance in valid_sites:
        ipot = scatterers.index(symbol) + 1
        atom_lines.append(
            f"  {x:9.5f}  {y:9.5f}  {z:9.5f}  {ipot}    {symbol:<4s}  {distance:9.5f}")

    return ("* feff.inp generated by artemis.py\n"
            f"TITLE {title}\n\n"
            f"HOLE {HOLE_INDEX[edge]}   1.0\n"
            "*         mphase,mpath,mfeff,mchi\n"
            "CONTROL   1      1     1     1\n"
            "PRINT     1      0     0     0\n\n"
            f"RMAX  {rmax:.1f}\n"
            "CORRECTIONS  0.0   0.0\n\n"
            "POTENTIALS\n"
            "*    ipot   Z  element\n"
            + "\n".join(potential_lines) + "\n\n"
            "ATOMS\n"
            "*   x          y          z      ipot  tag      distance\n"
            + "\n".join(atom_lines) + "\n")


def bcc_lattice_sites(a: float = 2.866, rmax: float = 6.0, symbol: str = "Fe"):
    """体心立方金属（Fe 箔等）的近邻坐标。a=2.866 Å 时第一壳 2.482 Å ×8。"""
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    n = int(rmax / a) + 2
    sites = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                for b in basis:
                    p = ((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a)
                    d = float(np.sqrt(sum(v * v for v in p)))
                    if 0.1 < d <= rmax:
                        sites.append((*p, symbol))
    return sites


def octahedron_sites(bond_length: float = 2.00, symbol: str = "O"):
    """单个 MO6 正八面体，用于无晶体结构时生成第一壳层路径。"""
    sites = []
    for axis in range(3):
        for sign in (+1, -1):
            p = [0.0, 0.0, 0.0]
            p[axis] = sign * bond_length
            sites.append((p[0], p[1], p[2], symbol))
    return sites


def read_cif_sites(cif_path, absorber: str | None = None, rmax: float = 6.0,
                   ignore=("H", "D"), absorber_fract=None,
                   with_occupancy: bool = False,
                   return_occupancy: bool = False,
                   merge_cocupancy: bool = True) -> list:
    """
    读 CIF 晶体结构，展开对称等价位置，返回以吸收原子为原点的近邻坐标。

    数学/流程:
        不对称单元 → 空间群操作展开 → 全晶胞原子
        → 分数坐标 × 晶胞矩阵 → 笛卡尔坐标（Å）
        → 吸收原子平移到原点，沿 a/b/c 周期平移补全 rmax 内的像

    参数:
        cif_path : CIF 文件路径
        absorber : 吸收原子元素符号（取第一个匹配位点作原点）
        rmax     : 团簇半径（Å），传给 build_feff_input 时保持一致
        ignore   : 忽略的元素。默认跳过 H/D——氢的背散射振幅可忽略，
                   且粉晶 CIF 里 H 的坐标常不可靠
        with_occupancy : True 时返回 (sites, occs)，occs 与 sites 平行
        return_occupancy : True 时返回 5 元组 (x, y, z, symbol, occ)；
                   默认 False 保持历史 4 元组，100% 向后兼容

    部分占据/共占据（B7-2）:
        - 读取 gemmi 的 ``site.occ`` 并钳制到 [0, 1]；
        - 展开后距离 < 0.08 Å 的散射体视为同一晶位共占据，合并为
          一个位点：元素取占据度较大者，有效占据度取
          ``min(1, Σocc)``——防止 FEFF 因两个原子坐标重合崩溃；
        - 吸收中心有多个同元素位点且未给 absorber_fract 时，按
          ``occ × 多重度`` 选择主位点（多重度 = 展开后的等价位点数）。

    限制: 默认 4 元组模式只保证几何正确，N 仍按满占据解读；
        占据敏感的定量必须用 with_occupancy/return_occupancy 通道。
        ``merge_cocupancy=False`` 时保留同一晶位的不同元素分支（B11-1
        异核共占位），不把 Ti/Ru 合并成单一元素近似。
    """
    import re

    import gemmi

    if not absorber:
        raise XAFSConfigError("CIF", "未指定吸收元素（absorber）——选吸收位必需")
    absorber = absorber.capitalize()
    ignored = {s.capitalize() for s in ignore}

    try:
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block() if len(doc) == 1 else doc[0]
        st = gemmi.make_small_structure_from_block(block)
        cell = st.cell
        all_sites = st.get_all_unit_cell_sites()
    except (RuntimeError, ValueError, IndexError) as exc:
        raise XAFSDataError(
            "CIF", f"解析 {Path(cif_path).name} 失败：{exc}。"
                   f"检查晶胞参数 / 空间群 / _atom_site_* 循环是否完整") from exc

    def _symbol(site):
        m = re.match(r"[A-Za-z]+", site.type_symbol.strip())
        return m.group(0).capitalize() if m else None

    def _occ(site):
        try:
            value = float(getattr(site, "occ", 1.0))
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(value):
            return 1.0
        return float(min(1.0, max(0.0, value)))

    def _cart_site(site):
        p = cell.orthogonalize(site.fract)
        return np.array([p.x, p.y, p.z])

    # ---- 吸收中心：无显式分数坐标时按 occ × 多重度选主位点 ----
    abs_site = None
    if absorber_fract is None:
        cands = [s for s in all_sites if _symbol(s) == absorber]
        if cands:
            # 以分数坐标 mod 1 分组（容差换算成笛卡尔 0.08 Å）
            groups = []
            for s in cands:
                f = np.array([s.fract.x, s.fract.y, s.fract.z], dtype=float)
                placed = False
                for g in groups:
                    d = f - g["f"]
                    d -= np.round(d)
                    p = cell.orthogonalize(gemmi.Fractional(*d))
                    if np.linalg.norm([p.x, p.y, p.z]) < 0.08:
                        g["members"].append(s)
                        placed = True
                        break
                if not placed:
                    groups.append({"f": f, "members": [s]})
            if groups:
                def _group_weight(g):
                    mult = len(g["members"])
                    occ_sum = sum(_occ(s) for s in g["members"]) / mult
                    return occ_sum * mult
                abs_site = max(groups, key=_group_weight)["members"][0]
    else:
        fx, fy, fz = (float(v) for v in absorber_fract)
        cands = [s for s in all_sites if _symbol(s) == absorber]

        def _d2(site):
            d = np.array([site.fract.x - fx, site.fract.y - fy,
                          site.fract.z - fz])
            d -= np.round(d)          # 最近像约定
            p = cell.orthogonalize(gemmi.Fractional(*d))
            return p.x ** 2 + p.y ** 2 + p.z ** 2

        abs_site = min(cands, key=_d2) if cands else None
    if abs_site is None:
        found = sorted({sym for sym in map(_symbol, all_sites) if sym})
        raise XAFSDataError(
            "CIF", f"{Path(cif_path).name} 中找不到元素 {absorber}，"
                   f"文件里只有 {found}")

    origin = _cart_site(abs_site)

    def _lattice_vec(fx, fy, fz):
        p = cell.orthogonalize(gemmi.Fractional(fx, fy, fz))
        return np.array([p.x, p.y, p.z])

    a_vec = _lattice_vec(1, 0, 0)
    b_vec = _lattice_vec(0, 1, 0)
    c_vec = _lattice_vec(0, 0, 1)

    # 周期像范围：rmax / 轴长 + 1 圈保险
    ranges = []
    for vec in (a_vec, b_vec, c_vec):
        n = int(np.ceil(rmax / max(np.linalg.norm(vec), 1e-6))) + 1
        ranges.append(range(-n, n + 1))

    # ---- 展开 + 共占位合并（先按距离筛选，再合并，避免 O(N²) 爆炸）----
    # 按占据度降序排列，保证同一晶位保留占据度较大的元素符号。
    raw = []
    for site in all_sites:
        sym = _symbol(site)
        if sym is None or sym in ignored:
            continue
        base = _cart_site(site) - origin
        occ = _occ(site)
        for i, j, k in ((i, j, k) for i in ranges[0] for j in ranges[1]
                        for k in ranges[2]):
            pos = base + i * a_vec + j * b_vec + k * c_vec
            d = float(np.linalg.norm(pos))
            if 0.1 < d <= rmax:
                raw.append((pos, sym, occ, d))
    raw.sort(key=lambda item: (-item[2], item[1]))

    # 空间哈希：网格边长 0.08 Å，只需比较相邻 3³ 个桶。
    cell_size = 0.08
    buckets = {}
    merged = []      # [{"pos": ndarray, "sym": str, "occ": float, "d": float}]
    for pos, sym, occ, d in raw:
        key = tuple(int(v) for v in np.floor(pos / cell_size))
        match = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for m in buckets.get((key[0] + dx, key[1] + dy,
                                          key[2] + dz), ()):
                        if float(np.linalg.norm(pos - m["pos"])) < 0.08:
                            if merge_cocupancy or m["sym"] == sym:
                                match = m
                                break
                    if match is not None:
                        break
                if match is not None:
                    break
            if match is not None:
                break
        if match is None:
            item = {"pos": pos, "sym": sym, "occ": occ, "d": d}
            merged.append(item)
            buckets.setdefault(key, []).append(item)
        else:
            if merge_cocupancy:
                match["occ"] = float(min(1.0, match["occ"] + occ))
            else:
                # 分支模式：同一元素同一位置的周期像不重复计占据。
                match["occ"] = float(max(match["occ"], occ))
    merged.sort(key=lambda m: (m["d"], m["sym"]))

    sites = []
    occs = []
    for m in merged:
        sites.append((float(m["pos"][0]), float(m["pos"][1]),
                      float(m["pos"][2]), m["sym"]))
        occs.append(float(m["occ"]))
    if return_occupancy:
        return [(x, y, z, sym, occ)
                for (x, y, z, sym), occ in zip(sites, occs)]
    return (sites, occs) if with_occupancy else sites


def cif_hetero_cocupancy_branches(cif_path, absorber: str, rmax: float = 6.0,
                                  delta_z: int = 3,
                                  occ_min: float = 0.10) -> dict:
    """B11-1 异核共占位的双分支（Ti/Ru、Fe/Ni 等）。

    普通 ``read_cif_sites`` 在 <0.08 Å 的同一晶位上只保留占据度较大的
    元素，等效于“单元素近似”。真实催化剂的共占位位点必须拆成两支：
    每个元素有自己的散射振幅和占据度，但共享同一个 R 与 σ²（在拟合层
    用 ``shared_bonds`` 绑定）。

    返回::

        {"single_element_approx": bool,
         "branches": [{"distance", "element", "occ", "degen",
                       "shell", "n_geom"}, ...],
         "shells": [{"distance", "n_geom", "branches": [...]}, ...]}

    简并度约定：同一晶位的几何多重度 N_geom 按占据度分给各元素，
    ``Σ degen = N_geom``。只保留 |Z_a−Z_b|≥delta_z 且 occ≥occ_min 的
    异核分支；同元素共占位按单元素处理。
    """
    sites5 = read_cif_sites(cif_path, absorber=absorber, rmax=rmax,
                            return_occupancy=True, merge_cocupancy=False)
    if not sites5:
        return {"single_element_approx": True, "branches": [], "shells": []}

    # 先按空间位置聚类：同一晶位 = 距离相同且坐标差 <0.08 Å。
    sites = []
    for (x, y, z, sym, occ) in sites5:
        pos = np.array([x, y, z], dtype=float)
        d = float(np.linalg.norm(pos))
        if d <= 0.1 or occ < float(occ_min):
            continue
        sites.append({"pos": pos, "element": str(sym), "occ": float(occ),
                      "distance": d})
    if not sites:
        return {"single_element_approx": True, "branches": [], "shells": []}
    sites.sort(key=lambda s: s["distance"])

    shells = []
    for s in sites:
        placed = False
        for sh in shells:
            # 粉末 EXAFS 的“壳层”按 Reff 归并，与方向无关；同一距离
            # 上的不同晶位（不同元素分支）必须留在同一壳层里，
            # 否则简并度会被拆成 N 个 1，BVS/占位统计全错。
            if abs(sh["distance"] - s["distance"]) < 0.12:
                sh["members"].append(s)
                placed = True
                break
        if not placed:
            shells.append({"distance": s["distance"], "ref_pos": s["pos"],
                           "members": [s]})

    branches = []
    out_shells = []
    any_hetero = False
    for si, sh in enumerate(shells):
        # 几何简并度 = **不同晶位**数，而不是展开后的原子条目数。
        # 同一晶位列了 Ti(0.7)+Ru(0.3) 时，对称展开会得到 8 个 Ti 镜像
        # 和 8 个 Ru 镜像；按条目计 n_geom=16 会把占据度重复乘一遍。
        unique_pos: list[np.ndarray] = []
        for m in sh["members"]:
            if not any(np.linalg.norm(m["pos"] - u) < 0.08
                       for u in unique_pos):
                unique_pos.append(m["pos"])
        n_geom = len(unique_pos)
        by_el: dict[str, dict] = {}
        for m in sh["members"]:
            el = m["element"]
            by_el.setdefault(el, {"occ_sum": 0.0, "count": 0})
            by_el[el]["occ_sum"] += m["occ"]
            by_el[el]["count"] += 1
        hetero = False
        if len(by_el) >= 2:
            zs = sorted(ATOMIC_NUMBER.get(el, 0) for el in by_el)
            hetero = (zs[-1] - zs[0]) >= int(delta_z)
        any_hetero = any_hetero or hetero
        shell_branches = []
        for el, agg in sorted(by_el.items()):
            occ_mean = agg["occ_sum"] / max(agg["count"], 1)
            branch = {
                "distance": float(sh["distance"]),
                "element": el,
                "occ": float(occ_mean),
                "degen": float(n_geom * occ_mean),
                "shell": int(si),
                "n_geom": int(n_geom),
            }
            shell_branches.append(branch)
            branches.append(branch)
        out_shells.append({
            "distance": float(sh["distance"]),
            "n_geom": int(n_geom),
            "single_element_approx": bool(not hetero),
            "branches": shell_branches,
        })

    return {"single_element_approx": bool(not any_hetero),
            "branches": branches, "shells": out_shells}


# 【并发安全】相位来源原先只写在函数属性上，属于进程级"最后一次调用"
# 状态：多线程/协程并发时会把 A 路径的来源记到 B 结果上。现在改为：
#   - 每次调用通过 ``source_out`` 把来源返回给调用方（准确、无共享状态）；
#   - thread-local 保存本线程最近一次来源；
#   - 函数属性 ``last_source`` 仅作向后兼容的诊断/回归断言，用锁保护写入。
_PHASE_SOURCE_LOCK = threading.Lock()
_PHASE_SOURCE_LOCAL = threading.local()


def _set_phase_source(value: str, source_out: dict | None) -> None:
    _PHASE_SOURCE_LOCAL.value = value
    if source_out is not None:
        source_out["source"] = value
    with _PHASE_SOURCE_LOCK:
        path_total_phase.last_source = value


def phase_source_for_current_thread() -> str | None:
    """返回本线程最近一次 ``path_total_phase`` 的来源（并发安全）。"""
    return getattr(_PHASE_SOURCE_LOCAL, "value", None)


def path_total_phase(datfile, k, source_out: dict | None = None):
    """B11-1 返回一条 FEFF 路径的总相移 φ_tot(k)=2δ_c+φ_s。

    优先使用 FEFF 文件自带的 ``pha`` 表（FeffDatFile.pha），按请求的
    k 网格线性插值；老版本/裁剪文件没有 ``pha`` 时，退化为对 ``chi``
    做 Hilbert 解析信号提取相位，并显式记录来源。

    ``source_out``：可选字典，本次调用会把 ``{"source": "feff_pha" |
    "hilbert_fallback"}`` 写入其中。**并发调用请使用这个通道**；
    ``path_total_phase.last_source`` 是"全进程最后一次调用"的兼容属性，
    只适合单线程诊断与回归断言，不能用来给并发结果归属来源。

    注意：相位只定义到一个 2π 常数；FT 只关心 e^{−iφ}，常数相位不改变
    峰位，因此不需要绝对相位零点。
    """
    from larch.xafs import feffpath

    k = np.asarray(k, dtype=float)
    p = feffpath(str(datfile))
    p.create_path_params()
    fd = getattr(p, "_feffdat", None)
    if fd is not None and hasattr(fd, "pha") and hasattr(fd, "k"):
        phi = np.interp(k, np.asarray(fd.k), np.asarray(fd.pha))
        _set_phase_source("feff_pha", source_out)
        return phi
    p._calc_chi()
    from scipy.signal import hilbert
    chi = np.asarray(p.chi, dtype=float)
    kp = np.asarray(p.k, dtype=float)
    z = hilbert(chi)
    arg = np.unwrap(np.angle(z))
    phi = np.interp(k, kp, arg - 2.0 * kp * float(p.reff))
    _set_phase_source("hilbert_fallback", source_out)
    return phi


path_total_phase.last_source = None


def cif_shell_occupancy_scale(cif_path, absorber: str,
                              rmax: float = 6.0,
                              absorber_fract=None) -> list:
    """返回 CIF 展开后每个近邻的 ``(distance, element, occ)``。

    供 ``xafs_fit._ss_rows`` 按距离/元素匹配壳层平均占据度，把 FEFF
    几何简并度缩放为 ``N_cryst = N_geom·mean(occ)``。距离是几何
    距离（与 FEFF 的 Reff 同一口径，不含相移）。
    """
    sites5 = read_cif_sites(cif_path, absorber=absorber,
                            absorber_fract=absorber_fract, rmax=rmax,
                            return_occupancy=True)
    return [(float(np.linalg.norm([x, y, z])), str(sym), float(occ))
            for (x, y, z, sym, occ) in sites5]


# ================================================================
# 2. FEFF 计算与路径读取
# ================================================================

def run_feff6l(feffinp, workdir) -> list:
    """
    调用 feff6l.exe，返回生成的 feffNNNN.dat 列表。

    feffinp 可以是文本内容，也可以是 .inp 文件路径。
    """
    from larch.xafs import feff6l

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "feff.inp"
    text = feffinp if "\n" in str(feffinp) else Path(feffinp).read_text()
    target.write_text(text, encoding="utf-8")

    feff6l(feffinp=str(target.resolve()), folder=str(workdir.resolve()), verbose=False)
    files = sorted(workdir.glob("feff0*.dat"))
    if not files:
        raise XAFSDataError("FEFF", f"{workdir} 未生成任何 feffNNNN.dat，"
                                     f"检查 feff.inp 的 POTENTIALS / ATOMS 段")
    return files


def load_feff_path(datfile):
    """
    读入一条散射路径。

    返回 FeffPathGroup，其 params 需要用 set_path_parameters 显式写入。
    """
    from larch.xafs import feffpath

    path = feffpath(str(datfile))
    path.create_path_params()
    return path


def describe_paths(datfiles, max_show: int = 6) -> list:
    """列出路径的 (文件, Reff, 简并度, 散射腿数)，用于挑选壳层。"""
    from larch.xafs import feffpath

    summary = []
    for f in datfiles[:max_show]:
        p = feffpath(str(f))
        summary.append({"file": Path(f).name,
                        "reff": round(float(p.reff), 4),
                        "degeneracy": float(p.degen),
                        "legs": int(p.nleg)})
    return summary
