#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""catalytic_builder_v2.py — 晶体库驱动与纯算术畸变催化 XAFS 建模引擎 (v2.0)

核心定位：
  完全脱离高耗时的 DFT 弛豫，基于“晶体学母体提取 + 空间算法智能寻位 + 参数化物理微扰注入”
  的三段式流水线，秒级生成包含真实缺陷（界面锚定、出平面起伏、高斯静态无序、姜泰勒扭曲）
  的非周期性局域团簇，并闭环导出原点对齐的 feff.inp 文件。

支持核心场景：
  1. 表面切片与智能吸附（Top/Bridge/Hollow 自动寻位锚定）
  2. 体相/骨架掺杂（指定元素替换与局域弛豫）
  3. 参数化微扰注入（Buckling 面外起伏、高斯静态无序、键角扭转）
  4. 离线内置经典载体 + 本地 CIF 解析 + 在线晶体数据库扩展接口
"""

import argparse
import math
import sys
from pathlib import Path
import numpy as np

import xafs_writer

# 元素原子序数与常用共价半径 (Å)
ATOMIC_INFO = {
    "H":  (1, 0.31),  "C":  (6, 0.76),  "N":  (7, 0.71),  "O":  (8, 0.66),
    "F":  (9, 0.57),  "Al": (13, 1.21), "Si": (14, 1.11), "P":  (15, 1.07),
    "S":  (16, 1.05), "Cl": (17, 1.02), "Ti": (22, 1.60), "V":  (23, 1.53),
    "Cr": (24, 1.39), "Mn": (25, 1.39), "Fe": (26, 1.32), "Co": (27, 1.26),
    "Ni": (28, 1.24), "Cu": (29, 1.32), "Zn": (30, 1.22), "Mo": (42, 1.54),
    "Ru": (44, 1.46), "Rh": (45, 1.42), "Pd": (46, 1.39), "Ag": (47, 1.45),
    "Ce": (58, 2.04), "Ir": (77, 1.41), "Pt": (78, 1.36), "Au": (79, 1.36)
}

# ---------------------------------------------------------------------------
# 物理硬约束（P0.1 / P0.3）
# ---------------------------------------------------------------------------
# FEFF 的 Muffin-Tin 近似为每个原子分配一个电子势球，势球半径重叠容差
# 通常只有 10%~15%。两个原子间距落到 0.1~0.3 Å 时内部势场发散，FEFF 直接
# 报 "Muffin-tin radii overlap too much" 并终止整条路径计算。
CENTER_PROTECT_TOL_A = 1e-3        # 原点吸收原子的保护半径 (Å)
MIN_BOND_RATIO = 0.85              # 允许的 (r_a+r_b) 下限系数 = 1 - 15% 重叠容差
HARD_MIN_DISTANCE_A = 0.80         # 任意两原子的绝对硬下限 (Å)，低于即拒绝导出
UNKNOWN_COVALENT_RADIUS_A = 1.0    # 表外元素的保守半径 (Å)


def _covalent_radius(symbol: str) -> float:
    info = ATOMIC_INFO.get(str(symbol).capitalize())
    return float(info[1]) if info else UNKNOWN_COVALENT_RADIUS_A


def minimum_bond_distance(el_a: str, el_b: str,
                          min_ratio: float = MIN_BOND_RATIO) -> float:
    """刚体相切下限：min_ratio × (共价半径 a + 共价半径 b)。"""
    return float(min_ratio) * (_covalent_radius(el_a) + _covalent_radius(el_b))


def check_collision_and_elevate(anchor_pos, anchor_el, surf_coords,
                                surf_elements, min_ratio: float = MIN_BOND_RATIO,
                                collision_info: dict | None = None):
    """基于共价半径的表面锚定碰撞排斥抬升（P0.3）。

    单纯按 z 轴找最高点再硬编码一个吸附距离，会把 M–O 压到 1.6 Å 量级；
    在 sin(2kR+φ) 中这会产生极高频振荡并摧毁第一壳径向分布。此门禁沿 +z
    把吸附原子抬到与所有表面原子刚体相切的位置，保证任一 M–X 距离不小于
    ``min_ratio × (r_M + r_X)``。

    ``collision_info`` 可选，用于接回诊断而不引入模块级全局状态。
    """
    pos = np.array(anchor_pos, dtype=float).reshape(-1).copy()
    if pos.size != 3:
        raise ValueError(f"anchor_pos 必须是三维坐标，收到 {pos!r}")
    coords = np.asarray(surf_coords, dtype=float).reshape(-1, 3)
    elements = [str(e) for e in surf_elements]
    if coords.shape[0] != len(elements):
        raise ValueError("surf_coords 与 surf_elements 长度不一致")

    info = {"elevated_A": 0.0, "n_violations": 0,
            "min_allowed_A": None, "nearest_before_A": None}
    if coords.shape[0]:
        allowed = np.array([minimum_bond_distance(anchor_el, el, min_ratio)
                            for el in elements], dtype=float)
        delta = pos[None, :] - coords
        current = np.linalg.norm(delta, axis=1)
        xy2 = delta[:, 0] ** 2 + delta[:, 1] ** 2
        violating = current < allowed
        info["n_violations"] = int(violating.sum())
        info["min_allowed_A"] = float(allowed.min())
        info["nearest_before_A"] = float(current.min())
        shift = 0.0
        for i in np.flatnonzero(violating):
            need2 = float(allowed[i] ** 2 - xy2[i])
            if need2 <= 0.0:
                # 水平距离已经足够，任何 z 都满足，无需抬升
                continue
            needed_dz = math.sqrt(need2)
            actual_dz = float(pos[2] - coords[i, 2])
            shift = max(shift, needed_dz - actual_dz)
        if shift > 0.0:
            pos[2] += shift
            info["elevated_A"] = float(shift)
    if collision_info is not None:
        collision_info.update(info)
    return pos


def validate_cluster_distances(cluster_atoms, *, strict: bool = True):
    """团簇两两最小间距校验（P0.1 第二道防线）。

    - ``< HARD_MIN_DISTANCE_A``：严格非物理，MT 势球必然崩溃 → 抛错拒绝导出；
    - ``< 0.85(r_a+r_b)``：共价半径下限被突破 → 打印警告（真实离子晶体的
      键长有时略短于共价半径和，不宜一刀切拒绝）。

    绝不做静默 clamp：要么显式抬升，要么诚实失败。
    """
    pts = np.asarray([[x, y, z] for el, x, y, z in cluster_atoms], dtype=float)
    els = [str(el) for el, x, y, z in cluster_atoms]
    n = len(els)
    if n < 2:
        return {"min_distance_A": None, "soft_violations": 0}
    hard_pairs, soft_pairs = [], []
    global_min = float("inf")
    for i in range(n - 1):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        j_rel = int(np.argmin(d))
        j = i + 1 + j_rel
        dmin = float(d[j_rel])
        global_min = min(global_min, dmin)
        if dmin < HARD_MIN_DISTANCE_A:
            hard_pairs.append((i, j, dmin, els[i], els[j]))
        elif dmin < minimum_bond_distance(els[i], els[j]):
            soft_pairs.append((i, j, dmin, els[i], els[j]))
    if hard_pairs and strict:
        i, j, dmin, ea, eb = min(hard_pairs, key=lambda p: p[2])
        raise ValueError(
            f"非物理团簇：{ea}(#{i})–{eb}(#{j}) 间距 {dmin:.3f} Å < "
            f"{HARD_MIN_DISTANCE_A:.2f} Å，FEFF Muffin-Tin 势球将严重重叠。"
            "常见原因是中心吸收原子被微扰后与原点补写的原子构成鬼影对；"
            "请检查 inject_parameterized_distortions 的中心保护或团簇构建流程。")
    if soft_pairs:
        i, j, dmin, ea, eb = min(soft_pairs, key=lambda p: p[2])
        print(f"[WARN] 团簇存在 {len(soft_pairs)} 处键长低于共价半径下限："
              f"最近 {ea}–{eb} = {dmin:.3f} Å；FEFF 可能报 MT 势球重叠。")
    return {"min_distance_A": global_min,
            "soft_violations": len(soft_pairs),
            "hard_violations": len(hard_pairs)}


# 内置经典催化载体晶体结构原型 (用于离线与快速基准)
BUILTIN_PROTOTYPES = {
    "tio2_rutile": {
        "formula": "TiO2", "spacegroup": "P42/mnm",
        "cell": (4.594, 4.594, 2.959, 90.0, 90.0, 90.0),
        "atoms": [
            ("Ti", 0.0000, 0.0000, 0.0000),
            ("Ti", 0.5000, 0.5000, 0.5000),
            ("O",  0.3053, 0.3053, 0.0000),
            ("O",  0.6947, 0.6947, 0.0000),
            ("O",  0.8053, 0.1947, 0.5000),
            ("O",  0.1947, 0.8053, 0.5000),
        ]
    },
    "tio2_anatase": {
        "formula": "TiO2", "spacegroup": "I41/amd",
        "cell": (3.784, 3.784, 9.515, 90.0, 90.0, 90.0),
        "atoms": [
            ("Ti", 0.0, 0.0, 0.0),
            ("Ti", 0.0, 0.5, 0.25),
            ("Ti", 0.5, 0.5, 0.5),
            ("Ti", 0.5, 0.0, 0.75),
            ("O",  0.0, 0.0, 0.2081),
            ("O",  0.0, 0.0, 0.7919),
            ("O",  0.0, 0.5, 0.4581),
            ("O",  0.0, 0.5, 0.9581),
            ("O",  0.5, 0.5, 0.7081),
            ("O",  0.5, 0.5, 0.2919),
            ("O",  0.5, 0.0, 0.0419),
            ("O",  0.5, 0.0, 0.5419),
        ]
    },
    "ceo2": {
        "formula": "CeO2", "spacegroup": "Fm-3m",
        "cell": (5.411, 5.411, 5.411, 90.0, 90.0, 90.0),
        "atoms": [
            ("Ce", 0.0, 0.0, 0.0), ("Ce", 0.0, 0.5, 0.5),
            ("Ce", 0.5, 0.0, 0.5), ("Ce", 0.5, 0.5, 0.0),
            ("O",  0.25, 0.25, 0.25), ("O", 0.25, 0.25, 0.75),
            ("O",  0.25, 0.75, 0.25), ("O", 0.25, 0.75, 0.75),
            ("O",  0.75, 0.25, 0.25), ("O", 0.75, 0.25, 0.75),
            ("O",  0.75, 0.75, 0.25), ("O", 0.75, 0.75, 0.75),
        ]
    },
    "co3o4": {
        "formula": "Co3O4", "spacegroup": "F d -3 m :2",
        # 真实尖晶石非对称单元（原点选择 2，与 COD:9005888 一致）：
        # Co2+ 8a(1/8,1/8,1/8) + Co3+ 16d(1/2,1/2,1/2) + O 32e(x,x,x)。
        # 按 F d -3 m :2 展开为 8+16+32 = 56 原子（Co3O4，化学计量正确，
        # Co–O≈1.93 Å）。旧 6 原子表在 :1 设置下会给出 Co32O32(CoO) 且
        # 存在 0.18 Å 的 O–Co 鬼影对，已修正。
        "expand": True,
        "cell": (8.084, 8.084, 8.084, 90.0, 90.0, 90.0),
        "atoms": [
            ("Co", 0.125, 0.125, 0.125),
            ("Co", 0.500, 0.500, 0.500),
            ("O",  0.263, 0.263, 0.263),
        ]
    },
    "zsm5_model": {
        "formula": "AlSiO4", "spacegroup": "Pnma",
        "cell": (20.07, 19.92, 13.42, 90.0, 90.0, 90.0),
        "atoms": [
            ("Si", 0.11, 0.06, 0.03), ("Si", 0.19, 0.17, 0.03),
            ("Al", 0.25, 0.12, 0.25), ("Si", 0.07, 0.12, 0.35),
            ("O",  0.15, 0.10, 0.00), ("O",  0.20, 0.12, 0.15),
            ("O",  0.25, 0.18, 0.25), ("O",  0.10, 0.05, 0.25),
        ]
    }
}


def _cell_matrix_from_params(a, b, c, alpha, beta, gamma):
    # alpha,beta,gamma in radians
    v = math.sqrt(1.0 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2
                  + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))
    return np.array([
        [a, b * math.cos(gamma), c * math.cos(beta)],
        [0.0, b * math.sin(gamma), c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)],
        [0.0, 0.0, c * v / math.sin(gamma)]
    ], dtype=float)


class CrystalModel:
    """轻量自包含晶体与超胞模型"""
    def __init__(self, cell_params, atoms, spacegroup="P1", prototype_id=None):
        # cell_params: (a, b, c, alpha, beta, gamma)
        self.a, self.b, self.c = cell_params[:3]
        self.alpha, self.beta, self.gamma = [math.radians(x) for x in cell_params[3:]]
        self.atoms = list(atoms)  # [(element, fx, fy, fz), ...]
        self.spacegroup = str(spacegroup or "P1")
        self.prototype_id = prototype_id
        self._init_cartesian_matrix()

    def _init_cartesian_matrix(self):
        # 晶胞分数坐标 -> 笛卡尔坐标转换矩阵
        self.m_frac2cart = _cell_matrix_from_params(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def to_cartesian(self, fx, fy, fz):
        return self.m_frac2cart @ np.array([fx, fy, fz])

    def _expanded_atoms(self):
        sg = (self.spacegroup or "P1").replace(" ", "")
        if sg in ("P1",):
            return list(self.atoms)
        try:
            import gemmi  # noqa: F401
            have_gemmi = True
        except Exception:
            have_gemmi = False
        if not have_gemmi:
            raise RuntimeError(
                f"晶体空间群为 {self.spacegroup}（非 P1），"
                f"需先用 gemmi 展开对称等效位点后才能构建超胞；"
                f"当前环境未安装 gemmi，禁止静默返回残缺原子")
        # 用 gemmi 空间群操作展开对称等效位点，得到 P1 全晶胞原子。
        # 旧实现依赖 gemmi.Structure + Fractional.to_orth +
        # Structure.get_all_unit_cell_sites，这三个 API 在 gemmi 0.7.x 上
        # 不存在，展开必然抛异常，再被下方 except 静默吞掉返回未展开原子。
        try:
            import gemmi
            sgroup = gemmi.find_spacegroup_by_name(self.spacegroup)
            if sgroup is None:
                raise ValueError(f"gemmi 无法识别空间群 {self.spacegroup!r}")
            out, seen = [], []
            for el, fx, fy, fz in self.atoms:
                for op in sgroup.operations():
                    x, y, z = op.apply_to_xyz([float(fx), float(fy), float(fz)])
                    f = np.array([x % 1.0, y % 1.0, z % 1.0], dtype=float)
                    dup = False
                    for g_el, g in seen:
                        if g_el != el:
                            continue
                        d = f - g
                        d -= np.round(d)
                        if float(np.linalg.norm(d)) < 1e-4:
                            dup = True
                            break
                    if dup:
                        continue
                    seen.append((el, f))
                    out.append((el, float(f[0]), float(f[1]), float(f[2])))
            return out if out else list(self.atoms)
        except Exception as exc:
            raise RuntimeError(
                f"空间群 {self.spacegroup} 需 gemmi 展开对称等效位点，但展开失败: {exc}"
            ) from exc

    def build_supercell(self, nx=3, ny=3, nz=3):
        """生成超胞避免周期截断"""
        base_atoms = self._expanded_atoms()
        new_atoms = []
        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(-nz, nz + 1):
                    for el, fx, fy, fz in base_atoms:
                        cx, cy, cz = self.to_cartesian(fx + ix, fy + iy, fz + iz)
                        new_atoms.append((el, cx, cy, cz))
        return new_atoms


def parse_cif_simple(cif_path: Path):
    """gemmi 完整空间群展开 CIF→全晶胞 CrystalModel；坏/空/缺失输入显式报错，不静默换材料。"""
    import re

    import gemmi

    from xafs_core import XAFSDataError

    cif_path = Path(cif_path)
    if not cif_path.is_file():
        raise XAFSDataError("CIF", f"显式 CIF 输入不存在: {cif_path}")
    try:
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block() if len(doc) == 1 else doc[0]
        st = gemmi.make_small_structure_from_block(block)
        cell = st.cell
        all_sites = st.get_all_unit_cell_sites()
    except (RuntimeError, ValueError, IndexError) as exc:
        raise XAFSDataError(
            "CIF", f"解析 {cif_path.name} 失败：{exc}。"
                   f"检查晶胞参数 / 空间群 / _atom_site_* 循环是否完整") from exc

    atoms = []
    seen = []
    for site in all_sites:
        m = re.match(r"[A-Za-z]+", site.type_symbol.strip())
        if not m:
            continue
        el = m.group(0).capitalize()
        f = np.array([site.fract.x, site.fract.y, site.fract.z], dtype=float)
        # 同元素特殊位置原子去重；不同元素共占同一分数坐标时必须保留双方
        dup = False
        for g_el, g in seen:
            if g_el != el:
                continue
            d = f - g
            d -= np.round(d)
            if float(np.linalg.norm(d)) < 1e-4:
                dup = True
                break
        if dup:
            continue
        seen.append((el, f))
        atoms.append((el, float(f[0]), float(f[1]), float(f[2])))

    if not atoms:
        raise XAFSDataError(
            "CIF", f"{cif_path.name} 解析后无原子（空晶胞 / 缺 _atom_site_ 循环）；"
                   f"拒绝静默回退内置原型")
    return CrystalModel(
        (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma), atoms)


# =========================================================================
# 任务 2：算法级拓扑缺陷与锚定位点生成引擎
# =========================================================================

def _parse_miller(surface):
    if surface is None:
        return (0, 0, 1)
    if isinstance(surface, str):
        s = surface.strip().strip("()[]")
        if s.isdigit() and len(s) == 3:
            vals = [int(ch) for ch in s]
        else:
            parts = s.replace(",", " ").split()
            vals = [int(float(p)) for p in parts if p.strip()]
    else:
        vals = [int(v) for v in surface]
    if len(vals) != 3 or vals == [0, 0, 0]:
        raise ValueError(f"surface Miller 指数必须为 3 个非全零整数，收到: {surface!r}")
    return tuple(vals)


def _rotation_align_to_z(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return np.eye(3)
    v = v / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, z))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    axis = np.cross(v, z)
    axis_n = np.linalg.norm(axis)
    if axis_n == 0:
        return np.eye(3)
    axis = axis / axis_n
    x, y, az = axis
    s = math.sqrt(max(0.0, 1.0 - c * c))
    C = 1.0 - c
    R = np.array([
        [c + x*x*C,     x*y*C - az*s, x*az*C + y*s],
        [y*x*C + az*s,  c + y*y*C,    y*az*C - x*s],
        [az*x*C - y*s,  az*y*C + x*s, c + az*az*C],
    ], dtype=float)
    return R


def cleave_miller_slab(model, hkl, nx: int = 2, ny: int = 2, nz: int = 2,
                       z_cleave: float | None = None):
    """按 Miller 指数 (hkl) 真实切割晶体平面，返回朝向 +z 的平整 Slab 原子列表。

    物理逻辑：
      1. 构造超胞（若晶轴很短则自动加厚，保证半厚度 > 6 Å 避免底面截断）；
      2. 由倒格矢 b1,b2,b3 计算晶面法向 n_cart = h·b1 + k·b2 + l·b3 并旋转对齐 +z；
      3. 在超胞中心高度附近的最大层间距处做**真实平面切割**（z ≤ z_cut），移除
         上半空间原子，将目标 (hkl) 晶面暴露为最表层；
      4. 将顶层台面横向中心平移至 (x=0, y=0)，避开有限超胞边缘棱角。
    """
    hkl_t = _parse_miller(hkl)
    M = model.m_frac2cart
    a1, a2, a3 = M[:, 0], M[:, 1], M[:, 2]
    vcell = float(np.dot(a1, np.cross(a2, a3)))
    if abs(vcell) < 1e-12:
        raise ValueError("晶胞体积为零，无法计算 Miller 面法向")
    b1 = np.cross(a2, a3) / vcell
    b2 = np.cross(a3, a1) / vcell
    b3 = np.cross(a1, a2) / vcell
    n_cart = hkl_t[0] * b1 + hkl_t[1] * b2 + hkl_t[2] * b3
    R = _rotation_align_to_z(n_cart)

    # 保证沿任一方向超胞半厚度足够容纳 rcut=5~6 Å
    nx_e = max(int(nx), int(math.ceil(6.5 / max(float(np.linalg.norm(a1)), 1.0))))
    ny_e = max(int(ny), int(math.ceil(6.5 / max(float(np.linalg.norm(a2)), 1.0))))
    nz_e = max(int(nz), int(math.ceil(6.5 / max(float(np.linalg.norm(a3)), 1.0))))
    supercell = model.build_supercell(nx=nx_e, ny=ny_e, nz=nz_e)
    rot_atoms = [(el, *(R @ np.array([x, y, z], dtype=float)))
                 for el, x, y, z in supercell]
    coords = np.array([[x, y, z] for _, x, y, z in rot_atoms], dtype=float)
    zs = coords[:, 2]

    if z_cleave is None:
        z_med = float(np.median(zs))
        # 在中心截面附近（避开立方体顶角）找最大晶面层间隙作为切割面
        xy_med = np.median(coords[:, :2], axis=0)
        core_mask = np.linalg.norm(coords[:, :2] - xy_med[None, :], axis=1) <= 6.0
        z_core = np.sort(zs[core_mask]) if np.sum(core_mask) >= 12 else np.sort(zs)
        band = z_core[(z_core >= z_med - 2.0) & (z_core <= z_med + 2.0)]
        if band.size >= 4:
            gaps = np.diff(band)
            # 只在原子层之间的间隙（>0.35 Å）中选最宽间隙的中点
            best_i = int(np.argmax(gaps))
            if gaps[best_i] > 0.35:
                z_cut = float(0.5 * (band[best_i] + band[best_i + 1]))
            else:
                z_cut = z_med
        else:
            z_cut = z_med
    else:
        z_cut = float(z_cleave)

    slab = [atom for atom in rot_atoms if atom[3] <= z_cut + 1e-6]
    if len(slab) < 4:
        raise ValueError(f"Miller 晶面 {hkl_t} 在 z_cut={z_cut:.2f} Å 切割后原子过少")

    # 横向居中：以顶表面原子 (max_z - 1.5 Å) 的中位 (x, y) 为台面中心 (0, 0)
    slab_coords = np.array([[x, y, z] for _, x, y, z in slab], dtype=float)
    top_z = float(np.max(slab_coords[:, 2]))
    top_mask = slab_coords[:, 2] >= (top_z - 1.5)
    cx, cy = np.median(slab_coords[top_mask, :2], axis=0)
    return [(el, float(x - cx), float(y - cy), float(z))
            for el, x, y, z in slab]


def find_surface_anchoring_site(slab_atoms, site_type="top", anchor_atom="Pt",
                                target_bond=2.02, surface=None,
                                terrace_radius: float | None = None):
    """
    智能几何寻位：在 Slab 表面自动计算 Top、Bridge、Hollow 锚定吸附点。
    若给定 terrace_radius，则只在横向台面中心区域内搜索表面原子，避开有限超胞边缘。
    """
    coords = np.array([[x, y, z] for el, x, y, z in slab_atoms], dtype=float)
    elements = [el for el, x, y, z in slab_atoms]

    # 提取顶表面原子集合（z 处于最高 1.5 Å 范围内的原子）
    max_z = np.max(coords[:, 2])
    surf_mask = coords[:, 2] >= (max_z - 1.5)
    if terrace_radius is not None and terrace_radius > 0:
        xy_center = np.median(coords[surf_mask, :2], axis=0)
        in_terrace = np.linalg.norm(coords[:, :2] - xy_center[None, :], axis=1) <= float(terrace_radius)
        if np.any(surf_mask & in_terrace):
            surf_mask = surf_mask & in_terrace
    surf_indices = np.where(surf_mask)[0]
    surf_coords = coords[surf_indices]
    surf_elements = [elements[i] for i in surf_indices]

    # 选择表面最近邻点
    center_idx = np.argmin(np.linalg.norm(surf_coords[:, :2], axis=1))
    c0 = surf_coords[center_idx]

    # 【P0.3】target_bond=2.02 是 Pt–O 的经验值，对 Ce–O (≈2.70 Å) 等
    # 大半径吸附原子会把键长硬压到共价半径和以下。先按元素对抬到物理下限，
    # 末尾再由碰撞门禁对全部表面原子做刚体相切校验。
    physical_target = minimum_bond_distance(
        anchor_atom, surf_elements[center_idx], min_ratio=1.0)
    target_bond = max(float(target_bond), physical_target)
    
    if site_type.lower() == "top":
        # 正上方直接锚定
        anchor_pos = c0 + np.array([0.0, 0.0, target_bond])
    elif site_type.lower() == "bridge":
        # 寻找最近邻表面配对原子
        dists = np.linalg.norm(surf_coords - c0, axis=1)
        valid_pairs = np.where((dists > 1.5) & (dists < 3.5))[0]
        if len(valid_pairs) > 0:
            c1 = surf_coords[valid_pairs[0]]
            mid = 0.5 * (c0 + c1)
            half_d = 0.5 * np.linalg.norm(c0 - c1)
            h = math.sqrt(max(0.1, target_bond**2 - half_d**2))
            anchor_pos = mid + np.array([0.0, 0.0, h])
        else:
            anchor_pos = c0 + np.array([0.0, 0.0, target_bond])
    else:  # hollow
        # 三原子空位中心
        dists = np.linalg.norm(surf_coords - c0, axis=1)
        sorted_idx = np.argsort(dists)
        if len(sorted_idx) >= 3:
            pts = surf_coords[sorted_idx[:3]]
            tri_center = np.mean(pts, axis=0)
            r_arm = np.mean(np.linalg.norm(pts[:, :2] - tri_center[:2], axis=1))
            h = math.sqrt(max(0.2, target_bond**2 - r_arm**2))
            anchor_pos = np.array([tri_center[0], tri_center[1], max_z + h])
        else:
            anchor_pos = c0 + np.array([0.0, 0.0, target_bond])

    collision = {}
    anchor_pos = check_collision_and_elevate(
        anchor_pos, anchor_atom, surf_coords, surf_elements,
        collision_info=collision)
    if collision.get("elevated_A", 0.0) > 0.05:
        print(f"[P0.3] 锚定位点因共价半径排斥抬升 "
              f"{collision['elevated_A']:.3f} Å"
              f"（最近键长 {collision.get('nearest_before_A')} Å → 下限 "
              f"{collision.get('min_allowed_A')} Å）")
    return anchor_pos


def apply_bulk_doping(supercell_atoms, replace_el="Ti", dopant_el="Fe", relaxation=0.03):
    """
    体相晶格掺杂：替换目标元素，并自动对第一配位层施加局域径向伸缩 (Relaxation)
    """
    replace_el = str(replace_el).strip().capitalize()
    dopant_el = str(dopant_el).strip().capitalize()
    dopant_idx = None
    # 寻找最靠近几何中心的被置换原子
    center_dists = [math.sqrt(x**2 + y**2 + z**2) for el, x, y, z in supercell_atoms if el == replace_el]
    if not center_dists:
        present = sorted({el for el, *_ in supercell_atoms})
        raise ValueError(
            f"超胞中不存在被替换元素 {replace_el!r}（现有: {present}），禁止静默替换其他元素")
        
    min_d = float("inf")
    for idx, (el, x, y, z) in enumerate(supercell_atoms):
        if el == replace_el:
            d = math.sqrt(x**2 + y**2 + z**2)
            if d < min_d:
                min_d = d
                dopant_idx = idx

    dopant_pos = np.array(supercell_atoms[dopant_idx][1:])
    new_atoms = []
    
    # 替换中心并进行球形径向弛豫
    for idx, (el, x, y, z) in enumerate(supercell_atoms):
        pos = np.array([x, y, z])
        if idx == dopant_idx:
            new_atoms.append((dopant_el, x, y, z))
        else:
            r_vec = pos - dopant_pos
            dist = np.linalg.norm(r_vec)
            if dist < 2.8 and dist > 0.1:  # 第一配位层
                pos = dopant_pos + r_vec * (1.0 + relaxation)
            new_atoms.append((el, pos[0], pos[1], pos[2]))
            
    return new_atoms, dopant_pos


# =========================================================================
# 任务 3：参数化畸变与静态无序注入 (Distortion Injector)
# =========================================================================

def inject_parameterized_distortions(cluster_atoms, center_pos, buckling_z=0.0,
                                     disorder_sigma=0.0, twist_angle=0.0):
    """
    注入纯数学参数化微扰：
      - buckling_z: 沿 z 轴施加面外起伏/极化拉伸
      - disorder_sigma: 三维高斯随机静态无序 N(0, sigma^2)
      - twist_angle: 绕 z 轴施加角度扭转（模拟 Jahn-Teller 形变）
    """
    distorted = []
    rad_twist = math.radians(twist_angle)
    
    for el, x, y, z in cluster_atoms:
        vec = np.array([x, y, z]) - center_pos
        dist = np.linalg.norm(vec)

        # 【P0.1】保护中心吸收原子：FEFF 导出时会在 (0,0,0) 手工补写
        # ipot=0 的吸收原子。若此处把原点原子微扰出去（buckling 对 vec=0
        # 会直接加上 z 位移，形成 0.1~0.3 Å 的"鬼影 Fe"），两个 MT 势球
        # 会灾难性重叠并让 FEFF 硬报错。中心原子必须严格留在原点。
        if dist <= CENTER_PROTECT_TOL_A:
            distorted.append((el, float(center_pos[0]),
                              float(center_pos[1]), float(center_pos[2])))
            continue

        # 1. 面外起伏拉伸 (Buckling)
        if buckling_z != 0.0 and dist < 2.6:
            vec[2] += buckling_z
            
        # 2. 键角扭矩旋转 (Angular Twist)
        if twist_angle != 0.0 and dist < 3.2:
            rot_z = np.array([
                [math.cos(rad_twist), -math.sin(rad_twist), 0.0],
                [math.sin(rad_twist),  math.cos(rad_twist), 0.0],
                [0.0,                  0.0,                 1.0]
            ])
            vec = rot_z @ vec
            
        # 3. 三维高斯静态无序微扰 (Gaussian Disorder)
        if disorder_sigma > 0.0 and dist > 0.01:
            noise = np.random.normal(0.0, disorder_sigma, 3)
            vec += noise
            
        new_pos = center_pos + vec
        distorted.append((el, new_pos[0], new_pos[1], new_pos[2]))

    # 【P0.1】第二道防线：畸变后仍然做一次两两最小间距校验，防止 twist /
    # disorder 把某个近邻推进中心势球。绝不静默 clamp。
    validate_cluster_distances(distorted, strict=True)
    return distorted


# =========================================================================
# 任务 4：FEFF 闭环导出
# =========================================================================

def export_cluster_feff(center_el, cluster_atoms, center_pos, rcut,
                        output_path: Path, edge: str | None = None):
    """截取 rcut 半径内的局域团簇，原点对齐后导出 feff.inp。

    【P1.2】单一真源：格式与吸收边一律复用 ``artemis.build_feff_input``，
    不再维护第二套 ``EDGE K`` / ``S02 1.0`` / 6 参 CONTROL 的生成器。

    - FEFF 用 **HOLE** 卡指定芯孔（K=1, L1=2, L2=3, L3=4）。把 Pt / Ir /
      Ce 等 L3 吸收边写成 ``EDGE K`` 会调取错误的中心原子相移 δ_c(k)，
      键长产生 0.3–0.5 Å 的系统偏差；``EDGE`` 本身也不是 FEFF 的边选择卡。
    - S0² 是**拟合层**参数（由 absorber.py 静态表 / T5 / CLI 解析链给出），
      绝不能写死在理论路径输入里，否则与 0.803 / 0.43–0.86 的标定值冲突。
    - 吸收原子 ipot=0 只能有一个；旧生成器把近邻也赋 ipot=0 会让散射势
      完全错误。

    ``edge`` 为空时从 ``absorber.ABSORBERS`` 动态读取；元素未收录时**显式
    报错**，不静默回退到 K 边。
    """
    from absorber import ABSORBERS
    # 延迟导入：脚本主体（构建/寻位/畸变）保持零 larch 依赖，只有真正导出
    # FEFF 输入时才引入生产链的同一个生成器。
    from artemis import build_feff_input

    absorber = str(center_el).capitalize()
    if edge is None:
        cfg = ABSORBERS.get(absorber)
        if cfg is None:
            raise ValueError(
                f"元素 {center_el!r} 不在 absorber.ABSORBERS 中，无法判定吸收边；"
                "请用 edge= 显式传入 'K'/'L1'/'L2'/'L3'。"
                "未知元素一律不静默回退到 K（L3 重金属相移会完全错）。")
        edge = cfg.edge
    edge = str(edge).upper()
    if edge not in ("K", "L1", "L2", "L3"):
        raise ValueError(f"未知吸收边 {edge!r}；可选 K / L1 / L2 / L3")

    # 【P0.1】导出前拒绝含鬼影原子对的团簇，避免 FEFF MT 势球重叠硬报错。
    validate_cluster_distances(cluster_atoms, strict=True)

    cx, cy, cz = (float(v) for v in center_pos)
    sites = []
    for el, x, y, z in cluster_atoms:
        r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        if 0.001 < r <= float(rcut):
            sites.append((x - cx, y - cy, z - cz, str(el)))
    if not sites:
        raise ValueError(
            f"rcut={rcut} Å 内没有任何散射原子，拒绝导出空的 feff.inp")

    text = build_feff_input(
        sites, absorber=absorber, edge=edge, rmax=float(rcut),
        title=(f"Catalytic algorithmic cluster: absorber {absorber}, "
               f"rcut={rcut} A, edge={edge}"))
    output_path = Path(output_path)
    xafs_writer.write_text(output_path, text, encoding="utf-8")
    print(f"[OK] 导出 FEFF 输入（HOLE {edge}）: {output_path} "
          f"(散射原子 {len(sites)} 个)")


def main():
    parser = argparse.ArgumentParser(description="催化 XAFS 建模工具 v2.0 (晶体库驱动与算术畸变版)")
    parser.add_argument("--local-cif", default=None, help="本地 CIF 晶体结构路径")
    parser.add_argument("--db", choices=["cod", "matproj", "builtin"], default="builtin", help="晶体数据库来源")
    parser.add_argument("--formula", default="TiO2", help="化学式检索关键词")
    parser.add_argument("--id", default=None, help="数据库 ID (如 COD 编号)")
    parser.add_argument("--surface", default=None, help="切片表面密勒指数，如 '1 1 1' 或 '1 1 0'")
    parser.add_argument("--anchor-atom", default=None, help="表面锚定吸附金属原子 (如 Pt, Au)")
    parser.add_argument("--site-type", choices=["top", "bridge", "hollow"], default="bridge", help="表面锚定位点类型")
    parser.add_argument("--replace", default=None, help="体相掺杂被替换元素 (如 Ti, Al)")
    parser.add_argument("--dopant", default=None, help="体相掺杂元素 (如 Fe, Co)")
    parser.add_argument("--center", default=None, help="指定中心吸收原子元素")
    
    # 参数化畸变
    parser.add_argument("--buckling-z", type=float, default=0.0, help="轴向面外起伏/拉伸 (Å)")
    parser.add_argument("--disorder-sigma", type=float, default=0.0, help="高斯静态无序微扰方差 sigma (Å)")
    parser.add_argument("--twist-angle", type=float, default=0.0, help="配位键角扭转角度 (deg)")
    parser.add_argument("--rcut", type=float, default=5.0, help="局域团簇切取半径 (Å)")
    parser.add_argument("--edge", default=None, choices=["K", "L1", "L2", "L3"],
                        help="吸收边；默认按 absorber.py 的元素表动态判定（如 Ce/Pt/Ir 为 L3）。"
                             "元素未收录时必须显式指定，绝不静默回退 K 边")
    parser.add_argument("-o", "--output", default="feff_cluster.inp", help="导出 FEFF 文件名")

    args = parser.parse_args()
    np.random.seed(42)  # 保证高斯无序注入可复现

    # 1. 获取母体晶体结构
    if args.local_cif:
        model = parse_cif_simple(Path(args.local_cif))
    else:
        # 使用内置高精度原型或模拟数据库拉取
        key = "tio2_rutile"
        f_low = str(args.formula).lower()
        if "zsm" in f_low:
            key = "zsm5_model"
        elif "co" in f_low or (args.center and args.center.lower() == "co"):
            key = "co3o4"
        elif "ceo2" in f_low:
            key = "ceo2"
        elif "anatase" in f_low:
            key = "tio2_anatase"
        proto = BUILTIN_PROTOTYPES.get(key, BUILTIN_PROTOTYPES["tio2_rutile"])
        # 与 structure_lib.build_algorithmic_catalytic_motif 一致：仅声明
        # expand=True 的原型（真实非对称单元）才做空间群展开。
        sg = proto.get("spacegroup", "P1") if proto.get("expand") else "P1"
        model = CrystalModel(proto["cell"], proto["atoms"],
                             spacegroup=sg, prototype_id=key)

    # 2. 超胞扩建 / 晶面切割
    if args.surface:
        hkl = _parse_miller(args.surface)
        supercell = cleave_miller_slab(model, hkl, nx=2, ny=2, nz=2)
    else:
        supercell = model.build_supercell(nx=2, ny=2, nz=2)

    # 3. 支线 A：表面切片与智能吸附锚定 (Surface Cleaving & Anchoring)
    if args.surface and args.anchor_atom:
        anchor_pos = find_surface_anchoring_site(
            supercell, site_type=args.site_type,
            anchor_atom=args.anchor_atom, surface=args.surface,
            terrace_radius=4.5)
        center_el = args.anchor_atom
        center_pos = anchor_pos
        cluster_atoms = supercell + [(args.anchor_atom, anchor_pos[0], anchor_pos[1], anchor_pos[2])]
        
    # 3. 支线 B：体相/骨架掺杂 (Bulk Doping)
    elif args.replace and args.dopant:
        doped_atoms, d_pos = apply_bulk_doping(supercell, replace_el=args.replace, dopant_el=args.dopant)
        center_el = args.dopant
        center_pos = d_pos
        cluster_atoms = doped_atoms
        
    # 3. 支线 C：指定中心原子提取无序母体
    else:
        center_el = args.center or model.atoms[0][0]
        # 找最靠近原点的中心原子
        c_candidates = [(math.sqrt(x**2+y**2+z**2), np.array([x, y, z])) for el, x, y, z in supercell if el == center_el]
        if not c_candidates:
            raise ValueError(f"超胞中未找到指定中心原子 {center_el!r}，无法提取局域团簇")
        c_candidates.sort(key=lambda item: item[0])
        center_pos = c_candidates[0][1]
        cluster_atoms = supercell

    # 4. 参数化物理微扰注入
    distorted_atoms = inject_parameterized_distortions(
        cluster_atoms, center_pos, buckling_z=args.buckling_z,
        disorder_sigma=args.disorder_sigma, twist_angle=args.twist_angle
    )

    # 5. 闭环导出 FEFF 输入（吸收边动态判定，格式复用 artemis.build_feff_input）
    export_cluster_feff(center_el, distorted_atoms, center_pos, rcut=args.rcut,
                        output_path=Path(args.output), edge=args.edge)


if __name__ == "__main__":
    main()
