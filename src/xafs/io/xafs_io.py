# -*- coding: utf-8 -*-
"""
xafs_io.py — 数据进出层
================================================================
职责边界：**只做格式转换，不做任何物理计算**。
  读：XDAC / XDI / 纯文本  →  RawScan
  写：ndarray → 文本表 / JSON

本模块不 import athena / artemis / larch_wt，因此可被任意阶段复用。
μ(E) 的换算也放在这里，因为它是「通道角色」问题，不是「谱学处理」问题：
透射取对数、荧光直接比值，选哪条由文件里有哪些通道决定。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

import xafs_writer
from xafs_core import RawScan, XAFSConfigError, XAFSDataError

def measurement_identity(scan) -> tuple[bool | None, str]:
    cached = getattr(scan, "_measurement_identity", None)
    if cached is not None:
        return cached
    meta = getattr(scan, "meta", None) or {}
    mode = str(meta.get("mode") or "unknown")
    if mode in ("fluor", "fluo", "fluor_multichannel"):
        return True, f"mode={mode}"
    def present(attr):
        value = getattr(scan, attr, None)
        try:
            return value is not None and np.asarray(value).size > 0
        except (TypeError, ValueError):
            return False

    if mode in ("given", "trans"):
        roles = meta.get("mu_channel_roles")
        pair = ("it", "iref") if roles == "ln(It/Ir)" else ("i0", "it")
        try:
            arrays = [np.asarray(getattr(scan, key, None), dtype=float)
                      for key in (*pair, "mu")]
            incident, transmitted, mu = arrays
            valid = (mu.ndim == 1 and mu.size > 0
                     and incident.shape == transmitted.shape == mu.shape
                     and all(np.all(np.isfinite(a)) for a in arrays)
                     and np.all(incident > 0) and np.all(transmitted > 0))
            if valid and np.allclose(mu, np.log(incident / transmitted),
                                     rtol=0.0, atol=1e-6):
                return False, (f"原始 xmu 与 ln({pair[0]}/{pair[1]}) "
                               "逐点一致（≤1e-6）")
        except (TypeError, ValueError):
            return None, f"mode={mode!r} 的原始通道不是有效数值"
    if mode != "trans" and present("ifluor") and not present("it"):
        return True, f"存在 If 通道且无 It：mode={mode}"
    return None, f"mode={mode!r} 且缺少可判别的原始通道证据"


def capture_measurement_identity(scan) -> tuple[bool | None, str]:
    identity = measurement_identity(scan)
    scan._measurement_identity = identity
    scan.meta["measurement_identity"] = {
        "is_fluorescence": identity[0], "basis": identity[1], "stage": "raw"}
    return identity


# ================================================================
# 1. 格式识别
# ================================================================

def sniff_format(path) -> str:
    """
    按首行特征判断格式，不做推测性解析。

    返回: "XDAC" | "XDI" | "SIMPLE"
    """
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(2048).decode("latin-1", errors="replace")
    first = head.splitlines()[0] if head else ""
    if first.upper().startswith("XDAC"):
        return "XDAC"
    if first.lstrip().upper().startswith("# XDI"):
        return "XDI"
    if "# Column." in head:
        return "XDI"
    return "SIMPLE"


def _safe_float(value):
    """容忍 '7112.0 eV' 这类带单位的字段；解析失败返回 None 而不是崩。"""
    if value is None:
        return None
    try:
        return float(str(value).split()[0]) or None
    except (ValueError, IndexError):
        return None


def _to_float(token: str):
    """'0.05k' / '15k' / '2.80' → float；无法转换返回 None。"""
    try:
        return float(token.strip().rstrip("kK"))
    except ValueError:
        return None


# ================================================================
# 2. 通道角色校验
# ================================================================

def assert_channel_roles(i0, it, energy, name=""):
    """
    物理合理性检查：挡住「把 I0 当成 μ」这类静默错误。

    判据:
        I0/It 中位数 ∈ (1.01, 1e5)   —— 入射必须大于透射
        μ 中位数 ∈ (0.05, 20)        —— 吸收系数不可能跨几个数量级

    这两条来自实测：把 I0 当 μ 时，E0 只偏 1.25 eV 看不出来，
    但 edge_step 会从 0.47 变成 979.9。
    """
    if np.any(i0 <= 0) or np.any(it <= 0):
        raise XAFSDataError("S1", f"{name}: I0/It 含非正值，列角色可能判错")
    ratio = float(np.median(i0 / it))
    if not (1.01 < ratio < 1e5):
        raise XAFSDataError(
            "S1", f"{name}: I0/It 中位数 = {ratio:.4g}，不像入射/透射比。"
                  f"检查列顺序（XDAC 为 Energy,I0,It,Ir,IntTime）")
    mu = np.log(i0 / it)
    if not (0.05 < float(np.median(mu)) < 20):
        raise XAFSDataError(
            "S1", f"{name}: μ 中位数 = {np.median(mu):.4g}，超出合理范围 0.05–20")


def _convert_energy_unit(energy, unit, name=""):
    """XDI 能量列单位换算 → eV。未声明单位按 eV（向后兼容）。

    绝不静默把 keV 当 eV：7.112 keV 若按 eV 读，边位会错 1000 倍，
    后续 E0/k 轴/FT 全部失真。返回 (energy_eV, 声明单位, 换算因子)。
    """
    e = np.asarray(energy, dtype=float)
    u = (unit or "ev").strip().lower()
    if u in ("", "ev", "electronvolt", "electron_volts"):
        return e, "eV", 1.0
    if u in ("kev", "kiloelectronvolt", "kiloelectron_volts"):
        return e * 1000.0, "keV", 1000.0
    raise XAFSDataError(
        "S1", f"{name}: 能量列单位 '{unit}' 无法识别（仅支持 eV/keV）；"
              f"不得静默假设单位")


def _order_energy_axis(scan: RawScan, name: str = "") -> RawScan:
    """能量轴单调性契约：严格递增放行；严格递减整体翻转；非单调/重复拒绝。

    下游 np.interp / find_e0_derivative_max / FT 均要求能量严格递增。
    翻转时所有通道（i0/it/ifluor/iref/dtime/mu/sigma_mu 及 MCA 矩阵）
    同步反序，保持逐点对应；非单调或重复能量无法唯一插值，如实拒绝
    而非静默排序（排序会掩盖乱序/重复采集的数据问题）。
    """
    e = np.asarray(scan.energy, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise XAFSDataError("S1", f"{name}: 能量轴至少需 2 个一维点")
    if not np.all(np.isfinite(e)):
        raise XAFSDataError("S1", f"{name}: 能量轴含非有限值")
    # U1-06：非 None 通道必须与能量轴逐点对齐（1D 且等长），否则翻转/插值
    # 会把形状错位静默带进下游（RawScan 允许通道缺省，但不允许错位）。
    for ch_name in ("i0", "it", "ifluor", "iref", "dtime", "mu", "sigma_mu"):
        ch_val = getattr(scan, ch_name, None)
        if ch_val is not None:
            ch_arr = np.asarray(ch_val)
            if ch_arr.ndim != 1 or ch_arr.shape[0] != e.shape[0]:
                raise XAFSDataError(
                    "S1", f"{name}: 通道 {ch_name} 形状 {ch_arr.shape} "
                          f"与能量轴 {e.shape} 不一致")
    mca_raw = (scan.meta or {}).get("mca")
    if isinstance(mca_raw, dict):
        for mk in ("channels", "dtime"):
            mv = mca_raw.get(mk)
            if mv is None:
                continue
            mv = np.asarray(mv)
            if mv.ndim != 2 or mv.shape[0] != e.shape[0]:
                raise XAFSDataError(
                    "S1", f"{name}: mca.{mk} 形状 {mv.shape} 与能量轴 "
                          f"{e.shape} 不一致")
    d = np.diff(e)
    if np.all(d > 0):
        return scan                       # 严格递增：原样放行（基线不变）
    if not np.all(d < 0):
        raise XAFSDataError(
            "S1", f"{name}: 能量轴非单调或含重复点，无法唯一插值；"
                  f"请检查数据是否乱序或重复采集")
    order = slice(None, None, -1)         # 严格递减：整体翻转

    def rev(a):
        return None if a is None else np.asarray(a)[order].copy()

    meta = dict(scan.meta or {})
    meta["energy_axis_reversed"] = True
    mca = meta.get("mca")
    if isinstance(mca, dict):
        mca = dict(mca)
        if mca.get("channels") is not None:
            mca["channels"] = np.asarray(mca["channels"])[order].copy()
        if mca.get("dtime") is not None:
            mca["dtime"] = np.asarray(mca["dtime"])[order].copy()
        meta["mca"] = mca
    return RawScan(energy=e[order].copy(), i0=rev(scan.i0), it=rev(scan.it),
                   ifluor=rev(scan.ifluor), iref=rev(scan.iref),
                   dtime=rev(scan.dtime), mu=rev(scan.mu),
                   sigma_mu=rev(scan.sigma_mu), meta=meta)


# ================================================================
# 3. 各格式解析器
# ================================================================

def read_xdac(path) -> RawScan:
    """
    XDAC V1.x（SSRL / NSLS 老格式）。

    结构: 14 行头部 + 分隔线 + 数值表
    坑位: ① CRLF 行尾（splitlines 可解）
          ② SRB/SRSS 里有 '15k'、'0.05k'，float() 会抛异常
          ③ 列表固定为 Energy, I0, It, Ir, IntTime

    红线：数据行一旦开始，遇到既不是空行/注释也不是合法 key=val 头、
    且无法解析为全浮点数的行，或列数与首行不一致的行，或含 NaN/Inf 的行，
    立即抛出 XAFSDataError("S1", ...)（包含文件名和行号），禁止静默跳过。
    """
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    header, rows = {}, []
    for idx, line in enumerate(lines, 1):
        s = line.strip()
        # 跳过空行、分隔线和列名标题行
        if not s or s.startswith("-") or s.lower().startswith("energy"):
            continue
        tokens = s.split()
        values = [_to_float(t) for t in tokens]
        # 若所有 tokens 都能转为 float 且列数 >= 3，视为数据行
        if all(v is not None for v in values) and len(values) >= 3:
            # 列数一致性与 NaN/Inf 检查
            if rows and len(values) != len(rows[0]):
                raise XAFSDataError("S1", f"{path.name}: 第 {idx} 行列数 "
                                      f"({len(values)}) 与首行不一致 ({len(rows[0])})")
            if not all(np.isfinite(v) for v in values):
                raise XAFSDataError("S1", f"{path.name}: 第 {idx} 行含 NaN/Inf")
            rows.append(values)
            continue
        # 合法 key=val 头部行（在数据行开始前可作为头部；数据行开始后若仍是纯头部格式也按头部处理）
        if "=" in s:
            key, val = s.split("=", 1)
            header[key.strip()] = val.strip()
            continue
        # 数据行已开始后，遇到不可解行必须报错
        if rows:
            raise XAFSDataError("S1", f"{path.name}: 第 {idx} 行无法解析为数值行")
        # 数据行未开始前的纯文本头部（如 XDAC V1.2 Datafile V1）保持兼容跳过
        # （非 key=val 且非全浮点的行，若看起来像损坏数据行则报错）
        if len(tokens) >= 2 and not all(v is None for v in values):
            # 部分可解析但不完整的行，视为损坏
            raise XAFSDataError("S1", f"{path.name}: 第 {idx} 行无法解析为数值行")

    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        raise XAFSDataError("S1", f"{path.name}: 未解析到数值行")
    if arr.shape[1] < 3:
        raise XAFSDataError("S1", f"{path.name}: 仅 {arr.shape[1]} 列，无法区分 I0/It")

    energy = arr[:, 0]
    i0, it = arr[:, 1], arr[:, 2]
    iref = arr[:, 3] if arr.shape[1] >= 4 else None
    dtime = arr[:, 4] if arr.shape[1] >= 5 else None
    assert_channel_roles(i0, it, energy, path.name)

    line2 = lines[1] if len(lines) > 1 else ""
    meta = {
        "format": "XDAC",
        "mode": "trans",
        "path": str(path),
        "n_points": len(energy),
        "beamline": line2.rsplit(" on ", 1)[-1].strip() if " on " in line2 else None,
        "e0_nominal": _to_float(header.get("E0", "")),
        "header": {k: v for k, v in header.items()
                   if k in ("E0", "NUM_REGIONS", "SRB", "SRSS", "Offsets", "Gains")},
        "channels": ["Energy", "I0", "It"] + (["Ir"] if iref is not None else []),
    }
    return RawScan(energy=energy, i0=i0, it=it, iref=iref, dtime=dtime,
                   mu=np.log(i0 / it), meta=meta)


def read_xdi(path) -> RawScan:
    """
    XDI 1.0（NSLS-II / APS / ESRF 现代输出）。

    结构: '#' 注释 + '# Column.N: name unit' 声明 + 数值表
    坑位: **列名行本身也被 '#' 注释**，按「跳过所有 # 行」实现就读不到列名。

    通道角色按优先级判定：
        xmu 列存在        → mode="given"
        I0 与 It 都在     → mode="trans"
        I0 与 If/FF 都在  → mode="fluor"
    """
    path = Path(path)
    kv, colmap, colunits, colnames, rows = {}, {}, {}, [], []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("# ").strip()
            if body.startswith("Column.") and ":" in body:
                key, val = body.split(":", 1)
                parts = val.strip().split()
                if parts:
                    cname = parts[0].lower()
                    try:
                        col_idx = int(key.split(".")[1]) - 1
                    except ValueError as exc:
                        raise XAFSDataError(
                            "S1", f"{path.name}: 第 {line_number} 行列声明编号"
                                  f"非法: {key}") from exc
                    if col_idx < 0:
                        raise XAFSDataError(
                            "S1", f"{path.name}: 第 {line_number} 行列声明编号"
                                  f"必须 >= 1: {key}")
                    if col_idx in colmap.values():
                        raise XAFSDataError(
                            "S1", f"{path.name}: 第 {line_number} 行列号重复: {key}")
                    colmap[cname] = col_idx
                    colunits[cname] = parts[1] if len(parts) > 1 else None
            elif ":" in body:
                key, val = body.split(":", 1)
                kv[key.strip()] = val.strip()
            elif " " in body:
                cand = body.split()
                if any(c.lower() in ("energy", "xmu") for c in cand):
                    colnames = [c.lower() for c in cand]
            continue
        try:
            row = [float(t) for t in s.split()]
        except ValueError as exc:
            raise XAFSDataError(
                "S1", f"{path.name}: 第 {line_number} 行有无效数值") from exc
        width = len(rows[0]) if rows else (max(colmap.values()) + 1
                                         if colmap else len(colnames))
        if width and len(row) != width:
            raise XAFSDataError(
                "S1", f"{path.name}: 第 {line_number} 行列数 {len(row)} != {width}")
        if not np.all(np.isfinite(row)):
            raise XAFSDataError(
                "S1", f"{path.name}: 第 {line_number} 行包含非有限数值")
        rows.append(row)

    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        raise XAFSDataError("S1", f"{path.name}: 未解析到数值行")
    names = colnames or sorted(colmap, key=colmap.get)
    # 真实列位置：# Column.N 声明时用编号-1；仅有列名行时按出现顺序
    name_pos = {}
    if colmap:
        name_pos.update(colmap)
    else:
        for i, nm in enumerate(names):
            name_pos[nm] = i

    def pick(*candidates):
        for name in candidates:
            if name in name_pos:
                pos = name_pos[name]
                if 0 <= pos < arr.shape[1]:
                    return arr[:, pos]
        return None

    energy = pick("energy")
    if energy is None:
        raise XAFSDataError("S1", f"{path.name}: 找不到 energy 列")
    energy, energy_unit, energy_factor = _convert_energy_unit(
        energy, colunits.get("energy"), path.name)

    # 列名兼容 XDI 规范的两套写法：
    #   Demeter 系 it/if/ir/xmu ；larch/BMM 系 itrans/ifluor/irefer/mutrans。
    #   BMM 标准库(mds2-4032)中透射谱为 energy,...,mutrans,i0,itrans,irefer，
    #   缺这些别名时 70/359 个谱全部误判为「无 xmu，无法配对通道」。
    i0 = pick("i0")
    it = pick("it", "itrans")
    iref = pick("ir", "irefer")
    ifluor = pick("if", "ff", "i1", "ifluor", "fluor", "fluorescence")
    mu_given = pick("xmu", "mu", "mutrans", "mufluor")
    # 积分时间（measurement_time/seconds）是采集时长，不是荧光死时间分数；
    # 只记入 meta 溯源，绝不喂给 scan.dtime——否则 apply_deadtime_correction
    # 会把 0.5 s 当死时间分数 → I/(1−0.5) 把整条谱系统性放大 2 倍。
    integ = pick("measurement_time", "inttime", "integration_time", "count_time")
    # 只有显式命名的单通道死时间列（dtime/deadtime）才进 scan.dtime，
    # 且仅荧光模式有意义；透射谱的 dtime 语义是积分时间，不做死时间校正。
    dtime_col = pick("dtime", "deadtime", "dead_time")

    # ---- B9-3 多通道荧光探测器（MCA / 多 If 通道）----
    def _numbered(prefix):
        pairs = []
        for nm in name_pos:
            tail = nm[len(prefix):] if nm.startswith(prefix) else ""
            if tail.isdigit():
                pairs.append((int(tail), nm))
        return [nm for _, nm in sorted(pairs)]

    mca_names = _numbered("mca") or _numbered("if")
    mca_channels = dtime_channels = None
    has_xmu_with_mca = False
    if len(mca_names) >= 2:
        mca_channels = np.column_stack([arr[:, name_pos[nm]]
                                        for nm in mca_names if nm in name_pos])
        dt_names = _numbered("dt")
        if len(dt_names) == len(mca_names):
            dtime_channels = np.column_stack([arr[:, name_pos[nm]]
                                              for nm in dt_names if nm in name_pos])
        ifluor = np.sum(mca_channels, axis=1)
        # 通道求和只是"物理总荧光"的第一步；坏道剔除/死时间校正由
        # athena.merge_mca_channels 在 S1.2 完成。此处不得提前平均，
        # 否则坏道的 Bragg spike 会被稀释到无法识别。
        if i0 is not None and mu_given is None:
            mode = "fluor"
            if np.any(i0 <= 0):
                raise XAFSDataError(
                    "S1", f"{path.name}: 多通道荧光模式入射通道 I0 含非正值")
            if np.any(ifluor <= 0):
                raise XAFSDataError(
                    "S1", f"{path.name}: 多通道荧光总和 If 含非正值")
            mu = ifluor / i0
        else:
            mode = "given" if mu_given is not None else "fluor"
            mu = mu_given if mu_given is not None else ifluor
            has_xmu_with_mca = mu_given is not None

    if mca_channels is None:
        if mu_given is not None:
            mode, mu = "given", mu_given
        elif i0 is not None and it is not None:
            mode = "trans"
            assert_channel_roles(i0, it, energy, path.name)
            mu = np.log(i0 / it)
        elif i0 is not None and ifluor is not None:
            mode = "fluor"
            if np.any(i0 <= 0):
                raise XAFSDataError("S1", f"{path.name}: 荧光模式入射通道 I0 含非正值")
            if np.any(ifluor <= 0):
                raise XAFSDataError("S1", f"{path.name}: 荧光通道 If 含非正值")
            mu = ifluor / i0            # 荧光模式绝不取对数
        else:
            raise XAFSDataError("S1", f"{path.name}: 无 xmu，也无法配对 I0/It 或 I0/If")

    # 单通道死时间列只在荧光模式生效；多通道荧光的死时间矩阵在 meta["mca"]
    # 里由 merge_mca_channels 处理，此处 scan.dtime 保持 None。
    dtime = dtime_col if (mode == "fluor" and mca_channels is None) else None

    meta = {
        "format": "XDI",
        "mode": mode,
        "path": str(path),
        "n_points": len(energy),
        "beamline": kv.get("Beamline.name"),
        "sample": kv.get("Sample.name"),
        "element": kv.get("Element.symbol"),
        "e0_nominal": _safe_float(kv.get("Scan.edge_energy")),
        "channels": names,
        "energy_unit": energy_unit,
        "energy_unit_factor": energy_factor,
    }
    plot_hint = str(kv.get("Scan.plot_hint") or "").split("--", 1)[0].strip()
    if plot_hint in ("ln(I0/It)", "ln(It/Ir)"):
        meta["mu_channel_roles"] = plot_hint
    if integ is not None:
        meta["integration_time_s"] = float(np.median(integ))
    if mca_channels is not None:
        meta["mca"] = {
            "channels": mca_channels,
            "dtime": dtime_channels,
            "channel_names": mca_names,
        }
        # 当同时存在 xmu 与 mca1,mca2 时，保持 mode="given"，不覆写为 fluor_multichannel
        if not has_xmu_with_mca:
            meta["mode"] = "fluor_multichannel"
        meta["channel_sum_note"] = ("mu 为各通道求和（坏道剔除由 "
                                    "athena.merge_mca_channels 执行）")
    if mu_given is not None and i0 is not None and it is not None:
        # 非正 I0/It 取对数会写 NaN/inf 进 meta；此时只跳过自检，不改数据。
        if np.all(i0 > 0) and np.all(it > 0):
            meta["xmu_selfcheck"] = float(
                np.max(np.abs(np.log(i0 / it) - mu_given)))
    return RawScan(energy=energy, i0=i0, it=it, ifluor=ifluor, iref=iref,
                   dtime=dtime, mu=mu, meta=meta)


def read_simple(path) -> RawScan:
    """
    无格式声明的纯文本。
        2 列 → 视作 (E, μ) 已算好（查中位数量级，挡住把 I0 当 μ）
        3~4 列 → 视作 (E, I0, It[, Ir])，按透射处理
    """
    path = Path(path)
    rows, header_names = [], []
    in_numeric = False
    expected_cols = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("# ").strip()
            if body and not any(ch.isdigit() for ch in body[:3]):
                cand = body.replace(",", " ").split()
                if len(cand) >= 2:
                    header_names = [c.lower() for c in cand]
            continue
        tokens = s.replace(",", " ").split()
        try:
            vals = [float(t) for t in tokens]
        except ValueError as exc:
            if in_numeric:
                raise XAFSDataError("S1", f"{path.name}: 第 {line_no} 行有无效数值") from exc
            if len(tokens) >= 2 and not header_names:
                header_names = [t.lower() for t in tokens]
            continue
        if not np.all(np.isfinite(vals)):
            raise XAFSDataError("S1", f"{path.name}: 第 {line_no} 行含 NaN/Inf")
        if expected_cols is None:
            expected_cols = len(vals)
        elif len(vals) != expected_cols:
            raise XAFSDataError("S1", f"{path.name}: 第 {line_no} 行列数 ({len(vals)}) 与前文 ({expected_cols}) 不一致")
        in_numeric = True
        rows.append(vals)

    if not rows:
        raise XAFSDataError("S1", f"{path.name}: 未解析到有效数值行")
    arr = np.asarray(rows, dtype=float)
    ncol = arr.shape[1] if arr.ndim == 2 else 0
    base = {"path": str(path), "n_points": len(arr), "channels": header_names}

    if ncol == 2:
        mu = arr[:, 1]
        mu_median = float(np.median(mu))
        # 量级检查：μ(E)（无论透射对数还是荧光比值）中位数不可能到几百以上；
        # 若给的是 I0/It 等原始光强（典型 1e4–1e6），在此挡住而不是静默当 μ 用。
        if not (1e-3 < mu_median < 50.0):
            raise XAFSDataError(
                "S1", f"{path.name}: 第 2 列中位数 = {mu_median:.4g}，不像 μ(E)"
                      f"（合理范围约 0.001–50）。若该列是 I0/It 等原始光强，"
                      f"请改用 3 列格式 (E, I0, It)，由程序取对数算 μ")
        return RawScan(energy=arr[:, 0], mu=mu,
                       meta={**base, "format": "SIMPLE_2COL", "mode": "given"})
    if ncol >= 3 and len(header_names) == ncol:
        e_aliases = {"energy", "e", "en", "e(ev)", "energy(ev)", "e_ev"}
        fluo_aliases = {"i1/i0", "if/i0", "ff/i0", "ifluor/i0", "fluo/i0", "fluo"}
        trans_mu_aliases = {"ln(i0/i1)", "ln(i0/it)", "log(i0/i1)", "log(i0/it)", "mu", "xmu"}
        idx_e = next((i for i, h in enumerate(header_names) if h in e_aliases), None)
        if idx_e is not None:
            energy_col = arr[:, idx_e]
            idx_i0 = next((i for i, h in enumerate(header_names) if h == "i0"), None)
            idx_it = next((i for i, h in enumerate(header_names) if h in {"it", "itrans", "i1"}), None)
            idx_ir = next((i for i, h in enumerate(header_names) if h in {"ir", "iref", "i2"}), None)
            i0_col = arr[:, idx_i0] if idx_i0 is not None else None
            it_col = arr[:, idx_it] if idx_it is not None else None
            ir_col = arr[:, idx_ir] if idx_ir is not None else None
            idx_fluo = next((i for i, h in enumerate(header_names) if h in fluo_aliases), None)
            idx_mu = next((i for i, h in enumerate(header_names) if h in trans_mu_aliases), None)
            if idx_fluo is not None:
                return RawScan(
                    energy=energy_col, i0=i0_col, ifluor=it_col, iref=ir_col,
                    mu=arr[:, idx_fluo],
                    meta={**base, "format": "SIMPLE_NAMED", "mode": "fluo"},
                )
            if idx_mu is not None:
                h_mu = header_names[idx_mu]
                mode_val = "trans" if h_mu.startswith(("ln(", "log(")) else "given"
                return RawScan(
                    energy=energy_col, i0=i0_col, it=it_col, iref=ir_col,
                    mu=arr[:, idx_mu],
                    meta={**base, "format": "SIMPLE_NAMED", "mode": mode_val},
                )
            if i0_col is not None and it_col is not None:
                assert_channel_roles(i0_col, it_col, energy_col, path.name)
                return RawScan(
                    energy=energy_col, i0=i0_col, it=it_col, iref=ir_col,
                    mu=np.log(i0_col / it_col),
                    meta={**base, "format": "SIMPLE_NAMED", "mode": "trans"},
                )
    if ncol >= 3:
        assert_channel_roles(arr[:, 1], arr[:, 2], arr[:, 0], path.name)
        return RawScan(energy=arr[:, 0], i0=arr[:, 1], it=arr[:, 2],
                       iref=arr[:, 3] if ncol > 3 else None,
                       mu=np.log(arr[:, 1] / arr[:, 2]),
                       meta={**base, "format": "SIMPLE_3COL", "mode": "trans"})
    raise XAFSDataError("S1", f"{path.name}: 无法识别的列数 {ncol}")


# ================================================================
# 4. 统一入口
# ================================================================

_PARSERS = {"XDAC": read_xdac, "XDI": read_xdi, "SIMPLE": read_simple}


def split_scan_spec(spec) -> tuple[Path, str | None]:
    """拆分 ``file.prj::Group`` 规格为 ``(Path, group_or_none)``（兼容 Windows 盘符单冒号）。"""
    s = str(spec)
    if "::" in s:
        left, right = s.rsplit("::", 1)
        lp = Path(left)
        if lp.suffix.lower() == ".prj" and right.strip():
            return lp, right.strip()
    return Path(s), None


def scan_spec_label(spec) -> str:
    """提取可读标签：``.prj::Group`` 返回 Group，普通文件返回 stem。"""
    path, group = split_scan_spec(spec)
    return group if group is not None else path.stem


def scan_spec_stem(spec) -> str:
    """生成安全文件名前缀：``.prj::Group`` 返回 ``prjstem__Group``，普通文件保持 ``stem``。"""
    import re

    path, group = split_scan_spec(spec)
    if group is None:
        return path.stem
    safe_g = re.sub(r"[^\w.-]+", "_", group).strip("_") or "group"
    return f"{path.stem}__{safe_g}"


def _resolve_prj_scan(doc: dict, file_path: Path, group: str | None) -> tuple[RawScan, str]:
    scans = list(doc.get("scans") or [])
    if not scans:
        raise XAFSConfigError("S1", f"{file_path.name}: .prj 中未解析到有效扫描")
    avail = [
        str((s.meta or {}).get("label") or (s.meta or {}).get("tag") or f"scan_{i}")
        for i, s in enumerate(scans)
    ]
    if group is None:
        if len(scans) == 1:
            return scans[0], avail[0]
        raise XAFSConfigError(
            "S1",
            f"{file_path.name} 包含 {len(scans)} 条扫描，请用 '{file_path.name}::<组名>' "
            f"或 --prj-group 指定：{avail}",
        )
    want = str(group).strip()
    norm = lambda s: " ".join(str(s).replace("_", " ").split()).lower()
    for idx, sc in enumerate(scans):
        meta = sc.meta or {}
        if want in (str(meta.get("label") or ""), str(meta.get("tag") or "")):
            return sc, avail[idx]
    want_n = norm(want)
    for idx, sc in enumerate(scans):
        meta = sc.meta or {}
        if want_n in (norm(meta.get("label") or ""), norm(meta.get("tag") or "")):
            return sc, avail[idx]
    raise XAFSConfigError("S1", f"{file_path.name}: 未找到组 {group!r}；可用组：{avail}")


def read_scan(path, *, prj_group: str | None = None,
              apply_eshift: bool = False) -> RawScan:
    """按格式分发解析，返回 RawScan；原生支持 Athena ``file.prj::Group``。"""
    file_path, spec_group = split_scan_spec(path)
    if file_path.suffix.lower() == ".prj":
        doc = read_prj(file_path, as_scans=True, apply_eshift=apply_eshift)
        scan, chosen = _resolve_prj_scan(doc, file_path, spec_group or prj_group)
        scan.meta = dict(scan.meta or {})
        scan.meta["path"] = str(file_path)
        scan.meta["prj_group"] = chosen
        scan.meta.setdefault("mode", "given")
        scan.meta.setdefault(
            "format_detected",
            str((doc.get("meta") or {}).get("source_format") or "PRJ"),
        )
        return _order_energy_axis(scan, f"{file_path.name}::{chosen}")
    fmt = sniff_format(file_path)
    scan = _PARSERS[fmt](file_path)
    scan.meta.setdefault("format_detected", fmt)
    return _order_energy_axis(scan, file_path.name)


def read_scans(paths, *, prj_group: str | None = None,
               apply_eshift: bool = False) -> list:
    """批量读取；单文件也返回长度为 1 的列表。"""
    return [read_scan(p, prj_group=prj_group, apply_eshift=apply_eshift)
            for p in paths]


# ================================================================
# 5. 写出
# ================================================================

def write_columns(path, header: str, columns, column_names) -> Path:
    """
    写多列文本表。

    columns     : 可迭代的 ndarray，长度必须一致
    column_names: 列名列表，写进表头
    """
    path = Path(path)
    arrays = [np.asarray(c, dtype=float) for c in columns]
    if not arrays:
        raise XAFSConfigError("write", f"{path.name}: 没有列可写出")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise XAFSConfigError("write", f"{path.name}: 各列长度不一致")
    # 格式化/编码/防覆盖统一交给 xafs_writer；此处只保留异常类型契约。
    return xafs_writer.write_columns(path, header, column_names, arrays)


def write_wide_csv(path, index, index_name: str, series: dict) -> Path:
    """
    宽表：一列自变量 + 每条谱一列，可直接导入 Origin。
    series 形如 {"scan1": arr1, "merge": arr2}
    """
    # 宽表语义（一列 X + 每谱一列 Y）由 xafs_writer 统一保证，绝不改成长表。
    return xafs_writer.write_wide_csv(path, index, index_name, series)


def write_json(path, payload) -> Path:
    """把阶段状态落盘，便于断点续跑与复现。

    序列化规则与原子写/防覆盖统一委托 ``xafs_writer``（红线 5）。
    """
    return xafs_writer.safe_write_json(path, payload,
                                       default=xafs_writer.json_default)


# ================================================================
# 5.5 `.prj` 工程容器（B9-3）
# ================================================================

_PRJ_SCAN_FIELDS = ("energy", "i0", "it", "ifluor", "iref", "dtime",
                    "mu", "sigma_mu")


def _prj_default(obj):
    """gzip-JSON 容器的序列化器：numpy / Path / dataclass 都落成纯 JSON。"""
    import dataclasses

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: v for k, v in dataclasses.asdict(obj).items()}
    raise TypeError(f"无法序列化 {type(obj)}")


def write_prj(path, payload, protect: bool = True) -> Path:
    """写 gzip 压缩的 JSON 工程容器（`.prj`）。

    容器结构::

        {"xafs_prj": 1, "scans": [...], "standards": {...}, "meta": {...}}

    ``payload`` 可为任意 JSON 可序列化字典；``scans`` 里的 RawScan
    dataclass 会被自动展开。使用 gzip 是为了避免大数组把工程文件
    膨胀到几百 MB（实测 3 条 1000 点扫描 JSON≈70 KB，gzip≈18 KB）。
    注意：这是本项目的容器，**不是** Athena/Demeter 的原生 Perl `.prj`；
    后者请先导出为文本/JSON。

    写入前调用 xafs_writer._prepare 备份已有文件，并先写临时文件再
    os.replace 原子替换（与 xafs_writer 同口径）。
    """
    import gzip

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, dict):
        raise XAFSConfigError("prj", "write_prj 的 payload 必须是字典")
    xafs_writer._prepare(path, protect=protect)
    doc = {"xafs_prj": 1}
    doc.update(payload)
    raw = json.dumps(doc, ensure_ascii=False, default=_prj_default,
                     separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with gzip.open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def _athena_split_top(text: str) -> list[str]:
    """按顶层逗号切分 Perl 列表，跳过引号与嵌套 ``{}`` / ``[]`` / ``()``。"""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = None
    esc = False
    for ch in text:
        if quote:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _athena_unquote(token: str) -> str:
    """去掉 Perl 单/双引号并还原最小转义；非字符串 token 原样返回。"""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        quote = token[0]
        body = token[1:-1]
        return body.replace("\\" + quote, quote).replace("\\\\", "\\")
    return token


def _athena_balanced(text: str, start: int) -> str | None:
    """从 ``@args = (`` / ``@x = (`` 的左括号起取平衡括号内容。"""
    depth = 0
    quote = None
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def _athena_args(block: str) -> dict:
    """解析 ``@args = (key, value, ...)``；嵌套结构整体保留为字符串值。"""
    import re

    match = re.search(r"@args\s*=\s*\(", block)
    if match is None:
        return {}
    body = _athena_balanced(block, match.end() - 1)
    if body is None:
        return {}
    tokens = _athena_split_top(body)
    args: dict = {}
    for i in range(0, len(tokens) - 1, 2):
        args[_athena_unquote(tokens[i])] = _athena_unquote(tokens[i + 1])
    return args


def _athena_array(block: str, name: str):
    """解析 ``@x = (...)`` 浮点数组；缺失返回 None，坏 token 记 NaN。"""
    import re

    match = re.search(r"@" + re.escape(name) + r"\s*=\s*\(", block)
    if match is None:
        return None
    body = _athena_balanced(block, match.end() - 1)
    if body is None:
        return None
    values = []
    for token in _athena_split_top(body):
        try:
            values.append(float(_athena_unquote(token)))
        except ValueError:
            values.append(np.nan)
    if not values:
        return None
    return np.asarray(values, dtype=float)


def read_athena_perl_prj(path_or_text, as_scans: bool = False,
                         include_derived: bool = False,
                         apply_eshift: bool = False) -> dict:
    """B16-1：只读解析 Athena 原生 Perl ``.prj`` 工程文件。

    判定为合法 Athena 工程的充要条件是文本同时含 ``@args =`` 与至少一组
    非空的 ``@x =`` / ``@y =`` 数组；否则抛 ``XAFSConfigError``（保护
    既有 #50：任意 ``$VAR1`` Perl 片段不得被猜成扫描）。

    记录以行首 ``[record]`` 分隔（后面常带 ``# create`` 注释）。每条记录：
      - ``@args`` 偶数下标为键、奇数下标为值（嵌套 ``{}``/``[]`` 跳过）；
      - ``label`` 缺省时退到 ``tag`` / ``old_group``；
      - 默认过滤参比通道派生记录（``label`` 去空格以 ``Ref `` 开头或
        ``tag`` 以 ``d_Ref_`` 开头）与合并派生记录（``is_merge∉{"", "0"}``）；
      - ``ln=1`` 且有等长非零 ``@i0``/``@signal`` 时按 μ=ln(I₀/I_signal)
        还原（实测 Ref 记录 ``@y`` 为负值）；否则直接取 ``@y``。
    """
    import gzip
    import re

    if isinstance(path_or_text, bytes):
        blob = path_or_text
        source = "<bytes>"
    elif isinstance(path_or_text, Path):
        if not path_or_text.is_file():
            raise XAFSConfigError("prj", f"工程文件不存在: {path_or_text}")
        blob = path_or_text.read_bytes()
        source = str(path_or_text)
    elif isinstance(path_or_text, str):
        if "\n" in path_or_text or "\r" in path_or_text:
            blob = path_or_text.encode("utf-8")
            source = "<text>"
        else:
            p = Path(path_or_text)
            if p.is_file():
                blob = p.read_bytes()
                source = str(p)
            else:
                blob = path_or_text.encode("utf-8")
                source = "<text>"
    else:
        raise XAFSConfigError(
            "prj", f"read_athena_perl_prj 不支持 {type(path_or_text)}")
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except OSError as exc:
            raise XAFSConfigError("prj", f"{source}: gzip 解压失败: {exc}")
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")

    if not (re.search(r"@args\s*=", text)
            and re.search(r"@x\s*=", text)
            and re.search(r"@y\s*=", text)):
        raise XAFSConfigError(
            "prj", f"{source}: 检测到 Perl 文本但缺少 @args/@x/@y 扫描记录，"
                   f"不是可识别的 Athena 原生工程（拒绝猜测格式）")

    blocks = re.split(r"(?m)^\[record\].*$", text)
    records = []
    for idx, block in enumerate(blocks):
        if "@args" not in block:
            continue
        args = _athena_args(block)
        x = _athena_array(block, "x")
        y = _athena_array(block, "y")
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue
        label = str(args.get("label") or args.get("tag")
                    or args.get("old_group") or f"record_{idx}")
        tag = str(args.get("tag", ""))
        is_ref = (label.strip().startswith("Ref ")
                  or tag.startswith("d_Ref_"))
        is_merge = str(args.get("is_merge", "")) not in ("", "0")
        records.append({"index": idx, "block": block, "args": args,
                        "x": x, "y": y, "label": label,
                        "is_ref": is_ref, "is_merge": is_merge})

    if not records:
        raise XAFSConfigError(
            "prj", f"{source}: @args/@x/@y 标记存在但没有非空扫描记录")

    scans = []
    for rec in records:
        if not include_derived and (rec["is_ref"] or rec["is_merge"]):
            continue
        args = rec["args"]
        x, y = rec["x"], rec["y"]
        if x.size != y.size:
            raise XAFSConfigError(
                "prj", f"{source}: 记录 {rec['label']!r} 的 @x/@y 长度"
                       f"不一致（{x.size} vs {y.size}）")
        i0 = _athena_array(rec["block"], "i0")
        signal = _athena_array(rec["block"], "signal")
        if i0 is not None and i0.size != x.size:
            i0 = None
        if signal is not None and signal.size != x.size:
            signal = None
        mu = y
        computed_trans = False
        if (i0 is not None and signal is not None
                and np.any(i0 != 0) and np.any(signal != 0)
                and str(args.get("ln", "1")) == "1"
                and str(args.get("is_nor", "0")) != "1"):
            ratio = np.divide(i0, signal,
                              out=np.full_like(i0, np.nan),
                              where=signal != 0)
            if np.all(np.isfinite(ratio)) and np.all(ratio > 0):
                mu = np.log(ratio)
                computed_trans = True
        ln_flag = str(args.get("ln", ""))
        is_nor_flag = str(args.get("is_nor", "0"))
        if computed_trans:
            mode_val = "trans"
            it_arr, if_arr = signal, None
        elif ln_flag == "0" and signal is not None and is_nor_flag != "1":
            mode_val = "fluor"
            it_arr, if_arr = None, signal
        else:
            mode_val = "given"
            it_arr, if_arr = signal, None
        e0_value = args.get("bkg_e0")
        try:
            e0_float = float(e0_value) if e0_value is not None else None
        except (TypeError, ValueError):
            e0_float = None
        try:
            eshift_float = float(args["bkg_eshift"]) if args.get("bkg_eshift") is not None else 0.0
        except (TypeError, ValueError):
            eshift_float = 0.0
        if not np.isfinite(eshift_float):
            eshift_float = 0.0
        athena_bkg = {}
        for bk_key in ("bkg_rbkg", "bkg_pre1", "bkg_pre2", "bkg_nor1", "bkg_nor2", "bkg_spl1", "bkg_spl2"):
            if args.get(bk_key) is not None:
                try:
                    athena_bkg[bk_key.removeprefix("bkg_")] = float(args[bk_key])
                except (TypeError, ValueError):
                    pass
        x_out = (x + eshift_float) if (apply_eshift and eshift_float != 0.0) else x
        meta = {
            "format_detected": "athena_perl",
            "source_format": "athena_perl",
            "mode": mode_val,
            "label": rec["label"],
            "tag": str(args.get("tag", "")),
            "absorber": str(args.get("bkg_z", "")),
            "e0": e0_float,
            "eshift": eshift_float,
            "eshift_applied": bool(apply_eshift),
            "athena_bkg": athena_bkg,
            "is_merge": str(args.get("is_merge", "")),
            "is_nor": str(args.get("is_nor", "")),
            "ln": str(args.get("ln", "")),
            "source": str(args.get("source", "")),
            "record_index": rec["index"],
        }
        if abs(eshift_float) > 1e-3 and not apply_eshift:
            meta["eshift_unapplied_warning"] = (
                f"Athena .prj 记录了能量平移校准 bkg_eshift={eshift_float:+.3f} eV，"
                f"默认保持原始能量未平移（若需按 Athena 对齐刻度读取请传 apply_eshift=True）"
            )
        scans.append(RawScan(energy=x_out, i0=i0, it=it_arr, ifluor=if_arr, mu=mu, meta=meta))

    doc = {
        "xafs_prj": "athena_perl_v1",
        "scans": ([_prj_default(s) for s in scans] if not as_scans else scans),
        "standards": {},
        "meta": {"source_format": "athena_perl",
                 "n_records": len(records),
                 "n_scans": len(scans),
                 "source": source},
    }
    return doc


def read_prj(path, as_scans: bool = False,
             include_derived: bool = False,
             apply_eshift: bool = False) -> dict:
    """读 `.prj` 工程容器；兼容未压缩 JSON 与 Athena 原生 Perl 文本。

    JSON 通道保持 B9-3 契约；非 JSON 文本只有在同时含 ``@args =``、
    ``@x =``、``@y =`` 且至少一条非空扫描时才转发给
    ``read_athena_perl_prj``，否则抛 ``XAFSConfigError``（保护 #50）。
    """
    import gzip

    path = Path(path)
    if not path.is_file():
        raise XAFSConfigError("prj", f"工程文件不存在: {path}")
    blob = path.read_bytes()
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except OSError as exc:
            raise XAFSConfigError("prj", f"{path.name}: gzip 解压失败: {exc}")
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        import re
        if (re.search(r"@args\s*=", text)
                and re.search(r"@x\s*=", text)
                and re.search(r"@y\s*=", text)):
            return read_athena_perl_prj(
                text, as_scans=as_scans, include_derived=include_derived,
                apply_eshift=apply_eshift)
        raise XAFSConfigError(
            "prj", f"{path.name}: 不是有效 JSON（{exc.msg}），也不是含 "
                   f"@args/@x/@y 扫描记录的 Athena 原生工程。"
                   f"若为其他 Perl 序列化，请先在 Athena 导出为文本/JSON")
    if not isinstance(doc, dict) or doc.get("xafs_prj") is None:
        raise XAFSConfigError(
            "prj", f"{path.name}: 缺少 xafs_prj 版本标记，拒绝猜测格式")
    if as_scans and isinstance(doc.get("scans"), list):
        scans = []
        for item in doc["scans"]:
            if not isinstance(item, dict) or "energy" not in item:
                continue
            kwargs = {}
            for key in _PRJ_SCAN_FIELDS:
                if item.get(key) is not None:
                    kwargs[key] = np.asarray(item[key], dtype=float)
            kwargs["meta"] = dict(item.get("meta") or {})
            scans.append(RawScan(**kwargs))
        doc["scans"] = scans
    return doc


# ================================================================
# 6. 自测
# ================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = (Path(__file__).resolve().parents[3] / "data" / "练手数据" /
                  "02_Fe_standards" / "Fe_lepidocrocite.000")
    scan = read_scan(target)
    print(f"文件      : {Path(target).name}")
    print(f"格式      : {scan.meta['format_detected']}  mode={scan.meta['mode']}")
    print(f"点数      : {len(scan)}")
    print(f"能量范围  : {scan.energy.min():.1f} – {scan.energy.max():.1f} eV")
    print(f"μ 范围    : {scan.mu.min():.4f} – {scan.mu.max():.4f}")
