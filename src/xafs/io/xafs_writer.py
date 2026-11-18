# -*- coding: utf-8 -*-
"""xafs_writer.py — 统一产物落盘层（格式化 + 防静默覆盖 + 并发安全）

架构分工（对应架构审查建议「巨型 CLI 入口导致职责耦合」）：

    run_xafs.py   CLI 参数解析、调度、流程编排
    xafs_report / xafs_io   内容组装（指标、表格、图、阶段状态）
    **xafs_writer**  序列化、编码、目录创建、原子写、防覆盖、并发锁

把落盘收拢到这里有两个目的：

1. **红线 5「保留用户既有分析记录」**：旧实现全链路用 ``open(..., "w")``
   或 ``Path.write_text``，对同一 ``--outdir`` 重跑会静默抹掉上次的
   ``*_fit_table.csv`` / ``*_meta.json`` / ``fit.json`` / ``*_report.html``。
   本模块在覆盖前先把旧文件复制到 ``.backup/<时间戳>/``，既不丢数据，
   也不阻断重复运行。所有目录（包括临时目录）遵循相同的保护语义。
2. **并发安全**：Web 后端多线程共用一个 outdir 时，写文件用锁 + 临时文件
   原子替换，避免两个请求交错写出半截 JSON/CSV。

所有函数返回 ``Path``，与旧 API 完全兼容；不改变任何文件的内容格式。
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# numpy 只在数值列格式化 / JSON 序列化时才需要；保持顶层零第三方依赖，
# 使 run_ci.py 等入口在编译检查阶段也能安全导入本模块。
# 原定义在 xafs_report.FIT_HEADER，收拢到落盘层，
# xafs_report 以别名 re-export 保持公共 API 不变。
FIT_HEADER = ["Sample", "Shell", "CN", "R (Å)", "σ² (Å²)",
              "ΔE₀ (eV)", "R-factor"]

PROTECT_EXISTING = True          # 全局开关；设为 False 可恢复旧的直接覆盖语义
_WRITE_LOCK = threading.RLock()


def _backup_existing(path: Path) -> Path | None:
    """把既有文件复制到同级 .backup/<时间戳>/，返回备份路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest_dir = path.parent / ".backup" / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    shutil.copy2(path, dest)
    return dest


def _prepare(path: Path, protect: bool | None) -> Path | None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    do_protect = PROTECT_EXISTING if protect is None else bool(protect)
    if do_protect and path.exists():
        return _backup_existing(path)
    return None


def _atomic_write(path: Path, text: str, *, encoding: str,
                  newline: str | None) -> None:
    """先写同目录临时文件再 os.replace，保证读者永远看到完整文件。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline=newline) as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def safe_write_text(path, text: str, *, encoding: str = "utf-8",
                    newline: str | None = None,
                    protect: bool | None = None) -> Path:
    """写纯文本（HTML / feff.inp / 阶段文本），覆盖前备份旧版本。"""
    path = Path(path)
    with _WRITE_LOCK:
        _prepare(path, protect)
        _atomic_write(path, text, encoding=encoding, newline=newline)
    return path


def json_default(obj: Any) -> Any:
    """与旧 xafs_io.write_json 一致的 numpy / Path 序列化规则。"""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"无法序列化 {type(obj)}")


def safe_write_json(path, payload: Any, *, default=None,
                    protect: bool | None = None) -> Path:
    """写阶段状态 / meta / fit JSON；indent=2 + ensure_ascii=False 保持不变。"""
    path = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False,
                      default=default or json_default)
    return safe_write_text(path, text, encoding="utf-8", protect=protect)


def safe_write_csv(path, header: Iterable[str], rows: Iterable[Iterable[Any]],
                   *, encoding: str = "utf-8-sig",
                   protect: bool | None = None) -> Path:
    """写 CSV（utf-8-sig，Excel/Origin 可直接打开）。"""
    import io

    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(list(header))
    writer.writerows([list(r) for r in rows])
    return safe_write_text(path, buf.getvalue(), encoding=encoding,
                           newline="", protect=protect)


@contextmanager
def csv_buffer(path, *, encoding: str = "utf-8-sig",
               protect: bool | None = None):
    """替换 ``open(path, "w", newline="", ...)`` + ``csv.writer`` 组合。

    用法（把原来的 ``with open(...) as fh:`` 直接换成一行即可，块内代码
    与 ``csv.writer(fh)`` 完全不用改）::

        with xafs_writer.csv_buffer(path, encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    缓冲内容在块**正常结束**后由 ``safe_write_text`` 原子落盘并按需备份；
    块内若抛异常则不写任何文件，避免留下半截 CSV。StringIO 与文件对象
    在 ``newline=""`` 下的换行/编码行为一致，产物逐字节不变。
    """
    import io

    buf = io.StringIO(newline="")
    completed = False
    try:
        yield buf
        completed = True
    finally:
        if completed:
            safe_write_text(Path(path), buf.getvalue(), encoding=encoding,
                            newline="", protect=protect)
        buf.close()


def write_fit_table(rows: list, out, stem: str,
                    *, protect: bool | None = None) -> Path:
    """拟合参数表 CSV（从 xafs_report.export_fit_table 抽离）。

    有 C3 值时追加第 8 列，与历史产物逐位一致。
    """
    has_c3 = any(len(r) > 7 and str(r[7]).strip() for r in rows)
    header = list(FIT_HEADER) + (["C3 (Å³)"] if has_c3 else [])
    body = rows if has_c3 else [r[:7] for r in rows]
    return safe_write_csv(Path(out) / f"{stem}_fit_table.csv",
                          header, body, protect=protect)


def write_columns(path, header: str, column_names: Iterable[str],
                  columns: Iterable[Any], *,
                  protect: bool | None = None) -> Path:
    """ASCII 列数据（# 注释头 + 科学计数法），格式与旧 xafs_io 一致。"""
    import numpy as np

    path = Path(path)
    arrays = [np.asarray(c, dtype=float) for c in columns]
    if not arrays:
        raise ValueError(f"{path.name}: 没有列")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError(f"{path.name}: 各列长度不一致")
    lines = [header if header.endswith("\n") else header + "\n",
             "# " + "  ".join(str(x) for x in column_names) + "\n"]
    lines += ["  ".join(f"{a[i]:.6e}" for a in arrays) + "\n"
              for i in range(n)]
    return safe_write_text(path, "".join(lines), encoding="utf-8",
                           protect=protect)


def write_wide_csv(path, index, index_name: str, series: dict,
                   *, protect: bool | None = None) -> Path:
    """宽表：一列自变量 + 每条谱一列（Origin 可直接导入）。

    保持用户约定的宽表格式，绝不改成长表。
    """
    path = Path(path)
    names = list(series)
    n_idx = len(index)
    for n in names:
        if len(series[n]) != n_idx:
            raise ValueError(f"write_wide_csv: 列 {n!r} 长度 {len(series[n])} 与索引长度 {n_idx} 不一致")
    lines = [",".join([index_name] + names) + "\n"]
    lines += [",".join([f"{index[i]:.6f}"]
                       + [f"{series[n][i]:.6e}" for n in names]) + "\n"
              for i in range(len(index))]
    return safe_write_text(path, "".join(lines), encoding="utf-8-sig",
                           protect=protect)


def safe_savefig(fig, path, *, protect: bool | None = None, **kwargs) -> Path:
    """matplotlib 图件落盘（PNG/PDF/SVG）：覆盖前备份旧版本，与文本产物同口径。

    savefig 是二进制写，不走 _atomic_write 的文本临时文件。这里先 _prepare
    把旧图备份到 .backup/<时间戳>/，再写同目录临时文件、os.replace 原子替换
    （按目标扩展名显式指定 format，保证字节与直接 fig.savefig(path) 逐位一致）。
    异常时清理临时文件，绝不留下 0 字节或损坏的目标图件。
    红线 5：所有产物落盘都必须防静默覆盖，图件也不例外。
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    save_kw = dict(kwargs)
    if "format" not in save_kw and path.suffix:
        save_kw["format"] = path.suffix.lstrip(".").lower()
    with _WRITE_LOCK:
        _prepare(path, protect)
        try:
            fig.savefig(tmp, **save_kw)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return path


# 简洁别名（文档/调用方常用名），全部指向同一套防覆盖实现，避免出现
# "两套 IO 入口" 再次造成静默覆盖。
write_text = safe_write_text
write_json = safe_write_json
write_csv = safe_write_csv
savefig = safe_savefig
