# -*- coding: utf-8 -*-
"""xafs — 分层 XAFS 光谱处理与定量结构解析核心包。

层级架构（严格单向向下依赖）：
  cli      -> 展示、报表、交互调窗与论文出图
  core     -> S1-S7 管线编排、结构识别、催化建模、拟合与物理门禁诊断
  io       -> 数据读写、原子写盘防覆盖、CIF 门禁与相库溯源
  adapters -> 第三方黑盒科学计算库隔离层（xraylarch / xraydb / FEFF / 小波）
  domain   -> 物理常数、冻结数据结构契约、12 锁定元素表与配置
"""
from __future__ import annotations

import sys
from pathlib import Path

from .paths import REPO_ROOT

_LAYER_NAMES = ("domain", "adapters", "io", "core", "cli")


def ensure_repo_paths() -> Path:
    """将仓库根、src/、五层包目录及 scripts/ 注入 sys.path（幂等、保序）。"""
    pkg_dir = Path(__file__).resolve().parent
    src_dir = pkg_dir.parent
    candidates = [
        str(REPO_ROOT),
        str(src_dir),
        *[str(pkg_dir / layer) for layer in _LAYER_NAMES],
        str(REPO_ROOT / "scripts"),
    ]
    for path_str in reversed(candidates):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return REPO_ROOT


ensure_repo_paths()
