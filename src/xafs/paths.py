# -*- coding: utf-8 -*-
"""src/xafs/paths.py — 全局仓库根目录与关键目录权威路径定位。

无论模块位于 src/xafs/{domain,adapters,io,core,cli}/ 哪一层，
均通过本模块解析仓库根目录，严禁在业务子层硬编码 ``Path(__file__).resolve().parent``。
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
STRUCTURE_LIB_ROOT = REPO_ROOT / "_structure_lib"
FEFF_CACHE_ROOT = REPO_ROOT / "_feff_cache"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
