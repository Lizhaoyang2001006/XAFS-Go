# -*- coding: utf-8 -*-
"""XAFS 回归测试包：由 regression_test.py 纯拆分的 mixin 集合。"""
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import xafs  # noqa: F401
