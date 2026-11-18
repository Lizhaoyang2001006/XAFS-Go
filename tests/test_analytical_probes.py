# -*- coding: utf-8 -*-
"""L0/L1/L2 解析解探针与物理不变量测试（对齐 PHYSICS_SPEC.md 与 AGENTS.md）。

说明：
- [EQ-01], [EQ-04], [EQ-06], [EQ-07] 均为纯 NumPy/SciPy 数学与门禁契约测试；
- [EQ-03] `ft_to_r_space` 依赖真实 `larch.xafs.xftf`：仅在安装真实 `larch` 时运行，
  未安装时显式 skip（绝不用 stub 伪造第三方库数值行为）。
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import xafs  # noqa: E402,F401


def _ensure_optional_stubs() -> bool:
    """仅当环境缺失 larch/xraydb/gemmi 时注入最小导入占位符（不用于跳过实测探针）。"""
    used_stub = False
    if "larch" not in sys.modules and importlib.util.find_spec("larch") is None:
        used_stub = True
        larch_mod = types.ModuleType("larch")
        larch_mod._is_xafs_test_stub = True
        larch_mod.Group = type("Group", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})
        larch_xafs = types.ModuleType("larch.xafs")
        larch_xafs.autobk = lambda *a, **kw: None
        larch_xafs.xftf = lambda *a, **kw: None
        larch_xafs.xftr = lambda *a, **kw: None
        larch_xafs.pre_edge = lambda *a, **kw: None
        larch_mod.xafs = larch_xafs
        sys.modules["larch"] = larch_mod
        sys.modules["larch.xafs"] = larch_xafs
    elif getattr(sys.modules.get("larch"), "_is_xafs_test_stub", False):
        used_stub = True
    if "xraydb" not in sys.modules and importlib.util.find_spec("xraydb") is None:
        used_stub = True
        xraydb_mod = types.ModuleType("xraydb")
        for fn in ("material_mu", "material_mu_components", "mu_elam", "xray_edge", "xray_line"):
            setattr(xraydb_mod, fn, lambda *a, **kw: 0.0)
        sys.modules["xraydb"] = xraydb_mod
    if "gemmi" not in sys.modules and importlib.util.find_spec("gemmi") is None:
        used_stub = True
        gemmi_mod = types.ModuleType("gemmi")
        gemmi_mod.cif = types.SimpleNamespace(read_file=lambda *a, **kw: None)
        sys.modules["gemmi"] = gemmi_mod
    return used_stub


_USED_STUBS = _ensure_optional_stubs()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_xafs_test")

import artemis_stats  # noqa: E402
import athena_bkg_ft  # noqa: E402
import fit_conditional_cn  # noqa: E402
import fit_shell_plan  # noqa: E402
import structure_lib  # noqa: E402
from xafs_core import K2, FitResult, PathFit  # noqa: E402


class AnalyticalProbeTests(unittest.TestCase):
    """[PHYSICS_SPEC] 解析解探针与物理不变量校验。"""

    def test_eq01_wave_number_analytical_and_below_edge_clamp(self):
        """[EQ-01] E <= E0 严格截断为 0；E > E0 精确服从 sqrt(K2*(E-E0))。"""
        e0 = 7112.0
        energies = np.array([7090.0, 7112.0, 7112.0 + 100.0, 7112.0 + 400.0])
        k = athena_bkg_ft.wave_number(energies, e0=e0)
        self.assertTrue(np.all(np.isfinite(k)))
        self.assertEqual(float(k[0]), 0.0)
        self.assertEqual(float(k[1]), 0.0)
        self.assertAlmostEqual(float(k[2]), math.sqrt(K2 * 100.0), places=10)
        self.assertAlmostEqual(float(k[3]), math.sqrt(K2 * 400.0), places=10)

    def test_eq04_stern_nidp_exact_no_plus2_and_overfit_boundary(self):
        """[EQ-04] N_idp 严格按 2*dk*dr/pi（不加 2），等号边界同样触发 overfit 与壳层截断。"""
        nidp = artemis_stats.independent_points(kmin=3.0, kmax=13.0, rmin=1.0, rmax=3.0)
        self.assertAlmostEqual(nidp, 2.0 * 10.0 * 2.0 / math.pi, places=12)

        dummy_path = PathFit(
            label="Fe-O@2.00", reff=2.0, degeneracy=6.0,
            delta_r=0.0, delta_r_err=0.01, sigma2=0.005, sigma2_err=0.001,
        )
        # N_var == N_idp 必须触发 overfit=True
        fit_eq = FitResult(
            paths=[dummy_path], s02=0.803, delta_e0=0.0, delta_e0_err=0.5,
            r_factor=0.01, n_independent=10.0, n_variables=10,
            kmin=3.0, kmax=13.0, rmin=1.0, rmax=3.0,
        )
        self.assertTrue(fit_eq.overfit)

        #窄 R/k 窗下 N_idp 仅够容纳 1~2 壳层时，远壳必须被截断记入 truncated
        p1 = SimpleNamespace(nleg=2, reff=2.00, degen=6.0,
                             geom=[["Fe", 0, 0, 0, 0, 0, 0], ["O", 1, 0, 0, 2.0, 0, 0]])
        p2 = SimpleNamespace(nleg=2, reff=2.85, degen=4.0,
                             geom=[["Fe", 0, 0, 0, 0, 0, 0], ["Fe", 2, 0, 0, 2.85, 0, 0]])
        p3 = SimpleNamespace(nleg=2, reff=3.40, degen=8.0,
                             geom=[["Fe", 0, 0, 0, 0, 0, 0], ["Fe", 2, 0, 0, 3.40, 0, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            mapping = {}
            for idx, obj in enumerate((p1, p2, p3), start=1):
                fname = f"feff{idx:04d}.dat"
                (tdir / fname).write_text("", encoding="utf-8")
                mapping[fname] = obj
            motif = structure_lib.Motif(
                id="probe_nidp", absorber="Fe", kind="crystal", nuclearity="crystal",
            )
            # N_idp = 2*(8-3)*(3.6-1.5)/pi = 6.68 -> 3n+1 < 6.68 => n <= 1
            with patch("fit_shell_plan.structure_lib.ensure_feff", return_value=tdir), \
                 patch("fit_shell_plan.artemis.load_feff_path", side_effect=lambda f: mapping[Path(f).name]):
                plan = fit_shell_plan.select_shell_paths(
                    motif, edge="K", rmin=1.5, rmax=3.6, kmin=3.0, kmax=8.0,
                )
        self.assertLess(len(plan["shells"]) * 3 + 1, plan["n_independent"])
        self.assertGreaterEqual(len(plan["truncated"]), 1)

    def test_eq06_nanoparticle_forward_inverse_closure_invariant(self):
        """[EQ-06] 正向粒径压低比与反向粒径反演在球形/半球形下严格互逆闭合。"""
        r1 = 2.55
        for shape in ("spherical", "hemispherical"):
            for d_nm in (1.5, 2.5, 4.0):
                fwd = fit_conditional_cn.parse_defect_model({
                    "type": "nanoparticle",
                    "diameter_nm": d_nm,
                    "lattice": "fcc",
                    "shape": shape,
                    "r1_A": r1,
                })
                n1_meas = 12.0 * fwd["expected_n_ratio"]
                inv = artemis_stats.estimate_nanoparticle_size(
                    n1=n1_meas, r1=r1, lattice="fcc", shape=shape,
                )
                self.assertAlmostEqual(inv["diameter_nm"], d_nm, places=3)

    def test_eq07_rdf_area_conservation_and_grid_invariance(self):
        """[EQ-07] 非高斯 g(R) 总面积严格归一到 N，加密网格下热力学均值与峰值稳定。"""
        r_coarse = np.linspace(1.70, 2.35, 300)
        r_fine = np.linspace(1.70, 2.35, 1200)
        out_c = artemis_stats.reconstruct_pair_distribution(
            r_coarse, n=6.0, r0=2.00, sigma2=0.005, c3=2.5e-5, c4=0.0,
        )
        out_f = artemis_stats.reconstruct_pair_distribution(
            r_fine, n=6.0, r0=2.00, sigma2=0.005, c3=2.5e-5, c4=0.0,
        )
        self.assertAlmostEqual(out_c["area"], 6.0, places=4)
        self.assertAlmostEqual(out_f["area"], 6.0, places=4)
        self.assertAlmostEqual(out_c["r_thermodynamic"], out_f["r_thermodynamic"], places=6)
        self.assertLess(out_f["r_mode"], out_f["r_mean"])
        self.assertGreater(out_f["r_thermodynamic"], out_f["r_mean"])

    @unittest.skipIf(_USED_STUBS, "需要真实 xraylarch 环境运行 ft_to_r_space 解析峰值探针")
    def test_eq03_ft_analytical_single_shell_peak_position(self):
        """[EQ-03] 理想无相移单壳层 chi(k) 经 larch xftf 变换后主峰精确落在 R0=2.500 Å。"""
        r_true = 2.500
        n_true = 6.0
        sigma2_true = 0.004
        k = np.arange(2.0, 16.0, 0.05)
        chi_k = (n_true / (k * r_true ** 2)) * np.exp(-2.0 * (k ** 2) * sigma2_true) * np.sin(2.0 * k * r_true)
        ft = athena_bkg_ft.ft_to_r_space(k, chi_k, kmin=3.0, kmax=15.0, kweight=2, dk=1.0)
        r_peak = float(ft.r[int(np.argmax(ft.magnitude))])
        self.assertLess(abs(r_peak - r_true), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
