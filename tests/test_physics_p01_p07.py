# -*- coding: utf-8 -*-
"""针对 P-01 至 P-07 物理与预期一致性修复的单元测试（归档方案见 docs/archive/P01_P07_PHYSICS_FIX_PLAN.md）。

说明：
- 若当前环境缺少 `larch` 或 `gemmi`（纯轻量沙箱环境），仅对缺失顶层包做局部 stub，
  以允许不依赖 larch/gemmi 的纯 NumPy/SciPy 数学与编排逻辑完整自测；
- 若环境已安装真实 `larch` 与 `gemmi`，不注入任何 stub，直接使用真实包运行。
"""
from __future__ import annotations

import importlib.util
import json
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
    """当环境未安装 larch/gemmi 时注入最小导入占位符。返回是否使用了 stub。"""
    used_stub = False
    if "larch" not in sys.modules and importlib.util.find_spec("larch") is None:
        used_stub = True
        larch_mod = types.ModuleType("larch")
        larch_mod._is_xafs_test_stub = True

        class _Group:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        larch_mod.Group = _Group
        larch_xafs = types.ModuleType("larch.xafs")
        larch_xafs.autobk = lambda *a, **kw: None
        larch_xafs.xftf = lambda *a, **kw: None
        larch_xafs.xftr = lambda *a, **kw: None
        larch_xafs.pre_edge = lambda *a, **kw: None
        larch_xafs.feffdat = types.SimpleNamespace(
            feffpath=lambda *a, **kw: None,
            path2chi=lambda *a, **kw: None,
            ff2chi=lambda *a, **kw: None,
        )
        larch_mod.xafs = larch_xafs
        sys.modules["larch"] = larch_mod
        sys.modules["larch.xafs"] = larch_xafs
    elif getattr(sys.modules.get("larch"), "_is_xafs_test_stub", False):
        used_stub = True
    if "gemmi" not in sys.modules and importlib.util.find_spec("gemmi") is None:
        used_stub = True
        gemmi_mod = types.ModuleType("gemmi")
        gemmi_mod.cif = types.SimpleNamespace(read_file=lambda *a, **kw: None)
        sys.modules["gemmi"] = gemmi_mod
    if "xraydb" not in sys.modules and importlib.util.find_spec("xraydb") is None:
        used_stub = True
        xraydb_mod = types.ModuleType("xraydb")
        xraydb_mod.material_mu = lambda *a, **kw: 0.0
        xraydb_mod.material_mu_components = lambda *a, **kw: {}
        xraydb_mod.mu_elam = lambda *a, **kw: 0.0
        xraydb_mod.xray_edge = lambda *a, **kw: (7112.0, 0.0, 0.0)
        xraydb_mod.xray_line = lambda *a, **kw: (6404.0, 0.0)
        sys.modules["xraydb"] = xraydb_mod
    return used_stub


_USED_STUBS = _ensure_optional_stubs()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_xafs_test")

import artemis  # noqa: E402
import artemis_stats  # noqa: E402
import athena  # noqa: E402
import catalytic_builder as cb  # noqa: E402
import fit_conditional_cn  # noqa: E402
import fit_shell_plan  # noqa: E402
import fit_single  # noqa: E402
import run_xafs  # noqa: E402
import structure_lib  # noqa: E402
import xafs_pipeline  # noqa: E402
import xafs_report  # noqa: E402
from xafs_core import (  # noqa: E402
    ChiK,
    FitResult,
    FourierTransform,
    NormalizedScan,
    PathFit,
    PipelineResult,
    RawScan,
    WaveletMap,
    XAFSConfigError,
)


class PhysicsP01ToP07Tests(unittest.TestCase):
    """覆盖 P-01 至 P-07 修复的契约与物理量纲测试。"""

    # ------------------------------------------------------------------
    # P-01: 纳米颗粒半球模型 Å / nm 量纲对齐与边界校验
    # ------------------------------------------------------------------
    def test_p01_nanoparticle_unit_alignment_and_bounds(self):
        # d = 2.5 nm = 25.0 Å, default r1 = 2.55 Å -> u = 0.102
        res = fit_conditional_cn.parse_defect_model({
            "type": "nanoparticle",
            "diameter_nm": 2.5,
            "lattice": "fcc",
        })
        self.assertEqual(res["r1_A"], 2.55)
        self.assertAlmostEqual(res["u_ratio"], 2.55 / 25.0, places=6)
        expected = 1.0 - 1.5 * (2.55 / 25.0) + 0.5 * (2.55 / 25.0) ** 3
        self.assertAlmostEqual(res["expected_n_ratio"], expected, places=5)
        # 绝不能退化成错把 nm 当成 Å 的 1 - 1.5/2.5 = 0.4
        self.assertGreater(res["expected_n_ratio"], 0.80)

        # absorber 自动查共价半径 2*r_cov
        res_pt = fit_conditional_cn.parse_defect_model({
            "type": "nanoparticle",
            "diameter_nm": 2.0,
            "lattice": "fcc",
            "absorber": "Pt",
        })
        r1_pt = 2.0 * artemis.COVALENT_RADII["Pt"]
        self.assertAlmostEqual(res_pt["r1_A"], round(r1_pt, 4), places=4)
        self.assertAlmostEqual(res_pt["u_ratio"], r1_pt / 20.0, places=5)

        # hemispherical 形状有效球径缩小 d_eff = d / 2^(1/3) -> CN 压低更明显
        res_hemi = fit_conditional_cn.parse_defect_model({
            "type": "nanoparticle",
            "diameter_nm": 2.0,
            "lattice": "fcc",
            "shape": "hemispherical",
            "r1_A": 2.77,
        })
        d_eff_a = 20.0 / (2.0 ** (1.0 / 3.0))
        self.assertAlmostEqual(res_hemi["u_ratio"], 2.77 / d_eff_a, places=5)
        self.assertLess(res_hemi["expected_n_ratio"], res_pt["expected_n_ratio"])

        # 非物理极小粒径 (u >= 1) 必须抛出 XAFSConfigError
        with self.assertRaises(XAFSConfigError):
            fit_conditional_cn.parse_defect_model({
                "type": "nanoparticle",
                "diameter_nm": 0.1,
                "lattice": "fcc",
                "r1_A": 2.55,
            })

    # ------------------------------------------------------------------
    # P-02: 单文件/预合并单条扫描在 align=True 时仍走参考箔对齐
    # ------------------------------------------------------------------
    def test_p02_single_scan_triggers_reference_alignment(self):
        e = np.linspace(7050.0, 7250.0, 41)
        scan = RawScan(
            energy=e,
            i0=np.ones_like(e),
            it=np.exp(-0.5 * np.ones_like(e)),
            iref=np.exp(-0.5 * np.ones_like(e)),
            mu=0.5 * np.ones_like(e),
            meta={"format_detected": "XDI", "mode": "trans"},
        )
        aligned_marker = {"called_with_len": None}

        def fake_align(scans_arg, edge_energy):
            aligned_marker["called_with_len"] = len(scans_arg)
            return scans_arg, {"shifts_eV": [0.25], "note": "aligned"}

        class _StopAfterAlign(RuntimeError):
            pass

        with patch("xafs_pipeline.xafs_io.read_scans", return_value=[scan]), \
             patch("xafs_pipeline.xafs_io.capture_measurement_identity", return_value=(False, "mock")), \
             patch("xafs_pipeline.athena.align_by_reference", side_effect=fake_align), \
             patch("xafs_pipeline.athena.merge_scans", side_effect=_StopAfterAlign()):
            with self.assertRaises(_StopAfterAlign):
                xafs_pipeline.process_sample(
                    ["single_scan.xdi"], absorber="Fe", align=True, verbose=False,
                )
        self.assertEqual(aligned_marker["called_with_len"], 1)

    # ------------------------------------------------------------------
    # P-03: 真实 Miller 平面切割 vs 旧体相顶角保护
    # ------------------------------------------------------------------
    def test_p03_cleave_miller_slab_and_terrace_contract(self):
        ceo2 = cb.BUILTIN_PROTOTYPES["ceo2"]
        model = cb.CrystalModel(ceo2["cell"], ceo2["atoms"], spacegroup="P1")
        slab = cb.cleave_miller_slab(model, "111")
        coords = np.array([[x, y, z] for _, x, y, z in slab], dtype=float)
        top_z = float(np.max(coords[:, 2]))
        outer_mask = coords[:, 2] >= (top_z - 0.5)
        top_mask = coords[:, 2] >= (top_z - 1.5)
        # 真实 (111) 平面应暴露出平整密集的顶表面原子台面（而非孤立单点棱角）
        self.assertGreaterEqual(int(np.sum(outer_mask)), 4)
        # 顶层台面 (max_z - 1.5 Å) 横向中位中心应精确平移至 (0, 0)
        xy_med = np.median(coords[top_mask, :2], axis=0)
        self.assertLess(float(np.linalg.norm(xy_med)), 1e-6)

        anchor = cb.find_surface_anchoring_site(
            slab, site_type="top", anchor_atom="Pt", target_bond=2.02, terrace_radius=4.5,
        )
        # 锚定坐标应当落在台面内部而非超胞边缘
        self.assertLess(float(math.hypot(anchor[0], anchor[1])), 4.5)
        self.assertGreater(float(anchor[2]), top_z)

        # 向后兼容契约：裸 "111"/"100"/"110" 继续抛 XAFSConfigError；"cleave:111" 显式启用切割
        for legacy_facet in ("111", "100", "110"):
            with self.assertRaises(XAFSConfigError):
                structure_lib.build_algorithmic_catalytic_motif(
                    "Pt", {"surface": legacy_facet, "support": "ceo2"},
                )

        # 验证 cleave:111 能调用真实 Miller 切割生成 Motif
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(structure_lib, "DYNAMIC_CACHE", Path(tmp)):
                motif = structure_lib.build_algorithmic_catalytic_motif(
                    "Pt",
                    {"surface": "cleave:111", "support": "ceo2", "site": "top"},
                )
        self.assertEqual(motif.absorber, "Pt")
        self.assertIn("cleave_111", motif.id)

    # ------------------------------------------------------------------
    # P-04: 四腿相干往返 4x 标度 & 桥角放行 & artemis_fit_single 兼容性
    # ------------------------------------------------------------------
    def test_p04_collinear_ms_sigma2_scaling_and_bridge_angle(self):
        # Path 1: 4-leg coherent rattle (Pt -> O -> Pt -> O -> Pt), Reff=4.0, first=2.0
        p_coherent = SimpleNamespace(
            nleg=4, reff=4.0, degen=6.0,
            geom=[
                ["Pt", 0, 0, 0, 0.0, 0.0, 0.0],
                ["O",  1, 0, 0, 2.0, 0.0, 0.0],
                ["Pt", 0, 0, 0, 0.0, 0.0, 0.0],
                ["O",  1, 0, 0, 2.0, 0.0, 0.0],
            ],
        )
        # Path 2: 4-leg heteronuclear symmetric chain (Pt -> O1 -> Pt -> O2 -> Pt), Reff=4.0
        p_hetero = SimpleNamespace(
            nleg=4, reff=4.0, degen=6.0,
            geom=[
                ["Pt", 0, 0, 0,  0.0, 0.0, 0.0],
                ["O",  1, 0, 0,  2.0, 0.0, 0.0],
                ["Pt", 0, 0, 0,  0.0, 0.0, 0.0],
                ["O",  2, 0, 0, -2.0, 0.0, 0.0],
            ],
        )
        # Path 3: 3-leg bent bridge (Pt1 -> O -> Pt2 -> Pt1) with bridge angle 150° (cos ≈ -0.866)
        ang = math.radians(30.0)  # 180° - 150° = 30° deflection from straight
        p2 = (2.0 + 2.0 * math.cos(ang), 2.0 * math.sin(ang), 0.0)
        reff_bent = 0.5 * (2.0 + 2.0 + math.hypot(p2[0], p2[1]))
        p_bent = SimpleNamespace(
            nleg=3, reff=reff_bent, degen=12.0,
            geom=[
                ["Pt", 0, 0, 0, 0.0, 0.0, 0.0],
                ["O",  1, 0, 0, 2.0, 0.0, 0.0],
                ["Pt", 2, 0, 0, p2[0], p2[1], p2[2]],
            ],
        )
        # 检查几何内角确实约 150° (cos ≈ -0.866 < 0.95)
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            for idx in (1, 2, 3):
                (tdir / f"feff{idx:04d}.dat").write_text("", encoding="utf-8")
            mapping = {
                "feff0001.dat": p_coherent,
                "feff0002.dat": p_hetero,
                "feff0003.dat": p_bent,
            }
            motif = structure_lib.Motif(
                id="test_ms", absorber="Pt", kind="crystal",
                nuclearity="crystal",
            )
            with patch("fit_shell_plan.structure_lib.ensure_feff", return_value=tdir), \
                 patch("fit_shell_plan.artemis.load_feff_path", side_effect=lambda f: mapping[Path(f).name]):
                strict_rows = fit_shell_plan._collinear_ms_rows(
                    motif, "L3", rmax=5.0, min_abs_cos=0.95, min_bridge_cos=None,
                )
                bridge_rows = fit_shell_plan._collinear_ms_rows(
                    motif, "L3", rmax=5.0, min_abs_cos=0.95, min_bridge_cos=0.65,
                )

        # 未开 min_bridge_cos 时，150° 弯曲桥连（顶点 cos = -0.866 < 0.95 且远端也不严格共线？检查）
        # 注意远端散射原子在等腰三角形底角约 15° -> |cos| = cos(15°) ≈ 0.966 >= 0.95；
        # 仅桥顶点 |cos(150°)| = 0.866 < 0.95 被 strict_rows 拒绝，而被 bridge_rows 放行
        strict_ids = {r["path_id"]: r for r in strict_rows}
        bridge_ids = {r["path_id"]: r for r in bridge_rows}
        self.assertNotIn(3, strict_ids)
        self.assertIn(3, bridge_ids)
        self.assertAlmostEqual(bridge_ids[3]["bridge_angle_deg"], 150.0, delta=0.5)

        self.assertTrue(strict_ids[1]["coherent_rattle"])
        self.assertAlmostEqual(strict_ids[1]["sigma2_ratio"], 4.0, places=5)
        self.assertFalse(strict_ids[2]["coherent_rattle"])
        self.assertAlmostEqual(strict_ids[2]["sigma2_ratio"], 2.0, places=5)

    # ------------------------------------------------------------------
    # P-05: 自由度缩放误差 N_err_nidp / R_err_nidp 导出且不破坏 CSV 基线
    # ------------------------------------------------------------------
    def test_p05_nidp_scaled_error_exports(self):
        path_fit = PathFit(
            label="Fe-O@1.98", reff=1.98, degeneracy=6.0,
            delta_r=0.01, delta_r_err=0.01, sigma2=0.005, sigma2_err=0.001,
            n_err=0.5,
        )
        fit_res = FitResult(
            paths=[path_fit], s02=0.803, delta_e0=1.2, delta_e0_err=0.5,
            r_factor=0.01, n_independent=10.0, n_variables=4,
            kmin=3.0, kmax=12.0, rmin=1.0, rmax=3.0,
            nidp_scale_factor=2.0,
        )
        fit_out = {
            "mode": "motif",
            "fit": fit_res,
            "flags": {"nidp_scale_factor": 2.0},
            "cn_reliable": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            meta = run_xafs._fit_outputs(fit_out, Path(tmp), "sample")
            csv_text = (Path(tmp) / "sample_fit.csv").read_text(encoding="utf-8")
            json_meta = json.loads((Path(tmp) / "sample_fit.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["fit"]["nidp_scale_factor"], 2.0)
        p_json = json_meta["fit"]["paths"][0]
        self.assertAlmostEqual(p_json["N_err"], 0.5)
        self.assertAlmostEqual(p_json["N_err_nidp"], 1.0)
        self.assertAlmostEqual(p_json["R_err"], 0.01)
        self.assertAlmostEqual(p_json["R_err_nidp"], 0.02)
        # CSV 基线 N_err 列必须仍保持未乘 nidp_scale_factor 的原始值 0.5
        self.assertIn(",0.5,", csv_text)

        summary = xafs_report.fit_summary(fit_out)
        self.assertEqual(summary["fit"]["nidp_scale_factor"], 2.0)
        self.assertAlmostEqual(summary["fit"]["paths"][0]["N_err_nidp"], 1.0)
        self.assertAlmostEqual(summary["fit"]["paths"][0]["R_err_nidp"], 0.02)

        budget = fit_single.systematic_budget(fit_res, flags=fit_out["flags"])
        self.assertEqual(budget["nidp_scale_factor"], 2.0)

    # ------------------------------------------------------------------
    # P-06: 合并壳层静态劈裂方差 σ²_geom 与跨度统计
    # ------------------------------------------------------------------
    def test_p06_shell_static_splitting_metrics(self):
        p_short = SimpleNamespace(
            nleg=2, reff=1.88, degen=2.0,
            geom=[["Fe", 0, 0, 0, 0, 0, 0], ["O", 1, 0, 0, 1.88, 0, 0]],
        )
        p_long = SimpleNamespace(
            nleg=2, reff=2.04, degen=4.0,
            geom=[["Fe", 0, 0, 0, 0, 0, 0], ["O", 1, 0, 0, 2.04, 0, 0]],
        )
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            (tdir / "feff0001.dat").write_text("", encoding="utf-8")
            (tdir / "feff0002.dat").write_text("", encoding="utf-8")
            mapping = {"feff0001.dat": p_short, "feff0002.dat": p_long}
            motif = structure_lib.Motif(
                id="test_split", absorber="Fe", kind="crystal",
                nuclearity="crystal",
            )
            with patch("fit_shell_plan.structure_lib.ensure_feff", return_value=tdir), \
                 patch("fit_shell_plan.artemis.load_feff_path", side_effect=lambda f: mapping[Path(f).name]):
                plan = fit_shell_plan.select_shell_paths(
                    motif, edge="K", rmin=1.0, rmax=3.0, merge_tol=0.20,
                    kmin=3.0, kmax=12.0,
                )
        self.assertEqual(len(plan["shells"]), 1)
        sh = plan["shells"][0]
        r_mean = (2.0 * 1.88 + 4.0 * 2.04) / 6.0
        s2_geom = (2.0 * (1.88 - r_mean) ** 2 + 4.0 * (2.04 - r_mean) ** 2) / 6.0
        self.assertAlmostEqual(sh["split_span_A"], 0.16, places=4)
        self.assertAlmostEqual(sh["sigma2_geom"], s2_geom, places=5)
        self.assertAlmostEqual(sh["rep_reff_offset_A"], 2.04 - r_mean, places=4)

    # ------------------------------------------------------------------
    # P-07: 透射过厚警告（不触动荧光 cn_gate） & 累积量热力学键长/单峰诊断
    # ------------------------------------------------------------------
    def test_p07_transmission_thickness_and_cumulant_rdf_diagnostics(self):
        e = np.linspace(7050.0, 7250.0, 10)
        for edge_step, expected_risk in ((3.0, "high"), (1.8, "moderate"), (0.9, "low")):
            raw = RawScan(
                energy=e, i0=np.ones_like(e), it=np.full_like(e, 0.5),
                mu=np.full_like(e, math.log(2.0)),
                meta={"mode": "trans", "mu_channel_roles": "ln(I0/It)"},
            )
            res = SimpleNamespace(
                merged=raw,
                norm=SimpleNamespace(edge_step=edge_step),
                self_absorption=None,
            )
            diag = fit_single._self_absorption_diag(res)
            self.assertEqual(diag["risk"], "none")
            self.assertFalse(diag["cn_gate"])
            self.assertEqual(diag["transmission_thickness_risk"], expected_risk)

        r_grid = np.linspace(1.75, 2.30, 500)
        rdf_ok = artemis_stats.reconstruct_pair_distribution(
            r_grid, n=6.0, r0=2.00, sigma2=0.005, c3=2.0e-5, c4=0.0, lambda_eff=8.0,
        )
        expected_shift = (2.0 * 0.005 / 2.00) * (1.0 + 2.00 / 8.0)
        self.assertAlmostEqual(rdf_ok["cumulant_c1_shift_A"], expected_shift, places=7)
        self.assertAlmostEqual(rdf_ok["r_thermodynamic"], 2.00 + expected_shift, places=7)
        self.assertTrue(rdf_ok["unimodal"])
        self.assertFalse(rdf_ok["gram_charlier_clipped"])

        # 过大的 C3 / C4 导致 Gram-Charlier 截断与非物理旁瓣（双峰/多峰）
        rdf_bad = artemis_stats.reconstruct_pair_distribution(
            r_grid, n=6.0, r0=2.00, sigma2=0.004, c3=0.0, c4=1.5e-4,
        )
        self.assertTrue(rdf_bad["gram_charlier_clipped"])
        self.assertFalse(rdf_bad["unimodal"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
