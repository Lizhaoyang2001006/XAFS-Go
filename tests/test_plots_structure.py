# -*- coding: utf-8 -*-
"""tests/test_plots_structure.py — 催化剂结构解析统一绘图系统单元测试。

覆盖 Scheme A 的纯数学与出图契约：
  - `xafs_plots.hann_window`（0 端点 / 平台 1 / 单调渐变）；
  - `xafs_plots.build_structure_main_figure`（6 面板 PNG/PDF 产出与非空校验）；
  - `xafs_plots.render_plot_suite(mode="none")`（不生成图件）；
  - `xafs_plot_special.write_rdf_figure`（C3/C4 非高斯 g(R) 与 r_thermodynamic/clipped/unimodal）；
  - `xafs_plot_special.write_summary_csv` & `write_fit_table_markdown`（格式、BOM 与防覆盖）；
  - CLI `--help` 对话（`run_xafs` / `xafs_report` 暴露 `--plot` 架构，剔除已退役开关）。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
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
        xdb = types.ModuleType("xraydb")
        xdb._is_xafs_test_stub = True
        sys.modules["xraydb"] = xdb
    elif getattr(sys.modules.get("xraydb"), "_is_xafs_test_stub", False):
        used_stub = True

    if "gemmi" not in sys.modules and importlib.util.find_spec("gemmi") is None:
        used_stub = True
        gem = types.ModuleType("gemmi")
        gem._is_xafs_test_stub = True
        sys.modules["gemmi"] = gem
    elif getattr(sys.modules.get("gemmi"), "_is_xafs_test_stub", False):
        used_stub = True
    return used_stub


_USED_STUBS = _ensure_optional_stubs()

import xafs_plot_special  # noqa: E402
import xafs_plots  # noqa: E402


def _synthetic_pipeline_result():
    e = np.linspace(7080.0, 7400.0, 321)
    e0 = 7120.0
    mu_norm = 0.5 + 0.5 * np.tanh((e - e0) / 3.0) + 0.35 * np.exp(-((e - (e0 + 5.0)) / 4.0) ** 2)
    k = np.linspace(0.0, 14.0, 281)
    chi = 0.05 * np.sin(2.0 * 1.9 * k + 0.2) * np.exp(-0.008 * k ** 2)
    r = np.linspace(0.0, 6.0, 121)
    mag = 1.6 * np.exp(-((r - 1.55) / 0.20) ** 2) + 0.7 * np.exp(-((r - 2.65) / 0.22) ** 2)
    chi_r = mag * np.exp(1j * 2.0 * r)
    kc = np.arange(3.0, 11.6, 0.25)
    wt_mag = np.exp(-((kc[:, None] - 5.2) / 2.2) ** 2) * np.exp(-((r[None, :] - 1.55) / 0.25) ** 2)

    norm = SimpleNamespace(energy=e, mu=mu_norm, pre_edge_line=np.zeros_like(e),
                           mu_norm=mu_norm, flattened=mu_norm, edge_step=1.2, plateau=1.0)
    chi_k = SimpleNamespace(k=k, chi=chi)
    ft = SimpleNamespace(r=r, chi_r=chi_r, real=chi_r.real, imag=chi_r.imag,
                         magnitude=mag, kmin=3.0, kmax=12.0, kweight=2, dk=1.0)
    wt = SimpleNamespace(k_centers=kc, r=r, magnitude=wt_mag, sigma=1.5, kweight=2)
    return SimpleNamespace(
        absorber_symbol="Fe",
        e0={"e0_sample": e0},
        norm=norm,
        chi_k=chi_k,
        ft=ft,
        wt=wt,
        r1=1.55,
        r_wt=1.55,
        k_wt=5.2,
        xanes_features={"neck_shoulder_present": False, "derivative_split_eV": 0.0},
        structure={"status": "confident", "ranked": [{"id": "fe_n4_planar", "score": 0.06}]},
        self_absorption={"mode": "trans", "risk": "none", "corrected": False, "cn_gate": False},
    )


class StructurePlotsUnitTests(unittest.TestCase):
    def test_hann_window_contract(self):
        k = np.linspace(0.0, 15.0, 1501)
        w = xafs_plots.hann_window(k, 3.0, 12.0, 1.0)
        self.assertTrue(np.allclose(w[k < 3.0], 0.0))
        self.assertTrue(np.allclose(w[k > 12.0], 0.0))
        mid = (k > 4.05) & (k < 10.95)
        self.assertTrue(np.allclose(w[mid], 1.0))

    def test_render_none_produces_nothing(self):
        res = _synthetic_pipeline_result()
        with tempfile.TemporaryDirectory() as td:
            out = xafs_plots.render_plot_suite(res, None, Path(td), "demo", mode="none")
            self.assertEqual(out["mode"], "none")
            self.assertEqual(out["images"], [])
            self.assertEqual(out["tables"], [])
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_build_structure_main_figure_and_summary_tables(self):
        res = _synthetic_pipeline_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            imgs = xafs_plots.build_structure_main_figure(
                res, None, out, "cat_sac", sample="Fe-N-C SAC", edge_label="Fe K-edge",
                formats=("png", "pdf"), dpi=90)
            self.assertEqual(len(imgs), 2)
            for p in imgs:
                self.assertTrue(p.is_file() and p.stat().st_size > 1000, f"图件异常: {p}")

            sc = xafs_plot_special.write_summary_csv(
                out, "cat_sac", res, None,
                indicator_rows=[("custom_metric", "1.23", "eV", "自定义指标")])
            text = sc.read_text(encoding="utf-8-sig")
            self.assertIn("structure_top1", text)
            self.assertIn("fe_n4_planar", text)
            self.assertIn("custom_metric", text)

            md = xafs_plot_special.write_fit_table_markdown(
                [["Fe-N-C", "Fe-N", "4.1±0.3", "1.99±0.01", "0.0042", "+1.2±0.5", "0.0120"]],
                out, "cat_sac")
            md_txt = md.read_text(encoding="utf-8")
            self.assertIn("| Sample | Shell |", md_txt)
            self.assertIn("Fe-N-C", md_txt)

    def test_write_rdf_figure_cumulants(self):
        path_obj = SimpleNamespace(
            label="Ni-O", degeneracy=6.0, r=2.08, sigma2=0.006,
            c3=1.8e-4, c4=4.0e-5, reff=2.06)
        fit_obj = SimpleNamespace(paths=[path_obj], r_factor=0.015)
        with tempfile.TemporaryDirectory() as td:
            written = xafs_plot_special.write_rdf_figure(
                {"fit": fit_obj}, Path(td), "nio", formats=("png",), dpi=90)
            self.assertEqual(len(written), 1)
            self.assertTrue(written[0].is_file() and written[0].stat().st_size > 1000)

    @unittest.skipIf(_USED_STUBS, "CLI import pulls larch/xraydb; verified on full env")
    def test_cli_exposes_plot_switches_without_legacy_flags(self):
        import run_xafs
        import xafs_report

        for mod in (run_xafs, xafs_report):
            buf = io.StringIO()
            with patch.object(sys, "argv", ["prog", "--help"]), \
                    contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    mod.main()
            help_txt = buf.getvalue()
            for flag in ("--plot", "--plot-formats", "--plot-dpi"):
                self.assertIn(flag, help_txt)
            for retired in ("--diagnostics", "--figure-pack", "--wt-sweep", "--no-window-tuner"):
                self.assertNotIn(retired, help_txt)

    @unittest.skipIf(_USED_STUBS, "multi-k-weight FT in shell dissection requires real xraylarch")
    def test_render_plot_suite_structure_mode_real_ft(self):
        res = _synthetic_pipeline_result()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            suite = xafs_plots.render_plot_suite(
                res, None, out, "cat_sac", mode="structure", formats=("png",), dpi=80)
            self.assertEqual(suite["mode"], "structure")
            self.assertTrue((out / "cat_sac_structure_main.png").is_file())
            self.assertTrue((out / "cat_sac_shell_dissection.png").is_file())
            self.assertTrue((out / "cat_sac_summary.csv").is_file())

    def test_ft_anchor_near_bond_uses_path_contrib(self):
        r = np.linspace(0.5, 4.5, 81)
        mag = 0.8 * np.exp(-((r - 2.75) / 0.14) ** 2) + 0.55 * np.exp(-((r - 3.20) / 0.14) ** 2)
        naive = xafs_plot_special.ft_anchor_near_bond(r, mag, 3.39)
        self.assertAlmostEqual(naive, 2.75, delta=0.1)
        path_contrib = {
            "r": r,
            "magnitude": np.exp(-((r - 3.20) / 0.15) ** 2),
        }
        anchored = xafs_plot_special.ft_anchor_near_bond(r, mag, 3.39, path_contrib=path_contrib)
        self.assertAlmostEqual(anchored, 3.20, delta=0.1)

    def test_assess_wt_shell_sanity_catches_clipping_and_order_inversion(self):
        res = _synthetic_pipeline_result()
        r = res.ft.r
        kc = np.arange(4.5, 11.6, 0.25)
        wt_bad = (
            np.exp(-((kc[:, None] - 5.7) / 1.5) ** 2) * np.exp(-((r[None, :] - 1.55) / 0.20) ** 2)
            + 0.4 * np.exp(-((kc[:, None] - 4.0) / 1.2) ** 2) * np.exp(-((r[None, :] - 2.65) / 0.20) ** 2)
        )
        res.wt = SimpleNamespace(k_centers=kc, r=r, magnitude=wt_bad, sigma=1.5, kweight=2)
        targets = [
            {"label": "Pt-O@2.00", "r_anchor": 1.55, "r_fit": 2.00, "sigma2": 0.0023, "element": "O", "z": 8},
            {"label": "Pt-Ce@3.39", "r_anchor": 2.65, "r_fit": 3.39, "sigma2": 0.0139, "element": "Ce", "z": 58},
        ]
        bad_sanity = xafs_plot_special.assess_wt_shell_sanity(res, None, shell_targets=targets)
        self.assertEqual(bad_sanity["status"], "alert")
        self.assertTrue(bad_sanity["boundary_clipped"])
        self.assertTrue(bad_sanity["wt_k_order_anomaly"])
        self.assertEqual(bad_sanity["high_sigma2_shells"], ["Pt-Ce@3.39"])

        wt_good = (
            np.exp(-((kc[:, None] - 5.2) / 1.5) ** 2) * np.exp(-((r[None, :] - 1.55) / 0.20) ** 2)
            + 0.5 * np.exp(-((kc[:, None] - 8.2) / 1.6) ** 2) * np.exp(-((r[None, :] - 2.65) / 0.20) ** 2)
        )
        res.wt = SimpleNamespace(k_centers=kc, r=r, magnitude=wt_good, sigma=1.5, kweight=2)
        targets_good = [
            {"label": "Fe-O@2.00", "r_anchor": 1.55, "r_fit": 2.00, "sigma2": 0.0040, "element": "O", "z": 8},
            {"label": "Fe-Fe@3.05", "r_anchor": 2.65, "r_fit": 3.05, "sigma2": 0.0060, "element": "Fe", "z": 26},
        ]
        good_sanity = xafs_plot_special.assess_wt_shell_sanity(res, None, shell_targets=targets_good)
        self.assertEqual(good_sanity["status"], "ok")
        self.assertFalse(good_sanity["boundary_clipped"])
        self.assertFalse(good_sanity["wt_k_order_anomaly"])

    def test_gate_card_blocks_false_pass_on_diagnostic_ftest_or_wt_anomaly(self):
        from matplotlib.axes import Axes

        res = _synthetic_pipeline_result()
        fit_obj = SimpleNamespace(
            paths=[
                SimpleNamespace(label="Pt-O@2.00", reff=2.00, delta_r=0.0, r=2.00,
                                degeneracy=4.0, n_err=0.1, sigma2=0.0023, delta_r_err=0.01, sigma2_err=0.0005),
                SimpleNamespace(label="Pt-Ce@3.39", reff=3.39, delta_r=0.0, r=3.39,
                                degeneracy=2.5, n_err=0.4, sigma2=0.0139, delta_r_err=0.01, sigma2_err=0.0015),
            ],
            s02=0.8075, delta_e0=7.2, delta_e0_err=0.2, r_factor=0.0289,
            n_independent=13.8, n_variables=7, kmin=3.0, kmax=12.0, rmin=1.15, rmax=3.55,
            model_k_grid=None, model_k=None, model_r=None, model_chi_r=None, correlation=None,
        )
        r = res.ft.r
        kc = np.arange(4.5, 11.6, 0.25)
        res.absorber_symbol = "Pt"
        res.wt = SimpleNamespace(
            k_centers=kc, r=r,
            magnitude=(np.exp(-((kc[:, None] - 5.7) / 1.5) ** 2) * np.exp(-((r[None, :] - 1.55) / 0.20) ** 2)
                       + 0.4 * np.exp(-((kc[:, None] - 4.0) / 1.2) ** 2) * np.exp(-((r[None, :] - 2.65) / 0.20) ** 2)),
            sigma=1.5, kweight=2,
        )
        bad_payload = {
            "mode": "motif", "status": "ambiguous",
            "cn_reliable": True, "cn_claim_allowed": False,
            "diagnostic_only": True, "cn_status": "hypothesis_model_only",
            "s02": 0.8075, "s02_source": "Pt_foil_T5",
            "fit": fit_obj, "flags": {},
            "shell_f_test": {"F": 0.86, "confidence": 0.493, "significant": False},
        }
        texts = []
        orig_text = Axes.text

        def _capture_text(self_ax, *args, **kwargs):
            if len(args) >= 3:
                texts.append(str(args[2]))
            elif "s" in kwargs:
                texts.append(str(kwargs["s"]))
            return orig_text(self_ax, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with patch.object(Axes, "text", new=_capture_text):
                xafs_plots.build_structure_main_figure(
                    res, bad_payload, out, "pt_bad", sample="Pt/CeO2", dpi=80)
            joined = "\n".join(texts)
            self.assertNotIn("PASS — quantitative CN allowed", joined)
            self.assertIn("HYPOTHESIS/GATED", joined)
            self.assertIn("F-test <95%", joined)

            sc = xafs_plot_special.write_summary_csv(out, "pt_bad", res, bad_payload)
            sc_txt = sc.read_text(encoding="utf-8-sig")
            self.assertIn("wt_k_sanity", sc_txt)
            self.assertIn("outer_shells_f_test_confidence", sc_txt)

            # Positive case: formal CN + valid F-test + ordered WT -> PASS allowed
            res.absorber_symbol = "Fe"
            res.wt = SimpleNamespace(
                k_centers=kc, r=r,
                magnitude=(np.exp(-((kc[:, None] - 5.2) / 1.5) ** 2) * np.exp(-((r[None, :] - 1.55) / 0.20) ** 2)
                           + 0.6 * np.exp(-((kc[:, None] - 8.2) / 1.5) ** 2) * np.exp(-((r[None, :] - 2.65) / 0.20) ** 2)),
                sigma=1.5, kweight=2,
            )
            fit_good = SimpleNamespace(**{**fit_obj.__dict__, "paths": [
                SimpleNamespace(label="Fe-O@2.00", reff=2.00, delta_r=0.0, r=2.00,
                                degeneracy=6.0, n_err=0.2, sigma2=0.0040, delta_r_err=0.01, sigma2_err=0.0005),
                SimpleNamespace(label="Fe-Fe@3.05", reff=3.05, delta_r=0.0, r=3.05,
                                degeneracy=4.0, n_err=0.3, sigma2=0.0060, delta_r_err=0.01, sigma2_err=0.0008),
            ]})
            good_payload = {
                "mode": "motif", "status": "confident",
                "cn_reliable": True, "cn_claim_allowed": True,
                "diagnostic_only": False, "cn_status": "formal_cn",
                "s02": 0.803, "s02_source": "absorber_static",
                "fit": fit_good, "flags": {},
                "shell_f_test": {"F": 12.4, "confidence": 0.999, "significant": True},
            }
            texts.clear()
            with patch.object(Axes, "text", new=_capture_text):
                xafs_plots.build_structure_main_figure(
                    res, good_payload, out, "fe_ok", sample="Fe-oxide", dpi=80)
            self.assertIn("Gate check: PASS — quantitative CN allowed", "\n".join(texts))

    def test_wt_sanity_bidirectional_light_on_heavy_masking_vs_unmodeled(self):
        """轻散射体错落高 k 重散射区时，必须区分“邻近重壳掩蔽”与“漏建重原子模型”两种情形。"""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        res = _synthetic_pipeline_result()
        r = res.ft.r
        kc = np.arange(4.5, 11.6, 0.25)
        res.absorber_symbol = "Pt"
        res.wt = SimpleNamespace(
            k_centers=kc, r=r,
            magnitude=(np.exp(-((kc[:, None] - 5.4) / 1.4) ** 2) * np.exp(-((r[None, :] - 1.50) / 0.20) ** 2)
                       + 0.7 * np.exp(-((kc[:, None] - 9.8) / 1.5) ** 2) * np.exp(-((r[None, :] - 3.06) / 0.25) ** 2)),
            sigma=1.5, kweight=2,
        )

        # Case A: 晶态氧化物模型含邻近重原子 Pt-Pt，轻外层 Pt-O2 被 Pt-Pt 高 k 峰脊掩蔽 -> 标 masked，禁标白圈误导
        masked_targets = [
            {"label": "Pt-O@2.02", "r_anchor": 1.50, "r_fit": 2.02, "sigma2": 0.0025, "element": "O", "z": 8, "k_peak_th": 5.1},
            {"label": "Pt-Pt@3.11", "r_anchor": 3.06, "r_fit": 3.11, "sigma2": 0.0040, "element": "Pt", "z": 78, "k_peak_th": 9.5},
            {"label": "Pt-O2@3.65", "r_anchor": 3.18, "r_fit": 3.65, "sigma2": 0.0030, "element": "O", "z": 8, "k_peak_th": 5.0},
        ]
        masked_sanity = xafs_plot_special.assess_wt_shell_sanity(res, None, shell_targets=masked_targets)
        self.assertEqual(masked_sanity["status"], "warn")
        self.assertEqual(masked_sanity["alerts"], [])
        self.assertFalse(masked_sanity["wt_k_order_anomaly"])
        o2_entry = [s for s in masked_sanity["shells"] if "Pt-O2" in s["label"]][0]
        self.assertTrue(o2_entry["masked_by_heavy"])
        self.assertEqual(o2_entry["masked_by"], "Pt-Pt@3.11")

        # 确认 Panel (e) 严禁画出 "Pt-O2 (9.8, ...)" 这类错误 WT 归属标记
        fit_masked = SimpleNamespace(
            paths=[
                SimpleNamespace(label="Pt-O@2.02", reff=2.02, delta_r=0.0, r=2.02,
                                degeneracy=6.0, n_err=0.2, sigma2=0.0025, delta_r_err=0.01, sigma2_err=0.0005),
                SimpleNamespace(label="Pt-Pt@3.11", reff=3.11, delta_r=0.0, r=3.11,
                                degeneracy=6.0, n_err=0.3, sigma2=0.0040, delta_r_err=0.01, sigma2_err=0.0005),
                SimpleNamespace(label="Pt-O2@3.65", reff=3.65, delta_r=0.0, r=3.65,
                                degeneracy=6.0, n_err=0.4, sigma2=0.0030, delta_r_err=0.01, sigma2_err=0.0005),
            ],
            s02=0.8257, delta_e0=6.5, delta_e0_err=0.5, r_factor=0.038,
            n_independent=14.3, n_variables=7, kmin=3.0, kmax=12.0, rmin=1.15, rmax=3.85,
            model_k_grid=None, model_k=None, model_r=None, model_chi_r=None, correlation=None,
        )
        masked_payload = {
            "mode": "motif", "status": "confident",
            "cn_reliable": True, "cn_claim_allowed": True,
            "diagnostic_only": False, "cn_status": "formal_cn",
            "s02": 0.8257, "fit": fit_masked, "flags": {},
        }
        fig, ax_e = plt.subplots()
        texts: list[str] = []
        orig_text = Axes.text

        def _cap(self_ax, *args, **kwargs):
            if len(args) >= 3:
                texts.append(str(args[2]))
            elif "s" in kwargs:
                texts.append(str(kwargs["s"]))
            return orig_text(self_ax, *args, **kwargs)

        try:
            with patch.object(Axes, "text", new=_cap):
                xafs_plots._annotate_wt_shell_panel(ax_e, res.wt, [], masked_sanity, 4.5, explicit_shells=False)
            panel_e_txt = "\n".join(texts)
            self.assertIn("masked by Pt-Pt@3.11", panel_e_txt)
            self.assertNotIn("Pt-O2@3.65 (9.8", panel_e_txt)
            self.assertIn("Pt-Pt@3.11 (9.8", panel_e_txt)
        finally:
            plt.close(fig)

        # Case B: 模型漏掉 Pt-Pt，仅用轻原子 Pt-O2 拟合 R≈3.06 Å（k=9.8）-> 硬告警拦截
        unmodeled_targets = [
            {"label": "Pt-O@2.02", "r_anchor": 1.50, "r_fit": 2.02, "sigma2": 0.0025, "element": "O", "z": 8, "k_peak_th": 5.1},
            {"label": "Pt-O2@3.50", "r_anchor": 3.06, "r_fit": 3.50, "sigma2": 0.0030, "element": "O", "z": 8, "k_peak_th": 5.0},
        ]
        bad_sanity = xafs_plot_special.assess_wt_shell_sanity(res, None, shell_targets=unmodeled_targets)
        self.assertEqual(bad_sanity["status"], "alert")
        self.assertTrue(bad_sanity["wt_k_order_anomaly"])
        self.assertTrue(bad_sanity["shells"][1]["light_heavy_mismatch"])
        self.assertIn("unmodeled heavy scatterer", " ".join(bad_sanity["alerts"]))

        # Case C: Panel (f) 面对漏建重原子拟合时阻断 PASS
        with tempfile.TemporaryDirectory() as td:
            texts.clear()
            with patch.object(xafs_plot_special, "select_wt_shell_targets", return_value=unmodeled_targets), \
                 patch.object(Axes, "text", new=_cap):
                xafs_plots.build_structure_main_figure(
                    res, masked_payload, Path(td), "pt_unmodeled", sample="PtO2-missing-Pt", dpi=80)
            import re
            joined_f = re.sub(r"\s+", " ", "\n".join(texts))
            self.assertNotIn("PASS — quantitative CN allowed", joined_f)
            self.assertIn("unmodeled heavy scatterer", joined_f)
            self.assertIn("[mismatch!]", joined_f)

    def test_compute_path_r_contributions_uses_res_ft_kweight(self):
        """单路径 FT 拆解与理论 WT 检查应使用与实验面板一致的 res.ft.kweight。"""
        res = _synthetic_pipeline_result()
        res.ft = SimpleNamespace(**{**res.ft.__dict__, "kweight": 2})
        seen_kw: list[int] = []

        def _fake_ft(k_arr, chi, e_arr, kmin, kmax, kw, dk):
            seen_kw.append(int(kw))
            return SimpleNamespace(r=res.ft.r, chi_r=np.zeros_like(res.ft.r, dtype=complex),
                                   magnitude=np.exp(-((res.ft.r - 1.55) / 0.25) ** 2),
                                   real=np.zeros_like(res.ft.r))

        with tempfile.TemporaryDirectory() as td:
            pfile = Path(td) / "feff0001.dat"
            pfile.write_text("dummy", encoding="utf-8")
            fit_obj = SimpleNamespace(
                paths=[SimpleNamespace(label="Pt-O@2.00", r=2.00, degeneracy=4.0,
                                       delta_r=0.0, sigma2=0.0025)],
                s02=0.80, delta_e0=1.0, kmin=3.0, kmax=11.0, kweights=(1, 2, 3),
            )
            payload = {"mode": "motif", "fit": fit_obj, "path_files": [str(pfile)]}
            with patch.object(xafs_plot_special.artemis, "load_feff_path", return_value=SimpleNamespace()), \
                 patch.object(xafs_plot_special.artemis, "set_path_parameters"), \
                 patch.object(xafs_plot_special.artemis, "exafs_model_k",
                              return_value=0.05 * np.sin(4.0 * res.chi_k.k)), \
                 patch.object(xafs_plot_special.athena, "ft_to_r_space", side_effect=_fake_ft):
                contribs = xafs_plot_special.compute_path_r_contributions(res, payload)
        self.assertEqual(seen_kw, [2])
        self.assertEqual(len(contribs), 1)
        self.assertIn("k_peak_th", contribs[0])

    def test_l1_first_shell_live_fit_and_lcf_figure(self):
        """L1 回退保留 FitResult 时应绘制拟合曲线、锁定 DIAGNOSTIC/HYPOTHESIS ONLY 并产出 LCF 图表。"""
        import csv
        import re
        from matplotlib.axes import Axes

        self.assertEqual(xafs_plot_special._scatterer_element_and_z("Pt-O", "Pt"), ("O", 8))
        self.assertEqual(xafs_plot_special._scatterer_element_and_z("Pt-Pt_1", "Pt"), ("Pt", 78))

        res = _synthetic_pipeline_result()
        res.absorber_symbol = "Pt"
        res.structure = {
            "status": "no_match",
            "ranked": [{"id": "pt_o2", "score": 0.45}],
            "coverage_ok": False,
            "l1_first_shell": {
                "ligand": "O", "geometry": "oct",
                "N": 4.2, "N_err": 0.4, "R": 2.01, "R_err": 0.02,
                "sigma2": 0.003, "rf": 0.045, "delta_e0": 3.0,
                "n_independent": 10.0, "n_variables": 4,
            },
        }
        fit_obj = SimpleNamespace(
            r_factor=0.045, delta_e0=3.0, delta_e0_err=1.1,
            n_variables=4, n_independent=10.0, s02=0.80,
            rmin=1.0, rmax=2.4, kmin=3.0, kmax=11.0, kweights=(2,),
            paths=[SimpleNamespace(
                label="Pt-O", r=2.01, reff=2.00,
                degeneracy=4.2, n_err=0.4,
                delta_r=0.01, delta_r_err=0.02,
                sigma2=0.003, sigma2_err=0.0005, sigma2_at_bound=False,
            )],
            model_k_grid=res.chi_k.k,
            model_k=0.9 * res.chi_k.chi,
            model_r=res.ft.r,
            model_chi_r=0.9 * res.ft.chi_r,
            correlation=np.eye(4),
        )
        fit_out = {
            "mode": "l1_first_shell",
            "status": "no_match",
            "cn_reliable": True,
            "s02": 0.80,
            "s02_source": "--s02 runtime override",
            "l1": res.structure["l1_first_shell"],
            "fit": fit_obj,
            "shells": [{"label": "Pt-O", "scatterer": "O"}],
        }
        res.lcf = {
            "x": np.linspace(7100.0, 7160.0, 31),
            "y": np.linspace(0.0, 1.0, 31),
            "y_fit": np.linspace(0.01, 0.99, 31),
            "weights": {"Lep": 0.6, "Pyr": 0.4},
            "weight_err": {"Lep": 0.02, "Pyr": 0.02},
            "r_factor": 0.0012,
            "energy_scale": "absolute",
            "alignment_check": {
                "recommended_shift_eV": -1.2,
                "base_r_factor": 0.01,
                "shifted_r_factor": 0.002,
            },
            "alignment_warning": "shift -1.2 eV",
        }

        texts: list[str] = []
        orig_text = Axes.text

        def _cap(self_ax, *args, **kwargs):
            if len(args) >= 3:
                texts.append(str(args[2]))
            elif "s" in kwargs:
                texts.append(str(kwargs["s"]))
            return orig_text(self_ax, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with patch.object(Axes, "text", new=_cap):
                suite = xafs_plots.render_plot_suite(
                    res, fit_out, out, "l1_lcf_test",
                    mode="structure", sample="Pt-L1", dpi=80,
                )
            joined = re.sub(r"\s+", " ", "\n".join(texts))
            self.assertIn("L1 first-shell fallback: O (oct)", joined)
            self.assertIn("Rf = 0.0450", joined)
            self.assertIn("DIAGNOSTIC/HYPOTHESIS ONLY (l1_diagnostic_only)", joined)
            self.assertNotIn("PASS — quantitative CN allowed", joined)

            lcf_png = out / "l1_lcf_test_xanes_lcf.png"
            self.assertTrue(lcf_png.is_file())
            self.assertGreater(lcf_png.stat().st_size, 1024)
            self.assertIn(lcf_png, suite["images"])

            with (out / "l1_lcf_test_summary.csv").open("r", encoding="utf-8-sig", newline="") as fh:
                items = {row[0]: row for row in csv.reader(fh) if row}
            self.assertIn("xanes_lcf", items)
            self.assertIn("energy_scale=absolute", items["xanes_lcf"][3])

            xlabels: list[str] = []
            orig_xlabel = Axes.set_xlabel

            def _cap_xl(self_ax, xlabel, *args, **kwargs):
                xlabels.append(str(xlabel))
                return orig_xlabel(self_ax, xlabel, *args, **kwargs)

            op_mock = {
                "names": ["t0", "t1", "t2"],
                "n_components": 2,
                "accepted_standards": [],
                "lcf_series": [],
                "descriptors": [{"e0": 11564.0}, {"e0": 11565.0}, {"e0": 11566.0}],
                "mcr_als": {
                    "concentrations": np.array([[0.8, 0.5, 0.2], [0.2, 0.5, 0.8]]),
                    "components": np.ones((21, 2)),
                    "x": np.linspace(11540.0, 11620.0, 21),
                },
                "mode": "xanes",
                "energy_scale": "absolute",
            }
            with patch.object(Axes, "set_xlabel", new=_cap_xl):
                xafs_plot_special.write_operando_diagnostics(op_mock, out / "op_abs")
            self.assertIn("photon energy (eV)", xlabels)

            import paper_figures

            fit_out_paper = {**fit_out, "path_files": [str(out / "feff0001.dat")]}
            seen_kwargs: list[dict] = []
            orig_f2j = paper_figures.fit_result_to_json

            def _wrap_f2j(*args, **kwargs):
                seen_kwargs.append(dict(kwargs))
                return {"fit": {"paths": []}}

            with patch.object(paper_figures, "fit_result_to_json", side_effect=_wrap_f2j), \
                 patch.object(paper_figures, "make_paper_figure",
                              return_value={"png": str(out / "mock_paper.png")}) as mpf:
                (out / "mock_paper.png").write_bytes(b"png")
                xafs_plots.render_plot_suite(
                    res, fit_out_paper, out, "l1_paper_test",
                    mode="paper", sample="Pt-L1", dpi=80,
                )
            self.assertEqual(len(seen_kwargs), 1)
            self.assertNotIn("cn_reliable", seen_kwargs[0])
            self.assertIs(seen_kwargs[0]["diagnostic_only"], True)
            self.assertEqual(seen_kwargs[0]["cn_status"], "l1_diagnostic_only")
            self.assertEqual(mpf.call_args.kwargs["kmax"], 12.0)


if __name__ == "__main__":
    unittest.main()
