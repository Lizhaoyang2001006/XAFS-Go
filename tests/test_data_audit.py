# -*- coding: utf-8 -*-
"""tests/test_data_audit.py — XAFS 数据预检、单扫噪声、Rbkg 入侵与 LCF 对齐自审计单测。"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
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

_HAS_LARCH = importlib.util.find_spec("larch") is not None

from xafs_core import ChiK, NormalizedScan, RawScan  # noqa: E402
import data_audit  # noqa: E402
import xafs_lcf  # noqa: E402


class DataAuditTests(unittest.TestCase):
    def test_single_scan_kmax_estimator_separates_clean_and_noisy(self):
        import athena

        k = np.arange(0.0, 14.01, 0.05)
        chi_clean = (0.18 / np.maximum(k, 1.5)) * np.sin(2.0 * 2.0 * k + 0.1) * np.exp(-0.012 * k ** 2)
        ck_clean = ChiK(k=k, chi=chi_clean, background=np.zeros_like(k),
                        mu_norm=np.ones_like(k), rbkg=1.0)
        # high_k_snr(sigma_mu=None) 契约严禁改动
        self.assertIsNone(athena.high_k_snr(ck_clean)["k_snr2"])
        est_clean = athena.estimate_single_scan_kmax(ck_clean, edge_step=1.0, kweight=2)
        self.assertIsNone(est_clean["k_snr2"])

        rng = np.random.default_rng(7)
        chi_noisy = chi_clean + rng.normal(0.0, 0.0022, size=k.size)
        ck_noisy = ChiK(k=k, chi=chi_noisy, background=np.zeros_like(k),
                        mu_norm=np.ones_like(k), rbkg=1.0)
        est_noisy = athena.estimate_single_scan_kmax(ck_noisy, edge_step=1.0, kweight=2)
        self.assertIsNotNone(est_noisy["k_snr2"])
        self.assertLess(float(est_noisy["k_snr2"]), 12.0)

    def test_audit_preprocessing_catches_all_silent_failure_classes(self):
        e = np.linspace(11400.0, 12300.0, 451)
        # 制造 2 个卡滞能量微步
        e[50] = e[49] + 0.005
        e[51] = e[50] + 0.005
        sc = RawScan(energy=e, mu=np.ones_like(e),
                     meta={"label": "Pt_std", "eshift": -1.5, "eshift_applied": False})
        norm_bad = NormalizedScan(energy=e, mu=np.ones_like(e), mu_norm=np.ones_like(e),
                                  pre_edge_line=np.zeros_like(e), edge_step=-0.35,
                                  plateau=1.12, e0=11564.0)
        fake_res = SimpleNamespace(
            scans=[sc],
            merged=sc,
            norm=norm_bad,
            rbkg={
                "rbkg": 1.30, "r1": 1.60, "intrusion_risk": True,
                "safe_rbkg_hint": 1.00,
                "intrusion_warning": "Rbkg=1.30 Å 已逼近第一壳 r1=1.60 Å",
            },
            ft=SimpleNamespace(kmax=12.0),
            snr={"k_snr2": None, "single_scan_estimate": {"k_snr2": 10.5}},
        )
        rep = data_audit.audit_preprocessing(fake_res)
        self.assertEqual(rep["status"], "fail")
        ids = {c["id"] for c in rep["checks"]}
        self.assertIn("prj_eshift_unapplied", ids)
        self.assertIn("step_polarity", ids)
        self.assertIn("normalization_plateau_edge", ids)
        self.assertIn("energy_tiny_steps", ids)
        self.assertIn("rbkg_first_shell_intrusion", ids)
        self.assertIn("high_k_noise_cutoff", ids)
        self.assertAlmostEqual(rep["recommended_kmax"], 10.5, places=2)
        self.assertAlmostEqual(rep["safe_rbkg_hint"], 1.00, places=2)

    def test_lcf_detects_global_energy_misalignment(self):
        e = np.linspace(11540.0, 11620.0, 321)
        std1 = 0.5 * (1.0 + np.tanh((e - 11564.0) / 1.8)) + 0.3 * np.exp(-((e - 11566.5) / 2.5) ** 2)
        std2 = 0.5 * (1.0 + np.tanh((e - 11566.5) / 1.8)) + 1.4 * np.exp(-((e - 11568.0) / 2.2) ** 2)
        # 目标谱已校准，但传入的标准谱整体偏高 +1.5 eV（需 -1.5 eV 偏移还原）
        target = 0.25 * std1 + 0.75 * std2
        std1_unshifted = np.interp(e - 1.5, e, std1)
        std2_unshifted = np.interp(e - 1.5, e, std2)

        res_mis = xafs_lcf.run_lcf(
            e, target, {"Pt_foil": (e, std1_unshifted), "PtO2": (e, std2_unshifted)},
            xmin=11545.0, xmax=11615.0,
        )
        self.assertIn("alignment_warning", res_mis)
        self.assertAlmostEqual(res_mis["alignment_check"]["recommended_shift_eV"], -1.5, delta=0.6)

        res_ok = xafs_lcf.run_lcf(
            e, target, {"Pt_foil": (e, std1), "PtO2": (e, std2)},
            xmin=11545.0, xmax=11615.0,
        )
        self.assertNotIn("alignment_warning", res_ok)

    @unittest.skipUnless(_HAS_LARCH, "optimize_rbkg requires real larch")
    def test_optimize_rbkg_flags_first_shell_intrusion_without_drifting_lepidocrocite(self):
        import athena
        import xafs_io

        lep = xafs_io.read_scan(ROOT / "data" / "练手数据" / "02_Fe_standards" / "Fe_lepidocrocite.000")
        e0 = athena.determine_e0(lep, method="derivative", guess=7112.0)["e0_sample"]
        nm = athena.normalize_absorption(lep, e0)
        best, _ = athena.optimize_rbkg(nm, kmin=3.0, kmax=12.0, kweight=2)
        self.assertAlmostEqual(best["rbkg"], 1.20, places=2)
        self.assertFalse(best.get("intrusion_risk", False))


if __name__ == "__main__":
    unittest.main()
