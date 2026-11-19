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

    def test_lcf_absolute_energy_scale_and_coherent_tail_audit(self):
        e = np.linspace(11540.0, 11620.0, 321)
        shape = lambda e0: (
            0.5 * (1.0 + np.tanh((e - e0) / 1.8))
            + 0.45 * np.exp(-((e - (e0 + 2.5)) / 2.2) ** 2)
        )
        y0 = shape(11564.0)
        y4 = shape(11568.0)
        zeros = np.zeros_like(e)
        s0 = NormalizedScan(energy=e, mu=y0, mu_norm=y0, pre_edge_line=zeros,
                            edge_step=1.0, plateau=1.0, e0=11564.0, flattened=y0)
        s4 = NormalizedScan(energy=e, mu=y4, mu_norm=y4, pre_edge_line=zeros,
                            edge_step=1.0, plateau=1.0, e0=11568.0, flattened=y4)
        ymix = 0.30 * y0 + 0.70 * y4
        targ = NormalizedScan(energy=e, mu=ymix, mu_norm=ymix, pre_edge_line=zeros,
                              edge_step=1.0, plateau=1.0, e0=11567.2, flattened=ymix)

        res_abs = xafs_lcf.lcf_from_scans(
            targ, {"S0": s0, "S4": s4}, mode="xanes",
            xmin=11545.0, xmax=11615.0, energy_scale="absolute",
        )
        self.assertEqual(res_abs["energy_scale"], "absolute")
        self.assertAlmostEqual(res_abs["weights"]["S0"], 0.30, delta=0.02)
        self.assertAlmostEqual(res_abs["weights"]["S4"], 0.70, delta=0.02)
        self.assertLess(res_abs["r_factor"], 1e-6)

        # 验证 lcf_summary 保留对齐告警与刻度
        summ = xafs_lcf.lcf_summary({
            **res_abs,
            "alignment_check": {"recommended_shift_eV": -1.5,
                                "base_r_factor": 0.05, "shifted_r_factor": 0.01},
            "alignment_warning": "shift -1.5 eV",
        })
        self.assertEqual(summ["energy_scale"], "absolute")
        self.assertEqual(summ["alignment_warning"], "shift -1.5 eV")
        self.assertAlmostEqual(summ["alignment_check"]["recommended_shift_eV"], -1.5)

        # 原位 NormalizedScan 序列默认自动选 absolute，保留化学态边移轨迹
        fracs = (0.1, 0.35, 0.65, 0.9)
        series = []
        for idx, f4 in enumerate(fracs):
            yi = (1.0 - f4) * y0 + f4 * y4
            sc = NormalizedScan(energy=e, mu=yi, mu_norm=yi, pre_edge_line=zeros,
                                edge_step=1.0, plateau=1.0,
                                e0=11564.0 + 4.0 * f4, flattened=yi)
            series.append((f"t{idx}", sc))
        with patch.object(xafs_lcf.athena, "extract_xanes_features",
                          return_value={"e_half_height": 11565.0, "white_line_height": 1.2}):
            op = xafs_lcf.run_operando_series(series, standards={"S0": s0, "S4": s4})
        self.assertEqual(op["energy_scale"], "absolute")
        w4 = [pt["weights"]["S4"] for pt in op["lcf_series"]]
        self.assertTrue(all(w4[i] < w4[i + 1] for i in range(len(w4) - 1)))

        # 重原子金属相干高 k 振幅（tail_hf_ratio 低 + reliable_kmax 高）不得误报噪声
        res_pt_foil = SimpleNamespace(
            merged=SimpleNamespace(meta={}),
            norm=SimpleNamespace(e0=11564.0, edge_step=1.2, plateau=1.0,
                                 energy=np.linspace(11500.0, 12500.0, 100)),
            rbkg={"rbkg": 0.90, "r1": 2.48, "intrusion_risk": False},
            ft=SimpleNamespace(kmax=12.0),
            snr={"k_snr2": None, "single_scan_estimate": {
                "k_snr2": None, "reliable_kmax": 16.0,
                "tail_rise_ratio": 1.40, "tail_hf_ratio": 0.06,
            }},
        )
        rep_foil = data_audit.audit_preprocessing(res_pt_foil)
        self.assertNotIn("high_k_tail_amplification", {c["id"] for c in rep_foil["checks"]})

        res_noisy_tail = SimpleNamespace(
            merged=SimpleNamespace(meta={}),
            norm=SimpleNamespace(e0=11564.0, edge_step=1.2, plateau=1.0,
                                 energy=np.linspace(11500.0, 12500.0, 100)),
            rbkg={"rbkg": 0.90, "r1": 1.60, "intrusion_risk": False},
            ft=SimpleNamespace(kmax=12.0),
            snr={"k_snr2": None, "single_scan_estimate": {
                "k_snr2": None, "reliable_kmax": 11.0,
                "tail_rise_ratio": 1.20, "tail_hf_ratio": 0.35,
            }},
        )
        rep_noisy = data_audit.audit_preprocessing(res_noisy_tail)
        self.assertIn("high_k_tail_amplification", {c["id"] for c in rep_noisy["checks"]})

    @unittest.skipUnless(_HAS_LARCH, "process_xanes_scan uses real larch pre_edge")
    def test_process_xanes_scan_handles_short_xanes_only_standards(self):
        import xafs_pipeline

        e = np.linspace(7070.0, 7220.0, 151)
        y_low = 0.5 * (1.0 + np.tanh((e - 7112.0) / 2.2)) + 0.30 * np.exp(-((e - 7115.0) / 3.0) ** 2)
        y_hi = 0.5 * (1.0 + np.tanh((e - 7116.0) / 2.2)) + 0.55 * np.exp(-((e - 7119.0) / 3.0) ** 2)
        y_mix = 0.40 * y_low + 0.60 * y_hi
        zeros = np.zeros_like(e)
        targ = NormalizedScan(energy=e, mu=y_mix, mu_norm=y_mix, pre_edge_line=zeros,
                              edge_step=1.0, plateau=1.0, e0=7114.8, flattened=y_mix)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "std_low.dat"
            p2 = Path(tmp) / "std_hi.dat"
            p1.write_text("# E mu\n" + "\n".join(f"{a:.2f} {0.2 + 0.8 * b:.6f}" for a, b in zip(e, y_low)), encoding="utf-8")
            p2.write_text("# E mu\n" + "\n".join(f"{a:.2f} {0.1 + 0.9 * b:.6f}" for a, b in zip(e, y_hi)), encoding="utf-8")
            n1 = xafs_pipeline.process_xanes_scan([str(p1)], absorber="Fe")
            n2 = xafs_pipeline.process_xanes_scan([str(p2)], edge_energy=7114.8)
        self.assertAlmostEqual(n1.edge_step, 0.8, delta=0.05)
        self.assertAlmostEqual(n2.edge_step, 0.9, delta=0.05)
        lcf = xafs_lcf.lcf_from_scans(
            targ, {"low": n1, "hi": n2}, mode="xanes",
            xmin=7095.0, xmax=7170.0, energy_scale="absolute",
        )
        self.assertAlmostEqual(lcf["weights"]["low"], 0.40, delta=0.03)
        self.assertAlmostEqual(lcf["weights"]["hi"], 0.60, delta=0.03)


if __name__ == "__main__":
    unittest.main()
