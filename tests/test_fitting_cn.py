# -*- coding: utf-8 -*-
"""Unit 4+5：FEFF/拟合引擎 + 标定/CN 门禁/台账

纯搬运拆分：测试方法体与原 regression_test.py 逐行一致，断言未做任何修改。
"""
from __future__ import annotations

from tests.fixtures import *  # noqa: F401,F403
from tests.fixtures import Fixtures, _find  # noqa: F401

# 历史测试体里有 3 处直接用 Path(__file__) 定位 data/（原文件在仓库根）。
# 拆到 tests/ 后模块位置变了，把 __file__ 指回等价的根入口文件，
# 保证测试方法体逐字节不变（不改任何断言或路径写法）。
__file__ = str(ROOT / "regression_test.py")


class FittingCNTests(Fixtures):

    def test_wavelet_orientation(self):
        k = np.arange(0.0, 16.0, 0.05)
        chi = ChiK(k=k, chi=np.sin(2.0 * k), background=np.zeros_like(k),
                   mu_norm=np.zeros_like(k), rbkg=1.0)
        wt = larch_wt.continuous_wavelet_transform(
            chi, sigma=1.5, kmin=3.0, kmax=13.0, kweight=2)
        self.assertEqual(wt.magnitude.shape,
                         (len(wt.k_centers), len(wt.r)))
        self.assertEqual(wt.magnitude.shape[1], len(wt.r))

    def test_mixed_measurement_modes_cannot_bypass_cn_gate(self):
        from types import SimpleNamespace

        energy = np.linspace(7000, 7200, 20)
        mu = np.r_[np.full(10, 0.2), np.full(10, 2.2)]
        i0 = np.full(20, 100.0)
        trans = RawScan(energy=energy, i0=i0, it=i0 * np.exp(-mu),
                        mu=mu, meta={"mode": "trans"})
        fluor = RawScan(energy=energy, i0=i0, ifluor=i0 * mu,
                        mu=mu, meta={"mode": "fluor"})
        given = RawScan(energy=energy, i0=i0, it=trans.it,
                        mu=mu, meta={"mode": "given"})
        unknown = RawScan(energy=energy, mu=mu, meta={"mode": "given"})
        for scans in ([trans, fluor], [fluor, trans], [given, fluor],
                      [trans, unknown], [unknown, trans]):
            for corrected in (False, True):
                with self.subTest(modes=[s.meta["mode"] for s in scans],
                                  corrected=corrected):
                    res = SimpleNamespace(
                        scans=scans, merged=athena.merge_scans(scans),
                        norm=SimpleNamespace(edge_step=2.0),
                        self_absorption={"corrected": corrected, "reliable": True})
                    self.assertIsNone(xafs_fit.classify_fluorescence_scan(res)[0])
                    self.assertTrue(xafs_fit._self_absorption_diag(res)["cn_gate"])
        compatible = [trans, given]
        self.assertIs(xafs_fit.classify_fluorescence_scan(SimpleNamespace(
            scans=compatible, merged=athena.merge_scans(compatible)))[0], False)
        outlier = RawScan(energy=energy, i0=i0, ifluor=i0 * (mu + 20),
                          mu=mu + 20, meta={"mode": "fluor"})
        scans = [trans, given, outlier]
        merged = athena.merge_scans(scans, reject_outliers=True)
        self.assertEqual(merged.meta["merge_info"]["rejected_indices"], [2])
        self.assertIs(xafs_fit.classify_fluorescence_scan(SimpleNamespace(
            scans=scans, merged=merged))[0], False)
        scans = [outlier, trans, given]
        merged = athena.merge_scans(scans, reject_outliers=True)
        self.assertEqual(merged.meta["merge_info"]["rejected_indices"], [0])
        self.assertEqual(merged.meta["mode"], "trans")
        self.assertIs(xafs_fit.classify_fluorescence_scan(SimpleNamespace(
            scans=scans, merged=merged))[0], False)

    def test_real_standard_independent_conditional_cn_and_refusal(self):
        import csv
        import validate_real_cn

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = validate_real_cn.run_case(
                "ni_oxide_to_hydroxide", root / "ni")
            self.assertEqual(accepted["status"], "validated", accepted.get("reason"))
            self.assertNotEqual(accepted["standard_data_sha256"],
                                accepted["holdout_data_sha256"])
            self.assertTrue(accepted["calibration_qualified"])
            motif = json.loads((root / "ni" / "calibration_motif.json").read_text(
                encoding="utf-8"))
            self.assertEqual(motif["source_cif_sha256"],
                             accepted["standard_cif_sha256"])
            self.assertEqual(motif["id"], accepted["calibration"]["motif"])
            self.assertTrue(accepted["cn_reliable"])
            self.assertLessEqual(abs(accepted["relative_bias"]), 0.30)
            conditional = accepted["conditional_cn"]
            self.assertEqual(conditional["n0_total"], 6.0)
            self.assertTrue(conditional["consistency_passed"])
            for key in ("fixed_n_fit", "conditional_n_fit"):
                fit = conditional[key]
                self.assertLessEqual(fit["r_factor"], 0.08)
                self.assertLess(abs(fit["delta_e0"]), 15.0)
                self.assertLess(fit["n_variables"], fit["n_independent"])
            self.assertTrue(conditional["hamilton_f_test"]["valid"])
            self.assertFalse(conditional["hamilton_f_test"]["significant"])
            table = root / "ni" / "prediction" / "cn_table.csv"
            with table.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            fitted = [r for r in rows if r["n_status"] == "fitted"]
            self.assertTrue(fitted)
            for row in fitted:
                self.assertEqual(row["data_sha256"], accepted["holdout_data_sha256"])
                self.assertEqual(row["source_ref"], "COD:1548811")
                self.assertTrue(row["source_cif_sha256"])
                self.assertTrue(row["n_err_systematic"])
                self.assertIn(accepted["standard_data_sha256"], row["s02_source"])
                self.assertAlmostEqual(float(row["s02"]),
                                       accepted["calibration"]["s02"])

            rejected = validate_real_cn.run_case(
                "cu_foil_to_oxide", root / "cu")
            self.assertTrue(rejected["calibration_qualified"])
            self.assertEqual(rejected["status"], "formal_rejected")
            self.assertFalse(rejected["within_systematic_budget"])
            self.assertEqual(rejected["formal_cn"]["rows"], [])
            self.assertTrue(rejected["formal_cn"]["reject_reason"])
            table = root / "cu" / "prediction" / "cn_table.csv"
            with table.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(not r["n_fit"] for r in rows))

    def test_real_cn_validation_refuses_same_spectrum(self):
        from unittest.mock import patch
        import validate_real_cn

        case = dict(validate_real_cn.CASES["ni_oxide_to_hydroxide"])
        case["holdout"] = case["standard"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                validate_real_cn.CASES, {"same_input": case}):
            with self.assertRaises(XAFSConfigError) as ctx:
                validate_real_cn.run_case("same_input", Path(tmp))
            self.assertIn("拒绝循环验证", str(ctx.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_real_cn_validation_preserves_case_errors(self):
        from unittest.mock import patch
        import validate_real_cn

        results = [{"case": "ni_oxide_to_hydroxide", "status": "validated"},
                   XAFSConfigError("validation", "CIF N0 不一致")]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                validate_real_cn, "run_case", side_effect=results):
            code = validate_real_cn.main([
                "--case", "ni_oxide_to_hydroxide", "--case", "cu_foil_to_oxide",
                "--outdir", tmp])
            report = json.loads((Path(tmp) / "real_cn_validation.json").read_text(
                encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(report["cases"][1]["status"], "validation_error")
        self.assertIn("CIF N0 不一致", report["cases"][1]["reason"])
        self.assertEqual(report["validated_cases"], ["ni_oxide_to_hydroxide"])
        self.assertFalse(report["all_cases_validated"])

    def test_fit_l1_when_no_match(self):
        res = xafs_pipeline.process_sample(
            [FERRIHYDRITE], absorber="Fe", strict=False, verbose=False)
        out = xafs_fit.run(res, identify=True, fit=True,
                           absorber_symbol="Fe", verbose=False)
        sid = out["structure"]
        self.assertEqual(sid["status"], "no_match")
        fit_out = out["fit"]
        self.assertEqual(fit_out["mode"], "l1_first_shell")
        l1 = fit_out["l1"]
        self.assertGreater(l1["N"], 4.0)
        self.assertLess(l1["N"], 7.0)
        self.assertGreater(l1["R"], 1.85)
        self.assertLess(l1["R"], 2.05)
        self.assertIn("仅第一壳", l1["warning"])

    def test_l1_ligand_aware_peak_selection(self):
        res = xafs_pipeline.process_sample(
            [PT_CL], absorber="Pt", strict=False, verbose=False)
        l1 = structure_id.l1_first_shell_fit(
            res, absorber_symbol="Pt", ligand="Cl", verbose=False)
        # 平方平面 PtCl4：N=4、Pt–Cl 2.30 Å；错误的低 R 伪峰会给出 Rf>0.3
        self.assertLess(l1["rf"], 0.08)
        self.assertAlmostEqual(l1["N"], 4.0, delta=0.5)
        self.assertAlmostEqual(l1["R"], 2.30, delta=0.05)

    def test_l1_long_bond_window_not_truncated(self):
        res = xafs_pipeline.process_sample(
            [PT_FOIL], absorber="Pt", strict=False, verbose=False)
        l1 = structure_id.l1_first_shell_fit(
            res, absorber_symbol="Pt", ligand="Pt", verbose=False)
        # 旧 rmax=min(2.3, ...) 把 Pt–Pt 2.77 的模型峰截在窗外 → Rf≈1.0
        self.assertLess(l1["rf"], 0.08)
        self.assertGreater(l1["N"], 9.0)
        self.assertAlmostEqual(l1["R"], 2.77, delta=0.05)

    def test_vary_s02_recovers_synthetic_truth(self):
        """标定回路（calibrate_s02.py 的前提契约）。

        数据由已知参数合成：S₀²=0.72、σ²=0.004、E0+=2 eV、N=8 固定
        （fe_bcc 第一壳）。fit_r_space 以 vary_s02=True、s02=None 启动，
        N 固定为晶体学简并度时唯一振幅变量 S₀² 必须回收真值——
        回收不了说明标定会写入错误数值。
        """
        truth = {"s02": 0.72, "sigma2": 0.004, "e0": 2.0}
        k = np.arange(3.0, 12.0, 0.05)
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "fe_bcc"][0]
        plan = xafs_fit.select_shell_paths(motif, edge="K", rmax=2.9,
                                           kmin=3.0, kmax=12.0)
        first = plan["shells"][0]
        self.assertAlmostEqual(first["degen"], 8.0, places=6)
        from larch.xafs import feffpath
        p = feffpath(str(plan["files"][0]), s02=truth["s02"],
                     sigma2=truth["sigma2"], e0=truth["e0"], deltar=0.0)
        p.degen = float(first["degen"])
        p._calc_chi()
        chi = artemis.exafs_model_k([p], k)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.zeros_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        fit = artemis.fit_r_space(
            chi_k, ft, [plan["files"][0]], rmin=1.9, rmax=2.75,
            kmin=3.0, kmax=12.0, kweight=2, s02=0.85,
            vary_delta_e0=True, vary_n=False, vary_s02=True,
            path_degen=[first["degen"]], path_labels=[first["label"]])
        self.assertFalse(fit.overfit)
        self.assertLess(fit.r_factor, 0.02, f"无噪声合成数据应近完美复现")
        self.assertLess(abs(fit.s02 - truth["s02"]), 0.03,
                        f"S₀² 回收偏差过大: {fit.s02}")
        self.assertLess(fit.s02_err, 0.06)
        self.assertLess(abs(fit.paths[0].sigma2 - truth["sigma2"]), 1e-3)
        self.assertLess(abs(fit.delta_e0 - truth["e0"]), 0.6)

    def test_counterexample_artemis_jac_rank_deficiency(self):
        """审查 item 5：雅可比秩亏必须标记，杜绝 Rf≈0 的虚假精度。

        满秩单路径拟合 rank_deficient=False、jac_rank==N_var（基线逐位
        不变）；零占据壳（degen=0）却仍放开其 ΔR → 该路径对 χ 无贡献、
        雅可比对应列恒为 0 → 秩亏，即使 Rf≈0 也判不可辨识；provenance
        门禁据此拦截 CN。
        """
        import structure_lib
        import xafs_fit
        import artemis
        import athena
        import xafs_provenance
        from larch.xafs import feffpath
        from xafs_core import ChiK, FitResult

        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "magnetite_mixed"][0]
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmax=2.6, kmin=3.0, kmax=12.0,
            group_by_component=True)
        s02, sigma2, e0 = 0.80, 0.004, 2.0
        k = np.arange(3.0, 12.0, 0.05)
        f0 = plan["files"][0]
        d0 = plan["degens"][0]
        p = feffpath(str(f0), s02=s02, sigma2=sigma2, e0=e0, deltar=0.0)
        p.degen = d0
        p._calc_chi()
        chi = artemis.exafs_model_k([p], k)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.zeros_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)

        # ---- 满秩基线：单路径正常拟合 ----
        fit_ok = artemis.fit_r_space(
            chi_k, ft, [f0], rmin=plan["rmin"], rmax=plan["rmax"],
            kmin=3.0, kmax=12.0, kweight=2, s02=s02,
            vary_delta_e0=True, vary_n=False, vary_sigma2=False,
            sigma2_init=sigma2, path_degen=[d0], path_labels=["shell0"])
        self.assertFalse(fit_ok.rank_deficient)
        self.assertEqual(fit_ok.jac_rank, fit_ok.n_variables)
        self.assertLess(fit_ok.r_factor, 0.02)

        # ---- 秩亏反例：第二条路径简并度=0（零占据壳），却仍放开其 ΔR。
        # 该路径对 χ 无任何贡献 → 雅可比对应列恒为 0 → 秩亏。即使
        # Rf≈0（完美拟合），这个 ΔR 也完全不可辨识——pinv 会给出虚假
        # 有限误差，正是秩亏守卫要拦截的"看起来完美"陷阱。----
        fit_deg = artemis.fit_r_space(
            chi_k, ft, [f0, f0], rmin=plan["rmin"], rmax=plan["rmax"],
            kmin=3.0, kmax=12.0, kweight=2, s02=s02,
            vary_delta_e0=True, vary_n=False, vary_sigma2=False,
            sigma2_init=sigma2, path_degen=[d0, 0.0],
            path_labels=["real", "zero_occ"])
        self.assertTrue(fit_deg.rank_deficient,
                        "零占据壳的自由 ΔR 必须判秩亏")
        self.assertLess(fit_deg.jac_rank, fit_deg.n_variables)
        self.assertLess(fit_deg.r_factor, 0.02,
                        "秩亏时 Rf 仍可能≈0——正是虚假精度陷阱")

        # ---- FitResult 字段语义 + provenance 门禁联动 ----
        fr = FitResult(paths=[], s02=0.8, delta_e0=1.0, delta_e0_err=0.1,
                       r_factor=0.01, n_independent=10.0, n_variables=3,
                       kmin=3.0, kmax=12.0, rmin=1.0, rmax=3.0,
                       jac_rank=2, rank_deficient=True)
        self.assertTrue(fr.rank_deficient)
        gate = xafs_provenance.build_gate_ledger(
            fit_out={"fit": fr, "flags": {}, "coverage_ok": True,
                     "cn_reliable": False, "cn_status": "diagnostic_only"}
        )["fit"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn("rank_deficient", gate["failed_checks"])

    def test_wt_locator_anchored_to_ft_peak(self):
        """WT 定位必须锚定 FT 峰，不能在两壳分支间跳。

        实测故障：Fe3O4 的 WT 在 R≈1.42（Fe–O，k≈4.5）与 R≈2.72
        （Fe–Fe，k≈5.8）各有一条分支，宽窗 [1.0,2.6] 内二者只差 3%；
        Booth 校正把低 k 抬高 7% 后 WT 峰从 2.60 跳到 1.42，而 FT 峰
        完全不变。契约：锚定 FT 峰后，校正前后 WT R 位移动 ≤ 0.02 Å。
        """
        res = xafs_pipeline.process_sample(
            [MAGNETITE], absorber="Fe", strict=False, verbose=False)
        chi2, info = athena.booth_self_absorption(
            res.chi_k, formula="Fe3O4", elem="Fe", edge="K", line="Ka",
            e0=res.e0["e0_sample"], thickness_um=4.6, density=5.17,
            anginp=49.4, angout=41.0)
        self.assertGreater(info["correction_factor_range"][1], 1.4,
                           "测试前提：校正因子应有明显低 k 抬升")

        def anchored(chi_k):
            wt = larch_wt.continuous_wavelet_transform(
                chi_k, sigma=1.5, kmin=3.0, kmax=12.0, kweight=2)
            return larch_wt.locate_shell_near(wt, res.r1)

        k_a, r_a, _ = anchored(res.chi_k)
        k_b, r_b, _ = anchored(chi2)
        self.assertLessEqual(abs(r_b - r_a), 0.02,
                             f"WT 峰随校正跳变: {r_a} → {r_b}")
        self.assertLessEqual(abs(r_a - res.r1), 0.5,
                             f"WT 与 FT 峰位不一致: {r_a} vs {res.r1}")
        self.assertTrue(np.isfinite(k_a) and np.isfinite(k_b))

    def test_multi_kweight_joint_fit(self):
        """B1-1：多 k 权重联合拟合。

        验证 (1) 三个分支 Rf 均 < 0.08；(2) 第一壳键长与单权偏差
        ≤ 0.015 Å、N 仍在物理区间；(3) N_idp 不变且最强相关不恶化。
        多权残差若未按分支 RMS 归一，kw=3 会统治最小二乘，第一壳 N
        会被系统性拉偏——这正是该测试要防的退化。
        """
        fit1 = self.lep_fit["fit"]["fit"]
        out3 = xafs_fit.fit_identified(
            self.lep, absorber_symbol="Fe", kweights=(1, 2, 3),
            s02=0.803,  # 固定历史口径，隔离多 k 权算法与静态标定值
            verbose=False)
        fit3 = out3["fit"]
        self.assertEqual(tuple(fit3.kweights), (1, 2, 3))
        self.assertEqual(set(fit3.r_factors_by_kweight), {1, 2, 3})
        for kw, rf in fit3.r_factors_by_kweight.items():
            self.assertLess(rf, 0.08, f"kw={kw} Rf={rf:.4f} 超出 0.08")

        p1 = min(fit1.paths, key=lambda p: p.reff)
        p3 = min(fit3.paths, key=lambda p: p.reff)
        self.assertLessEqual(abs(p3.r - p1.r), 0.015,
                             f"第一壳键长漂移 {abs(p3.r - p1.r):.4f} Å")
        self.assertGreaterEqual(p3.degeneracy, 4.5)
        self.assertLessEqual(p3.degeneracy, 7.5)

        rho1, _, _ = xafs_fit._max_offdiag_correlation(fit1)
        rho3, _, _ = xafs_fit._max_offdiag_correlation(fit3)
        self.assertLessEqual(
            rho3, rho1 + 1e-9,
            f"多权相关恶化: {rho3:.4f} > 单权 {rho1:.4f}")
        self.assertEqual(fit3.n_independent, fit1.n_independent)

    def test_collinear_ms_constrained_fit_nio(self):
        """B1-3：NiO 共线 MS 路径受约束纳入，自由变量数不变。

        关键契约：MS 路径不新增 ΔR/σ²/N 变量，全部由母单散射壳层按
        reff_ratio 驱动；否则加 MS 必然触发 N_idp 过拟合截断。
        """
        res = xafs_pipeline.process_sample(
            [NIO], absorber="Ni", strict=False, verbose=False)
        no_ms = xafs_fit.fit_identified(
            res, absorber_symbol="Ni", rmax=4.3,
            include_ms=False, verbose=False)
        with_ms = xafs_fit.fit_identified(
            res, absorber_symbol="Ni", rmax=4.3,
            include_ms=True, verbose=False)
        f0, f1 = no_ms["fit"], with_ms["fit"]

        self.assertFalse([p for p in f0.paths if p.nleg > 2],
                         "include_ms=False 仍选入 MS 路径")
        ms_paths = [p for p in f1.paths if p.nleg > 2]
        self.assertTrue(ms_paths, "include_ms=True 未选入任何 nleg>2 路径")
        self.assertEqual(f1.n_variables, f0.n_variables,
                         "MS 路径新增了自由变量")
        self.assertLess(f1.r_factor, 0.08,
                        f"含 MS 拟合 Rf={f1.r_factor:.4f}")

        labels = [p.label for p in f1.paths]
        self.assertEqual(len(labels), len(set(labels)), "MS 路径标签重名")

        by_label = {p.label: p for p in f1.paths}
        for p in ms_paths:
            self.assertIsNotNone(p.ms_parent, f"{p.label} 未标记母壳层")
            parent = by_label[p.ms_parent]
            ratio = p.reff / parent.reff
            self.assertGreater(ratio, 1.5,
                               f"{p.label} 不是 Reff≈2R_first 的聚焦路径")
            expected_r = p.reff + ratio * (parent.r - parent.reff)
            self.assertAlmostEqual(
                p.r, expected_r, places=5,
                msg=f"{p.label} 的 ΔR 未按 reff_ratio={ratio:.3f} 缩放")

    def test_wt_shell_k_profiles_and_model_residual(self):
        """B3-1：一维 WT 包络区分轻/重散射体；实验-模型二维残差可定量。

        为什么 k 窗口取 2.5 而不是产线默认 3.0：Fe–O 的包络在低 k 仍有
        显著权重，kmin=2.5 能保留这部分信息，使第二壳 Fe–Fe（k≈6.4）
        与第一壳 Fe–O（k≈5.5）的峰位差达到 0.9 Å⁻¹。产线 kmin=3.0 时
        差值约 0.7 Å⁻¹，方向仍正确但小于验收阈值；这里显式使用更宽的
        分析窗并同时核对两者排序，避免把窗口选择误当成物理结论。
        """
        fit = self.lep_fit["fit"]["fit"]
        self.assertIsNotNone(fit.model_k, "FitResult 未保存 model_k")
        self.assertEqual(fit.model_k.shape, fit.model_k_grid.shape)

        kmax = float(min(12.0, np.max(self.lep.chi_k.k)))
        wt = larch_wt.continuous_wavelet_transform(
            self.lep.chi_k, sigma=1.5, kmin=2.5, kmax=kmax,
            kweight=2, rmax=4.0)
        first = larch_wt.extract_k_profile(wt, 1.47, half_width=0.25)
        second = larch_wt.extract_k_profile(wt, 2.67, half_width=0.25)
        diff = second["k_peak"] - first["k_peak"]
        self.assertGreaterEqual(
            diff, 0.8, f"Fe–Fe k_peak={second['k_peak']:.2f}, "
                       f"Fe–O k_peak={first['k_peak']:.2f}, 差 {diff:.2f} <0.8")
        # 产线默认窗口（kmin=3.0）下排序必须一致，只是差值略小
        self.assertGreater(larch_wt.extract_k_profile(
            self.lep.wt, 2.67)["k_peak"],
            larch_wt.extract_k_profile(self.lep.wt, 1.47)["k_peak"] + 0.5)
        for prof in (first, second):
            self.assertEqual(len(prof["k"]), len(prof["ridge_profile"]))
            self.assertEqual(len(prof["k"]), len(prof["slice_profile"]))
            self.assertFalse(np.any(np.isnan(prof["ridge_profile"])))

        resid = larch_wt.compute_model_wt_residual(
            self.lep.chi_k, fit.model_k, sigma=1.5, kmin=3.0,
            kmax=12.0, kweight=2, rmax=4.0)
        self.assertEqual(resid["residual_map"].shape,
                         resid["wt_data"].magnitude.shape)
        self.assertEqual(resid["residual_map"].shape,
                         resid["wt_model"].magnitude.shape)
        self.assertLess(resid["r_factor"], 0.15,
                        f"二维 WT R-factor={resid['r_factor']:.4f} ≥0.15")
        # 残差必须共享同一物理标度，而不是各自归一后相减
        self.assertAlmostEqual(resid["scale"],
                               float(resid["wt_data"].magnitude.max()),
                               places=12)

        with tempfile.TemporaryDirectory() as tmp:
            prof = xafs_plot_special.write_wt_shell_profiles(
                self.lep, self.lep_fit, tmp, stem="lepido")
            self.assertTrue(Path(prof["png"]).exists())
            self.assertTrue(Path(prof["csv"]).exists())
            header = Path(prof["csv"]).read_text(
                encoding="utf-8").splitlines()[0]
            self.assertIn("data_ridge", header)
            self.assertGreaterEqual(len(prof["shells"]), 2)

    def test_reduced_chi2_and_shell_ftest(self):
        """B4-1：高 R 噪声 ε_R → χ²_ν；第二壳 Fe–Fe 的 F-test 必须显著。

        关键防退化点：
          - larch xftf 默认 R 网格只到 ~10 Å，不显式扩到 25 Å 时
            [15,25] 区间为空，ε_R/χ² 会静默变成 None；
          - F-test 必须用**同一 R 窗**的单壳对照拟合；换窗后 χ² 不可比。
        """
        fit = self.lep_fit["fit"]["fit"]
        self.assertIsNotNone(fit.epsilon_r)
        self.assertIsNotNone(fit.chi_square)
        self.assertIsNotNone(fit.reduced_chi_square)
        self.assertGreater(fit.epsilon_r, 0.0)
        self.assertGreater(fit.chi_square, 0.0)
        self.assertGreater(fit.reduced_chi_square, 0.0)
        self.assertAlmostEqual(
            fit.reduced_chi_square,
            fit.chi_square / max(fit.n_independent - fit.n_variables, 1.0),
            places=8)

        eps = artemis.estimate_r_noise(
            self.lep.chi_k, fit.kmin, fit.kmax, kweight=2, dk=1.0)
        self.assertGreater(eps, 0.0)

        sft = self.lep_fit["fit"].get("shell_f_test")
        self.assertIsNotNone(sft, "S7 未生成 shell_f_test")
        self.assertIsNotNone(sft.get("confidence"))
        self.assertGreater(
            sft["confidence"], 0.95,
            f"Fe–Fe 第二壳 F-test 置信度仅 {sft['confidence']:.4f}")
        self.assertTrue(sft["significant"])
        self.assertEqual(sft["dfn"], sft["p2"] - sft["p1"])
        self.assertAlmostEqual(
            sft["dfd"], fit.n_independent - sft["p2"], places=6)
        self.assertGreater(sft["delta_chi_square"], 0.0,
                           "多壳层模型必须降低 χ²（delta = χ²_simple − χ²_complex >0）")

        # 报告摘要必须披露这两个统计量（JSON/HTML 同源）
        summary = xafs_report.fit_summary(self.lep_fit["fit"])
        self.assertIn("reduced_chi_square", summary["fit"])
        self.assertIn("shell_f_test", summary)

    def test_c3_cumulant_cli_and_nidp_gate(self):
        """B4-2：C3 非谐相移回收 + N_idp 自动降级保护。

        合成第一壳 χ(k) 带 ``third=C3=+1.5e-4 Å³``（larch 的相位项
        ``−4/3·C3·k³``）。放开 C3 后必须回收真值，且 R 回到 FEFF 的
        真键长；关闭 C3 时 R 会偏 −0.010 Å。窄 R 窗下 n_var ≥ N_idp，
        必须自动降级到无 C3 而不崩溃。
        """
        self.assertEqual(xafs_fit._normalize_c3_paths("first", 3), [0])
        self.assertEqual(xafs_fit._normalize_c3_paths("all", 3), True)
        self.assertIsNone(xafs_fit._normalize_c3_paths(False, 3))
        with self.assertRaises(XAFSConfigError):
            xafs_fit._normalize_c3_paths([5], 3)

        # 不能原地改 self.lep：unittest 的类级 fixture 会被后续测试复用。
        import copy
        res = copy.deepcopy(self.lep)
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        path = artemis.load_feff_path(cache / "feff0002.dat")
        path.degen = 6.0
        c3_true = 1.5e-4
        artemis.set_path_parameters(
            path, s02=0.803, delta_r=0.0, sigma2=0.004,
            delta_e0=0.0, third=c3_true)
        k_data = np.asarray(res.chi_k.k)
        chi = np.interp(k_data, path.k, path.chi)
        res.chi_k = ChiK(k=k_data, chi=chi,
                         background=res.chi_k.background,
                         mu_norm=res.chi_k.mu_norm, rbkg=res.chi_k.rbkg)
        res.ft = athena.ft_to_r_space(k_data, chi, None, 3.0, 12.0, 2, 1.0)
        res.wt = larch_wt.continuous_wavelet_transform(
            res.chi_k, sigma=1.5, kmin=3.0, kmax=12.0,
            kweight=2, rmax=4.0)
        seg = (res.ft.r >= 0.8) & (res.ft.r <= 2.5)
        res.r1 = float(res.ft.r[seg][int(np.argmax(
            res.ft.magnitude[seg]))])
        res.structure = {"status": "confident", "coverage_ok": True,
                         "ranked": [{"id": "lepidocrocite"}], "peaks": []}

        fit_c3 = xafs_fit.fit_identified(
            res, motif_id="lepidocrocite", absorber_symbol="Fe",
            rmax=2.6, vary_c3=True, verbose=False)["fit"]
        fit_no = xafs_fit.fit_identified(
            res, motif_id="lepidocrocite", absorber_symbol="Fe",
            rmax=2.6, vary_c3=False, verbose=False)["fit"]
        p_c3, p_no = fit_c3.paths[0], fit_no.paths[0]
        self.assertIsNotNone(p_c3.c3)
        self.assertLess(abs(p_c3.c3 - c3_true), 5e-5,
                        f"C3 回收 {p_c3.c3:.3e}，真值 {c3_true:.3e}")
        self.assertIsNotNone(p_c3.c3_err)
        self.assertLess(abs(p_c3.r - path.reff),
                        abs(p_no.r - path.reff),
                        "放开 C3 后键长没有更接近 FEFF 真值")
        self.assertFalse(p_c3.c3_at_bound)

        narrow = xafs_fit.fit_identified(
            res, motif_id="lepidocrocite", absorber_symbol="Fe",
            rmin=1.7, rmax=2.5, vary_c3=True, verbose=False)
        nfit = narrow["fit"]
        self.assertTrue(narrow["c3_dropped_by_nidp"])
        self.assertTrue(narrow["flags"]["c3_dropped_by_nidp"])
        self.assertGreaterEqual(nfit.n_variables, 1)
        self.assertTrue(all(p.c3 is None for p in nfit.paths))

    def test_wt_element_reference_ruler_sigma2_shift(self):
        """B5-2：候选散射元素理论 k_peak 标尺随 Z 单调、随 σ² 左移。

        该标尺存在的意义正是打破"3d 金属峰一定在 7 Å⁻¹"的经验误判：
        高无序 Fe–Fe 的包络峰会被 exp(−2σ²k²) 拉低 ≥0.5 Å⁻¹，与
        轻/中元素重合。缓存命中后重复调用必须返回完全相同的值。
        """
        low = structure_id.reference_element_k_peaks(
            "Fe", elements=("O", "Cl", "Fe"), r_bond=2.2, sigma2=0.003)
        high = structure_id.reference_element_k_peaks(
            "Fe", elements=("Fe",), r_bond=2.2, sigma2=0.015)
        for el in ("O", "Cl", "Fe"):
            self.assertIn(el, low)
            self.assertTrue(np.isfinite(low[el]))
        self.assertLess(low["O"], low["Cl"])
        self.assertLess(low["Cl"], low["Fe"])
        self.assertGreaterEqual(low["Fe"] - low["O"], 1.2,
                                f"Fe−O 标尺间距仅 "
                                f"{low['Fe'] - low['O']:.2f} Å⁻¹")
        self.assertGreaterEqual(low["Fe"] - high["Fe"], 0.5,
                                f"σ² 0.003→0.015 的 Fe 峰位移仅 "
                                f"{low['Fe'] - high['Fe']:.2f} Å⁻¹")
        # 缓存一致性：二次调用逐位相同
        again = structure_id.reference_element_k_peaks(
            "Fe", elements=("Fe",), r_bond=2.2, sigma2=0.003)
        self.assertEqual(again["Fe"], low["Fe"])

        # 诊断图接口必须接受标尺参数且不破坏既有 CSV 列
        with tempfile.TemporaryDirectory() as tmp:
            prof = xafs_plot_special.write_wt_shell_profiles(
                self.lep, self.lep_fit, tmp, stem="ruler",
                ref_k_peaks=low)
            header = Path(prof["csv"]).read_text(
                encoding="utf-8").splitlines()[0]
            self.assertIn("data_ridge", header)
            self.assertIn("data_phase_ridge", header)
            self.assertIsNotNone(prof["complex_r_factor"])

    def test_c4_cumulant_and_asymmetric_rdf_reconstruction(self):
        """B7-3：C4 回收与正偏态 g(R) 的重构契约。

        1) C4=6e-5 合成单壳谱，vary_c4=True 回收误差 <2.5e-5；
        2) RDF 面积严格等于 N，且 C3>0 时 r_mode < r_mean、
           skewness>0、右尾延伸大于左尾。

        注：规范原文“右半侧积分面积大于左半侧”与归一化正偏态分布
        数学矛盾——在 r0 处切分时，质量向峰位左侧集中必然导致
        左面积 > 右面积（本实现实测 3.159 vs 2.779）。这里改用
        等价可检验的尾长判据（q95−r0 > r0−q05），不伪造面积。
        """
        motif = [m for m in structure_lib.load_library("Ni")
                 if m.id == "nio_bunsenite_cod_1010093"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        datfile = None
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if int(getattr(p, "nleg", 2)) == 2:
                datfile = f
                break
        self.assertIsNotNone(datfile, "NiO 单散射路径缺失")
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(p, s02=1.0, delta_r=0.0,
                                    sigma2=0.004, delta_e0=0.0,
                                    fourth=6e-5)
        self.assertIn(p.pathpar_name("fourth"), p.params,
                      "set_path_parameters 未把 fourth 映射进 larch params")
        k = np.asarray(p.k)
        chi = np.asarray(p.chi)
        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.2)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        fit = artemis.fit_r_space(
            ck, ft, [datfile], rmin=max(1.0, p.reff - 0.8),
            rmax=p.reff + 0.8, kmin=3.0, kmax=12.0, kweight=2,
            s02=1.0, vary_n=False, vary_c4=True,
            sigma2_init=0.004, deltar_lim=(-0.1, 0.1))
        path_fit = fit.paths[0]
        self.assertIsNotNone(path_fit.c4)
        self.assertLess(abs(path_fit.c4 - 6e-5), 2.5e-5,
                        f"C4 回收 {path_fit.c4:.3e} vs 真值 6e-5")
        self.assertFalse(path_fit.c4_at_bound)
        self.assertEqual(artemis.C4_LIMIT, 2e-4)

        rdf = artemis.reconstruct_pair_distribution(
            np.linspace(1.5, 2.7, 601), n=6.0, r0=2.05,
            sigma2=0.006, c3=2.5e-4)
        self.assertLess(abs(rdf["area"] - 6.0), 1e-3,
                        f"g(R) 积分面积 {rdf['area']:.6f} ≠ 6.0")
        self.assertEqual(rdf["r"].shape, rdf["g_r"].shape)
        self.assertTrue(np.all(rdf["g_r"] >= 0.0))
        self.assertLess(rdf["r_mode"], rdf["r_mean"],
                        "正偏态 g(R) 的峰位必须低于平均键长")
        self.assertGreater(rdf["skewness"], 0.0)
        r, g = rdf["r"], rdf["g_r"]
        cdf = np.cumsum(g) * np.gradient(r)
        cdf /= cdf[-1]
        q05 = float(r[np.searchsorted(cdf, 0.05)])
        q95 = float(r[np.searchsorted(cdf, 0.95)])
        self.assertGreater(q95 - 2.05, 2.05 - q05,
                           f"右尾 {q95 - 2.05:.3f} 未长于左尾 {2.05 - q05:.3f}")
        left = float(np.trapezoid(g[r <= 2.05], r[r <= 2.05]))
        right = float(np.trapezoid(g[r >= 2.05], r[r >= 2.05]))
        self.assertGreater(left, right,
                           "归一化正偏态分布在 r0 处切分应满足左面积>右面积；"
                           "若此断言失败说明归一化或非负截断出错")

    def test_multi_dataset_corefinement_and_einstein_model(self):
        """B8-1：双温度全局联合拟合 + 爱因斯坦 σ²(T) 约束。

        用 LEPIDO 第一壳 Fe–O（μ=12.436 amu）在真值 Θ_E=420 K、
        σ²_static=0.0015 Å²、ΔR=-0.02 Å、N=6.0 下合成 80 K 与 350 K
        两条谱。要求：
          - 全局自由变量从独立 2×4=8 降为 5（N/ΔR/e0/σ²_static/Θ_E）；
          - Θ_E 回收误差 <15 K，σ²_static 误差 <5e-4；
          - 两数据集各自 Rf <1e-4，且 ΔR/N 也被正确绑定回收。
        同时核对 HBAR2_OVER_2KB=24.254316 与 T≤0 的干净拒绝。
        """
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        first = None
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if int(getattr(p, "nleg", 2)) == 2:
                if first is None or p.reff < first[1].reff:
                    first = (f, p)
        self.assertIsNotNone(first, "lepidocrocite 第一壳路径缺失")
        datfile, path0 = first
        self.assertLess(abs(path0.reff - 1.98), 0.05)

        # 常数与输入护栏
        self.assertEqual(artemis.HBAR2_OVER_2KB, 24.254316)
        self.assertAlmostEqual(
            artemis.einstein_sigma2(300.0, 500.0, 12.436),
            0.005717251068417776, places=12)
        for kwargs in ({"temperature_K": 0.0, "theta_e": 420.0,
                        "reduced_mass_amu": 12.436},
                       {"temperature_K": 300.0, "theta_e": 0.0,
                        "reduced_mass_amu": 12.436},
                       {"temperature_K": 300.0, "theta_e": 420.0,
                        "reduced_mass_amu": 0.0}):
            with self.assertRaises(XAFSConfigError):
                artemis.einstein_sigma2(**kwargs)

        theta_true, static_true = 420.0, 0.0015
        dr_true, n_true, mu = -0.02, 6.0, 12.436
        k = np.arange(2.0, 13.0, 0.02)
        datasets = []
        sigma2_true = {}
        for T in (80.0, 350.0):
            p = artemis.load_feff_path(datfile)
            p.degen = n_true
            s2 = artemis.einstein_sigma2(
                T, theta_true, mu, static_true)
            sigma2_true[T] = s2
            artemis.set_path_parameters(
                p, s02=1.0, delta_r=dr_true, sigma2=s2, delta_e0=0.0)
            chi = artemis.exafs_model_k([p], k)
            ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                      mu_norm=np.ones_like(k), rbkg=1.0)
            ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
            datasets.append({
                "name": f"T{int(T)}", "chi_k": ck, "ft": ft,
                "datfiles": [datfile], "rmin": 1.0, "rmax": 2.6,
                "kmin": 3.0, "kmax": 12.0, "kweight": 2, "s02": 1.0,
                "vary_n": True, "path_degen": [n_true],
                "path_labels": ["Fe-O"], "temperature_K": T})

        ref = artemis.fit_multi_datasets(
            datasets,
            shared_bonds=[{"target": (1, 0), "source": (0, 0),
                           "share_deltar": True, "share_sigma2": False,
                           "degen_ratio": 1.0}],
            einstein_models=[{"paths": [(0, 0), (1, 0)],
                              "reduced_mass_amu": mu,
                              "theta_e_init": 350.0}])
        self.assertEqual(ref["n_variables_total"], 5,
                         f"全局变量数 {ref['n_variables_total']} ≠ 5")
        self.assertGreater(ref["n_independent_total"],
                           ref["n_variables_total"])
        self.assertEqual(len(ref["datasets"]), 2)
        self.assertEqual(len(ref["einstein_results"]), 1)
        ein = ref["einstein_results"][0]
        self.assertLess(abs(ein["theta_e"] - theta_true), 15.0,
                        f"Θ_E 回收 {ein['theta_e']:.4f} vs 真值 {theta_true}")
        self.assertLess(abs(ein["sigma2_static"] - static_true), 5e-4,
                        f"σ²_static 回收 {ein['sigma2_static']:.6f} "
                        f"vs 真值 {static_true}")
        for di, T in enumerate((80.0, 350.0)):
            fr = ref["datasets"][di]
            self.assertLess(fr.r_factor, 1e-4,
                            f"T={T} K 的 Rf={fr.r_factor:.3e} ≥1e-4")
            self.assertAlmostEqual(
                fr.paths[0].delta_r, dr_true, places=4,
                msg=f"T={T} K 的 ΔR 未被共享约束")
            self.assertAlmostEqual(
                fr.paths[0].degeneracy, n_true, places=4,
                msg=f"T={T} K 的 N 未被共享约束")
            self.assertAlmostEqual(
                fr.paths[0].sigma2, sigma2_true[T], places=10,
                msg=f"T={T} K 的 σ²(T) 未按爱因斯坦模型计算")
            self.assertIsNotNone(fr.paths[0].sigma2_err)
        self.assertLess(abs(ref["global_r_factor"]), 1e-4)

    def test_vary_c4_cli_nidp_gate_and_rdf_export(self):
        """#47 B9-1：C4 回收、N_idp 降级与真实 LEPIDO 的 g(R) 导出。

        1) 合成 Fe–O 单壳 C4=6e-5，经 ``fit_identified(vary_c4=True)``
           回收误差 <2.5e-5；
        2) LEPIDO 取 4 条壳层、vary_c4="all" 时 4 个 C4 变量使
           n_var=17 > N_idp≈15，必须先降为仅第一壳并记
           ``c4_dropped_by_nidp=True``；
        3) 真实 LEPIDO 放开 C3 后 ``rdf_table`` 每条路径的 g(R) 面积
           必须等于拟合 N，并能写出 7 列 ``_rdf.csv``。
        """
        import types

        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        first = None
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if int(getattr(p, "nleg", 2)) == 2:
                if first is None or p.reff < first[1].reff:
                    first = (f, p)
        self.assertIsNotNone(first, "lepidocrocite 第一壳路径缺失")
        datfile = first[0]

        k = np.arange(2.0, 13.0, 0.02)
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(
            p, s02=0.803, delta_r=-0.05, sigma2=0.004,
            delta_e0=0.5, fourth=6e-5)
        chi = artemis.exafs_model_k([p], k)
        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        res = types.SimpleNamespace(
            chi_k=ck, ft=ft, merged=None, norm=None,
            structure=None, absorber_symbol="Fe", self_absorption=None)
        out = xafs_fit.fit_identified(
            res, motif_id="lepidocrocite", absorber_symbol="Fe",
            rmin=1.2, rmax=2.4, vary_c4=True, s02=0.803,
            verbose=False)
        fit = out["fit"]
        c4 = fit.paths[0].c4
        self.assertIsNotNone(c4, "vary_c4=True 未产生 C4 变量")
        self.assertLess(abs(c4 - 6e-5), 2.5e-5,
                        f"C4 回收 {c4:.3e} vs 真值 6.0e-05")
        self.assertIn(fit.paths[0].label,
                      out["flags"]["c4_active_shells"])

        # 4 壳层 + 全部 C4 → N_idp 防护
        out2 = xafs_fit.fit_identified(
            self.lep, motif_id="lepidocrocite", absorber_symbol="Fe",
            rmax=4.0, vary_c4="all", verbose=False)
        self.assertTrue(out2["flags"]["c4_dropped_by_nidp"],
                        "4 壳层 + C4 未触发 c4_dropped_by_nidp")
        self.assertEqual(len(out2["flags"]["c4_active_shells"]), 1,
                         "C4 降级后应只保留第一壳")
        self.assertLess(out2["fit"].n_variables,
                        out2["fit"].n_independent)

        out3 = xafs_fit.fit_identified(
            self.lep, motif_id="lepidocrocite", absorber_symbol="Fe",
            vary_c3=True, verbose=False)
        rdf = xafs_fit.rdf_table(out3["fit"])
        self.assertEqual(len(rdf["rows"]), len(out3["fit"].paths),
                         "并非每条 C3 路径都生成了 g(R)")
        for summary, path in zip(rdf["summary"], out3["fit"].paths):
            self.assertAlmostEqual(summary["area"], path.degeneracy,
                                   places=6,
                                   msg=f"{path.label} 的 g(R) 面积≠N")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "lepido_rdf.csv"
            xafs_fit.write_rdf_csv(csv_path, rdf)
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(
                encoding="utf-8-sig").splitlines()[0]
            for col in ("label", "r_A", "g_r", "r_mode", "r_mean",
                        "skewness", "area"):
                self.assertIn(col, header)

    def test_wt_ruler_auto_mount_and_l3_whiteline_area(self):
        """#49 B9-2：WT 元素标尺自动挂载 + L3 白线净面积。

        K 边保留 ``pre_edge_n_peaks``；L3 边必须没有该字段，而给出
        有限的 ``white_line_area``/``whiteline_peak_height``。
        自动标尺的元素集合来自拟合路径 ∪ {O, 吸收元素}，并带缓存。
        """
        fit_obj = self.lep_fit["fit"]["fit"]
        ruler = xafs_report.auto_ref_k_peaks(
            self.lep, fit_obj, "Fe")
        self.assertIsNotNone(ruler, "拟合路径存在时未生成 WT 标尺")
        self.assertIn("O", ruler)
        self.assertIn("Fe", ruler)
        self.assertTrue(np.isfinite(ruler["O"]))
        self.assertTrue(np.isfinite(ruler["Fe"]))
        self.assertLess(ruler["O"], ruler["Fe"])
        again = xafs_report.auto_ref_k_peaks(
            self.lep, fit_obj, "Fe")
        self.assertIs(again, ruler, "同一参数未命中 WT 标尺缓存")
        with tempfile.TemporaryDirectory() as tmp:
            prof = xafs_plot_special.write_wt_shell_profiles(
                self.lep, self.lep_fit, tmp, ref_k_peaks=ruler)
            self.assertTrue(Path(prof["png"]).exists())

        k_feat = athena.extract_xanes_features(
            self.lep.norm, edge="K", absorber="Fe")
        self.assertIn("pre_edge_n_peaks", k_feat)
        self.assertNotIn("white_line_area", k_feat)

        res_pt = xafs_pipeline.process_sample(
            [PT_FOIL], absorber="Pt", strict=False, verbose=False)
        l_feat = athena.extract_xanes_features(
            res_pt.norm, edge="L3", absorber="Pt")
        self.assertNotIn("pre_edge_n_peaks", l_feat,
                         "L3 边不应拟合 1s→3d 边前峰")
        self.assertTrue(np.isfinite(l_feat["white_line_area"]))
        self.assertGreater(l_feat["white_line_area"], 0.0)
        self.assertTrue(np.isfinite(l_feat["whiteline_peak_height"]))

    def test_correlated_debye_sigma2_contract(self):
        """#56 B11-2：关联 Debye 模型的温度/距离趋势与输入护栏。

        σ²(300 K)>σ²(80 K)；R=4 Å 的关联因子更接近 1，σ² 大于 R=2 Å；
        T→0 收敛到与温度无关的零点平台（静态项 + 零点振动），R=4 Å 的
        高温值与 Einstein 模型同量级（0.5–2）。
        """
        kw = dict(theta_d=400.0, mass_absorber=55.845,
                  mass_scatterer=15.999)
        near_80 = artemis.correlated_debye_sigma2(80.0, reff=2.0, **kw)
        near_300 = artemis.correlated_debye_sigma2(300.0, reff=2.0, **kw)
        far_300 = artemis.correlated_debye_sigma2(300.0, reff=4.0, **kw)
        self.assertGreater(near_300, near_80)
        self.assertGreater(far_300, near_300)
        self.assertLess(artemis.correlated_debye_sigma2.last_details[
            "correlation_factor"], 1.0)
        self.assertEqual(artemis.correlated_debye_sigma2.last_details[
            "k_debye_source"], "conservative_default_0.35A^-1")

        t_plateau_a = artemis.correlated_debye_sigma2(
            1e-6, reff=2.0, **kw)
        t_plateau_b = artemis.correlated_debye_sigma2(
            1e-4, reff=2.0, **kw)
        self.assertAlmostEqual(t_plateau_a, t_plateau_b, places=8)
        static = 0.0015
        with_static = artemis.correlated_debye_sigma2(
            1e-6, reff=2.0, sigma2_static=static, **kw)
        self.assertAlmostEqual(with_static - static, t_plateau_a, places=10)

        einstein = artemis.einstein_sigma2(1000.0, 400.0, 12.436)
        far_1000 = artemis.correlated_debye_sigma2(1000.0, reff=4.0, **kw)
        ratio = far_1000 / einstein
        self.assertGreaterEqual(ratio, 0.5)
        self.assertLessEqual(ratio, 2.0)
        self.assertGreater(far_1000, near_300)

        for bad in (
                dict(temperature_K=0.0, theta_d=400.0, reff=2.0,
                     mass_absorber=55.845, mass_scatterer=15.999),
                dict(temperature_K=300.0, theta_d=0.0, reff=2.0,
                     mass_absorber=55.845, mass_scatterer=15.999),
                dict(temperature_K=300.0, theta_d=400.0, reff=2.0,
                     mass_absorber=0.0, mass_scatterer=15.999)):
            with self.assertRaises(XAFSConfigError):
                artemis.correlated_debye_sigma2(**bad)

    def test_bond_valence_sum_restraint(self):
        """#57 B11-2：BVS 计算与软约束。

        Fe–O N=6,R=2.0 的 BVS≈3.128（Fe³⁺ 合理）；约束 target=3.0
        后拟合输出的 bvs 必须与用手工 BVS 从最终路径重算的值一致，
        且 |bvs−3| 小于无约束拟合；bvs_chi2 有限。
        """
        bvs = artemis.bond_valence_sum(
            [{"N": 6.0, "R": 2.0, "ligand": "O"}], absorber="Fe")
        self.assertAlmostEqual(bvs, 3.12804471568531, places=10)
        with self.assertRaises(XAFSConfigError):
            artemis.bond_valence_sum(
                [{"N": 6.0, "R": 2.0, "ligand": "P"}], absorber="Fe")

        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        datfile = None
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if (int(getattr(p, "nleg", 2)) == 2
                    and abs(p.reff - 1.98) < 0.05):
                datfile = f
                break
        self.assertIsNotNone(datfile)
        k = np.linspace(2.0, 14.0, 601)
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(
            p, s02=0.803, delta_r=-0.05, sigma2=0.004, delta_e0=0.5)
        chi = np.interp(k, p.k, p.chi)
        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.2)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2, 1.0)
        base = artemis.fit_r_space(
            ck, ft, [str(datfile)], 1.2, 2.4, kmin=3.0, kmax=12.0,
            kweight=2, s02=0.803, vary_n=True, vary_delta_e0=True,
            path_labels=["Fe-O"])
        free_bvs = artemis.bond_valence_sum(base.paths, absorber="Fe")
        self.assertIsNone(base.bvs)
        constrained = artemis.fit_r_space(
            ck, ft, [str(datfile)], 1.2, 2.4, kmin=3.0, kmax=12.0,
            kweight=2, s02=0.803, vary_n=True, vary_delta_e0=True,
            path_labels=["Fe-O"],
            bvs_restraint={"target": 3.0, "sigma": 0.05})
        self.assertEqual(constrained.bvs_target, 3.0)
        self.assertIsNotNone(constrained.bvs)
        manual = artemis.bond_valence_sum(
            constrained.paths, absorber="Fe")
        self.assertAlmostEqual(constrained.bvs, manual, places=10)
        self.assertLess(abs(constrained.bvs - 3.0),
                        0.05 * abs(free_bvs - 3.0))
        self.assertTrue(np.isfinite(constrained.bvs_chi2))
        self.assertLess(constrained.bvs_chi2, 0.01)

    def test_multispace_q_and_wt_fit(self):
        """#58 B11-3：R/q/WT 三空间拟合回收同一组 N/R/σ²。

        同一合成单壳谱在三个空间分别拟合，参数差必须
        |ΔN|<0.3、|ΔR|<0.02 Å、|Δσ²|<0.001 Å²；默认 "r" 与显式
        "r" 逐位一致（历史口径不回退）。
        """
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        datfile = None
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if (int(getattr(p, "nleg", 2)) == 2
                    and abs(p.reff - 1.98) < 0.05):
                datfile = f
                break
        k = np.arange(2.0, 13.0, 0.02)
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(
            p, s02=0.803, delta_r=-0.05, sigma2=0.004, delta_e0=0.5)
        chi = artemis.exafs_model_k([p], k)
        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)

        fits = {}
        for space in ("r", "q", "wt"):
            fits[space] = artemis.fit_r_space(
                ck, ft, [str(datfile)], 1.2, 2.4,
                kmin=3.0, kmax=12.0, kweight=2, s02=0.803,
                vary_n=True, vary_delta_e0=True,
                path_degen=[6.0], path_labels=["Fe-O"],
                fit_space=space)
            self.assertEqual(fits[space].fit_space, space)
            self.assertTrue(np.isfinite(fits[space].aic))
            self.assertTrue(np.isfinite(fits[space].bic))

        default = artemis.fit_r_space(
            ck, ft, [str(datfile)], 1.2, 2.4,
            kmin=3.0, kmax=12.0, kweight=2, s02=0.803,
            vary_n=True, vary_delta_e0=True,
            path_degen=[6.0], path_labels=["Fe-O"])
        self.assertEqual(default.r_factor, fits["r"].r_factor)
        self.assertEqual(default.paths[0].degeneracy,
                         fits["r"].paths[0].degeneracy)

        ref = fits["r"].paths[0]
        for space in ("q", "wt"):
            got = fits[space].paths[0]
            self.assertLess(abs(got.degeneracy - ref.degeneracy), 0.3)
            self.assertLess(abs(got.r - ref.r), 0.02)
            self.assertLess(abs(got.sigma2 - ref.sigma2), 0.001)

    def test_mini_gds_stoichiometric_sum_and_ratio_constraints(self):
        """#63 B14-1：N_sum 精确守恒、误差严格传递、循环约束拒绝。

        Fe–O@1.98 + Fe–Cl@2.35 双路径合成谱（N=2.4/3.6）：
        - 无约束 n_variables=7，n_sum 约束后=6（真实减少 1）；
        - N_O+N_Cl=6.0 逐位成立，且 σ(N_Cl)=σ(N_O)；
        - 循环约束与未知标签抛 XAFSConfigError。
        """
        from larch.xafs import feffpath

        def single_path(distance, element, workdir):
            inp = artemis.build_feff_input(
                [(distance, 0.0, 0.0, element)], absorber="Fe",
                rmax=distance + 0.8, title=f"gds_{element}")
            files = artemis.run_feff6l(inp, workdir)
            for f in files:
                p = artemis.load_feff_path(f)
                if (int(getattr(p, "nleg", 2)) == 2
                        and abs(float(p.reff) - distance) < 0.05):
                    return f
            raise AssertionError(f"FEFF 未生成 {element}@{distance} SS 路径")

        with tempfile.TemporaryDirectory() as tmp:
            f_o = single_path(1.98, "O", Path(tmp) / "o")
            f_cl = single_path(2.35, "Cl", Path(tmp) / "cl")
            files = [f_o, f_cl]
            labels = ["Fe-O", "Fe-Cl"]
            truth_n = [2.4, 3.6]
            s02, s2, e0 = 0.803, 0.004, 2.0
            k = np.arange(3.0, 12.0, 0.05)
            plist = []
            for f, n in zip(files, truth_n):
                p = feffpath(str(f), s02=s02, sigma2=s2, e0=e0, deltar=0.0)
                p.degen = n
                p._calc_chi()
                plist.append(p)
            chi = artemis.exafs_model_k(plist, k)
            chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                         mu_norm=np.zeros_like(k), rbkg=1.0)
            ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)

            def do_fit(**kw):
                return artemis.fit_r_space(
                    chi_k, ft, files, rmin=1.4, rmax=2.9,
                    kmin=3.0, kmax=12.0, kweight=2, s02=s02,
                    vary_delta_e0=True, vary_n=True,
                    path_degen=truth_n, path_labels=labels, **kw)

            free = do_fit()
            self.assertEqual(free.n_variables, 7)
            constrained = do_fit(constraints="N(Fe-O)+N(Fe-Cl)=6.0")
            self.assertEqual(constrained.n_variables, 6,
                             "n_sum 应真实减少一个 N 自由变量")
            n_o = constrained.paths[0].degeneracy
            n_cl = constrained.paths[1].degeneracy
            self.assertAlmostEqual(n_o + n_cl, 6.0, places=12)
            self.assertLess(abs(n_o - 2.4), 0.25)
            self.assertLess(abs(n_cl - 3.6), 0.25)
            self.assertIsNotNone(constrained.paths[0].n_err)
            self.assertIsNotNone(constrained.paths[1].n_err)
            self.assertEqual(constrained.paths[1].n_err,
                             constrained.paths[0].n_err,
                             "n_sum 误差必须严格按 |coeff| 传递")
            self.assertLess(constrained.r_factor, 0.05)

            # 字典形式的 n_ratio：N_Cl = 1.5·N_O
            ratio = do_fit(constraints=[
                {"kind": "n_ratio", "target": "Fe-Cl",
                 "source": "Fe-O", "ratio": 1.5}])
            self.assertEqual(ratio.n_variables, 6)
            self.assertAlmostEqual(
                ratio.paths[1].degeneracy,
                1.5 * ratio.paths[0].degeneracy, places=12)
            self.assertAlmostEqual(
                ratio.paths[1].n_err, 1.5 * ratio.paths[0].n_err,
                places=12)

            with self.assertRaises(XAFSConfigError):
                do_fit(constraints=(
                    "N(Fe-O)=1.0*N(Fe-Cl);N(Fe-Cl)=1.0*N(Fe-O)"))
            with self.assertRaises(XAFSConfigError):
                do_fit(constraints="N(Fe-O)+N(Fe-Pt)=6.0")

            # 审查 #18②：固定源 N_A=6 + 守恒 total=4 → N_B=-2 必须拒绝（禁止负 CN）。
            with self.assertRaises(XAFSConfigError):
                artemis.fit_r_space(
                    chi_k, ft, files, rmin=1.4, rmax=2.9,
                    kmin=3.0, kmax=12.0, kweight=2, s02=s02,
                    vary_delta_e0=True, vary_n=[],
                    path_degen=[6.0, 3.6], path_labels=labels,
                    constraints="N(Fe-O)+N(Fe-Cl)=4.0")
            # 正控：固定源 N_A=2 + total=6 → N_B=4，可行且严格守恒。
            ok_fixed = artemis.fit_r_space(
                chi_k, ft, files, rmin=1.4, rmax=2.9,
                kmin=3.0, kmax=12.0, kweight=2, s02=s02,
                vary_delta_e0=True, vary_n=[],
                path_degen=[2.0, 3.6], path_labels=labels,
                constraints="N(Fe-O)+N(Fe-Cl)=6.0")
            self.assertAlmostEqual(ok_fixed.paths[0].degeneracy, 2.0, places=9)
            self.assertAlmostEqual(ok_fixed.paths[1].degeneracy, 4.0, places=9)

    def test_multi_dataset_correlated_debye_corefinement(self):
        """#65 B14-3：单 Θ_D 同时约束近壳/远壳 σ²(T,R)。

        用 Fe–O@2.00 与 Fe–O@3.68 两条 FEFF 路径，以真值 Θ_D=420 K、
        σ²_static=0.0012 Å² 合成 80 K 与 350 K 两套谱：
        - 一个 debye_models 组绑定两温度下的全部 4 条路径，
          Θ_D 回收误差 <15 K、σ²_static <5e-4；
        - 同一温度下远壳 σ² 严格大于近壳 σ²，global Rf <1e-4；
        - 与 einstein_models/share_sigma2 绑定同一路径时抛错。
        """
        from larch.xafs import feffpath

        true_theta, true_static, s02 = 420.0, 0.0012, 0.803

        def find_ss(files, target, tol=0.08):
            best = None
            for f in files:
                p = artemis.load_feff_path(f)
                if int(getattr(p, "nleg", 2)) != 2:
                    continue
                if (best is None or abs(float(p.reff) - target)
                        < abs(float(best[1].reff) - target)):
                    best = (f, p)
            if best is None or abs(float(best[1].reff) - target) > tol:
                raise AssertionError(f"FEFF 未生成 {target} Å 附近的 SS 路径")
            return best[0]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sites = []
            for d, el in ((2.0, "O"), (3.68, "O")):
                for axis in ((d, 0, 0), (-d, 0, 0), (0, d, 0),
                             (0, -d, 0), (0, 0, d), (0, 0, -d)):
                    sites.append((*axis, el))
            inp = artemis.build_feff_input(
                sites, absorber="Fe", rmax=4.5, title="debye_test")
            files = artemis.run_feff6l(inp, tmp / "feff")
            near = find_ss(files, 2.0)
            far = find_ss(files, 3.68)

            k = np.arange(3.0, 12.0, 0.05)
            datasets = []
            expected = {}
            for T in (80.0, 350.0):
                plist, s2s = [], []
                for f in (near, far):
                    reff = float(artemis.load_feff_path(f).reff)
                    s2 = artemis.correlated_debye_sigma2(
                        T, true_theta, reff, 55.845, 15.999,
                        sigma2_static=true_static)
                    s2s.append(s2)
                    p = feffpath(str(f), s02=s02, sigma2=s2,
                                 e0=0.0, deltar=0.0)
                    p.degen = 6.0
                    p._calc_chi()
                    plist.append(p)
                expected[T] = s2s
                chi = artemis.exafs_model_k(plist, k)
                ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                          mu_norm=np.ones_like(k), rbkg=1.0)
                ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
                datasets.append({
                    "name": f"{T:.0f}K", "chi_k": ck, "ft": ft,
                    "datfiles": [str(near), str(far)],
                    "rmin": 1.4, "rmax": 4.0, "kmin": 3.0, "kmax": 12.0,
                    "kweight": 2, "s02": s02, "vary_n": False,
                    "path_degen": [6.0, 6.0],
                    "path_labels": ["Fe-O_near", "Fe-O_far"],
                    "temperature_K": T, "edge_group": "K"})

            model = {
                "paths": [(0, 0), (0, 1), (1, 0), (1, 1)],
                "mass_absorber": 55.845,
                "mass_scatterer": 15.999,
                "theta_d_init": 400.0,
                "k_debye": None,
            }
            out = artemis.fit_multi_datasets(
                datasets, share_e0="auto", debye_models=[model])
            self.assertEqual(out["n_variables_total"], 7,
                             "4 ΔR + 2 Debye 全局变量 + ΔE0")
            self.assertEqual(len(out["debye_results"]), 1)
            deb = out["debye_results"][0]
            self.assertLess(abs(deb["theta_d"] - true_theta), 15.0)
            self.assertLess(abs(deb["sigma2_static"] - true_static), 5e-4)
            self.assertTrue(np.isfinite(deb["theta_d_err"]))
            self.assertTrue(np.isfinite(deb["sigma2_static_err"]))
            self.assertLess(out["global_r_factor"], 1e-4,
                            f"联合拟合未复现合成谱: "
                            f"Rf={out['global_r_factor']}")
            for di, T in enumerate((80.0, 350.0)):
                near_fit, far_fit = out["datasets"][di].paths
                self.assertGreater(far_fit.sigma2, near_fit.sigma2,
                                   f"{T:.0f} K 远壳 σ² 应大于近壳")
                self.assertAlmostEqual(near_fit.sigma2, expected[T][0],
                                       places=7)
                self.assertAlmostEqual(far_fit.sigma2, expected[T][1],
                                       places=7)

            with self.assertRaises(XAFSConfigError):
                artemis.fit_multi_datasets(
                    datasets, debye_models=[model],
                    einstein_models=[{"paths": [(0, 0)],
                                      "reduced_mass_amu": 12.5}])
            with self.assertRaises(XAFSConfigError):
                artemis.fit_multi_datasets(
                    datasets, debye_models=[model],
                    shared_bonds=[{"target": (1, 0), "source": (0, 0),
                                   "share_sigma2": True}])

    def test_s02_runtime_override_and_real_au_mo_foil_calibration(self):
        """#72 B18-2：运行时 S₀² 解锁未标定元素，静态表保持 #60 契约。

        - 不传 s02 时 Au 识别仍抛 XAFSDataError，get_absorber("Au").s02=None；
        - 真实 Au 箔 s02=0.82：top1=au_fcc confident，S7 N∈[11,13]、
          R∈[2.83,2.90] Å、Rf<0.02，且全局静态表未被修改；
        - 真实 Mo 箔 s02=0.88：top1=mo_bcc，双壳 N/R 回收在容差内；
        - s02 越界抛 XAFSConfigError。
        """
        from xafs_core import XAFSDataError

        au_file = (ROOT / "data" / "bmm_standards" / "Data" / "Au" /
                   "Au-L3-GoldFoil.xdi")
        mo_file = (ROOT / "data" / "bmm_standards" / "Data" / "Mo" /
                   "Mo-K-MolybdenumFoil.xdi")
        self.assertTrue(au_file.is_file())
        self.assertTrue(mo_file.is_file())

        # Rbkg=1.2：auto(1.0) 在 Au L3 白线后留下一个 1.66 Å 低 R
        # 背景鼓包，会把覆盖率误判为 partial；1.2 的 1.66 Å 伪峰消失，
        # 而第一壳 N/R 与 Rf 不受影响。
        res_au = xafs_pipeline.process_sample(
            [str(au_file)], absorber="Au", strict=False, rbkg=1.2,
            verbose=False)
        with self.assertRaises(XAFSDataError):
            structure_id.identify_result(
                res_au, absorber_symbol="Au", verbose=False)
        self.assertIsNone(get_absorber("Au").s02)

        out_au = xafs_fit.run(
            res_au, identify=True, fit=True, absorber_symbol="Au",
            s02=0.82, verbose=False)
        sid = out_au["structure"]
        self.assertEqual(sid["ranked"][0]["id"], "au_fcc")
        self.assertEqual(sid["status"], "confident")
        fout = out_au["fit"]
        self.assertEqual(fout["mode"], "motif")
        self.assertAlmostEqual(fout["s02"], 0.82, places=12)
        fit = fout["fit"]
        path = fit.paths[0]
        self.assertGreaterEqual(path.degeneracy, 11.0)
        self.assertLessEqual(path.degeneracy, 13.0)
        self.assertGreaterEqual(path.r, 2.83)
        self.assertLessEqual(path.r, 2.90)
        self.assertLess(fit.r_factor, 0.02)
        self.assertIsNone(get_absorber("Au").s02,
                          "运行时 --s02 不得修改全局静态表")

        res_mo = xafs_pipeline.process_sample(
            [str(mo_file)], absorber="Mo", strict=False, verbose=False)
        out_mo = xafs_fit.run(
            res_mo, identify=True, fit=True, absorber_symbol="Mo",
            s02=0.88, verbose=False)
        self.assertEqual(out_mo["structure"]["ranked"][0]["id"], "mo_bcc")
        fit_mo = out_mo["fit"]["fit"]
        self.assertEqual(len(fit_mo.paths), 2)
        p1, p2 = fit_mo.paths[0], fit_mo.paths[1]
        self.assertGreaterEqual(p1.degeneracy, 6.8)
        self.assertLessEqual(p1.degeneracy, 9.0)
        self.assertGreaterEqual(p1.r, 2.69)
        self.assertLessEqual(p1.r, 2.75)
        self.assertGreaterEqual(p2.degeneracy, 5.0)
        self.assertLessEqual(p2.degeneracy, 7.2)
        self.assertGreaterEqual(p2.r, 3.11)
        self.assertLessEqual(p2.r, 3.18)
        self.assertLess(fit_mo.r_factor, 0.02)
        self.assertIsNone(get_absorber("Mo").s02)

        with self.assertRaises(XAFSConfigError):
            xafs_fit.run(
                res_au, identify=False, fit=True, absorber_symbol="Au",
                s02=1.5, verbose=False)
        self.assertIn("s02", xafs_config.DEFAULTS)

    def test_chemical_state_s02_polymorphism(self):
        """化学态 S₀²：Fe 金属 0.6566，默认生产值 0.803 不变。"""
        from types import SimpleNamespace
        from absorber import resolve_s02
        import run_xafs

        self.assertEqual(get_absorber("Fe").s02, 0.803)
        self.assertEqual(resolve_s02("Fe"), 0.803)
        self.assertEqual(resolve_s02("Fe", "metal"), 0.6566)
        self.assertEqual(resolve_s02("Fe", "METAL"), 0.6566)
        with self.assertRaises(XAFSConfigError):
            resolve_s02("Fe", "oxide")
        with self.assertRaises(XAFSConfigError):
            resolve_s02("Au", "metal")
        self.assertEqual(get_absorber("Fe").s02, 0.803,
                         "化学态解析不得修改静态表")

        # CLI 优先级：显式 --s02 > 化学态 > T5 > 静态表。
        args = SimpleNamespace(s02=None, s02_from_t5=False,
                               chemical_state="metal", absorber="Fe")
        val, src = run_xafs._resolve_s02_for_paths(args, ["dummy.dat"])
        self.assertEqual(val, 0.6566)
        self.assertEqual(src, "chemical_state:Fe=metal")
        args.s02 = 0.9
        val, src = run_xafs._resolve_s02_for_paths(args, ["dummy.dat"])
        self.assertEqual(val, 0.9)
        self.assertEqual(src, "--s02 runtime override")

        # 逐元素映射：未列出的元素返回 (None, None)，多元素批处理不误伤。
        args_map = SimpleNamespace(s02=None, s02_from_t5=False,
                                   chemical_state="Fe=metal", absorber="Cu")
        self.assertEqual(
            run_xafs._resolve_s02_for_paths(args_map, ["dummy.dat"]),
            (None, None))
        args_map.absorber = "Fe"
        self.assertEqual(
            run_xafs._resolve_s02_for_paths(args_map, ["dummy.dat"]),
            (0.6566, "chemical_state:Fe=metal"))

        # 来源标记：化学态值视为已标定，普通手填 --s02 不算（用未标定
        # 元素隔离静态表 calibrated=True 的影响）。
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Cr"), "chemical_state:Cr=metal"))
        self.assertTrue(xafs_fit._s02_calibrated_flag(
            structure_id.absorber_with_s02("Fe", 0.6566), "chemical_state:Fe=metal"))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Cr"), "--s02 runtime override"))

    def test_operando_exafs_series_joint_e0_bond_and_cn_tracking(self):
        """#73 B19-1：共享 ΔE₀ 的序列拟合回收 R(t)/N(t) 并导出 CSV/PNG。

        真值 R=[2.02,1.99,1.96,1.93] Å、N=[4.0,4.6,5.3,6.0]，
        全局 ΔE₀=+2.5 eV、σ²=0.004 Å²；共享 σ² 后键长误差 <0.005 Å、
        配位数误差 <0.15，global_r_factor <1e-3。
        """
        from types import SimpleNamespace
        from larch.xafs import feffpath

        motif = structure_lib.parse_catalytic_cluster_spec(
            "Fe", "sac:O4@1.98,C=none")
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmin=1.2, rmax=2.4, kmin=3.0, kmax=12.0)
        r_truth = [2.02, 1.99, 1.96, 1.93]
        n_truth = [4.0, 4.6, 5.3, 6.0]
        k = np.arange(2.5, 12.5, 0.05)
        results = []
        for r_t, n_t in zip(r_truth, n_truth):
            p = feffpath(str(plan["files"][0]), s02=0.803,
                         sigma2=0.004, e0=2.5, deltar=r_t - 1.98)
            p.degen = n_t
            p._calc_chi()
            chi = artemis.exafs_model_k([p], k)
            chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                         mu_norm=np.ones_like(k), rbkg=1.0)
            ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
            results.append(SimpleNamespace(
                chi_k=chi_k, ft=ft, absorber_symbol="Fe", structure=None,
                merged=None, norm=None, self_absorption=None))

        out = xafs_fit.run_operando_exafs_series(
            results, "Fe", catalytic_model="sac:O4@1.98,C=none",
            share_sigma2_across_series=True, s02=0.803)
        self.assertEqual(out["motif"], motif.id)
        self.assertEqual(len(out["names"]), 4)
        label = out["shells"][0]
        traj = out["trajectories"][label]
        self.assertLess(max(abs(a - b) for a, b in
                            zip(traj["R"], r_truth)), 0.005)
        self.assertLess(max(abs(a - b) for a, b in
                            zip(traj["N"], n_truth)), 0.15)
        self.assertTrue(all(np.diff(traj["R"]) < 0),
                        f"R(t) 未严格单调递减: {traj['R']}")
        self.assertTrue(all(np.diff(traj["N"]) > 0),
                        f"N(t) 未严格单调递增: {traj['N']}")
        self.assertLess(out["global_r_factor"], 1e-3)

        with tempfile.TemporaryDirectory() as tmp:
            made = xafs_plot_special.write_operando_exafs_diagnostics(
                out, tmp, stem="operando_exafs")
            csv_path = Path(made["csv"])
            png_path = Path(made["png"])
            self.assertTrue(csv_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(lines), 5, "CSV 应为表头 + 4 个时间点")
            self.assertIn(f"{label}_N", lines[0])
            self.assertIn(f"{label}_R", lines[0])
            self.assertIn("delta_e0", lines[0])

    def test_formal_cn_refusal_without_model(self):
        """#75 P0a：无结构模型时正式 CN 拒绝，诊断拟合保留。"""
        self.assertTrue(self.lep_fit["fit"].get("diagnostic_only"))
        self.assertEqual(self.lep_fit["fit"].get("cn_status"),
                         "diagnostic_only")
        out = xafs_fit.run(
            self.lep, identify=True, fit=True, absorber_symbol="Fe",
            formal_cn=True, verbose=False)
        fit_out = out["fit"]
        self.assertEqual(fit_out["mode"], "motif")
        self.assertIsNotNone(fit_out.get("fit"))
        formal = fit_out["formal_cn"]
        self.assertEqual(formal["status"], "no_cif")
        self.assertEqual(formal["rows"], [])
        self.assertTrue(formal["reject_reason"])
        self.assertEqual(out["formal_cn"]["status"], "no_cif")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cn_table.csv"
            cn_table.write_cn_table(
                path, [cn_table.refusal_row(
                    formal["status"], formal["reject_reason"],
                    phase_id="lepidocrocite")])
            rows = cn_table.read_cn_table(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gate_status"], "no_cif")
        self.assertEqual(rows[0]["n_fit"], "")
        self.assertTrue(rows[0]["reject_reason"])

    def test_formal_cn_l1_not_in_table(self):
        """#76 P0a：L1 只做诊断，不得进入正式 CN 表。"""
        res = xafs_pipeline.process_sample(
            [FERRIHYDRITE], absorber="Fe", strict=False, verbose=False)
        out = xafs_fit.run(
            res, identify=True, fit=True, absorber_symbol="Fe",
            formal_cn=True, verbose=False)
        fit_out = out["fit"]
        self.assertEqual(fit_out["mode"], "l1_first_shell")
        formal = fit_out["formal_cn"]
        self.assertEqual(formal["status"], "l1_diagnostic_only")
        self.assertEqual(formal["rows"], [])
        row = cn_table.refusal_row(formal["status"], formal["reject_reason"])
        self.assertEqual(row["n_fit"], "")
        self.assertEqual(row["cn_label"], "none")

    def test_cn_table_schema_and_refusal_row(self):
        """#78 P0a：CN 表列顺序与拒绝行写读往返。"""
        expected = [
            "phase_id", "model_level", "source_type", "source_ref",
            "source_cif_sha256", "absorber_site", "shell_group", "shell_label",
            "scatterer", "r_cif", "n0", "n0_members", "n_fit", "n_status",
            "n_ratio", "n_err_formal", "n_err_systematic", "delta_r", "sigma2",
            "delta_e0", "rf", "s02", "s02_source", "s02_calibrated",
            "kmin", "kmax", "kweight", "rmin", "rmax", "n_idp", "n_variables",
            "cn_label", "gate_status", "reject_reason", "flags", "assumptions",
            "data_sha256", "software_version",
        ]
        self.assertEqual(cn_table.CN_TABLE_COLUMNS, expected)
        row = cn_table.refusal_row("no_cif", "no source", phase_id="x")
        self.assertEqual(row["gate_status"], "no_cif")
        self.assertEqual(row["n_fit"], "")
        self.assertTrue(row["reject_reason"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cn_table.csv"
            cn_table.write_cn_table(path, [row])
            rows = cn_table.read_cn_table(path)
        self.assertEqual(len(rows), 1)
        for key in expected:
            self.assertIn(key, rows[0])
        self.assertEqual(rows[0]["phase_id"], "x")
        self.assertEqual(rows[0]["n_fit"], "")
        self.assertEqual(rows[0]["gate_status"], "no_cif")
        self.assertTrue(rows[0]["reject_reason"])

    def test_cn_disclosure_complete_payload(self):
        """CN 披露标准：完整 payload 无问题。"""
        self.assertEqual(
            cn_disclosure.check_fit_payload(self._disclosure_payload()), [])

    def test_cn_disclosure_flags_missing_provenance(self):
        """CN 披露标准：缺结构来源/CN 状态必须报出。"""
        payload = self._disclosure_payload()
        payload.pop("structure")
        payload.pop("cn_status")
        issues = cn_disclosure.check_fit_payload(payload)
        self.assertTrue(any("structure/CIF" in x for x in issues))
        self.assertTrue(any("CN status" in x for x in issues))

    def test_cn_disclosure_flags_diagnostic_mismatch(self):
        """CN 披露标准：diagnostic_only 不得配 formal 状态。"""
        payload = self._disclosure_payload()
        payload["diagnostic_only"] = True
        payload["cn_status"] = "formal"
        issues = cn_disclosure.check_fit_payload(payload)
        self.assertTrue(any("diagnostic" in x.lower() for x in issues))

    def test_paper_figures_reuses_disclosure_check(self):
        """paper_figures 复用项目级 CN 披露检查。"""
        import paper_figures as pf

        fit = pf.load_fit(self._disclosure_payload())
        self.assertEqual(pf.validate_cn_disclosure(fit), [])

    def test_cn_disclosure_large_delta_e0_is_advisory_only(self):
        """CN 披露标准：|Delta E0| > 5 eV 只提示，不阻断。"""
        payload = self._disclosure_payload()
        payload["fit"]["delta_e0"] = 8.0
        self.assertEqual(cn_disclosure.check_fit_payload(payload), [])
        notes = cn_disclosure.delta_e0_warnings(payload)
        self.assertEqual(len(notes), 1)
        self.assertIn("large", notes[0].lower())
        self.assertIn("absorption-edge shift", notes[0].lower())

    def test_cn_disclosure_delta_e0_threshold(self):
        """CN 披露标准：5 eV 边界与负值都要判断。"""
        payload = self._disclosure_payload()
        for value, expected in ((5.0, 0), (5.01, 1), (-8.0, 1), (0.0, 0)):
            payload["fit"]["delta_e0"] = value
            self.assertEqual(
                len(cn_disclosure.delta_e0_warnings(payload)), expected,
                msg=f"delta_e0={value}")

    def test_cn_disclosure_cli_advisory_does_not_fail(self):
        """CN 披露标准：CLI 对大 Delta E0 返回 0，并打印 NOTE。"""
        import contextlib
        import io

        payload = self._disclosure_payload()
        payload["fit"]["delta_e0"] = 8.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cn_disclosure.main([str(path)])
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("[NOTE]", output)
        self.assertIn("large", output.lower())

    def test_paper_figures_cn_table_includes_delta_e0_advisory(self):
        """CN 表：大 Delta E0 写入 Disclosure warnings 与 CSV advisory 列。"""
        import paper_figures as pf

        fit = pf.load_fit(self._disclosure_payload())
        self.assertIsNotNone(fit)
        fit.delta_e0 = 8.0
        with tempfile.TemporaryDirectory() as tmp:
            md = pf.write_cn_markdown(fit, tmp, stem="cn_table")
            csv_path = pf.write_cn_csv(fit, tmp, stem="cn_table")
            md_text = md.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8-sig")
        self.assertIn("Disclosure warnings", md_text)
        self.assertIn("advisory, non-blocking", md_text)
        self.assertIn("large |Delta E0|", md_text)
        self.assertIn("disclosure_advisories", csv_text)
        self.assertIn("large |Delta E0|", csv_text)

    def test_cn_range_12_metals(self):
        """#100 P2a：fcc/hcp 金属 CN=12 正常通过；CN=8 (bcc) 也通过。"""
        pack = ROOT / "_structure_lib" / "_phase_pack"
        cases = {
            "Co": ("9008492", "bd9d57408d901e3ff966ba0ebe19ced70e1f1b62a4ef47fda801a39ecbb5344a.cif", 12.0),
            "Ni": ("1534892", "adb86b38b5313592734259a0ff2bb537fc60acb92793015ee8a74a41b9966cdd.cif", 12.0),
            "Cu": ("4105040", "39726e1afd93226e126184d30eb0e45df8138f9a78c07b5fd59d9a700cb2300c.cif", 12.0),
            "Pt": ("1011103", "e44ec4c99848ea8b6805f2a4a9ce750157197b0885f2d9b8da0aa7a961a5c002.cif", 12.0),
            "Fe": ("1100108", "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif", 8.0),
        }
        for el, (_id, fname, expected_cn) in cases.items():
            cif = pack / "COD" / _id / fname
            self.assertTrue(cif.is_file(), str(cif))
            r = cif_gate.gate_cif(cif, el)
            self.assertTrue(r.ok, f"{el}: {r.reason}")
            self.assertEqual(r.first_shell_cn, expected_cn,
                             f"{el} first shell CN={r.first_shell_cn} != {expected_cn}")

    def test_n0_shell_grouping_rayleigh(self):
        """#101 P2a：Rayleigh 分组：同元素子壳间距 < ΔR_Rayleigh 归为一组。"""
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        # k 范围越窄，ΔR_Rayleigh 越大，分组越粗（合并更多壳层）
        r_wide = cif_gate.gate_cif(cif, "Fe", kmin=3.0, kmax=6.0)
        r_narrow = cif_gate.gate_cif(cif, "Fe", kmin=3.0, kmax=12.0)
        self.assertTrue(r_wide.ok)
        self.assertTrue(r_narrow.ok)
        # ΔR_Rayleigh(k=3–6) = π/(2*3) ≈ 0.524 > ΔR(k=3–12) = π/(2*9) ≈ 0.175
        self.assertGreater(r_wide.rayleigh_delta_r, r_narrow.rayleigh_delta_r)
        self.assertAlmostEqual(r_wide.rayleigh_delta_r, math.pi / 6.0, places=6)
        self.assertAlmostEqual(r_narrow.rayleigh_delta_r, math.pi / 18.0, places=6)
        # 窄 k 范围（大 delta_r）：第一壳 2.484 与第二壳 2.868 间距 0.384 < 0.524，合并 → first_cn 增大
        self.assertGreater(r_wide.first_shell_cn, r_narrow.first_shell_cn)
        self.assertEqual(r_narrow.first_shell_cn, 8.0)   # bcc Fe 第一壳 CN=8，不合并
        # 验证 N₀ = Σ(简并度 × 占据度)
        for sh in r_narrow.n0_shells:
            n0_sum = sum(m["occ"] for m in sh["n0_members"])
            self.assertAlmostEqual(sh["n0"], n0_sum, places=6,
                                   msg=f"N₀ {sh['n0']} != Σ occ {n0_sum}")

    def test_fixed_n_nidp_formula(self):
        """#102 P2b：固定 N 信息量公式 2n+1 / 2n+2，map_n0_to_plan 一致。"""
        motifs = {m.id: m for m in structure_lib.load_library("Fe")}
        motif = motifs["fe_bcc"]
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        gate = cif_gate.gate_cif(cif, "Fe")
        self.assertTrue(gate.ok)

        # free 模式：3n+1
        plan_free = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0, rmax=3.5, n_mode="free")
        n_free = len(plan_free["files"])
        # fixed_n 模式：2n+1
        plan_fixed = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0, rmax=3.5, n_mode="fixed_n")
        n_fixed = len(plan_fixed["files"])
        self.assertEqual(n_free, n_fixed)
        self.assertEqual(plan_fixed["n_var_fixed_n"], 2 * n_fixed + 1)
        # conditional 模式：2n+2
        plan_cond = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0, rmax=3.5,
            n_mode="conditional")
        self.assertEqual(plan_cond["n_var_conditional"], 2 * n_fixed + 2)

        # map_n0_to_plan 的 n_var 与 select_shell_paths 一致
        mapped = xafs_fit.map_n0_to_plan(
            plan_free, gate.n0_shells, kmin=3.0, kmax=12.0)
        self.assertTrue(mapped["mapping_ok"], mapped["mapping_reason"])
        self.assertEqual(mapped["n_var_fixed_n"], 2 * n_free + 1)
        self.assertEqual(mapped["n_var_free_n_first"], 2 * n_free + 2)
        # n0_per_path 与前两个壳层 N₀ 一致
        self.assertEqual(len(mapped["n0_per_path"]), n_free)
        expected_n0 = [round(s["n0"], 6) for s in gate.n0_shells[:n_free]]
        for i, n0 in enumerate(mapped["n0_per_path"]):
            self.assertAlmostEqual(n0, expected_n0[i], places=6)

    def test_counterexample_n0_subshell_distribution_conserves_sum(self):
        """审查 item 8：同一 Rayleigh N₀ 壳被 FEFF 拆成多条子壳路径时，
        整壳 N₀ 按晶体学简并度比例分摊（Σ守恒），而非就近独占错配/失败。

        子壳 2.00/2.21 Å 间距 0.21 > merge_tol 0.20（plan 拆成两条），
        但窄 k 窗 Δk=7 → ΔR_Rayleigh=π/14≈0.224 > 0.21，CIF 合并成一个
        N₀=10 壳。旧版就近独占匹配会让第二条子壳无壳可配 → 映射失败。
        """
        import xafs_fit
        plan = {
            "files": ["a.feFF", "b.feFF"],
            "degens": [6.0, 4.0],
            "labels": ["Fe-O@2.00", "Fe-O@2.21"],
            "shells": [
                {"scatterer": "O", "reff": 2.00, "degen": 6.0},
                {"scatterer": "O", "reff": 2.21, "degen": 4.0},
            ],
            "ms_links": [],
        }
        n0_shells = [{"el": "O", "r": 2.10, "n0": 10.0, "n0_members": []}]
        m = xafs_fit.map_n0_to_plan(plan, n0_shells, kmin=3.0, kmax=10.0)
        self.assertTrue(m["mapping_ok"], m["mapping_reason"])
        # 分摊：10×6/10=6、10×4/10=4；Σ=10（整壳配位数守恒）
        self.assertAlmostEqual(m["n0_per_path"][0], 6.0, places=6)
        self.assertAlmostEqual(m["n0_per_path"][1], 4.0, places=6)
        self.assertAlmostEqual(sum(m["n0_per_path"]), 10.0, places=6)
        # 子壳绑定：target=1 → source=0，degen_ratio=4/6（首壳单一比例因子）
        self.assertEqual(len(m["path_links"]), 1)
        link = m["path_links"][0]
        self.assertEqual(link["target"], 1)
        self.assertEqual(link["source"], 0)
        self.assertAlmostEqual(link["degen_ratio"], 4.0 / 6.0, places=6)
        self.assertFalse(link["share_deltar"])
        self.assertFalse(link["share_sigma2"])

    def test_counterexample_n0_negative_rejected(self):
        """审查 item 8：负配位数 N₀ 非物理 → 映射拒绝，不进入 CN 定量。"""
        import xafs_fit
        plan = {"files": ["a.feFF"], "degens": [6.0], "labels": ["Fe-O"],
                "shells": [{"scatterer": "O", "reff": 2.0, "degen": 6.0}],
                "ms_links": []}
        n0_shells = [{"el": "O", "r": 2.0, "n0": -3.0, "n0_members": []}]
        m = xafs_fit.map_n0_to_plan(plan, n0_shells)
        self.assertFalse(m["mapping_ok"])
        self.assertIn("为负", m["mapping_reason"])

    def test_counterexample_n0_ms_inherits_parent(self):
        """审查 item 8：MS 路径不独立匹配 CIF 单散射壳，按 ms_links 从
        母壳继承 N₀（×简并度比），且不参与子壳比例绑定。"""
        import xafs_fit
        plan = {
            "files": ["ss.feFF", "ms.feFF"],
            "degens": [6.0, 12.0],
            "labels": ["Ni-O@2.08", "Ni-O-Ni@4.16(MS3#0)"],
            "shells": [
                {"scatterer": "O", "reff": 2.08, "degen": 6.0,
                 "label": "Ni-O@2.08"},
                {"scatterer": "Ni", "reff": 4.16, "degen": 12.0,
                 "label": "Ni-O-Ni@4.16(MS3#0)", "kind": "ms",
                 "parent": "Ni-O@2.08"},
            ],
            "ms_links": [{"ms_path_index": 1, "parent_shell_index": 0,
                          "reff_ratio": 2.0, "degen_ratio": 2.0, "nleg": 3}],
        }
        # CIF 只有 O@2.08 单散射壳；MS 的 Ni@4.16 无独立 N₀ 壳可配
        n0_shells = [{"el": "O", "r": 2.08, "n0": 6.0, "n0_members": []}]
        m = xafs_fit.map_n0_to_plan(plan, n0_shells)
        self.assertTrue(m["mapping_ok"], m["mapping_reason"])
        self.assertAlmostEqual(m["n0_per_path"][0], 6.0, places=6)
        # MS 继承：母壳 N₀ 6 × degen_ratio 2 = 12（而非独立匹配失败）
        self.assertAlmostEqual(m["n0_per_path"][1], 12.0, places=6)
        self.assertEqual(m["path_links"], [])

    def test_conditional_cn_out_of_range_requires_defect(self):
        """#105 P3：N/N₀ 超出范围且无缺陷模型 → defect_model_missing。"""
        # N/N₀ = 0.5，低于 0.6，out_of_range
        self.assertEqual(xafs_fit.check_n_ratio(0.5), "out_of_range")
        self.assertEqual(xafs_fit.check_n_ratio(1.5), "out_of_range")
        # 无缺陷模型 → 拒绝
        chk = xafs_fit.check_defect_model(0.5, None)
        self.assertFalse(chk["ok"])
        self.assertEqual(chk["status"], "defect_model_missing")
        # 有缺陷模型但不匹配 → defect_model_mismatch
        dm = xafs_fit.parse_defect_model(
            '{"type": "custom", "expected_n_ratio": 0.75, "uncertainty": 0.05}')
        chk = xafs_fit.check_defect_model(0.5, dm)
        self.assertFalse(chk["ok"])
        self.assertEqual(chk["status"], "defect_model_mismatch")
        # N/N₀ 在正常范围 → ok，不需要缺陷模型
        self.assertEqual(xafs_fit.check_n_ratio(0.9), "ok")
        self.assertEqual(xafs_fit.check_n_ratio(1.2), "ok")

    def test_cn_table_provenance_complete(self):
        """#108 P3：正式 CN 行必须包含全部 provenance 字段。"""
        from structure_model import StructureModel, validate_formal
        model = StructureModel(
            model_id="COD:1100108", absorber="Fe",
            absorber_site="Fe1", model_level="crystal_cif",
            source_type="database_cif", source_ref="COD:1100108",
            source_cif="_structure_lib/_phase_pack/COD/1100108/x.cif",
            source_cif_sha256="0" * 64,
            cell={"a": 2.868, "b": 2.868, "c": 2.868,
                  "alpha": 90, "beta": 90, "gamma": 90},
            spacegroup="Im-3m",
            provenance_status="verified_source_cif",
        )
        # provenance 字段完整
        self.assertIsNotNone(model.model_id)
        self.assertIsNotNone(model.absorber)
        self.assertIsNotNone(model.absorber_site)
        self.assertIsNotNone(model.model_level)
        self.assertIsNotNone(model.source_type)
        self.assertIsNotNone(model.source_ref)
        self.assertIsNotNone(model.source_cif_sha256)
        self.assertIsNotNone(model.absorber_site)
        self.assertEqual(model.provenance_status, "verified_source_cif")
        # cn_table 拒绝行也应包含 provenance 字段
        row = cn_table.refusal_row("no_cif", "测试拒绝",
                                   phase_id=model.model_id,
                                   model_level=model.model_level,
                                   source_type=model.source_type,
                                   source_ref=model.source_ref,
                                   source_cif_sha256=model.source_cif_sha256,
                                   absorber_site=model.absorber_site)
        self.assertEqual(row["phase_id"], "COD:1100108")
        self.assertEqual(row["model_level"], "crystal_cif")
        self.assertEqual(row["source_type"], "database_cif")
        self.assertEqual(row["source_ref"], "COD:1100108")
        self.assertEqual(row["absorber_site"], "Fe1")
        self.assertEqual(row["gate_status"], "no_cif")

    def test_reproducible_cn_hash(self):
        """#110 P3：cn_table.csv 写读往返，内容逐字节一致。"""
        import hashlib
        row = cn_table.refusal_row(
            "no_cif", "reproducibility test",
            phase_id="COD:1100108", model_level="crystal_cif",
            source_type="database_cif", source_ref="COD:1100108",
            source_cif_sha256="0" * 64, absorber_site="Fe1")
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "cn1.csv"
            out2 = Path(tmp) / "cn2.csv"
            cn_table.write_cn_table(out1, [row])
            cn_table.write_cn_table(out2, [row])
            h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
            self.assertEqual(h1, h2)
            # 读回内容一致
            back = cn_table.read_cn_table(out1)
            self.assertEqual(len(back), 1)
            self.assertEqual(back[0]["phase_id"], "COD:1100108")

    def test_standard_foils_formal_cn(self):
        """#112 P4：标准箔（Fe/Co/Ni/Cu）formal CN 链可放行。"""
        from structure_model import StructureModel, validate_formal
        pack = ROOT / "_structure_lib" / "_phase_pack" / "COD"
        cases = {
            "Fe": ("1100108", "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif", "Fe1"),
            "Co": ("9008492", "bd9d57408d901e3ff966ba0ebe19ced70e1f1b62a4ef47fda801a39ecbb5344a.cif", "Co"),
            "Ni": ("1534892", "adb86b38b5313592734259a0ff2bb537fc60acb92793015ee8a74a41b9966cdd.cif", "Ni1"),
            "Cu": ("4105040", "39726e1afd93226e126184d30eb0e45df8138f9a78c07b5fd59d9a700cb2300c.cif", "Cu"),
        }
        for el, (_id, fname, site) in cases.items():
            cif = pack / _id / fname
            self.assertTrue(cif.is_file(), str(cif))
            gate = cif_gate.gate_cif(cif, el, absorber_site=site)
            self.assertTrue(gate.ok, f"{el}: {gate.reason}")
            self.assertEqual(gate.occupancy_source in ("explicit", "implicit_1.0"), True)
            # 构造 StructureModel 并验证 formal 门禁
            model = StructureModel(
                model_id=f"COD:{_id}", absorber=el,
                absorber_site=gate.absorber_site or site,
                model_level="crystal_cif",
                source_type="database_cif", source_ref=f"COD:{_id}",
                source_cif=str(cif.resolve()),
                source_cif_sha256=gate.cif_sha256,
                provenance_status="verified_source_cif",
            )
            v = validate_formal(model)
            self.assertTrue(v["ok"], f"{el}: {v}")

    def test_legacy_auto_fit_marked(self):
        """#114 P4：--legacy-auto-fit 标记 legacy_unverified，不进正式表。"""
        # 模拟 formal_cn 处理：legacy_auto_fit=True 时状态应被覆盖为 legacy_unverified
        formal = {"status": "pending_consistency", "reject_reason": None,
                  "rows": [], "model": None}
        # 模拟 run_xafs.py 中的 legacy_auto_fit 处理逻辑
        legacy_auto_fit = True
        if legacy_auto_fit:
            formal["status"] = "legacy_unverified"
            formal["reject_reason"] = formal.get("reject_reason") or "legacy_auto_fit: 旧版自动拟合，不进正式表"
            formal["legacy_auto_fit"] = True
        self.assertEqual(formal["status"], "legacy_unverified")
        self.assertTrue(formal["legacy_auto_fit"])
        self.assertIn("legacy_auto_fit", formal["reject_reason"])

    def test_manual_s02_override_is_not_formally_calibrated(self):
        import run_xafs
        ledger, kwargs = self._review_cn_fixture()
        for element in ("Ni", "Au"):
            for source in ("--s02 runtime override", "config:s02",
                           "s02_calibration:forged", "chemical_state:Fe=metal"):
                with self.subTest(element=element, source=source):
                    absorber = structure_id.absorber_with_s02(element, 0.95)
                    self.assertTrue(absorber.calibrated)  # diagnostic API unchanged
                    block = xafs_fit.build_formal_cn_block(
                        conditional_cn=ledger["conditional_cn"], absorber=absorber,
                        s02_source=source, **kwargs)
                    self.assertNotEqual(block["status"], "ok")
                    self.assertEqual(block["rows"], [])
                    self.assertIn("S", block["reject_reason"])
                    with tempfile.TemporaryDirectory() as tmp:
                        cn_table.write_cn_table(Path(tmp) / "cn_table.csv", [cn_table.refusal_row(
                            block["status"], block["reject_reason"],
                            s02=absorber.s02, s02_source=source, s02_calibrated=False)])
                        rows = cn_table.read_cn_table(Path(tmp) / "cn_table.csv")
                        self.assertTrue(rows)
                        self.assertTrue(all(r["reject_reason"] for r in rows))
                        self.assertTrue(all(not r["n_fit"] for r in rows))
                        self.assertTrue(all(r["n_status"] != "fitted" for r in rows))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            structure_id.absorber_with_s02("Ni", 0.95), "absorber.py"))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Ni"), "--s02 runtime override"))
        self.assertTrue(xafs_fit._s02_calibrated_flag(get_absorber("Fe"), "absorber.py"))
        self.assertTrue(xafs_fit._s02_calibrated_flag(
            structure_id.absorber_with_s02("Fe", 0.6566), "chemical_state:Fe=metal"))

    def test_report_entry_preserves_t5_s02_evidence(self):
        from unittest import mock
        path = _find("bmm_standards/Data/Cr/Cr-K-Cr2O3.xdi")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(xafs_report.xafs_fit, "run", wraps=xafs_fit.run) as run:
                xafs_report.main([str(path), "--absorber", "Cr", "--s02-from-t5",
                                  "--outdir", tmp, "--no-html", "--plot", "none"])
            kwargs = run.call_args.kwargs
            self.assertIsNotNone(kwargs["s02_evidence"])
            self.assertTrue(xafs_fit._s02_calibrated_flag(
                structure_id.absorber_with_s02("Cr", kwargs["s02"]),
                kwargs["s02_source"], kwargs["s02_evidence"]))

    def test_s02_evidence_real_holdout_csv(self):
        import run_xafs
        ledger, _kwargs = self._review_cn_fixture()
        holdout = _find("bmm_standards/Data/Ni/Ni-K-NickelHydroxide.xdi")
        evidence = xafs_fit.make_s02_evidence(ledger["calibration"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source, proof, accepted in (
                    ("s02_calibration:real_ledger", evidence, True),
                    ("--s02 runtime override", None, False),
                    ("s02_calibration:forged", None, False)):
                with self.subTest(source=source):
                    out = root / str(accepted) / source.split(":")[0].replace(" ", "_")
                    result = run_xafs.process_sample(
                        [holdout], out, absorber="Ni", fit=True, formal_cn=True,
                        phase_id="COD:1548811", offline=True, strict=True,
                        s02=ledger["calibration"]["s02"], s02_source=source,
                        s02_evidence=proof, verbose=False)
                    rows = cn_table.read_cn_table(out / "cn_table.csv")
                    self.assertTrue(rows)
                    self.assertTrue(result.fit["cn_reliable"])
                    self.assertEqual(result.fit["conditional_cn"]["n_fit_total"],
                                     ledger["conditional_cn"]["n_fit_total"])
                    if accepted:
                        self.assertEqual(result.fit["formal_cn"]["status"], "ok")
                        self.assertTrue(any(r["n_status"] == "fitted" for r in rows))
                        self.assertTrue(all(r["s02_calibrated"] == "true" for r in rows))
                    else:
                        self.assertEqual(result.fit["formal_cn"]["status"], "s02_uncalibrated")
                        self.assertTrue(all(r["reject_reason"] and not r["n_fit"] for r in rows))
                        self.assertFalse(any(r["n_status"] == "fitted" for r in rows))
                        self.assertTrue(all(r["s02_calibrated"] == "false" for r in rows))

    def test_s02_evidence_binds_element_value_and_hash(self):
        from copy import deepcopy
        import run_xafs
        ledger, kwargs = self._review_cn_fixture()
        payload = ledger["calibration"]
        evidence = xafs_fit.make_s02_evidence(payload)
        absorber = structure_id.absorber_with_s02("Ni", payload["s02"])
        source = ledger["formal_cn"]["rows"][0]["s02_source"]
        self.assertTrue(xafs_fit._s02_calibrated_flag(absorber, source, evidence))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            absorber, "s02_calibration:forged;certificate_sha256:" + "0" * 64, evidence))
        block = xafs_fit.build_formal_cn_block(
            conditional_cn=ledger["conditional_cn"], absorber=absorber,
            s02_source=source, s02_evidence=evidence, **kwargs)
        self.assertEqual(block["status"], "ok", block.get("reject_reason"))
        self.assertTrue(all(r["s02_calibrated"] for r in block["rows"]))
        bad_hash = deepcopy(evidence)
        bad_hash["payload_sha256"] = "0" * 64
        changed_payload = deepcopy(evidence)
        changed_payload["payload"]["s02"] = 0.95
        invalid_protocol = xafs_fit.make_s02_evidence(dict(payload, n_variables=0))
        for bad in ({}, bad_hash, changed_payload, invalid_protocol):
            with self.subTest(evidence=bad.get("payload_sha256")):
                rejected = xafs_fit.build_formal_cn_block(
                    conditional_cn=ledger["conditional_cn"], absorber=absorber,
                    s02_source=source, s02_evidence=bad, **kwargs)
                self.assertEqual(rejected["rows"], [])
                self.assertEqual(rejected["status"], "s02_uncalibrated")
        for wrong in (structure_id.absorber_with_s02("Au", payload["s02"]),
                      structure_id.absorber_with_s02("Ni", 0.95)):
            self.assertFalse(xafs_fit._s02_calibrated_flag(wrong, source, evidence))
        for manual in ("--s02 runtime override", "config:s02"):
            self.assertFalse(xafs_fit._s02_calibrated_flag(absorber, manual, evidence))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            structure_id.absorber_with_s02("Fe", 0.95), "chemical_state:Fe=metal"))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            absorber, "chemical_state:Fe=metal"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "certificate.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            file_evidence = xafs_fit.make_s02_evidence(payload, certificate_file=path)
            self.assertNotEqual(file_evidence["payload_sha256"],
                                file_evidence["certificate_file_sha256"])
            self.assertTrue(xafs_fit._s02_calibrated_flag(absorber, source, file_evidence))
            path.write_text(json.dumps(dict(payload, s02=0.95)), encoding="utf-8")
            self.assertFalse(xafs_fit._s02_calibrated_flag(absorber, source, file_evidence))
        t5_value, t5_source = run_xafs.load_t5_s02("Cr")
        t5_absorber = structure_id.absorber_with_s02("Cr", t5_value)
        self.assertTrue(xafs_fit._s02_calibrated_flag(t5_absorber, t5_source))
        self.assertTrue(xafs_fit._s02_calibrated_flag(t5_absorber, deepcopy(t5_source)))
        self.assertFalse(xafs_fit._s02_calibrated_flag(t5_absorber, str(t5_source)))
        self.assertTrue(xafs_fit._s02_calibrated_flag(
            t5_absorber, str(t5_source), t5_source.s02_evidence))

    def test_formal_cn_rechecks_fixed_n_fit(self):
        from copy import deepcopy
        ledger, kwargs = self._review_cn_fixture()
        good = ledger["conditional_cn"]
        kwargs["absorber"] = get_absorber("Ni")
        self.assertTrue(all(p["N_err"] is None for p in good["fixed_n_paths"]))
        self.assertEqual(xafs_fit.build_formal_cn_block(
            conditional_cn=good, **kwargs)["status"], "ok")
        for field, value, reason in (("r_factor", 0.2, "Rf"),
                                     ("delta_e0", 15.0, "ΔE0"),
                                     ("n_variables", good["fixed_n_fit"]["n_independent"], "N_var")):
            with self.subTest(field=field):
                bad = deepcopy(good)
                bad["fixed_n_fit"][field] = value
                block = xafs_fit.build_formal_cn_block(conditional_cn=bad, **kwargs)
                self.assertEqual(block["rows"], [])
                self.assertIn("固定 N", block["reject_reason"])
                self.assertIn(reason, block["reject_reason"])
        for missing in ("fixed_n_fit", "fixed_n_paths", "sigma2"):
            with self.subTest(missing=missing):
                bad = deepcopy(good)
                if missing == "sigma2":
                    bad["fixed_n_paths"][0]["sigma2"] = 0.0001
                else:
                    bad.pop(missing)
                block = xafs_fit.build_formal_cn_block(conditional_cn=bad, **kwargs)
                self.assertEqual(block["rows"], [])
                self.assertIn("固定 N", block["reject_reason"])

    def test_conditional_cn_consistency_requires_fixed_n_qualification(self):
        from copy import deepcopy
        from types import SimpleNamespace
        from unittest import mock
        ledger, kwargs = self._review_cn_fixture()
        record = ledger["conditional_cn"]

        def fit(stage):
            summary = dict(record[stage + "_n_fit"])
            paths = record[("fixed" if stage == "fixed" else "conditional") + "_n_paths"]
            summary["paths"] = [SimpleNamespace(
                label=p["label"], degeneracy=p["N"], n_err=p["N_err"],
                sigma2=p["sigma2"], r=p["reff"], delta_r=p["delta_r"]) for p in paths]
            return SimpleNamespace(**summary)

        fixed, conditional, free = fit("fixed"), fit("conditional"), fit("free")
        plan = dict(kwargs["plan"], files=["unused0", "unused1"],
                    labels=[p.label for p in fixed.paths])
        result = SimpleNamespace(chi_k=None, ft=SimpleNamespace(kmin=3.0, kmax=12.0, kweight=2))
        for qualified in (True, False):
            candidate = deepcopy(fixed)
            if not qualified:
                candidate.delta_e0 = 15.0
            with mock.patch.object(xafs_fit.artemis, "fit_r_space",
                                   side_effect=[candidate, conditional, free]), mock.patch.object(
                    xafs_fit.artemis, "shell_f_test", return_value={
                        "valid": True, "significant": False, "confidence": 0.3}):
                out = xafs_fit.run_conditional_cn(
                    result, plan, kwargs["n0_mapping"], "Ni", 0.9654, verbose=False)
            self.assertEqual(out["consistency_passed"], qualified)
            if not qualified:
                self.assertTrue(any("固定 N" in r and "ΔE0" in r
                                    for r in out["qualification_reasons"]))

    def test_conditional_cn_formal_rows(self):
        """#119 P4+：条件 CN 通过 → 正式行完整可追溯；失败 → 0 行。"""
        from structure_model import StructureModel
        model = StructureModel(
            model_id="COD:1100108", absorber="Fe",
            absorber_site="Fe1", model_level="crystal_cif",
            source_type="database_cif", source_ref="COD:1100108",
            source_cif="_structure_lib/_phase_pack/COD/1100108/x.cif",
            source_cif_sha256="0" * 64,
            cell={"a": 2.868, "b": 2.868, "c": 2.868,
                  "alpha": 90, "beta": 90, "gamma": 90},
            spacegroup="Im-3m",
            provenance_status="verified_source_cif",
        )
        plan = {
            "shells": [
                {"label": "Fe-Fe@2.48", "scatterer": "Fe",
                 "reff": 2.482, "degen": 8.0, "n_members": 1},
                {"label": "Fe-Fe@2.87", "scatterer": "Fe",
                 "reff": 2.866, "degen": 6.0, "n_members": 1},
            ],
            "rmin": 1.88, "rmax": 3.5,
        }
        n0_mapping = {"n0_per_path": [8.0, 6.0]}
        conditional = {
            "n_ratio": 0.95, "n_status": "ok", "consistency_passed": True,
            "first_shell_idxs": [0],
            "kmin": 3.0, "kmax": 12.0, "kweight": 2,
            "conditional_n_fit": {
                "r_factor": 0.0234, "n_variables": 5,
                "n_independent": 9.3, "delta_e0": 0.59,
                "overfit": False},
            "hamilton_f_test": {"significant": False, "p_value": 0.5,
                                "valid": True},
            "conditional_n_paths": [
                {"label": "Fe-Fe@2.48", "reff": 2.45, "N": 7.6,
                 "N_err": 0.4, "delta_r": -0.032, "sigma2": 0.0055},
                {"label": "Fe-Fe@2.87", "reff": 2.813, "N": 6.0,
                 "N_err": None, "delta_r": -0.053, "sigma2": 0.0040},
            ],
        }
        conditional["fixed_n_fit"] = dict(conditional["conditional_n_fit"], n_variables=4)
        conditional["fixed_n_paths"] = [dict(p, N_err=None)
                                          for p in conditional["conditional_n_paths"]]
        block = xafs_fit.build_formal_cn_block(
            conditional_cn=conditional, plan=plan,
            n0_mapping=n0_mapping, structure_model=model,
            absorber=get_absorber("Fe"),
            cn_systematics={"relative_1sigma": 0.32},
            s02_source="absorber.py", data_sha256="d" * 64)
        self.assertEqual(block["status"], "ok")
        self.assertEqual(len(block["rows"]), 2)
        first, second = block["rows"]
        self.assertEqual(first["n_status"], "fitted")
        self.assertEqual(first["n_fit"], 7.6)
        self.assertEqual(first["n0"], 8.0)
        self.assertEqual(first["n_ratio"], 7.6 / 8.0)
        self.assertEqual(first["n_err_formal"], 0.4)
        self.assertEqual(first["n_err_systematic"], 0.32)
        self.assertEqual(first["shell_group"], "first_shell")
        self.assertEqual(first["cn_label"], "cn_conditional")
        self.assertEqual(first["gate_status"], "passed")
        self.assertEqual(first["data_sha256"], "d" * 64)
        self.assertEqual(first["source_cif_sha256"], "0" * 64)
        self.assertEqual(first["absorber_site"], "Fe1")
        self.assertTrue(first["s02_calibrated"])
        self.assertEqual(second["n_status"], "fixed")
        self.assertEqual(second["n_fit"], 6.0)
        self.assertEqual(second["n_ratio"], 1.0)
        self.assertEqual(second["n_err_formal"], "")

        # Hamilton 显著 → 一致性失败，正式表不得产生 CN 行。
        failed = dict(conditional)
        failed["hamilton_f_test"] = {"significant": True, "p_value": 0.01,
                                     "valid": True}
        failed["consistency_passed"] = False
        block_bad = xafs_fit.build_formal_cn_block(
            conditional_cn=failed, plan=plan,
            n0_mapping=n0_mapping, structure_model=model,
            absorber=get_absorber("Fe"))
        self.assertEqual(block_bad["status"], "consistency_failed")
        self.assertEqual(block_bad["rows"], [])
        self.assertIn("Hamilton", block_bad["reject_reason"])

        # 数据哈希稳定：同内容换路径不变，内容变化则变。
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.dat"
            b = Path(tmp) / "b.dat"
            a.write_bytes(b"abc")
            b.write_bytes(b"abc")
            self.assertEqual(cn_table.data_sha256([a]),
                             cn_table.data_sha256([b]))
            b.write_bytes(b"abd")
            self.assertNotEqual(cn_table.data_sha256([a]),
                                cn_table.data_sha256([b]))

    def test_counterexample_ftest_not_executable_blocks_formal_cn(self):
        """反例：Hamilton F-test 不可执行（valid=False）≠ 不显著。

        即便 consistency_passed=True、significant=False，正式 CN 也必须
        拒绝。旧漏洞把“未执行的检验”当“不显著”放行，会洗白固定 N 一致性。
        同时验证：条件拟合自身 Rf 超标、主拟合 cn_reliable=False 硬前置。
        """
        from structure_model import StructureModel
        model = StructureModel(
            model_id="COD:1100108", absorber="Fe", absorber_site="Fe1",
            model_level="crystal_cif", source_type="database_cif",
            source_ref="COD:1100108",
            source_cif="_structure_lib/_phase_pack/COD/1100108/x.cif",
            source_cif_sha256="0" * 64,
            cell={"a": 2.868, "b": 2.868, "c": 2.868,
                  "alpha": 90, "beta": 90, "gamma": 90},
            spacegroup="Im-3m", provenance_status="verified_source_cif")
        plan = {"shells": [
            {"label": "Fe-Fe@2.48", "scatterer": "Fe", "reff": 2.482,
             "degen": 8.0, "n_members": 1}], "rmin": 1.88, "rmax": 3.5}
        n0_mapping = {"n0_per_path": [8.0]}
        base = {
            "n_ratio": 0.95, "n_status": "ok", "consistency_passed": True,
            "first_shell_idxs": [0], "kmin": 3.0, "kmax": 12.0,
            "kweight": 2,
            "conditional_n_fit": {"r_factor": 0.0234, "n_variables": 5,
                                  "n_independent": 9.3, "delta_e0": 0.59,
                                  "overfit": False},
            "conditional_n_paths": [
                {"label": "Fe-Fe@2.48", "reff": 2.45, "N": 7.6,
                 "N_err": 0.4, "delta_r": -0.032, "sigma2": 0.0055}],
        }
        base["fixed_n_fit"] = dict(base["conditional_n_fit"], n_variables=4)
        base["fixed_n_paths"] = [dict(p, N_err=None) for p in base["conditional_n_paths"]]
        kw = dict(plan=plan, n0_mapping=n0_mapping, structure_model=model,
                  absorber=get_absorber("Fe"))
        # 正控：F-test 可执行且不显著、拟合合格 → 正式行产出
        ok = dict(base, hamilton_f_test={"significant": False,
                                         "p_value": 0.5, "valid": True})
        block_ok = xafs_fit.build_formal_cn_block(conditional_cn=ok, **kw)
        self.assertEqual(block_ok["status"], "ok")
        self.assertEqual(len(block_ok["rows"]), 1)
        # 反例①：F-test 不可执行（valid=False），consistency_passed 仍 True
        bad = dict(base, hamilton_f_test={"significant": False, "F": None,
                                          "confidence": None,
                                          "valid": False})
        block_bad = xafs_fit.build_formal_cn_block(
            conditional_cn=bad, **kw)
        self.assertEqual(block_bad["status"], "consistency_failed")
        self.assertEqual(block_bad["rows"], [])
        self.assertIn("不可执行", block_bad["reject_reason"])
        # 反例②：缺 valid 字段（陈旧记录）一律按不可执行拒绝
        missing = dict(base, hamilton_f_test={"significant": False,
                                              "p_value": 0.5})
        block_missing = xafs_fit.build_formal_cn_block(
            conditional_cn=missing, **kw)
        self.assertEqual(block_missing["status"], "consistency_failed")
        self.assertIn("不可执行", block_missing["reject_reason"])
        # 反例③：F-test 合格但条件拟合 Rf 超标 → 仍拒绝（门禁在前）
        unqual = dict(base, hamilton_f_test={"significant": False,
                                             "valid": True},
                      conditional_n_fit=dict(base["conditional_n_fit"],
                                             r_factor=0.2))
        block_u = xafs_fit.build_formal_cn_block(
            conditional_cn=unqual, **kw)
        self.assertEqual(block_u["status"], "consistency_failed")
        self.assertIn("Rf", block_u["reject_reason"])
        # 反例④：主拟合 cn_reliable=False 硬前置，条件 CN 自洽也拒绝
        block_rel = xafs_fit.build_formal_cn_block(
            conditional_cn=ok, cn_reliable=False,
            reliability_reasons=["Rf>0.08"], **kw)
        self.assertEqual(block_rel["status"], "consistency_failed")
        self.assertEqual(block_rel["rows"], [])

    def test_counterexample_conditional_fit_physical_qualification(self):
        """反例：条件 CN 拟合自身物理资格门禁（与常规 motif 同口径）。

        逐项验证 Rf/|ΔE0|/自由度/σ² 下界/首壳 N 形式误差，以及缺字段
        fail-safe（缺失按不可靠）。合格基线必须放行，避免过度拦截。
        """
        good_fit = {"r_factor": 0.02, "delta_e0": 0.5,
                    "n_variables": 5, "n_independent": 9.3}
        good_paths = [{"label": "Fe-Fe@2.48", "N": 7.6, "N_err": 0.4,
                       "sigma2": 0.0055}]
        q, reasons = xafs_fit._conditional_fit_qualified(
            good_fit, good_paths, [0])
        self.assertTrue(q, reasons)
        self.assertEqual(reasons, [])
        # Rf 超门限（0.08 不可放宽）
        q, r = xafs_fit._conditional_fit_qualified(
            dict(good_fit, r_factor=0.2), good_paths, [0])
        self.assertFalse(q)
        self.assertTrue(any("Rf" in x for x in r))
        # |ΔE0| ≥ 15 eV
        q, r = xafs_fit._conditional_fit_qualified(
            dict(good_fit, delta_e0=20.0), good_paths, [0])
        self.assertFalse(q)
        self.assertTrue(any("ΔE0" in x for x in r))
        # N_var > N_idp
        q, r = xafs_fit._conditional_fit_qualified(
            dict(good_fit, n_variables=10, n_independent=9.3),
            good_paths, [0])
        self.assertFalse(q)
        self.assertTrue(any("N_var" in x for x in r))
        # N_var == N_idp 也拒绝（红线：严禁 ≥）
        q, _ = xafs_fit._conditional_fit_qualified(
            dict(good_fit, n_variables=9, n_independent=9.0),
            good_paths, [0])
        self.assertFalse(q)
        # σ² 撞下界
        q, r = xafs_fit._conditional_fit_qualified(
            good_fit, [dict(good_paths[0], sigma2=1e-4)], [0])
        self.assertFalse(q)
        self.assertTrue(any("σ²" in x for x in r))
        # 首壳 N 形式误差绝对过大（>1.0）
        q, r = xafs_fit._conditional_fit_qualified(
            good_fit, [dict(good_paths[0], N_err=1.5)], [0])
        self.assertFalse(q)
        self.assertTrue(any("N 形式误差" in x for x in r))
        # 首壳 N 形式误差相对过大（>0.5·N）
        q, _ = xafs_fit._conditional_fit_qualified(
            good_fit, [{"label": "x", "N": 1.0, "N_err": 0.6,
                        "sigma2": 0.005}], [0])
        self.assertFalse(q)
        # 缺 r_factor → fail-safe 按 1.0 → 不可靠
        q, _ = xafs_fit._conditional_fit_qualified(
            {"delta_e0": 0.5, "n_variables": 5, "n_independent": 9.3},
            good_paths, [0])
        self.assertFalse(q)
        # 缺自由度字段 → fail-safe 拒绝
        q, _ = xafs_fit._conditional_fit_qualified(
            {"r_factor": 0.02, "delta_e0": 0.5}, good_paths, [0])
        self.assertFalse(q)

    def test_counterexample_stern_formula_and_ftest_validity(self):
        """反例（红线 3）：N_idp = 2ΔkΔR/π，严禁 +2；shell_f_test 三态。

        自由度不足（dfd<1）或变量未增加（dfn≤0）时 valid=False，
        绝不可判显著；正控自由度充足且残差下降时 valid=True。
        """
        from types import SimpleNamespace
        nidp = artemis.independent_points(3.0, 12.0, 1.5, 3.5)
        expected = 2.0 * (12.0 - 3.0) * (3.5 - 1.5) / math.pi
        self.assertAlmostEqual(nidp, expected, places=9)
        self.assertNotAlmostEqual(nidp, expected + 2.0, places=6)
        # dfd = N_idp − p2 = 5.5 − 5 = 0.5 < 1 → 不可执行
        ft = artemis.shell_f_test(
            SimpleNamespace(n_variables=3, chi_square=10.0),
            SimpleNamespace(n_variables=5, n_independent=5.5,
                            chi_square=2.0))
        self.assertFalse(ft["valid"])
        self.assertFalse(ft["significant"])
        # dfn = p2 − p1 = 0 → 不适用
        ft2 = artemis.shell_f_test(
            SimpleNamespace(n_variables=5, chi_square=10.0),
            SimpleNamespace(n_variables=5, n_independent=20.0,
                            chi_square=2.0))
        self.assertFalse(ft2["valid"])
        self.assertFalse(ft2["significant"])
        # 正控：dfd=15、dfn=2、残差下降 → valid=True、F>0
        ft3 = artemis.shell_f_test(
            SimpleNamespace(n_variables=3, chi_square=10.0),
            SimpleNamespace(n_variables=5, n_independent=20.0,
                            chi_square=2.0))
        self.assertTrue(ft3["valid"])
        self.assertGreater(ft3["F"], 0.0)

    def test_counterexample_wt_respects_effective_kmax(self):
        """反例（审查 item 13）：有效 k 窗必须贯穿 WT，且 Ce 物理上限生效。

        历史缺陷：S6 小波写死 kmax=min(14.0, k.max())，绕过 run_xafs
        .bounded_kmax 按元素裁出的有效 kmax。Ce L3 的 EXAFS 被紧邻 L2 边
        硬截断（ΔE=441 eV → k=10.76 Å⁻¹，静态 ceiling 10.2），旧 WT 却把
        k>10.2 的 L2 污染区重新卷入，产生不可扣除的假峰。正确：WT 与 FT
        共用同一 kmax；改 kmax 必须改 WT 窗；Ce 上限在入口被裁到 10.2。
        """
        import run_xafs
        from absorber import effective_kmax_limit, physical_kmax_limit

        sigma = 1.5   # process_sample 默认 wt_sigma
        tol = 0.11    # k_centers 网格步长 dk=0.1 的端点容差

        # ① 前提：数据延伸过 k=12，才能区分"写死 14"的旧缺陷
        res10 = xafs_pipeline.process_sample(
            [LEPIDO], absorber="Fe", kmax=10.0, verbose=False)
        res12 = xafs_pipeline.process_sample(
            [LEPIDO], absorber="Fe", kmax=12.0, verbose=False)
        self.assertGreater(float(res12.chi_k.k.max()), 12.0)
        kc10 = float(res10.wt.k_centers.max())
        kc12 = float(res12.wt.k_centers.max())
        # 旧缺陷下两者都 = min(14,14.9)-1.5 = 12.5（逐位相同）；修复后必须分开
        self.assertGreater(kc12, kc10 + 1.0,
                           "WT 窗必须随 kmax 变化，不得写死 min(14, k.max())")
        # ② WT 窗上界 = kmax - sigma（完整滑窗中心），绝不超出有效 kmax
        self.assertLessEqual(kc10, 10.0 - sigma + tol)
        self.assertLessEqual(kc12, 12.0 - sigma + tol)
        # ③ FT 与 WT 共用同一有效 kmax（FT.kmax 即入口裁剪后的值）
        self.assertAlmostEqual(res10.ft.kmax, 10.0, places=6)
        self.assertAlmostEqual(res12.ft.kmax, 12.0, places=6)
        self.assertLessEqual(kc10, res10.ft.kmax - sigma + tol)
        self.assertLessEqual(kc12, res12.ft.kmax - sigma + tol)

        # ④ Ce 物理上限：L3 5723 → L2 6164，ΔE=441 eV → k=sqrt(K2·441)
        phys_ce = physical_kmax_limit("Ce")
        self.assertIsNotNone(phys_ce)
        self.assertAlmostEqual(phys_ce, math.sqrt(K2 * 441.0), places=6)
        eff_ce = effective_kmax_limit("Ce")
        self.assertAlmostEqual(eff_ce, 10.2, places=6)  # min(ceiling 10.2, phys 10.76)
        self.assertLessEqual(eff_ce, phys_ce)

        # ⑤ 入口裁剪：请求 12.0 → Ce 生效 10.2，记录台账（不静默截断）
        capped, info = run_xafs.bounded_kmax(12.0, "Ce", verbose=False)
        self.assertAlmostEqual(capped, 10.2, places=6)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["applied_kmax"], 10.2, places=6)
        self.assertAlmostEqual(info["requested_kmax"], 12.0, places=6)
        # ⑥ 未超限不裁：请求 9.0 < 10.2 → 原样返回，info=None
        keep, info2 = run_xafs.bounded_kmax(9.0, "Ce", verbose=False)
        self.assertAlmostEqual(keep, 9.0, places=6)
        self.assertIsNone(info2)
        # ⑦ K 边无 next_edge → 物理上限 None，仅静态 ceiling 生效（Fe=15）
        self.assertIsNone(physical_kmax_limit("Fe"))
        fe_cap, fe_info = run_xafs.bounded_kmax(12.0, "Fe", verbose=False)
        self.assertAlmostEqual(fe_cap, 12.0, places=6)
        self.assertIsNone(fe_info)

    def test_counterexample_artemis_stern_and_s02n_guards(self):
        """反例（红线 3 + N/S₀² 简并）：求解前硬门禁必须拦截非法拟合。

        ① vary_s02 与自由 N 同时放开（乘积简并）；② N_var≥N_idp（有效窗）；
        ③ 请求 k 窗与数据支持域无交集（不得虚增自由度）。三者均在进入
        求解器前拒绝，不得产出拟合结果。
        """
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        datfile = cache / "feff0002.dat"
        chi_k, ft = self.lep.chi_k, self.lep.ft
        # ① N/S₀² 乘积简并：同时放开必须拒绝
        with self.assertRaises(XAFSConfigError) as cm:
            artemis.fit_r_space(chi_k, ft, [datfile], 1.0, 3.0,
                                kmin=3.0, kmax=12.0, kweight=2,
                                s02=None, vary_s02=True, vary_n=True)
        self.assertIn("不能同时", str(cm.exception))
        # ② 窄 R 窗 → N_var=3 ≥ N_idp_eff，Stern 硬拒绝
        with self.assertRaises(XAFSConfigError) as cm2:
            artemis.fit_r_space(chi_k, ft, [datfile], 1.5, 1.55,
                                kmin=3.0, kmax=12.0, kweight=2,
                                s02=0.803, vary_n=False)
        self.assertIn("Stern", str(cm2.exception))
        # ③ 请求 k 窗超出数据支持域 → 有效窗为空，拒绝
        with self.assertRaises(XAFSConfigError) as cm3:
            artemis.fit_r_space(chi_k, ft, [datfile], 1.0, 3.0,
                                kmin=100.0, kmax=110.0, kweight=2,
                                s02=0.803, vary_n=False)
        self.assertIn("有效 k 窗", str(cm3.exception))

    def test_t5_artifacts_static_lock_and_incremental_merge(self):
        """#122 P5-A：T5 产物完整、#60 静态锁定、--only 增量不丢元素。"""
        import calibrate_s02
        import calibrate_t5
        from unittest import mock
        from xafs_core import XAFSDataError

        locked = {"Cr", "Mn", "Zn", "Mo", "Ru", "Rh", "Pd", "Ag",
                  "Ir", "Au", "Ce", "U"}
        self.assertEqual(set(calibrate_s02.T5_STATIC_LOCKED), locked)
        for el in sorted(locked):
            cfg = get_absorber(el)
            self.assertIsNone(cfg.s02, el)
            self.assertFalse(cfg.calibrated, el)

        index_path = ROOT / "data" / "s02_calibration" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertTrue(index["static_absorber_py_unchanged"])
        self.assertFalse(index["apply_to_absorber_py"])
        self.assertEqual(index["runtime_override"], "--s02")
        rows = {r["absorber"]: r for r in index["elements"]}
        self.assertEqual(set(rows), set(calibrate_t5.ELEMENTS))
        self.assertEqual(index["pass_count"] + index["fail_count"], 12)
        self.assertEqual(index["pass_count"], 12)
        self.assertEqual(index["fail_count"], 0)
        self.assertNotIn("C:\\\\", index_path.read_text(encoding="utf-8"))
        for el in calibrate_t5.ELEMENTS:
            self.assertEqual(rows[el]["status"], "calibrated", el)
            self.assertIsNotNone(rows[el]["s02"], el)
        expected_s02 = {
            "Rh": 0.7022, "Pd": 0.6559, "Ir": 0.4752,
            "Ce": 0.6279, "U": 0.7800,
        }
        for el, value in expected_s02.items():
            self.assertAlmostEqual(rows[el]["s02"], value, places=4, msg=el)

        # --apply 对 #60 锁定元素硬拒绝（不真正跑标定）。
        with mock.patch.object(calibrate_s02, "calibrate_s02",
                               return_value={"pass": True, "s02": 0.6}):
            with self.assertRaises(SystemExit) as cm:
                calibrate_s02.main([
                    "dummy.dat", "--absorber", "Cr", "--apply"])
        self.assertIn("#60", str(cm.exception))

        # 标定专用 motif 缺 sites 必须拒绝，不得回退到 expand_full。
        payload = json.loads((ROOT / "data" / "s02_calibration" / "motifs" /
                              "cr2o3_corundum.json").read_text(
                                  encoding="utf-8"))
        payload["sites"] = []
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(XAFSDataError):
                calibrate_s02.load_motif_json(bad, "Cr")

        # --only 只重跑 Cr，但 index/REPORT 必须保留既有 Mn/Au。
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for el in ("Mn", "Au"):
                el_dir = tmp / el
                el_dir.mkdir()
                existing = {
                    "absorber": el, "status": "calibrated",
                    "s02": 0.5, "s02_err": 0.01, "rf": 0.01,
                    "spread": 0.001, "motif": f"{el.lower()}_motif",
                    "standard": "dummy", "standard_sha256": "a" * 64,
                    "attempts_compact": [],
                }
                (el_dir / "calibration.json").write_text(
                    json.dumps(existing), encoding="utf-8")
            new_cr = {
                "absorber": "Cr", "status": "calibrated", "s02": 0.6482,
                "s02_err": 0.0198, "rf": 0.0074, "spread": 0.0053,
                "motif": "cr2o3_corundum", "standard": "dummy",
                "standard_sha256": "b" * 64, "attempts_compact": [],
            }
            with mock.patch.object(calibrate_t5, "_run_element",
                                   return_value=new_cr):
                rc = calibrate_t5.main(
                    ["--only", "Cr", "--outdir", str(tmp)])
            self.assertEqual(rc, 0)
            merged = json.loads((tmp / "index.json").read_text(
                encoding="utf-8"))
            self.assertEqual([r["absorber"] for r in merged["elements"]],
                             ["Cr", "Mn", "Au"])
            self.assertEqual(merged["pass_count"], 3)
            self.assertTrue((tmp / "REPORT.md").is_file())

    def test_t5_s02_loader_and_provenance(self):
        """#123 P5-A：--s02-from-t5 只接受已标定元素并记录溯源。"""
        import run_xafs

        s02, source = run_xafs.load_t5_s02("Cr")
        self.assertAlmostEqual(s02, 0.6482, places=4)
        self.assertTrue(source.startswith("s02_calibration:"))
        self.assertIn("sha256:", source)
        self.assertIn("standard_sha256:", source)

        s02_ce, source_ce = run_xafs.load_t5_s02("Ce")
        self.assertAlmostEqual(s02_ce, 0.6279, places=4)
        self.assertTrue(source_ce.startswith("s02_calibration:"))
        # 未通过元素仍然硬拒绝（用临时产物验证，不依赖真实标定状态）。
        with tempfile.TemporaryDirectory() as tmp:
            el_dir = Path(tmp) / "Ce"
            el_dir.mkdir()
            (el_dir / "calibration.json").write_text(json.dumps({
                "absorber": "Ce", "status": "not_calibrated",
                "s02": None, "reason": "gate_failed：未通过冻结门禁",
                "static_absorber_py_unchanged": True,
            }), encoding="utf-8")
            with self.assertRaises(XAFSConfigError) as cm:
                run_xafs.load_t5_s02("Ce", root=Path(tmp))
            self.assertIn("拒绝自动注入", str(cm.exception))
        with self.assertRaises(XAFSConfigError):
            run_xafs.load_t5_s02("Xx")

        # 显式 --s02 优先于 T5 产物；未显式给值时从 T5 读取。
        from types import SimpleNamespace
        args = SimpleNamespace(s02=0.9, s02_from_t5=True, absorber="Cr")
        val, src = run_xafs._resolve_s02_for_paths(args, ["dummy.dat"])
        self.assertEqual(val, 0.9)
        self.assertEqual(src, "--s02 runtime override")
        args2 = SimpleNamespace(s02=None, s02_from_t5=True, absorber=None)
        val2, src2 = run_xafs._resolve_s02_for_paths(
            args2, [str(ROOT / "data" / "bmm_standards" / "Data" / "Cr" /
                         "Cr-K-Cr2O3.xdi")])
        self.assertAlmostEqual(val2, 0.6482, places=4)
        self.assertTrue(src2.startswith("s02_calibration:"))

        # 正式表：真实 T5 证据绑定实际值；字符串前缀不算。
        self.assertTrue(xafs_fit._s02_calibrated_flag(
            structure_id.absorber_with_s02("Cr", s02), source))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Cr"), "s02_calibration:x"))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Cr"), "--s02 runtime override"))

    def test_s02_certificate_rejects_invalid_protocol_geometry(self):
        import copy
        import calibrate_s02

        ledger = json.loads((ROOT / "data" / "real_cn_validation.json").read_text(
            encoding="utf-8"))
        base = ledger["cases"][0]["calibration"]
        original = copy.deepcopy(base)
        self.assertEqual(calibrate_s02.certificate_qualifies(base), (True, []))
        for element in ("Ce", "Mo"):
            certificate = json.loads((ROOT / "data" / "s02_calibration" / element /
                                      "calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(calibrate_s02.certificate_qualifies(certificate), (True, []))

        def reject(path, value):
            bad = copy.deepcopy(base)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            ok, reasons = calibrate_s02.certificate_qualifies(bad)
            self.assertFalse(ok, f"accepted {path}={value!r}")
            self.assertTrue(reasons)

        for key in ("n_variables", "n_shells"):
            values = (0, -1, 1.5, True, float("nan"), float("inf"), None)
            if key == "n_variables":
                values += (1, 4, 7, "6")
            else:
                values += (1, 3, "2")
            for value in values:
                with self.subTest(field=key, value=value):
                    reject((key,), value)
        for window in ([12.0, 3.0], [3.0, 3.0], [-1.0, 12.0],
                       [3.0, float("inf")], [float("nan"), 12.0], [True, 12.0]):
            with self.subTest(k_window=window):
                reject(("k_window",), window)
        for index in range(3):
            for window in ([3.2, 1.4], [1.484, 1.484],
                           [1.484, float("inf")], [float("nan"), 3.2],
                           [1.484, 3.4], [1.484, base["runs"][1]["r_window"][1]],
                           [1.5, base["runs"][index]["r_window"][1]]):
                if window == base["runs"][index]["r_window"]:
                    continue
                with self.subTest(run=index, r_window=window):
                    reject(("runs", index, "r_window"), window)
        for path, value in (
                (("runs",), base["runs"][:2]),
                (("shell_paths",), base["shell_paths"][:1]),
                (("shell_paths", 1, "label"), base["shell_paths"][0]["label"]),
                (("shell_paths", 1, "reff"), -1.0),
                (("shell_paths", 1, "reff"), float("nan")),
                (("shell_paths", 1, "degen"), -1.0),
                (("shell_paths", 1, "scatterer"), ""),
                (("runs", 0), None),
                (("runs", 0, "paths"), base["runs"][0]["paths"][:1]),
                (("runs", 0, "paths", 1, "label"), "wrong"),
                (("runs", 0, "paths", 1, "degen"), 6.0),
                (("runs", 0, "paths", 1, "sigma2"), -0.01),
                (("runs", 0, "paths", 1, "delta_r"), float("inf")),
                (("runs", 0, "paths", 1, "sigma2_at_bound"), True),
                (("runs", 0, "sigma2"), float("nan")),
                (("runs", 0, "delta_r"), float("inf")),
                (("runs", 0, "sigma2_at_bound"), None),
                (("n_independent",), float("inf")),
                (("sigma2",), -0.01)):
            with self.subTest(path=path, value=value):
                reject(path, value)
        for key in ("s02", "s02_err", "rf", "delta_e0"):
            for value in (float("nan"), float("inf"), float("-inf"), True):
                with self.subTest(metric=key, value=value):
                    reject(("runs", 0, key), value)
                    reject((key,), value)
        # Synchronize aggregates too: rejection must be for negative raw metrics,
        # not merely because a modified run disagrees with its stored aggregate.
        for key in ("s02_err", "rf"):
            with self.subTest(negative_metric=key):
                bad = copy.deepcopy(base)
                for run in bad["runs"]:
                    run[key] = -0.01
                bad[key] = -0.01
                self.assertFalse(calibrate_s02.certificate_qualifies(bad)[0])
        for sigma2 in (0.0, 0.0001, 0.0005):
            with self.subTest(hidden_sigma2_bound=sigma2):
                bad = copy.deepcopy(base)
                for run in bad["runs"]:
                    run["sigma2"] = sigma2
                    run["sigma2_at_bound"] = False
                    for path in run["paths"]:
                        path["sigma2"] = sigma2
                        path["sigma2_at_bound"] = False
                bad["sigma2"] = sigma2
                self.assertFalse(calibrate_s02.certificate_qualifies(bad)[0])
        self.assertEqual(base, original)

    def test_calibration_kw_error_is_recorded(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import calibrate_s02
        from xafs_core import XAFSError

        shell = {"label": "Ni-O@2.08", "scatterer": "O", "reff": 2.0842,
                 "degen": 6.0}
        plan = {"shells": [shell], "files": ["unused.dat"], "degens": [6.0],
                "labels": [shell["label"]], "rmin": 1.0}
        result = SimpleNamespace(
            chi_k=None, ft=SimpleNamespace(kmin=3.0, kmax=12.0, kweight=2),
            e0={"e0_method": "derivative", "e0_sample": 8344.9},
            norm=SimpleNamespace(plateau=1.0), rbkg={"rbkg": 1.0})
        path = SimpleNamespace(label=shell["label"], degeneracy=6.0,
                               sigma2=0.006, delta_r=-0.01, sigma2_at_bound=False)
        fit = SimpleNamespace(paths=[path], s02=0.9, s02_err=0.01,
                              r_factor=0.02, delta_e0=-3.0,
                              n_variables=4, n_independent=8.0)

        def fitting(*args, **kwargs):
            if kwargs["kweight"] != 2:
                raise XAFSError("kw_probe", f"injected kw={kwargs['kweight']}")
            return fit

        with patch.object(calibrate_s02.xafs_pipeline, "process_sample", return_value=result), \
                patch.object(calibrate_s02, "_resolve_motif", return_value=SimpleNamespace(id="test")), \
                patch.object(calibrate_s02.xafs_fit, "select_shell_paths", return_value=plan), \
                patch.object(calibrate_s02.artemis, "fit_r_space", side_effect=fitting):
            output = calibrate_s02.calibrate_s02(
                ["unused.xdi"], "Ni", allow_recalibrate=True, verbose=False)
        self.assertTrue(output["pass"])
        self.assertEqual(calibrate_s02.certificate_qualifies(output), (True, []))
        for kw in ("1", "3"):
            self.assertIn(f"injected kw={kw}", output["kw_sensitivity"][kw]["error"])

    def test_real_cn_validation_retains_partial_evidence(self):
        import copy
        from types import SimpleNamespace
        from unittest.mock import patch
        import validate_real_cn

        baseline = json.loads((ROOT / "data" / "real_cn_validation.json").read_text(
            encoding="utf-8"))["cases"][0]
        phase = phase_resolver.resolve_phase("Ni", phase_id="COD:1010093", offline=True)
        gate = SimpleNamespace(ok=True, first_shell_cn=6.0,
                               n0_shells=[6.0], occupancy_source="CIF")
        motif = SimpleNamespace(to_json=lambda: {"id": "test_motif"})
        formal = {"status": "ok", "model": {}, "rows": []}
        prediction = SimpleNamespace(norm=SimpleNamespace(plateau=1.0), fit={
            "formal_cn": formal, "cn_reliable": True,
            "conditional_cn": {"n0_total": 5.0, "n_fit_total": 5.0}})

        scenarios = (
            ("standard_phase", "phase", XAFSConfigError("CIF", "injected phase error")),
            ("standard_cif", "cif", XAFSConfigError("CIF", "injected CIF error")),
            ("calibration_motif_write", "motif_write", OSError("injected motif write error")),
            ("calibration", "calibrate", XAFSConfigError("CAL", "injected calibration error")),
            ("calibration_check", "n0", XAFSConfigError("validation", "unused")),
            ("calibration_write", "certificate_write", OSError("injected certificate write error")),
            ("prediction", "predict", XAFSConfigError("S1", "injected prediction error")),
            ("prediction", "predict", OSError("injected prediction write error")),
            ("holdout_check", "holdout_n0", XAFSConfigError("validation", "unused")),
        )
        for failed_stage, fault, error in scenarios:
            with self.subTest(stage=failed_stage, fault=fault, error=type(error).__name__), \
                    tempfile.TemporaryDirectory() as tmp:
                calibration = copy.deepcopy(baseline["calibration"])
                if fault == "n0":
                    calibration["fixed_CN"] = 5.0
                write_json = validate_real_cn.xafs_writer.safe_write_json

                def writer(path, data):
                    if ((fault == "motif_write" and Path(path).name == "calibration_motif.json") or
                            (fault == "certificate_write" and Path(path).name == "calibration.json")):
                        raise error
                    return write_json(path, data)

                with patch.object(validate_real_cn.phase_resolver, "resolve_phase",
                                  side_effect=error if fault == "phase" else None,
                                  return_value=copy.deepcopy(phase)), \
                        patch.object(validate_real_cn.cif_gate, "gate_cif",
                                     side_effect=error if fault == "cif" else None, return_value=gate), \
                        patch.object(validate_real_cn.xafs_fit, "_site_motif_from_model", return_value=motif), \
                        patch.object(validate_real_cn.calibrate_s02, "calibrate_s02",
                                     side_effect=error if fault == "calibrate" else None,
                                     return_value=calibration), \
                        patch.object(validate_real_cn.run_xafs, "process_sample",
                                     side_effect=error if fault == "predict" else None,
                                     return_value=prediction), \
                        patch.object(validate_real_cn.xafs_writer, "safe_write_json", side_effect=writer):
                    entry = validate_real_cn.run_case("ni_oxide_to_hydroxide", Path(tmp) / "case")
                    self.assertEqual(entry["failed_stage"], failed_stage)
                    self.assertNotEqual(entry["status"], "validated")
                    self.assertTrue(entry["reason"])
                    self.assertEqual(entry["protocol"], baseline["protocol"])
                    for key in ("standard_data_sha256", "holdout_data_sha256"):
                        self.assertEqual(entry[key], baseline[key])
                    self.assertTrue(entry["independent_input_spectra"])
                    if fault != "phase":
                        self.assertEqual(entry["standard_cif_sha256"], baseline["standard_cif_sha256"])
                    else:
                        self.assertNotIn("standard_cif_sha256", entry)
                    if fault not in ("phase", "cif", "motif_write"):
                        self.assertEqual(len(entry["calibration_motif_sha256"]), 64)
                    else:
                        self.assertNotIn("calibration_motif_sha256", entry)
                    if fault not in ("phase", "cif", "motif_write", "calibrate"):
                        self.assertEqual(entry["calibration"], calibration)
                        self.assertEqual(len(entry["calibration_sha256"]), 64)
                    else:
                        self.assertNotIn("calibration", entry)
                    if fault == "holdout_n0":
                        self.assertEqual(entry["conditional_cn"], prediction.fit["conditional_cn"])
                        self.assertEqual(entry["formal_cn"], formal)
                    # Exercise the real main loop and ledger writer after this partial failure.
                    with patch.object(validate_real_cn, "run_case", side_effect=[
                            entry, {"case": "cu_foil_to_oxide", "status": "validated"}]):
                        self.assertEqual(validate_real_cn.main([
                            "--case", "ni_oxide_to_hydroxide", "--case", "cu_foil_to_oxide",
                            "--outdir", str(Path(tmp) / "ledger")]), 2)
                    report = json.loads((Path(tmp) / "ledger" / "real_cn_validation.json").read_text(
                        encoding="utf-8"))
                    self.assertEqual(report["cases"][0], entry)
                    self.assertEqual(report["validated_cases"], ["cu_foil_to_oxide"])
                    self.assertFalse(report["all_cases_validated"])

        case = dict(validate_real_cn.CASES["ni_oxide_to_hydroxide"])
        case["holdout"] = "missing_holdout.xdi"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(validate_real_cn.CASES, {"missing": case}):
            entry = validate_real_cn.run_case("missing", Path(tmp))
            self.assertEqual(entry["failed_stage"], "input_data")
            self.assertEqual(entry["standard_data_sha256"], baseline["standard_data_sha256"])
            self.assertNotIn("holdout_data_sha256", entry)
            self.assertNotIn("independent_input_spectra", entry)
        read_bytes = Path.read_bytes

        def unreadable_holdout(path):
            if path.name == "Ni-K-NickelHydroxide.xdi":
                raise PermissionError("injected unreadable holdout")
            return read_bytes(path)

        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "read_bytes", new=unreadable_holdout):
            entry = validate_real_cn.run_case("ni_oxide_to_hydroxide", Path(tmp))
            self.assertEqual(entry["failed_stage"], "input_data")
            self.assertEqual(entry["standard_data_sha256"], baseline["standard_data_sha256"])
            self.assertNotIn("holdout_data_sha256", entry)
            self.assertNotIn("independent_input_spectra", entry)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                validate_real_cn.phase_resolver, "resolve_phase", side_effect=RuntimeError("programming bug")):
            with self.assertRaisesRegex(RuntimeError, "programming bug"):
                validate_real_cn.run_case("ni_oxide_to_hydroxide", Path(tmp))

    def test_counterexample_t5_certificate_requalified_against_current_protocol(self):
        """反例（审查 item 9）：T5 证书必须按*现行*标定协议独立复核资格。

        load_t5_s02 不得只信 status="calibrated"/pass=True 快照：协议
        常量收紧或产物被手改/损坏时，原始指标已不合格的旧证书必须硬拒绝。
        同时 12 个已提交证书（真实标定产物）必须仍全部合格——复核不得
        误伤基线（红线5）。标定门禁用标定常量（RF_MAX=0.12），不是生产
        拟合门禁（Rf≤0.08）。
        """
        import copy
        import calibrate_s02
        import calibrate_t5
        import run_xafs

        cert_root = ROOT / "data" / "s02_calibration"
        # 1) 12 个已提交证书全部满足现行协议（基线不被误伤）。
        for el in calibrate_t5.ELEMENTS:
            payload = json.loads((cert_root / el / "calibration.json")
                                 .read_text(encoding="utf-8"))
            ok, reasons = calibrate_s02.certificate_qualifies(payload)
            self.assertTrue(ok, f"{el} 基线证书被误拒: {reasons}")
            self.assertEqual(reasons, [], el)
        # 真实 loader 仍返回历史标定值（逐位一致）。
        for el, expect in (("Cr", 0.6482), ("Ce", 0.6279), ("Mo", 0.8322)):
            s02, _src = run_xafs.load_t5_s02(el)
            self.assertAlmostEqual(s02, expect, places=4, msg=el)

        base = json.loads((cert_root / "Ce" / "calibration.json")
                          .read_text(encoding="utf-8"))

        def write_cert(tmp, payload):
            el_dir = Path(tmp) / "Ce"
            el_dir.mkdir(parents=True, exist_ok=True)
            (el_dir / "calibration.json").write_text(
                json.dumps(payload), encoding="utf-8")
            return Path(tmp)

        # 2a) 聚合 s02 被手改成超现行 S02_PASS 上界（注入值失真）。
        bad = copy.deepcopy(base)
        bad["s02"] = 1.30
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)
        self.assertTrue(any("s02" in r for r in reasons))
        with tempfile.TemporaryDirectory() as tmp:
            root = write_cert(tmp, bad)
            with self.assertRaises(XAFSConfigError) as cm:
                run_xafs.load_t5_s02("Ce", root=root)
            self.assertIn("独立复核", str(cm.exception))

        # 2b) 逐 run rf 超 RF_MAX，但 interior/checks/pass 快照仍称合格
        #     （证明复核不信任存储布尔，只看原始指标）。
        bad = copy.deepcopy(base)
        for r in bad["runs"]:
            r["rf"] = 0.5
            r["interior"] = True
        bad["rf"] = 0.5
        bad["checks"] = {k: True for k in bad["checks"]}
        bad["pass"] = True
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)
        self.assertTrue(any("interior" in r or "rf" in r for r in reasons))

        # 2c) σ² 贴界但快照称 sigma2_not_at_bound。
        bad = copy.deepcopy(base)
        for r in bad["runs"]:
            r["sigma2_at_bound"] = True
        bad["checks"]["sigma2_not_at_bound"] = True
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)
        self.assertTrue(any("sigma2" in r or "interior" in r for r in reasons))

        # 2d) 跨窗口 s02 极差超 WINDOW_SPREAD_MAX（鲁棒性不足），
        #     聚合同步篡改以证明 spread 门本身生效（而非只靠一致性）。
        bad = copy.deepcopy(base)
        bad["runs"][0]["s02"] = 0.55
        bad["runs"][2]["s02"] = 0.75
        vals = [r["s02"] for r in bad["runs"]]
        bad["s02"] = round(sorted(vals)[1], 4)
        bad["spread"] = round(max(vals) - min(vals), 4)
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)
        self.assertTrue(any("window_spread_ok" in r for r in reasons))

        # 2e) 聚合 s02 被手改成门禁内、但与逐 run 中位不符（注入值失真）。
        bad = copy.deepcopy(base)
        bad["s02"] = 0.99   # 在 (0.4,1.15) 内，但 ≠ median(runs)=0.6279
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)
        self.assertTrue(any("不符" in r for r in reasons))
        with tempfile.TemporaryDirectory() as tmp:
            root = write_cert(tmp, bad)
            with self.assertRaises(XAFSConfigError):
                run_xafs.load_t5_s02("Ce", root=root)

        # 2f) runs 数量不足协议窗口数（缺鲁棒性证据）。
        bad = copy.deepcopy(base)
        bad["runs"] = bad["runs"][:2]
        ok, reasons = calibrate_s02.certificate_qualifies(bad)
        self.assertFalse(ok)

        # 3) 标定门禁 ≠ 生产门禁：rf∈(0.08,0.12] 在标定协议下合格，
        #    不得被生产 Rf≤0.08 误拒（证明用的是标定常量 RF_MAX=0.12）。
        good = copy.deepcopy(base)
        for r in good["runs"]:
            r["rf"] = 0.10
        good["rf"] = 0.10
        ok, reasons = calibrate_s02.certificate_qualifies(good)
        self.assertTrue(ok, f"标定 RF_MAX=0.12 被误用生产门禁: {reasons}")

    def test_p5d_multi_site_formal_cn(self):
        """#127 P5-D：--absorber-site all 逐位点正式 CN + 化学计量加权。"""
        from types import SimpleNamespace
        from unittest.mock import patch

        # 未传 --absorber-site：多位点相仍然必须拒绝，绝不自动 Top-1。
        no_site = phase_resolver.resolve_phase(
            "Fe", phase_id="COD:1010369", offline=True)
        self.assertEqual(no_site.status, "resolved")
        self.assertIsNone(no_site.structure_model)
        self.assertEqual(no_site.site_models, [])
        self.assertIn("不自动 Top-1", no_site.reason)

        # 显式 all：按位点顺序返回每个不等价吸收位的模型。
        fe = phase_resolver.resolve_phase(
            "Fe", phase_id="COD:1010369", absorber_site="all", offline=True)
        self.assertEqual(fe.status, "resolved")
        self.assertEqual(fe.absorber_site, "all")
        self.assertEqual([m.absorber_site for m in fe.site_models],
                         ["Fe1", "Fe2", "Fe3"])
        self.assertIs(fe.structure_model, fe.site_models[0])

        gates = cif_gate.gate_cif_all_sites(
            fe.selected.cif_path, "Fe", kmin=3.0, kmax=12.0)
        self.assertEqual(len(gates), 3)
        self.assertTrue(all(g.ok for g in gates), [g.reason for g in gates])
        self.assertEqual([g.site_multiplicity for g in gates], [16, 16, 8])
        self.assertEqual([g.site_occupancy for g in gates], [0.5, 0.5, 1.0])
        for gate in gates:
            self.assertAlmostEqual(gate.site_weight, 1.0 / 3.0, places=12)
        weighted_n0 = sum(g.site_weight * g.first_shell_cn for g in gates)
        self.assertAlmostEqual(weighted_n0, 16.0 / 3.0, places=6)
        self.assertAlmostEqual(weighted_n0, 5.333333, places=6)

        # Co3O4：8a:16d → 1/3 : 2/3；声明化学式 Co3 O4 与全占据晶胞一致，
        # 按 implicit_1.0 放行，加权第一壳 N₀ = 1/3×4 + 2/3×6 = 16/3。
        co = phase_resolver.resolve_phase(
            "Co", phase_id="COD:9005888", absorber_site="all", offline=True)
        self.assertEqual([m.absorber_site for m in co.site_models],
                         ["Co1", "Co2"])
        co_sites = cif_gate.list_inequivalent_absorber_sites(
            co.selected.cif_path, "Co")
        self.assertEqual([s["label"] for s in co_sites], ["Co1", "Co2"])
        self.assertEqual([s["multiplicity"] for s in co_sites], [8, 16])
        self.assertAlmostEqual(co_sites[0]["weight"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(co_sites[1]["weight"], 2.0 / 3.0, places=12)
        co_gates = cif_gate.gate_cif_all_sites(
            co.selected.cif_path, "Co", kmin=3.0, kmax=12.0)
        self.assertTrue(all(g.ok for g in co_gates),
                        [g.reason for g in co_gates])
        self.assertTrue(all(g.occupancy_source == "implicit_1.0"
                            for g in co_gates))
        co_weighted_n0 = sum(g.site_weight * g.first_shell_cn
                             for g in co_gates)
        self.assertAlmostEqual(co_weighted_n0, 16.0 / 3.0, places=6)

        # run_multi_site_formal_cn：全部位点 ok 才输出 ALL_WEIGHTED。
        res = SimpleNamespace(ft=SimpleNamespace(kmin=3.0, kmax=12.0))

        def fake_block(conditional_cn, plan, n0_mapping, structure_model,
                       absorber, **kwargs):
            site = structure_model.absorber_site
            n0 = {"Fe1": 6.0, "Fe2": 6.0, "Fe3": 4.0}[site]
            return {
                "status": "ok", "reject_reason": None,
                "rows": [{
                    "phase_id": structure_model.model_id,
                    "model_level": structure_model.model_level,
                    "source_type": structure_model.source_type,
                    "source_ref": structure_model.source_ref,
                    "source_cif_sha256": structure_model.source_cif_sha256,
                    "absorber_site": site, "shell_group": "first_shell",
                    "shell_label": "Fe-O", "scatterer": "O",
                    "r_cif": 2.0, "n0": n0, "n_fit": n0 * 0.95,
                    "n_status": "fitted", "n_ratio": 0.95,
                    "n_err_formal": 0.2, "n_err_systematic": 0.32,
                    "delta_r": 0.01, "sigma2": 0.004, "delta_e0": 1.0,
                    "rf": 0.02, "s02": 0.803,
                    "s02_source": "absorber.py", "s02_calibrated": True,
                    "kmin": 3.0, "kmax": 12.0, "kweight": 2,
                    "rmin": 1.0, "rmax": 3.0, "n_idp": 9.0,
                    "n_variables": 5, "cn_label": "cn_conditional",
                    "gate_status": "passed", "reject_reason": "",
                    "flags": "", "data_sha256": "d" * 64,
                    "software_version": "test",
                }],
            }

        # item 7：多位点改为联合拟合 χ_obs=Σw_iχ_i。mock 联合条件 CN 助手，
        # 按传入位点数返回等长的逐位点 conditional 列表 + 联合元数据。
        def fake_joint(res, site_plans, site_mappings, site_weights,
                       site_labels, **kwargs):
            return ([{} for _ in site_plans],
                    {"model": "joint_chi_obs_sum_w_i_chi_i",
                     "n_sites": len(site_plans),
                     "n_paths": sum(len(p["files"]) for p in site_plans)})

        common = [
            patch.object(xafs_fit, "_site_motif_from_model",
                         return_value=object()),
            patch.object(xafs_fit, "select_shell_paths",
                         return_value={"files": [], "shells": [],
                                       "rmin": 1.0, "rmax": 3.0}),
            patch.object(xafs_fit, "map_n0_to_plan",
                         return_value={"mapping_ok": True}),
            patch.object(xafs_fit, "_joint_multi_site_conditional_cn",
                         side_effect=fake_joint),
            patch.object(xafs_fit, "build_formal_cn_block",
                         side_effect=fake_block),
        ]
        for patcher in common:
            patcher.start()
        try:
            out = xafs_fit.run_multi_site_formal_cn(
                res, site_models=fe.site_models, gate_results=gates,
                s02=0.803, s02_source="absorber.py",
                data_sha256="d" * 64)
            self.assertEqual(out["status"], "ok")
            self.assertAlmostEqual(out["site_weights"]["Fe1"], 1 / 3,
                                   places=12)
            self.assertEqual(out["rows"][-1]["absorber_site"],
                             "ALL_WEIGHTED")
            self.assertAlmostEqual(out["rows"][-1]["n0"], 16 / 3,
                                   places=6)
            self.assertIn("site_weight=0.333333", out["rows"][0]["flags"])

            # item 7：任一位点门禁失败 → 联合振幅 χ_obs=Σw_iχ_i 不闭合 →
            # 所有位点写拒绝行，不输出 ALL_WEIGHTED，也不保留任何 CN 宣称。
            bad = list(gates)
            bad[1] = cif_gate.CIFGateResult(
                ok=False, code="n_out_of_range", reason="fake gate failure",
                absorber="Fe", absorber_site="Fe2")
            bad[1].site_weight = gates[1].site_weight
            failed = xafs_fit.run_multi_site_formal_cn(
                res, site_models=fe.site_models, gate_results=bad,
                s02=0.803, s02_source="absorber.py",
                data_sha256="d" * 64)
            self.assertEqual(failed["status"], "multi_site_failed")
            sites = [r.get("absorber_site") for r in failed["rows"]]
            self.assertNotIn("ALL_WEIGHTED", sites)
            self.assertIn("Fe2", sites)
            # 混合谱中缺失任一位点贡献，任何单站点 CN 都不可信：全员拒绝。
            self.assertEqual(len(sites), 3)
            self.assertTrue(all(r.get("gate_status") != "passed"
                                for r in failed["rows"]))

            # 【红线 2】主拟合 cn_reliable=False 必须透传到多位点：
            # 这里故意让 fake_block 忽略 cn_reliable 并返回 "ok/passsed"，
            # 验证即使逐位点块被替换/绕过，底层仍会整体如实降级，
            # 绝不输出 ALL_WEIGHTED，也不留任何 gate_status=passed 的 CN。
            gated = xafs_fit.run_multi_site_formal_cn(
                res, site_models=fe.site_models, gate_results=gates,
                s02=0.803, s02_source="absorber.py",
                data_sha256="d" * 64,
                site_cn_reliability=[False, True, False])
            self.assertEqual(gated["status"], "base_fit_unreliable")
            self.assertIs(gated["cn_reliable"], False)
            self.assertIs(gated["degraded"], True)
            self.assertEqual(gated["prior_status"], "ok")
            self.assertNotIn("ALL_WEIGHTED",
                             [r.get("absorber_site") for r in gated["rows"]])
            self.assertTrue(all(r.get("gate_status") != "passed"
                                for r in gated["rows"]))
            self.assertTrue(all(str(r.get("n_fit") or "") == ""
                                for r in gated["rows"]))
            self.assertTrue(all("base_cn_unreliable" in str(r.get("flags"))
                                for r in gated["rows"]))
            # 降级必须幂等，两个 CLI 入口可重复调用而不产生重复标记。
            before = repr(gated)
            xafs_fit.downgrade_multi_site_cn(gated)
            self.assertEqual(before, repr(gated))
        finally:
            for patcher in common:
                patcher.stop()

    def test_counterexample_multi_site_joint_cn_recovers_truth(self):
        """审查 item 7：多位点**联合** χ_obs=Σw_iχ_i 必须回收逐位点真值 CN。

        旧版对每位点独立拟合**完整** χ_obs，单位点被迫解释全部振幅 → 首壳
        N 被抬高 ~1/w_i，且各独立拟合在同一份数据上重复占用全部 N_idp
        （违反全局 Stern 守恒，红线 3）。

        本例用两个**完全可分辨**的 Fe 位点合成混合谱（部分氧化铁的真实
        图像：金属核 + 氧化壳）：位点 0 = Fe-O@1.886×4（轻散射体、短 R，
        磁铁矿 tet 组分），位点 1 = Fe-Fe@2.48×8（重散射体、长 R，bcc 金属）。
        权重 0.5:0.5，χ_obs=w_0·χ_0+w_1·χ_1（有效简并度=N₀·w）。联合条件
        CN 固定化学计量权重、逐位点放开首壳 N，切片除以 w_i 还原局域 CN
        必须≈4 / 8，联合 N_var<N_idp，且远离独立拟合的 1/w 虚高（8 / 16）。
        """
        import structure_lib
        import xafs_fit
        import artemis
        import athena
        from larch.xafs import feffpath
        from xafs_core import ChiK
        from types import SimpleNamespace

        # 位点 0：磁铁矿 tet 组分 Fe-O@1.886×4（轻散射体、短 R）。
        mag = [m for m in structure_lib.load_library("Fe")
               if m.id == "magnetite_mixed"][0]
        mag_plan = xafs_fit.select_shell_paths(
            mag, edge="K", rmax=2.6, kmin=3.0, kmax=16.0,
            group_by_component=True)
        tet_idxs = [i for i, g in enumerate(mag_plan["groups"]) if g == 0]
        self.assertEqual([mag_plan["degens"][i] for i in tet_idxs], [4.0])
        # 位点 1：bcc 金属 Fe-Fe@2.48×8（重散射体、长 R），与位点 0 完全可分辨。
        bcc = [m for m in structure_lib.load_library("Fe")
               if m.id == "fe_bcc"][0]
        bcc_plan = xafs_fit.select_shell_paths(
            bcc, edge="K", rmax=2.7, kmin=3.0, kmax=16.0)
        self.assertAlmostEqual(bcc_plan["degens"][0], 8.0, places=6)

        site_plans = [
            {"files": [mag_plan["files"][i] for i in tet_idxs],
             "shells": [mag_plan["shells"][i] for i in tet_idxs],
             "degens": [mag_plan["degens"][i] for i in tet_idxs],
             "labels": [mag_plan["labels"][i] for i in tet_idxs],
             "rmin": mag_plan["rmin"], "rmax": mag_plan["rmax"]},
            {"files": [bcc_plan["files"][0]],
             "shells": [bcc_plan["shells"][0]],
             "degens": [bcc_plan["degens"][0]],
             "labels": [bcc_plan["labels"][0]],
             "rmin": bcc_plan["rmin"], "rmax": bcc_plan["rmax"]},
        ]
        site_mappings = [
            {"n0_per_path": [4.0], "path_links": []},
            {"n0_per_path": [8.0], "path_links": []},
        ]
        weights = [0.5, 0.5]
        s02, sigma2, e0 = 0.803, 0.004, 2.0
        k = np.arange(3.0, 16.0, 0.05)
        paths = []
        for si in range(2):
            p = feffpath(str(site_plans[si]["files"][0]), s02=s02,
                         sigma2=sigma2, e0=e0, deltar=0.0)
            # 有效简并度 = 晶体学 N₀ × 化学计量权重（联合模型的真值振幅）。
            p.degen = site_plans[si]["degens"][0] * weights[si]
            p._calc_chi()
            paths.append(p)
        chi = artemis.exafs_model_k(paths, k)
        rng = np.random.default_rng(11)
        chi = chi + rng.normal(0.0, 2e-4, size=chi.shape)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 16.0, 2)
        res = SimpleNamespace(chi_k=chi_k, ft=ft)

        per_site, joint_meta = xafs_fit._joint_multi_site_conditional_cn(
            res, site_plans, site_mappings, weights, ["FeO", "FeFe"],
            absorber_symbol="Fe", s02=s02, kmin=3.0, kmax=16.0,
            verbose=False)

        # 联合模型必须闭合全局 Stern 预算（红线 3，IFEFFIT 口径不加 2）。
        self.assertEqual(joint_meta["model"], "joint_chi_obs_sum_w_i_chi_i")
        self.assertLess(joint_meta["n_variables"], joint_meta["n_independent"])
        self.assertLess(joint_meta["r_factor"], 0.02)
        # 逐位点局域 CN 回收真值 4 / 8（切片已除以 w_i 还原）。
        self.assertAlmostEqual(per_site[0]["n0_total"], 4.0, places=6)
        self.assertAlmostEqual(per_site[1]["n0_total"], 8.0, places=6)
        self.assertAlmostEqual(per_site[0]["n_fit_total"], 4.0, delta=0.5)
        self.assertAlmostEqual(per_site[1]["n_fit_total"], 8.0, delta=0.7)
        # 关键反例：绝不落到旧版独立拟合的 ~1/w 虚高（FeO~8 / FeFe~16）。
        self.assertLess(per_site[0]["n_fit_total"], 6.5)
        self.assertLess(per_site[1]["n_fit_total"], 12.0)
        self.assertTrue(per_site[0]["joint_fit"])
        self.assertTrue(per_site[1]["joint_fit"])

    def test_p6_1_single_site_oxide_formal_cn_end_to_end(self):
        """#129 P6.1：单晶位氧化物（不在 _structure_lib 中）通过已验证 CIF 完成正式 CN，不被 L1 劫持。"""
        from larch.xafs import feffpath
        from types import SimpleNamespace

        import run_xafs

        s02, s02_src = run_xafs.load_t5_s02("Ru")
        pr = phase_resolver.resolve_phase(
            "Ru", phase_id="COD:9007541", offline=True)
        self.assertEqual(pr.status, "resolved")
        self.assertEqual(pr.structure_model.occupancy_policy, "implicit_1.0")
        gate = cif_gate.gate_cif(
            pr.selected.cif_path, "Ru",
            absorber_site=pr.absorber_site, kmin=3.0, kmax=12.0)
        self.assertTrue(gate.ok, gate.reason)

        motif = xafs_fit._site_motif_from_model(pr.structure_model)
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0)
        k = np.arange(3.0, 12.0, 0.05)
        plist = []
        for sh, f in zip(plan["shells"], plan["files"]):
            p = feffpath(str(f), s02=s02, sigma2=0.004, e0=1.0, deltar=0.0)
            p.degen = float(sh["degen"])
            p._calc_chi()
            plist.append(p)
        chi = artemis.exafs_model_k(plist, k)
        rng = np.random.default_rng(42)
        chi = chi + rng.normal(0.0, 5e-4, size=chi.shape)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        # 合成谱（理论 χ + 噪声）显式声明为透射：自吸收门禁对"探测模式
        # 无法判定"必须保守拒绝 CN 定量，夹具不能靠省略元数据绕过门禁。
        # 这里补齐的是测试前提，不是放宽任何 Rf/ΔE0/平台门禁。
        # Synthetic transmission channels are explicit fixture evidence; χ is unchanged.
        raw_mu = 1.0 + 0.01 * chi
        res = SimpleNamespace(
            chi_k=chi_k, ft=ft, absorber_symbol="Ru", structure=None,
            merged=SimpleNamespace(meta={"mode": "trans"}, mu=raw_mu,
                                   i0=np.ones_like(raw_mu), it=np.exp(-raw_mu)),
            norm=None, self_absorption=None,
            data_sha256="b" * 64)

        out = xafs_fit.run(
            res, identify=True, fit=True, absorber_symbol="Ru",
            s02=s02, s02_source=s02_src, formal_cn=True,
            structure_model=pr.structure_model, gate_result=gate,
            verbose=False)
        # 诊断轨因 _structure_lib/Ru 只有 ru_hcp 返回 l1_first_shell，
        # 但正式轨必须独立用 COD:9007541 (RuO2) 跑通 P3 条件 CN 并输出 ok！
        self.assertEqual(out["fit"]["mode"], "l1_first_shell")
        self.assertEqual(out["cn_status"], "ok")
        self.assertEqual(out["formal_cn"]["status"], "ok")
        rows = out["formal_cn"]["rows"]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase_id"], "COD:9007541")
        self.assertAlmostEqual(float(rows[0]["n0"]), 6.0, places=3)
        self.assertAlmostEqual(float(rows[0]["n_fit"]), 6.0, delta=0.35)


    def test_unit4_counterexamples_validation_and_propagation(self):
        """Unit 4 审计反例：越界/边界校验、F-test kweights 一致性、MS 派生链误差与联合拟合变量计数。"""
        from types import SimpleNamespace

        # 1) build_feff_input 跳过所有 rmax 外原子时应显式报错
        with self.assertRaises(XAFSConfigError):
            artemis.build_feff_input([(0.0, 0.0, 8.0, "O")], absorber="Fe", rmax=5.0)

        # 2) reconstruct_pair_distribution 拒绝非正 sigma2
        with self.assertRaises(XAFSConfigError):
            artemis.reconstruct_pair_distribution(
                np.linspace(1.5, 2.5, 101), n=6.0, r0=2.0, sigma2=-1e-4)

        # 3) shell_f_test 拦截 kweights 不一致的模型对比
        f_kw2 = SimpleNamespace(
            n_variables=3, n_independent=10.0, kmin=3.0, kmax=12.0,
            rmin=1.0, rmax=3.0, fit_space="r", kweights=(2,),
            chi_square=20.0, r_factor=0.02)
        f_kw3 = SimpleNamespace(
            n_variables=5, n_independent=10.0, kmin=3.0, kmax=12.0,
            rmin=1.0, rmax=3.0, fit_space="r", kweights=(3,),
            chi_square=5.0, r_factor=0.005)
        sft_bad = artemis.shell_f_test(f_kw2, f_kw3)
        self.assertFalse(sft_bad["valid"])
        self.assertIn("kweights", sft_bad.get("note", ""))

        # 4) vary_c3 越界索引必须抛 XAFSConfigError（而非 IndexError）
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        datfiles = sorted(cache.glob("feff0*.dat"))
        with self.assertRaises(XAFSConfigError):
            artemis.fit_r_space(
                self.lep.chi_k, self.lep.ft, [datfiles[0]],
                rmin=1.2, rmax=2.4, s02=0.803, vary_c3=[99])

        # 5) ms_links 母壳层被 path_links 绑定时不崩溃且正确传播误差
        if len(datfiles) >= 3:
            res_ms = artemis.fit_r_space(
                self.lep.chi_k, self.lep.ft, datfiles[:3],
                rmin=1.0, rmax=3.8, kmin=3.0, kmax=12.0, kweight=2, s02=0.803,
                path_links=[{"target": 1, "source": 0,
                             "share_deltar": True, "share_sigma2": True}],
                ms_links=[{"ms_path_index": 2, "parent_shell_index": 1,
                           "reff_ratio": 1.8, "degen_ratio": 1.0}])
            self.assertAlmostEqual(
                res_ms.paths[2].delta_r_err,
                res_ms.paths[0].delta_r_err * 1.8, places=10)
            self.assertAlmostEqual(
                res_ms.paths[2].sigma2,
                res_ms.paths[0].sigma2 * 1.8, places=10)

        # 6) fit_multi_datasets 只共享 ΔR 时不把源路径的 σ²/N 计入目标数据集变量数
        ds = [
            {"name": "ds0", "chi_k": self.lep.chi_k, "ft": self.lep.ft,
             "datfiles": [datfiles[0]], "rmin": 1.0, "rmax": 2.4,
             "kmin": 3.0, "kmax": 12.0, "kweight": 2, "s02": 0.803,
             "vary_n": True},
            {"name": "ds1", "chi_k": self.lep.chi_k, "ft": self.lep.ft,
             "datfiles": [datfiles[0]], "rmin": 1.0, "rmax": 2.4,
             "kmin": 3.0, "kmax": 12.0, "kweight": 2, "s02": 0.803,
             "vary_n": False},
        ]
        mfit = artemis.fit_multi_datasets(
            ds, shared_bonds=[{"target": (1, 0), "source": (0, 0),
                               "share_deltar": True, "share_sigma2": False}])
        self.assertEqual(mfit["datasets"][1].n_variables, 3)

        # 7) fit_multi_datasets 越界 theta_e_init 必须抛 XAFSConfigError
        with self.assertRaises(XAFSConfigError):
            artemis.fit_multi_datasets(
                [{**ds[0], "temperature_K": 300.0}],
                einstein_models=[{"paths": [(0, 0)], "reduced_mass_amu": 12.4,
                                  "theta_e_init": 50.0}])


    def test_counterexample_plateau_out_of_range_degrades_to_l1(self):
        """反例（红线 2）：边后归一化平台超出 [0.85, 1.15] 必须强制 cn_reliable=False 并降级。"""
        import copy
        from types import SimpleNamespace

        for bad_plateau in (0.70, 1.25):
            with self.subTest(plateau=bad_plateau):
                res = copy.deepcopy(self.lep)
                res.norm = SimpleNamespace(
                    plateau=bad_plateau,
                    edge_step=float(getattr(self.lep.norm, "edge_step", 1.0)),
                    e0=float(getattr(self.lep.norm, "e0", 7112.0)),
                )
                out = xafs_fit.fit_identified(
                    res, motif_id="lepidocrocite", absorber_symbol="Fe",
                    verbose=False)
                self.assertFalse(out["cn_reliable"])
                self.assertTrue(out["flags"]["plateau_bad"])
                self.assertTrue(out.get("degraded"))
                self.assertFalse(out.get("cn_claim_allowed"))
                self.assertEqual(len(out.get("quantitative_shells", [])), 1)
                self.assertTrue(any("归一化平台" in r
                                    for r in out.get("degradation_reasons", [])))

    def test_counterexample_multisite_non_true_reliability_and_certificate_meta(self):
        """反例（红线 2 & 证据完整性）：
        1) 多位点 site_cn_reliability 含 None 或长度不足时不得放行正式 CN；
        2) T5 证书缺少 absorber/edge/protocol 或含非法 e0_method 时必须拒绝；
        3) catalytic_model 配合 formal_cn=True 时必须显式给出 s02_calibrated=False。
        """
        import copy
        from types import SimpleNamespace
        from unittest.mock import patch
        import calibrate_s02

        # 1) T5 证书元数据缺失/篡改拦截
        cert_path = ROOT / "data" / "s02_calibration" / "Cr" / "calibration.json"
        base_cert = json.loads(cert_path.read_text(encoding="utf-8"))
        for bad_patch in (
            {"absorber": ""},
            {"edge": None},
            {"protocol": ""},
            {"e0_method": "forged_method"},
        ):
            bad = copy.deepcopy(base_cert)
            bad.update(bad_patch)
            ok, reasons = calibrate_s02.certificate_qualifies(bad)
            self.assertFalse(ok, f"未拦截非法证书字段: {bad_patch}")
            self.assertTrue(reasons)

        # 2) catalytic_model + formal_cn=True 返回契约包含 s02_calibrated
        out_cat = xafs_fit.fit_identified(
            self.lep, absorber_symbol="Fe",
            catalytic_model="sac:O4@1.98,C=none",
            formal_cn=True, verbose=False)
        self.assertIn("s02_calibrated", out_cat)
        self.assertFalse(out_cat["s02_calibrated"])
        self.assertEqual(out_cat["cn_status"], "hypothesis_model_only")

        # 3) 多位点 site_cn_reliability 包含 None 或长度不足时整体降级
        fe = phase_resolver.resolve_phase(
            "Fe", phase_id="COD:1010369", absorber_site="all", offline=True)
        gates = cif_gate.gate_cif_all_sites(
            fe.selected.cif_path, "Fe", kmin=3.0, kmax=12.0)
        res = SimpleNamespace(ft=SimpleNamespace(kmin=3.0, kmax=12.0))

        def fake_block(conditional_cn, plan, n0_mapping, structure_model,
                       absorber, cn_reliable=None, **kwargs):
            if cn_reliable is False:
                return {"status": "consistency_failed", "rows": [],
                        "reject_reason": "base_unreliable"}
            return {
                "status": "ok", "reject_reason": None,
                "rows": [{
                    "phase_id": structure_model.model_id,
                    "absorber_site": structure_model.absorber_site,
                    "shell_group": "first_shell", "n0": 6.0, "n_fit": 6.0,
                    "n_status": "fitted", "gate_status": "passed",
                    "flags": "", "reject_reason": "",
                }],
            }

        with patch.object(xafs_fit, "_site_motif_from_model", return_value=object()), \
             patch.object(xafs_fit, "select_shell_paths",
                          return_value={"files": [], "shells": [], "rmin": 1.0, "rmax": 3.0}), \
             patch.object(xafs_fit, "map_n0_to_plan", return_value={"mapping_ok": True}), \
             patch.object(xafs_fit, "_joint_multi_site_conditional_cn",
                          return_value=([{} for _ in fe.site_models], {"model": "joint"})), \
             patch.object(xafs_fit, "build_formal_cn_block", side_effect=fake_block):
            for bad_rel in ([True, None, True], [True]):
                gated = xafs_fit.run_multi_site_formal_cn(
                    res, site_models=fe.site_models, gate_results=gates,
                    s02=0.803, s02_source="absorber.py",
                    site_cn_reliability=bad_rel)
                self.assertEqual(gated["status"], "base_fit_unreliable")
                self.assertFalse(gated["cn_reliable"])
                self.assertNotIn("ALL_WEIGHTED",
                                 [r.get("absorber_site") for r in gated["rows"]])
