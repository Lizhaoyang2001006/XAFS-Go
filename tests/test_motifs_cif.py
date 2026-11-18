# -*- coding: utf-8 -*-
"""Unit 3：晶体学与 Motif 建模（CIF/结构识别/SAC-DAC/相解析）

纯搬运拆分：测试方法体与原 regression_test.py 逐行一致，断言未做任何修改。
"""
from __future__ import annotations

from tests.fixtures import *  # noqa: F401,F403
from tests.fixtures import Fixtures, _find  # noqa: F401

# 历史测试体里有 3 处直接用 Path(__file__) 定位 data/（原文件在仓库根）。
# 拆到 tests/ 后模块位置变了，把 __file__ 指回等价的根入口文件，
# 保证测试方法体逐字节不变（不改任何断言或路径写法）。
__file__ = str(ROOT / "regression_test.py")


class MotifsCIFTests(Fixtures):

    def test_feff_hole_card_for_l_edges(self):
        sites = [(0.0, 0.0, 0.0, "Pt"), (2.77, 0.0, 0.0, "Pt")]
        expected = {"K": 1, "L1": 2, "L2": 3, "L3": 4}
        for edge, idx in expected.items():
            inp = artemis.build_feff_input(
                sites, absorber="Pt", rmax=3.0, edge=edge)
            self.assertIn(f"HOLE {idx}   1.0", inp)
        with self.assertRaises(XAFSConfigError):
            artemis.build_feff_input(sites, absorber="Pt", edge="M5")

    def test_fit_motif_returns_shell_cn(self):
        fit_out = self.lep_fit["fit"]
        self.assertEqual(fit_out["mode"], "motif")
        self.assertEqual(fit_out["motif"], "lepidocrocite")
        self.assertTrue(fit_out["coverage_ok"])
        self.assertTrue(fit_out["cn_reliable"])
        fit = fit_out["fit"]
        self.assertFalse(fit.overfit)
        first = fit.paths[0]
        self.assertGreater(first.degeneracy, 4.5)
        self.assertLess(first.degeneracy, 8.0)
        self.assertGreater(first.r, 1.90)
        self.assertLess(first.r, 2.15)
        self.assertLess(fit.r_factor, 0.10)

    def test_goethite_motif_is_confident(self):
        res = xafs_pipeline.process_sample(
            [GOETHITE], absorber="Fe", strict=False, verbose=False)
        sid = structure_id.identify_result(
            res, absorber_symbol="Fe", verbose=False)
        self.assertEqual(sid["status"], "confident")
        self.assertEqual(sid["ranked"][0]["id"], "goethite")
        self.assertTrue(sid["coverage_ok"])
        self.assertEqual(sid["uncovered_peaks"], [])

    def test_magnetite_mixed_is_confident(self):
        """磁铁矿多位点识别（曾出问题的案例）。

        历史：只有 16d 八面体位点时 score=0.145、实验峰 1.35/3.712
        无理论峰解释（覆盖率失败→ partial）。补 8a 四面体位点与
        mixture 联合模型后，联合模型 score=0.083 成为 Top-1，且覆盖
        按"复合谱 ∪ 各分量分别解释"通过——tet 分量解释 1.35/3.712，
        oct 分量解释 2.608。
        """
        res = xafs_pipeline.process_sample(
            [MAGNETITE], absorber="Fe", strict=False, verbose=False)
        sid = structure_id.identify_result(
            res, absorber_symbol="Fe", verbose=False)
        self.assertEqual(sid["status"], "confident")
        self.assertEqual(sid["ranked"][0]["id"], "magnetite_mixed")
        self.assertTrue(sid["coverage_ok"])
        self.assertEqual(sid["uncovered_peaks"], [])
        self.assertGreater(sid["confidence"]["margin"], 0.03)

    def test_leave_one_phase_out_no_false_confidence(self):
        """阶段5：留一物相盲测——真物相缺库时必须如实降级，不得错误授信。

        把样品真物相从结构库剔除后重跑 S6.5：最优候选复数谱残差必然超过
        CONFIDENT_SCORE，判定须落 no_match，而不是 confident 冒充次优物相
        （lepidocrocite 剔除后 goethite 多型体不得被误判；magnetite 全部表征
        剔除后特征 3.712 Å 峰无人解释）。同时校验识别台账完整记录
        best_score/margin/status/uncovered_peaks——识别错误与错误授信可审计。
        既有基线保持：全库下两者仍 confident、top1=真物相、无未解释峰。
        """
        orig_load = structure_lib.load_library

        def identify_with(res, exclude):
            if not exclude:
                return structure_id.identify(
                    res.ft, res.chi_k.k, absorber_symbol="Fe",
                    kmin=res.ft.kmin, kmax=res.ft.kmax,
                    kweight=res.ft.kweight, verbose=False)

            def filtered(absorber):
                return [m for m in orig_load(absorber)
                        if m.id not in exclude]
            structure_lib.load_library = filtered
            try:
                return structure_id.identify(
                    res.ft, res.chi_k.k, absorber_symbol="Fe",
                    kmin=res.ft.kmin, kmax=res.ft.kmax,
                    kweight=res.ft.kweight, verbose=False)
            finally:
                structure_lib.load_library = orig_load

        mag_res = xafs_pipeline.process_sample(
            [MAGNETITE], absorber="Fe", strict=False, verbose=False)
        cases = [
            ("lepidocrocite", self.lep, {"lepidocrocite"}),
            ("magnetite_mixed", mag_res,
             {m.id for m in orig_load("Fe") if "magnetite" in m.id}),
        ]
        for true_id, res, exclude in cases:
            full = identify_with(res, set())
            self.assertEqual(full["status"], "confident",
                             f"{true_id} 全库基线应 confident")
            self.assertEqual(full["ranked"][0]["id"], true_id)
            self.assertEqual(full["uncovered_peaks"], [])

            loo = identify_with(res, exclude)
            ranked_ids = [d["id"] for d in loo["ranked"]]
            self.assertNotIn(true_id, ranked_ids,
                             "真物相已剔库，不应再出现在候选表")
            self.assertNotEqual(loo["status"], "confident",
                                f"{true_id} 缺库后仍 confident = 错误授信")
            self.assertEqual(loo["status"], "no_match")
            self.assertGreater(loo["confidence"]["best_score"],
                               structure_id.CONFIDENT_SCORE,
                               "缺库后最优残差须超门禁，故判 no_match")
            conf = loo["confidence"]
            for key in ("best_score", "second_score", "margin",
                        "score_limit", "margin_limit"):
                self.assertIn(key, conf)
            self.assertEqual(conf["score_limit"], structure_id.CONFIDENT_SCORE)
            self.assertEqual(conf["margin_limit"],
                             structure_id.CONFIDENT_MARGIN)
            self.assertTrue(loo["uncovered_peaks"],
                            f"{true_id} 缺库后应有未解释峰（假设空间不完备）")
        self.assertIs(structure_lib.load_library, orig_load,
                      "monkeypatch 须还原，不污染后续测试")

    def test_mixture_shell_selection(self):
        """多位点吸收的壳层分割。

        磁铁矿 8a/16d 的两个子壳层 Fe-O 1.886 / 2.060 Å 只差 0.17 Å，
        单位点容差 0.20 Å 会把它们错并成一层；mixture 的简并度按
        位点权重缩放（tet O×4·⅓=1.33，oct O×6·⅔=4.0）。
        """
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "magnetite_mixed"][0]
        plan = xafs_fit.select_shell_paths(motif, rmax=3.5,
                                           kmin=3.0, kmax=12.0)
        first, second, third = plan["shells"][:3]
        self.assertEqual(first["label"], "Fe-O@1.89")
        self.assertAlmostEqual(first["degen"], 4.0 / 3.0, places=2)
        self.assertEqual(second["label"], "Fe-O@2.06")
        self.assertAlmostEqual(second["degen"], 4.0, places=2)
        self.assertEqual(third["label"], "Fe-Fe@2.97")
        self.assertAlmostEqual(third["degen"], 4.0, places=2)
        self.assertLessEqual(plan["merge_tol"], 0.10,
                             "mixture 必须收紧合并容差，否则子壳层被吞")

    def test_mixture_weight_recovery(self):
        """mixture 相分数回收（--fit-weights 的核心数学契约）。

        磁铁矿 tet/oct 第一壳 Fe-O 1.886/2.060 Å 用 0.25/0.75 权重
        合成 χ。拟合固定每路径的**晶体学简并度**（4/6）、固定 σ²，
        放开 softmax 权重 + ΔR + ΔE₀。无噪声时权重必须回收真值，
        有效简并度随之等于 N_crystal·w（1.0 / 4.5）——回收不了说明
        相分数只是把 N 换个写法，不能用于物相定量。
        """
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "magnetite_mixed"][0]
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmax=2.6, kmin=3.0, kmax=12.0,
            group_by_component=True)
        self.assertEqual(plan["group_names"],
                         ["magnetite_tet", "magnetite_oct"])
        self.assertEqual(plan["groups"], [0, 1])
        self.assertEqual(plan["degens"], [4.0, 6.0],
                         "权重模式必须保留未乘权重的晶体学简并度")
        self.assertEqual(plan["weight_degenerate_pairs"], [])

        true_w = np.array([0.25, 0.75])
        s02, sigma2, e0 = 0.80, 0.004, 2.0
        k = np.arange(3.0, 12.0, 0.05)
        from larch.xafs import feffpath
        paths = []
        for i, f in enumerate(plan["files"]):
            p = feffpath(str(f), s02=s02, sigma2=sigma2, e0=e0, deltar=0.0)
            p.degen = plan["degens"][i] * true_w[plan["groups"][i]]
            p._calc_chi()
            paths.append(p)
        chi = artemis.exafs_model_k(paths, k)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.zeros_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        fit = artemis.fit_r_space(
            chi_k, ft, plan["files"],
            rmin=plan["rmin"], rmax=plan["rmax"],
            kmin=3.0, kmax=12.0, kweight=2, s02=s02,
            vary_delta_e0=True, vary_n=False,
            vary_sigma2=False, sigma2_init=sigma2,
            path_degen=plan["degens"], path_labels=plan["labels"],
            path_groups=plan["groups"],
            path_group_names=plan["group_names"],
            vary_weights=True, weight_init=[0.5, 0.5])

        self.assertFalse(fit.overfit)
        self.assertLess(fit.r_factor, 0.02, "无噪声合成数据应近完美复现")
        self.assertIsNotNone(fit.mixture_weights)
        got = [w["weight"] for w in fit.mixture_weights]
        self.assertAlmostEqual(sum(got), 1.0, places=9)
        for kk in range(2):
            self.assertLess(abs(got[kk] - true_w[kk]), 0.03,
                            f"权重回收偏差过大: {got}")
            err = fit.mixture_weights[kk]["weight_err"]
            self.assertTrue(np.isfinite(err))
            self.assertLess(err, 0.03)
            self.assertAlmostEqual(
                fit.paths[kk].degeneracy,
                plan["degens"][kk] * true_w[kk], places=4,
                msg="有效简并度必须等于 N_crystal·w")
            self.assertIsNotNone(fit.paths[kk].n_err)
        self.assertLess(abs(fit.delta_e0 - e0), 0.6)
        self.assertEqual(len(fit.parameter_names), fit.n_variables)
        self.assertEqual(fit.correlation.shape,
                         (fit.n_variables, fit.n_variables))
        self.assertTrue(np.all(np.isfinite(fit.correlation)))
        self.assertTrue(np.allclose(fit.correlation, fit.correlation.T))
        self.assertTrue(np.allclose(np.diag(fit.correlation), 1.0))
        self.assertLessEqual(np.max(np.abs(fit.correlation
                                           - np.eye(fit.n_variables))), 1.0)

    def test_back_ft_q_space_phase_and_envelope(self):
        """B1-2：R 窗反傅里叶的 q 轴、包络与相位。

        合成单壳 χ(k)=sin(2kR₀)·exp(−2k²σ²)/k² 经 FT/Back-FT 后，
        Re χ(q) 必须与输入 k²χ(k) 同相（q∈[4,10] 相关 >0.95）。
        若反向变换符号写反或窗函数端点错误，相关会掉到 0 附近。
        """
        k = np.arange(1.0, 14.0, 0.05)
        R0, sigma2 = 2.0, 0.004
        chi = np.sin(2.0 * k * R0) * np.exp(-2.0 * k**2 * sigma2) / k**2
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2, 1.0)
        bft = athena.back_ft_to_q_space(ft.r, ft.chi_r, 1.3, 2.5, 0.1)

        self.assertEqual(len(bft.q), len(bft.chi_q))
        self.assertTrue(np.all(np.diff(bft.q) > 0), "q 轴不单调递增")
        self.assertTrue(np.all(bft.magnitude >= 0))
        self.assertTrue(np.all(np.isfinite(bft.magnitude)))
        self.assertEqual(len(bft.r_window), len(ft.r))

        mask = (bft.q >= 4.0) & (bft.q <= 10.0)
        self.assertGreater(int(mask.sum()), 50)
        target = np.interp(bft.q[mask], k, k**2 * chi)
        corr = float(np.corrcoef(bft.real[mask], target)[0, 1])
        self.assertGreater(corr, 0.95,
                           f"Re χ(q) 与 k²χ 相关仅 {corr:.4f}（符号/相位错？）")
        self.assertGreater(float(bft.magnitude[mask].max()), 0.0)

    def test_lcf_and_pca_synthetic_mixture(self):
        """B2-2：3 端元合成混合谱回收权重；PCA 前 3 主成分解释 >99.9%。"""
        import xafs_lcf

        goe = xafs_pipeline.process_sample(
            [GOETHITE], absorber="Fe", strict=False, verbose=False).norm
        mag = xafs_pipeline.process_sample(
            [MAGNETITE], absorber="Fe", strict=False, verbose=False).norm
        lep = self.lep.norm

        x = np.arange(-20.0, 60.0, 0.2)

        def flat(n):
            arr = n.flattened if n.flattened is not None else n.mu_norm
            return np.interp(x, n.energy - n.e0, arr)

        y_lep, y_goe, y_mag = flat(lep), flat(goe), flat(mag)
        rng = np.random.default_rng(42)
        y = 0.35 * y_lep + 0.65 * y_goe + rng.normal(0, 1e-3, x.size)
        target = NormalizedScan(
            energy=lep.e0 + x, mu=y, mu_norm=y,
            pre_edge_line=np.zeros_like(y), edge_step=1.0,
            plateau=1.0, e0=lep.e0, flattened=y.copy())
        res = xafs_lcf.lcf_from_scans(
            target, {"lepido": lep, "goethite": goe, "magnetite": mag},
            mode="xanes")
        self.assertLess(abs(res["weights"]["lepido"] - 0.35), 0.02)
        self.assertLess(abs(res["weights"]["goethite"] - 0.65), 0.02)
        self.assertLess(abs(res["weights"]["magnetite"]), 0.02)
        self.assertLess(res["r_factor"], 1e-3)

        specs = []
        for i, (a, b, c) in enumerate([
                (0.2, 0.8, 0.0), (0.5, 0.5, 0.0), (0.8, 0.2, 0.0),
                (0.4, 0.1, 0.5), (0.1, 0.4, 0.5), (0.3, 0.3, 0.4)]):
            yy = (a * y_lep + b * y_goe + c * y_mag
                  + rng.normal(0, 1e-4, x.size))
            specs.append((f"mix{i}", x, yy))
        pca = xafs_lcf.run_pca(specs, -20.0, 60.0)
        cum3 = float(np.sum(pca["explained_variance_ratio"][:3]))
        self.assertGreater(cum3, 0.999,
                           f"前 3 主成分累计解释仅 {cum3:.6f}")

    def test_wt_complex_residual_catches_phase_inversion(self):
        """B5-1：复数小波矩阵 + 相位残差必须抓住"幅值相等、相位反 π"。

        反相模型 χ_model = −χ_data 的 |W| 与实验逐点相同，旧版幅值
        残差严格为 0；复数残差应为 |W−(−W)|²/|W|² = 4。真实 S7 模型
        的复数 R-factor 必须保持在 0.25 以下，证明该指标有判别力而
        不是恒定 4。
        """
        kmax = float(min(12.0, np.max(self.lep.chi_k.k)))
        inverted = larch_wt.compute_model_wt_residual(
            self.lep.chi_k, -np.asarray(self.lep.chi_k.chi),
            sigma=1.5, kmin=3.0, kmax=kmax, kweight=2, rmax=4.0)
        self.assertLess(inverted["r_factor"], 1e-12,
                        f"反相模型幅值残差={inverted['r_factor']:.3e}，"
                        f"说明幅值口径本应完全失效")
        self.assertIsNotNone(inverted["complex_r_factor"])
        self.assertAlmostEqual(inverted["complex_r_factor"], 4.0, places=6)
        self.assertEqual(inverted["complex_residual_map"].shape,
                         inverted["wt_data"].magnitude.shape)

        wt = larch_wt.continuous_wavelet_transform(
            self.lep.chi_k, sigma=1.5, kmin=3.0, kmax=kmax,
            kweight=2, rmax=4.0, normalize=True)
        self.assertEqual(wt.complex_map.shape, wt.magnitude.shape)
        self.assertLess(float(np.max(np.abs(
            np.abs(wt.complex_map) - wt.magnitude))), 1e-12,
            "normalize=True 时 |complex_map| 与 magnitude 未同步归一")
        self.assertIsNotNone(wt.phase)
        self.assertEqual(wt.phase.shape, wt.magnitude.shape)

        fit = self.lep_fit["fit"]["fit"]
        self.assertIsNotNone(fit.model_k)
        real = larch_wt.compute_model_wt_residual(
            self.lep.chi_k, fit.model_k, sigma=1.5, kmin=3.0,
            kmax=kmax, kweight=2, rmax=4.0)
        self.assertLess(real["complex_r_factor"], 0.25,
                        f"真实 S7 模型 complex R-factor="
                        f"{real['complex_r_factor']:.4f} ≥0.25")
        # 相位数组必须与 k 网格等长，供诊断图直接消费
        prof = larch_wt.extract_k_profile(wt, 1.47, half_width=0.25)
        self.assertIsNotNone(prof["phase_ridge"])
        self.assertIsNotNone(prof["phase_slice"])
        self.assertEqual(len(prof["phase_ridge"]), len(prof["k"]))
        self.assertGreaterEqual(prof["beat_node_phase_jump"], 0.0)

    def test_isotropic_lattice_expansion_constraint(self):
        """B7-1：各向同性晶格约束把 M 个 ΔR 压缩为唯一全局 α。

        用 FEFF 四壳 NiO 按 ΔR_i = α_true·Reff_i（α_true=0.004）合成
        无噪声 χ(k)，要求 isotropic_deltar=True 后：
          - 自由变量数从 13 降为 10（4 个 ΔR → 1 个 α）；
          - Rf < 1e-4，α 回收误差 < 1e-9，四壳 ΔR/Reff 与 α 一致。
        实验 NiO 的各壳位移不服从统一晶格缩放（Rf≈0.135 > 0.08），
        必须诚实降级 cn_reliable=False 并给出同约束警告。
        """
        from types import SimpleNamespace

        motif = [m for m in structure_lib.load_library("Ni")
                 if m.id == "nio_bunsenite_cod_1010093"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        paths = []
        for f in sorted(cache.glob("feff*.dat")):
            p = artemis.load_feff_path(f)
            if int(getattr(p, "nleg", 2)) == 2:
                paths.append(p)
        paths.sort(key=lambda p: p.reff)
        paths = paths[:4]
        self.assertEqual(len(paths), 4, "NiO 四壳 FEFF 路径不完整")

        alpha_true = 0.004
        k = np.arange(2.0, 13.0, 0.02)
        for p in paths:
            artemis.set_path_parameters(
                p, s02=1.0, sigma2=0.004,
                delta_r=alpha_true * float(p.reff), delta_e0=0.0)
        chi = artemis.exafs_model_k(paths, k)
        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        synthetic = SimpleNamespace(
            chi_k=ck, ft=ft,
            structure={"status": "confident", "coverage_ok": True,
                       "ranked": [{"id": motif.id}]},
            merged=None, norm=None, self_absorption=None,
            absorber_symbol="Ni")
        free = xafs_fit.fit_identified(
            synthetic, absorber_symbol="Ni", rmax=4.3,
            verbose=False)["fit"]
        iso = xafs_fit.fit_identified(
            synthetic, absorber_symbol="Ni", rmax=4.3,
            isotropic_deltar=True, verbose=False)["fit"]
        self.assertEqual(free.n_variables, 13)
        self.assertEqual(iso.n_variables, 10,
                         f"各向同性约束后变量数 {iso.n_variables} ≠ 10")
        self.assertEqual(free.n_variables - iso.n_variables,
                         len(iso.paths) - 1)
        self.assertLess(iso.r_factor, 1e-4,
                        f"合成各向同性 NiO 的 Rf={iso.r_factor:.3e} ≥ 1e-4")
        self.assertIsNotNone(iso.lattice_strain)
        self.assertIsNotNone(iso.lattice_strain_err)
        self.assertLess(abs(iso.lattice_strain - alpha_true), 1e-9,
                        f"α 回收 {iso.lattice_strain:.12f} vs 真值 {alpha_true}")
        for p in iso.paths:
            self.assertLess(
                abs(p.delta_r / p.reff - iso.lattice_strain), 1e-12,
                f"{p.label} 的 ΔR/Reff 与全局 α 不一致")

        exp = xafs_pipeline.process_sample(
            [NIO], absorber="Ni", strict=False, verbose=False)
        exp_iso = xafs_fit.fit_identified(
            exp, absorber_symbol="Ni", rmax=4.3,
            isotropic_deltar=True, verbose=False)
        self.assertGreater(exp_iso["fit"].r_factor, 0.08,
                           "实验 NiO 的统一 α 约束意外达到 Rf<0.08，"
                           "需复核是否有人为补偿")
        self.assertFalse(exp_iso["cn_reliable"],
                         "实验 NiO 的 Rf>0.08 却仍标记 CN 可靠")
        self.assertIn("各向同性晶格约束", exp_iso["warning"])

    def test_cif_partial_occupancy_and_vacancy_scaling(self):
        """B7-2：CIF 部分占据、共占位合并与壳层简并度缩放。

        合成 Pm-3m CIF（a=4.10 Å）：第一壳 O 位于 3d (0.5,0,0)（R=2.05 Å，
        6 配位，occ=0.75）；第二壳 Co(0.6)/Ni(0.4) 同处 1b (0.5,0.5,0.5)
        （R=3.5507 Å，8 配位）共占位。要求：默认仍返回 4 元组；return_occupancy
        返 5 元组且第一壳 O 的 occ=0.75；共占位合并后任意两点间距 ≥0.08 Å，
        元素取高占据者 Co 且有效占据度钳制到 1.0，可正常跑 FEFF6；
        select_shell_paths 选出的第一壳简并度 degens[0] 精确等于 6×0.75=4.5，
        第二壳 Fe-Co 简并度等于 8×1.0=8.0。
        """
        cif_text = """data_test_occ
_cell_length_a 4.10
_cell_length_b 4.10
_cell_length_c 4.10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'Pm-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 1.0
O1 O 0.5 0.0 0.0 0.75
Co1 Co 0.5 0.5 0.5 0.6
Ni1 Ni 0.5 0.5 0.5 0.4
"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "occ.cif"
            cif.write_text(cif_text, encoding="utf-8")

            legacy = artemis.read_cif_sites(cif, absorber="Fe", rmax=6.0)
            self.assertGreater(len(legacy), 10)
            self.assertTrue(all(len(s) == 4 for s in legacy),
                            "默认模式不再是 4 元组，破坏既有 CIF 调用")
            sites2, occs2 = artemis.read_cif_sites(
                cif, absorber="Fe", rmax=6.0, with_occupancy=True)
            self.assertEqual(len(sites2), len(occs2))
            five = artemis.read_cif_sites(
                cif, absorber="Fe", rmax=6.0, return_occupancy=True)
            self.assertEqual(len(five[0]), 5)
            self.assertEqual(five[0][3], "O")
            self.assertAlmostEqual(five[0][4], 0.75, places=12)

            pos = np.array([s[:3] for s in five])
            dmin = min(float(np.linalg.norm(pos[i] - pos[j]))
                       for i in range(len(pos))
                       for j in range(i + 1, len(pos)))
            self.assertGreaterEqual(dmin, 0.08,
                                    f"共占位合并后仍有间距 {dmin:.4f} Å < 0.08")
            co_sites = [s for s in five if s[3] == "Co"]
            self.assertGreaterEqual(len(co_sites), 8)
            self.assertTrue(all(abs(s[4] - 1.0) < 1e-12 for s in co_sites),
                            f"Co/Ni 共占位未钳制到 occ=1.0: {co_sites[0][4]}")
            self.assertFalse(any(s[3] == "Ni" for s in five),
                             "Co/Ni 共占位未合并成单一晶位")
            self.assertAlmostEqual(
                float(np.linalg.norm(co_sites[0][:3])),
                2.05 * np.sqrt(3.0), places=6)

            # 合并后的位点必须能正常生成 FEFF 路径而不崩溃
            feff_dir = Path(tmp) / "feff_occ"
            feff_inp = artemis.build_feff_input(
                legacy, absorber="Fe", rmax=4.2, edge="K", title="occ_test")
            datfiles = artemis.run_feff6l(feff_inp, feff_dir)
            self.assertGreaterEqual(len(datfiles), 2)

            scale = artemis.cif_shell_occupancy_scale(cif, "Fe", rmax=6.0)
            o_scale = [occ for (dist, el, occ) in scale
                       if el == "O" and abs(dist - 2.05) < 1e-6]
            self.assertEqual(len(o_scale), 6)
            self.assertTrue(all(abs(occ - 0.75) < 1e-12 for occ in o_scale),
                            f"第一壳 O 占据度缩放错误: {o_scale}")

            motif = structure_lib.Motif(
                id="occ_test", absorber="Fe", kind="crystal",
                nuclearity="crystal", cell={"a": 4.10},
                spacegroup="Pm-3m",
                asym_sites=[
                    {"el": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"el": "O", "x": 0.5, "y": 0.0, "z": 0.0, "occ": 0.75},
                    {"el": "Co", "x": 0.5, "y": 0.5, "z": 0.5,
                     "occ": 0.6},
                    {"el": "Ni", "x": 0.5, "y": 0.5, "z": 0.5,
                     "occ": 0.4}],
                cif_path=str(cif),
                sites=[(x, y, z, el) for (x, y, z, el) in legacy])
            plan = xafs_fit.select_shell_paths(
                motif, edge="K", rmax=3.8, kmin=3.0, kmax=12.0)
            self.assertEqual(plan["labels"][0], "Fe-O@2.05")
            self.assertAlmostEqual(plan["degens"][0], 4.5, places=12,
                                   msg=f"第一壳 Fe-O 简并度 {plan['degens'][0]} ≠ 6×0.75")
            co_idx = plan["labels"].index("Fe-Co@3.55")
            self.assertAlmostEqual(plan["degens"][co_idx], 8.0, places=12,
                                   msg=f"第二壳 Fe-Co 简并度 {plan['degens'][co_idx]} ≠ 8×1.0")

    def test_phase_corrected_ft_and_wt(self):
        """#54 B11-1：FEFF 总相移校正把 FT/WT 峰位拉回 Reff。

        未校正 FT 峰位在 1.4–1.7 Å（表观 R），校正后 |R_peak−Reff|
        必须 <0.05 Å；WT 使用同一相移也必须回到 Reff。
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
        datfile, path0 = first
        reff = float(path0.reff)
        k = np.arange(2.0, 13.0, 0.02)
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(
            p, s02=1.0, delta_r=0.0, sigma2=0.004, delta_e0=0.0)
        chi = artemis.exafs_model_k([p], k)
        phi = artemis.path_total_phase(datfile, k)
        self.assertEqual(phi.shape, k.shape)
        self.assertTrue(np.all(np.isfinite(phi)))
        self.assertIn(artemis.path_total_phase.last_source,
                      ("feff_pha", "hilbert_fallback"))

        def ft_peak(phase_correct):
            ft = athena.ft_to_r_space(
                k, chi, None, 3.0, 12.0, 2, phase_correct=phase_correct)
            m = (ft.r >= 0.8) & (ft.r <= 2.8)
            return float(ft.r[m][np.argmax(np.abs(ft.chi_r[m]))])

        raw_peak = ft_peak(None)
        corrected_peak = ft_peak(phi)
        self.assertGreater(raw_peak, 1.3)
        self.assertLess(raw_peak, 1.7)
        self.assertLess(abs(corrected_peak - reff), 0.05,
                        f"校正 FT 峰 {corrected_peak:.4f} vs Reff {reff:.4f}")

        ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                  mu_norm=np.ones_like(k), rbkg=1.0)

        def wt_peak(phase_correct):
            wt = larch_wt.continuous_wavelet_transform(
                ck, sigma=1.5, kmin=3.0, kmax=12.0, kweight=2,
                rmax=3.0, phase_correct=phase_correct)
            m = (wt.r >= 0.8) & (wt.r <= 2.8)
            mag = wt.magnitude[:, m]
            ij = np.unravel_index(int(np.argmax(mag)), mag.shape)
            return float(wt.r[m][ij[1]])

        wt_raw = wt_peak(None)
        wt_corrected = wt_peak(phi)
        self.assertLess(wt_raw, 1.7)
        self.assertLess(abs(wt_corrected - reff), 0.05,
                        f"校正 WT 峰 {wt_corrected:.4f} vs Reff {reff:.4f}")

    def test_hetero_cocupancy_dual_branch_paths(self):
        """#55 B11-1：异核共占位按距离聚壳并拆成元素分支。

        Pm-3m 中 Ti(0.7)/Ru(0.3) 共占 1b 位（8 个镜像，R=3.5507 Å）：
        两支占据度和=1、简并度合计=8（不是 16——不能按展开条目重复
        乘占据度），且 single_element_approx=False。
        """
        cif_text = """data_test_hetero
_cell_length_a 4.10
_cell_length_b 4.10
_cell_length_c 4.10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'Pm-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 1.0
Ti1 Ti 0.5 0.5 0.5 0.7
Ru1 Ru 0.5 0.5 0.5 0.3
"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "hetero.cif"
            cif.write_text(cif_text, encoding="utf-8")
            out = artemis.cif_hetero_cocupancy_branches(
                cif, "Fe", rmax=5.0, delta_z=3, occ_min=0.10)
        self.assertFalse(out["single_element_approx"],
                         "Ti/Ru 共占位被误判为单元素近似")
        hetero_shells = [s for s in out["shells"]
                         if not s["single_element_approx"]]
        self.assertEqual(len(hetero_shells), 1)
        shell = hetero_shells[0]
        self.assertAlmostEqual(shell["distance"], 3.5507041555161982,
                               places=10)
        self.assertEqual(shell["n_geom"], 8)
        branches = {b["element"]: b for b in shell["branches"]}
        self.assertEqual(set(branches), {"Ti", "Ru"})
        self.assertAlmostEqual(branches["Ti"]["occ"], 0.7, places=12)
        self.assertAlmostEqual(branches["Ru"]["occ"], 0.3, places=12)
        self.assertAlmostEqual(
            branches["Ti"]["occ"] + branches["Ru"]["occ"], 1.0, places=12)
        self.assertAlmostEqual(
            branches["Ti"]["degen"] + branches["Ru"]["degen"],
            shell["n_geom"], places=12)
        self.assertAlmostEqual(branches["Ti"]["degen"], 5.6, places=12)
        self.assertAlmostEqual(branches["Ru"]["degen"], 2.4, places=12)

    def test_sac_dac_cluster_generator_and_element_expansion(self):
        """#60 B12-2：SAC/DAC 团簇生成、FEFF 可跑与元素扩充。

        M–N₄ 恰好 4 条且键长误差 <1e-9；C 壳层在 ring_radius±0.2；
        build_feff_input + feff6l 成功；12 个新元素全部未标定，
        Fe/Co/Ni/Cu/Pt 的 S₀² 逐位不变。
        """
        sites = artemis.sac_cluster_sites(
            bond_m_ligand=2.0, ligand="N", n_coord=4,
            axial=None, carbon_shell=True, ring_radius=2.9)
        n_sites = [s for s in sites if s[3] == "N"]
        c_sites = [s for s in sites if s[3] == "C"]
        self.assertEqual(len(n_sites), 4)
        self.assertEqual(len(c_sites), 8)
        for s in n_sites:
            self.assertLess(abs(float(np.linalg.norm(s[:3])) - 2.0), 1e-9)
        for s in c_sites:
            self.assertLess(
                abs(float(np.linalg.norm(s[:3])) - 2.9), 0.2)
        inp = artemis.build_feff_input(
            sites, absorber="Fe", rmax=4.0, title="sac_test")
        with tempfile.TemporaryDirectory() as tmp:
            files = artemis.run_feff6l(inp, tmp)
            self.assertGreaterEqual(len(files), 1, "FEFF 未生成路径文件")
            # files 的路径位于临时目录内，必须在 with 块内读路径文件；
            # 出了块目录已被删除，load_feff_path 会报文件不存在。
            p0 = artemis.load_feff_path(files[0])
            self.assertGreater(p0.reff, 1.0)

        dac = artemis.diatomic_m1m2_sites(
            "Fe", "Co", bond_length=2.5, ligand="N",
            n_coord=4, ligand_bond=2.0)
        m2 = [s for s in dac if s[3] == "Co"]
        self.assertEqual(len(m2), 1)
        self.assertAlmostEqual(m2[0][2], 2.5, places=12)
        self.assertEqual(sum(1 for s in dac if s[3] == "N"), 8)

        new_elements = ("Cr", "Mn", "Zn", "Mo", "Ru", "Rh",
                        "Pd", "Ag", "Ir", "Au", "Ce", "U")
        for sym in new_elements:
            cfg = get_absorber(sym)
            self.assertFalse(cfg.calibrated, f"{sym} 不应标记为已标定")
            self.assertIsNone(cfg.s02, f"{sym} 的 S₀² 必须为 None")
            self.assertTrue(np.isfinite(cfg.e0_edge))
            self.assertTrue(np.isfinite(cfg.kmax_ceiling))
            self.assertIn(sym, artemis.COVALENT_RADII)
        calibrated = {"Fe": 0.803, "Co": 0.7308, "Ni": 0.7795,
                      "Cu": 0.8670, "Pt": 0.8075}
        for sym, s02 in calibrated.items():
            self.assertEqual(get_absorber(sym).s02, s02,
                             f"{sym} 的标定 S₀² 被改动")

    def test_dynamic_motif_collision_facet_alloy_guards(self):
        """#19③：动态催化 motif 全原子碰撞拒绝 / 未实现晶面拒绝 / 合金组成诚实披露。"""
        # 碰撞：键长压到原子重叠（<0.80 Å 硬下限）必须诚实拒绝；中心补回后两两查距
        with self.assertRaises(XAFSConfigError):
            structure_lib.build_sac_motif("Fe", bond_m_ligand=0.3, carbon_shell=False)
        with self.assertRaises(XAFSConfigError):
            structure_lib.build_dac_motif("Fe", "Co", bond_length=0.3)
        with self.assertRaises(XAFSConfigError):
            structure_lib.build_alloy_motif("Pt", "Co", r_aa=0.3, r_ab=0.3)
        # 未实现晶面：Miller 指数标签不得当作已实现几何（当前无 Miller 切割）
        for facet in ("111", "100", "110"):
            with self.assertRaises(XAFSConfigError):
                structure_lib.build_algorithmic_catalytic_motif("Fe", {"surface": facet})
        # 合金组成诚实：x=0.01 在 n=12 单壳无法精确表示 → 显式披露实际 1/12，不静默
        al = structure_lib.build_alloy_motif("Pt", "Co", x_partner=0.01, n_total=12)
        self.assertAlmostEqual(al.source["x_partner_actual"], 1.0 / 12.0, places=9)
        self.assertIn("组成离散化", al.notes)
        # 精确组成 x=0.25 不触发披露（保持 #70 基线）
        al2 = structure_lib.build_alloy_motif("Pt", "Co", x_partner=0.25, n_total=12)
        self.assertAlmostEqual(al2.source["x_partner_actual"], 0.25, places=9)
        self.assertNotIn("组成离散化", al2.notes)

    def test_sac_axial_and_dac_mm_bond_ftest_pipeline(self):
        """#61 B13-1：动态 SAC/DAC motif + M₁–M₂ 金属键显著性判据。

        用 DAC motif 的真实 FEFF 路径合成两组谱（ε≈1e-3）：
        - 纯 Fe–N₄ 谱按 DAC 模型拟合：mm_bond_ftest.significant=False
          （p>0.05，N_MM 不显著大于误差）；
        - 真 Fe–N₄ + Fe–Co 谱：R_Co∈[2.47,2.53]、N_Co∈[0.7,1.3]，
          significant=True 且 p<0.01；
        - SAC 轴向氧谱：N_N≈4、N_O≈1（|Δ|<0.5）。
        """
        from types import SimpleNamespace
        from larch.xafs import feffpath

        s02 = 0.803

        def synth(files, degens, noise=1e-3, seed=13):
            k = np.arange(3.0, 12.0, 0.05)
            plist = []
            for f, d in zip(files, degens):
                p = feffpath(str(f), s02=s02, sigma2=0.004, e0=2.0,
                             deltar=0.0)
                p.degen = float(d)
                p._calc_chi()
                plist.append(p)
            chi = artemis.exafs_model_k(plist, k)
            if noise:
                rng = np.random.default_rng(seed)
                chi = chi + rng.normal(0.0, noise, size=chi.shape)
            chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                         mu_norm=np.zeros_like(k), rbkg=1.0)
            ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
            return SimpleNamespace(
                chi_k=chi_k, ft=ft, absorber_symbol="Fe", structure=None,
                merged=None, norm=None, self_absorption=None)

        dac = structure_lib.parse_catalytic_cluster_spec(
            "Fe", "dac:Co@2.50,N4@2.00")
        self.assertEqual(dac.nuclearity, "binuclear")
        self.assertIn("_dynamic_cache", str(dac.cache_dir))
        plan = xafs_fit.select_shell_paths(
            dac, edge="K", rmin=1.0, rmax=2.6, kmin=3.0, kmax=12.0)
        self.assertEqual([s["scatterer"] for s in plan["shells"]],
                         ["N", "Co"])
        self.assertEqual(plan["degens"], [4.0, 1.0])

        # ---- 纯 SAC 谱：DAC 模型的 Co 分支必须判为不显著 ----
        res_sac = synth(plan["files"][:1], plan["degens"][:1])
        out_sac = xafs_fit.fit_identified(
            res_sac, absorber_symbol="Fe",
            catalytic_model="dac:Co@2.50,N4@2.00",
            rmin=1.0, rmax=2.6, s02=s02, verbose=False)
        self.assertEqual(out_sac["mode"], "motif")
        mm = out_sac["mm_bond_ftest"]
        self.assertIsNotNone(mm)
        self.assertEqual(mm["partner"], "Co")
        self.assertFalse(mm["significant"],
                         f"纯 SAC 谱不应检出 Fe–Co: {mm}")
        self.assertGreater(mm["p_value"], 0.05)

        # ---- 真 DAC 谱：回收 Fe–Co 距离/配位并判显著 ----
        res_dac = synth(plan["files"], plan["degens"])
        out_dac = xafs_fit.fit_identified(
            res_dac, absorber_symbol="Fe",
            catalytic_model="dac:Co@2.50,N4@2.00",
            rmin=1.0, rmax=2.6, s02=s02, verbose=False)
        self.assertEqual(out_dac["mode"], "motif")
        co = [p for p in out_dac["fit"].paths if "Co" in p.label]
        self.assertEqual(len(co), 1)
        self.assertGreaterEqual(co[0].r, 2.47)
        self.assertLessEqual(co[0].r, 2.53)
        self.assertGreaterEqual(co[0].degeneracy, 0.7)
        self.assertLessEqual(co[0].degeneracy, 1.3)
        mm_dac = out_dac["mm_bond_ftest"]
        self.assertTrue(mm_dac["significant"],
                        f"真 DAC 谱未检出 Fe–Co: {mm_dac}")
        self.assertLess(mm_dac["p_value"], 0.01)
        self.assertGreaterEqual(mm_dac["delta_r_factor"], 0.0)

        # ---- SAC 轴向氧：N_N≈4、N_O≈1（共享 σ² 避免 0.1 Å 近简并） ----
        sac = structure_lib.parse_catalytic_cluster_spec(
            "Fe", "sac:N4@2.00,axial=O@2.10,C=none")
        self.assertEqual(sac.nuclearity, "mononuclear")
        self.assertIn("_dynamic_cache", str(sac.cache_dir))
        splan = xafs_fit.select_shell_paths(
            sac, edge="K", rmin=1.0, rmax=2.6, kmin=3.0, kmax=12.0)
        self.assertEqual([s["scatterer"] for s in splan["shells"]],
                         ["N", "O"])
        deg_axial = list(splan["degens"])
        for i, sh in enumerate(splan["shells"]):
            if sh["scatterer"] == "O":
                deg_axial[i] = 1.0
        res_ax = synth(splan["files"], deg_axial)
        out_ax = xafs_fit.fit_identified(
            res_ax, absorber_symbol="Fe",
            catalytic_model="sac:N4@2.00,axial=O@2.10,C=none",
            rmin=1.0, rmax=2.6, shared_sigma2="all", s02=s02,
            verbose=False)
        by_el = {("O" if "O" in p.label else "N"): p
                 for p in out_ax["fit"].paths}
        self.assertLess(abs(by_el["N"].degeneracy - 4.0), 0.5,
                        f"SAC N_N 回收失败: {by_el['N'].degeneracy}")
        self.assertLess(abs(by_el["O"].degeneracy - 1.0), 0.5,
                        f"SAC N_O 回收失败: {by_el['O'].degeneracy}")
        self.assertLess(by_el["N"].n_err, 0.5)
        self.assertLess(by_el["O"].n_err, 0.5)

    def test_hetero_cocupancy_end_to_end_dual_branch_fit(self):
        """#62 B13-2：共占位 CIF → 双分支 FEFF → 共享 ΔR/σ² 精修。

        Ti(0.7)/Ru(0.3) 共占 N_geom=8（N_Ti=5.6, N_Ru=2.4）：
        path_links 把自由变量从 7 压到 5；回收两分支 N、ΔR 与 σ²
        逐位相等；双分支 Rf 比单元素近似小 5 倍以上。
        """
        from larch.xafs import feffpath

        cif_text = """data_test_hetero
_cell_length_a 4.10
_cell_length_b 4.10
_cell_length_c 4.10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'Pm-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 1.0
Ti1 Ti 0.5 0.5 0.5 0.7
Ru1 Ru 0.5 0.5 0.5 0.3
"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "hetero.cif"
            cif.write_text(cif_text, encoding="utf-8")
            motif = structure_lib.import_cif(cif, "test_hetero_b13", "Fe")
            motif.cif_path = str(cif)
            motif.sites = structure_lib.expand_crystal(motif, rmax=5.0)
            motif.shell_expect = structure_lib.geometric_shells(motif.sites)
            plan = xafs_fit.select_shell_paths(
                motif, edge="K", rmin=2.8, rmax=4.3,
                kmin=3.0, kmax=12.0, hetero_cocupancy=True)

        self.assertEqual(plan["hetero_shells"][0]["n_geom"], 8)
        branches = {b["element"]: b for b in
                    plan["hetero_shells"][0]["branches"]}
        self.assertAlmostEqual(branches["Ti"]["degen"], 5.6, places=10)
        self.assertAlmostEqual(branches["Ru"]["degen"], 2.4, places=10)
        self.assertEqual(len(plan["path_links"]), 1)
        link = plan["path_links"][0]
        self.assertTrue(link["share_deltar"] and link["share_sigma2"])
        self.assertIsNone(link["degen_ratio"])

        # 合成真值谱：N_Ti=5.6, N_Ru=2.4, ΔR=-0.02, σ²=0.004
        s02, dr_true, s2_true, e0_true = 0.803, -0.02, 0.004, 2.0
        k = np.arange(3.0, 12.0, 0.05)
        plist = []
        for f, d in zip(plan["files"], plan["degens"]):
            p = feffpath(str(f), s02=s02, sigma2=s2_true,
                         e0=e0_true, deltar=dr_true)
            p.degen = float(d)
            p._calc_chi()
            plist.append(p)
        chi = artemis.exafs_model_k(plist, k)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.zeros_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)

        def do_fit(files, labels, degens, links=None):
            return artemis.fit_r_space(
                chi_k, ft, files, rmin=plan["rmin"], rmax=plan["rmax"],
                kmin=3.0, kmax=12.0, kweight=2, s02=s02,
                vary_delta_e0=True, vary_n=True,
                path_degen=degens, path_labels=labels, path_links=links)

        fit = do_fit(plan["files"], plan["labels"], plan["degens"],
                     plan["path_links"])
        self.assertEqual(fit.n_variables, 5,
                         "共享 ΔR/σ² 后应为 4 个壳层变量 + ΔE0")
        free = do_fit(plan["files"], plan["labels"], plan["degens"])
        self.assertEqual(free.n_variables, 7,
                         "无 path_links 时双分支应为 2×3+1=7 个变量")

        by_el = {("Ti" if "Ti" in p.label else "Ru"): p
                 for p in fit.paths}
        ti, ru = by_el["Ti"], by_el["Ru"]
        self.assertGreaterEqual(ti.degeneracy, 5.2)
        self.assertLessEqual(ti.degeneracy, 6.0)
        self.assertGreaterEqual(ru.degeneracy, 2.0)
        self.assertLessEqual(ru.degeneracy, 2.8)
        self.assertAlmostEqual(ti.delta_r, ru.delta_r, places=12)
        self.assertAlmostEqual(ti.sigma2, ru.sigma2, places=12)
        self.assertAlmostEqual(ti.delta_r_err, ru.delta_r_err, places=12)
        self.assertAlmostEqual(ti.sigma2_err, ru.sigma2_err, places=12)
        self.assertLess(fit.r_factor, 1e-4,
                        f"双分支合成谱应近完美复现: Rf={fit.r_factor}")

        ti_idx = [i for i, s in enumerate(plan["shells"])
                  if s["scatterer"] == "Ti"][0]
        single = do_fit([plan["files"][ti_idx]],
                        [plan["labels"][ti_idx]],
                        [plan["degens"][ti_idx]])
        self.assertGreater(single.r_factor, 5.0 * fit.r_factor,
                           "单元素近似应显著劣于双分支共享精修")

    def test_cif_symmetry_expansion_and_no_silent_fallback(self):
        """#19①：CIF 必须完整空间群展开；坏/空/显式缺失输入显式报错，绝不静默换材料。

        - fcc Ag(Fm-3m, a=4.0862)：非对称单元仅 1 原子，展开后常规晶胞 4 原子，
          第一壳 12×a/√2≈2.889 Å（旧版只平移非对称单元 → 6×a=4.086，错误）；
        - 坏 CIF（有晶胞无原子循环）→ XAFSDataError，不回退金红石；
        - 显式 CIF 路径不存在 → XAFSDataError。
        """
        import catalytic_builder as cb
        from xafs_core import XAFSDataError

        repo = Path(__file__).resolve().parent
        ag_cif = repo / "data" / "cif_raw" / "Ag_COD_9008459.cif"
        self.assertTrue(ag_cif.is_file(), f"缺测试夹具 {ag_cif}")

        model = cb.parse_cif_simple(ag_cif)
        self.assertEqual(len(model.atoms), 4,
                         "fcc 常规晶胞应 4 个 Ag（非对称 1 × F 心 4）")
        self.assertAlmostEqual(model.a, 4.0862, places=3)
        supercell = model.build_supercell(nx=2, ny=2, nz=2)
        coords = np.array([[x, y, z] for el, x, y, z in supercell])
        center = coords[np.argmin(np.linalg.norm(coords, axis=1))]
        d = np.linalg.norm(coords - center, axis=1)
        d = d[d > 0.1]
        r1 = float(d.min())
        nn = int(np.sum(np.abs(d - r1) < 0.05))
        self.assertEqual(nn, 12, f"fcc 第一壳应 12 近邻，实得 {nn}@{r1:.3f}")
        self.assertAlmostEqual(r1, model.a / np.sqrt(2.0), places=2)
        self.assertLess(r1, model.a - 0.5,
                        "第一壳必须 a/√2≈2.889，而非旧版简单立方的 a=4.086")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad = tmp / "bad.cif"
            bad.write_text(
                "data_bad\n_cell_length_a 4.0\n_cell_length_b 4.0\n"
                "_cell_length_c 4.0\n_cell_angle_alpha 90\n"
                "_cell_angle_beta 90\n_cell_angle_gamma 90\n",
                encoding="utf-8")
            with self.assertRaises(XAFSDataError):
                cb.parse_cif_simple(bad)
            with self.assertRaises(XAFSDataError):
                cb.parse_cif_simple(tmp / "does_not_exist.cif")

    def test_phase_corrected_diagnostics_and_chi2_contour_report(self):
        """#64 B14-2：相移校正真键长与二维 χ² 等高线全链接入。

        用自带 lepidocrocite + S7 拟合结果：
        - 未校正 FT 峰在 1.4–1.7 Å（表观 R），E0 补偿相移校正后
          |r_peak_corrected − r_fit| < 0.06 Å；
        - χ² 网格 (15,15)、min_chi2 有限，CSV/PNG 非空；
        - xafs_report 开 --phase-correct/--chi2-contour 后 HTML 内联
          两张图并含标题。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pc = xafs_plot_special.write_phase_corrected_diagnostics(
                self.lep, self.lep_fit, tmp, stem="lepido")
            self.assertGreaterEqual(pc["r_peak_raw"], 1.4)
            self.assertLessEqual(pc["r_peak_raw"], 1.7)
            self.assertLess(abs(pc["r_peak_corrected"] - pc["r_fit"]), 0.06,
                            f"相移校正未回到拟合键长: {pc}")
            self.assertEqual(pc["phase_source"], "feff_pha")
            csv_path, png_path = Path(pc["csv"]), Path(pc["png"])
            self.assertTrue(csv_path.is_file()
                            and csv_path.stat().st_size > 0)
            self.assertTrue(png_path.is_file()
                            and png_path.stat().st_size > 0)
            header = csv_path.read_text(
                encoding="utf-8-sig").splitlines()[0]
            self.assertEqual(
                header, "R_A,mag_raw,real_raw,"
                        "mag_phase_corrected,real_phase_corrected")

            cc = xafs_plot_special.write_chi2_contour_diagnostics(
                self.lep, self.lep_fit, tmp, stem="lepido")
            self.assertIsNotNone(cc, "χ² 等高线未生成")
            self.assertTrue(np.isfinite(cc["min_chi2"]))
            self.assertEqual(cc["chi2_grid"].shape, (15, 15))
            self.assertEqual((cc["x_name"], cc["y_name"]), ("N", "sigma2"))
            for key in ("csv", "png"):
                p = Path(cc[key])
                self.assertTrue(p.is_file() and p.stat().st_size > 0)
            self.assertEqual(len(cc["levels"]), 3)
            self.assertAlmostEqual(cc["levels"][0] - cc["min_chi2"],
                                   2.30, places=6)
            self.assertLessEqual(cc["min_chi2"],
                                 np.nanmin(cc["chi2_grid"]) + 1e-9)

            report_dir = tmp / "report"
            xafs_report.main([
                str(LEPIDO), "--absorber", "Fe", "-o", str(report_dir),
                "--plot", "full", "--phase-correct", "--chi2-contour"])
            html_files = sorted(report_dir.glob("*_report.html"))
            self.assertEqual(len(html_files), 1)
            html_text = html_files[0].read_text(encoding="utf-8")
            self.assertIn("相移校正真键长", html_text)
            self.assertIn("二维 χ² 置信椭圆", html_text)
            self.assertIn("data:image/png;base64,", html_text)

    def test_catalysis_library_motifs_for_co_ni_cu_pt(self):
        """#68 B16-2：四大催化元素催化基准 Motif 入库 + PtCl2 实测命中。

        - Co/Ni/Cu 平面 M-N4 + 碳壳、Cu2O 赤铜矿、PtCl2 平面 PtCl4；
        - 每个 Motif 均能 ensure_feff 生成 feff0001.dat；
        - 真实 Pt-L3-PlatinumChloride.xdi（Sample.name=PtCl2）：
          top1=ptcl4_square、status=confident，S7 第一壳
          N∈[3.5,4.5]、R∈[2.26,2.35] Å、Rf<0.05。
        """
        expected = {
            "Co": [("con4_planar", "N", 1.96, 4, "C", 2.88)],
            "Ni": [("nin4_planar", "N", 1.92, 4, "C", 2.85)],
            "Cu": [("cun4_planar", "N", 1.95, 4, "C", 2.88),
                   ("cu2o_cuprite_cod_1010941", "O", 1.8446, 2,
                    "Cu", 3.0123)],
            "Pt": [("ptcl4_square", "Cl", 2.31, 4, "Cl", 3.45)],
        }
        for el, entries in expected.items():
            lib = {m.id: m for m in structure_lib.load_library(el)}
            edge = get_absorber(el).edge
            for mid, el1, r1, n1, el2, r2 in entries:
                self.assertIn(mid, lib, f"{el} 库缺 {mid}")
                motif = lib[mid]
                shells = motif.shell_expect
                self.assertEqual(shells[0]["el"], el1)
                self.assertAlmostEqual(shells[0]["r"], r1, places=3)
                self.assertEqual(shells[0]["n"], n1)
                self.assertEqual(shells[1]["el"], el2)
                self.assertAlmostEqual(shells[1]["r"], r2, places=3)
                cache = structure_lib.ensure_feff(motif, edge=edge)
                self.assertTrue(
                    (cache / "feff0001.dat").is_file(),
                    f"{el}/{mid} 未生成 feff0001.dat")

        res = xafs_pipeline.process_sample(
            [PT_CL], absorber="Pt", strict=False, verbose=False)
        out = xafs_fit.run(
            res, identify=True, fit=True, absorber_symbol="Pt",
            verbose=False)
        sid = out["structure"]
        self.assertEqual(sid["status"], "confident")
        self.assertEqual(sid["ranked"][0]["id"], "ptcl4_square")
        self.assertEqual(sid["uncovered_peaks"], [])
        fit = out["fit"]["fit"]
        self.assertEqual(out["fit"]["motif"], "ptcl4_square")
        path = fit.paths[0]
        self.assertTrue(path.label.startswith("Pt-Cl"))
        self.assertGreaterEqual(path.degeneracy, 3.5)
        self.assertLessEqual(path.degeneracy, 4.5)
        self.assertGreaterEqual(path.r, 2.26)
        self.assertLessEqual(path.r, 2.35)
        self.assertLess(fit.r_factor, 0.05)

    def test_nanoparticle_size_and_dispersion_inversion(self):
        """#69 B17-1：Calvin 球截断方程反解粒径、误差传播与形貌判据。

        - N₁=8.0、r₁=2.774 Å 反解 u 精确满足截断方程（误差 <1e-9）；
        - N₁ 增大时直径单调增大、分散度单调减小，N₁=12 为体相极限；
        - 显著偏低的第二壳 N₂ 判为 raft_or_flattened。
        """
        est = artemis.estimate_nanoparticle_size(
            n1=8.0, r1=2.774, n1_err=0.5, lattice="fcc")
        u = 2.774 / est["diameter_A"]
        residual = 12.0 * (1.0 - 1.5 * u + 0.5 * u ** 3) - 8.0
        self.assertLess(abs(residual), 1e-9,
                        f"球截断方程残差 {residual:.3e}")
        self.assertGreaterEqual(est["diameter_nm"], 1.0)
        self.assertLessEqual(est["diameter_nm"], 1.5)
        self.assertGreater(est["diameter_nm_err"], 0.0)
        self.assertGreater(est["dispersion"], 0.3)
        self.assertLessEqual(est["dispersion"], 1.0)
        self.assertEqual(est["shape_hint"], "single_shell_only")

        diameters = [artemis.estimate_nanoparticle_size(
            n1=n, r1=2.774)["diameter_nm"] for n in (6.0, 9.0, 11.0)]
        self.assertLess(diameters[0], diameters[1])
        self.assertLess(diameters[1], diameters[2])
        dispersions = [artemis.estimate_nanoparticle_size(
            n1=n, r1=2.774)["dispersion"] for n in (6.0, 9.0, 11.0)]
        self.assertGreater(dispersions[0], dispersions[1])
        self.assertGreater(dispersions[1], dispersions[2])

        bulk = artemis.estimate_nanoparticle_size(n1=12.0, r1=2.774)
        self.assertEqual(bulk["regime"], "bulk_limit")
        self.assertEqual(bulk["dispersion"], 0.0)
        self.assertEqual(bulk["diameter_nm"], float("inf"))

        raft = artemis.estimate_nanoparticle_size(
            n1=8.0, r1=2.774, n2=1.0)
        self.assertGreater(raft["n2_spherical_expected"], 3.0)
        self.assertEqual(raft["shape_hint"], "raft_or_flattened")
        consistent = artemis.estimate_nanoparticle_size(
            n1=8.0, r1=2.774, n2=raft["n2_spherical_expected"])
        self.assertEqual(consistent["shape_hint"], "spherical_consistent")

        bcc = artemis.estimate_nanoparticle_size(
            n1=6.0, r1=2.48, lattice="bcc")
        self.assertEqual(bcc["n1_bulk"], 8.0)
        self.assertEqual(bcc["regime"], "nanoparticle")
        with self.assertRaises(XAFSConfigError):
            artemis.estimate_nanoparticle_size(8.0, 2.774, lattice="diamond")

    def test_bimetallic_alloy_motif_and_cowley_sro_pipeline(self):
        """#70 B17-2：alloy: 动态建模 + Cowley SRO 端到端回收。

        - α 对随机固溶体/核壳偏析/异核有序分别精确给出 0 / 0.6 / −1；
        - PtCo 合金谱按真值 N_Pt-Pt=6、N_Pt-Co=2 合成后 S7 回收
          误差 <0.5，α≈0、nanoparticle_size≈1.0–1.5 nm。
        """
        random_sro = artemis.compute_cowley_sro(9.0, 3.0, 0.25)
        self.assertAlmostEqual(random_sro["alpha_cowley"], 0.0, places=12)
        self.assertAlmostEqual(random_sro["alloying_extent_pct"], 100.0,
                               places=12)
        self.assertEqual(random_sro["tendency"], "random_solid_solution")

        segregated = artemis.compute_cowley_sro(8.0, 2.0, 0.50)
        self.assertAlmostEqual(segregated["alpha_cowley"], 0.60, places=12)
        self.assertEqual(segregated["tendency"],
                         "homoatomic_clustering_or_segregation")

        ordered = artemis.compute_cowley_sro(6.0, 6.0, 0.25)
        self.assertAlmostEqual(ordered["alpha_cowley"], -1.0, places=12)
        self.assertEqual(ordered["tendency"], "heteroatomic_ordering")

        sro_err = artemis.compute_cowley_sro(
            9.0, 3.0, 0.25, n_aa_err=0.5, n_ab_err=0.3)
        expected_err = float(np.hypot(3.0 * 0.5, 9.0 * 0.3)
                             / (0.25 * 12.0 ** 2))
        self.assertAlmostEqual(sro_err["alpha_err"], expected_err, places=12)

        alloy = structure_lib.parse_catalytic_cluster_spec(
            "Pt", "alloy:Co@2.64,Pt@2.76,x=0.25,n=12")
        self.assertEqual(alloy.nuclearity, "alloy")
        self.assertIn("_dynamic_cache", str(alloy.cache_dir))
        counts = {sh["el"]: sh["n"] for sh in alloy.shell_expect}
        self.assertEqual(counts.get("Co"), 3)
        self.assertEqual(counts.get("Pt"), 9)
        alloy_dict = structure_lib.parse_catalytic_cluster_spec(
            "Pt", {"type": "alloy", "partner": "Co", "r_ab": 2.64,
                   "r_aa": 2.76, "x_partner": 0.25, "n_total": 12})
        self.assertEqual(alloy_dict.id, alloy.id)

        from types import SimpleNamespace
        from larch.xafs import feffpath

        plan = xafs_fit.select_shell_paths(
            alloy, edge="L3", rmin=1.6, rmax=3.2,
            kmin=3.0, kmax=12.0)
        truth = {"Co": 2.0, "Pt": 6.0}
        k = np.arange(3.0, 12.0, 0.05)
        plist = []
        for sh, f in zip(plan["shells"], plan["files"]):
            p = feffpath(str(f), s02=0.8075, sigma2=0.004, e0=2.0,
                         deltar=0.0)
            p.degen = truth[sh["scatterer"]]
            p._calc_chi()
            plist.append(p)
        chi = artemis.exafs_model_k(plist, k)
        rng = np.random.default_rng(11)
        chi = chi + rng.normal(0.0, 1e-3, size=chi.shape)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        res = SimpleNamespace(
            chi_k=chi_k, ft=ft, absorber_symbol="Pt", structure=None,
            merged=None, norm=None, self_absorption=None)
        out = xafs_fit.fit_identified(
            res, absorber_symbol="Pt",
            catalytic_model="alloy:Co@2.64,Pt@2.76,x=0.25,n=12",
            rmin=1.6, rmax=3.2, shared_sigma2="all", verbose=False)
        self.assertEqual(out["mode"], "motif")
        by_scatterer = {}
        for sh, p in zip(plan["shells"], out["fit"].paths):
            by_scatterer[sh["scatterer"]] = p
        self.assertLess(abs(by_scatterer["Pt"].degeneracy - 6.0), 0.5)
        self.assertLess(abs(by_scatterer["Co"].degeneracy - 2.0), 0.5)
        sro = out["alloy_sro"]
        self.assertIsNotNone(sro)
        self.assertEqual(sro["tendency"], "random_solid_solution")
        self.assertLess(abs(sro["alpha_cowley"]), 0.10)
        np_size = out["nanoparticle_size"]
        self.assertIsNotNone(np_size)
        self.assertGreaterEqual(np_size["diameter_nm"], 1.0)
        self.assertLessEqual(np_size["diameter_nm"], 1.5)

    def test_rayleigh_resolution_and_light_element_z_degeneracy_guard(self):
        """#71 B18-1：π/(2Δk) 分辨率门禁与 SAC 近简并拆壳告警。

        - k∈[3,12] 时 ΔR_Rayleigh = π/18 ≈ 0.17453 Å；
        - Fe-N@1.98 vs Fe-O@2.06（ΔZ=1, ΔR=0.08）判 light_element_z_degeneracy；
        - Z 差异大或 ΔR 超限时 resolvable=True；
        - SAC N4@2.00 + axial O@2.10 拟合后 flags 置 True 并追加建议告警。
        """
        rayleigh = artemis.check_shell_resolution_and_z_degeneracy(
            [{"label": "Fe-O@1.98", "scatterer": "O", "r": 1.98}],
            kmin=3.0, kmax=12.0)
        self.assertAlmostEqual(rayleigh["rayleigh_limit_A"],
                               float(np.pi / 18.0), places=12)
        self.assertTrue(rayleigh["resolvable"])

        close = artemis.check_shell_resolution_and_z_degeneracy(
            [{"label": "Fe-N@1.98", "scatterer": "N", "r": 1.98},
             {"label": "Fe-O@2.06", "scatterer": "O", "r": 2.06}])
        self.assertFalse(close["resolvable"])
        self.assertEqual(len(close["pair_warnings"]), 1)
        warning = close["pair_warnings"][0]
        self.assertEqual(warning["kind"], "light_element_z_degeneracy")
        self.assertEqual(warning["delta_z"], 1)
        self.assertLess(warning["delta_r"], warning["rayleigh_limit_A"])

        far_z = artemis.check_shell_resolution_and_z_degeneracy(
            [{"label": "Fe-N@1.98", "scatterer": "N", "r": 1.98},
             {"label": "Fe-Co@2.50", "scatterer": "Co", "r": 2.50}])
        self.assertTrue(far_z["resolvable"])
        self.assertEqual(far_z["pair_warnings"], [])

        far_r = artemis.check_shell_resolution_and_z_degeneracy(
            [{"label": "Fe-O@1.98", "scatterer": "O", "r": 1.98},
             {"label": "Fe-O@2.25", "scatterer": "O", "r": 2.25}])
        self.assertTrue(far_r["resolvable"])
        self.assertEqual(far_r["pair_warnings"], [])

        from types import SimpleNamespace
        from larch.xafs import feffpath

        motif = structure_lib.parse_catalytic_cluster_spec(
            "Fe", "sac:N4@2.00,axial=O@2.10,C=none")
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmin=1.0, rmax=2.6, kmin=3.0, kmax=12.0)
        k = np.arange(3.0, 12.0, 0.05)
        plist = []
        for sh, f in zip(plan["shells"], plan["files"]):
            p = feffpath(str(f), s02=0.803, sigma2=0.004, e0=2.0,
                         deltar=0.0)
            p.degen = float(sh["degen"])
            p._calc_chi()
            plist.append(p)
        chi = artemis.exafs_model_k(plist, k)
        rng = np.random.default_rng(7)
        chi = chi + rng.normal(0.0, 1e-3, size=chi.shape)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        res = SimpleNamespace(
            chi_k=chi_k, ft=ft, absorber_symbol="Fe", structure=None,
            merged=None, norm=None, self_absorption=None)
        out = xafs_fit.fit_identified(
            res, absorber_symbol="Fe",
            catalytic_model="sac:N4@2.00,axial=O@2.10,C=none",
            rmin=1.0, rmax=2.6, verbose=False)
        self.assertFalse(out["resolution_check"]["resolvable"])
        self.assertTrue(out["flags"]["sub_rayleigh_z_degeneracy"])
        self.assertIn("Rayleigh", out["warning"])

    def test_occupancy_scale_failure_hard_fail(self):
        """#77 P0a：占据度展开失败硬报错，不按满占据继续。"""
        import dataclasses
        from unittest.mock import patch

        lep = [m for m in structure_lib.load_library("Fe")
               if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(lep, edge="K")
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "fake.cif"
            cif.write_text("data_fake\n", encoding="utf-8")
            motif = dataclasses.replace(
                lep, id="_p0a_occ_fail", cif_path=str(cif))
            with patch.object(structure_lib, "ensure_feff",
                              return_value=cache), \
                 patch.object(artemis, "cif_shell_occupancy_scale",
                              side_effect=XAFSError("FIT", "boom")):
                with self.assertRaises(structure_model.CNError) as ctx:
                    xafs_fit._ss_rows(motif, edge="K")
        self.assertEqual(ctx.exception.code, "occupancy_scale_failed")
        self.assertIn("occupancy_scale_failed", str(ctx.exception))

    def test_cn_disclosure_explicit_no_cif(self):
        """CN 披露标准：cif=None 视为显式声明，不报 CIF 缺失。"""
        issues = cn_disclosure.check_fit_payload(self._disclosure_payload())
        self.assertFalse(any("CIF status" in x for x in issues))

    def test_motif_provenance_schema(self):
        """#79 P0b：全部 motif 可加载；新字段齐全且状态合法。"""
        allowed_status = set(motif_provenance.PROVENANCE_STATUSES)
        allowed_levels = set(motif_provenance.MODEL_LEVELS)
        allowed_occ = set(motif_provenance.OCCUPANCY_SOURCES)

        report_path = ROOT / "_structure_lib" / "provenance_report.json"
        self.assertTrue(report_path.exists(), "缺少 provenance_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entries = {(e["element"], e["id"]): e
                   for e in report["entries"]}

        total = 0
        for element_dir in sorted((ROOT / "_structure_lib").iterdir()):
            if not element_dir.is_dir() or element_dir.name.startswith("_"):
                continue
            for motif in structure_lib.load_library(element_dir.name):
                total += 1
                self.assertIn(motif.provenance_status, allowed_status,
                              motif.id)
                self.assertIn(motif.model_level, allowed_levels, motif.id)
                self.assertIn(motif.occupancy_source, allowed_occ, motif.id)
                if motif.kind == "crystal":
                    self.assertTrue(motif.absorber_site, motif.id)
                key = (element_dir.name, motif.id)
                self.assertIn(key, entries, motif.id)
                self.assertEqual(entries[key]["provenance_status"],
                                 motif.provenance_status)
                self.assertEqual(entries[key]["model_level"],
                                 motif.model_level)

        self.assertEqual(total, report["summary"]["motifs"])
        self.assertEqual(total, 43)
        self.assertEqual(report["summary"]["unclassified"], 0)

    def test_generated_cif_not_formal_source(self):
        """#80 P0b：现场生成 CIF 不得成为正式来源。"""
        fe = next(m for m in structure_lib.load_library("Fe")
                  if m.id == "fe_bcc")
        self.assertEqual(fe.provenance_status, "generated_only")
        self.assertIsNone(fe.source_cif_sha256)
        self.assertIsNone(fe.cif_path)

        generated = structure_lib._cif_path(fe)
        self.assertTrue(generated.exists())
        self.assertTrue(motif_provenance.is_generated_cif(generated))

        model = structure_model.model_from_motif(fe)
        self.assertEqual(model.provenance_status, "generated_only")
        self.assertIsNone(model.source_cif)
        self.assertIsNone(model.source_cif_sha256)
        gate = structure_model.validate_formal(model)
        self.assertFalse(gate["ok"])
        self.assertIn(gate["code"], ("unverified_source", "no_cif"))

        lep = next(m for m in structure_lib.load_library("Fe")
                   if m.id == "lepidocrocite")
        self.assertIn(lep.provenance_status,
                      ("source_ref_only", "generated_only"))
        self.assertIsNone(lep.source_cif_sha256)
        self.assertEqual(lep.source_ref, "COD:9002763")
        lep_model = structure_model.model_from_motif(lep)
        self.assertIsNone(lep_model.source_cif)
        self.assertFalse(structure_model.validate_formal(lep_model)["ok"])

        # 即使把生成 CIF 伪造成“哈希正确 + 吸收位明确”，门禁仍拒绝。
        fake = structure_model.StructureModel(
            model_id="fake_generated",
            absorber="Fe",
            absorber_site="Fe1@(0,0,0)",
            model_level="crystal_cif",
            source_type="user_cif",
            source_cif=str(generated),
            source_cif_sha256=structure_model.sha256_file(generated),
            provenance_status="verified_source_cif",
        )
        gate = structure_model.validate_formal(fake)
        self.assertFalse(gate["ok"])
        self.assertEqual(gate["code"], "generated_cif_not_formal")

        # 审查 #19⑥：把生成 CIF 复制到外部目录（脱离 _structure_lib/*/_cif/
        # 路径判据），伪造"哈希正确 + 吸收位明确 + verified_source_cif"，
        # 门禁仍须靠持久化生成标记拒绝——来源分类不认目录，只认证据。
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "laundered_fe_bcc.cif"
            copied.write_bytes(generated.read_bytes())
            self.assertTrue(
                motif_provenance.is_generated_cif(copied),
                "复制到外部目录的生成 CIF 应靠持久化标记被识别")
            fake_copy = structure_model.StructureModel(
                model_id="fake_generated_copy",
                absorber="Fe",
                absorber_site="Fe1@(0,0,0)",
                model_level="crystal_cif",
                source_type="user_cif",
                source_cif=str(copied),
                source_cif_sha256=structure_model.sha256_file(copied),
                provenance_status="verified_source_cif",
            )
            gate_copy = structure_model.validate_formal(fake_copy)
            self.assertFalse(gate_copy["ok"])
            self.assertEqual(gate_copy["code"], "generated_cif_not_formal")

    def test_user_cif_is_verified_source(self):
        """#81 P0b：用户真实 CIF + sha256 + 吸收位可升级为正式来源。"""
        source = ROOT / "lepidocrocite.cif"
        self.assertTrue(source.exists())
        with tempfile.TemporaryDirectory() as tmp:
            user_cif = Path(tmp) / "user_lepidocrocite.cif"
            user_cif.write_bytes(source.read_bytes())
            motif = structure_lib.Motif(
                id="user_lep",
                absorber="Fe",
                kind="crystal",
                nuclearity="crystal",
                source={"type": "user_cif", "ref": "user:test"},
                cell={"a": 3.072, "b": 12.516, "c": 3.873,
                      "alpha": 90, "beta": 90, "gamma": 90},
                spacegroup="C m c m",
                asym_sites=[], absorber_fract=[0.5, 0.1778, 0.25],
                cif_path=str(user_cif),
                edits=[{"op": "complete_site", "site": "O1",
                        "reason": "文献表残缺，几何补全"}],
            )
            fields = motif_provenance.classify_motif(motif)
            self.assertEqual(fields["provenance_status"],
                             "verified_source_cif")
            self.assertEqual(fields["model_level"], "crystal_edited")
            self.assertEqual(fields["source_cif_sha256"],
                             structure_model.sha256_file(user_cif))
            self.assertFalse(fields["_has_generated_cif"])

            model = structure_model.model_from_motif(motif)
            self.assertEqual(model.provenance_status, "verified_source_cif")
            self.assertEqual(model.model_level, "crystal_edited")
            self.assertEqual(model.source_cif, str(user_cif))
            self.assertTrue(model.absorber_site)
            gate = structure_model.validate_formal(model)
            self.assertTrue(gate["ok"], gate)

    def test_angular_dedup_distinguishes_planar_tetrahedral(self):
        """#19⑤：径向壳层相同但角向拓扑不同（平面 MN4 vs 四面体 MN4）
        不得判为重复——_same_fingerprint 须用角向终判，不能只看径向。
        同构型旋转后仍判重复（角向指纹旋转不变）。"""
        r = 1.96
        planar = [(r, 0, 0, "N"), (-r, 0, 0, "N"),
                  (0, r, 0, "N"), (0, -r, 0, "N")]
        t = r / np.sqrt(3.0)
        tetra = [(t, t, t, "N"), (t, -t, -t, "N"),
                 (-t, t, -t, "N"), (-t, -t, t, "N")]
        sh_p = structure_lib.geometric_shells(planar)
        sh_t = structure_lib.geometric_shells(tetra)
        self.assertEqual(sh_p[0]["el"], "N")
        self.assertEqual(sh_p[0]["n"], 4)
        self.assertAlmostEqual(sh_p[0]["r"], r, places=3)
        self.assertEqual(sh_t[0]["n"], 4)
        self.assertAlmostEqual(sh_t[0]["r"], r, places=3)
        mp = structure_lib.Motif(id="planar_probe", absorber="Fe",
                                 kind="cluster", nuclearity="m1",
                                 sites=planar, shell_expect=sh_p)
        mt = structure_lib.Motif(id="tetra_probe", absorber="Fe",
                                 kind="cluster", nuclearity="m1",
                                 sites=tetra, shell_expect=sh_t)
        self.assertFalse(
            structure_lib._same_fingerprint(mp, mt),
            "平面/四面体 MN4 径向相同但角向不同，不应判重复")
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        rot = [(x * c - y * s, x * s + y * c, z, el)
               for (x, y, z, el) in planar]
        mr = structure_lib.Motif(
            id="planar_rot", absorber="Fe", kind="cluster", nuclearity="m1",
            sites=rot, shell_expect=structure_lib.geometric_shells(rot))
        self.assertTrue(
            structure_lib._same_fingerprint(mp, mr),
            "同构型旋转后角向指纹应不变，仍判重复")

    def test_feff_cache_completion_manifest(self):
        """#19⑤：半截缓存（有 feff*.dat 但无完成清单）判不完整；清单
        原子落盘后判完整；输入哈希不符或清单文件缺失 → 不完整（重跑）。"""
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "lib_probe_key"
            cache.mkdir()
            manifest = cache / "_feff_complete.json"
            inp_hash = hashlib.sha256(b"probe-input").hexdigest()
            (cache / "feff0001.dat").write_text("x", encoding="utf-8")
            self.assertFalse(structure_lib._feff_cache_complete(
                cache, manifest, inp_hash))
            structure_lib._write_feff_manifest(
                cache, manifest, inp_hash, "K", 6.0)
            self.assertTrue(manifest.is_file())
            self.assertFalse((cache / "_feff_complete.json.tmp").exists())
            self.assertTrue(structure_lib._feff_cache_complete(
                cache, manifest, inp_hash))
            self.assertFalse(structure_lib._feff_cache_complete(
                cache, manifest, "deadbeef"))
            (cache / "feff0001.dat").unlink()
            self.assertFalse(structure_lib._feff_cache_complete(
                cache, manifest, inp_hash))

    def test_feff_cache_key_covers_rmax_edge_version(self):
        """#19⑤：缓存键须覆盖 rmax/吸收边/引擎版本——即便近邻坐标完全
        相同（旧键 md5(sites) 会复用），rmax 或边变化也必须使键变化。"""
        sites = [(2.0, 0, 0, "O"), (0, 2.0, 0, "O"),
                 (-2.0, 0, 0, "O"), (0, -2.0, 0, "O")]
        inp6 = artemis.build_feff_input(sites, absorber="Fe",
                                        rmax=6.0, edge="K")
        inp5 = artemis.build_feff_input(sites, absorber="Fe",
                                        rmax=5.0, edge="K")
        inp_l3 = artemis.build_feff_input(sites, absorber="Fe",
                                          rmax=6.0, edge="L3")
        k6 = structure_lib._feff_cache_key(inp6, "K")
        k5 = structure_lib._feff_cache_key(inp5, "K")
        k_l3 = structure_lib._feff_cache_key(inp_l3, "L3")
        self.assertNotEqual(k6, k5, "rmax 变化必须使缓存键变化")
        self.assertNotEqual(k6, k_l3, "吸收边变化必须使缓存键变化")
        orig = structure_lib._FEFF_CACHE_VERSION
        try:
            structure_lib._FEFF_CACHE_VERSION = orig + "-bump"
            self.assertNotEqual(
                k6, structure_lib._feff_cache_key(inp6, "K"),
                "引擎/卡片版本变化必须使缓存键变化")
        finally:
            structure_lib._FEFF_CACHE_VERSION = orig

    def test_b20_motifs_not_degraded(self):
        """#82 P0b：B17–B20 motif 保留、指纹不变、constructed 不升级。"""
        expected = {
            "Fe": {"fen4_planar", "fe2_mu_oxo", "fe2n6_dual"},
            "Co": {"co3o4_mixed", "co3o4_oct",
                   "co3o4_spinel_cod_9005888", "con4_planar"},
            "Ni": {"nin4_planar", "nioh2_theophrastite_cod_1548811"},
            "Cu": {"cun4_planar", "cuo_tenorite_cod_1011148"},
            "Pt": {"ptcl4_square", "pto2_beta"},
        }
        libraries = {}
        for element, ids in expected.items():
            lib = {m.id: m for m in structure_lib.load_library(element)}
            libraries[element] = lib
            for motif_id in ids:
                self.assertIn(motif_id, lib, f"{element}/{motif_id}")

        co_oct = libraries["Co"]["co3o4_oct"]
        co_tet = libraries["Co"]["co3o4_spinel_cod_9005888"]
        con4 = libraries["Co"]["con4_planar"]
        self.assertEqual(co_oct.shell_expect[0],
                         {"el": "O", "r": 1.9217, "n": 6, "n_occ": 6.0})
        self.assertEqual(co_tet.shell_expect[0],
                         {"el": "O", "r": 1.9353, "n": 4, "n_occ": 4.0})
        self.assertEqual(con4.shell_expect[0],
                         {"el": "N", "r": 1.96, "n": 4})

        for motif_id in ("co3o4_mixed", "con4_planar"):
            motif = libraries["Co"][motif_id]
            self.assertEqual(motif.provenance_status, "constructed")
            self.assertEqual(motif.model_level, "cluster")
            model = structure_model.model_from_motif(motif)
            self.assertEqual(model.provenance_status, "constructed")
            self.assertFalse(structure_model.validate_formal(model)["ok"])

        # 压力测试会用 overwrite=True 重注册种子；注册通道必须保留
        # provenance，不能把刚补好的字段静默写丢。
        with tempfile.TemporaryDirectory() as tmp:
            original_root = structure_lib.LIB_ROOT
            try:
                structure_lib.LIB_ROOT = Path(tmp)
                motif = structure_lib.Motif(
                    id="persist_check", absorber="Fe", kind="cluster",
                    nuclearity="m1", source={"type": "constructed"},
                    sites=[[2.0, 0.0, 0.0, "N"]])
                saved_path = structure_lib.register(motif)
                saved = json.loads(saved_path.read_text(encoding="utf-8"))
            finally:
                structure_lib.LIB_ROOT = original_root
        self.assertEqual(saved["provenance_status"], "constructed")
        self.assertEqual(saved["model_level"], "cluster")
        self.assertIn("edits", saved)

    def test_motif_coverage_audit(self):
        """#83 P1a：motif 覆盖审计覆盖 17/17 元素且无未分类。"""
        coverage = json.loads((ROOT / "_structure_lib" /
                               "motif_coverage.json")
                              .read_text(encoding="utf-8"))
        report = json.loads((ROOT / "_structure_lib" /
                             "provenance_report.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(coverage["schema_version"], 1)
        self.assertEqual(set(coverage["absorber_elements"]),
                         set(phase_pack.ABSORBER_ELEMENTS))
        self.assertEqual(len(coverage["absorber_elements"]), 17)
        self.assertEqual(coverage["summary"]["elements_total"], 17)
        self.assertEqual(coverage["summary"]["elements_covered"], 17)
        self.assertEqual(coverage["summary"]["elements_missing"], 0)
        self.assertEqual(coverage["summary"]["motifs"], 43)
        self.assertEqual(set(coverage["missing_elements"]), set())
        self.assertEqual(set(coverage["covered_elements"]) |
                         set(coverage["missing_elements"]),
                         set(coverage["absorber_elements"]))
        self.assertFalse(set(coverage["covered_elements"]) &
                         set(coverage["missing_elements"]))
        self.assertEqual(sum(row["motifs"]
                             for row in coverage["elements"].values()), 43)
        self.assertEqual(coverage["summary"]["formal_cn_ready_elements"], [])
        self.assertEqual(report["summary"]["unclassified"], 0)
        self.assertEqual(report["summary"]["by_status"].get(
            "verified_source_cif", 0), 0)
        self.assertEqual(report["summary"]["motifs"], 43)

    def test_phase_pack_import_and_index(self):
        """#84 P1a：44 条来源（5 金属种子 + 39 抓取目标）可校验、可索引。"""
        from tools import cif_crawler

        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        manifest = phase_pack.load_manifest(pack_root)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["pack_id"], "formal-cn-phase-pack")
        self.assertEqual(len(manifest["entries"]), 44)

        seeds = {
            "1100108": ("Fe", ["Fe1"]),
            "9008492": ("Co", ["Co"]),
            "1534892": ("Ni", ["Ni1"]),
            "4105040": ("Cu", ["Cu"]),
            "1011103": ("Pt", ["Pt1"]),
        }
        crawled = {
            "1011240": ("Fe", ["Fe1"]),
            "1010369": ("Fe", ["Fe1", "Fe2", "Fe3"]),
            "1533087": ("Co", ["Co1"]),
            "9005888": ("Co", ["Co1", "Co2"]),
            "1010093": ("Ni", ["Ni1"]),
            "1548811": ("Ni", ["Ni1"]),
            "1011148": ("Cu", ["Cu1"]),
            "1010941": ("Cu", ["Cu1"]),
            "1562989": ("Ce", ["Ce"]),
            "9008589": ("Mn", ["Mn1", "Mn2", "Mn3", "Mn4"]),
            "9008513": ("Ru", ["Ru"]),
            "9008482": ("Rh", ["Rh"]),
            "9008478": ("Pd", ["Pd"]),
            "9008459": ("Ag", ["Ag"]),
            "9008470": ("Ir", ["Ir"]),
            "9010987": ("Ce", ["Ce"]),
            "1527120": ("U", ["U1"]),
            "9008531": ("Cr", ["Cr"]),
            "9008522": ("Zn", ["Zn"]),
            "9008543": ("Mo", ["Mo"]),
            "9008463": ("Au", ["Au"]),
        }
        by_id = {}
        for entry in manifest["entries"]:
            result = phase_pack.validate_entry(pack_root, entry)
            self.assertTrue(result["ok"], (entry["id"], result))
            self.assertEqual(entry["source"], "COD")
            self.assertTrue(entry["experimental"])
            self.assertEqual(entry["model_level"], "crystal_cif")
            self.assertEqual(len(entry["absorber_elements"]), 1)
            by_id[entry["id"]] = entry
        for entry_id, (element, sites) in {**seeds, **crawled}.items():
            self.assertIn(entry_id, by_id)
            entry = by_id[entry_id]
            self.assertEqual(entry["absorber_elements"], [element])
            self.assertEqual(entry["absorber_sites"], sites)

        # P5-C：18 个常见相补齐目标必须都已导入。
        p5c_ids = {t["cod_id"] for t in cif_crawler.VERIFIED_TARGETS.values()
                   if t["tier"] == "P5C"}
        self.assertEqual(len(p5c_ids), 18)
        self.assertTrue(p5c_ids <= set(by_id), p5c_ids - set(by_id))

        index = json.loads((ROOT / "_structure_lib" /
                            "phase_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(index["pack_id"], manifest["pack_id"])
        self.assertEqual(len(index["entries"]), 44)
        self.assertEqual(set(index["summary"]["elements_covered"]), {
            "Fe", "Co", "Ni", "Cu", "Pt", "Cr", "Mn", "Zn", "Mo",
            "Ru", "Rh", "Pd", "Ag", "Ir", "Au", "Ce", "U"})
        indexed = {(e["source"], e["id"]): e for e in index["entries"]}
        for entry in manifest["entries"]:
            row = indexed[(entry["source"], entry["id"])]
            self.assertEqual(row["cif_sha256"], entry["cif_sha256"])
            self.assertEqual(row["pack_path"], entry["cif_path"])
            self.assertTrue((ROOT / row["cif_path"]).is_file())

        coverage = json.loads((ROOT / "_structure_lib" /
                               "phase_coverage.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(len(coverage["absorber_elements"]), 17)
        self.assertEqual(coverage["summary"]["elements_with_motif"], 17)
        self.assertEqual(coverage["summary"]["elements_without_motif"], 0)
        self.assertEqual(coverage["summary"]["without_motif_elements"], [])
        self.assertEqual(set(coverage["summary"]["verified_elements"]), {
            "Fe", "Co", "Ni", "Cu", "Pt", "Cr", "Mn", "Zn", "Mo",
            "Ru", "Rh", "Pd", "Ag", "Ir", "Au", "Ce", "U"})
        for element in {v[0] for v in {**seeds, **crawled}.values()}:
            row = coverage["elements"][element]
            self.assertTrue(row["formal_cn_ready"], element)
            self.assertEqual(row["coverage_status"], "verified")
            self.assertGreaterEqual(row["verified_cif"], 1)

        # 导入幂等：同一文件导入两次，manifest 与目标文件保持不变。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pack = Path(tmp) / "pack"
            src = pack_root / by_id["1011240"]["cif_path"]
            first = phase_pack.import_cif(
                src, tmp_pack, source="user", id="seed_fe",
                tags=["metal"], meta={})
            manifest_1 = (tmp_pack / "manifest.json").read_text(
                encoding="utf-8")
            second = phase_pack.import_cif(
                src, tmp_pack, source="user", id="seed_fe",
                tags=["metal"], meta={})
            manifest_2 = (tmp_pack / "manifest.json").read_text(
                encoding="utf-8")
            self.assertEqual(first["cif_sha256"], second["cif_sha256"])
            self.assertEqual(manifest_1, manifest_2)
            target_dir = tmp_pack / "user" / "seed_fe"
            self.assertEqual(len(list(target_dir.glob("*.cif"))), 1)
            self.assertEqual(len(list(target_dir.glob("*.meta.json"))), 1)
            tmp_index = phase_pack.build_index(tmp_pack)
            self.assertEqual(tmp_index["summary"]["entries"], 1)
            tmp_cov = phase_pack.build_coverage(tmp_pack, index=tmp_index)
            self.assertTrue(tmp_cov["elements"]["Fe"]["formal_cn_ready"])

    def test_phase_pack_sha256_mismatch_rejected(self):
        """#85 P1a：sha256 不符、同 ID 冲突与显式覆盖。"""
        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        manifest = phase_pack.load_manifest(pack_root)
        fe = next(e for e in manifest["entries"]
                  if e["absorber_elements"] == ["Fe"])
        source = pack_root / fe["cif_path"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "COD-1100108.cif"
            src.write_bytes(source.read_bytes())
            meta_path = src.with_suffix(".meta.json")
            meta_path.write_text(json.dumps({
                "source": "COD", "id": "1100108",
                "cif_sha256": "0" * 64,
            }), encoding="utf-8")
            with self.assertRaises(phase_pack.PhasePackError) as ctx:
                phase_pack.import_cif(src, tmp / "pack")
            self.assertEqual(ctx.exception.code, "sha256_mismatch")
            meta_path.unlink()

            first = phase_pack.import_cif(
                src, tmp / "pack", source="user", id="conflict")
            old_cif = tmp / "pack" / first["cif_path"]
            self.assertTrue(old_cif.is_file())

            changed = tmp / "changed.cif"
            changed.write_bytes(source.read_bytes() + b"\n# changed\n")
            with self.assertRaises(phase_pack.PhasePackError) as ctx:
                phase_pack.import_cif(
                    changed, tmp / "pack", source="user", id="conflict")
            self.assertEqual(ctx.exception.code, "id_sha_conflict")

            replaced = phase_pack.import_cif(
                changed, tmp / "pack", source="user", id="conflict",
                replace=True)
            self.assertNotEqual(replaced["cif_sha256"],
                                first["cif_sha256"])
            rebuilt = phase_pack.load_manifest(tmp / "pack")
            self.assertEqual(len(rebuilt["entries"]), 1)
            self.assertEqual(rebuilt["entries"][0]["cif_sha256"],
                             replaced["cif_sha256"])
            self.assertFalse(old_cif.exists())

            tampered = dict(rebuilt["entries"][0])
            tampered["cif_sha256"] = "f" * 64
            result = phase_pack.validate_entry(tmp / "pack", tampered)
            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "sha256_mismatch")

    def test_uncovered_element_no_formal_cn(self):
        """#86 P1a：generated_only motif 不升级正式 CN；phase pack 覆盖 17/17。"""
        coverage = json.loads((ROOT / "_structure_lib" /
                               "phase_coverage.json")
                              .read_text(encoding="utf-8"))
        index = json.loads((ROOT / "_structure_lib" /
                            "phase_index.json").read_text(encoding="utf-8"))
        # P4+ 的 21 条 + P5-C 的 18 条真实 CIF 已补齐全部 17 个吸收元素。
        self.assertEqual(coverage["summary"]["elements_verified"], 17)
        self.assertEqual(coverage["summary"]["elements_not_covered"], 0)
        self.assertEqual(coverage["summary"]["not_covered_elements"], [])
        self.assertEqual(coverage["summary"]["elements_motif_only"], 0)
        for element in phase_pack.ABSORBER_ELEMENTS:
            self.assertTrue(
                coverage["elements"][element]["formal_cn_ready"], element)
            self.assertEqual(
                coverage["elements"][element]["coverage_status"], "verified")
            self.assertGreaterEqual(
                len(phase_pack.verified_entries(index, element)), 1, element)
        # 但 generated_only motif 仍不能冒充正式来源。
        zn = next(m for m in structure_lib.load_library("Zn")
                  if m.id == "zn_hcp")
        self.assertEqual(zn.provenance_status, "generated_only")
        model = structure_model.model_from_motif(zn)
        gate = structure_model.validate_formal(model)
        self.assertFalse(gate["ok"])
        row = cn_table.refusal_row(
            "no_cif", gate["reason"], phase_id=model.model_id,
            source_type=model.source_type)
        self.assertEqual(row["gate_status"], "no_cif")
        self.assertEqual(row["n_fit"], "")
        self.assertEqual(row["cn_label"], "none")
        self.assertIn("refused", row["flags"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cn_table.csv"
            cn_table.write_cn_table(out, [row])
            back = cn_table.read_cn_table(out)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["gate_status"], "no_cif")
        self.assertEqual(back[0]["n_fit"], "")

    def test_phase_spec_parser_no_eval(self):
        """#87 P1b：--phase SPEC 只做字符串拆分，禁止 eval/exec。"""
        # 合法 SPEC
        spec = phase_resolver.parse_phase_spec(
            "name=fcc Fe;formula=Fe;spacegroup=Im-3m;"
            "cell=a=2.868,b=2.868,c=2.868,alpha=90,beta=90,gamma=90;"
            "source=COD;id=1100108")
        self.assertEqual(spec["name"], "fcc Fe")
        self.assertEqual(spec["formula"], "Fe")
        self.assertEqual(spec["spacegroup"], "Im-3m")
        self.assertEqual(spec["cell"]["a"], 2.868)
        self.assertEqual(spec["cell"]["gamma"], 90.0)
        self.assertEqual(spec["source"], "COD")
        self.assertEqual(spec["id"], "1100108")

        # 禁止 eval/exec：任何试图注入代码的片段都应被当作普通字符串
        # 或因不允许的键 / 格式错误而被拒绝，绝不会执行。
        # 下面这些值即使看起来像代码，也只是被当作字符串，不执行；
        # 真正应拒绝的是非法 key、缺少 '='、cell 格式错误。
        bad_key_or_format = [
            "evil=1",                 # 非法 key
            "novalue",                # 缺少 '='
            "cell=1,2,3",             # cell 不是 6 个数字
            "cell=1,2,3,4,5,6,7",     # cell 太多数字
            "cell=a=1,b=2,c=3,d=4,e=5",  # cell 缺 gamma
        ]
        for b in bad_key_or_format:
            with self.assertRaises(phase_resolver.PhaseResolutionError):
                phase_resolver.parse_phase_spec(b)

        # 代码注入字符串只被当作普通字符串，不会执行（也不报错，因为 key 合法）
        harmless = [
            "name=__import__('os')",
            "formula=eval('1+1')",
            "spacegroup=open('/etc/passwd')",
        ]
        for h in harmless:
            spec = phase_resolver.parse_phase_spec(h)

    def test_phase_cif_user_source(self):
        """#88 P1b：--phase-cif 用户真实 CIF 作为正式来源。"""
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        self.assertTrue(cif.is_file())

        # 无 absorber_site：自动取候选首个吸收位
        res = phase_resolver.resolve_phase("Fe", phase_cif=[str(cif)])
        self.assertEqual(res.status, "resolved")
        self.assertIsNotNone(res.selected)
        self.assertEqual(res.selected.source, "user")
        self.assertTrue(res.selected.cif_path.endswith(".cif"))
        self.assertEqual(res.selected.cif_sha256,
                         "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe")
        self.assertIsNotNone(res.structure_model)
        self.assertEqual(res.structure_model.provenance_status,
                         "verified_source_cif")
        self.assertEqual(res.structure_model.source_type, "user_cif")
        self.assertTrue(res.structure_model.source_ref.startswith("user:"))
        # 用户 CIF 始终优先，不是 _structure_lib/*/_cif/ 生成文件
        from structure_model import validate_formal
        v = validate_formal(res.structure_model)
        self.assertTrue(v["ok"], v)

    def test_phase_id_cache_hit(self):
        """#89 P1b：--phase-id 精确编号，本地缓存命中（幂等）。"""
        cache_root = ROOT / "_structure_lib" / "_phase_cache"
        # 先确保缓存为空（测试隔离）
        import shutil
        fe_cache = cache_root / "COD" / "1100108"
        if fe_cache.exists():
            shutil.rmtree(fe_cache)

        res = phase_resolver.resolve_phase("Fe", phase_id="COD:1100108",
                                           absorber_site="Fe1")
        self.assertEqual(res.status, "resolved")
        self.assertIsNotNone(res.selected)
        self.assertEqual(res.selected.source, "COD")
        self.assertEqual(res.selected.id, "1100108")
        self.assertIsNotNone(res.structure_model)
        self.assertEqual(res.structure_model.provenance_status,
                         "verified_source_cif")

        # 缓存文件应已写入
        meta_files = list(fe_cache.glob("*.meta.json")) if fe_cache.exists() else []
        self.assertGreaterEqual(len(meta_files), 1)

        # 再次调用：缓存命中，结果一致
        res2 = phase_resolver.resolve_phase("Fe", phase_id="COD:1100108",
                                            absorber_site="Fe1")
        self.assertEqual(res2.status, "resolved")
        self.assertEqual(res2.selected.cif_sha256,
                         res.selected.cif_sha256)
        self.assertEqual(len(list(fe_cache.glob("*.cif"))), 1)

        # 清理测试产生的缓存
        shutil.rmtree(fe_cache)

    def test_phase_id_cache_miss_offline(self):
        """#90 P1b：--phase-id 未命中且离线 → offline_miss，不退回 motif。"""
        cache_root = ROOT / "_structure_lib" / "_phase_cache"
        import shutil
        # 确保不存在该 ID 的缓存
        weird_cache = cache_root / "COD" / "99999999"
        if weird_cache.exists():
            shutil.rmtree(weird_cache)
        # phase pack 中也不存在该 ID
        res = phase_resolver.resolve_phase("Fe", phase_id="COD:99999999",
                                           offline=True)
        self.assertEqual(res.status, "offline_miss")
        self.assertIsNone(res.selected)
        self.assertIsNone(res.structure_model)
        self.assertIn("离线", res.reason)
        if weird_cache.exists():
            shutil.rmtree(weird_cache)

    def test_phase_name_ambiguous(self):
        """#91 P1b：多候选 → ambiguous，不自动 Top-1。"""
        # 给 Fe 指定 phase_index，Fe 有 1 条 verified；此处用不存在的元素
        # 制造无候选；要制造“多候选”，需在同一 pack 下有 2 条含同一吸收元素。
        # 用临时 pack 造两条 Fe CIF。
        import shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            tmp_pack = tmp / "pack"
            tmp_pack.mkdir()
            # 复制 Fe CIF 为两条不同 id（同 sha 不行，改文件名即可）
            src = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
                / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
            sha = phase_pack.sha256_file(src)
            d1 = tmp_pack / "COD" / "1100108"
            d1.mkdir(parents=True)
            shutil.copy2(src, d1 / f"{sha}.cif")
            meta1 = {"source": "COD", "id": "1100108", "formula": "Fe",
                     "spacegroup": "Im-3m", "cell": {"a": 2.868, "b": 2.868,
                     "c": 2.868, "alpha": 90, "beta": 90, "gamma": 90},
                     "absorber_sites": ["Fe1"], "absorber_elements": ["Fe"],
                     "cif_path": f"{sha}.cif", "cif_sha256": sha,
                     "experimental": True, "model_level": "crystal_cif"}
            (d1 / f"{sha}.meta.json").write_text(
                json.dumps(meta1, ensure_ascii=False, indent=2), encoding="utf-8")
            # 第二条：不同 id，同 sha（内容相同），构成另一个候选
            d2 = tmp_pack / "COD" / "1100109"
            d2.mkdir(parents=True)
            shutil.copy2(src, d2 / f"{sha}.cif")
            meta2 = dict(meta1, id="1100109")
            (d2 / f"{sha}.meta.json").write_text(
                json.dumps(meta2, ensure_ascii=False, indent=2), encoding="utf-8")
            phase_pack.rebuild_manifest(tmp_pack)

            res = phase_resolver.resolve_phase(
                "Fe", phase_index_path=str(tmp_pack / "manifest.json"),
                phase_pack_path=str(tmp_pack))
            self.assertEqual(res.status, "ambiguous")
            self.assertEqual(len(res.candidates), 2)
            self.assertIsNone(res.selected)
            self.assertIsNone(res.structure_model)
            self.assertIn("不自动 Top-1", res.reason)
        finally:
            shutil.rmtree(tmp)

    def test_phase_index_schema(self):
        """#92 P1b：phase_index.json schema 与必要字段齐全。"""
        index = json.loads((ROOT / "_structure_lib" /
                            "phase_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["schema_version"], 1)
        self.assertIn("pack_id", index)
        self.assertIn("pack_root", index)
        self.assertIn("manifest_sha256", index)
        self.assertIn("summary", index)
        self.assertEqual(index["summary"]["entries"], 44)
        self.assertEqual(index["summary"]["verified_cif"], 44)
        self.assertEqual(set(index["summary"]["elements_covered"]), {
            "Fe", "Co", "Ni", "Cu", "Pt", "Cr", "Mn", "Zn", "Mo",
            "Ru", "Rh", "Pd", "Ag", "Ir", "Au", "Ce", "U"})
        for row in index["entries"]:
            for key in ("source", "id", "formula", "spacegroup", "cell",
                        "absorber_sites", "absorber_elements", "cif_path",
                        "cif_sha256", "experimental", "model_level"):
                self.assertIn(key, row, row.get("id"))
            self.assertTrue(row["experimental"])
            self.assertIn(row["model_level"], ("crystal_cif", "crystal_edited"))
            # cif_path 必须是 repo-relative，且文件存在
            cif = ROOT / row["cif_path"]
            self.assertTrue(cif.is_file(), row["cif_path"])
            # sha256 必须与文件一致
            actual = phase_pack.sha256_file(cif)
            self.assertEqual(actual, row["cif_sha256"])

    def test_cif_gate_parse_failed(self):
        """#94 P2a：缺 _atom_site_* 循环的 CIF → parse_failed。"""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.cif"
            bad.write_text(
                "data_test\n_cell_length_a 1.0\n_cell_length_b 1.0\n",
                encoding="utf-8")
            r = cif_gate.gate_cif(bad, "Fe")
            self.assertFalse(r.ok)
            self.assertEqual(r.code, "parse_failed")
            self.assertIn("_atom_site", r.reason)

    def test_cif_gate_absorber_missing(self):
        """#95 P2a：CIF 中无吸收元素 → absorber_missing。"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "oxygen.cif"
            cif.write_text("""data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3.0 _cell_length_b 3.0 _cell_length_c 3.0
_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90
loop_
_atom_site_label _atom_site_type_symbol _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z _atom_site_occupancy
O1 O 0 0 0 1
""", encoding="utf-8")
            r = cif_gate.gate_cif(cif, "Fe")
            self.assertFalse(r.ok)
            self.assertEqual(r.code, "absorber_missing")

    def test_cif_gate_absorber_site_ambiguous(self):
        """#96 P2a：多元素结构中 Fe 有两个晶位，未指定 → absorber_site_ambiguous。"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "two_fe.cif"
            cif.write_text("""data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 4.0 _cell_length_b 4.0 _cell_length_c 4.0
_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90
loop_
_atom_site_label _atom_site_type_symbol _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z _atom_site_occupancy
Fe1 Fe 0 0 0 1
Fe2 Fe 0.5 0.5 0.5 1
O1 O 0.25 0.25 0.25 1
""", encoding="utf-8")
            r = cif_gate.gate_cif(cif, "Fe")
            self.assertFalse(r.ok)
            self.assertEqual(r.code, "absorber_site_ambiguous")
            # 指定吸收位后应放行（测试 CIF 中 Fe1 第一壳 CN=1，超出正常范围，需开关）
            r2 = cif_gate.gate_cif(cif, "Fe", absorber_site="Fe1",
                                   allow_unusual_coordination=True)
            self.assertTrue(r2.ok, r2.reason)
            self.assertEqual(r2.absorber_site, "Fe1")

    def test_cif_gate_occupancy_implicit_ok(self):
        """#97 P2a：缺 occupancy 但单元素有序 → implicit_1.0 放行（COD:9008492 Co）。"""
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "9008492" \
            / "bd9d57408d901e3ff966ba0ebe19ced70e1f1b62a4ef47fda801a39ecbb5344a.cif"
        self.assertTrue(cif.is_file())
        r = cif_gate.gate_cif(cif, "Co")
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.occupancy_source, "implicit_1.0")
        self.assertEqual(r.first_shell_cn, 12.0)   # hcp Co 第一壳 CN=12

    def test_cif_gate_occupancy_disorder_rejected(self):
        """#98 P2a：多元素结构缺 occupancy → occupancy_missing。"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "disorder.cif"
            cif.write_text("""data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 4.0 _cell_length_b 4.0 _cell_length_c 4.0
_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90
loop_
_atom_site_label _atom_site_type_symbol _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
Fe1 Fe 0 0 0
O1 O 0.5 0.5 0.5
""", encoding="utf-8")
            r = cif_gate.gate_cif(cif, "Fe")
            self.assertFalse(r.ok)
            self.assertEqual(r.code, "occupancy_missing")

    def test_cif_gate_cocupancy_unresolved(self):
        """#99 P2a：同一坐标多元素共占位，未拆分 → cocupancy_unresolved。"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "cocupancy.cif"
            cif.write_text("""data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 4.0 _cell_length_b 4.0 _cell_length_c 4.0
_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90
loop_
_atom_site_label _atom_site_type_symbol _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z _atom_site_occupancy
Fe1 Fe 0 0 0 0.5
O1 O 0 0 0 0.5
""", encoding="utf-8")
            r = cif_gate.gate_cif(cif, "Fe")
            self.assertFalse(r.ok)
            self.assertEqual(r.code, "cocupancy_unresolved")

    def test_cif_gate_occupancy_value_and_same_element_bond(self):
        """#19②：显式占据度原始值非法（未知/越界）→ occupancy_scale_failed；
        同元素第一壳键长荒谬 → bond_length_implausible；真实金属箔仍放行。"""
        pack = ROOT / "_structure_lib" / "_phase_pack" / "COD"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # 占据度未知 / 越界：必须在 read_cif_sites 钳位到 [0,1] 之前拒绝
            for idx, bad_occ in enumerate(("?", ".", "1.5", "-0.2")):
                cif = tmp / f"fe_occ_{idx}.cif"
                cif.write_text(
                    "data_test\n_symmetry_space_group_name_H-M 'P 1'\n"
                    "_cell_length_a 4.0 _cell_length_b 4.0 _cell_length_c 4.0\n"
                    "_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90\n"
                    "loop_\n_atom_site_label _atom_site_type_symbol "
                    "_atom_site_fract_x _atom_site_fract_y _atom_site_fract_z "
                    "_atom_site_occupancy\n"
                    f"Fe1 Fe 0 0 0 {bad_occ}\n", encoding="utf-8")
                r = cif_gate.gate_cif(cif, "Fe")
                self.assertFalse(r.ok, f"occ={bad_occ} 不应放行")
                self.assertEqual(r.code, "occupancy_scale_failed",
                                 f"occ={bad_occ}: {r.code} / {r.reason}")
            # 同元素第一壳键长荒谬：过短（原子重叠）/ 过长（不可能是第一近邻）
            for a, tag in ((1.4, "short"), (5.0, "long")):
                cif = tmp / f"fe_sc_{tag}.cif"
                cif.write_text(
                    "data_test\n_symmetry_space_group_name_H-M 'P 1'\n"
                    f"_cell_length_a {a} _cell_length_b {a} _cell_length_c {a}\n"
                    "_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90\n"
                    "loop_\n_atom_site_label _atom_site_type_symbol "
                    "_atom_site_fract_x _atom_site_fract_y _atom_site_fract_z "
                    "_atom_site_occupancy\n"
                    "Fe1 Fe 0 0 0 1\n", encoding="utf-8")
                r = cif_gate.gate_cif(cif, "Fe")
                self.assertFalse(r.ok, f"a={a} 同元素荒谬键长不应放行")
                self.assertEqual(r.code, "bond_length_implausible",
                                 f"a={a}: {r.code} / {r.reason}")
        # 正控：真实 hcp Co 箔（同元素第一壳 r1≈2.50, CN=12）仍放行
        co_cif = pack / "9008492" \
            / "bd9d57408d901e3ff966ba0ebe19ced70e1f1b62a4ef47fda801a39ecbb5344a.cif"
        self.assertTrue(co_cif.is_file())
        r = cif_gate.gate_cif(co_cif, "Co")
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.first_shell_cn, 12.0)

    def test_wrong_phase_consistency_failed(self):
        """#103 P2b：错误相 → N₀ 映射失败（0 CN）；正确相 → 映射通过。"""
        motifs = {m.id: m for m in structure_lib.load_library("Fe")}
        motif = motifs["fe_bcc"]

        # 正确相：Fe bcc 的 n0_shells 与 plan 匹配
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        gate = cif_gate.gate_cif(cif, "Fe")
        self.assertTrue(gate.ok)
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0, rmax=3.5, n_mode="free")
        mapped_ok = xafs_fit.map_n0_to_plan(plan, gate.n0_shells,
                                            kmin=3.0, kmax=12.0)
        self.assertTrue(mapped_ok["mapping_ok"], mapped_ok["mapping_reason"])
        self.assertGreater(len(mapped_ok["n0_per_path"]), 0)

        # 错误相：n0_shells 中的元素与 plan 的 scatterer 不匹配
        wrong_shells = [
            {"el": "Zn", "r": 2.484, "n0": 8.0, "n0_members": [],
             "sub_rayleigh_grouped": False},
            {"el": "Cu", "r": 2.868, "n0": 6.0, "n0_members": [],
             "sub_rayleigh_grouped": False},
        ]
        mapped_bad = xafs_fit.map_n0_to_plan(plan, wrong_shells,
                                             kmin=3.0, kmax=12.0)
        self.assertFalse(mapped_bad["mapping_ok"])
        self.assertIn("无匹配 N₀ 壳层", mapped_bad["mapping_reason"])
        self.assertEqual(len(mapped_bad.get("n0_per_path", [])), 0)

    def test_conditional_cn_nanoparticle_model(self):
        """#106 P3：纳米粒子缺陷模型，N/N₀ 在预期不确定度内。"""
        dm = xafs_fit.parse_defect_model(
            '{"type": "nanoparticle", "diameter_nm": 2.5, "lattice": "fcc"}')
        self.assertEqual(dm["type"], "nanoparticle")
        self.assertTrue(dm["model_dependent"])
        self.assertIn("expected_n_ratio", dm)
        self.assertIn("uncertainty", dm)
        # N/N₀ 在模型预期范围内 → defect_ok
        ratio = dm["expected_n_ratio"]
        chk = xafs_fit.check_defect_model(ratio, dm)
        self.assertTrue(chk["ok"])
        self.assertEqual(chk["status"], "defect_ok")
        # 偏离不确定度范围 → mismatch
        chk = xafs_fit.check_defect_model(ratio + 3 * dm["uncertainty"], dm)
        self.assertFalse(chk["ok"])
        self.assertEqual(chk["status"], "defect_model_mismatch")
        # 非 fcc/hcp/bcc 晶格拒绝
        with self.assertRaises(xafs_fit.XAFSConfigError):
            xafs_fit.parse_defect_model(
                '{"type": "nanoparticle", "diameter_nm": 2.5, "lattice": "diamond"}')

    def test_multiphase_no_auto_pick(self):
        """#107 P3：多相候选 → ambiguous，不自动 Top-1。"""
        import shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            tmp_pack = tmp / "pack"
            tmp_pack.mkdir()
            # 两条含 Fe 的 CIF（相同内容，不同 id）
            src = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
                / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
            sha = phase_pack.sha256_file(src)
            for mid in ("1100108", "1100109"):
                d = tmp_pack / "COD" / mid
                d.mkdir(parents=True)
                shutil.copy2(src, d / f"{sha}.cif")
                meta = {"source": "COD", "id": mid, "formula": "Fe",
                        "spacegroup": "Im-3m",
                        "cell": {"a": 2.868, "b": 2.868, "c": 2.868,
                                 "alpha": 90, "beta": 90, "gamma": 90},
                        "absorber_sites": ["Fe1"], "absorber_elements": ["Fe"],
                        "cif_path": f"{sha}.cif", "cif_sha256": sha,
                        "experimental": True, "model_level": "crystal_cif"}
                (d / f"{sha}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            phase_pack.rebuild_manifest(tmp_pack)
            res = phase_resolver.resolve_phase(
                "Fe", phase_index_path=str(tmp_pack / "manifest.json"),
                phase_pack_path=str(tmp_pack))
            self.assertEqual(res.status, "ambiguous")
            self.assertEqual(len(res.candidates), 2)
            self.assertIsNone(res.selected)
            self.assertIsNone(res.structure_model)
        finally:
            shutil.rmtree(tmp)

    def test_prototype_cluster_separate(self):
        """#109 P3：prototype/cluster 模型不进正式 CN 表。"""
        from structure_model import StructureModel, validate_formal
        # prototype
        m1 = StructureModel(
            model_id="proto_test", absorber="Fe",
            model_level="prototype", source_type="none",
            provenance_status="source_ref_only",
        )
        self.assertFalse(validate_formal(m1)["ok"])
        # cluster
        m2 = StructureModel(
            model_id="cluster_test", absorber="Fe",
            model_level="cluster", source_type="constructed",
            provenance_status="constructed",
        )
        self.assertFalse(validate_formal(m2)["ok"])

    def test_no_cif_report_empty(self):
        """#113 P4：无 CIF 样品 → cn_table 拒绝行，n_fit 为空，cn_label=none。"""
        row = cn_table.refusal_row(
            "no_cif", "无合格结构来源",
            phase_id="", model_level="", source_type="",
            source_ref="", source_cif_sha256="", absorber_site="")
        self.assertEqual(row["gate_status"], "no_cif")
        self.assertEqual(row["n_fit"], "")
        self.assertEqual(row["cn_label"], "none")
        self.assertEqual(row["reject_reason"], "无合格结构来源")
        self.assertIn("refused", row["flags"])
        # 写读往返后仍为空 CN
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cn_empty.csv"
            cn_table.write_cn_table(out, [row])
            back = cn_table.read_cn_table(out)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["n_fit"], "")
        self.assertEqual(back[0]["cn_label"], "none")

    def test_cif_crawler_target_table(self):
        """#115 P4+：39 个抓取目标 ID 唯一、元素/化学式/空间群自洽。"""
        from tools import cif_crawler
        self.assertEqual(len(cif_crawler.VERIFIED_TARGETS), 39)
        cod_ids = [t["cod_id"] for t in
                   cif_crawler.VERIFIED_TARGETS.values()]
        self.assertEqual(len(cod_ids), len(set(cod_ids)))
        seen_elements = set()
        for name, target in cif_crawler.VERIFIED_TARGETS.items():
            self.assertIn(target["tier"], ("T3", "T4", "P5C"), name)
            self.assertIn(target["absorber"],
                          phase_pack.ABSORBER_ELEMENTS, name)
            parsed = cif_crawler.parse_formula(target["formula"])
            self.assertEqual(parsed, {el: float(n) for el, n in
                                      target["stoich"].items()}, name)
            self.assertTrue(target["spg_nums"], name)
            self.assertTrue(all(isinstance(n, int) and n > 0
                                for n in target["spg_nums"]), name)
            self.assertTrue(target["tags"], name)
            seen_elements.add(target["absorber"])
        self.assertEqual(seen_elements, set(phase_pack.ABSORBER_ELEMENTS))

    def test_cif_crawler_offline_import(self):
        """#116 P4+：离线缓存 → 核验 → phase pack 导入，sha256 可追溯。"""
        import shutil
        from tools import cif_crawler

        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        manifest = phase_pack.load_manifest(pack_root)
        fe2o3 = next(e for e in manifest["entries"] if e["id"] == "1011240")
        src = pack_root / fe2o3["cif_path"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "cache"
            cache.mkdir()
            shutil.copy2(src, cache / "Fe2O3_COD_1011240.cif")
            tmp_pack = tmp / "pack"
            report_path = tmp / "report.json"
            rc = cif_crawler.main([
                "--target", "Fe2O3", "--offline",
                "--cache", str(cache), "--pack", str(tmp_pack),
                "--report", str(report_path), "--no-index",
            ])
            self.assertEqual(rc, 0)
            rebuilt = phase_pack.load_manifest(tmp_pack)
            self.assertEqual(len(rebuilt["entries"]), 1)
            entry = rebuilt["entries"][0]
            self.assertEqual(entry["source"], "COD")
            self.assertEqual(entry["id"], "1011240")
            self.assertEqual(entry["absorber_sites"], ["Fe1"])
            self.assertEqual(entry["cif_sha256"],
                             phase_pack.sha256_file(src))
            self.assertIn("source_url=", entry.get("notes", ""))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["targets"]["Fe2O3"]["status"],
                             "imported")
            self.assertEqual(report["summary"]["imported"], 1)

    def test_cif_crawler_offline_miss_and_wrong_cif(self):
        """#117 P4+：离线缺缓存 → offline_miss；错相 CIF → 拒绝导入。"""
        import shutil
        from tools import cif_crawler

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = tmp / "cache"
            cache.mkdir()
            tmp_pack = tmp / "pack"
            report_path = tmp / "report.json"
            rc = cif_crawler.main([
                "--target", "CeO2", "--offline",
                "--cache", str(cache), "--pack", str(tmp_pack),
                "--report", str(report_path), "--no-index",
            ])
            self.assertEqual(rc, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["targets"]["CeO2"]["status"],
                             "offline_miss")
            self.assertFalse((tmp_pack / "manifest.json").exists())

            # 把 Fe3O4 错相当成 CeO2 缓存：必须被化学计量核验拒绝。
            pack_root = ROOT / "_structure_lib" / "_phase_pack"
            manifest = phase_pack.load_manifest(pack_root)
            fe3o4 = next(e for e in manifest["entries"]
                         if e["id"] == "1010369")
            shutil.copy2(pack_root / fe3o4["cif_path"],
                         cache / "CeO2_COD_1562989.cif")
            rc = cif_crawler.main([
                "--target", "CeO2", "--offline",
                "--cache", str(cache), "--pack", str(tmp_pack),
                "--report", str(report_path), "--no-index",
            ])
            self.assertEqual(rc, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = report["targets"]["CeO2"]
            self.assertEqual(row["status"], "cache_invalid")
            self.assertIn("composition_mismatch", row["reason"])
            self.assertFalse((tmp_pack / "manifest.json").exists())

    def test_multi_site_phase_requires_absorber_site(self):
        """#118 P4+：Fe3O4 多个吸收位必须显式 --absorber-site。"""
        res = phase_resolver.resolve_phase("Fe", phase_id="COD:1010369")
        self.assertEqual(res.status, "resolved")
        self.assertIsNotNone(res.selected)
        self.assertEqual(res.selected.id, "1010369")
        self.assertEqual(len(res.selected.absorber_sites), 3)
        self.assertIsNone(res.structure_model)
        self.assertIn("多个吸收位点", res.reason)
        self.assertIn("--absorber-site", res.reason)

        res_ok = phase_resolver.resolve_phase(
            "Fe", phase_id="COD:1010369", absorber_site="Fe1")
        self.assertIsNotNone(res_ok.structure_model)
        self.assertEqual(res_ok.structure_model.absorber_site, "Fe1")

        res_bad = phase_resolver.resolve_phase(
            "Fe", phase_id="COD:1010369", absorber_site="Fe9")
        self.assertIsNone(res_bad.structure_model)
        self.assertIn("没有吸收位", res_bad.reason)

    def test_p5c_common_phase_coverage(self):
        """#125 P5-C：18 个固定 COD ID 补齐常见相，coverage 无缺口。"""
        from types import SimpleNamespace
        from tools import cif_crawler
        import phase_resolver

        expected = {
            ("Fe", "FeO"), ("Fe", "FeOOH"), ("Pt", "PtO2"),
            ("Cr", "Cr2O3"), ("Mn", "MnO"), ("Mn", "MnO2"),
            ("Mn", "Mn3O4"), ("Zn", "ZnO"), ("Mo", "MoO2"),
            ("Mo", "MoO3"), ("Ru", "RuO2"), ("Rh", "Rh2O3"),
            ("Pd", "PdO"), ("Ag", "Ag2O"), ("Ir", "IrO2"),
            ("Au", "Au2O3"), ("U", "UO2"), ("U", "U3O8"),
        }
        p5c = {name: target for name, target in
               cif_crawler.VERIFIED_TARGETS.items()
               if target["tier"] == "P5C"}
        self.assertEqual(len(p5c), 18)
        self.assertEqual({(t["absorber"], name) for name, t in p5c.items()},
                         expected)
        selected = cif_crawler._select_targets(SimpleNamespace(
            all=False, t3=False, t4=False, p5c=True, target=None))
        self.assertEqual(len(selected), 18)

        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        manifest = phase_pack.load_manifest(pack_root)
        by_id = {entry["id"]: entry for entry in manifest["entries"]}
        for name, target in p5c.items():
            self.assertIn(target["cod_id"], by_id, name)
            entry = by_id[target["cod_id"]]
            self.assertEqual(entry["formula"], target["formula"], name)
            self.assertEqual(entry["absorber_elements"],
                             [target["absorber"]], name)
            self.assertEqual(entry["model_level"], "crystal_cif", name)
            self.assertTrue(entry["experimental"], name)
            self.assertEqual(entry["source"], "COD", name)
            check = phase_pack.validate_entry(pack_root, entry)
            self.assertTrue(check["ok"], (name, check))

        coverage = json.loads((ROOT / "_structure_lib" /
                               "phase_coverage.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(coverage["summary"]["verified_cif"], 44)
        for element, row in coverage["elements"].items():
            self.assertEqual(row["missing_common_phases"], [], element)
            self.assertTrue(row["formal_cn_ready"], element)

        # 多位点常见相仍必须显式 --absorber-site，不能自动 Top-1。
        for element, phase_name in (("Mn", "Mn3O4"), ("U", "U3O8")):
            target = p5c[phase_name]
            entry = by_id[target["cod_id"]]
            self.assertGreaterEqual(len(entry["absorber_sites"]), 2,
                                    phase_name)
            res = phase_resolver.resolve_phase(
                element, phase_id=f"COD:{target['cod_id']}")
            self.assertEqual(res.status, "resolved", phase_name)
            self.assertIsNone(res.structure_model, phase_name)
            self.assertIn("多个吸收位点", res.reason, phase_name)

    def test_p5_standard_source_isolation_pool(self):
        """#126 P5-prep/P5-S02：5 元素谱源池闭环与门禁冻结。"""
        from tools import build_source_pool

        pool = json.loads((ROOT / "data" / "s02_calibration" /
                           "source_pool" / "index.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(pool["schema_version"], 2)
        self.assertEqual(pool["pool_id"], "s02-standard-source-isolation")
        self.assertEqual(pool["elements_total"], 5)
        self.assertEqual(pool["pool_state_counts"], {"resolved": 5})
        # 索引必须能由标定产物与门禁常量确定性重建。
        self.assertEqual(build_source_pool.build_pool(), pool)

        rows = {row["element"]: row for row in pool["elements"]}
        self.assertEqual(set(rows), {"Rh", "Pd", "Ir", "Ce", "U"})
        expected_class = {
            "Rh": "short_spectrum_replace",
            "Pd": "short_spectrum_replace",
            "Ir": "short_spectrum_replace",
            "Ce": "oxide_baseline_reprocess",
            "U": "oxide_phase_verification",
        }
        expected_s02 = {
            "Rh": 0.7022, "Pd": 0.6559, "Ir": 0.4752,
            "Ce": 0.6279, "U": 0.7800,
        }
        for element, row in rows.items():
            self.assertEqual(row["classification"], expected_class[element])
            self.assertEqual(row["pool_state"], "resolved")
            self.assertTrue(row["usable_for_s02"])
            self.assertAlmostEqual(row["s02"], expected_s02[element],
                                   places=4, msg=element)
            self.assertEqual(row["last_status"], "calibrated")
            self.assertEqual(row["attempt_status"], "passed")
            self.assertEqual(row["gate_failures"], [])
            self.assertGreater(row["k_max_estimate"], 10.0, element)
            # #60：静态表仍然 s02=None / calibrated=False。
            cfg = get_absorber(element)
            self.assertIsNone(cfg.s02, element)
            self.assertFalse(cfg.calibrated, element)
            # 旧谱的历史隔离证据必须保留，不能因为替换而丢失审计链。
            legacy = row["quarantined_legacy_source"]
            self.assertEqual(legacy["attempt_status"], "gated_out")
            self.assertTrue(legacy["gate_failures"], element)
            self.assertEqual(len(legacy["source_sha256"]), 64, element)
            self.assertEqual(row["resolved_by"]["gate_failures"], [])
            self.assertEqual(row["resolved_by"]["standard"],
                             row["current_source"])

        # 旧谱谱长判定：Rh/Pd/Ir 是短谱；Ce/U 不是截断问题。
        for element in ("Rh", "Pd", "Ir"):
            self.assertLess(
                rows[element]["quarantined_legacy_source"]["k_max_estimate"],
                6.0, element)
        self.assertGreater(
            rows["Ce"]["quarantined_legacy_source"]["k_max_estimate"], 10.0)
        self.assertGreater(
            rows["U"]["quarantined_legacy_source"]["k_max_estimate"], 10.0)

        # Ir 的 k_min 建议只能换谱后诊断，不能作为通过手段。
        ir = rows["Ir"]
        self.assertFalse(ir["kmin_tuning_assessment"]
                         ["sufficient_with_current_source"])
        self.assertFalse(ir["kmin_tuning_assessment"]["may_relax_gates"])
        self.assertEqual(ir["resolved_by"]["kmin"], 3.0)

        # Ce 使用 CeO2 基准；U 的 UO2 切换已完成并记录标定专用 motif。
        self.assertFalse(rows["Ce"]["phase_switch_needed"])
        self.assertIn("Ce_metal_foil.dat", rows["Ce"]["current_source"])
        self.assertIn("CeO2", rows["Ce"]["phase_baseline"])
        self.assertFalse(rows["U"]["phase_switch_needed"])
        self.assertIn("UO2", rows["U"]["phase_baseline"])
        self.assertEqual(rows["U"]["resolved_by"]["motif"], "uo2_fluorite")

        # 门禁冻结：与 calibrate_s02.py 常量一致，且未放宽。
        policy = pool["policy"]
        self.assertEqual(policy["s02_pass_range"], [0.4, 1.15])
        self.assertEqual(policy["s02_err_max"], 0.06)
        self.assertEqual(policy["rf_max"], 0.12)
        self.assertEqual(policy["abs_delta_e0_max_ev"], 10.0)
        self.assertEqual(policy["window_fracs"], [0.12, 0.18, 0.26])
        self.assertEqual(policy["window_spread_max"], 0.05)
        self.assertEqual(policy["delta_r_limits"], [-0.03, 0.03])
        self.assertEqual(policy["gate_relaxation"], "forbidden")
        self.assertEqual(policy["manual_window_trimming"], "forbidden")
        self.assertEqual(policy["auto_top1_selection"], "forbidden")

        # T5 总索引：5 元素全部 calibrated、s02 非空。
        t5_index = json.loads((ROOT / "data" / "s02_calibration" /
                               "index.json").read_text(encoding="utf-8"))
        t5 = {row["absorber"]: row for row in t5_index["elements"]}
        for element in rows:
            self.assertEqual(t5[element]["status"], "calibrated", element)
            self.assertIsNotNone(t5[element]["s02"], element)

    def test_p6_all_phase_pack_entries_pass_cif_gate(self):
        """#128 P6：phase_pack 全部 44 个入库晶体相 100% 通过 cif_gate。"""
        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        manifest = phase_pack.load_manifest(pack_root)
        self.assertEqual(len(manifest["entries"]), 44)

        single_count = 0
        multi_count = 0
        for entry in manifest["entries"]:
            el = entry["absorber_elements"][0]
            cif_path = pack_root / entry["cif_path"]
            sites = entry["absorber_sites"]
            is_comp = not entry.get("experimental", True)
            if len(sites) == 1:
                single_count += 1
                g = cif_gate.gate_cif(cif_path, el, absorber_site=None,
                                      is_computed=is_comp)
                self.assertTrue(
                    g.ok, f"{entry['id']} {entry['formula']} failed: "
                          f"{g.code} ({g.reason})")
                self.assertEqual(g.absorber_site, sites[0])
                self.assertGreaterEqual(g.first_shell_cn, 2.0)
                self.assertLessEqual(g.first_shell_cn, 16.0)
            else:
                multi_count += 1
                g_amb = cif_gate.gate_cif(cif_path, el, absorber_site=None,
                                          is_computed=is_comp)
                parsed = phase_pack.parse_cif(cif_path)
                if len(parsed["formula_elements"]) == 1:
                    # 单元素多晶位：所有位点对称等价，自动选第一个
                    self.assertTrue(g_amb.ok, g_amb.reason)
                else:
                    # 多元素多晶位：未传 --absorber-site 必须继续拒绝
                    self.assertFalse(g_amb.ok)
                    self.assertEqual(g_amb.code, "absorber_site_ambiguous")
                gs = cif_gate.gate_cif_all_sites(
                    cif_path, el, is_computed=is_comp)
                self.assertEqual(len(gs), len(sites))
                self.assertTrue(
                    all(g.ok for g in gs),
                    f"{entry['id']} {entry['formula']} multi-site failed: "
                    f"{[(g.absorber_site, g.code, g.reason) for g in gs]}")
                self.assertAlmostEqual(
                    sum(g.site_weight for g in gs), 1.0, places=9)
        self.assertEqual(single_count, 39)
        self.assertEqual(multi_count, 5)

        # 催化位点假设模型与 --formal-cn 同时开启：明确写 hypothesis_model_only
        from larch.xafs import feffpath
        from types import SimpleNamespace

        motif = structure_lib.parse_catalytic_cluster_spec(
            "Fe", "sac:N4@2.00,axial=O@2.10,C=none")
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmin=1.0, rmax=2.6, kmin=3.0, kmax=12.0)
        k = np.arange(3.0, 12.0, 0.05)
        plist = []
        for sh, f in zip(plan["shells"], plan["files"]):
            p = feffpath(str(f), s02=0.803, sigma2=0.004, e0=2.0,
                         deltar=0.0)
            p.degen = float(sh["degen"])
            p._calc_chi()
            plist.append(p)
        chi = artemis.exafs_model_k(plist, k)
        rng = np.random.default_rng(7)
        chi = chi + rng.normal(0.0, 1e-3, size=chi.shape)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.ones_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        res = SimpleNamespace(
            chi_k=chi_k, ft=ft, absorber_symbol="Fe", structure=None,
            merged=None, norm=None, self_absorption=None)
        out = xafs_fit.fit_identified(
            res, absorber_symbol="Fe",
            catalytic_model="sac:N4@2.00,axial=O@2.10,C=none",
            rmin=1.0, rmax=2.6, verbose=False, formal_cn=True)
        self.assertEqual(out["formal_cn"]["status"], "hypothesis_model_only")
        self.assertEqual(out["cn_status"], "hypothesis_model_only")

    def test_p6_3_entry_parity_and_operando_guards(self):
        """#131 P6.3：双入口配置键对齐、operando 冲突守卫、B10 配置接线与 S3 提示。"""
        import argparse
        import contextlib
        import io
        from unittest.mock import patch

        # 1) P1b/P5 配置合并：9 键 + post_window；CLI 显式值优先
        args = argparse.Namespace(
            s02_from_t5=False, xanes_cross_check=False, offline=False,
            phase_id=None, phase=None, phase_index=None, phase_pack=None,
            absorber_site=None, phase_cif=None, post_window=None)
        cfg = {
            "s02_from_t5": True, "xanes_cross_check": True, "offline": True,
            "phase_cif": "a.cif,b.cif", "phase_id": "COD:9007541",
            "phase": "name=rutile", "phase_index": "idx.json",
            "phase_pack": "pack", "absorber_site": "all",
            "post_window": [180, 650],
        }
        xafs_config.apply_p1b_config(args, cfg)
        self.assertTrue(args.s02_from_t5)
        self.assertTrue(args.xanes_cross_check)
        self.assertTrue(args.offline)
        self.assertEqual(args.phase_cif, ["a.cif", "b.cif"])
        self.assertEqual(args.phase_id, "COD:9007541")
        self.assertEqual(args.phase, "name=rutile")
        self.assertEqual(args.phase_index, "idx.json")
        self.assertEqual(args.phase_pack, "pack")
        self.assertEqual(args.absorber_site, "all")
        self.assertEqual(args.post_window, (180.0, 650.0))

        args2 = argparse.Namespace(
            s02_from_t5=True, xanes_cross_check=False, offline=False,
            phase_id="COD:1", phase=None, phase_index=None, phase_pack=None,
            absorber_site="Fe1", phase_cif=["explicit.cif"],
            post_window="140,620")
        xafs_config.apply_p1b_config(args2, cfg)
        self.assertEqual(args2.phase_id, "COD:1")
        self.assertEqual(args2.absorber_site, "Fe1")
        self.assertEqual(args2.phase_cif, ["explicit.cif"])
        self.assertEqual(args2.post_window, "140,620")

        # 2) B10 配置键接线：target_transform / mcr_als
        kw = xafs_config.operando_kwargs_from_config({
            "target_transform": {"spoil_good": 2.5, "spoil_poor": 5.0},
            "mcr_als": {"n_components": 3, "max_iter": 500, "closure": False},
        })
        self.assertEqual(kw, {"spoil_good": 2.5, "spoil_poor": 5.0,
                              "n_components": 3, "max_iter": 500,
                              "closure": False})

        # 3) --operando 与 S6.5/S7 参数显式冲突（不再静默忽略）
        import run_xafs
        for extra in (["--formal-cn", "--fit"], ["--fit"],
                      ["--xanes-cross-check"]):
            argv = (["run_xafs.py", str(FE_FOIL), str(FE_FOIL), str(FE_FOIL),
                     "--operando", "--absorber", "Fe"] + extra)
            with patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_xafs.main()

        # 4) 新入口开关在后端可见（--help 即解析器全量）
        for module, flag in (
                (run_xafs, "--allow-unusual-coordination"),
                (run_xafs, "--post-window"),
                (xafs_report, "--xanes-cross-check"),
                (xafs_report, "--allow-unusual-coordination"),
                (xafs_report, "--post-window"),
                (xafs_report, "--phase-id")):
            buf = io.StringIO()
            with patch.object(sys, "argv", ["prog", "--help"]), \
                    contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main()
            self.assertIn(flag, buf.getvalue(), f"{module.__name__} 缺少 {flag}")

        # 5) S3 strict 报错给出 --no-strict 出口
        with self.assertRaises(XAFSError) as ctx:
            xafs_pipeline.process_sample([FE_FOIL], absorber="Fe",
                                         strict=True, verbose=False)
        self.assertIn("归一化失败", str(ctx.exception))
        self.assertIn("--no-strict", str(ctx.exception))


    def test_unit3a_anti_bypass_and_silent_fallback_guards(self):
        """Unit 3A 审计反例：
        1) 12 锁定元素在 --chemical-state / L1 入口均不可绕过 s02=None；
        2) build_algorithmic_catalytic_motif 不再 NameError、Co3O4 正确对称展开、
           未知 host / 缺失中心原子 / 非 P1 无 gemmi / 缺失 replace_el 均显式拒绝；
        3) parse_cif_simple 保留异核共占位双方原子；
        4) cluster motif 即使挂外部 cif_path 也不得升级为 verified_source_cif；
        5) structure_id 四舍五入边界 (0.16004 / 0.02992) 不得软放宽门禁。
        """
        import absorber as absorber_mod
        import catalytic_builder as cb
        from types import SimpleNamespace
        from unittest.mock import patch

        # 1) 12 锁定元素不可被 chemical-state 绕过；L1 入口早拦截（不跑 FEFF）
        for sym in ("Cr", "Mn", "Zn", "Mo", "Ru", "Rh",
                    "Pd", "Ag", "Ir", "Au", "Ce", "U"):
            self.assertEqual(
                absorber_mod.resolve_s02_spec(sym, "Fe=metal"),
                (None, None))
            with self.assertRaises(XAFSConfigError):
                absorber_mod.resolve_s02(sym, "metal")
            dummy_res = SimpleNamespace(
                ft=self.lep.ft, r1=2.0, absorber_symbol=sym)
            with self.assertRaises(XAFSError):
                structure_id.l1_first_shell_fit(
                    dummy_res, absorber_symbol=sym, verbose=False)

        # 2) 算法催化建模：修复 NameError、Co3O4 展开与未知 host 静默回退
        m_co = structure_lib.build_algorithmic_catalytic_motif(
            "Co", {"host": "Co3O4"})
        self.assertEqual(m_co.absorber, "Co")
        self.assertGreater(len(m_co.sites), 10)
        with self.assertRaises(XAFSConfigError):
            structure_lib.build_algorithmic_catalytic_motif(
                "Fe", {"host": "Zirconia"})
        with self.assertRaises(XAFSConfigError):
            structure_lib.build_algorithmic_catalytic_motif(
                "Fe", {"host": "TiO2", "center": "Pt"})
        with self.assertRaises(ValueError):
            cb.apply_bulk_doping(
                [("Ti", 0.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)],
                replace_el="Zr", dopant_el="Fe")

        # 3) parse_cif_simple 保留异核共占位原子
        hetero_cif = """data_hetero_probe
_cell_length_a 4.10
_cell_length_b 4.10
_cell_length_c 4.10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'Pm-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0.0 0.0 0.0 1.0
Ti1 Ti 0.5 0.5 0.5 0.7
Ru1 Ru 0.5 0.5 0.5 0.3
"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "hetero.cif"
            cif.write_text(hetero_cif, encoding="utf-8")
            cm = cb.parse_cif_simple(cif)
            els = {a[0] for a in cm.atoms}
            self.assertEqual(els, {"Fe", "Ti", "Ru"})

            # 4) cluster motif 挂真实 CIF 仍不可升级为 verified_source_cif
            cl = structure_lib.Motif(
                id="cluster_with_cif", absorber="Fe", kind="cluster",
                nuclearity="m1", cif_path=str(cif),
                absorber_site="Fe1", sites=[[2.0, 0.0, 0.0, "O"]])
            sm = structure_model.model_from_motif(cl)
            self.assertNotEqual(sm.provenance_status, "verified_source_cif")
            self.assertFalse(structure_model.validate_formal(sm)["ok"])

        # 5) structure_id 边界分数 0.16004 不得因 round(4) 变成 confident
        fake_lib = [
            structure_lib.Motif(
                id="m_a", absorber="Fe", kind="cluster", nuclearity="m1",
                sites=[[2.0, 0.0, 0.0, "O"]]),
            structure_lib.Motif(
                id="m_b", absorber="Fe", kind="cluster", nuclearity="m1",
                sites=[[2.1, 0.0, 0.0, "O"]]),
        ]
        k = self.lep.chi_k.k
        chi0 = np.sin(2.0 * 2.0 * k)
        with patch.object(structure_lib, "load_library", return_value=fake_lib), \
             patch.object(structure_id, "theory_chi", return_value=chi0), \
             patch.object(structure_id, "complex_score", side_effect=[0.16004] * 27 + [0.25] * 27):
            out = structure_id.identify(
                self.lep.ft, k, absorber_symbol="Fe",
                kmin=self.lep.ft.kmin, kmax=self.lep.ft.kmax,
                kweight=self.lep.ft.kweight, verbose=False)
            self.assertEqual(out["status"], "no_match")


    def test_unit3b_spec_cell_and_tamper_rejected(self):
        """Unit 3B：含 cell= 的 --phase SPEC 可匹配；篡改 CIF/哈希/注释绕过必须拒绝。"""
        # 1. 含 cell= 的 --phase SPEC 不得因二次解析 dict 失败
        res = phase_resolver.resolve_phase(
            "Fe",
            phase_spec=(
                "name=Fe;formula=Fe;spacegroup=Im-3m;"
                "cell=a=2.868,b=2.868,c=2.868,alpha=90,beta=90,gamma=90;"
                "source=COD;id=1100108"
            ),
            absorber_site="Fe1",
        )
        self.assertEqual(res.status, "resolved", res.reason)
        self.assertIsNotNone(res.structure_model)
        self.assertEqual(res.selected.id, "1100108")

        # 2. phase_index 中 CIF 被篡改时，resolve_phase 必须拒绝
        pack_root = ROOT / "_structure_lib" / "_phase_pack"
        src = (pack_root / "COD" / "1100108"
               / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_pack = tmp / "pack"
            entry = phase_pack.import_cif(
                src, tmp_pack, source="COD", id="1100108")
            tampered_cif = tmp_pack / entry["cif_path"]
            tampered_cif.write_bytes(tampered_cif.read_bytes() + b"\n# tampered\n")
            with self.assertRaises(phase_resolver.PhaseResolutionError) as ctx:
                phase_resolver.resolve_phase(
                    "Fe",
                    phase_index_path=str(tmp_pack / "manifest.json"),
                    phase_pack_path=str(tmp_pack),
                )
            self.assertEqual(ctx.exception.code, "sha256_mismatch")

            # 3. motif_provenance：伪造/过期 source_cif_sha256 不得升级为 verified_source_cif
            user_cif = tmp / "user_lep.cif"
            user_cif.write_bytes((ROOT / "lepidocrocite.cif").read_bytes())
            bad_motif = structure_lib.Motif(
                id="tampered_motif",
                absorber="Fe",
                kind="crystal",
                nuclearity="crystal",
                source={"type": "user_cif", "ref": "user:tamper"},
                cell={"a": 3.072, "b": 12.516, "c": 3.873,
                      "alpha": 90, "beta": 90, "gamma": 90},
                spacegroup="C m c m",
                asym_sites=[],
                absorber_fract=[0.5, 0.1778, 0.25],
                cif_path=str(user_cif),
                source_cif_sha256="0" * 64,
            )
            classified = motif_provenance.classify_motif(bad_motif)
            self.assertEqual(classified["provenance_status"], "unverified")
            self.assertIsNone(classified["source_cif_sha256"])

            # 4. cif_gate：注释含 _atom_site_occupancy 不得绕过 occupancy_missing
            comment_bypass = tmp / "comment_bypass.cif"
            comment_bypass.write_text(
                "data_test\n# _atom_site_occupancy\n"
                "_symmetry_space_group_name_H-M 'P 1'\n"
                "_cell_length_a 4.0 _cell_length_b 4.0 _cell_length_c 4.0\n"
                "_cell_angle_alpha 90 _cell_angle_beta 90 _cell_angle_gamma 90\n"
                "loop_\n_atom_site_label _atom_site_type_symbol "
                "_atom_site_fract_x _atom_site_fract_y _atom_site_fract_z\n"
                "Fe1 Fe 0 0 0\nO1 O 0.5 0.5 0.5\n",
                encoding="utf-8",
            )
            g = cif_gate.gate_cif(comment_bypass, "Fe")
            self.assertFalse(g.ok)
            self.assertEqual(g.code, "occupancy_missing")
