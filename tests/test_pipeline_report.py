# -*- coding: utf-8 -*-
"""Unit 6：CLI/管线/出图/报告/溯源

纯搬运拆分：测试方法体与原 regression_test.py 逐行一致，断言未做任何修改。
"""
from __future__ import annotations

from tests.fixtures import *  # noqa: F401,F403
from tests.fixtures import Fixtures, _find  # noqa: F401

# 历史测试体里有 3 处直接用 Path(__file__) 定位 data/（原文件在仓库根）。
# 拆到 tests/ 后模块位置变了，把 __file__ 指回等价的根入口文件，
# 保证测试方法体逐字节不变（不改任何断言或路径写法）。
__file__ = str(ROOT / "regression_test.py")


class PipelineReportTests(Fixtures):

    def test_indistinguishable_excludes_best(self):
        sid = self.lep_fit["structure"]
        best = sid["ranked"][0]["id"]
        self.assertEqual(best, "lepidocrocite")
        self.assertNotIn(best, sid["indistinguishable"])
        self.assertEqual(sid["status"], "confident")
        self.assertGreater(sid["confidence"]["margin"], 0.02)

    def test_strict_gate_and_downgrade(self):
        with self.assertRaises(XAFSError) as ctx:
            xafs_pipeline.process_sample([FE_FOIL], absorber="Fe",
                                         strict=True, verbose=False)
        self.assertIn("归一化失败", str(ctx.exception))

        res = xafs_pipeline.process_sample(
            [FE_FOIL], absorber="Fe", strict=False, verbose=False)
        self.assertLess(res.norm.plateau, 0.85,
                        "该标样本应触发平台门禁，否则测试前提失效")

    def test_lepidocrocite_frozen_numerical_and_csv_baseline(self):
        import csv
        import hashlib
        import io

        reference = json.loads(
            (ROOT / "data" / "xafs_validation_baseline.json").read_text(
                encoding="utf-8"))
        protocol = reference["protocol"]
        self.assertEqual(get_absorber("Fe").s02, protocol["s02"])
        for key in ("single_scan", "three_scan"):
            with self.subTest(case=key):
                expected = reference[key]
                paths = [ROOT / name for name in expected["scans"]]
                self.assertTrue(all(path.is_file() for path in paths))
                self.assertEqual(cn_table.data_sha256(paths),
                                 expected["data_sha256"])
                fit_out = self._fit_frozen_baseline(reference, key)
                self.assertTrue(fit_out["cn_reliable"])
                fit = fit_out["fit"]
                self.assertLess(fit.n_variables, fit.n_independent)
                # 跨平台 BLAS/SciPy/Python 小版本会引起 ~1e-14 R-factor 与微电子伏级 ΔE0
                # 收敛终止浮点尾差；保持严格科学精度容差同时维持 CSV SHA256 逐位一致。
                self.assertAlmostEqual(fit.r_factor, expected["r_factor"], places=10)
                self.assertAlmostEqual(fit.delta_e0, expected["delta_e0"], places=4)
                self.assertEqual(len(fit.paths), len(expected["paths"]))
                for got_p, exp_p in zip(fit.paths, expected["paths"]):
                    self.assertAlmostEqual(got_p.degeneracy, exp_p["N"], places=5)
                    self.assertAlmostEqual(got_p.r, exp_p["R"], places=6)
                    self.assertAlmostEqual(got_p.sigma2, exp_p["sigma2"], places=8)
                rows = xafs_report.fit_rows_from_output(
                    "lepidocrocite", fit_out)
                with tempfile.TemporaryDirectory() as tmp:
                    table = xafs_report.export_fit_table(
                        rows, Path(tmp), "baseline")
                    text = table.read_text(encoding="utf-8-sig")
                self.assertTrue(all(len(row) == 7 for row in
                                    csv.reader(io.StringIO(text))))
                self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 expected["csv_sha256"])

    def test_frozen_baseline_uses_declared_protocol(self):
        from unittest.mock import patch
        from types import SimpleNamespace

        reference = json.loads((ROOT / "data" / "xafs_validation_baseline.json")
                               .read_text(encoding="utf-8"))
        reference["protocol"] = dict(reference["protocol"],
                                     e0_method="half_height", strict=False,
                                     kmin=3.1, kmax=11.8, kweight=3, s02=0.9)
        for key in ("single_scan", "three_scan"):
            with patch.object(xafs_pipeline, "process_sample",
                              return_value=SimpleNamespace()) as process, patch.object(
                    xafs_fit, "run", return_value={"fit": "sentinel"}) as fit:
                self.assertEqual(self._fit_frozen_baseline(reference, key), "sentinel")
            args = process.call_args.kwargs
            for field in ("absorber", "kmin", "kmax", "kweight", "strict", "e0_method"):
                self.assertEqual(args[field], reference["protocol"][field])
            self.assertEqual(args["align"], reference[key]["align"])
            self.assertEqual(fit.call_args.kwargs["s02"], 0.9)
            self.assertTrue(fit.call_args.kwargs["identify"])
            self.assertTrue(fit.call_args.kwargs["fit"])
            self.assertNotIn("motif_id", fit.call_args.kwargs)

    def test_shell_selection_respects_n_idp(self):
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "goethite"][0]
        plan = xafs_fit.select_shell_paths(
            motif, rmax=3.5, merge_tol=0.20, kmin=3.0, kmax=12.0)
        self.assertEqual(plan["shells"][0]["label"], "Fe-O@2.03")
        self.assertEqual(plan["shells"][0]["degen"], 6.0)
        self.assertEqual(plan["shells"][1]["label"], "Fe-Fe@3.02")
        self.assertEqual(plan["max_paths_by_idp"], 3)
        self.assertTrue(plan["truncated"])

    def test_sigma2_shared_group(self):
        """σ² 共享组：多条路径共享一个 σ² 变量，减少自由参数。

        对标 Artemis 的路径约束。用 NiO 双壳层（Ni-O + Ni-Ni）验证：
        - shared_sigma2="all"：两条路径共享一个 σ²，拟合收敛，Rf 合理
        - 显式分组 [[0, 1]]：等价于 "all"（两条路径都在同一组）
        - 共享组输出的 σ² 值在两条路径中相同
        若 σ² 共享组机制失效，拟合会报错或输出不同的 σ²。
        """
        res = xafs_pipeline.process_sample(
            [NIO], absorber="Ni", strict=False, verbose=False)
        out = xafs_fit.fit_identified(
            res, absorber_symbol="Ni", shared_sigma2="all", verbose=False)
        fit = out["fit"]
        self.assertEqual(out["mode"], "motif")
        self.assertTrue(out["sigma2_shared"])
        self.assertEqual(len(out["sigma2_groups"]), 1)
        n_path = len(out["fit"].paths)
        self.assertEqual(out["sigma2_groups"][0], list(range(n_path)))
        # 所有路径的 σ² 应相同（共享同一变量）
        s2s = [p.sigma2 for p in fit.paths]
        for s2 in s2s:
            self.assertAlmostEqual(s2, s2s[0], places=6)
        # σ²_err 也应相同
        e0s = [p.sigma2_err for p in fit.paths]
        for e in e0s:
            self.assertIsNotNone(e)
        for e in e0s:
            self.assertAlmostEqual(e, e0s[0], places=6)
        # Rf 应在合理范围（双壳层 NiO 合成谱，Rf 应 < 0.02）
        self.assertLess(fit.r_factor, 0.02,
                        f"σ² 共享后 Rf={fit.r_factor:.4f} 过大，拟合可能失败")

        # 显式分组 [[0, 1, ..., n-1]] 等价于 "all"
        out2 = xafs_fit.fit_identified(
            res, absorber_symbol="Ni",
            shared_sigma2=[list(range(n_path))], verbose=False)
        fit2 = out2["fit"]
        s2s2 = [p.sigma2 for p in fit2.paths]
        for s2 in s2s2:
            self.assertAlmostEqual(s2, s2s[0], places=6)
        self.assertAlmostEqual(fit.r_factor, fit2.r_factor, places=6)

    def test_systematic_budget_dynamic(self):
        """CN 系统误差预算必须随本次拟合条件变化，而不是恒为 ±30%。

        基础预算 0.30 是历史实测锚点；额外项（自吸收残留、参数强相关、
        远壳层截断、σ² 撞界）按平方和合成。这里用构造对象验证每一类
        额外项的数值贡献，防止预算退化成写死的常数。
        """
        class FakeFit:
            correlation = np.array([[1.0, 0.95], [0.95, 1.0]])
            parameter_names = ["Fe-Fe@2.97:N", "Fe-Fe@2.97:σ²"]

        base = xafs_fit.systematic_budget(None)
        self.assertAlmostEqual(base["relative_1sigma"], 0.30, places=4)
        self.assertEqual(base["extra_components"], {})
        self.assertAlmostEqual(base["max_parameter_correlation"], 0.0,
                               places=4)

        # 0.30 base + 0.10 相关 + 0.10 自吸收残留 + 0.05 截断 + 0.10 σ²撞界
        rich = xafs_fit.systematic_budget(
            FakeFit(), flags={"sigma2_at_bound": True},
            self_abs={"corrected": True, "reliable": True},
            truncated_shells=2)
        expected = float(np.sqrt(0.30**2 + 0.10**2 + 0.10**2
                                 + 0.05**2 + 0.10**2))
        self.assertAlmostEqual(rich["relative_1sigma"], expected, places=4)
        self.assertEqual(rich["extra_components"],
                         {"parameter_correlation": 0.10,
                          "self_absorption_residual": 0.10,
                          "truncated_shells": 0.05,
                          "sigma2_at_bound": 0.10})
        self.assertIn("Fe-Fe@2.97:N vs Fe-Fe@2.97:σ²", rich["note"])

        # 未校正但 risk=possible → +0.15；risk=low → 不加项
        possible = xafs_fit.systematic_budget(
            None, self_abs={"corrected": False, "risk": "possible"})
        self.assertAlmostEqual(possible["relative_1sigma"],
                               float(np.hypot(0.30, 0.15)), places=4)
        low = xafs_fit.systematic_budget(
            None, self_abs={"corrected": False, "risk": "low"})
        self.assertAlmostEqual(low["relative_1sigma"], 0.30, places=4)

    def test_html_report_single_file(self):
        """HTML 报告：图片内联为 data URI，强相关/不可靠项标红。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([0, 1], [0, 1])
            fig.savefig(out / "demo_structure_main.png", dpi=40)
            plt.close(fig)
            rows = [
                ("cn_reliable", "False", "",
                 "CN 是否可作结论"),
                ("max_parameter_correlation",
                 "0.950 (a:N vs a:σ²)", "",
                 "最强参数相关"),
                ("r_factor", "0.1234", "", "拟合优度"),
            ]
            fit_out = {"status": "confident", "cn_reliable": False,
                       "flags": {"r_factor_bad": True},
                       "cn_systematics": {"relative_1sigma": 0.35,
                                          "note": "test budget"},
                       "fit": None}
            path = xafs_report.export_html_report(
                out, "demo", sample="demo sample",
                edge_label="Fe K-edge", indicator_rows=rows,
                table_rows=[["demo", "Fe-O", "6.0", "2.00", "0.004",
                             "0.0", "0.003"]],
                fit_out=fit_out)
            self.assertTrue(path.exists())
            doc = path.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", doc)
            self.assertGreaterEqual(doc.count("class='bad'"), 2)
            self.assertIn("r_factor_bad", doc)
            self.assertIn("35%", doc)

    def test_electrocatalysis_oxide_hydroxide_library_real_standards(self):
        """#74 B20-1：7 个电催化 Motif 入库 + 4 条真实 BMM 标样盲测。

        - CoO / Co3O4 / Ni(OH)2 / CuO / PtO2 几何指纹逐项核验；
        - Co-K-CoO：top1=coo_rocksalt_cod_1533087 confident、score<0.06，S7 Rf<0.02；
        - Co-K-Co3O4：top1=co3o4_mixed confident、score<0.05、无未解释峰；
        - Ni-K-NickelHydroxide：top1=nioh2_theophrastite_cod_1548811
          confident、score<0.14，S7 Rf<0.05 且 Ni-O/Ni-Ni 键长落在窗口内；
        - Cu-CuO：top1=cuo_tenorite_cod_1011148、score<0.10。
        """
        co_lib = {m.id: m for m in structure_lib.load_library("Co")}
        ni_lib = {m.id: m for m in structure_lib.load_library("Ni")}
        cu_lib = {m.id: m for m in structure_lib.load_library("Cu")}
        pt_lib = {m.id: m for m in structure_lib.load_library("Pt")}
        for mid in ("coo_rocksalt_cod_1533087", "co3o4_spinel_cod_9005888",
                    "co3o4_oct", "co3o4_mixed"):
            self.assertIn(mid, co_lib)
        self.assertIn("nioh2_theophrastite_cod_1548811", ni_lib)
        self.assertIn("cuo_tenorite_cod_1011148", cu_lib)
        self.assertIn("pto2_beta", pt_lib)

        def _shells(motif):
            rows = motif.shell_expect or []
            return [(s["el"], float(s["r"]), int(s["n"])) for s in rows]

        coo = _shells(co_lib["coo_rocksalt_cod_1533087"])
        self.assertEqual(coo[0][0], "O")
        self.assertAlmostEqual(coo[0][1], 2.1315, places=3)
        self.assertEqual(coo[0][2], 6)
        self.assertEqual(coo[1][0], "Co")
        self.assertAlmostEqual(coo[1][1], 3.0144, places=3)
        self.assertEqual(coo[1][2], 12)

        tet = _shells(co_lib["co3o4_spinel_cod_9005888"])
        self.assertEqual(tet[0][2], 4)
        self.assertAlmostEqual(tet[0][1], 1.9353, places=3)
        octa = _shells(co_lib["co3o4_oct"])
        self.assertEqual(octa[0][2], 6)
        self.assertAlmostEqual(octa[0][1], 1.9217, places=3)
        self.assertEqual(octa[1][0], "Co")
        self.assertAlmostEqual(octa[1][1], 2.8581, places=3)
        self.assertEqual(octa[1][2], 6)
        mixed = co_lib["co3o4_mixed"]
        self.assertEqual(mixed.kind, "mixture")
        weights = {c["id"]: float(c["weight"]) for c in mixed.components}
        self.assertAlmostEqual(weights["co3o4_spinel_cod_9005888"],
                               1.0 / 3.0, places=12)
        self.assertAlmostEqual(weights["co3o4_oct"], 2.0 / 3.0, places=12)

        brucite = _shells(ni_lib["nioh2_theophrastite_cod_1548811"])
        self.assertEqual(brucite[0][0], "O")
        self.assertAlmostEqual(brucite[0][1], 2.0597, places=3)
        self.assertEqual(brucite[0][2], 6)
        self.assertEqual(brucite[1][0], "Ni")
        self.assertAlmostEqual(brucite[1][1], 3.1268, places=3)
        self.assertEqual(brucite[1][2], 6)

        tenorite = _shells(cu_lib["cuo_tenorite_cod_1011148"])
        self.assertEqual(tenorite[0][0], "O")
        self.assertAlmostEqual(tenorite[0][1], 1.9475, places=3)
        self.assertEqual(tenorite[0][2], 4)

        pto2 = _shells(pt_lib["pto2_beta"])
        self.assertEqual(pto2[0][0], "O")
        self.assertAlmostEqual(pto2[0][1], 2.0495, places=3)
        self.assertEqual(pto2[0][2], 6)
        self.assertEqual(pto2[1][0], "Pt")
        self.assertAlmostEqual(pto2[1][1], 3.100, places=3)
        self.assertEqual(pto2[1][2], 6)

        data_dir = ROOT / "data" / "bmm_standards" / "Data"

        def _identify(rel, absorber):
            res = xafs_pipeline.process_sample(
                [str(data_dir / rel)], absorber=absorber, strict=False,
                verbose=False)
            sid = structure_id.identify_result(
                res, absorber_symbol=absorber, verbose=False)
            return res, sid

        # ---- CoO ----
        res_coo, sid_coo = _identify("Co/Co-K-CoO.xdi", "Co")
        self.assertEqual(sid_coo["ranked"][0]["id"],
                         "coo_rocksalt_cod_1533087")
        self.assertEqual(sid_coo["status"], "confident")
        self.assertLess(sid_coo["ranked"][0]["score"], 0.06)
        out_coo = xafs_fit.run(
            res_coo, identify=False, fit=True, absorber_symbol="Co",
            verbose=False)
        self.assertEqual(out_coo["fit"]["mode"], "motif")
        self.assertEqual(out_coo["fit"]["motif"],
                         "coo_rocksalt_cod_1533087")
        self.assertLess(out_coo["fit"]["fit"].r_factor, 0.02)

        # ---- Co3O4（尖晶石双位点 mixture）----
        _, sid_co3o4 = _identify("Co/Co-K-Co3O4.xdi", "Co")
        self.assertEqual(sid_co3o4["ranked"][0]["id"], "co3o4_mixed")
        self.assertEqual(sid_co3o4["status"], "confident")
        self.assertLess(sid_co3o4["ranked"][0]["score"], 0.05)
        self.assertEqual(sid_co3o4["uncovered_peaks"], [])

        # ---- β-Ni(OH)2 ----
        res_nioh, sid_nioh = _identify(
            "Ni/Ni-K-NickelHydroxide.xdi", "Ni")
        self.assertEqual(sid_nioh["ranked"][0]["id"],
                         "nioh2_theophrastite_cod_1548811")
        self.assertEqual(sid_nioh["status"], "confident")
        self.assertLess(sid_nioh["ranked"][0]["score"], 0.14)
        out_nioh = xafs_fit.run(
            res_nioh, identify=False, fit=True, absorber_symbol="Ni",
            verbose=False)
        self.assertEqual(out_nioh["fit"]["mode"], "motif")
        fit_nioh = out_nioh["fit"]["fit"]
        self.assertLess(fit_nioh.r_factor, 0.05)
        by_label = {p.label: p for p in fit_nioh.paths}
        ni_o = next(p for label, p in by_label.items()
                    if label.startswith("Ni-O"))
        ni_ni = next(p for label, p in by_label.items()
                     if label.startswith("Ni-Ni"))
        self.assertGreaterEqual(ni_o.r, 2.03)
        self.assertLessEqual(ni_o.r, 2.11)
        self.assertGreaterEqual(ni_ni.r, 3.08)
        self.assertLessEqual(ni_ni.r, 3.17)

        # ---- CuO ----
        _, sid_cuo = _identify("Cu/Cu-CuO.xdi", "Cu")
        self.assertEqual(sid_cuo["ranked"][0]["id"],
                         "cuo_tenorite_cod_1011148")
        self.assertLess(sid_cuo["ranked"][0]["score"], 0.10)

    def test_cache_sha256_reuse(self):
        """#93 P1b：缓存复用前校验 sha256；不匹配则拒绝。"""
        import shutil
        cache_root = ROOT / "_structure_lib" / "_phase_cache"
        fe_cache = cache_root / "COD" / "1100108"
        if fe_cache.exists():
            shutil.rmtree(fe_cache)

        # 首次解析：写入缓存
        res = phase_resolver.resolve_phase("Fe", phase_id="COD:1100108",
                                           absorber_site="Fe1")
        self.assertEqual(res.status, "resolved")
        meta_files = list(fe_cache.glob("*.meta.json"))
        self.assertEqual(len(meta_files), 1)
        cached_sha = json.loads(meta_files[0].read_text(encoding="utf-8"))[
            "cif_sha256"]
        self.assertEqual(cached_sha,
                         "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe")

        # 篡改缓存中的 CIF 字节（改哈希），复用时应拒绝
        cif_file = list(fe_cache.glob("*.cif"))[0]
        original_bytes = cif_file.read_bytes()
        try:
            cif_file.write_bytes(original_bytes + b"\n# tampered\n")
            res2 = phase_resolver.resolve_phase("Fe", phase_id="COD:1100108",
                                                absorber_site="Fe1")
            # 缓存 sha256 不匹配，应回退到 phase pack 重新获取（仍 resolved）
            self.assertEqual(res2.status, "resolved")
        finally:
            cif_file.write_bytes(original_bytes)

        # 清理测试产生的缓存
        shutil.rmtree(fe_cache)

    def test_counterexample_freed_n_zero_degeneracy_rejected(self):
        """审查 item 8：零占据壳放开 N → lower=0.5 > upper=0 不可行边界，
        必须显式拒绝（而非 scipy 静默崩溃或给出虚假 CN）。"""
        import artemis
        import athena
        import structure_lib
        import xafs_fit
        from larch.xafs import feffpath
        from xafs_core import ChiK, XAFSConfigError
        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "magnetite_mixed"][0]
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", rmax=2.6, kmin=3.0, kmax=12.0,
            group_by_component=True)
        f0, d0 = plan["files"][0], plan["degens"][0]
        k = np.arange(3.0, 12.0, 0.05)
        p = feffpath(str(f0), s02=0.8, sigma2=0.004, e0=2.0, deltar=0.0)
        p.degen = d0
        p._calc_chi()
        chi = artemis.exafs_model_k([p], k)
        chi_k = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                     mu_norm=np.zeros_like(k), rbkg=1.0)
        ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
        with self.assertRaises(XAFSConfigError):
            artemis.fit_r_space(
                chi_k, ft, [f0, f0], rmin=plan["rmin"], rmax=plan["rmax"],
                kmin=3.0, kmax=12.0, kweight=2, s02=0.8,
                vary_delta_e0=True, vary_n=[1], sigma2_init=0.004,
                path_degen=[d0, 0.0], path_labels=["real", "zero_occ"])

    def test_first_shell_single_scale_factor(self):
        """#104 P3：首壳单一比例因子，仅放开参考路径 N。"""
        motifs = {m.id: m for m in structure_lib.load_library("Fe")}
        motif = motifs["fe_bcc"]
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        gate = cif_gate.gate_cif(cif, "Fe")
        self.assertTrue(gate.ok)
        plan = xafs_fit.select_shell_paths(
            motif, edge="K", kmin=3.0, kmax=12.0, rmax=3.5, n_mode="free")
        mapping = xafs_fit.map_n0_to_plan(plan, gate.n0_shells,
                                          kmin=3.0, kmax=12.0)
        self.assertTrue(mapping["mapping_ok"])
        args = xafs_fit.build_conditional_fit_args(plan, mapping)
        # 仅放开参考路径 N
        self.assertEqual(len(args["fit_args"]["vary_n"]), 1)
        self.assertIn(args["ref_idx"], args["fit_args"]["vary_n"])
        # path_degen = n0_per_path（所有路径 N₀ 初值）
        self.assertEqual(args["fit_args"]["path_degen"],
                         mapping["n0_per_path"])
        # 非参考首壳路径用 degen_ratio 绑定到参考路径
        links = args["fit_args"]["path_links"]
        # 不同 N₀ 壳层（如 Fe bcc 的 2.48 与 2.87 Å）不得互相绑定：
        # 远壳必须固定 N₀，不能跟着首壳 N 漂移。
        self.assertNotIn(1, [link["target"] for link in links])
        for link in links:
            self.assertEqual(link["share_deltar"], False)
            self.assertEqual(link["share_sigma2"], False)
            self.assertIsNotNone(link["degen_ratio"])
            self.assertGreater(link["degen_ratio"], 0)
        # n0_total = 首壳 N₀ 总和
        self.assertAlmostEqual(args["n0_total"], 8.0)

    def test_offline_full_chain(self):
        """#111 P4：离线全链——相同输入 + 相同缓存 → 相同输出哈希。"""
        import hashlib
        cif = ROOT / "_structure_lib" / "_phase_pack" / "COD" / "1100108" \
            / "41085e65debf2bd82a772bcee9e7c0a43ef42b67666d2f233c8f2d822769b8fe.cif"
        self.assertTrue(cif.is_file())
        # 两次独立调用，结果应字节一致
        r1 = cif_gate.gate_cif(cif, "Fe")
        r2 = cif_gate.gate_cif(cif, "Fe")
        self.assertTrue(r1.ok and r2.ok)
        self.assertEqual(r1.cif_sha256, r2.cif_sha256)
        self.assertEqual(r1.first_shell_cn, r2.first_shell_cn)
        self.assertEqual(r1.occupancy_source, r2.occupancy_source)
        # 整个结果 dict 的哈希也应一致
        h1 = hashlib.sha256(repr(sorted(r1.__dict__.items())).encode()).hexdigest()
        h2 = hashlib.sha256(repr(sorted(r2.__dict__.items())).encode()).hexdigest()
        self.assertEqual(h1, h2)

    def test_counterexample_report_entry_shares_effective_kmax(self):
        """反例（审查 item 16）：xafs_report 入口必须与 run_xafs 共用有效 kmax。

        历史缺陷：process_scans 直接调 xafs_pipeline.process_sample(kmax=raw)，
        从不经过 run_xafs.bounded_kmax——只有 run_xafs.process_sample 入口裁剪。
        于是 xafs_report 的单谱 / operando / phase / 演示四条路径（全部经
        process_scans 这一唯一 chokepoint）对超限 kmax 原样放行，把下一条边
        的阶跃卷入 χ(k)/FT/WT：#13 在主管线修好的缺陷在报告入口原样复发。
        正确：process_scans 必须按元素裁剪并把记录挂到 result 供台账写入。
        这里用 Fe 真实静态 ceiling=15 验证（请求 20 → 生效 15），无需 Ce 数据。
        """
        sigma = 1.5   # process_sample 默认 wt_sigma
        tol = 0.11    # k_centers 网格步长 dk=0.1 的端点容差

        # ① 未超限（Fe ceiling=15，请求 12）：原样生效，无裁剪记录（不回归）。
        st_fe = xafs_report.process_scans(
            [LEPIDO], absorber="Fe", kmax=12.0, strict=False)
        self.assertAlmostEqual(st_fe["result"].ft.kmax, 12.0, places=6)
        self.assertIsNone(st_fe["result"].kmax_clamp)

        # ② 反例核心：请求 kmax=20 > Fe ceiling=15 → 必须裁到 15。
        #    旧 process_scans 忽略 bounded_kmax，ft.kmax 会停在 20。
        st_cap = xafs_report.process_scans(
            [LEPIDO], absorber="Fe", kmax=20.0, strict=False)
        self.assertAlmostEqual(st_cap["result"].ft.kmax, 15.0, places=6)
        self.assertIsNotNone(st_cap["result"].kmax_clamp)
        clamp = st_cap["result"].kmax_clamp
        self.assertAlmostEqual(clamp["requested_kmax"], 20.0, places=6)
        self.assertAlmostEqual(clamp["applied_kmax"], 15.0, places=6)
        self.assertAlmostEqual(clamp["static_ceiling"], 15.0, places=6)
        # ③ 有效 kmax 必须同样贯穿 WT（与 FT 共用上限，不得写死）。
        self.assertLessEqual(float(st_cap["wt"].k_centers.max()),
                             15.0 - sigma + tol)
        self.assertLessEqual(float(st_fe["wt"].k_centers.max()),
                             12.0 - sigma + tol)

    def test_counterexample_boundary_input_crash_guards(self):
        """反例（审查 item 14）：边界输入须如实降级/明确报错，不得原始崩溃。

        ① 零噪声（σ_μ≡0，扫描完全相同）：high_k_snr 旧实现把 0 并入
           "非正数"一律 raise，在 S5 杀掉整条管线；正确应诚实降级
           snr_k=None 并说明原因（SNR=信号/0 无定义）。
        ② 负/非有限 σ_μ 仍须拒绝（真非法输入，不可放过）。
        ③ 正噪声仍给出判据（不回归）。
        ④ 短 k 窗（宽度<2σ）：WT 旧实现 k_centers 空 → wave_map.max()
           抛原始 ValueError；正确应抛带可操作信息的 XAFSConfigError。
        ⑤ 恰好 2σ 宽度 → ≥1 个滑窗中心，边界不误杀。
        """
        ck = self.lep.chi_k
        edge = abs(float(self.lep.norm.edge_step))

        # ① 零噪声：降级而非崩溃
        zero = athena.high_k_snr(ck, sigma_mu=np.zeros_like(ck.k),
                                 edge_step=self.lep.norm.edge_step, kweight=2)
        self.assertIsNone(zero["snr_k"])
        self.assertIsNone(zero["k_snr2"])
        self.assertIsNone(zero["reliable_kmax"])
        self.assertIn("零噪声", zero["reason"])
        # ② 负 σ_μ / 非有限 σ_μ：仍拒绝
        with self.assertRaises(XAFSConfigError):
            athena.high_k_snr(ck, sigma_mu=np.full_like(ck.k, -1e-3),
                              edge_step=self.lep.norm.edge_step)
        with self.assertRaises(XAFSConfigError):
            athena.high_k_snr(ck, sigma_mu=np.full_like(ck.k, np.nan),
                              edge_step=self.lep.norm.edge_step)
        # ③ 正噪声仍给出判据（不回归）
        pos = athena.high_k_snr(ck, sigma_mu=np.full_like(ck.k, 1e-2 * edge),
                                edge_step=self.lep.norm.edge_step, kweight=2)
        self.assertIsNotNone(pos["snr_k"])

        # ④ 短 k 窗 WT：明确 XAFSConfigError，不是原始 ValueError
        with self.assertRaises(XAFSConfigError) as cm:
            larch_wt.continuous_wavelet_transform(
                ck, sigma=1.5, kmin=3.0, kmax=5.0, kweight=2)  # 宽2 < 2σ=3
        self.assertIn("WT", str(cm.exception))
        # ⑤ 恰好 2σ 宽度（kmax-kmin=3）→ ≥1 个中心，边界不误杀
        wm = larch_wt.continuous_wavelet_transform(
            ck, sigma=1.5, kmin=3.0, kmax=6.0, kweight=2)
        self.assertGreaterEqual(wm.k_centers.size, 1)

    def test_multi_element_report_no_top1(self):
        """#120 T7：多元素自动分元素跑批；报告并列、不推荐 Top-1。"""
        import run_xafs
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fe = tmp / "sample_fe.dat"
            fe.write_text("# Absorber: Fe\n# Energy(eV)\tmu\n"
                          "7112\t1.0\n", encoding="utf-8")
            cu = tmp / "sample_cu.dat"
            cu.write_text("# Absorber: Cu\n# Energy(eV)\tmu\n"
                          "8979\t1.0\n", encoding="utf-8")
            unknown = tmp / "unknown.dat"
            unknown.write_text("# no element header\n1 1\n",
                               encoding="utf-8")

            def fake_run_one(path, element, sub):
                sub.mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(
                    structure={
                        "status": "confident",
                        "ranked": [{"id": f"{element.lower()}_phase",
                                    "score": 0.02}]},
                    fit={
                        "mode": "l1_first_shell", "cn_reliable": True,
                        "l1": {"N": 5.0, "R": 2.0}, "warning": ""},
                    xanes_cross_check={
                        "warnings": [{"code": "edge_step_weak",
                                      "message": "demo warning"}]})

            out = tmp / "out"
            summary = run_xafs.run_multi_element(
                [str(fe), str(cu), str(unknown)], out,
                run_one=fake_run_one, verbose=False)
            self.assertEqual(summary["n_ok"], 2)
            self.assertEqual(summary["n_failed"], 1)
            self.assertEqual(summary["elements"], ["Cu", "Fe"])
            self.assertIsNone(summary["recommended"])
            self.assertIsNone(summary["best_element"])
            self.assertIn("不自动选择", summary["recommendation_policy"])
            self.assertTrue((out / "multi_element_summary.json").is_file())
            md = (out / "multi_element_summary.md").read_text(
                encoding="utf-8")
            self.assertIn("不自动选择/推荐 Top-1", md)
            fe_row = next(r for r in summary["rows"]
                          if r["absorber"] == "Fe")
            self.assertEqual(fe_row["first_shell"]["n_fit"], 5.0)
            self.assertEqual(fe_row["candidates"][0]["id"], "fe_phase")
            self.assertEqual(fe_row["absorber_source"], "header_absorber")
            self.assertIn("demo warning", fe_row["warnings"])

    def test_p5b_mo_double_shell_and_ce_rejection(self):
        """#124 P5-B：Mo 双壳 bcc 标定；Ce σ² 压界/伪迹明确拒绝。"""
        import inspect
        import calibrate_s02
        import calibrate_t5
        import run_xafs

        sig = inspect.signature(calibrate_s02.calibrate_s02)
        for name in ("e0_method", "manual_e0", "fit_shells"):
            self.assertIn(name, sig.parameters)

        mo_cases = calibrate_t5.CASES["Mo"]
        mo_foil = next(c for c in mo_cases
                       if Path(c["standard"]).name ==
                       "Mo-K-MolybdenumFoil.xdi")
        self.assertEqual(mo_foil.get("e0_method"), "half_height")
        self.assertEqual(mo_foil.get("fit_shells"), 2)
        self.assertTrue(mo_foil.get("strict"))
        ce_cases = calibrate_t5.CASES["Ce"]
        self.assertEqual(len(ce_cases), 3)
        self.assertTrue(ce_cases[1].get("strict"))
        self.assertTrue(ce_cases[2].get("strict"))
        self.assertEqual(ce_cases[2].get("rmin"), 1.5)

        mo = json.loads((ROOT / "data" / "s02_calibration" / "Mo" /
                         "calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(mo["status"], "calibrated")
        self.assertAlmostEqual(mo["s02"], 0.8322, places=4)
        self.assertEqual(mo["n_shells"], 2)
        self.assertEqual([s["degen"] for s in mo["shell_paths"]],
                         [8.0, 6.0])
        self.assertAlmostEqual(mo["shell_paths"][0]["reff"], 2.7254,
                               places=4)
        self.assertAlmostEqual(mo["shell_paths"][1]["reff"], 3.147,
                               places=3)
        self.assertEqual(mo["e0_method"], "half_height")
        self.assertAlmostEqual(mo["e0"], 20012.8624, places=3)
        self.assertIn("sigma2_not_at_bound", mo["checks"])
        self.assertTrue(all(mo["checks"].values()))
        self.assertEqual(mo["n_variables"], 6)
        self.assertGreater(mo["n_independent"], mo["n_variables"])
        for run in mo["runs"]:
            self.assertTrue(run["interior"])
            self.assertEqual(len(run["paths"]), 2)
            self.assertFalse(run["sigma2_at_bound"])
            self.assertFalse(any(p["sigma2_at_bound"]
                                 for p in run["paths"]))

        ce = json.loads((ROOT / "data" / "s02_calibration" / "Ce" /
                         "calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(ce["status"], "calibrated")
        self.assertAlmostEqual(ce["s02"], 0.6279, places=4)
        first = next(a for a in ce["attempts"]
                     if a["standard"].endswith("Ce-L3-CeO2.xdi"))
        self.assertEqual(first["status"], "gated_out")
        self.assertFalse(first["result"]["checks"]["sigma2_not_at_bound"])
        self.assertTrue(all(r["sigma2_at_bound"]
                            for r in first["result"]["runs"]))
        second = next(a for a in ce["attempts"]
                      if a["standard"].endswith(
                          "Ce-L3-CeriumDioxide.xdi"))
        self.assertEqual(second["status"], "error")
        self.assertIn("归一化失败", second["error"])
        self.assertIn("2.271", second["error"])
        third = next(a for a in ce["attempts"]
                     if a["standard"].endswith("Ce_metal_foil.dat"))
        self.assertEqual(third["status"], "passed")
        self.assertTrue(all(third["result"]["checks"].values()))

        s02, source = run_xafs.load_t5_s02("Mo")
        self.assertAlmostEqual(s02, 0.8322, places=4)
        self.assertIn("Mo", source)
        self.assertIn("sha256:", source)
        s02_ce, source_ce = run_xafs.load_t5_s02("Ce")
        self.assertAlmostEqual(s02_ce, 0.6279, places=4)
        self.assertIn("Ce", source_ce)

    def test_p6_4_output_ledger_and_print_flags(self):
        """#132 P6.4：参数/门禁/决策/摘要台账、打印开关与阈值不漂移。"""
        import argparse
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch

        # 1) 阈值镜像必须与冻结常量一致（防止台账与物理门禁脱节）
        self.assertEqual(xafs_provenance.PLATEAU_LO, 0.85)
        self.assertEqual(xafs_provenance.PLATEAU_HI, 1.15)
        self.assertEqual(xafs_provenance.RF_LIMIT, 0.08)
        self.assertEqual(xafs_provenance.DELTA_E0_LIMIT, 15.0)
        self.assertEqual(xafs_provenance.N_ERR_LIMIT, 1.0)
        self.assertEqual(xafs_provenance.N_ERR_FRACTION_LIMIT, 0.5)
        self.assertEqual(xafs_provenance.CIF_CN_NORMAL, (2, 12))
        self.assertEqual(xafs_provenance.CIF_CN_WARN_HI, 16)
        self.assertEqual(xafs_provenance.CN_RATIO_OK, (0.7, 1.3))
        self.assertEqual(xafs_provenance.CN_RATIO_WARN, (0.6, 1.4))
        self.assertEqual(xafs_provenance.SID_SCORE_LIMIT, 0.16)
        self.assertEqual(xafs_provenance.SID_MARGIN, 0.03)

        # 2) 参数来源：CLI / config / default / auto
        parser = argparse.ArgumentParser()
        parser.add_argument("--kmin", type=float, default=3.0)
        parser.add_argument("--kmax", type=float, default=12.0)
        parser.add_argument("--config", default=None)
        parser.add_argument("--print-gates", action="store_true")
        args = parser.parse_args(["--kmin", "3.5", "--print-gates"])
        prov = xafs_provenance.Provenance.from_args(
            args, {}, ["--kmin", "3.5", "--print-gates"],
            parser=parser, inputs=["a.xdi"])
        block = prov.param_block()
        self.assertEqual(block["sources"]["kmin"], "cli")
        self.assertEqual(block["sources"]["print_gates"], "cli")
        self.assertEqual(block["sources"]["kmax"], "default")
        self.assertEqual(block["effective"]["kmin"], 3.5)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text(json.dumps({"kmin": 3.2}), encoding="utf-8")
            args2 = parser.parse_args(["--config", str(cfg)])
            prov2 = xafs_provenance.Provenance.from_args(
                args2, {"kmin": 3.2}, ["--config", str(cfg)], parser=parser)
            block2 = prov2.param_block()
            self.assertEqual(block2["sources"]["kmin"], "config")
            self.assertEqual(block2["config_keys_applied"], ["kmin"])

        # 3) 门禁台账：通过 / 失败 / 不适用
        gates = xafs_provenance.build_gate_ledger(
            norm=SimpleNamespace(plateau=0.98, edge_step=0.4),
            strict=True,
            self_absorption={"mode": "transmission", "risk": "none",
                             "cn_gate": False, "corrected": False},
            structure={"status": "confident",
                       "confidence": {"best_score": 0.05, "margin": 0.1,
                                      "second_score": 0.15},
                       "coverage_ok": True, "ranked": [{"id": "fe_bcc"}]})
        self.assertEqual(gates["normalization"]["status"], "passed")
        self.assertEqual(gates["normalization"]["threshold"], [0.85, 1.15])
        self.assertEqual(gates["self_absorption"]["status"], "passed")
        self.assertEqual(gates["structure_id"]["status"], "passed")
        self.assertEqual(gates["fit"]["status"], "not_applicable")
        self.assertEqual(gates["summary"]["passed"],
                         ["normalization", "self_absorption", "structure_id"])

        fit_obj = SimpleNamespace(
            r_factor=0.2, delta_e0=2.0, overfit=False, paths=[],
            n_independent=10, n_variables=5)
        fit_gate = xafs_provenance.build_gate_ledger(
            fit_out={"fit": fit_obj, "flags": {}, "coverage_ok": True,
                     "cn_reliable": False, "cn_status": "diagnostic_only"}
        )["fit"]
        self.assertEqual(fit_gate["status"], "failed")
        self.assertIn("r_factor", fit_gate["failed_checks"])
        self.assertEqual(fit_gate["action"], "cn_quantification_blocked")

        l1_gate = xafs_provenance.build_gate_ledger(
            fit_out={"mode": "l1_first_shell",
                     "l1": {"rf": 0.05, "delta_e0": 1.0, "N": 4.0,
                            "N_err": 0.2, "n_independent": 8,
                            "n_variables": 4},
                     "flags": {}, "coverage_ok": True,
                     "cn_reliable": True, "cn_status": "ok"}
        )["fit"]
        self.assertEqual(l1_gate["status"], "passed")

        formal_gate = xafs_provenance.build_gate_ledger(
            fit_out={"conditional_cn": {
                "n_ratio": 0.5, "n_status": "out_of_range",
                "conditional_n_fit": {"r_factor": 0.10},
                "hamilton_f_test": {"significant": False},
                "consistency_passed": False}},
            formal_block={"status": "consistency_failed",
                          "reject_reason": "N/N0 out of range"}
        )["formal_cn"]
        self.assertEqual(formal_gate["status"], "failed")
        self.assertEqual(formal_gate["action"], "formal_cn_refused")

        # 4) run_xafs 单样品：_meta.json 必须包含四个台账块
        import run_xafs
        with tempfile.TemporaryDirectory() as tmp:
            args3 = parser.parse_args(["--kmin", "3.0"])
            prov3 = xafs_provenance.Provenance.from_args(
                args3, {}, ["--kmin", "3.0"], parser=parser,
                inputs=[str(FE_FOIL)])
            run_xafs.process_sample(
                [str(FE_FOIL)], tmp, absorber="Fe", strict=False,
                provenance=prov3, verbose=False)
            meta = json.loads(
                (Path(tmp) / "fe_meta.json").read_text(encoding="utf-8"))
            for key in ("params", "gates", "decisions", "summary"):
                self.assertIn(key, meta)
            self.assertEqual(meta["params"]["sources"]["kmin"], "cli")
            self.assertEqual(meta["gates"]["normalization"]["status"],
                             "warning")
            self.assertGreaterEqual(len(meta["decisions"]), 3)

        # 5) report 入口同样写 _meta.json，且 source.mode=report
        with tempfile.TemporaryDirectory() as tmp:
            xafs_report.main([
                str(FE_FOIL), "--absorber", "Fe", "--no-strict",
                "--no-fit", "--no-html", "--plot", "none",
                "-o", tmp, "--print-gates", "--print-params"])
            meta = json.loads(
                (Path(tmp) / "fe_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["source"]["mode"], "report")
            for key in ("params", "gates", "decisions", "summary"):
                self.assertIn(key, meta)

        # 6) 两个入口都暴露 --print-gates / --print-params
        for module, flag in (
                (run_xafs, "--print-gates"),
                (run_xafs, "--print-params"),
                (xafs_report, "--print-gates"),
                (xafs_report, "--print-params")):
            buf = io.StringIO()
            with patch.object(sys, "argv", ["prog", "--help"]), \
                    contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main()
            self.assertIn(flag, buf.getvalue(),
                          f"{module.__name__} 缺少 {flag}")

    def test_p6_5_unified_structure_plots(self):
        """Scheme A：统一催化剂结构剖析出图系统（主图 + 壳层鉴别 + 候选对比 + summary.csv）。"""
        import contextlib
        import io
        from unittest.mock import patch

        import run_xafs
        import xafs_plot_special
        import xafs_plots

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            suite = xafs_plots.render_plot_suite(
                self.lep, self.lep_fit, out, "lepido",
                mode="full", formats=("png",), dpi=72, verbose=False)
            self.assertEqual(suite["mode"], "full")
            for name in (
                "lepido_structure_main.png",
                "lepido_shell_dissection.png",
                "lepido_candidates_compare.png",
                "lepido_summary.csv",
            ):
                p = out / name
                self.assertTrue(p.is_file() and p.stat().st_size > 0,
                                f"缺少结构剖析产物 {name}")
            summary_lines = (out / "lepido_summary.csv").read_text(
                encoding="utf-8-sig").splitlines()
            self.assertGreater(len(summary_lines), 8)
            md_path = xafs_plot_special.write_fit_table_markdown(
                [["lepido", "Fe-O", "6.0", "2.00", "0.004", "+1.0", "0.0200"]],
                out, "lepido")
            self.assertTrue(md_path.is_file())
            self.assertIn("| Sample |", md_path.read_text(encoding="utf-8"))

        # 两个 CLI 入口统一暴露 --plot / --plot-formats / --plot-dpi
        for module in (run_xafs, xafs_report):
            buf = io.StringIO()
            with patch.object(sys, "argv", ["prog", "--help"]), \
                    contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main()
            for flag in ("--plot", "--plot-formats", "--plot-dpi"):
                self.assertIn(flag, buf.getvalue(),
                              f"{module.__name__} 缺少 {flag}")

    def test_p6_6_entry_parity_and_streamlined_outputs(self):
        """#134 P6.6 + Scheme A：入口产物对齐与统一结构剖析输出。"""
        import contextlib
        import io
        from unittest.mock import patch

        import run_xafs

        # 1) report 入口：meta 完整 + 统一结构剖析图包
        with tempfile.TemporaryDirectory() as tmp:
            xafs_report.main([
                str(FE_FOIL), "--absorber", "Fe", "--no-strict",
                "--fit-motif", "fe_bcc", "--no-html",
                "--plot", "full", "--plot-dpi", "72", "-o", tmp])
            out = Path(tmp)
            meta = json.loads(
                (out / "fe_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["rbkg"].get("table"),
                            "report 的 rbkg.table 未落盘")
            self.assertIn("plateau_ok", meta["normalization"])
            self.assertTrue(meta["source"].get("data_sha256"),
                            "report 的 source.data_sha256 缺失")
            self.assertIsNotNone(meta["fit"].get("r_factor"),
                                 "fit.r_factor 不应为 None")
            self.assertIsNotNone(meta["fit"].get("delta_e0"),
                                 "fit.delta_e0 不应为 None")
            self.assertFalse((out / "fe_fit_table.txt").exists(),
                             "冗余 _fit_table.txt 不应再生成")
            self.assertTrue((out / "fe_chiq.csv").is_file())
            self.assertTrue((out / "fe_structure_id.json").is_file())
            for name in ("fe_structure_main.png", "fe_shell_dissection.png",
                         "fe_candidates_compare.png", "fe_summary.csv",
                         "fe_fit_table.md"):
                self.assertTrue((out / name).is_file(), f"缺少 {name}")

        # 2) run_xafs 入口：q 空间、全量结构候选与统一结构图件落盘
        with tempfile.TemporaryDirectory() as tmp:
            run_xafs.process_sample(
                [str(FE_FOIL)], Path(tmp), absorber="Fe", strict=False,
                identify=True, fit=True, fit_motif="fe_bcc",
                plot="structure", plot_dpi=72, verbose=False)
            out = Path(tmp)
            chiq = out / "fe_chiq.csv"
            self.assertTrue(chiq.is_file(), "run_xafs 未落盘 _chiq.csv")
            header = chiq.read_text(encoding="utf-8").splitlines()[0]
            for col in ("q_inv_A", "chiq_re", "chiq_im", "chiq_mag"):
                self.assertIn(col, header)
            sid = json.loads(
                (out / "fe_structure_id.json").read_text(encoding="utf-8"))
            self.assertGreater(len(sid.get("ranked") or []), 3,
                               "run_xafs 仍把结构候选截断到 <=3")
            meta = json.loads(
                (out / "fe_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["source"].get("data_sha256"))
            self.assertTrue((out / "fe_structure_main.png").is_file())
            self.assertTrue((out / "fe_shell_dissection.png").is_file())
            self.assertTrue((out / "fe_summary.csv").is_file())

        # 3) report CLI 暴露 4 个参数；--batch-dir 不再漏传 fit_space/bvs
        buf = io.StringIO()
        with patch.object(sys, "argv", ["prog", "--help"]), \
                contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                xafs_report.main()
        for flag in ("--rbkg", "--e0-method", "--manual-e0", "--kweight"):
            self.assertIn(flag, buf.getvalue(), f"xafs_report 缺少 {flag}")

        src = Path(run_xafs.__file__).read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("fit_space=args.fit_space"), 2,
                                "--batch-dir 分支漏传 fit_space")
        self.assertGreaterEqual(src.count("bvs_restraint=bvs_restraint"), 2,
                                "--batch-dir 分支漏传 bvs_restraint")
        self.assertIn("_structure_id.json", src)
        self.assertIn("_chiq.csv", src)


    def test_unit6_regression_guards(self):
        """Unit 6 审计反例：multi_element 默认入口、绘图异常留痕、
        paper_figures cn_reliable=False 降级与软件溯源。"""
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        import paper_figures
        import run_xafs
        import xafs_plots

        # 1) U6-01：run_multi_element(run_one=None) 不得因未定义 wt_sigma 崩溃
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fe = tmp / "s.dat"
            fe.write_text("# Absorber: Fe\n7112 1.0\n", encoding="utf-8")
            with patch.object(
                run_xafs, "process_sample",
                return_value=SimpleNamespace(
                    structure={"status": "confident", "ranked": []},
                    fit={"mode": "l1_first_shell", "cn_reliable": True,
                         "l1": {"N": 6.0, "R": 2.0}},
                    xanes_cross_check=None,
                ),
            ) as mock_ps:
                summary = run_xafs.run_multi_element(
                    [str(fe)], tmp / "out", verbose=False)
            self.assertEqual(summary["n_ok"], 1)
            self.assertEqual(summary["n_failed"], 0)
            self.assertTrue(mock_ps.called)

        # 2) U6-04：paper_figures.load_fit 对 cn_reliable=False 必须标记 diagnostic_only
        fit_data = paper_figures.load_fit({
            "s02": 0.8,
            "cn_reliable": False,
            "fit": {"delta_e0": 1.0, "r_factor": 0.02, "s02": 0.8,
                    "paths": [{"label": "Fe-O", "N": 6.0, "R": 2.0,
                               "sigma2": 0.004, "path_file": __file__}]},
            "path_files": [__file__],
        })
        self.assertTrue(fit_data.diagnostic_only)

        # 3) U6-07：build_output_meta 必须包含 software 版本块
        prov = xafs_provenance.Provenance()
        meta_block = xafs_provenance.build_output_meta(prov)
        self.assertIn("software", meta_block)
        self.assertIn("python", meta_block["software"])

        # 4) U6-06：xafs_plots 异常在 verbose=False 时仍写 stderr 并记入 meta
        with tempfile.TemporaryDirectory() as tmp:
            err_buf = io.StringIO()
            with patch.object(xafs_plots, "render_plot_suite",
                              side_effect=RuntimeError("boom")), \
                    contextlib.redirect_stderr(err_buf):
                run_xafs.process_sample(
                    [str(FE_FOIL)], tmp, absorber="Fe", strict=False,
                    plot="structure", verbose=False)
            self.assertIn("boom", err_buf.getvalue())
            meta = json.loads(
                (Path(tmp) / "fe_meta.json").read_text(encoding="utf-8"))
            self.assertIn("plot_error", meta)
