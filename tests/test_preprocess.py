# -*- coding: utf-8 -*-
"""Unit 2：预处理与光谱特征（合并/归一化/背景/FT/小波/LCF）

纯搬运拆分：测试方法体与原 regression_test.py 逐行一致，断言未做任何修改。
"""
from __future__ import annotations

from tests.fixtures import *  # noqa: F401,F403
from tests.fixtures import Fixtures, _find  # noqa: F401

# 历史测试体里有 3 处直接用 Path(__file__) 定位 data/（原文件在仓库根）。
# 拆到 tests/ 后模块位置变了，把 __file__ 指回等价的根入口文件，
# 保证测试方法体逐字节不变（不改任何断言或路径写法）。
__file__ = str(ROOT / "regression_test.py")


class PreprocessTests(Fixtures):

    def test_chik_length_contract(self):
        chi = self.lep.chi_k
        self.assertEqual(len(chi.k), len(chi.chi))
        self.assertEqual(len(chi.background), len(chi.mu_norm))
        self.assertEqual(len(chi.mu_norm), len(self.lep.norm.energy))
        self.assertNotEqual(len(chi.mu_norm), len(chi.k),
                            "mu_norm 不应被误当成 k 网格数组")

    def test_theory_chi_filters_ms(self):
        from larch.xafs import feffpath

        motif = [m for m in structure_lib.load_library("Fe")
                 if m.id == "lepidocrocite"][0]
        cache = structure_lib.ensure_feff(motif, edge="K")
        absorber = get_absorber("Fe")
        k = np.arange(1.0, 13.0, 0.05)

        def model(max_legs):
            paths = []
            for f in sorted(cache.glob("feff*.dat")):
                p = feffpath(str(f), s02=absorber.s02 or 1.0,
                             sigma2=0.003, e0=0.0, deltar=0.0)
                nleg = getattr(p, "nleg", getattr(p, "legs", 2))
                if nleg <= max_legs and p.degen >= 1 and p.reff <= 5.5:
                    p._calc_chi()
                    paths.append(p)
            return artemis.exafs_model_k(paths, k)

        filtered = model(4)
        unfiltered = model(99)
        actual = structure_id.theory_chi(motif, k, absorber, sigma2=0.003)
        np.testing.assert_allclose(actual, filtered, rtol=1e-10, atol=1e-12)
        self.assertFalse(np.allclose(actual, unfiltered, rtol=1e-6, atol=1e-9),
                         "MS 路径仍被计入 theory_chi——nleg 过滤失效")

    def test_theory_chi_occupancy_weighting(self):
        """#19④：部分占据晶体指纹按壳层占据度加权 SS 简并度（N₀=Σdegen×occ），
        与拟合链 _ss_rows 同口径。参照用 theory_chi 自身（把占据表强制 occ=1.0
        即还原"未做占据缩放"的行为），不重实现脆弱的 MS 过滤——larch feffpath
        的 nleg 在重复 _calc_chi 后有状态性，旁路重实现会误纳 nleg>4 路径。
          · 赤铁矿 O occ=0.5 → 指纹相对全占据参照必须改变（缩放生效）；
          · 全占据体系（lepidocrocite、磁铁矿 tet/oct）→ 与全占据参照逐位
            一致（基线安全，no-op）。"""
        absorber = get_absorber("Fe")
        k = np.arange(1.0, 13.0, 0.05)
        lib = {m.id: m for m in structure_lib.load_library("Fe")}

        orig_scale = artemis.cif_shell_occupancy_scale

        def forced_full_occ(*a, **kw):
            return [(d, e, 1.0) for (d, e, _o) in orig_scale(*a, **kw)]

        def chi_full(motif):
            artemis.cif_shell_occupancy_scale = forced_full_occ
            try:
                return structure_id.theory_chi(
                    motif, k, absorber, sigma2=0.003)
            finally:
                artemis.cif_shell_occupancy_scale = orig_scale

        hem = lib["fe2o3_hematite_cod_1011240"]
        scale = orig_scale(structure_lib._cif_path(hem), "Fe", rmax=6.0)
        self.assertTrue(any(o < 0.999 for (_d, _e, o) in scale),
                        "赤铁矿占据表应含 occ<1 的近邻（O occ=0.5）")

        actual = structure_id.theory_chi(hem, k, absorber, sigma2=0.003)
        self.assertFalse(
            np.allclose(actual, chi_full(hem), rtol=1e-6, atol=1e-9),
            "赤铁矿 O occ=0.5 未加权——theory_chi 占据缩放失效")

        for mid in ("lepidocrocite", "magnetite_tet", "magnetite_oct"):
            np.testing.assert_allclose(
                structure_id.theory_chi(lib[mid], k, absorber, sigma2=0.003),
                chi_full(lib[mid]), rtol=1e-10, atol=1e-12,
                err_msg=f"{mid} 全占据理论谱被占据缩放改动（基线漂移）")

    def test_self_absorption_entry_rejects_mixed_identity(self):
        from unittest.mock import patch
        import copy

        path = _find("bmm_standards/Data/Ni/Ni-K-NickelHydroxide.xdi")
        trans = xafs_io.read_scan(path)
        fluor = copy.deepcopy(trans)
        fluor.meta["mode"] = "fluor"
        fluor.ifluor = fluor.i0 * fluor.mu
        for scans in ([trans, fluor], [fluor, trans]):
            with patch.object(xafs_io, "read_scans", return_value=copy.deepcopy(scans)):
                with self.assertRaises(XAFSError) as ctx:
                    xafs_pipeline.process_sample(
                        [path, path], absorber="Ni", strict=False,
                        self_absorption_formula="NiO", verbose=False)
                self.assertEqual(ctx.exception.stage, "S4.5")
                self.assertIn("身份不兼容", str(ctx.exception))

    def test_despike_preserves_verified_measurement_identity(self):
        from types import SimpleNamespace

        energy = np.arange(101, dtype=float)
        mu = np.linspace(0.3, 1.5, 101)
        spiky = mu.copy()
        spiky[50] += 2.0
        i0 = np.full(101, 100.0)
        scan = RawScan(energy=energy, i0=i0, it=i0 * np.exp(-spiky),
                       mu=spiky, meta={"mode": "given"})
        cleaned, info = athena.despike_scan(scan, kernel=5, threshold=3.0)
        self.assertGreater(info["n_spikes"], 0)
        self.assertFalse(np.allclose(cleaned.mu, np.log(cleaned.i0 / cleaned.it)))
        np.testing.assert_array_equal(cleaned.it, scan.it)
        self.assertIs(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=cleaned))[0], False)
        invalid = RawScan(energy=energy, i0=i0, it=i0 * np.exp(-mu),
                          mu=spiky, meta={"mode": "given", "verified": True})
        washed, _ = athena.despike_scan(invalid, kernel=5, threshold=3.0)
        self.assertIsNone(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=washed))[0])
        bad = SimpleNamespace(meta={"mode": "given", "verified": True},
                              i0=["BAD"], it=[1.0], mu=[0.0])
        self.assertIsNone(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=bad))[0])

    def test_fluorescence_self_absorption_gate(self):
        """荧光自吸收门禁（不做无法验证的校正，只做保守降级）。

        荧光信号被样品自身再吸收会压低 EXAFS 振幅与 N；没有组成/厚度
        就不能定量校正。契约：Δμ≥1.0 判 high 并拒绝定量；0.3–1.0 只
        警告；<0.3 视为稀样；透射/given 模式不受影响；Δμ 缺失时不放行。
        """
        from types import SimpleNamespace

        def diag(edge, mode="fluor"):
            mu = np.linspace(0.2, 1.2, 20)
            res = SimpleNamespace(
                merged=SimpleNamespace(meta={"mode": mode}, mu=mu,
                                       i0=np.full(20, 100.0),
                                       it=100.0 * np.exp(-mu)),
                norm=SimpleNamespace(edge_step=edge))
            return xafs_fit._self_absorption_diag(res)

        high = diag(1.2)
        self.assertEqual(high["risk"], "high")
        self.assertTrue(high["cn_gate"])
        self.assertFalse(high["corrected"])
        warn = diag(0.5)
        self.assertEqual(warn["risk"], "possible")
        self.assertFalse(warn["cn_gate"])
        low = diag(0.05)
        self.assertEqual(low["risk"], "low")
        self.assertFalse(low["cn_gate"])
        trans = diag(1.2, mode="trans")
        self.assertEqual(trans["risk"], "none")
        self.assertFalse(trans["cn_gate"])
        unknown = diag(float("nan"))
        self.assertEqual(unknown["risk"], "unknown")
        self.assertTrue(unknown["cn_gate"])

    def test_booth_self_absorption_recovery(self):
        """Booth 反解必须回收 Booth 前向模型（Eq.2 → Eq.4）。

        用 Eq.2 从已知 χ 生成 χ_exp，再走 booth_self_absorption；
        要求 max|Δχ| ≤ 0.01（含厚样修正项），薄极限严格恒等，
        厚极限保持有限。任何公式符号/单位退化都会先在这里失败。
        """
        k = np.arange(3.0, 12.01, 0.05)
        chi = 0.09 * np.sin(2 * 2.0 * k + 0.4) * np.exp(-2 * 0.004 * k**2)
        e0, d, rho = 7120.0, 4.6, 5.17
        anginp, angout = 49.4, 41.0
        mu_t, mu_a, mu_f, _, _ = athena._booth_mu_terms(
            "Fe3O4", "Fe", "K", "Ka", e0, k, rho)
        sin_phi = np.sin(np.deg2rad(anginp))
        g = sin_phi / np.sin(np.deg2rad(angout))
        alpha = mu_t + g * mu_f
        arg = alpha * d / sin_phi
        arg_true = (alpha + chi * mu_a) * d / sin_phi
        chi_exp = ((1 - np.exp(-arg_true)) / (1 - np.exp(-arg))
                   * alpha * (chi + 1.0) / (alpha + chi * mu_a) - 1.0)
        chi_in = ChiK(k=k, chi=chi_exp, background=np.zeros_like(k),
                      mu_norm=np.ones_like(k), rbkg=1.2)
        out, info = athena.booth_self_absorption(
            chi_in, formula="Fe3O4", elem="Fe", edge="K", line="Ka", e0=e0,
            thickness_um=d, density=rho, anginp=anginp, angout=angout)
        self.assertTrue(info["reliable"])
        self.assertLess(float(np.max(np.abs(out.chi - chi))), 0.01)
        self.assertGreater(info["correction_factor_median"], 1.0)

        thin, _ = athena.booth_self_absorption(
            chi_in, formula="Fe3O4", elem="Fe", edge="K", line="Ka", e0=e0,
            thickness_um=1e-6, density=rho, anginp=anginp, angout=angout)
        self.assertTrue(np.array_equal(thin.chi, chi_exp))

        thick, _ = athena.booth_self_absorption(
            chi_in, formula="Fe3O4", elem="Fe", edge="K", line="Ka", e0=e0,
            thickness_um=1e4, density=rho, anginp=anginp, angout=angout)
        self.assertTrue(np.all(np.isfinite(thick.chi)))

        # 垂直入射（90°）是合法几何，不能因边界检查误拒
        normal, _ = athena.booth_self_absorption(
            chi_in, formula="Fe3O4", elem="Fe", edge="K", line="Ka", e0=e0,
            thickness_um=d, density=rho, anginp=90.0, angout=45.0)
        self.assertTrue(np.all(np.isfinite(normal.chi)))
        with self.assertRaises(XAFSConfigError):
            athena.booth_self_absorption(
                chi_in, formula="Fe3O4", elem="Fe", edge="K", line="Ka",
                e0=e0, thickness_um=d, density=rho, anginp=0.0, angout=45.0)

    def test_booth_end_to_end_synthetic(self):
        """全链合成：真实 μ → Eq.2 前向荧光谱 → 归一化/autobk → Booth。

        这是对 S4.5 最有信息量的契约：只测公式反解（上一项）不够，
        还要覆盖 E0 漂移、边前基线、autobk 归一化和 k 空间形状。
        期望：未校正 χ 振幅明显低于真值（<0.85），校正后回到
        0.9–1.2；若 μ_a 基线或能量锚点退化，这一项会立刻失败。
        """
        import tempfile
        from xraydb import material_mu, material_mu_components, xray_line

        formula, elem, line, d, rho = "Fe3O4", "Fe", "Ka", 4.6, 5.17
        angin, angout = 49.4, 41.0
        scan = xafs_io.read_scan(MAGNETITE)
        E = np.asarray(scan.energy, dtype=float)
        e0 = athena.determine_e0(scan)["e0_sample"]
        norm = athena.normalize_absorption(scan, e0)
        chi_true = athena.extract_chi_k(norm, rbkg=1.10, kweight=2)

        comp = material_mu_components(formula, E, density=rho, kind="photo")
        mass = float(comp["mass"])
        mu_t = np.zeros_like(E)
        for el in comp["elements"]:
            stoich, amass, mu_mass = comp[el]
            mu_t += rho * float(stoich) * float(amass) * np.asarray(mu_mass) / mass
        mu_a = (rho * comp[elem][0] * comp[elem][1]
                * np.asarray(comp[elem][2]) / mass)
        comp_below = material_mu_components(
            formula, float(e0) - 200.0, density=rho, kind="photo")
        mu_a -= (rho * comp_below[elem][0] * comp_below[elem][1]
                 * float(comp_below[elem][2]) / float(comp_below["mass"]))
        mu_t *= 1e-4
        mu_a *= 1e-4
        mu_f = float(material_mu(formula, xray_line(elem, line).energy,
                                 density=rho, kind="total")) * 1e-4

        k_on_e = np.sqrt(np.maximum(K2 * (E - e0), 0.0))
        chi_on_e = np.interp(k_on_e, chi_true.k, chi_true.chi,
                             left=0.0, right=0.0)
        chi_on_e[E <= e0] = 0.0
        # 振荡幅度 = 边跳变 × χ（不是平滑 μ_a(E)·χ）
        c_lo = material_mu_components(formula, e0 - 10.0, density=rho,
                                      kind="photo")
        c_hi = material_mu_components(formula, e0 + 10.0, density=rho,
                                      kind="photo")
        ma_lo = (rho * c_lo[elem][0] * c_lo[elem][1]
                 * float(c_lo[elem][2]) / float(c_lo["mass"])) * 1e-4
        ma_hi = (rho * c_hi[elem][0] * c_hi[elem][1]
                 * float(c_hi[elem][2]) / float(c_hi["mass"])) * 1e-4
        jump = float(ma_hi - ma_lo)
        mu_a_e = mu_a + jump * chi_on_e
        mu_t_e = mu_t + jump * chi_on_e
        sin_phi = np.sin(np.deg2rad(angin))
        g = sin_phi / np.sin(np.deg2rad(angout))
        arg = (mu_t_e / sin_phi + mu_f / np.sin(np.deg2rad(angout))) * d
        arg0 = (mu_t / sin_phi + mu_f / np.sin(np.deg2rad(angout))) * d
        ifluo = (mu_a_e / (mu_t_e + g * mu_f)
                 * (1.0 - np.exp(-arg)) / (1.0 - np.exp(-arg0)) + 0.35)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "forward.xdi"
            lines = ["# XDI/1.0 synthetic-forward", "# Element.symbol: Fe",
                     "# Element.edge: K", "# Column.1: energy eV",
                     "# Column.2: i0", "# Column.3: ifluor", "# -----------"]
            for e, f in zip(E, ifluo):
                lines.append(f"{e:.6f} 1000000.000000 {f * 1e6:.8f}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            common = dict(absorber="Fe", strict=False, rbkg=1.10,
                          kmin=3.0, kmax=12.0, kweight=2, verbose=False)
            raw = xafs_pipeline.process_sample([str(path)], **common)
            corr = xafs_pipeline.process_sample(
                [str(path)], self_absorption_formula=formula,
                self_absorption_thickness=d, self_absorption_density=rho,
                self_absorption_angles=(angin, angout), **common)

        kg = np.linspace(3.0, 12.0, 900)
        truth = np.interp(kg, chi_true.k, chi_true.chi)
        raw_amp = np.std(np.interp(kg, raw.chi_k.k, raw.chi_k.chi)) / np.std(truth)
        cor_amp = np.std(np.interp(kg, corr.chi_k.k, corr.chi_k.chi)) / np.std(truth)
        self.assertLess(raw_amp, 0.85, f"前向谱未被压低: {raw_amp:.3f}")
        self.assertGreater(cor_amp, 0.90, f"校正不足: {cor_amp:.3f}")
        self.assertLess(cor_amp, 1.20, f"校正过头: {cor_amp:.3f}")

    def test_hann_window_contract(self):
        """Hanning 窗函数必须与产线同约定：两端 0、平台 1。"""
        k = np.linspace(0.0, 16.0, 1601)
        kmin, kmax, dk = 3.0, 12.0, 1.0
        w = xafs_plots.hann_window(k, kmin, kmax, dk)
        self.assertTrue(np.all(w >= -1e-12) and np.all(w <= 1 + 1e-12))
        # 窗外严格为 0（端点也含在窗内，但 Hanning 在端点处为 0）
        self.assertTrue(np.allclose(w[k < kmin], 0.0))
        self.assertTrue(np.allclose(w[k > kmax], 0.0))
        # 平台
        plateau = (k > kmin + dk + 0.05) & (k < kmax - dk - 0.05)
        self.assertGreater(plateau.sum(), 100)
        self.assertTrue(np.allclose(w[plateau], 1.0))
        # 左坡单调升、右坡单调降
        left = (k >= kmin) & (k <= kmin + dk)
        right = (k >= kmax - dk) & (k <= kmax)
        self.assertTrue(np.all(np.diff(w[left]) >= -1e-9))
        self.assertTrue(np.all(np.diff(w[right]) <= 1e-9))
        # 中点约 0.5
        mid = int(np.argmin(np.abs(k - (kmin + dk / 2))))
        self.assertAlmostEqual(float(w[mid]), 0.5, places=3)

    def test_xanes_flatten_and_neck_features(self):
        """B2-1：flattened 谱平台≈1，合成颈部双台阶被正确识别。"""
        norm = self.lep.norm
        self.assertIsNotNone(norm.flattened)
        self.assertEqual(len(norm.flattened), len(norm.mu_norm))
        mm = (norm.energy >= norm.e0 + 100) & (norm.energy <= norm.e0 + 400)
        self.assertGreater(int(mm.sum()), 5)
        flat_mean = float(np.mean(norm.flattened[mm]))
        self.assertGreaterEqual(flat_mean, 0.97)
        self.assertLessEqual(flat_mean, 1.03)
        # E<E0 必须保持 mu_norm 不变（flattening 只改边后）
        pre = norm.energy < norm.e0
        np.testing.assert_allclose(norm.flattened[pre], norm.mu_norm[pre])

        # 两个相距 5 eV 的 arctan 台阶（1s→4p_z 颈部肩峰）
        E = np.arange(7100.0, 7300.0, 0.05)
        e0 = 7150.0
        width = 0.8
        e1, e2 = e0 - 2.5, e0 + 2.5
        mu = (0.5 * (0.5 + np.arctan((E - e1) / width) / np.pi)
              + 0.5 * (0.5 + np.arctan((E - e2) / width) / np.pi))
        sn = NormalizedScan(
            energy=E, mu=mu, mu_norm=mu,
            pre_edge_line=np.zeros_like(E), edge_step=1.0,
            plateau=1.0, e0=e0, flattened=mu.copy())
        feat = athena.extract_xanes_features(sn)
        self.assertTrue(feat["neck_shoulder_present"])
        self.assertGreaterEqual(feat["derivative_split_eV"], 4.0)
        self.assertLessEqual(feat["derivative_split_eV"], 6.0)
        self.assertTrue(feat["e0_jump_risk"])

    def test_counterexample_lcf_identical_standards_not_false_precision(self):
        """审查 item 5：完全相同端元 → 权重不可辨识，绝不能报 50%±0。

        并精确复核 sum_to_one 约束的 dof 修正：自由参数是 k-1（约束消去
        1 个），残差方差 σ²=RSS/(n-(k-1))，不是旧的 RSS/(n-k)。
        """
        import xafs_lcf

        x = np.linspace(0.0, 10.0, 200)
        A = np.exp(-((x - 3.0) ** 2) / 0.5)
        B = np.exp(-((x - 7.0) ** 2) / 0.5)

        # ---- 反例：两端元完全相同 → Jp=0 → 秩亏 ----
        res_id = xafs_lcf.run_lcf(
            x, A.copy(), {"s1": (x, A), "s2": (x, A.copy())},
            0.0, 10.0, sum_to_one=True)
        best = res_id["best"]
        self.assertFalse(best["identifiable"], "相同端元必须判不可辨识")
        self.assertFalse(best["significant"],
                         "不可辨识时 significant 必须 False")
        for nm in ("s1", "s2"):
            self.assertFalse(np.isfinite(best["weight_err"][nm]),
                             "不可辨识权重误差必须非有限（杜绝 ±0 虚假精度）")
        self.assertAlmostEqual(sum(best["weights"].values()), 1.0, places=9)
        self.assertFalse(res_id["identifiable"])

        # ---- 基线：不同端元 + 噪声 → 可辨识，dof 用 k-1 ----
        rng = np.random.default_rng(0)
        y = 0.4 * A + 0.6 * B + rng.normal(0, 0.01, x.size)
        res = xafs_lcf.run_lcf(x, y, {"A": (x, A), "B": (x, B)},
                               0.0, 10.0, sum_to_one=True)
        b = res["best"]
        self.assertTrue(b["identifiable"])
        self.assertTrue(b["significant"])
        self.assertLess(abs(b["weights"]["A"] - 0.4), 0.03)
        self.assertLess(abs(b["weights"]["B"] - 0.6), 0.03)
        # 精确复核 dof 修正：σ²=RSS/(n-(k-1))，逐位重算形式误差
        n = b["n_points"]
        k = 2
        S = np.column_stack([A, B])
        w = b["weight_vector"]
        resid = y - S @ w
        dof_correct = n - (k - 1)
        sigma2 = float(np.sum(resid ** 2) / dof_correct)
        Jp = S[:, :-1] - S[:, -1:]
        cov_p = sigma2 * np.linalg.pinv(Jp.T @ Jp)
        Amat = np.vstack([np.eye(k - 1), -np.ones((1, k - 1))])
        cov = Amat @ cov_p @ Amat.T
        exp_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for i, nm in enumerate(["A", "B"]):
            self.assertAlmostEqual(b["weight_err"][nm], float(exp_err[i]),
                                   places=9)
        # 旧 dof(n-k) 会系统性高估残差方差 → 证明修正确实生效
        sigma2_old = float(np.sum(resid ** 2) / (n - k))
        self.assertGreater(sigma2_old, sigma2)

    def test_deadtime_and_explicit_deglitch(self):
        """B2-3：死时间倍乘/分数两条公式 + 宽 glitch 精确删点。"""
        # ---- 死时间：前半 dtime=2（倍乘），后半 dtime=0.5（分数）----
        E = np.arange(7000.0, 7200.0, 0.5)
        i0 = np.full_like(E, 1000.0)
        ifluor = np.full_like(E, 100.0)
        dt = np.where(np.arange(E.size) < E.size // 2, 2.0, 0.5)
        fs = RawScan(energy=E, i0=i0, ifluor=ifluor, mu=ifluor / i0,
                     dtime=dt, meta={"mode": "fluor"})
        out = athena.apply_deadtime_correction(fs)
        self.assertAlmostEqual(float(out.mu[0]), 0.2, places=12)
        self.assertAlmostEqual(float(out.mu[-1]), 0.2, places=12)
        self.assertAlmostEqual(float(fs.mu[0]), 0.1, places=12,
                               msg="死时间校正不得原地修改输入扫描")
        self.assertTrue(out.meta["deadtime_corrected"])

        # ---- 宽方波 glitch：自动 despike 只能削掉边缘，deglitch 精确删点 ----
        g = xafs_io.read_scans([str(LEPIDO)])[0]
        mu2 = np.array(g.mu, dtype=float, copy=True)
        near = np.flatnonzero((g.energy >= 7280.0) & (g.energy <= 7295.0))
        start = int(near[0]) if near.size else int(
            np.argmin(np.abs(g.energy - 7280.0)))
        gi = np.arange(start, min(start + 8, len(g.energy)))
        mu2[gi] += 2.0
        glitched = RawScan(energy=np.array(g.energy, copy=True),
                           mu=mu2, meta=dict(g.meta))
        # 复刻 pipeline 的去毛刺（kernel=5, 3×MAD）
        padded = np.pad(mu2, 2, mode="edge")
        med = np.array([np.median(padded[i:i + 5])
                        for i in range(len(mu2))])
        resid = np.abs(mu2 - med)
        mad = float(np.median(resid)) + 1e-12
        despiked = mu2.copy()
        mask = resid > 3.0 * mad
        despiked[mask] = med[mask]
        remain = float(np.max(np.abs(despiked[gi] - g.mu[gi])))
        self.assertGreater(remain, 1.0,
                           "自动 despike 不应能清除 8 点宽方波")
        lo, hi = float(g.energy[gi].min()), float(g.energy[gi].max())
        cleaned, info = athena.deglitch_ranges(glitched, [(lo, hi)])
        self.assertEqual(info["n_removed"], int(gi.size))
        self.assertFalse(np.any((cleaned.energy >= lo)
                                & (cleaned.energy <= hi)))
        with self.assertRaises(XAFSConfigError):
            athena.deglitch_ranges(glitched, [(7000.0, 8000.0)])

    def test_merge_scans_outlier_rejection_and_sigma(self):
        """B6-1：多扫描合并的离群剔除、σ_μ 传播与默认逐位兼容。

        默认 reject_outliers=False 必须与历史均值逐位一致；开启后
        第 4 条 +0.15 畸变扫描按中位谱 RMS+MAD 剔除，合并结果与
        三条干净扫描逐位相同；σ_μ 为保留扫描的逐点标准差。
        保底护栏：2-vs-1 时只剔 1 条，绝不把扫描集压到 1 条。
        """
        files = [LEPIDO.parent / f"Fe_lepidocrocite.{ext}"
                 for ext in ("000", "100", "200")]
        if not all(f.exists() for f in files):
            self.skipTest("缺少 lepidocrocite 三扫描数据")
        scans = xafs_io.read_scans([str(f) for f in files])
        self.assertEqual(len(scans), 3)

        clean = athena.merge_scans(scans)
        self.assertIsNotNone(clean.sigma_mu, "多扫描合并未传播 sigma_mu")
        self.assertEqual(clean.sigma_mu.shape, clean.energy.shape)
        self.assertTrue(np.all(np.isfinite(clean.sigma_mu)))
        self.assertTrue(np.all(np.asarray(clean.sigma_mu) >= 0.0))
        legacy = np.array([s.mu for s in scans]).mean(axis=0)
        self.assertTrue(np.array_equal(clean.mu, legacy),
                        "reject_outliers=False 改变了默认合并结果")
        self.assertFalse(clean.meta["merge_info"]["reject_outliers"])
        self.assertEqual(clean.meta["merge_info"]["n_kept"], 3)
        self.assertEqual(clean.meta["merge_info"]["rejected_indices"], [])

        bad = xafs_io.read_scans([str(files[0])])[0]
        bad.mu = np.asarray(bad.mu, dtype=float) + 0.15
        merged = athena.merge_scans(scans + [bad], reject_outliers=True)
        info = merged.meta["merge_info"]
        self.assertEqual(info["rejected_indices"], [3],
                         f"离群扫描未被正确剔除: {info}")
        self.assertEqual(info["n_kept"], 3)
        self.assertTrue(np.array_equal(merged.mu, clean.mu),
                        "剔除离群扫描后合并谱与干净三扫描不一致")
        self.assertIsNotNone(merged.sigma_mu)
        self.assertTrue(np.all(np.asarray(merged.sigma_mu) >= 0.0))

        # 保底护栏：3 条里 2 条接近、1 条远离 → 只剔离群者，保留 2 条
        E = np.arange(7100.0, 7200.0, 0.2)

        def _mk(value):
            return RawScan(energy=E, mu=np.full_like(E, float(value)))

        floored = athena.merge_scans(
            [_mk(0.0), _mk(100.0), _mk(101.0)], reject_outliers=True)
        f_info = floored.meta["merge_info"]
        self.assertEqual(f_info["n_kept"], 2,
                         f"保底护栏失效: {f_info}")
        self.assertEqual(f_info["rejected_indices"], [0])

    def test_xanes_fluo_self_absorption_restores_whiteline_and_lcf(self):
        """B6-2：XANES 荧光自吸收 Tröger 反解恢复白线与 LCF 权重。

        真值 = 0.5·LEPIDO + 0.5·GOETHITE flattened；按 Fe₂O₃ 浓样
        Eq.2 正演压缩（白线压低 >15%），反解后白线误差 <0.02，
        LCF 权重回到 0.50±0.02，R-factor 相对未校正下降 >10 倍。
        同时核对分母防爆门禁：α 较小的 45°/45° 几何下必须
        reliable=False 且 clamped_points>0，但输出仍有限。
        """
        import xafs_lcf

        goe = xafs_pipeline.process_sample(
            [GOETHITE], absorber="Fe", strict=False, verbose=False)
        x = np.asarray(self.lep.norm.energy, dtype=float)
        e0 = float(self.lep.norm.e0)
        y_lep = np.asarray(self.lep.norm.flattened, dtype=float)
        y_goe = np.interp(x, goe.norm.energy, goe.norm.flattened)
        y_true = 0.5 * y_lep + 0.5 * y_goe

        def _norm(y):
            return NormalizedScan(
                energy=np.array(x, copy=True), mu=np.asarray(y, dtype=float),
                mu_norm=np.asarray(y, dtype=float),
                pre_edge_line=np.zeros_like(x), edge_step=1.0,
                plateau=1.0, e0=e0, flattened=np.asarray(y, dtype=float))

        # 用公开 API 在占位谱上读取同一几何的 α，避免在测试里复制私有公式
        dummy = _norm(np.ones_like(x))
        _, info0 = athena.fluo_xanes_self_absorption(
            dummy, "Fe2O3", "Fe", "K", "Ka", 100.0, 5.24, 45.0, 20.0)
        alpha = float(info0["alpha"])
        self.assertGreater(alpha, 0.3)

        y_att = (alpha + 1.0) * y_true / (alpha + y_true)
        drop = 1.0 - float(np.max(y_att)) / float(np.max(y_true))
        self.assertGreater(drop, 0.15,
                           f"正演白线压缩仅 {100 * drop:.1f}% (<15%)，"
                           f"测试未覆盖浓样自吸收场景")

        norm_att = _norm(y_att)
        corr, info = athena.fluo_xanes_self_absorption(
            norm_att, "Fe2O3", "Fe", "K", "Ka", 100.0, 5.24, 45.0, 20.0)
        self.assertTrue(info["reliable"], info.get("warning"))
        self.assertEqual(info["clamped_points"], 0)
        self.assertAlmostEqual(info["alpha"], alpha, places=9)
        wl_err = abs(float(np.max(corr.flattened)) - float(np.max(y_true)))
        self.assertLess(wl_err, 0.02,
                        f"白线恢复误差 {wl_err:.4f} ≥0.02")
        # 输入对象不得被原地修改
        self.assertLess(float(np.max(np.abs(
            np.asarray(norm_att.flattened) - y_att))), 1e-12)

        xr = x - e0
        standards = {"lepido": (xr, y_lep), "goethite": (xr, y_goe)}
        r_att = xafs_lcf.run_lcf(xr, y_att, standards, -10.0, 60.0)
        r_cor = xafs_lcf.run_lcf(xr, corr.flattened, standards, -10.0, 60.0)
        for name in ("lepido", "goethite"):
            self.assertLess(abs(r_cor["weights"][name] - 0.5), 0.02,
                            f"校正后 {name} 权重 "
                            f"{r_cor['weights'][name]:.4f} 偏离 0.50 超过 0.02")
        self.assertGreater(r_att["r_factor"], 10.0 * r_cor["r_factor"],
                           f"LCF R-factor 未下降 >10 倍: "
                           f"{r_att['r_factor']:.4g} → "
                           f"{r_cor['r_factor']:.4g}")

        # 分母防爆门禁：α≈0.42 的 45°/45° 几何必然触界
        _, info45 = athena.fluo_xanes_self_absorption(
            dummy, "Fe2O3", "Fe", "K", "Ka", 100.0, 5.24, 45.0, 45.0)
        alpha45 = float(info45["alpha"])
        y_att45 = (alpha45 + 1.0) * y_true / (alpha45 + y_true)
        corr45, guard = athena.fluo_xanes_self_absorption(
            _norm(y_att45), "Fe2O3", "Fe", "K", "Ka",
            100.0, 5.24, 45.0, 45.0)
        self.assertFalse(guard["reliable"],
                         "45°/45° 下分母钳制未触发 reliable=False")
        self.assertGreater(guard["clamped_points"], 0)
        self.assertTrue(np.all(np.isfinite(corr45.flattened)))

    def test_pre_edge_voigt_deconvolution_and_centroid(self):
        """B6-3：主边拖尾基线 + 双 Pseudo-Voigt 分峰与面积加权质心。

        合成峰：E_c1=7113.0 (A=0.08)、E_c2=7114.6 (A=0.16)，
        真值质心 7114.067 eV。要求 n_peaks=2、质心误差 <0.15 eV、
        子峰中心误差 <0.20 eV；并核对 extract_xanes_features 与
        指标表已写入 pre_edge_centroid。
        """
        E = np.linspace(7090.0, 7160.0, 701)
        E0 = 7125.0

        def _pv(energy, area, center, gamma, eta):
            # 独立实现（不复用被测函数），避免公式错误相互抵消
            xx = energy - center
            lor = (gamma / np.pi) / (xx ** 2 + gamma ** 2)
            gau = (np.sqrt(np.log(2.0)) / (gamma * np.sqrt(np.pi))
                   * np.exp(-np.log(2.0) * (xx / gamma) ** 2))
            return area * (eta * lor + (1.0 - eta) * gau)

        y = (0.02 + 1.5 / ((E - (E0 + 2.0)) ** 2 + 3.0 ** 2)
             + _pv(E, 0.08, 7113.0, 0.7, 0.5)
             + _pv(E, 0.16, 7114.6, 0.7, 0.5))
        norm = NormalizedScan(
            energy=E, mu=y, mu_norm=y, pre_edge_line=np.zeros_like(E),
            edge_step=1.0, plateau=1.0, e0=E0, flattened=y)
        res = athena.fit_pre_edge_peaks(norm)
        self.assertEqual(res["n_peaks"], 2,
                         f"双峰未被识别: n_peaks={res['n_peaks']}")
        centroid_true = (0.08 * 7113.0 + 0.16 * 7114.6) / 0.24
        self.assertLess(abs(res["centroid_energy"] - centroid_true), 0.15,
                        f"质心 {res['centroid_energy']:.4f} vs 真值 "
                        f"{centroid_true:.4f}")
        centers = sorted(p["center"] for p in res["peaks"])
        self.assertLess(abs(centers[0] - 7113.0), 0.20,
                        f"子峰 1 中心 {centers[0]:.4f} vs 7113.0")
        self.assertLess(abs(centers[1] - 7114.6), 0.20,
                        f"子峰 2 中心 {centers[1]:.4f} vs 7114.6")
        self.assertLess(res["r_factor"], 0.05)
        self.assertAlmostEqual(
            res["total_area"], sum(p["area"] for p in res["peaks"]),
            places=12)

        # 集成：真实管线结果必须带该字段；指标表必须有对应行
        self.assertIn("pre_edge_centroid", self.lep.xanes_features)
        self.assertIn("pre_edge_n_peaks", self.lep.xanes_features)
        st = {"chi_k": self.lep.chi_k, "r1": self.lep.r1,
              "wt": self.lep.wt, "norm": self.lep.norm,
              "e0": self.lep.e0["e0_sample"], "rbkg": self.lep.rbkg,
              "xanes_features": self.lep.xanes_features}
        with tempfile.TemporaryDirectory() as tmp:
            rows = xafs_report.export_indicators_single(
                st, None, Path(tmp), "pe", "Fe2O3", "Fe K", 0.803, [])
            names = [r[0] for r in rows]
            self.assertIn("pre_edge_centroid", names)
            self.assertIn("pre_edge_n_peaks", names)

    def test_sigma_mu_export_and_high_k_snr_gate(self):
        """#48 B9-2：三扫描合并的 σ_μ 落盘与高 k SNR 单调性。

        ``_mu.dat`` 必须有第 4 列 sigma_mu；注入递增噪声时 k_snr2 必须
        单调下降；没有 sigma_mu 时返回 None 并给出原因，不伪造曲线。
        """
        import run_xafs

        files = [LEPIDO, LEPIDO.with_suffix(".100"),
                 LEPIDO.with_suffix(".200")]
        for f in files:
            self.assertTrue(f.is_file(), f"缺少测试扫描 {f}")
        with tempfile.TemporaryDirectory() as tmp:
            res = run_xafs.process_sample(
                [str(f) for f in files], Path(tmp), absorber="Fe",
                strict=False, verbose=False)
            self.assertIsNotNone(res.merged.sigma_mu,
                                 "三扫描合并未保留 sigma_mu")
            mu_path = Path(tmp) / f"{res.stem}_mu.dat"
            lines = mu_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("sigma_mu", lines[1])
            self.assertEqual(len(lines[2].split()), 4,
                             "_mu.dat 不是 4 列（E, mu, mu_norm, sigma_mu）")

        ck = res.chi_k
        edge = abs(float(res.norm.edge_step))
        k_snr2 = []
        for scale in (1e-3, 3e-3, 1e-2, 3e-2):
            out = athena.high_k_snr(
                ck, sigma_mu=np.full_like(ck.k, scale * edge),
                edge_step=res.norm.edge_step, kweight=2)
            self.assertIsNotNone(out["k_snr2"],
                                 f"噪声 scale={scale} 未给出 k_snr2")
            k_snr2.append(out["k_snr2"])
        for hi_noise, lo_noise in zip(k_snr2[1:], k_snr2[:-1]):
            self.assertLess(hi_noise, lo_noise - 0.2,
                            f"k_snr2 未随噪声单调下降: {k_snr2}")
        none = athena.high_k_snr(ck)
        self.assertIsNone(none["k_snr2"])
        self.assertIn("sigma_mu", none["reason"])
        self.assertIsNotNone(res.snr["k_snr2"])

    def test_mback_short_window_normalization(self):
        """#51 B10-1：短边后窗口 MBACK 归一化的 edge_step 回收。

        Fe K 理论截面合成 μ(E)（边后仅 60–150 eV），MBACK 的 edge_step
        相对误差必须 <3%；传统二次多项式归一化在同窗口平台发 NaN 且
        偏差更大。a≤0 时干净拒绝。
        """
        from xraydb import mu_elam

        energy = np.linspace(6900.0, 7280.0, 650)
        mu = (1.7 * mu_elam("Fe", energy)
              + 0.02 * (energy - 7112.0) + 0.3)
        scan = RawScan(energy=energy, mu=mu, meta={})
        truth = 1.7 * (mu_elam("Fe", 7162.0) - mu_elam("Fe", 7062.0))
        mback = athena.normalize_mback(
            scan, 7112.0, "Fe", post_window=(60.0, 150.0))
        rel = abs(mback.edge_step - truth) / truth
        self.assertLess(rel, 0.03,
                        f"MBACK edge_step 相对误差 {rel:.3e} ≥3%")
        self.assertAlmostEqual(mback.edge_step, truth, places=8)
        self.assertTrue(np.isfinite(mback.plateau))
        self.assertEqual(mback.engine, "mback")
        self.assertEqual(mback.mback["element"], "Fe")
        self.assertGreater(mback.mback["a"], 0.0)

        trad = athena.normalize_absorption(
            scan, 7112.0, post_window=(60.0, 150.0))
        trad_rel = abs(trad.edge_step - truth) / truth
        self.assertGreater(trad_rel, 0.01,
                           "传统短窗口归一化不应比 MBACK 更准")
        self.assertGreater(trad_rel, 5.0 * rel)
        self.assertFalse(np.isfinite(trad.plateau))

        negative = RawScan(
            energy=energy,
            mu=-1.7 * mu_elam("Fe", energy) + 0.02 * (energy - 7112.0) + 0.3,
            meta={})
        with self.assertRaises(XAFSError):
            athena.normalize_mback(
                negative, 7112.0, "Fe", post_window=(60.0, 150.0))

    def test_autobk_chi_std_prior_and_clamp(self):
        """#52 B10-1：AUTOBK 先验谱压低低 R 泄漏，clamp 生效且合法。

        合成谱 = 真实 Fe–O χ(k) + 0.7 Å 低频伪振荡 + 噪声。以真实
        χ(k) 作为 chi_std 时，R<1.2 Å 内“提取谱−真值”的泄漏必须
        降到无先验的一半以下；非法长度必须报 XAFSConfigError。
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
        datfile = first[0]
        k = np.arange(0.0, 15.0, 0.05)
        p = artemis.load_feff_path(datfile)
        p.degen = 6.0
        artemis.set_path_parameters(
            p, s02=0.803, delta_r=0.0, sigma2=0.004, delta_e0=0.0)
        true_chi = artemis.exafs_model_k([p], k)
        rng = np.random.default_rng(3)
        leak = 0.03 * np.sin(2.0 * 0.7 * k + 0.4)
        noise = rng.normal(0.0, 0.002, k.size)
        mu = 1.0 + true_chi + leak + noise
        energy = 7112.0 + k ** 2 / K2
        norm = NormalizedScan(
            energy=energy, mu=mu, mu_norm=mu,
            pre_edge_line=np.zeros_like(mu), edge_step=1.0,
            plateau=1.0, e0=7112.0, flattened=mu.copy())

        trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        ft_true = athena.ft_to_r_space(k, true_chi, None, 3.0, 12.0, 2)
        r_mask = (ft_true.r >= 0.2) & (ft_true.r < 1.2)

        def low_r_leak(chi_k):
            ft = athena.ft_to_r_space(
                chi_k.k, chi_k.chi, None, 3.0, 12.0, 2)
            re = np.interp(ft_true.r, ft.r, ft.chi_r.real)
            im = np.interp(ft_true.r, ft.r, ft.chi_r.imag)
            diff = (re + 1j * im) - ft_true.chi_r
            return float(trap(np.abs(diff[r_mask]), ft_true.r[r_mask]))

        plain = athena.extract_chi_k(norm, rbkg=1.2, kweight=2)
        with_std = athena.extract_chi_k(
            norm, rbkg=1.2, kweight=2, chi_std=true_chi, k_std=k)
        leak_plain = low_r_leak(plain)
        leak_std = low_r_leak(with_std)
        self.assertTrue(with_std.chi_std_used,
                        "chi_std 未被记录为已使用")
        self.assertLess(leak_std, 0.5 * leak_plain,
                        f"低 R 泄漏未下降: plain={leak_plain:.4f}, "
                        f"std={leak_std:.4f}")

        clamped = athena.extract_chi_k(
            norm, rbkg=1.2, kweight=2, clamp_lo=0.1, clamp_hi=0.1)
        self.assertAlmostEqual(clamped.clamp_lo, 0.1)
        self.assertAlmostEqual(clamped.clamp_hi, 0.1)
        self.assertTrue(np.all(np.isfinite(clamped.chi)))
        self.assertFalse(np.array_equal(clamped.chi, plain.chi),
                         "clamp_lo/hi 未改变 AUTOBK 解")

        with self.assertRaises(XAFSConfigError):
            athena.extract_chi_k(
                norm, rbkg=1.2, chi_std=np.ones(10), k_std=np.arange(9.0))
        with self.assertRaises(XAFSConfigError):
            athena.extract_chi_k(
                norm, rbkg=1.2, chi_std=np.ones(20))

    def test_pca_target_transform_spoil_and_mcr_als(self):
        """#53 B10-2：SPOIL 判定与 MCR-ALS 盲解谱。

        3 个独立端元 + 9 条混合谱（含纯端元，浓度和=1）加 0.5% 噪声：
        真实端元 SPOIL<3、随机谱>6；MCR-ALS 组分谱相关系数>0.99、
        浓度 RMSE<0.02，且 closure=True 时收敛。
        """
        import xafs_lcf

        x = np.linspace(0.0, 10.0, 400)

        def g(c, w):
            return np.exp(-0.5 * ((x - c) / w) ** 2)

        e1 = g(2.0, 0.35) + 0.10 * g(4.0, 0.6)
        e2 = g(5.0, 0.55) + 0.15 * g(7.5, 0.5)
        e3 = g(8.0, 0.40) + 0.20 * g(3.0, 0.4)
        truth = np.vstack([e1, e2, e3])
        weights = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.2, 0.5],
            [0.6, 0.1, 0.3], [0.15, 0.15, 0.7], [0.25, 0.55, 0.2]])
        rng = np.random.default_rng(11)
        scale = float(np.max(truth))
        specs = [(f"s{i}", x, weights[i] @ truth
                  + rng.normal(0.0, 0.005 * scale, x.size))
                 for i in range(len(weights))]
        pca = xafs_lcf.run_pca(specs, 0.0, 10.0)
        self.assertEqual(pca["x"].shape, x.shape)
        self.assertEqual(np.asarray(pca["components"]).shape,
                         (x.size, len(weights)))
        tt = xafs_lcf.target_transform(
            pca, [("true1", x, e1), ("true2", x, e2),
                  ("true3", x, e3),
                  ("rand", x, rng.normal(0.0, 1.0, x.size))],
            n_components=3)
        for name in ("true1", "true2", "true3"):
            self.assertLess(tt[name]["spoils"], 3.0,
                            f"{name} SPOIL={tt[name]['spoils']:.3f} ≥3")
            self.assertEqual(tt[name]["rating"], "good")
        self.assertGreater(tt["rand"]["spoils"], 6.0,
                           f"随机谱 SPOIL={tt['rand']['spoils']:.3f} ≤6")

        mcr = xafs_lcf.mcr_als(
            specs, 3, max_iter=1500, tol=1e-10, closure=True)
        self.assertTrue(mcr["converged"], "MCR-ALS 未收敛")
        self.assertLess(mcr["r_factor"], 0.03)
        comp = np.asarray(mcr["components"])
        conc = np.asarray(mcr["concentrations"])
        self.assertEqual(comp.shape, (x.size, 3))
        self.assertEqual(conc.shape, (3, len(weights)))
        # 组分顺序由数据决定，贪心匹配真值后计算相关与 RMSE
        perm = []
        for i in range(3):
            candidates = [j for j in range(3) if j not in perm]
            cors = [abs(np.corrcoef(comp[:, i], truth[j])[0, 1])
                    for j in candidates]
            perm.append(candidates[int(np.argmax(cors))])
        for i, j in enumerate(perm):
            corr = abs(np.corrcoef(comp[:, i], truth[j])[0, 1])
            self.assertGreater(corr, 0.99,
                               f"组分 {i} 与真值 {j} 相关 {corr:.4f} ≤0.99")
        rmse = float(np.sqrt(np.mean(
            (conc[perm, :] - weights.T) ** 2)))
        self.assertLess(rmse, 0.02, f"浓度 RMSE={rmse:.4f} ≥0.02")

    def test_multi_edge_e0_groups_and_aic_bic(self):
        """#59 B12-1：双吸收边独立 ΔE₀ 分组与 AIC/BIC 惩罚。

        两条同路径合成谱真值 ΔE₀=+2.0 / −6.0 eV，edge_group=K/L3：
        share_e0="auto" 分出 2 个槽位并各自回收 <0.5 eV；
        share_e0=True 只有 1 个槽位、Rf 显著变差；BIC 对多出的参数
        惩罚更重（auto 相对 True 的 BIC 优势小于 AIC 优势）。
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
        datasets = []
        for group, e0_true in (("K", 2.0), ("L3", -6.0)):
            p = artemis.load_feff_path(datfile)
            p.degen = 6.0
            artemis.set_path_parameters(
                p, s02=1.0, delta_r=-0.02, sigma2=0.004,
                delta_e0=e0_true)
            chi = artemis.exafs_model_k([p], k)
            ck = ChiK(k=k, chi=chi, background=np.zeros_like(k),
                      mu_norm=np.ones_like(k), rbkg=1.0)
            ft = athena.ft_to_r_space(k, chi, None, 3.0, 12.0, 2)
            datasets.append({
                "name": group, "chi_k": ck, "ft": ft,
                "datfiles": [str(datfile)], "rmin": 1.2, "rmax": 2.4,
                "kmin": 3.0, "kmax": 12.0, "kweight": 2, "s02": 1.0,
                "vary_n": False, "path_degen": [6.0],
                "path_labels": ["Fe-O"], "edge_group": group})

        auto = artemis.fit_multi_datasets(datasets, share_e0="auto")
        shared = artemis.fit_multi_datasets(datasets, share_e0=True)
        self.assertEqual(auto["e0_groups"], ["K", "L3"])
        self.assertEqual(shared["e0_groups"], ["global"])
        self.assertEqual(auto["n_variables_total"],
                         shared["n_variables_total"] + 1)
        self.assertAlmostEqual(auto["datasets"][0].delta_e0, 2.0, places=6)
        self.assertAlmostEqual(auto["datasets"][1].delta_e0, -6.0, places=6)
        self.assertLess(abs(auto["datasets"][0].delta_e0 - 2.0), 0.5)
        self.assertLess(abs(auto["datasets"][1].delta_e0 + 6.0), 0.5)
        self.assertLess(auto["global_r_factor"], 1e-6)
        self.assertGreater(shared["global_r_factor"], 0.01)

        for key in ("aic", "bic", "global_aic", "global_bic"):
            self.assertTrue(np.isfinite(auto[key]), f"auto {key} 非有限")
            self.assertTrue(np.isfinite(shared[key]),
                            f"shared {key} 非有限")
        for fit in auto["datasets"] + shared["datasets"]:
            self.assertTrue(np.isfinite(fit.aic))
            self.assertTrue(np.isfinite(fit.bic))
            self.assertGreater(fit.bic, fit.aic)
        # auto 多 1 个 ΔE₀ 参数：BIC 惩罚使其相对 shared 的优势变小
        aic_gain = shared["aic"] - auto["aic"]
        bic_gain = shared["bic"] - auto["bic"]
        self.assertGreater(aic_gain, bic_gain,
                           "BIC 未对多参数模型施加更重惩罚")

    def test_operando_series_pca_spoil_lcf_and_mcr_tracking(self):
        """#66 B15-1：原位序列一键解谱与演变轨迹。

        LEPIDO/GOETHITE 按 w=[0,0.2,…,1.0] 线性混合成 6 条序列，
        候选标样加一个纯噪声伪标样：
        - SPOIL 必须只拒绝 fake_noise；
        - LCF 回收 goethite 权重 RMSE <0.02；
        - MCR 两条组分浓度随序号单调（|r|>0.98），四宫格与 CSV 落盘。
        """
        import xafs_lcf

        goe = xafs_pipeline.process_sample(
            [GOETHITE], absorber="Fe", strict=False, verbose=False).norm
        lep = self.lep.norm
        x = np.arange(-20.0, 60.0, 0.2)

        def flat(n):
            arr = n.flattened if n.flattened is not None else n.mu_norm
            return np.interp(x, n.energy - n.e0, arr)

        y_lep, y_goe = flat(lep), flat(goe)
        rng = np.random.default_rng(66)
        w_true = np.linspace(0.0, 1.0, 6)
        items = []
        for i, w in enumerate(w_true):
            yy = ((1.0 - w) * y_lep + w * y_goe
                  + rng.normal(0, 1e-3, x.size))
            items.append((f"scan_{i:02d}", x, yy))
        fake_y = rng.normal(0.0, 0.1, x.size)
        standards = {"lepido": lep, "goethite": goe,
                     "fake_noise": (x, fake_y)}

        out = xafs_lcf.run_operando_series(
            items, standards=standards, xmin=-20.0, xmax=60.0)
        self.assertEqual(out["n_components"], 2)
        self.assertEqual(out["accepted_standards"],
                         ["lepido", "goethite"])
        self.assertEqual(out["rejected_standards"], ["fake_noise"])

        got = np.asarray([row["weights"].get("goethite", np.nan)
                          for row in out["lcf_series"]], dtype=float)
        self.assertEqual(got.shape, w_true.shape)
        rmse = float(np.sqrt(np.mean((got - w_true) ** 2)))
        self.assertLess(rmse, 0.02, f"LCF goethite 权重 RMSE={rmse}")

        conc = np.asarray(out["mcr_als"]["concentrations"], dtype=float)
        self.assertEqual(conc.shape, (2, 6))
        idx = np.arange(6, dtype=float)
        corr = [abs(float(np.corrcoef(idx, conc[j])[0, 1]))
                for j in range(2)]
        self.assertTrue(all(c > 0.98 for c in corr),
                        f"MCR 组分未随序号单调: {corr}")
        self.assertEqual(len(out["mcr_als"]["characteristic_energy"]), 2)
        self.assertLess(out["mcr_als"]["characteristic_energy"][0],
                        out["mcr_als"]["characteristic_energy"][1])
        self.assertEqual(len(out["descriptors"]), 6)
        for key in ("e0", "e_half_height", "white_line_height", "ft_r1"):
            self.assertIn(key, out["descriptors"][0])

        with tempfile.TemporaryDirectory() as tmp:
            made = xafs_plot_special.write_operando_diagnostics(
                out, tmp, stem="op")
            conc_csv = Path(made["concentrations_csv"])
            comp_csv = Path(made["components_csv"])
            png = Path(made["png"])
            for p in (conc_csv, comp_csv, png):
                self.assertTrue(p.is_file() and p.stat().st_size > 0,
                                f"缺少产物: {p}")
            lines = conc_csv.read_text(
                encoding="utf-8-sig").splitlines()
            self.assertEqual(len(lines), 7, "浓度表应为 6 条谱 + 表头")
            self.assertIn("lcf_goethite", lines[0])
            self.assertIn("mcr_component_1", lines[0])
            comp_lines = comp_csv.read_text(
                encoding="utf-8-sig").splitlines()
            self.assertGreater(len(comp_lines), 10)
            self.assertIn("mcr_component_2", comp_lines[0])

        with self.assertRaises(XAFSConfigError):
            xafs_lcf.run_operando_series(items[:2])

    def test_counterexample_xanes_self_absorption_thickness(self):
        """反例（审查 item 2）：XANES 荧光自吸收必须含厚度适用域。

        历史缺陷：fluo_xanes_self_absorption 收 thickness_um 却从不用于
        反解（纯厚样 Tröger），薄样被过度校正。正确：光学厚度
        τ=d·μ_a/sinφ 决定薄/中/厚——薄样→恒等不校正，中厚度→有限厚度
        正演数值反解，厚样→闭式 Tröger（与历史逐位一致）。且 τ 由真实
        厚度/密度/截面算出，绝不拿荧光 edge_step 当光学厚度。
        """
        E = np.arange(7000.0, 7300.0, 1.0)
        e0 = 7123.0
        # 温和白线（~1.44），45/20 几何 α 较大，不触分母钳制
        y_true = (0.5 * (1 + np.tanh((E - e0) / 3.0))
                  + 0.45 * np.exp(-((E - (e0 + 6)) / 10.0) ** 2))
        m = E >= e0 - 15.0
        args = ("Fe2O3", "Fe", "K", "Ka")
        geom = (45.0, 20.0)

        def _norm(y, es=1.0):
            return NormalizedScan(
                energy=np.array(E, copy=True), mu=np.asarray(y, float),
                mu_norm=np.asarray(y, float),
                pre_edge_line=np.zeros_like(E), edge_step=es,
                plateau=1.0, e0=e0, flattened=np.asarray(y, float))

        def _forward(y, th):
            # 独立写出有限厚度正演（Tröger）：μ_exp=(α+1)x/(α+x)·
            # [1−e^{−τ(α+x)}]/[1−e^{−τ(α+1)}]；α/τ 由公开 API 读取
            _, ii = athena.fluo_xanes_self_absorption(
                _norm(np.ones_like(E)), *args, th, 5.24, *geom)
            a = ii["alpha"]
            tau = ii["optical_thickness_tau"]
            ya = ((a + 1.0) * y / (a + y)
                  * (-np.expm1(-tau * (a + y)))
                  / (-np.expm1(-tau * (a + 1.0))))
            return ya, ii

        # ① 显式薄样分支：光学厚度可忽略 → 恒等，不校正
        y_att_thin, ii_thin = _forward(y_true, 1e-7)
        self.assertLess(float(np.max(np.abs(y_att_thin - y_true))), 1e-4)
        corr_thin, info_thin = athena.fluo_xanes_self_absorption(
            _norm(y_att_thin), *args, 1e-7, 5.24, *geom)
        self.assertEqual(info_thin["thickness_regime"], "thin")
        self.assertTrue(info_thin["thin_limit_no_correction"])
        self.assertTrue(np.array_equal(corr_thin.flattened, y_att_thin))

        # ② 现实薄样（0.01 μm）：校正可忽略（≈恒等）
        y_att_01, _ = _forward(y_true, 0.01)
        corr_01, _ = athena.fluo_xanes_self_absorption(
            _norm(y_att_01), *args, 0.01, 5.24, *geom)
        self.assertLess(
            float(np.max(np.abs(corr_01.flattened - y_att_01))), 0.01)

        # ③ 中厚度（20 μm）：正演压缩显著，反解精确回收真值
        y_att_20, ii_20 = _forward(y_true, 20.0)
        self.assertEqual(ii_20["thickness_regime"], "mid")
        comp20 = 1.0 - float(np.max(y_att_20)) / float(np.max(y_true))
        self.assertGreater(comp20, 0.10)
        corr_20, info_20 = athena.fluo_xanes_self_absorption(
            _norm(y_att_20), *args, 20.0, 5.24, *geom)
        self.assertTrue(info_20["reliable"], info_20.get("warning"))
        self.assertEqual(info_20["clamped_points"], 0)
        self.assertLess(
            float(np.max(np.abs(corr_20.flattened[m] - y_true[m]))), 0.02)

        # ④ 厚度依赖：压缩随厚度单调增并饱和；τ∝d；厚样回归闭式分支
        y_att_5, ii_5 = _forward(y_true, 5.0)
        y_att_40, ii_40 = _forward(y_true, 40.0)
        comp5 = 1.0 - float(np.max(y_att_5)) / float(np.max(y_true))
        comp40 = 1.0 - float(np.max(y_att_40)) / float(np.max(y_true))
        self.assertLess(comp5, comp20)
        self.assertLess(comp20, comp40)
        self.assertAlmostEqual(
            ii_40["optical_thickness_tau"] / ii_5["optical_thickness_tau"],
            8.0, places=6)
        _, ii_100 = athena.fluo_xanes_self_absorption(
            _norm(np.ones_like(E)), *args, 100.0, 5.24, *geom)
        self.assertEqual(ii_100["thickness_regime"], "thick")

        # ⑤ edge_step 无关：同厚度不同 edge_step → 校正逐位一致
        #    （光学厚度来自 d·μ_a，绝不来自荧光 edge_step）
        cA, _ = athena.fluo_xanes_self_absorption(
            _norm(y_att_20, 1.0), *args, 20.0, 5.24, *geom)
        cB, _ = athena.fluo_xanes_self_absorption(
            _norm(y_att_20, 2.5), *args, 20.0, 5.24, *geom)
        self.assertTrue(np.array_equal(cA.flattened, cB.flattened))

    def test_counterexample_merge_preserves_mode_and_channels(self):
        """反例（审查 item 12）：合并必须保留测量模式与通道来源。

        历史缺陷：merge_scans 只留 μ，把 I0/It/If/Ir 丢成 None，荧光谱
        合并后失去通道证据（S4.5 自吸收门禁的 _has_if 失效），参考通道
        无法在 S2 交叉校验。正确：kept 扫描逐通道在公共网格算术平均；
        μ 仍为逐扫描 μ 均值（均值对数 ≠ 对数均值，默认逐位不变）；模式与
        通道溯源保留；dtime 置 None 防二次校正；通道不齐不臆造平均。
        """
        E = np.arange(7000.0, 7300.0, 2.0)
        mu1 = 0.5 * (1 + np.tanh((E - 7125.0) / 4.0)) + 0.2
        mu2 = mu1 + 0.05
        i0_1 = 1e5 * np.ones_like(E)
        i0_2 = 0.9e5 * np.ones_like(E)
        if1 = i0_1 * mu1
        if2 = i0_2 * mu2
        s1 = RawScan(energy=E, i0=i0_1, ifluor=if1, mu=mu1,
                     meta={"mode": "fluor"})
        s2 = RawScan(energy=E, i0=i0_2, ifluor=if2, mu=mu2,
                     meta={"mode": "fluor"})
        merged = athena.merge_scans([s1, s2])

        # ① μ 逐位 = 逐扫描 μ 均值（绝不由平均强度重算）
        legacy = np.array([mu1, mu2]).mean(axis=0)
        self.assertTrue(np.array_equal(merged.mu, legacy))
        # ② 荧光通道与 I0 保留（kept 平均），模式保留
        self.assertEqual(merged.meta["mode"], "fluor")
        self.assertIsNotNone(merged.ifluor)
        self.assertTrue(np.allclose(merged.ifluor, (if1 + if2) / 2.0, rtol=1e-12))
        self.assertTrue(np.allclose(merged.i0, (i0_1 + i0_2) / 2.0, rtol=1e-12))
        # ③ 通道溯源
        mc = merged.meta["merge_channels"]
        self.assertTrue(mc["mode_consistent"])
        self.assertIn("ifluor", mc["averaged"])
        self.assertTrue(mc["mu_is_mean_of_per_scan_mu"])
        # ④ dtime 置 None（死时间已在 S1.2 逐扫描完成，防二次校正）
        self.assertIsNone(merged.dtime)
        # ⑤ 合并后仍具荧光通道证据：S4.5 门禁口径 _has_if and not _has_it
        self.assertTrue(merged.ifluor is not None and merged.it is None)

        # ⑥ 单扫描合并：通道逐位等于输入（mean of one）
        one = athena.merge_scans([s1])
        self.assertTrue(np.array_equal(one.ifluor, if1))
        self.assertTrue(np.array_equal(one.i0, i0_1))
        self.assertTrue(np.array_equal(one.mu, mu1))

        # ⑦ 通道不齐：一条缺 If → 不臆造平均，ifluor=None；i0 齐 → 保留
        s3 = RawScan(energy=E, i0=i0_1, mu=mu1, meta={"mode": "fluor"})
        partial = athena.merge_scans([s1, s3])
        self.assertIsNone(partial.ifluor)
        self.assertIsNotNone(partial.i0)
        self.assertNotIn("ifluor", partial.meta["merge_channels"]["averaged"])

        # ⑧ 离群剔除：通道只对 kept 扫描平均（bad 被剔，不进 If 平均）
        bad = RawScan(energy=E, i0=i0_1, ifluor=if1 * 5.0, mu=mu1 + 0.5,
                      meta={"mode": "fluor"})
        rej = athena.merge_scans([s1, s2, bad], reject_outliers=True)
        self.assertEqual(rej.meta["merge_info"]["rejected_indices"], [2])
        self.assertTrue(np.allclose(rej.ifluor, (if1 + if2) / 2.0, rtol=1e-12))

        # ⑨ 模式不一致：保留 mode 但记 warning（合并 μ 物理意义存疑）
        s_trans = RawScan(energy=E, i0=i0_1, it=i0_1 * np.exp(-mu1), mu=mu1,
                          meta={"mode": "trans"})
        het = athena.merge_scans([s1, s_trans])
        self.assertFalse(het.meta["merge_channels"]["mode_consistent"])
        self.assertIn("warning", het.meta["merge_channels"])

    def test_xanes_exafs_cross_check_warning_only(self):
        """#121 T8：白线/边步/归一化异常 → 警告；不改 cn_reliable。"""
        from types import SimpleNamespace

        norm = SimpleNamespace(plateau=0.70, edge_step=0.01, e0=7112.0)
        res = SimpleNamespace(
            norm=norm,
            xanes_features={"e0_jump_risk": True},
            absorber_symbol="Fe",
            fit=None)
        fit_out = {
            "mode": "l1_first_shell", "cn_reliable": True,
            "l1": {"N": 10.0, "R": 2.5},
            "self_absorption": {"risk": "high", "note": "荧光厚样"}}
        wl = {"reliable": True, "white_line_area_norm": 0.2,
              "whiteline_peak_height": 0.1,
              "white_line_energy": 7125.0, "note": ""}
        cc = xafs_fit.xanes_exafs_cross_check(res, fit_out, white_line=wl)
        codes = {w["code"] for w in cc["warnings"]}
        self.assertIn("normalization_plateau_out_of_range", codes)
        self.assertIn("edge_step_weak", codes)
        self.assertIn("e0_jump_risk", codes)
        self.assertIn("high_cn_low_white_line", codes)
        self.assertIn("self_absorption_risk", codes)
        self.assertFalse(cc["affects_cn_reliable"])
        self.assertTrue(fit_out["cn_reliable"])  # 未被修改

        # 干净情形：无警告；L3 边不做 CN–白线极端对比。
        clean_norm = SimpleNamespace(plateau=1.0, edge_step=1.0,
                                     e0=11564.0)
        clean_res = SimpleNamespace(
            norm=clean_norm, xanes_features={"e0_jump_risk": False},
            absorber_symbol="Pt", fit=None)
        clean_fit = {"mode": "l1_first_shell", "cn_reliable": True,
                     "l1": {"N": 10.0, "R": 2.5}}
        clean_wl = {"reliable": True, "white_line_area_norm": 0.2,
                    "whiteline_peak_height": 0.1,
                    "white_line_energy": 11570.0, "note": ""}
        cc2 = xafs_fit.xanes_exafs_cross_check(
            clean_res, clean_fit, white_line=clean_wl)
        self.assertEqual(cc2["warnings"], [])
        self.assertEqual(cc2["status"], "ok")
        self.assertEqual(
            cc2["checks"]["cn_white_line_contrast"]["status"],
            "unavailable")

    def test_p6_2_edge_entry_alignments(self):
        """#130 P6.2：frac:(x,y,z) 吸收位、P1b/P5 配置键镜像、XDI 自动元素回退与 report 异常分支。"""
        import json
        import run_xafs
        from xafs_core import XAFSConfigError

        # 1) --absorber-site "frac:(x,y,z)" 在 phase_resolver -> cif_gate -> _site_motif_from_model 贯通
        pr = phase_resolver.resolve_phase(
            "Ru", phase_id="COD:9007541", absorber_site="frac:(0,0,0)",
            offline=True)
        self.assertEqual(pr.status, "resolved")
        self.assertIsNotNone(pr.structure_model)
        self.assertEqual(pr.structure_model.absorber_site, "frac:(0,0,0)")
        gate = cif_gate.gate_cif(
            pr.selected.cif_path, "Ru",
            absorber_site=pr.absorber_site, kmin=3.0, kmax=12.0)
        self.assertTrue(gate.ok, gate.reason)
        motif = xafs_fit._site_motif_from_model(pr.structure_model)
        self.assertGreaterEqual(len(motif.sites), 2)
        self.assertGreaterEqual(len(motif.shell_expect), 1)

        # 非法 frac:(...) 坐标仍被拒绝
        pr_bad = phase_resolver.resolve_phase(
            "Ru", phase_id="COD:9007541", absorber_site="frac:(a,b,c)",
            offline=True)
        self.assertIsNone(pr_bad.structure_model)
        self.assertIn("非法分数坐标", pr_bad.reason)

        # 2) xafs_config.DEFAULTS 与 apply_config_to_kwargs 覆盖 P1b/P5 开关
        for k in ("s02_from_t5", "xanes_cross_check", "phase_cif",
                  "phase_id", "phase", "phase_index", "phase_pack",
                  "absorber_site", "offline"):
            self.assertIn(k, xafs_config.DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "p62.json"
            cfg_path.write_text(json.dumps({
                "s02_from_t5": True,
                "xanes_cross_check": True,
                "phase_cif": "a.cif,b.cif",
                "phase_id": "COD:9007541",
                "absorber_site": "all",
                "offline": True,
            }), encoding="utf-8")
            loaded = xafs_config.load_config(cfg_path)
            base_kw = {k: v for k, v in xafs_config.DEFAULTS.items()
                       if k not in ("self_absorption", "batch")}
            merged = xafs_config.apply_config_to_kwargs(loaded, base_kw)
        self.assertTrue(merged["s02_from_t5"])
        self.assertTrue(merged["xanes_cross_check"])
        self.assertEqual(merged["phase_cif"], ["a.cif", "b.cif"])
        self.assertEqual(merged["phase_id"], "COD:9007541")
        self.assertEqual(merged["absorber_site"], "all")
        self.assertTrue(merged["offline"])

        # 3) process_sample 在 absorber=None（靠 XDI 文件头自动识别元素）时支持 --formal-cn + --phase-id
        fe_xdi = ROOT / "data" / "bmm_standards" / "Data" / "Fe" / "Fe-Lepidocrocite.xdi"
        with tempfile.TemporaryDirectory() as tmp:
            res = run_xafs.process_sample(
                [str(fe_xdi)], tmp, absorber=None, strict=False,
                identify=True, fit=True, formal_cn=True,
                phase_id="COD:1011240", offline=True, verbose=False)
            self.assertEqual(res.absorber_symbol, "Fe")
            self.assertIsNotNone(res.fit.get("formal_cn"))
            self.assertNotEqual(res.fit["cn_status"], "no_cif")


    def test_unit2_counterexamples_edge_guards(self):
        """Unit 2 边界反例：有限厚度小正值、deglitch sigma_mu、NaN 拦截与 TT 乱序。"""
        import xafs_lcf

        # 1) U2-01: 中等光学厚度下极小正值点不应触发 brentq 括号同号崩溃
        E = np.arange(7000.0, 7200.0, 2.0)
        e0 = 7123.0
        y = np.where(E >= e0, 1.0, 1e-13)
        norm = NormalizedScan(
            energy=E, mu=y, mu_norm=y,
            pre_edge_line=np.zeros_like(E), edge_step=1.0,
            plateau=1.0, e0=e0, flattened=y.copy())
        corr, info = athena.fluo_xanes_self_absorption(
            norm, "Fe2O3", "Fe", "K", "Ka", 20.0, 5.24, 45.0, 20.0)
        self.assertEqual(info["thickness_regime"], "mid")
        self.assertTrue(np.all(np.isfinite(corr.flattened)))

        # 2) U2-03 & U2-04: deglitch 同步裁切 sigma_mu；dtime=None 保留 MCA 已校正标记
        sig = np.full_like(E, 0.01)
        scan = RawScan(
            energy=E, mu=y, sigma_mu=sig, dtime=None,
            meta={"mode": "fluor", "deadtime_corrected": True})
        cleaned, _ = athena.deglitch_ranges(scan, [(7150.0, 7156.0)])
        self.assertEqual(cleaned.sigma_mu.shape, cleaned.energy.shape)
        dt_out = athena.apply_deadtime_correction(cleaned)
        self.assertTrue(dt_out.meta["deadtime_corrected"])

        # 3) U2-06: FT / WT 拦截 NaN 输入，不静默产出假峰
        k = np.arange(0.0, 14.0, 0.05)
        chi_bad = np.sin(2.0 * 1.5 * k)
        chi_bad[50] = np.nan
        with self.assertRaises(XAFSError):
            athena.ft_to_r_space(k, chi_bad, None, 3.0, 12.0, 2)
        ck_bad = ChiK(k=k, chi=chi_bad, background=np.zeros_like(k),
                      mu_norm=np.zeros_like(k), rbkg=1.2)
        with self.assertRaises(XAFSConfigError):
            larch_wt.continuous_wavelet_transform(ck_bad)

        # 4) U2-07: target_transform 对乱序候选谱先排序，与升序结果逐位一致
        x = np.linspace(0.0, 10.0, 200)
        e1 = np.exp(-0.5 * ((x - 3.0) / 0.5) ** 2)
        e2 = np.exp(-0.5 * ((x - 7.0) / 0.5) ** 2)
        specs = [(f"m{i}", x, w * e1 + (1.0 - w) * e2)
                 for i, w in enumerate(np.linspace(0.1, 0.9, 5))]
        pca = xafs_lcf.run_pca(specs, 0.0, 10.0)
        tt_asc = xafs_lcf.target_transform(pca, [("c", x, e1)], n_components=2)
        tt_rev = xafs_lcf.target_transform(
            pca, [("c", x[::-1], e1[::-1])], n_components=2)
        self.assertAlmostEqual(
            tt_asc["c"]["spoils"], tt_rev["c"]["spoils"], places=12)
