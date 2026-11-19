# -*- coding: utf-8 -*-
"""Unit 1：IO 与配置底座（文件解析/写盘/配置/输入契约）

纯搬运拆分：测试方法体与原 regression_test.py 逐行一致，断言未做任何修改。
"""
from __future__ import annotations

from tests.fixtures import *  # noqa: F401,F403
from tests.fixtures import Fixtures, _find  # noqa: F401

# 历史测试体里有 3 处直接用 Path(__file__) 定位 data/（原文件在仓库根）。
# 拆到 tests/ 后模块位置变了，把 __file__ 指回等价的根入口文件，
# 保证测试方法体逐字节不变（不改任何断言或路径写法）。
__file__ = str(ROOT / "regression_test.py")


class IOConfigTests(Fixtures):

    def test_reference_transmission_requires_channel_roles(self):
        from types import SimpleNamespace

        path = _find("bmm_standards/Data/Au/Au-L3-GoldFoil.xdi")
        scan = xafs_io.read_scan(path)
        self.assertIs(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=scan))[0], False)
        raw = path.read_text(encoding="utf-8")
        no_roles = "\n".join(line for line in raw.splitlines()
                             if not line.startswith("# Scan.plot_hint:"))
        with tempfile.TemporaryDirectory() as tmp:
            unknown = Path(tmp) / "unknown_roles.xdi"
            unknown.write_text(no_roles, encoding="utf-8")
            ambiguous = xafs_io.read_scan(unknown)
        self.assertIsNone(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=ambiguous))[0])

    def test_xdac_rejects_corrupt_row_and_xdi_sparse_columns(self):
        from xafs_core import XAFSDataError
        import xafs_io

        # 1) read_xdac: corrupt/non-parseable/inconsistent-column/NaN-Inf numeric rows must raise S1 with file+line
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.060"
            base.write_text(
                "XDAC V1.2 Datafile V1\n"
                "Fe foil\n"
                "E0=7112\n"
                "----------------\n"
                "  7000.0  1000.0  500.0\n"
                "  7010.0  1000.0  480.0\n"
                "  7020.0  1000.0  460.0\n",
                encoding="latin-1",
            )
            # bad numeric token
            bad = Path(tmp) / "bad.060"
            bad.write_text(base.read_text(encoding="latin-1").replace("7010.0", "BAD"), encoding="latin-1")
            with self.assertRaises(XAFSDataError) as ctx:
                xafs_io.read_xdac(bad)
            self.assertIn("bad.060", str(ctx.exception))
            # inconsistent columns
            inc = Path(tmp) / "inc.060"
            inc.write_text(
                "XDAC V1.2 Datafile V1\n"
                "Fe foil\n"
                "E0=7112\n"
                "----------------\n"
                "  7000.0  1000.0  500.0\n"
                "  7010.0  1000.0\n",
                encoding="latin-1",
            )
            with self.assertRaises(XAFSDataError):
                xafs_io.read_xdac(inc)
            # NaN/Inf
            nanf = Path(tmp) / "nan.060"
            nanf.write_text(
                "XDAC V1.2 Datafile V1\n"
                "Fe foil\n"
                "E0=7112\n"
                "----------------\n"
                "  7000.0  1000.0  500.0\n"
                "  7010.0  NaN  480.0\n",
                encoding="latin-1",
            )
            with self.assertRaises(XAFSDataError):
                xafs_io.read_xdac(nanf)

            # normal 01_Fe_foil file still readable
            foil_dir = Path(__file__).resolve().parent / "data" / "练手数据" / "01_Fe_foil"
            foil = foil_dir / "fe.060"
            scan = xafs_io.read_xdac(foil)
            self.assertIsNotNone(scan.energy)
            self.assertGreater(len(scan.energy), 10)

        # 2) read_xdi: sparse Column.N positions and xmu+mca mode preservation
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse.xdi"
            sparse.write_text(
                "# XDI/1.0\n"
                "# Column.1: energy eV\n"
                "# Column.3: i0\n"
                "# Column.5: it\n"
                "# Column.7: xmu\n"
                "#-----------------\n"
                "7000  999  100  888  50  777  0.693147\n"
                "7010  999  110  888  55  777  0.693147\n"
                "7020  999  120  888  60  777  0.693147\n",
                encoding="utf-8",
            )
            sc = xafs_io.read_xdi(sparse)
            self.assertEqual(sc.meta.get("mode"), "given")
            self.assertTrue(np.allclose(sc.energy, [7000,7010,7020]))
            self.assertTrue(np.allclose(sc.i0, [100,110,120]))
            self.assertTrue(np.allclose(sc.it, [50,55,60]))
            self.assertTrue(np.allclose(sc.mu, [0.693147,0.693147,0.693147]))

            both = Path(tmp) / "both.xdi"
            both.write_text(
                "# XDI/1.0\n"
                "# Column.1: energy eV\n"
                "# Column.2: i0\n"
                "# Column.3: xmu\n"
                "# Column.4: mca1\n"
                "# Column.5: mca2\n"
                "#-----------------\n"
                "7000  100  0.5  10  20\n"
                "7010  100  0.6  11  21\n"
                "7020  100  0.7  12  22\n",
                encoding="utf-8",
            )
            sb = xafs_io.read_xdi(both)
            self.assertEqual(sb.meta.get("mode"), "given")
            self.assertIn("mca", sb.meta)
            self.assertTrue(np.allclose(sb.mu, [0.5,0.6,0.7]))

    def test_xdi_rejects_corrupt_numeric_rows(self):
        from xafs_core import XAFSDataError

        path = _find("bmm_standards/Data/Ni/Ni-K-NickelHydroxide.xdi")
        lines = path.read_text(encoding="utf-8").splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.strip() and not line.lstrip().startswith("#"))
        original = lines[row_index].split()
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("BAD", "missing", "extra", "NaN"):
                with self.subTest(kind=kind):
                    values = original.copy()
                    if kind == "missing":
                        values.pop()
                    elif kind == "extra":
                        values.append("1")
                    else:
                        values[4] = kind
                    damaged = lines.copy()
                    damaged[row_index] = " ".join(values)
                    target = Path(tmp) / (kind + ".xdi")
                    target.write_text("\n".join(damaged), encoding="utf-8")
                    with self.assertRaises(XAFSDataError) as ctx:
                        xafs_io.read_scan(target)
                    self.assertIn(target.name, str(ctx.exception))
                    self.assertIn(str(row_index + 1), str(ctx.exception))
        valid = xafs_io.read_scan(path)
        self.assertEqual(len(valid), 393)

    def test_given_xmu_transmission_requires_channel_agreement(self):
        from types import SimpleNamespace

        i0 = np.linspace(100.0, 120.0, 20)
        it = i0 * np.exp(-np.linspace(0.3, 1.5, 20))
        mu = np.log(i0 / it)

        def classify(values, incident=i0, transmitted=it, fluor=None,
                     mode="given"):
            merged = SimpleNamespace(
                meta={"mode": mode}, i0=incident, it=transmitted,
                ifluor=fluor, mu=values)
            return xafs_fit.classify_fluorescence_scan(
                SimpleNamespace(merged=merged))[0]

        self.assertIs(classify(np.round(mu, 6)), False)
        self.assertIsNone(classify(mu + 0.01))
        self.assertIsNone(classify(mu, incident=None))
        self.assertIsNone(classify(mu, transmitted=None))
        self.assertIsNone(classify(mu, incident=-i0))
        self.assertIsNone(classify(mu[:-1]))
        self.assertIsNone(classify(np.full_like(mu, np.nan)))
        self.assertIs(classify(mu, mode="fluor"), True)
        self.assertIs(classify(mu, mode="trans"), False)
        self.assertIsNone(classify(mu, incident=None, mode="trans"))
        self.assertIsNone(classify(mu + 0.01, mode="trans"))
        self.assertIs(classify(mu, transmitted=None, fluor=i0 * mu), True)
        self.assertIsNone(classify(mu + 0.01, fluor=i0 * (mu + 0.01)))

        first = SimpleNamespace(meta={"mode": "given"}, i0=i0, it=it, mu=mu)
        second = SimpleNamespace(meta={"mode": "given"}, i0=2 * i0,
                                 it=2 * it * np.exp(-0.2), mu=mu + 0.2)
        merged = SimpleNamespace(meta={"mode": "given"}, i0=1.5 * i0,
                                 it=(it + second.it) / 2, mu=mu + 0.1,
                                 ifluor=None)
        self.assertIs(xafs_fit.classify_fluorescence_scan(SimpleNamespace(
            merged=merged, scans=[first, second]))[0], False)
        first.mu = mu - 0.1
        second.mu = mu + 0.1
        merged.i0, merged.it, merged.mu = i0, it, mu
        self.assertIsNone(xafs_fit.classify_fluorescence_scan(SimpleNamespace(
            merged=merged, scans=[first, second]))[0])

        scan = xafs_io.read_scan(_find(
            "bmm_standards/Data/Ni/Ni-K-NickelHydroxide.xdi"))
        self.assertEqual(scan.meta["mode"], "given")
        self.assertIs(xafs_fit.classify_fluorescence_scan(
            SimpleNamespace(merged=scan))[0], False)

    def test_config_rejects_unknown_suffix(self):
        """不支持的扩展名必须给出干净 ValueError（曾因 p.sffix 拼写崩溃）。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.ini"
            p.write_text("kmin: 3.0\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                xafs_config.load_config(p)
            self.assertIn(".ini", str(ctx.exception))
            self.assertIn(".yaml", str(ctx.exception))

    def test_config_file_covers_all_cli_switches(self):
        """#46 B9-1：17 个 Phase I–III 配置键逐键生效，未配置键保持默认。

        优先级契约：explicit（CLI 显式 dest）> 配置文件 > 代码默认值。
        这里用 JSON（合法 YAML 子集）写全部 17 键，避免依赖 PyYAML；
        同时确认 ``bvs_restraint`` 已进入 DEFAULTS。
        """
        import json

        cfg = {
            "deglitch": [[7000.0, 7010.0]],
            "deadtime": True,
            "reject_outlier_scans": True,
            "fit_kweights": "1,2,3",
            "include_ms": True,
            "vary_c3": "first",
            "vary_c4": "all",
            "isotropic_deltar": True,
            "lcf_standards": "a.xdi,b.xdi",
            "shared_sigma2": "all",
            "mback": True,
            "chi_std": "std.dat",
            "clamp_lo": 0.1,
            "clamp_hi": 0.2,
            "fit_space": "q",
            "target_transform": {"spoil_good": 3.0},
            "mcr_als": {"n_components": 3},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            loaded = xafs_config.load_config(path)
            kwargs = {k: (list(v) if isinstance(v, list) else v)
                      for k, v in xafs_config.DEFAULTS.items()}
            kwargs.pop("self_absorption", None)
            kwargs.pop("batch", None)
            merged = xafs_config.apply_config_to_kwargs(loaded, kwargs)

        self.assertEqual(merged["deglitch"], [(7000.0, 7010.0)])
        for key in ("deadtime", "reject_outlier_scans", "include_ms",
                    "isotropic_deltar", "mback"):
            self.assertIs(merged[key], True, f"{key} 未由配置打开")
        self.assertEqual(merged["fit_kweights"], (1, 2, 3))
        self.assertEqual(merged["vary_c3"], [0])
        self.assertIs(merged["vary_c4"], True)
        self.assertEqual(merged["lcf_standards"], ["a.xdi", "b.xdi"])
        self.assertEqual(merged["shared_sigma2"], "all")
        self.assertEqual(merged["chi_std"], "std.dat")
        self.assertAlmostEqual(merged["clamp_lo"], 0.1)
        self.assertAlmostEqual(merged["clamp_hi"], 0.2)
        self.assertEqual(merged["fit_space"], "q")
        self.assertEqual(merged["target_transform"], {"spoil_good": 3.0})
        self.assertEqual(merged["mcr_als"], {"n_components": 3})
        # 未出现在配置里的键必须保持代码默认值
        self.assertEqual(merged["kmin"], 3.0)
        self.assertEqual(merged["kmax"], 12.0)
        self.assertIs(merged["despike"], xafs_config.DEFAULTS["despike"],
                      "未配置的 despike 不应被配置合并改写")
        self.assertIs(merged["despike"], False)
        self.assertIn("bvs_restraint", xafs_config.DEFAULTS)

        # CLI 显式值优先级最高
        merged2 = xafs_config.apply_config_to_kwargs(
            loaded, dict(kwargs, clamp_lo=0.9), explicit={"clamp_lo"})
        self.assertAlmostEqual(merged2["clamp_lo"], 0.9)

    def test_prj_roundtrip_and_multichannel_mca_merge(self):
        """#50 B9-3：gzip-JSON 工程容器 round-trip 与 MCA 坏道剔除。

        8 通道中第 2、6 道注入 Bragg spike，必须精确剔除且保底 ≥2 道；
        求和谱必须等于干净通道按 info["weights"] 的加权和。
        """
        rng = np.random.default_rng(4)
        energy = np.linspace(7100.0, 7300.0, 201)
        mu = (1.0 + 0.5 / (1.0 + np.exp(-(energy - 7120.0) / 2.0))
              + 0.01 * np.sin(energy / 3.0))
        scan = RawScan(energy=energy, mu=mu, meta={"mode": "trans"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.prj"
            xafs_io.write_prj(
                path, {"scans": [scan], "standards": {},
                       "meta": {"created": "test"}})
            self.assertEqual(path.read_bytes()[:2], b"\x1f\x8b",
                             ".prj 未使用 gzip 容器")
            out = xafs_io.read_prj(path, as_scans=True)
            got = out["scans"][0]
            self.assertTrue(np.array_equal(got.energy, scan.energy))
            self.assertTrue(np.array_equal(got.mu, scan.mu))
            self.assertEqual(out["meta"], {"created": "test"})

            # 兼容未压缩 JSON
            import json
            plain = Path(tmp) / "plain.prj"
            plain.write_text(json.dumps({
                "xafs_prj": 1,
                "scans": [{"energy": energy.tolist(),
                           "mu": mu.tolist(), "meta": {}}],
                "meta": {}}), encoding="utf-8")
            out2 = xafs_io.read_prj(plain, as_scans=True)
            self.assertTrue(np.array_equal(out2["scans"][0].energy, energy))

            # Athena Perl 工程必须干净拒绝
            perl = Path(tmp) / "athena.prj"
            perl.write_text("$VAR1 = { 'x' => 1 };", encoding="utf-8")
            with self.assertRaises(XAFSConfigError):
                xafs_io.read_prj(perl)

        n = 600
        base = (1.0 + 0.6 / (1.0 + np.exp(-(np.arange(n) - 300.0) / 12.0))
                + 0.005 * np.sin(np.arange(n) / 9.0))
        channels = (np.tile(base[:, None], (1, 8))
                    + rng.normal(0.0, 0.002, (n, 8)))
        for idx in (2, 6):
            for j in (120, 121, 400):
                channels[j, idx] += 2.5
        summed, info = athena.merge_mca_channels(
            channels, z_threshold=3.5, min_channels=2)
        self.assertEqual(info["rejected"], [2, 6])
        self.assertEqual(info["n_kept"], 6)
        self.assertGreaterEqual(info["n_kept"], 2)
        clean = [i for i in range(8) if i not in info["rejected"]]
        weights = np.asarray(info["weights"], dtype=float)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        ref = np.sum(channels[:, clean] * weights[None, :], axis=1)
        np.testing.assert_allclose(summed, ref, rtol=1e-12, atol=1e-12)

    def test_read_real_athena_perl_prj_fe_standards(self):
        """#67 B16-1：直接读通仓库自带 Fe_standards.prj。

        - 4 条主扫描（.000/.100/.200 + pyrite），Ref 与 merge 默认过滤；
        - include_derived=True 时 10 条全保留；
        - .100 的 energy/μ 与原始文件 462 点逐点一致；
        - 不含 @x/@y 的任意 Perl 片段仍抛 XAFSConfigError（保护 #50）。
        """
        prj = (ROOT / "data" / "练手数据" / "02_Fe_standards" /
               "Fe_standards.prj")
        out = xafs_io.read_prj(prj, as_scans=True)
        self.assertEqual(out["meta"]["source_format"], "athena_perl")
        self.assertEqual(len(out["scans"]), 4)
        labels = [s.meta["label"] for s in out["scans"]]
        for name in ("Fe_lepidocrocite.000", "Fe_lepidocrocite.100",
                     "Fe_lepidocrocite.200", "Fe_pyrite.000"):
            self.assertIn(name, labels)
        derived = xafs_io.read_prj(
            prj, as_scans=True, include_derived=True)
        self.assertEqual(len(derived["scans"]), 10)

        s_prj = [s for s in out["scans"]
                 if s.meta["label"] == "Fe_lepidocrocite.100"][0]
        s_raw = xafs_io.read_scan(
            ROOT / "data" / "练手数据" / "02_Fe_standards" /
            "Fe_lepidocrocite.100")
        self.assertEqual(len(s_prj.energy), len(s_raw.energy))
        self.assertEqual(len(s_prj.energy), 462)
        self.assertLess(
            float(np.max(np.abs(s_prj.energy - s_raw.energy))), 1e-4)
        self.assertGreater(
            float(np.corrcoef(s_prj.mu, s_raw.mu)[0, 1]), 0.999999)
        self.assertIsNotNone(s_prj.i0)
        self.assertIsNotNone(s_prj.it)

        with tempfile.TemporaryDirectory() as tmp:
            perl = Path(tmp) / "perl_only.prj"
            perl.write_text("$VAR1 = { 'x' => 1 };", encoding="utf-8")
            with self.assertRaises(XAFSConfigError):
                xafs_io.read_prj(perl)
            with self.assertRaises(XAFSConfigError):
                xafs_io.read_athena_perl_prj(perl, as_scans=True)

    def test_counterexample_config_merge_explicit_cli_precedence(self):
        """反例：config 合并不得覆盖 CLI 显式参数（即便其值等于默认）。

        _cli_dests 必须从 argv 识别 --x=y、--x y、短选项 -x；旧逻辑用
        “值==默认”判断会把显式给出的默认值误当未设置而被 config 覆盖。
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--kmax", type=float, default=12.0)
        parser.add_argument("--rbkg", type=float, default=1.0)
        parser.add_argument("-k", "--kweight", type=int, default=2)
        parser.add_argument("--strict", action="store_true", default=True)
        parser.add_argument("--absorber", default=None)
        dests = xafs_provenance._cli_dests(
            parser, ["--kmax=12.0", "--rbkg", "0.8", "-k", "3",
                     "--absorber", "Fe"])
        self.assertEqual(dests, {"kmax", "rbkg", "kweight", "absorber"})
        self.assertNotIn("strict", dests)
        self.assertEqual(xafs_provenance._cli_dests(parser, []), set())
        self.assertEqual(
            xafs_provenance._cli_dests(None, ["--kmax=1"]), set())
        self.assertEqual(
            xafs_provenance._cli_dests(parser, ["file.dat", "-"]), set())

    def test_counterexample_s02_config_below_explicit_cli(self):
        """反例（红线 1）：config 来源 s02 必须低于 CLI 显式 --s02/化学态/T5。

        config:s02 仅作数值兜底（高于静态表），且不算“已标定”；锁定元素
        Zn 在 config:s02 下仍 calibrated=False。宪法解析链不得被 config 破坏。
        """
        import run_xafs
        from types import SimpleNamespace
        # CLI 显式 --s02 最高优先
        args_cli = SimpleNamespace(s02=0.9, _s02_is_config=False,
                                   chemical_state=None, s02_from_t5=False,
                                   absorber="Fe")
        s02, src = run_xafs._resolve_s02_for_paths(args_cli, ["x.dat"])
        self.assertEqual(s02, 0.9)
        self.assertIn("--s02", src)
        # config s02 兜底（无化学态/T5）→ config:s02
        args_cfg = SimpleNamespace(s02=0.85, _s02_is_config=True,
                                   chemical_state=None, s02_from_t5=False,
                                   absorber="Fe")
        s02, src = run_xafs._resolve_s02_for_paths(args_cfg, ["x.dat"])
        self.assertEqual(s02, 0.85)
        self.assertEqual(src, "config:s02")
        # 无覆盖 → (None, None)，沿用静态表
        args_none = SimpleNamespace(s02=None, _s02_is_config=False,
                                    chemical_state=None, s02_from_t5=False,
                                    absorber="Fe")
        s02, src = run_xafs._resolve_s02_for_paths(args_none, ["x.dat"])
        self.assertIsNone(s02)
        self.assertIsNone(src)
        # config:s02 不算已标定；锁定元素 Zn 仍 False
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Zn"), "config:s02"))
        self.assertFalse(get_absorber("Zn").calibrated)
        # 前缀不能冒充元素/实际值绑定的标定证据。
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Zn"), "chemical_state:Fe2+"))
        self.assertFalse(xafs_fit._s02_calibrated_flag(
            get_absorber("Zn"), "s02_calibration:Fe"))

    def test_counterexample_run_manifest_freshness(self):
        """反例：run_manifest 只列本次新写/改写的产物。

        未改写的历史残留（mtime==基线）不得误列；.backup 排除；随后落盘
        的 meta 手动补入；改写旧文件（mtime 前进）必须列入。
        """
        import os
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "old.txt").write_text("stale", encoding="utf-8")
            os.utime(d / "old.txt", (1000, 1000))
            base = xafs_provenance.snapshot_artifacts(d)
            self.assertIn("old.txt", base)
            (d / "new.txt").write_text("fresh", encoding="utf-8")
            (d / "sub").mkdir()
            (d / "sub" / "fig.png").write_text("x", encoding="utf-8")
            (d / ".backup").mkdir()
            (d / ".backup" / "old.txt").write_text("b", encoding="utf-8")
            m = xafs_provenance.build_run_manifest(
                d, base, meta_name="s_meta.json")
            arts = m["artifacts"]
            self.assertNotIn("old.txt", arts)
            self.assertIn("new.txt", arts)
            self.assertIn("sub/fig.png", arts)
            self.assertIn("s_meta.json", arts)
            self.assertFalse(any(".backup" in a for a in arts))
            self.assertIn("generated_at", m)
            # 改写旧文件（mtime 前进）→ 列入本次产物
            os.utime(d / "old.txt", (2000, 2000))
            m2 = xafs_provenance.build_run_manifest(
                d, base, meta_name="s_meta.json")
            self.assertIn("old.txt", m2["artifacts"])
            # 全未改写（基线=当前）→ 仅 meta
            m3 = xafs_provenance.build_run_manifest(
                d, xafs_provenance.snapshot_artifacts(d),
                meta_name="s_meta.json")
            self.assertEqual(m3["artifacts"], ["s_meta.json"])

    def test_counterexample_run_manifest_covers_late_figures(self):
        """反例：run_manifest 必须在结构剖析图件/HTML/摘要表落盘后构建（顺序契约）。

        meta.json 先写盘以保护既有台账，随后 render_plot_suite 生成结构主图、
        壳层鉴别图与摘要表，最后补全 run_manifest。
        """
        import xafs_report
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            xafs_report.main([
                str(LEPIDO), "--absorber", "Fe",
                "--fit-motif", "lepidocrocite",
                "--plot", "full", "--no-strict",
                "-o", str(out)])
            metas = list(out.glob("*_meta.json"))
            self.assertEqual(len(metas), 1, "报告未产出唯一 meta.json")
            meta_path = metas[0]
            stem = meta_path.name[:-len("_meta.json")]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rm = meta.get("run_manifest")
            self.assertIsNotNone(rm, "meta.json 缺 run_manifest")
            arts = rm["artifacts"]
            for expected in (
                f"{stem}_report.html",
                f"{stem}_structure_main.png",
                f"{stem}_shell_dissection.png",
                f"{stem}_fit_table.png",
                f"{stem}_fit_table.md",
                f"{stem}_summary.csv",
                f"{stem}_meta.json",
            ):
                self.assertIn(expected, arts, f"run_manifest 漏列晚写产物 {expected}")
            for a in arts:
                self.assertTrue((out / a).is_file(), f"台账列了不存在的 {a}")
            self.assertFalse(any(".backup" in a for a in arts))
            sc = out / f"{stem}_summary.csv"
            self.assertGreater(
                len(sc.read_text(encoding="utf-8-sig").splitlines()), 5,
                "summary.csv 为空或记录不全")

    def test_counterexample_reference_channel_topology(self):
        """反例（审查 item 1）：参考通道拓扑与独立性必须由采集信息判定。

        历史缺陷：① μ_ref=ln(I0/Ir)=μ_sample+μ_foil，样品边污染参考边；
        ② 独立性靠 |e0_sample−e0_ref|>3 eV 反推（循环论证）。
        正确：串联 I0→样品→It→参考箔→Ir，μ_ref=ln(It/Ir) 隔离箔吸收；
        独立性由通道拓扑（It+Ir 俱在、Ir 非 It 常数比例）判定，不看边位差。
        合成样品边 7125 eV、参考箔边 7112 eV：参考边必须落在 7112 而非 7125。
        """
        from xafs_core import XAFSDataError

        E = np.arange(7000.0, 7300.0, 1.0)
        mu_sample = 0.2 + 1.5 * 0.5 * (1 + np.tanh((E - 7125.0) / 3.0))
        mu_foil = 0.1 + 0.8 * 0.5 * (1 + np.tanh((E - 7112.0) / 3.0))
        i0 = 1e5 * (1 - 1e-4 * (E - 7000.0))
        it = i0 * np.exp(-mu_sample)
        ir = it * np.exp(-mu_foil)

        # ① 串联三 chamber：参考边=箔边 7112，独立（不被样品边 7125 污染）
        scan = RawScan(energy=E, i0=i0, it=it, iref=ir,
                       mu=np.log(i0 / it), meta={"mode": "trans"})
        info = athena.determine_e0(scan, guess=7112.0)
        self.assertAlmostEqual(info["e0_sample"], 7125.0, delta=1.0)
        self.assertAlmostEqual(
            info["e0_reference"], 7112.0, delta=1.0,
            msg="参考边被样品边污染（ln(I0/Ir) 旧缺陷）")
        self.assertTrue(info["reference_independent"])
        self.assertEqual(info["reference_topology"], "series_3ch")
        # 显式对照：旧缺陷 ln(I0/Ir) 会把参考边判到样品边 7125
        e0_contaminated = athena.find_e0_derivative_max(
            E, np.log(i0 / ir), 7082.0, 7162.0)
        self.assertAlmostEqual(e0_contaminated, 7125.0, delta=1.0)

        # ② 只有 I0/Ir 无 It：Ir 是样品透射探测器，非独立参考
        ir_trans = i0 * np.exp(-mu_sample)
        two_ch = RawScan(energy=E, i0=i0, it=None, iref=ir_trans,
                         mu=np.log(i0 / ir_trans), meta={"mode": "trans"})
        info2 = athena.determine_e0(two_ch, guess=7112.0)
        self.assertFalse(info2["reference_independent"])
        self.assertEqual(info2["reference_topology"], "two_chamber_no_it")
        self.assertIsNone(info2["e0_reference"])

        # ③ Ir 与 It 成常数比例（两 chamber 间无吸收箔）：非独立
        degen = RawScan(energy=E, i0=i0, it=it, iref=0.5 * it,
                        mu=np.log(i0 / it), meta={"mode": "trans"})
        info3 = athena.determine_e0(degen, guess=7112.0)
        self.assertFalse(info3["reference_independent"])
        self.assertIsNone(info3["e0_reference"])

        # ④ align_by_reference：能量轴偏移的两扫描经参考对齐后残差→0
        scan_a = RawScan(energy=E + 3.0, i0=i0, it=it, iref=ir,
                         mu=np.log(i0 / it), meta={"mode": "trans"})
        scan_b = RawScan(energy=E - 2.0, i0=i0, it=it, iref=ir,
                         mu=np.log(i0 / it), meta={"mode": "trans"})
        aligned, ainfo = athena.align_by_reference([scan_a, scan_b], 7112.0)
        self.assertTrue(ainfo["ok"])
        self.assertLess(ainfo["residual_std_eV"], 0.2)
        for s in aligned:
            e0s = athena.determine_e0(s, guess=7112.0)["e0_sample"]
            self.assertAlmostEqual(e0s, 7125.0, delta=1.0)

        # ⑤ 非独立参考 → 拒绝对齐（管线据此降级，不硬凑平移量）
        with self.assertRaises(XAFSDataError):
            athena.align_by_reference([two_ch, two_ch], 7112.0)

    def test_counterexample_input_contract_units_axis_dtime(self):
        """反例（审查 item：输入契约）：单位/能量轴单调/积分时间≠死时间。

        历史缺陷：① XDI Column 单位被丢弃，keV 当 eV 读 → 边位错 1000 倍；
        ② 能量轴乱序/递减/重复未校验 → np.interp/FT 静默失真；③
        measurement_time（积分时间）若被当死时间分数 → I/(1−t) 系统性放大。
        正确：keV→eV 换算、未知单位拒绝；严格递增放行、严格递减整体翻转、
        非单调/重复拒绝；积分时间只进 meta 溯源，真死时间列才进 scan.dtime，
        且仅荧光模式生效（透射 dtime 是积分时间，不做死时间校正）。
        """
        from xafs_core import XAFSDataError

        def _write(td, name, header, rows):
            p = Path(td) / name
            p.write_text(header + "\n" + "\n".join(
                "  ".join(f"{v:.6f}" for v in r) for r in rows) + "\n",
                encoding="utf-8")
            return p

        E = np.arange(7000.0, 7300.0, 5.0)
        mu = 0.5 * (1 + np.tanh((E - 7125.0) / 4.0)) + 0.2
        i0 = 1e5 * (1 - 1e-4 * (E - 7000.0))
        it = i0 * np.exp(-mu)
        rows = list(zip(E, i0, it))
        hdr_ev = ("# Column.1: energy eV\n# Column.2: I0 nA\n"
                  "# Column.3: It nA\n# energy I0 It")

        with tempfile.TemporaryDirectory() as td:
            # ① eV 声明：能量逐位不变（基线），factor=1，不翻转
            s_ev = xafs_io.read_scan(_write(td, "ev.xdi", hdr_ev, rows))
            self.assertTrue(np.array_equal(s_ev.energy, E))
            self.assertEqual(s_ev.meta["energy_unit"], "eV")
            self.assertEqual(s_ev.meta["energy_unit_factor"], 1.0)
            self.assertNotIn("energy_axis_reversed", s_ev.meta)

            # ② keV 声明：能量 ×1000 → eV，边位回 7125 而非 7.125；μ 不变
            rows_kev = list(zip(E / 1000.0, i0, it))
            hdr_kev = ("# Column.1: energy keV\n# Column.2: I0 nA\n"
                       "# Column.3: It nA\n# energy I0 It")
            s_kev = xafs_io.read_scan(_write(td, "kev.xdi", hdr_kev, rows_kev))
            self.assertTrue(np.allclose(s_kev.energy, E, rtol=0, atol=1e-6))
            self.assertEqual(s_kev.meta["energy_unit"], "keV")
            self.assertEqual(s_kev.meta["energy_unit_factor"], 1000.0)
            self.assertTrue(np.allclose(s_kev.mu, s_ev.mu, rtol=1e-9, atol=1e-12))

            # ③ 未知单位：拒绝，不得静默假设
            hdr_bad = ("# Column.1: energy MeV\n# Column.2: I0 nA\n"
                       "# Column.3: It nA\n# energy I0 It")
            with self.assertRaises(XAFSDataError):
                xafs_io.read_scan(_write(td, "bad_unit.xdi", hdr_bad, rows))

            # ④ 严格递减：整体翻转，所有通道与递增文件逐位一致
            rows_rev = list(zip(E[::-1], i0[::-1], it[::-1]))
            s_rev = xafs_io.read_scan(_write(td, "rev.xdi", hdr_ev, rows_rev))
            self.assertTrue(np.all(np.diff(s_rev.energy) > 0))
            self.assertTrue(np.array_equal(s_rev.energy, s_ev.energy))
            self.assertTrue(np.array_equal(s_rev.i0, s_ev.i0))
            self.assertTrue(np.array_equal(s_rev.it, s_ev.it))
            self.assertTrue(np.array_equal(s_rev.mu, s_ev.mu))
            self.assertTrue(s_rev.meta["energy_axis_reversed"])

            # ⑤ 重复能量点：拒绝（无法唯一插值）
            rows_dup = list(rows)
            rows_dup[10] = rows_dup[9]
            with self.assertRaises(XAFSDataError):
                xafs_io.read_scan(_write(td, "dup.xdi", hdr_ev, rows_dup))

            # ⑥ 非单调（乱序）：拒绝，不得静默排序掩盖数据问题
            rows_shuf = list(rows)
            rows_shuf[5], rows_shuf[50] = rows_shuf[50], rows_shuf[5]
            with self.assertRaises(XAFSDataError):
                xafs_io.read_scan(_write(td, "shuf.xdi", hdr_ev, rows_shuf))

            # ⑦ 荧光 + measurement_time：积分时间进 meta，绝不进 scan.dtime
            ifl = i0 * mu * 0.1
            mt = np.full_like(E, 0.5)
            rows_fl = list(zip(E, i0, ifl, mt))
            hdr_fl = ("# Column.1: energy eV\n# Column.2: I0 nA\n"
                      "# Column.3: If nA\n# Column.4: measurement_time seconds\n"
                      "# energy I0 If measurement_time")
            s_fl = xafs_io.read_scan(_write(td, "fluor.xdi", hdr_fl, rows_fl))
            self.assertEqual(s_fl.meta["mode"], "fluor")
            self.assertIsNone(s_fl.dtime)
            self.assertAlmostEqual(s_fl.meta["integration_time_s"], 0.5)
            corr = athena.apply_deadtime_correction(s_fl)
            self.assertFalse(corr.meta["deadtime_corrected"])
            self.assertTrue(np.array_equal(corr.ifluor, s_fl.ifluor))

            # ⑧ 真死时间分数列（dtime=0.2）：进 scan.dtime，校正 I/(1−0.2)
            dt = np.full_like(E, 0.2)
            rows_dt = list(zip(E, i0, ifl, dt))
            hdr_dt = ("# Column.1: energy eV\n# Column.2: I0 nA\n"
                      "# Column.3: If nA\n# Column.4: dtime\n"
                      "# energy I0 If dtime")
            s_dt = xafs_io.read_scan(_write(td, "fluor_dt.xdi", hdr_dt, rows_dt))
            self.assertEqual(s_dt.meta["mode"], "fluor")
            self.assertIsNotNone(s_dt.dtime)
            self.assertTrue(np.allclose(s_dt.dtime, 0.2))
            corr_dt = athena.apply_deadtime_correction(s_dt)
            self.assertTrue(corr_dt.meta["deadtime_corrected"])
            self.assertTrue(np.allclose(corr_dt.ifluor, s_dt.ifluor * 1.25,
                                        rtol=1e-9))
            self.assertAlmostEqual(
                corr_dt.meta["deadtime_factor_range"][0], 1.25, places=6)

            # ⑨ 透射谱带 dtime 列：mode 门禁 → scan.dtime=None，不校正
            rows_tr = list(zip(E, i0, it, mt))
            hdr_tr = ("# Column.1: energy eV\n# Column.2: I0 nA\n"
                      "# Column.3: It nA\n# Column.4: dtime\n"
                      "# energy I0 It dtime")
            s_tr = xafs_io.read_scan(_write(td, "trans_dt.xdi", hdr_tr, rows_tr))
            self.assertEqual(s_tr.meta["mode"], "trans")
            self.assertIsNone(s_tr.dtime)
            corr_tr = athena.apply_deadtime_correction(s_tr)
            self.assertFalse(corr_tr.meta["deadtime_corrected"])

    def test_overwrite_protection_applies_in_temp(self):
        """E6：真实 Temp 的 CSV/JSON/PNG/PDF 按显式参数/全局开关防覆盖。"""
        import csv
        import io
        import os
        from unittest import mock
        from matplotlib.figure import Figure
        import xafs_writer

        def write_version(path, version, **kw):
            if path.suffix == ".json":
                payload = {"版本": version}
                expected = json.dumps(payload, indent=2, ensure_ascii=False)
                expected = expected.replace("\n", os.linesep).encode("utf-8")
                result = xafs_writer.safe_write_json(path, payload, **kw)
            elif path.suffix == ".csv":
                rows = [["sample", "Fe-O", version, 2.0, 0.003, 0.0, 0.02]]
                buf = io.StringIO(newline="")
                writer = csv.writer(buf)
                writer.writerow(xafs_writer.FIT_HEADER)
                writer.writerows(rows)
                expected = buf.getvalue().encode("utf-8-sig")
                result = xafs_writer.write_fit_table(
                    rows, path.parent, "sample", **kw)
                self.assertEqual(len(next(csv.reader(io.StringIO(
                    path.read_text(encoding="utf-8-sig"))))), 7)
            else:
                fig = Figure(figsize=(1, 1))
                fig.add_subplot().plot([0, 1], [0, version])
                fig_kw = {"dpi": 60}
                if path.suffix == ".pdf":
                    fig_kw["metadata"] = {"CreationDate": None, "ModDate": None}
                buf = io.BytesIO()
                fig.savefig(buf, format=path.suffix[1:], **fig_kw)
                expected = buf.getvalue()
                result = xafs_writer.safe_savefig(fig, path, **fig_kw, **kw)
            self.assertEqual(result, path)
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(list(path.parent.glob(".*.tmp")))
            return expected

        modes = [("default", {}), ("none", {"protect": None}),
                 ("true", {"protect": True}), ("false", {"protect": False})]
        with tempfile.TemporaryDirectory() as td:
            for name in ("state.json", "sample_fit_table.csv", "plot.png", "plot.pdf"):
                for global_protect in (True, False):
                    for mode, kw in modes:
                        with self.subTest(format=name, global_protect=global_protect,
                                          protect=mode):
                            out = Path(td) / name / str(global_protect) / mode
                            path = out / name
                            with mock.patch.object(xafs_writer, "PROTECT_EXISTING",
                                                   global_protect):
                                first = write_version(path, 1, **kw)
                                self.assertFalse((out / ".backup").exists())
                                second = write_version(path, 2, **kw)
                                self.assertNotEqual(first, second)
                                backups = list((out / ".backup").glob(f"*/{name}"))
                                protected = kw.get("protect")
                                if protected is None:
                                    protected = global_protect
                                self.assertEqual(len(backups), int(protected))
                                if protected:
                                    self.assertEqual(backups[0].read_bytes(), first)
                                third = write_version(path, 3, protect=False)
                                self.assertNotEqual(second, third)
                                self.assertEqual(
                                    list((out / ".backup").glob(f"*/{name}")), backups)
                                if protected:
                                    self.assertEqual(backups[0].read_bytes(), first)

    def test_writer_atomic_failure_preserves_temp_output(self):
        """E6：失败的原子替换仍保留旧字节/备份，并清理临时文件。"""
        from unittest import mock
        import xafs_writer

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            xafs_writer.safe_write_json(path, {"version": 1}, protect=True)
            first = path.read_bytes()
            with mock.patch.object(xafs_writer.os, "replace",
                                   side_effect=OSError("injected replace failure")):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    xafs_writer.safe_write_json(path, {"version": 2}, protect=True)
            self.assertEqual(path.read_bytes(), first)
            backups = list((path.parent / ".backup").glob("*/state.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), first)
            self.assertFalse(list(path.parent.glob(".*.tmp")))

    def test_counterexample_savefig_overwrite_protection(self):
        """反例（审查 item 15）：PNG/PDF 落盘必须防静默覆盖（红线 5）。

        历史缺陷：paper_figures / xafs_report / xafs_plots /
        xafs_plot_special 若直接 fig.savefig 会覆盖旧图，用户既有分析记录
        被静默冲掉。正确：safe_savefig 覆盖前把旧图备份到 .backup/<时间戳>/，
        新图字节不变；临时目录同样保护；protect=False 显式恢复
        旧覆盖语义。完成清单 run_manifest 在全部图件落盘后构建（另测）。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unittest import mock
        import xafs_writer

        def _fig(path, slope, **kw):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([0, 1], [0, slope])
            xafs_writer.safe_savefig(fig, path, dpi=60, **kw)
            plt.close(fig)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            path = out / "cn_table.png"
            # 真实 Temp 路径，不再通过路径 mock 绕过目录豁免
            with mock.patch.object(xafs_writer, "PROTECT_EXISTING", True):
                _fig(path, 1.0)
                first = path.read_bytes()
                # 首次写：无旧文件 → 不产生备份
                self.assertFalse((out / ".backup").exists())
                _fig(path, -1.0)
                second = path.read_bytes()
                self.assertNotEqual(first, second)
                backups = list((out / ".backup").glob(f"*/{path.name}"))
                self.assertEqual(len(backups), 1, "旧图必须备份且仅一份")
                self.assertEqual(backups[0].read_bytes(), first)
                self.assertEqual(path.read_bytes(), second)
                # protect=False：显式覆盖，不再新增备份
                _fig(path, 0.5, protect=False)
                self.assertEqual(
                    len(list((out / ".backup").glob(f"*/{path.name}"))), 1)
                # PDF 同口径：旧 PDF 备份逐字节保留
                pdf = out / "figure.pdf"
                _fig(pdf, 1.0)
                pdf_first = pdf.read_bytes()
                _fig(pdf, -1.0)
                pdf_backups = list((out / ".backup").glob("*/figure.pdf"))
                self.assertEqual(len(pdf_backups), 1)
                self.assertEqual(pdf_backups[0].read_bytes(), pdf_first)
            # 只有显式免责才不备份，Temp 路径本身不是免责依据
            tpath = out / "temp_explicit_overwrite.png"
            _fig(tpath, 1.0)
            tfirst = tpath.read_bytes()
            _fig(tpath, -1.0, protect=False)
            self.assertNotEqual(tpath.read_bytes(), tfirst)
            self.assertEqual(
                len(list((out / ".backup").glob("*/temp_explicit_overwrite.png"))), 0)

    def test_counterexample_unit1_io_config_writer_hardening(self):
        """Unit 1 反例：XDI 非法列声明/荧光零入射光强/空列写出/非法配置/图件原子写。"""
        from unittest import mock
        from matplotlib.figure import Figure
        from xafs_core import XAFSConfigError, XAFSDataError
        import xafs_writer

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)

            # 1) XDI Column.0 或重复 Column.N 必须干净拒绝
            col0 = td / "col0.xdi"
            col0.write_text(
                "# XDI/1.0\n# Column.0: energy eV\n# Column.1: i0\n"
                "# Column.2: it\n7000  100  50\n7010  110  55\n",
                encoding="utf-8",
            )
            with self.assertRaises(XAFSDataError):
                xafs_io.read_xdi(col0)

            dup_col = td / "dup_col.xdi"
            dup_col.write_text(
                "# XDI/1.0\n# Column.1: energy eV\n# Column.2: i0\n"
                "# Column.2: it\n7000  100  50\n7010  110  55\n",
                encoding="utf-8",
            )
            with self.assertRaises(XAFSDataError):
                xafs_io.read_xdi(dup_col)

            # 2) 荧光模式 I0 <= 0 必须抛出 XAFSDataError，防止 inf 穿透 S1
            fl_zero_i0 = td / "fl_zero_i0.xdi"
            fl_zero_i0.write_text(
                "# XDI/1.0\n# Column.1: energy eV\n# Column.2: i0\n"
                "# Column.3: if\n7000  0.0  10.0\n7010  100.0  12.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(XAFSDataError):
                xafs_io.read_xdi(fl_zero_i0)

            # 3) write_columns 空列必须抛 XAFSConfigError 而非裸 IndexError
            with self.assertRaises(XAFSConfigError):
                xafs_io.write_columns(td / "empty.dat", "# header", [], [])

            # 4) load_config 拒绝非字典顶层与拼写错误键
            bad_list_cfg = td / "list.json"
            bad_list_cfg.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                xafs_config.load_config(bad_list_cfg)

            typo_cfg = td / "typo.json"
            typo_cfg.write_text('{"k_max": 14.0}', encoding="utf-8")
            with self.assertWarns(UserWarning) as wctx:
                cfg_loaded = xafs_config.load_config(typo_cfg)
            self.assertIn("k_max", str(wctx.warning))
            self.assertEqual(cfg_loaded.get("_unknown_keys"), ["k_max"])

            # 5) safe_savefig 写盘失败时不得留下损坏的半成品图件，并清理临时文件
            fig = Figure(figsize=(1, 1))
            fig.add_subplot().plot([0, 1], [0, 1])
            img_path = td / "atomic_fail.png"
            with mock.patch.object(fig, "savefig",
                                   side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    xafs_writer.safe_savefig(fig, img_path)
            self.assertFalse(img_path.exists())
            self.assertFalse(list(td.glob(".*.tmp")))

            # 6) 通道与能量轴长度错位必须在 S1 拒绝（U1-06）
            import numpy as np
            from xafs_core import RawScan
            scan = RawScan(energy=np.array([7000.0, 7010.0, 7020.0]),
                           i0=np.array([1.0, 1.0]), it=np.array([0.5, 0.5, 0.4]))
            with self.assertRaises(XAFSDataError):
                xafs_io._order_energy_axis(scan, "bad_len.xdi")

    def test_read_simple_named_beamline_columns_and_prj_eshift_contract(self):
        """常见光束线命名多列文本（No. energy I0 I1 I1/I0 / ln(I0/I1)）直读，与 Athena .prj bkg_eshift 告警/应用契约。"""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            fluo_txt = td / "beamline_fluo.dat"
            fluo_txt.write_text(
                "No.  energy  I0  I1  I1/I0\n"
                "1  11540.0  100000  200000  2.00\n"
                "2  11550.0  100000  210000  2.10\n"
                "3  11570.0  100000  260000  2.60\n",
                encoding="utf-8",
            )
            sc_f = xafs_io.read_scan(fluo_txt)
            self.assertEqual(sc_f.meta.get("format"), "SIMPLE_NAMED")
            self.assertEqual(sc_f.meta.get("mode"), "fluo")
            self.assertTrue(np.allclose(sc_f.energy, [11540.0, 11550.0, 11570.0]))
            self.assertTrue(np.allclose(sc_f.mu, [2.00, 2.10, 2.60]))
            self.assertIsNotNone(sc_f.ifluor)

            trans_txt = td / "beamline_trans.dat"
            trans_txt.write_text(
                "No.  energy  I0  I1  ln(I0/I1)\n"
                "1  11540.0  100000  40000  0.9163\n"
                "2  11570.0  100000  15000  1.8971\n",
                encoding="utf-8",
            )
            sc_t = xafs_io.read_scan(trans_txt)
            self.assertEqual(sc_t.meta.get("mode"), "trans")
            self.assertTrue(np.allclose(sc_t.mu, [0.9163, 1.8971]))

            # Athena .prj bkg_eshift：默认保真并带告警，apply_eshift=True 精确平移能量轴
            prj = (ROOT / "data" / "练手数据" / "02_Fe_standards" / "Fe_standards.prj")
            raw_doc = xafs_io.read_prj(prj, as_scans=True, apply_eshift=False)
            shifted_doc = xafs_io.read_prj(prj, as_scans=True, apply_eshift=True)
            s_raw = [s for s in raw_doc["scans"] if s.meta["label"] == "Fe_lepidocrocite.100"][0]
            s_sh = [s for s in shifted_doc["scans"] if s.meta["label"] == "Fe_lepidocrocite.100"][0]
            self.assertAlmostEqual(float(s_raw.meta["eshift"]), -0.535, places=3)
            self.assertFalse(bool(s_raw.meta["eshift_applied"]))
            self.assertIn("eshift_unapplied_warning", s_raw.meta)
            self.assertTrue(bool(s_sh.meta["eshift_applied"]))
            self.assertNotIn("eshift_unapplied_warning", s_sh.meta)
            self.assertAlmostEqual(float(s_sh.energy[0] - s_raw.energy[0]), -0.535, places=4)

    def test_direct_prj_spec_eshift_hash_and_mode_identity(self):
        from types import SimpleNamespace
        import numpy as np
        import cn_table

        prj = ROOT / "data" / "练手数据" / "02_Fe_standards" / "Fe_standards.prj"
        spec = f"{prj}::Fe_lepidocrocite.100"
        sc_spec = xafs_io.read_scan(spec)
        sc_kw = xafs_io.read_scan(prj, prj_group="Fe_lepidocrocite.100")
        sc_sh = xafs_io.read_scan(spec, apply_eshift=True)

        self.assertEqual(len(sc_spec), len(sc_kw))
        self.assertEqual(sc_spec.meta.get("prj_group"), "Fe_lepidocrocite.100")
        self.assertEqual(sc_spec.meta.get("mode"), "trans")
        self.assertTrue(np.allclose(sc_kw.mu, sc_spec.mu))
        self.assertAlmostEqual(float(sc_sh.energy[0] - sc_spec.energy[0]), -0.535, places=4)
        self.assertEqual(
            xafs_io.scan_spec_label(spec),
            "Fe_lepidocrocite.100",
        )
        self.assertEqual(xafs_io.scan_spec_label(prj), "Fe_standards")
        stem = xafs_io.scan_spec_stem(spec)
        self.assertEqual(stem, "Fe_standards__Fe_lepidocrocite.100")
        self.assertEqual(cn_table.data_sha256([spec]), cn_table.data_sha256([prj]))

        self.assertIs(xafs_io.measurement_identity(sc_spec)[0], False)
        self.assertEqual(
            xafs_io.measurement_identity(SimpleNamespace(meta={"mode": "fluo"})),
            (True, "mode=fluo"),
        )
        self.assertEqual(
            xafs_io.measurement_identity(SimpleNamespace(meta={"mode": "fluor"})),
            (True, "mode=fluor"),
        )
        self.assertIn("prj_group", xafs_config.DEFAULTS)
        self.assertIn("apply_eshift", xafs_config.DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_p = Path(tmp) / "prj_cfg.json"
            cfg_p.write_text(
                '{"prj_group": "Fe_lepidocrocite.100", "apply_eshift": true}',
                encoding="utf-8",
            )
            loaded = xafs_config.load_config(cfg_p)
            self.assertNotIn("_unknown_keys", loaded)
            self.assertEqual(loaded["prj_group"], "Fe_lepidocrocite.100")
            self.assertIs(loaded["apply_eshift"], True)
