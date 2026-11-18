# -*- coding: utf-8 -*-
"""tests/fixtures.py — 共享夹具与常量（自 regression_test.py 纯搬移）。

ROOT 由 tests/ 上跳一级得到仓库根，其余内容与原文件逐行一致。
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np



ROOT = Path(__file__).resolve().parent.parent  # tests/ -> repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import xafs  # noqa: E402,F401

import artemis                        # noqa: E402
import athena                         # noqa: E402
import cn_table                       # noqa: E402
import cn_disclosure                  # noqa: E402
import larch_wt                       # noqa: E402
import motif_provenance               # noqa: E402
import phase_pack                     # noqa: E402
import phase_resolver                 # noqa: E402
import cif_gate                       # noqa: E402
import structure_id                   # noqa: E402
import structure_lib                  # noqa: E402
import structure_model                # noqa: E402
import xafs_config                    # noqa: E402
import xafs_fit                       # noqa: E402
import xafs_io                        # noqa: E402
import xafs_pipeline                  # noqa: E402
import xafs_plot_special              # noqa: E402
import xafs_plots                     # noqa: E402
import xafs_provenance                # noqa: E402
import xafs_report                    # noqa: E402
from absorber import get_absorber      # noqa: E402
from xafs_core import (ChiK, K2, NormalizedScan, RawScan,  # noqa: E402
                       XAFSConfigError, XAFSError)

def _find(pattern: str) -> Path:
    hits = sorted((ROOT / "data").glob(pattern))
    if not hits:
        raise unittest.SkipTest(f"测试数据缺失: data/{pattern}")
    return hits[0]


LEPIDO = _find("*/02_Fe_standards/Fe_lepidocrocite.000")
FE_FOIL = _find("*/01_Fe_foil/fe.060")
GOETHITE = _find("bmm_standards/Data/Fe/Fe-Goethite.xdi")
MAGNETITE = _find("bmm_standards/Data/Fe/Fe-Magnetite.xdi")
FERRIHYDRITE = _find("bmm_standards/Data/Fe/Fe-Ferrihydrite.xdi")
PT_FOIL = _find("bmm_standards/Data/Pt/Pt-L3-PlatinumFoil.xdi")
PT_CL = _find("bmm_standards/Data/Pt/Pt-L3-PlatinumChloride.xdi")
NIO = _find("bmm_standards/Data/Ni/Ni-K-NickelOxide.xdi")


class Fixtures:
    """共享夹具：setUpClass 与原文件里的三个私有辅助方法。"""

    @classmethod
    def setUpClass(cls):
        cls.lep = xafs_pipeline.process_sample(
            [LEPIDO], absorber="Fe", verbose=False)
        cls.lep_fit = xafs_fit.run(
            cls.lep, identify=True, fit=True,
            absorber_symbol="Fe", verbose=False)

    def _fit_frozen_baseline(self, reference, key):
        protocol = reference["protocol"]
        expected = reference[key]
        res = xafs_pipeline.process_sample(
            [ROOT / name for name in expected["scans"]],
            align=expected["align"], absorber=protocol["absorber"],
            kmin=protocol["kmin"], kmax=protocol["kmax"],
            kweight=protocol["kweight"], strict=protocol["strict"],
            e0_method=protocol["e0_method"], verbose=False)
        return xafs_fit.run(res, identify=True, fit=True,
                            absorber_symbol=protocol["absorber"],
                            s02=protocol["s02"], verbose=False)["fit"]

    @staticmethod
    def _disclosure_payload():
        return {
            "path_files": ["feff0001.dat"],
            "cn_status": "diagnostic_only",
            "cn_systematics": {"relative_1sigma": 0.30},
            "fit": {
                "paths": [{"label": "M-O", "N": 4.0, "R": 2.0,
                           "sigma2": 0.003}],
                "s02": 0.8,
                "delta_e0": 1.0,
                "kmin": 3.0,
                "kmax": 12.0,
                "kweights": [2],
                "rmin": 1.0,
                "rmax": 2.2,
                "r_factor": 0.02,
                "n_independent": 12.0,
                "n_variables": 5,
            },
            "structure": {
                "source_type": "constructed_cluster",
                "source_ref": "manual M-O4 cluster",
                "cif": None,
                "cif_declared": True,
                "feff_inputs": ["feff.inp"],
            },
        }

    def _review_cn_fixture(self):
        """Real ledger records for qualification counterexamples, not new spectra."""
        from dataclasses import fields
        from structure_model import StructureModel
        ledger = json.loads((ROOT / "data" / "real_cn_validation.json").read_text(
            encoding="utf-8"))["cases"][0]
        raw_model = ledger["formal_cn"]["model"]
        model = StructureModel(**{f.name: raw_model[f.name]
                                  for f in fields(StructureModel)
                                  if f.name in raw_model})
        rows = ledger["formal_cn"]["rows"]
        plan = {"shells": [{"label": r["shell_label"],
                            "scatterer": r["scatterer"], "reff": r["r_cif"],
                            "degen": r["n0"]} for r in rows],
                "rmin": rows[0]["rmin"], "rmax": rows[0]["rmax"]}
        return ledger, dict(plan=plan, structure_model=model,
                           n0_mapping={"n0_per_path": [r["n0"] for r in rows]})
