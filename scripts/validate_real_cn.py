from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import xafs  # noqa: E402,F401

import calibrate_s02  # noqa: E402
import cif_gate  # noqa: E402
import phase_resolver  # noqa: E402
import run_xafs  # noqa: E402
import xafs_fit  # noqa: E402
import xafs_writer  # noqa: E402
from xafs_core import XAFSError, XAFSConfigError  # noqa: E402

CASES = {
    "ni_oxide_to_hydroxide": {
        "absorber": "Ni", "standard": "Ni-K-NickelOxide.xdi",
        "standard_phase": "COD:1010093",
        "standard_n0": 6.0, "fit_shells": 2, "e0_method": "derivative",
        "holdout": "Ni-K-NickelHydroxide.xdi", "holdout_phase": "COD:1548811",
        "holdout_n0": 6.0,
    },
    "cu_foil_to_oxide": {
        "absorber": "Cu", "standard": "Cu-K-CopperFoil.xdi",
        "standard_phase": "COD:4105040",
        "standard_n0": 12.0, "fit_shells": 1, "e0_method": "half_height",
        "holdout": "Cu-CuO.xdi", "holdout_phase": "COD:1011148",
        "holdout_n0": 4.0,
    },
    "ni_foil_default_protocol": {
        "absorber": "Ni", "standard": "Ni-K-NickelFoil.xdi",
        "standard_phase": "COD:1534892",
        "standard_n0": 12.0, "fit_shells": 1, "e0_method": "derivative",
        "holdout": "Ni-K-NickelHydroxide.xdi", "holdout_phase": "COD:1548811",
        "holdout_n0": 6.0,
    },
    "co_foil_default_protocol": {
        "absorber": "Co", "standard": "Co-K-CobaltFoil.xdi",
        "standard_phase": "COD:9008492",
        "standard_n0": 12.0, "fit_shells": 1, "e0_method": "derivative",
        "holdout": "Co-K-CoO.xdi", "holdout_phase": "COD:1533087",
        "holdout_n0": 6.0,
    },
}


def run_case(name: str, outdir: Path) -> dict:
    """保留已完成阶段的证据；同谱配置错误仍按既有 API 契约抛出。"""
    case = dict(CASES[name])
    evidence = {
        "case": name, "protocol": case,
        "relative_budget": 0.30, "status": "pending",
    }

    def failed(exc, stage):
        status = {"calibration": "calibration_rejected",
                  "prediction": "prediction_rejected"}.get(stage, "validation_error")
        evidence.update(status=status, reason=str(exc), failed_stage=stage)
        if isinstance(exc, XAFSError):
            evidence["error_stage"] = exc.stage
        return evidence

    el = case["absorber"]
    folder = ROOT / "data" / "bmm_standards" / "Data" / el
    standard, holdout = (folder / case[key] for key in ("standard", "holdout"))
    try:
        for key, path in (("standard_data_sha256", standard),
                          ("holdout_data_sha256", holdout)):
            if not path.is_file():
                raise FileNotFoundError(path)
            # Read errors must not produce a digest of missing bytes.
            evidence[key] = hashlib.sha256(path.read_bytes() + b"\0").hexdigest()
    except (XAFSError, OSError) as exc:
        return failed(exc, "input_data")
    standard_sha = evidence["standard_data_sha256"]
    holdout_sha = evidence["holdout_data_sha256"]
    if standard_sha == holdout_sha:
        raise XAFSConfigError("validation", "标定谱与留出谱相同，拒绝循环验证")
    evidence["independent_input_spectra"] = True

    stage = "standard_phase"
    try:
        source = phase_resolver.resolve_phase(el, phase_id=case["standard_phase"],
                                             offline=True)
        if source.selected is not None and source.selected.cif_sha256:
            evidence["standard_cif_sha256"] = source.selected.cif_sha256
        if source.structure_model is None:
            raise XAFSConfigError("validation", source.reason)
        stage = "standard_cif"
        gate = cif_gate.gate_cif(source.selected.cif_path, el,
                                absorber_site=source.absorber_site,
                                kmin=3.0, kmax=12.0, rmax=6.0)
        if not gate.ok or gate.first_shell_cn != case["standard_n0"]:
            raise XAFSConfigError("validation", "标定相 CIF 门禁或预设 N0 不一致")
        source.structure_model.n0_shells = list(gate.n0_shells)
        source.structure_model.occupancy_policy = gate.occupancy_source
        stage = "calibration_motif"
        motif = xafs_fit._site_motif_from_model(source.structure_model)
        stage = "calibration_motif_write"
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        motif_file = xafs_writer.safe_write_json(
            outdir / "calibration_motif.json", motif.to_json())
        evidence["calibration_motif_sha256"] = hashlib.sha256(motif_file.read_bytes()).hexdigest()
        evidence["calibration_geometry"] = "expanded from the verified standard CIF; not a library seed"
        stage = "calibration"
        calibration = calibrate_s02.calibrate_s02(
            [standard], el, motif_file=motif_file,
            fit_shells=case["fit_shells"], e0_method=case["e0_method"],
            allow_recalibrate=True, strict=True, verbose=False)
        calibration["motif_file"] = "calibration_motif.json"
        evidence["calibration"] = calibration
        # Existing calibration_sha256 hashes the sorted JSON payload, not file bytes.
        certificate_sha = hashlib.sha256(json.dumps(
            calibration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        evidence["calibration_sha256"] = certificate_sha
        stage = "calibration_check"
        qualified, reasons = calibrate_s02.certificate_qualifies(calibration)
        evidence["calibration_qualified"] = qualified
        evidence["calibration_reasons"] = reasons
        if calibration["fixed_CN"] != case["standard_n0"]:
            raise XAFSConfigError("validation", "标定 motif 简并度与 CIF N0 不一致")
        if not qualified:
            evidence.update(status="calibration_rejected", failed_stage=stage)
            return evidence
        stage = "calibration_write"
        certificate_file = xafs_writer.safe_write_json(outdir / "calibration.json", {
            **calibration, "standard_sha256": standard_sha,
            "standard_cif_sha256": evidence["standard_cif_sha256"],
            "static_absorber_py_unchanged": True,
        })
        evidence["calibration_file_sha256"] = hashlib.sha256(certificate_file.read_bytes()).hexdigest()
        s02_source = (f"s02_calibration:heldout:{name};"
                      f"standard_sha256:{standard_sha};certificate_sha256:{certificate_sha}")
        stage = "prediction"
        result = run_xafs.process_sample(
            [holdout], outdir / "prediction", absorber=el,
            fit=True, formal_cn=True, phase_id=case["holdout_phase"],
            offline=True, strict=True, s02=calibration["s02"],
            s02_source=s02_source,
            s02_evidence=xafs_fit.make_s02_evidence(calibration), verbose=False)
        stage = "holdout_check"
        fit_out = result.fit
        formal = fit_out["formal_cn"]
        model = dict(formal.get("model") or {})
        for key in ("source_cif", "cif_path"):
            if model.get(key):
                _mp = Path(model[key])
                try:
                    model[key] = _mp.relative_to(ROOT).as_posix()
                except ValueError:
                    model[key] = _mp.as_posix()
        evidence.update(
            plateau=result.norm.plateau,
            cn_reliable=fit_out.get("cn_reliable"),
            conditional_cn=fit_out.get("conditional_cn"),
            formal_cn={**formal, "model": model},
        )
        conditional = evidence["conditional_cn"]
        if conditional is not None:
            if conditional["n0_total"] != case["holdout_n0"]:
                raise XAFSConfigError("validation", "留出相 CIF N0 与预设晶体学真值不一致")
            bias = conditional["n_fit_total"] / case["holdout_n0"] - 1.0
            evidence.update(relative_bias=bias,
                            within_systematic_budget=abs(bias) <= evidence["relative_budget"])
        if formal["status"] != "ok":
            evidence.update(status="formal_rejected", reason=formal.get("reject_reason"))
        elif evidence.get("within_systematic_budget") is not True:
            evidence["status"] = "budget_failed"
        else:
            evidence["status"] = "validated"
    except (XAFSError, OSError) as exc:
        return failed(exc, stage)
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="独立真实标样留出验证，不改静态 S02 或门禁")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = args.case or list(CASES)
    evidence = []
    for name in selected:
        try:
            entry = run_case(name, args.outdir / name)
        except (XAFSError, OSError) as exc:
            entry = {"case": name, "protocol": dict(CASES[name]),
                     "status": "validation_error", "failed_stage": "run_case",
                     "reason": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, XAFSError):
                entry["error_stage"] = exc.stage
        evidence.append(entry)
    report = {
        "schema_version": 1,
        "design": "retrospective independent real-spectrum holdout; not a prospective blinded assay",
        "selection_note": "Single- and two-shell models were examined on training spectra; all holdout results, including refusals, are retained. No holdout-fitted CN is used to calibrate S02.",
        "truth_scope": "ideal crystallographic N0; experimental sample stoichiometry/disorder is not independently certified",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ("numpy", "scipy", "xraylarch", "gemmi")}},
        "cases": evidence,
        "validated_cases": [r["case"] for r in evidence if r["status"] == "validated"],
        "all_cases_validated": all(r["status"] == "validated" for r in evidence),
        "universal_cn_accuracy_established": False,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = xafs_writer.safe_write_json(args.outdir / "real_cn_validation.json", report)
    for entry in evidence:
        print(f"{entry['case']}: {entry['status']}; bias={entry.get('relative_bias')}")
    print(path)
    return 0 if report["all_cases_validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
