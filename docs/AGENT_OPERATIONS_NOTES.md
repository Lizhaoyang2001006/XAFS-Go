# Some Things Not Noticed

*A practical guide for using an AI agent with this XAFS analysis repository (`xafs-auto-pipeline`).*

> **Core principle:** physical validity over fit residuals — do not relax gates, do not guess inputs, and degrade honestly when results are not reliable.

---

## 1. What the agent should read first

| Priority | Path | Why it matters |
|---|---|---|
| 1 | `README.md` | Project purpose, 3‑minute quick start, module map, and **known limitations** (unavailable standard spectra, Fe `S₀²` context, 12 uncalibrated elements, L-edge `k_max` limits, FEFF/cold-start dependencies). |
| 2 | `AGENTS.md` | **Hard constraints (must follow):** 12 locked elements must remain `s02=None, calibrated=False` (`Cr,Mn,Zn,Mo,Ru,Rh,Pd,Ag,Ir,Au,Ce,U`); Fe oxide production value `0.803` locked; gates/thresholds must not be relaxed (e.g. `R-factor ≤ 0.08`, `|ΔE0| ≤ 15 eV`, normalization plateau `[0.85,1.15]`, confident ID `0.16/0.03`, uncorrected concentrated fluorescence `Δμ ≥ 1.0` → `cn_reliable=False`); `N_idp = 2ΔkΔR/π` (no +2) with `N_var < N_idp`; one global `ΔE0` across shells; failures must downgrade (e.g. to L1), not be masked; preserve CI/regression baselines and non-overwriting output behavior. |
| 3 | `user_guid.md` | Hands-on workflow: data prep, common commands, red/yellow/green interpretation, troubleshooting/decision tree, `S₀²` guidance and calibration notes, and paper-ready reporting (use phase-corrected fit `R`, **not** apparent FT peak positions). |
| 4 | `python run_xafs.py --help` | Authoritative, up-to-date CLI options (absorber/edge, preprocessing, identification, fitting, structure sources/CIF/phase packs, catalytic models, windows/constraints, self-absorption, LCF/batch/operando, figure/diagnostic outputs, etc.). |
| 5 | `docs/CN_REPORTING_STANDARD.md` | What must be disclosed for any CN claim (structure/CIF or `cif=None`, FEFF inputs/SHA256, `S₀²` source, `ΔE0`, k/R windows & kweight, `N_idp`/variables/R-factor, per-shell CN status, systematics). Use `cn_disclosure.py` and/or `--strict-disclosure` as required. |
| 6 | Other `docs/*.md` (as needed) | `CATALYSIS_CAPABILITY_AND_GATES.md`, `CATALYTIC_MOTIF_PIPELINE.md`, `XAFS_SPECIATION_METHODOLOGY.md`, `PHASE_PACK_README.md`, `CIF_CRAWLER_INTEGRATION.md`, `FIGURE_OUTPUTS.md`. |
| 7 | Source entry points (for deeper debugging/extension) | `run_xafs.py`, `xafs_pipeline.py`, `data_audit.py`, `structure_id.py`, `structure_lib.py`, `catalytic_builder.py`, `xafs_fit.py`, `artemis*.py`, `athena*.py`, `xafs_report.py`, `xafs_plots.py`, `xafs_plot_special.py`, `paper_figures.py`, `absorber.py`. |

**Environment notes:** install dependencies (`pip install -r requirements.txt`); FEFF capability (via xraylarch/`feff6l`) and some caches/structure resolution may be required; certain structure retrieval paths may need network access unless run offline with local packs/indexes.

---

## 2. Information to provide the agent for data processing

Providing complete, explicit inputs prevents incorrect assumptions and unnecessary downgrades.

### 2.1 Minimum required
- **Raw spectra files**: path(s) and format (XDI / XDAC / plain-text `.dat`, etc.), including multiple scans if available. Note any special column mapping or multi-channel fluorescence layout.
- **Absorber element and edge**: e.g. `Fe K-edge`, `Pt L3-edge`. If not present in file headers, specify explicitly (`--absorber`, and edge/E0-related options as needed). **Do not ask the pipeline to guess the element.**
- **Output directory**: `-o/--outdir` (existing outputs are backed up to `.backup/<timestamp>/` rather than silently overwritten).

### 2.2 Strongly recommended (often decisive for quantitative reliability)
- **Measurement mode & sample details**: transmission vs fluorescence; composition/loading, thickness, effective density, incident/exit angles, dilution, etc.  
  - Uncorrected concentrated fluorescence (`Δμ ≥ 1.0`) triggers `cn_reliable=False`. Provide self-absorption parameters (`--self-absorption-*`, `--mode ...`) when applicable.
- **`S₀²` strategy** (priority: `--s02` > `--chemical-state` > `--s02-from-t5` > static table in `absorber.py`):  
  - Use calibrated/appropriate values per system (e.g. Fe oxide production value `0.803`; metallic Fe may require `--chemical-state metal`; see `user_guid.md`/`absorber.py`).  
  - **Locked uncalibrated elements** (`Cr,Mn,Zn,Mo,Ru,Rh,Pd,Ag,Ir,Au,Ce,U`, static `s02=None`) **cannot** produce automatic formal CN quantification without an explicit, justified `--s02` / `--s02-from-t5` / `--multi-element-s02`; otherwise expect diagnostic-only/refusal behavior.
- **Structural prior / model source** (critical for S6.5 identification and S7 fitting):  
  - Known phase/CIF: `--phase-cif`, `--phase-id`, `--phase`, `--phase-pack`, `--phase-index`, `--absorber-site`, `--offline`, or `--fit-motif`.  
  - Catalytic/non-ideal local models: `--catalytic-model` (e.g. `sac:...`, `dac:...`, `algo:...`).  
  - If no reliable prior exists, constrain to an L1 first-shell analysis (e.g. limit `--fit-rmax`, set `--l1-ligand`/geometry as appropriate). Do not force multi-shell fits for `ambiguous`/`no_match` cases unless explicitly intended (`--allow-ambiguous`) and risks are understood.
- **Fitting/processing intent**: run stages (`--identify`, `--fit`, `--formal-cn`, `--xanes-cross-check`, `--plot structure|full|paper`, operando/LCF/batch as needed), windows and weights (`--fit-rmin/--fit-rmax`, `--kmin/--kmax/--kweight`/`--fit-kweights`), background/E0/alignment/deglitch options (`--rbkg`, `--e0-method`, `--align`, `--despike`, etc.), and constraints/model toggles (`--constraints`, `--fit-weights`, `--include-ms`, `--vary-c3/--vary-c4`, isotropic/shared settings, etc.). Note: `--fit-rmax` refers to **apparent (phase-uncorrected) R**.
- **Reporting/compliance goal**: whether a **formal CN** table is required, and whether full structure/CIF/FEFF provenance and disclosure checks must pass.

### 2.3 A ready-to-use “agent brief” template
Provide a short, structured note like this:

```text
Data: <file paths / format / scans>
Absorber/edge: <e.g. Pt L3>
Mode & sample: <transmission|fluorescence; composition, thickness/density/angles, loading, etc.>
Goal: <preprocess only | identify | fit (L1/multi-shell) | formal CN | figures/LCF/batch/...>
Structure prior: <none | CIF/phase-id/motif | catalytic-model string | expected shells/distances>
S0² plan: <--s02 value/source | --chemical-state | --s02-from-t5 | accept diagnostic-only>
Windows/constraints: <k range/weight, R range, rbkg/E0, constraints, MS options>
Output: <outdir>
Notes/risks: <self-absorption, short bonds, disorder, known ambiguities>
```

---

## 3. Recommended execution workflow

1. **Sanity-check environment & constraints**: confirm dependencies (and FEFF availability as needed), then read `AGENTS.md` and relevant `README` limitations before running.
2. **Validate inputs**: ensure element/edge, mode, and any required `S₀²`/structure/self-absorption information are explicit. If key prerequisites are missing, state that results may be diagnostic-only or that formal CN must be refused (do not guess or relax gates).
3. **Run the pipeline** (examples):
   - Preprocess + identify + fit:  
     `python run_xafs.py <files> --absorber Fe --identify --fit -o xafs_out/<sample>`
   - With priors/model/calibration: add `--fit-motif ...` / `--phase-cif ...` / `--catalytic-model "..."` / `--s02 ...` / `--chemical-state ...` / `--fit-rmax ...` as appropriate.
   - Multi-element/batch or advanced modes: `--multi-element`, `--batch-dir`, `--operando`, LCF options, etc.
4. **Inspect outputs and gates** (primary evidence):
   - `*_meta.json` / decision trace
   - `*_summary.csv` & `*_indicators.csv` (R-factor, ΔE0, `S₀²` source, self-absorption/phase-correction flags, structure ID summary)
   - `*_fit_table.csv|md|png` (quantitative parameters and uncertainties)
   - `*_structure_main.png`, `*_shell_dissection.png`, `*_candidates_compare.png` (as generated via `--plot structure|full|paper`), `*_structure_id.json`
   - `cn_table.csv` / disclosure artifacts when `--formal-cn` or reporting checks are required (`cn_disclosure.py <fit.json>`, `--strict-disclosure`)
5. **Report with proper downgrades and run post-fit sanity verification**:
   - Inspect **all** 8 post-run verification items documented in `user_guid.md` (`第四步：终局结果八步核验清单`) before presenting conclusions:
     1. Energy-scale / `E0` alignment across standards vs sample (`e0`, `e_half_height`, white line).
     2. High-$k$ fluorescence noise cutoff & `epsilon_r` (avoid fitting noisy `k > 10` tails when `kmax` should be lowered).
     3. WT $k$-peak sanity (`wt_k_sanity` / `shell_dissection` panel c): flag boundary-pinned peaks (`[edge!]`, e.g. `k_peak` stuck at `kmin + sigma`) and never accept a heavy outer shell ($Z \ge 20$) whose `k_peak <= shell1 k_peak + 0.3` as WT-confirmed without investigating disorder/leakage/noise.
     4. Multi-$k$-weight (`k^1 -> k^2 -> k^3`) amplitude trend consistency (outer heavy shells must not shrink monotonically under $k^3$).
     5. High-$\sigma^2$ (> `0.010 Å²`) Ramsauer–Townsend / DW downshift vs default `σ² = 0.003` reference rulers.
     6. Hamilton F-test significance (`outer_shells_f_test_confidence >= 0.95` and `dfd >= 1`) before claiming any outer shell improves the model.
     7. Parameter bounds, correlation (`max|ρ| < 0.90`), `σ² ∈ (0, 0.020]`, global `ΔE0`, and `N_var < N_idp`.
     8. Agreement across `_summary.csv`, `_meta.json`, `_fit_table`, and structure-main panel `(f)` gate card (`cn_reliable`, `cn_claim_allowed`, `diagnostic_only`, `cn_status`). Report phase-corrected fitted `R` (not apparent FT peaks), and never claim a formal CN or confirmed outer shell when gates/F-test/WT sanity fail.

---

## 4. Outputs to review (quick reference)

| Output | What to check |
|---|---|
| `*_structure_main.png` / `*_shell_dissection.png` | XANES/dμ/dE, k-fit & residuals, path-by-path R decomposition, multi-k-weight FT discrimination, phase-corrected true R, and Morlet WT (qualitative attribution). |
| `*_summary.csv` / `*_indicators.csv` | Gate status: R-factor, ΔE0, `S₀²` provenance, plateau/self-absorption flags, key fit metrics. |
| `*_fit_table.csv` / `.md` / `.png` | `N, R, σ²`, uncertainties, and consistency (paper table source; `R` is phase-corrected). |
| `*_structure_id.json` | Candidate ranking/confidence (`confident/partial/ambiguous/no_match`) and implications for fitting scope. |
| `*_meta.json` | Full processing/decision record for auditability. |
| `cn_table.csv` / `*_cn_table.*` / disclosure warnings | Per-shell CN status and completeness of required provenance/disclosure for formal reporting. |

---

## 5. Common pitfalls (worth flagging early)

- **Do not guess** absorber/edge, structure, or `S₀²`; missing priors should lead to constrained/diagnostic analysis, not invented certainty.
- **Locked elements need explicit `S₀²`** for quantitative CN; without it, formal CN should be refused.
- **FT peak positions are apparent distances** (phase shift not corrected) — never report them as bond lengths.
- **`--fit-rmax`/R windows are in apparent R**; short bonds or shell overlap require careful `R_bkg`/window choices.
- **Freedom budget matters:** respect `N_idp = 2ΔkΔR/π` and avoid over-parameterization; high parameter correlation, non-physical `σ²`, boundary hits, or poor R-factor/ΔE0 require downgrade.
- **Fluorescence self-absorption** can bias amplitudes/CN; uncorrected concentrated cases are not CN-reliable.
- **WT is qualitative** (scatterer “fingerprint” aid), not a quantitative coordination proof.

*This note is intended to be packaged with the repository so an agent can start from the right documents, request the right inputs, and produce auditable, gate-compliant results.*