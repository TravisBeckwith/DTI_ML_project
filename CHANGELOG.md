# dwiforge — Changelog

---

## v1.4 — 2026-03-23
**Renamed from ML_v9_beta.sh to dwiforge.sh**

### Bug Fixes
- **DEC map eigenvector sign (abs on wrong target)** — `mrcalc -abs` was previously applied directly to `ev.mif`, forcing all x-components positive and making all corpus callosum fibres appear to lean right. Fix: `abs()` is now applied only to a separate `dec.nii.gz` output for visualisation. `ev.nii.gz` retains signed values for tractography and analysis.
- **-strides 1,2,3,4 introduced y-axis gradient flip** — Forcing stride reorientation on NIfTI data with native strides `-1 2 3 4` caused MRtrix3 to misapply its stride-based gradient rotation, producing an "n"-shaped corpus callosum in the DEC map. Confirmed via dtifit on raw BIDS data (correct "u" shape with unmodified bvec). Fix: all `-strides 1,2,3,4` flags removed from `mrconvert -fslgrad` import sites.

### New Outputs
- `sub-<id>_dec.nii.gz` — FA-modulated absolute-value eigenvector map for RGB/DEC visualisation in FSLeyes, ITK-SNAP, and other viewers that do not apply abs() automatically.

### Housekeeping
- Project renamed to **dwiforge** throughout (script, headers, log messages, report titles)
- Version string simplified from `1.4-ml-enhanced` to `1.4`

---

## v1.3 (ML_v9_beta.sh)

### Bug Fixes
- **-modulate fa added to tensor2metric** — Primary eigenvector output now FA-modulated by default, consistent with standard DEC map convention.
- **Tensor2metric vector output** — Added `-vector ev.mif` to primary tensor2metric call; eigenvector was previously not exported in the basic preprocessing stage.

---

## v1.2 (ML_v6.sh) — 12 improvements

### New Command-Line Options
- `--dry-run` — Preview what would be processed without executing anything
- `--resume` — Resume from last successful checkpoint (finer-grained)
- `--container-cmd <cmd>` — Override container runtime (docker/singularity/apptainer)

### Changes
1. **Input validation before processing begins** — `validate_subject_inputs()` and `validate_all_inputs()` run on all subjects before any processing starts. Checks for missing DWI/bval/bvec/T1w files, bval/bvec dimension mismatch, and NIfTI header sanity. Reports all problems up front.
2. **Structured JSONL event log** — `log()` now emits a parallel machine-readable log to `$LOG_DIR/pipeline_events.jsonl`. Each line is a JSON object with timestamp, level, subject, and message. Timing events include `duration_s`.
3. **`--dry-run` mode** — Walks through all subjects and stages, reporting which checkpoints exist and what would execute, without running anything.
4. **Subject lock files** — `acquire_subject_lock()` uses `flock` advisory locks to prevent two pipeline instances from processing the same subject concurrently.
5. **Failure context capture** — `capture_stage_failure()` captures stdout and stderr when a stage fails, saving them to `$LOG_DIR/${sub}_${stage}_failure.log` and showing the last 5 lines in the main log.
6. **Disk space checks between stages** — `check_disk_before_stage()` runs before each major stage. Aborts for a subject if free space drops below threshold (30–50 GB depending on stage).
7. **Per-stage elapsed time logging** — `timed_stage()` wraps every stage call, logging wall-clock duration. Also written to JSONL log with `duration_s`.
8. **Finer-grained checkpoints for `--resume`** — Added intermediate checkpoints within Stage 1: `synb0_complete`, `basic_preproc_complete`, `eddy_complete`.
9. **HTML summary report** — Generates `pipeline_report.html` at completion with summary statistics, per-subject status table, configuration details, and stage timing breakdown.
10. **Config file validation** — `load_config()` runs `bash -n` syntax check before sourcing, then validates known keys for path existence, numeric types, and boolean values.
11. **Unit test harness** — `test_helpers.sh` tests `safe_int`, `create_checkpoint`, `check_checkpoint`, `retry_operation`, and `move_with_verification` in isolation. Run with: `bash test_helpers.sh ./dwiforge.sh`
12. **Singularity/Apptainer container support** — `detect_container_runtime()` tries Docker → Singularity → Apptainer. `run_container()` abstracts bind-mount syntax across runtimes. Singularity builds and caches `.sif` image on first use.

---

## v1.1 (ML_v5.sh) — 16 improvements

### Critical Runtime Fixes
1. **`retry_operation mv` → `rsync`** — Synb0 output move used bare `mv` inside `retry_operation`. Replaced with `rsync -av --remove-source-files` to prevent data loss on partial failure.
2. **`find -o` without grouping** — `find ... -name "*.tmp" -o -name "*.mif" -delete` only deleted `.mif` files due to operator precedence. Fixed with explicit `\( ... \)` grouping.
3. **`cd` without guaranteed return** — Early-return paths left the working directory wrong for subsequent operations. Replaced all with `safe_cd`/`safe_cd_return` (pushd/popd wrappers).
4. **Integer comparison safety** — Added `safe_int()` helper. Wrapped `df`, `free`, `mrstats`, and `nvidia-smi` outputs to strip decimals and handle empty/error strings before arithmetic under `set -e`.
5. **Zero-division in final report** — `$(( n_successful * 100 / n_total ))` crashed if `n_total=0`. Added ternary guard. Also guarded `bc` processing-rate calculation against zero `total_duration`.

### Security / Correctness
6. **Heredoc injection safety** — Four `$PYTHON_EXECUTABLE << EOF` blocks used bash variable expansion inside Python code. Converted all to quoted `<< 'PYEOF'` with environment variable passing.
7. **Consensus mask: majority vote** — `mrcalc ... -mult` computed intersection (AND) of all masks — too conservative. Replaced with addition + threshold at ≥N/2 (majority vote).
8. **`cleanup_aggressive` age guard** — Now only deletes non-final `.mif` files older than 2 hours (`-mmin +120`).

### Maintainability
9. **Dead signal traps removed** — Three trap statements were overwritten by the final trap. Removed the dead ones.
10. **`export -f` block removed** — Removed ~30 exported functions intended for parallel processing that was never implemented and would silently break in subshells.
11. **Externalized Python scripts** — VoxelMorph (440 lines) and NODDI (393 lines) Python scripts available as standalone files for independent linting and editing. Falls back to inline heredocs if absent.
12. **Duplicate tool check consolidated** — `check_required_tools()` and `setup_environment()` no longer duplicate tool checks.
13. **`setup_environment` decomposed** — Extracted ML environment setup into `_setup_ml_environment()`.
14. **Python helper script** — `_py_helper.py` generated at runtime for JSON updates, import checks, and SNR estimation.
15. **Error handling philosophy documented** — Header block documents FATAL vs ADVISORY function contracts.
16. **Docker `--user` warning** — Added comment noting `--user` can cause permission failures with some synb0-disco image versions.
