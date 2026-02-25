# DTI_ML_project

Machine learning-enhanced diffusion tensor imaging (DTI) processing pipeline for multimodal neuroimaging research.

---

## Overview

This pipeline integrates traditional DTI processing with ML-based registration techniques to improve accuracy across structural, diffusion, and connectivity analyses. It was developed for use with BIDS-formatted datasets and is optimized for multi-drive storage environments.

**Current version:** `v1.3-ml-enhanced` (beta)

### What's new in v7 beta

- **Portable storage model** — hardcoded drive letters replaced with a user-configuration block and CLI flags (`--storage-fast`, `--storage-large`); paths are validated at startup with clear error messages
- **13 bug fixes** — including two undefined functions that caused pipeline aborts, a PYTHONPATH ordering conflict that broke ML dependencies, gradient-file path mismatches, unsafe bash arithmetic, and incomplete container abstraction (see [Changelog](#changelog) for the full list)
- **Full container support** — Synb0-DisCo now runs under Docker, Singularity, or Apptainer via a unified runtime abstraction
- **nibabel compatibility** — all Python QC blocks that previously attempted to load `.mif` files (unsupported by nibabel) now pre-convert to NIfTI

---

## Features

- **Full DTI pipeline** — preprocessing, eddy correction, bias correction, tractography, and connectivity analysis
- **ML-enhanced registration** — VoxelMorph, SynthMorph, and ANTs-based registration with automatic fallback to traditional methods
- **NODDI estimation** — neurite orientation dispersion and density imaging parameter fitting
- **Synb0 distortion correction** — via Synb0-DisCo with traditional fallback
- **Storage-optimized** — configurable multi-drive data management with BIDS compliance throughout
- **GPU acceleration** — CUDA support via TensorFlow; graceful CPU fallback
- **Robust error handling** — fatal vs. advisory error contracts; checkpoint-based resume
- **Quality control** — per-subject QC reports, JSONL structured logs, and HTML pipeline summary

---

## Pipeline Stages

| Stage | Description | Error Contract |
|-------|-------------|---------------|
| 1 | Basic preprocessing (denoise, Gibbs correction, DWI metrics) | Fatal |
| 2 | Eddy current & bias correction | Fatal |
| 3 | Post-hoc refinement & ML registration | Advisory |
| 4 | Connectivity analysis & tractography | Advisory |
| 5 | NODDI parameter estimation | Advisory |

---

## Dependencies

### Neuroimaging Tools
- [FSL](https://fsl.fmrib.ox.ac.uk/) (including `eddy`, `topup`, `bet`)
- [MRtrix3](https://www.mrtrix.org/)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [ANTs](http://stnava.github.io/ANTs/)

### Python / ML
- Python 3.7+ (virtualenv or conda recommended)
- TensorFlow ≥ 2.x
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- scikit-learn, scipy
- nibabel, numpy

### Containers (for Synb0-DisCo)
- Docker, **or**
- Singularity ≥ 3.x, **or**
- Apptainer ≥ 1.x

### System
- Bash 4+
- CUDA 12.3+ (optional, for GPU acceleration)

---

## Installation

```bash
git clone https://github.com/travisbeckwith/DTI_ML_project.git
cd DTI_ML_project
chmod +x ML_v7_beta.sh
```

Activate your Python environment before running:

```bash
# conda
conda activate neuroimaging_env

# or virtualenv
source /path/to/venv/bin/activate
```

---

## Configuration

Before running, set the three required paths. You can do this in one of two ways:

### Option A: Edit the configuration block

Open `ML_v7_beta.sh` and fill in the `USER CONFIGURATION` block near the top:

```bash
USER_BIDS_DIR="/data/study01/BIDS"         # Root of your BIDS dataset
USER_STORAGE_FAST="/scratch/study01"        # Fast SSD for pipeline outputs
USER_STORAGE_LARGE="/archive/study01"       # Large drive for FreeSurfer outputs
```

### Option B: Pass paths on the command line

```bash
./ML_v7_beta.sh \
  --bids /data/study01/BIDS \
  --storage-fast /scratch/study01 \
  --storage-large /archive/study01
```

CLI flags override the configuration block. The pipeline will abort with a clear message if any path is unset.

### Storage layout

| Path | Purpose | Recommended disk | Size estimate |
|------|---------|-----------------|---------------|
| `BIDS_DIR` | Input data + processing intermediates | Fast SSD | ~100 GB free |
| `--storage-fast` | Synb0, MRtrix3, post-hoc, and QC outputs | Fast SSD | ~100 GB free |
| `--storage-large` | FreeSurfer recon-all outputs | Large capacity | ~200–300 GB free |

---

## Usage

```bash
# All subjects
./ML_v7_beta.sh

# Single subject
./ML_v7_beta.sh -s sub-001 --omp-threads 8

# ML-enhanced registration with auto-install
./ML_v7_beta.sh -s sub-001 --use-ml-registration --auto-install-ml

# Specific ML method
./ML_v7_beta.sh --ml-method synthmorph --ml-full-mode

# Skip connectivity analysis
./ML_v7_beta.sh --skip-connectome

# Dry run — preview without executing
./ML_v7_beta.sh --dry-run

# Resume from last checkpoint
./ML_v7_beta.sh --resume

# Override container runtime
./ML_v7_beta.sh --container-cmd singularity
```

---

## CLI Reference

### Basic options

| Flag | Description |
|------|-------------|
| `-b`, `--bids <dir>` | Path to BIDS directory |
| `-s`, `--subject <id>` | Process a single subject |
| `--pe <dir>` | Phase encoding direction: `AP`, `PA`, `LR`, `RL` (default: `AP`) |
| `--echo <val>` | Echo spacing in seconds (default: `0.062`) |
| `--slm-model <model>` | Eddy SLM model: `linear` or `quadratic` (default: `linear`) |
| `--omp-threads <n>` | OpenMP threads (default: auto-detected) |
| `--storage-fast <path>` | Fast SSD for outputs |
| `--storage-large <path>` | Large drive for FreeSurfer |
| `--config <file>` | Load options from a configuration file |

### Processing toggles

| Flag | Description |
|------|-------------|
| `--skip-synb0` | Skip Synb0-DisCo distortion correction |
| `--skip-connectome` | Skip connectivity analysis |
| `--no-cleanup` | Keep temporary files |
| `--dry-run` | Preview execution plan without running |
| `--resume` | Resume from last successful checkpoint |
| `--container-cmd <cmd>` | Override container runtime (`docker`, `singularity`, `apptainer`) |

### ML registration options

| Flag | Description |
|------|-------------|
| `--use-ml-registration` | Enable ML-based registration |
| `--ml-method <method>` | `auto`, `voxelmorph`, `synthmorph`, or `ants` (default: `auto`) |
| `--ml-model-path <path>` | Path to custom ML model weights |
| `--auto-install-ml` | Auto-install missing ML Python packages |
| `--ml-quick-mode` | Fast ML registration (default) |
| `--ml-full-mode` | Full ML registration (slower, more accurate) |
| `--force-gpu` | Force GPU usage |
| `--skip-quality-check` | Skip registration quality assessment |

### Backward compatibility

The legacy flags `--storage-e` and `--storage-f` are still accepted as aliases for `--storage-fast` and `--storage-large`.

---

## Output Structure

```
<storage-fast>/derivatives/
├── synb0-disco/
│   └── sub-001/                       # Synb0 distortion correction outputs
├── mrtrix3/
│   ├── sub-001_fa.nii.gz              # Fractional anisotropy
│   ├── sub-001_md.nii.gz              # Mean diffusivity
│   ├── sub-001_ndi.nii.gz             # NODDI — neurite density index
│   ├── sub-001_odi.nii.gz             # NODDI — orientation dispersion
│   ├── sub-001_fwf.nii.gz             # NODDI — free water fraction
│   └── sub-001_connectome_*.csv       # Structural connectomes
├── posthoc/
│   └── sub-001/                       # Post-hoc refinement outputs
└── qc_integrated/
    ├── sub-001_qc.txt                 # Per-subject QC report
    └── pipeline_final_report.txt      # Pipeline summary

<storage-large>/derivatives/
└── freesurfer/
    └── sub-001/                       # FreeSurfer recon-all output
```

---

## Companion Files

| File | Description |
|------|-------------|
| `ML_v7_beta.sh` | Main pipeline script |
| `voxelmorph_registration.py` | Externalized VoxelMorph registration (co-locate with script) |
| `noddi_fitting.py` | Externalized NODDI fitting (co-locate with script) |
| `test_helpers.sh` | Unit tests for pure-logic helper functions |

---

## Error Handling

Functions follow one of two contracts:

- **Fatal** — pipeline aborts for the subject on failure (`run_basic_preprocessing`, `run_eddy_and_bias_correction`)
- **Advisory** — failure is logged and processing continues (`run_synb0`, `run_posthoc_refinement`, all ML registration functions)

---

## Changelog

### v1.3-ml-enhanced beta (v7)

**Critical fixes**
1. Defined missing `run_synthmorph_t1_dwi_registration()` and `run_enhanced_ants_t1_dwi_registration()` — previously caused "command not found" aborts when `USE_ML_REGISTRATION=true`
2. Fixed PYTHONPATH ordering in `setup_environment()` — FSL packages were prepending over venv, breaking TensorFlow/VoxelMorph imports
3. Fixed gradient file path derivation in `check_connectivity_readiness()` — `_preproc` suffix caused lookup failures

**Moderate fixes**
4. Replaced `((errors++))` with `errors=$((errors + 1))` — safe under `set -e`
5. Changed `return $errors` to `return $(( errors > 0 ? 1 : 0 ))` — prevents silent success when error count wraps past 255
6. Fixed `safe_int()` regex — negative numbers no longer stripped to empty string
7. Fixed `num_dirs` calculation — now counts gradient directions (`head -1 | wc -w`) instead of file lines (always 3)
8. Synb0-DisCo container invocation refactored — supports Docker, Singularity, and Apptainer via `case` dispatch

**Low-severity fixes**
9. Header version string updated to match `SCRIPT_VERSION`
10. `enhance_brain_mask()` Method 3 indentation corrected
11. `cleanup_aggressive()` only runs `docker system prune` when runtime is actually Docker
12. All `nib.load('*.mif')` calls replaced with `mrconvert` → `.nii.gz` pre-conversion (4 locations)
13. Added cleanup of temporary QC `.nii.gz` files

**Portability**
- Replaced all hardcoded paths (`/mnt/c/CLS/...`, `/mnt/e/CLS`, `/mnt/f/CLS`) with user-configurable placeholders
- Added `USER CONFIGURATION` block at top of script
- Renamed `--storage-e` / `--storage-f` to `--storage-fast` / `--storage-large` (old names kept as aliases)
- Genericized all log messages, comments, and reports (no more "C drive", "E drive", "F drive")
- Storage directories are auto-created if they don't exist

---

## Citation

If you use this pipeline in your research, please cite the relevant tools:

- Tournier et al. (2019) MRtrix3. *NeuroImage* 202, 116137
- Jenkinson et al. (2012) FSL. *NeuroImage* 62(2), 782–790
- Balakrishnan et al. (2019) VoxelMorph. *IEEE TMI* 38(8), 1788–1800
- Fischl (2012) FreeSurfer. *NeuroImage* 62(2), 774–781

---

## Author

**Travis Beckwith, Ph.D.**
Neuroimaging Scientist | Bloomington, IN
[travis.beckwith@gmail.com](mailto:travis.beckwith@gmail.com) · [ORCID](https://orcid.org/0000-0001-6128-8464) · [Google Scholar](https://scholar.google.com/citations?user=wolY848AAAAJ&hl=en)
