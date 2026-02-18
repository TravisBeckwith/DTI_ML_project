# DTI_ML_project

Machine learning-enhanced diffusion tensor imaging (DTI) processing pipeline for multimodal neuroimaging research.

---

## Overview

This pipeline integrates traditional DTI processing with ML-based registration techniques to improve accuracy across structural, diffusion, and connectivity analyses. It was developed for use with BIDS-formatted datasets on SLURM-based HPC infrastructure and is optimized for multi-drive storage environments.

**Current version:** `v1.3-ml-enhanced`

---

## Features

- **Full DTI pipeline** — preprocessing, eddy correction, bias correction, tractography, and connectivity analysis
- **ML-enhanced registration** — VoxelMorph, SynthMorph, and ANTs-based registration with automatic fallback to traditional methods
- **NODDI estimation** — neurite orientation dispersion and density imaging parameter fitting
- **SyN B0 distortion correction** — via Synb0-DisCo with traditional fallback
- **Storage-optimized** — intelligent multi-drive (C/E/F) data management with BIDS compliance throughout
- **GPU acceleration** — CUDA/RTX 3070 support via TensorFlow; graceful CPU fallback
- **Robust error handling** — fatal vs. advisory error contracts; checkpoint-based resume
- **Quality control** — per-subject QC reports, JSONL structured logs, and HTML pipeline summary

---

## Pipeline Stages

| Stage | Description | Error Contract |
|-------|-------------|---------------|
| 1 | Basic preprocessing (denoise, gibbs, DWIdenoise) | Fatal |
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
- Python 3.x (virtualenv or conda recommended)
- TensorFlow ≥ 2.x
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- scikit-learn
- nibabel, numpy

### System
- Bash 4+
- CUDA 12.3+ (optional, for GPU acceleration)
- SLURM (optional, for HPC batch submission)

---

## Installation

```bash
git clone https://github.com/travisbeckwith/DTI_ML_project.git
cd DTI_ML_project
chmod +x ML_v6.sh
```

Activate your Python environment before running:

```bash
# conda
conda activate neuroimaging_env

# or virtualenv
source /path/to/venv/bin/activate
```

---

## Usage

```bash
# Basic run — all subjects in BIDS directory
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output

# Single subject
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output --subject sub-001

# ML-enhanced registration
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output --use-ml-registration

# Dry run — preview without executing
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output --dry-run

# Resume from last checkpoint
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output --resume

# Override container runtime
./ML_v6.sh --bids-dir /path/to/bids --output-dir /path/to/output --container-cmd singularity
```

---

## Output Structure

```
derivatives/
├── mrtrix/
│   ├── sub-001_fa.nii.gz          # Fractional anisotropy
│   ├── sub-001_md.nii.gz          # Mean diffusivity
│   ├── sub-001_ndi.nii.gz         # NODDI — neurite density index
│   ├── sub-001_odi.nii.gz         # NODDI — orientation dispersion
│   ├── sub-001_fwf.nii.gz         # NODDI — free water fraction
│   └── sub-001_connectome_*.csv   # Structural connectomes
├── freesurfer/
│   └── sub-001/                   # FreeSurfer recon-all output
└── qc/
    ├── sub-001_qc.txt             # Per-subject QC report
    └── pipeline_final_report.txt  # Pipeline summary report
```

---

## Companion Files

| File | Description |
|------|-------------|
| `ML_v6.sh` | Main pipeline script |
| `voxelmorph_registration.py` | Externalized VoxelMorph registration (co-locate with script) |
| `noddi_fitting.py` | Externalized NODDI fitting (co-locate with script) |
| `test_helpers.sh` | Unit tests for pure-logic helper functions |

---

## Operational Modes

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview execution plan without running |
| `--resume` | Resume from last successful checkpoint |
| `--use-ml-registration` | Enable ML-based registration (VoxelMorph/SynthMorph/ANTs) |
| `--ml-method` | Specify `voxelmorph`, `synthmorph`, `ants`, or `auto` |
| `--container-cmd` | Override container runtime (`docker`, `singularity`, `apptainer`) |

---

## Error Handling

Functions follow one of two contracts:

- **Fatal** — pipeline aborts for the subject on failure (`run_basic_preprocessing`, `run_eddy_and_bias_correction`)
- **Advisory** — failure is logged and processing continues (`run_synb0`, `run_posthoc_refinement`, all ML registration functions)

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
