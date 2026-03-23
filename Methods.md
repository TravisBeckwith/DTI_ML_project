# Methods

## dwiforge: DWI Processing Pipeline with ML-Enhanced Registration (v1.4)

### 2.1 Pipeline Overview

Diffusion-weighted MRI data were processed using dwiforge (v1.4), an automated, storage-optimized pipeline that integrates conventional neuroimaging preprocessing with optional machine learning (ML)-enhanced registration techniques. The pipeline is implemented as a single Bash script operating on Brain Imaging Data Structure (BIDS)-formatted datasets and orchestrating tools from MRtrix3 [11], FSL [19], FreeSurfer [10], and ANTs [9]. Processing proceeds through four sequential stages: (1) basic DWI preprocessing with optional Synb0-DisCo distortion correction and ML-enhanced motion correction; (2) post-hoc refinement including advanced bias correction, intensity normalization, and enhanced brain masking; (3) structural connectivity analysis via anatomically-constrained tractography; and (4) Neurite Orientation Dispersion and Density Imaging (NODDI) microstructural parameter estimation. Each subject is processed serially with an advisory-lock mechanism to prevent concurrent processing of the same subject. A checkpoint system enables resumption from the last successfully completed stage.

---

### 2.2 Computational Environment and Storage Architecture

The pipeline requires three user-specified storage paths: a BIDS root directory for source data, a fast SSD for pipeline outputs and quality-control reports, and a large-capacity drive for FreeSurfer recon-all outputs. All intermediate processing files are written to a derivatives directory within the BIDS tree, and final outputs are moved to external storage via rsync with verification. Disk space is checked before each major stage (minimum thresholds range from 20–100 GB depending on the stage).

Python execution preferentially uses an active virtual environment or Conda environment (where ML packages reside), falling back to the system Python 3 installation. The pipeline verifies availability of TensorFlow [22], VoxelMorph [7], scikit-learn, SciPy, and NiBabel at startup. GPU configuration targets an NVIDIA RTX 3070 via CUDA 12.3, with TensorFlow GPU memory growth enabled to prevent out-of-memory errors. If no GPU is detected, all ML operations transparently fall back to CPU execution. Thread counts for FSL and MRtrix3 are dynamically optimized based on current system load and are reduced when GPU-accelerated ML registration is active to avoid resource contention.

Container runtime support is provided for Docker, Singularity, and Apptainer, auto-detected at startup. The container abstraction is used primarily for the Synb0-DisCo stage. Signal handlers ensure clean shutdown on interruption, and structured JSONL event logs are maintained alongside human-readable console output.

---

### 2.3 Input Validation

Before processing begins, a pre-flight validation pass runs across all subjects. The pipeline verifies the existence of the 4D DWI NIfTI file, gradient encoding files (bval and bvec), and optionally the T1-weighted anatomical image (required for connectivity analysis). Dimensional consistency is enforced: the number of b-values and b-vector columns must match the fourth dimension of the DWI image. Subjects with validation errors are flagged but processing continues, with failures logged per-stage.

---

### 2.4 Stage 1: Basic DWI Preprocessing

#### 2.4.1 Synb0-DisCo Synthetic Field Map Estimation

Unless explicitly skipped, the pipeline first generates a synthetic undistorted b=0 image using Synb0-DisCo [4] (version 3.1), executed within a Docker or Singularity container. The first b=0 volume is extracted from the DWI series using mrconvert (MRtrix3), paired with the T1-weighted anatomical image, and submitted to the Synb0-DisCo container with the `--notopup` flag (i.e., the synthetic b=0 is used directly rather than being fed into FSL TOPUP). An acquisition-parameters file encoding the phase-encoding direction (default: anterior–posterior) and echo spacing (default: 0.062 s) is generated automatically. Upon completion, outputs are immediately transferred to external fast storage via rsync.

#### 2.4.2 Denoising and Gibbs Ringing Removal

Raw DWI data are converted to MRtrix Image Format (.mif) using mrconvert with embedded FSL gradient information (`-fslgrad`). The gradient table is embedded using the native NIfTI header strides without stride forcing, which ensures MRtrix3's voxel-to-scanner gradient rotation is consistent with the orientation encoded by dcm2niix at conversion. Signal denoising is performed using the Marchenko–Pastur Principal Component Analysis (MP-PCA) method [1] as implemented in MRtrix3's dwidenoise command. A noise map is retained for quality control. Gibbs ringing artifacts are subsequently removed using the local subvoxel-shift approach [2] via mrdegibbs.

#### 2.4.3 Motion and Distortion Correction with ML-Enhanced Registration

The pipeline supports three ML-enhanced registration methods that can optionally replace or augment traditional eddy-current and motion correction. The method is selected automatically or by user override:

**VoxelMorph registration.** When selected, denoised DWI volumes are exported to NIfTI format and registered to the first b=0 volume using a VoxelMorph [7] dense deformable registration network. The network architecture consists of a lightweight U-Net encoder–decoder (feature channels: [8, 16, 16] / [16, 16, 8, 8]) with 5 integration steps and a downsampling factor of 2. The model is compiled with a mean squared error (MSE) image similarity loss and an L2 regularization loss on the displacement field gradient (weight = 0.01), optimized with Adam (learning rate = 1e−4). Each DWI volume is independently registered to the reference b=0 volume. If the full VoxelMorph model fails to initialize (e.g., due to TensorFlow compatibility issues), a simpler fallback 3D convolutional network with 8–16 feature channels is used. In quick mode, a SciPy-based translation-only registration using normalized cross-correlation with L-BFGS-B optimization (bounds: ±10 voxels in-plane, ±5 through-plane) is employed instead.

**SynthMorph registration.** If FreeSurfer 7.3+ is available, SynthMorph [8] is used for T1w-to-DWI registration in post-processing stages rather than DWI volume-to-volume correction, as it was designed for contrast-invariant inter-modal registration.

**Enhanced ANTs registration.** When the ANTs method is selected, antsRegistration [9] is invoked with a multi-stage transform comprising rigid, affine, and SyN (Symmetric Normalization) components. Mutual information (MI) is used as the similarity metric for rigid and affine stages (32 bins, regular sampling at 25%), while cross-correlation (CC, radius = 4) is used for the SyN stage. A four-level multi-resolution pyramid (shrink factors: 8×4×2×1; smoothing sigmas: 3×2×1×0 vox) is applied at each stage, with convergence monitored at 1e−6 tolerance.

If ML registration is not enabled or fails quality checks, the pipeline falls back to the traditional FSL dwifslpreproc workflow [3, 19]. When a synthetic b=0 from Synb0-DisCo is available, dwifslpreproc is run with the `-rpe_pair` option and aligned spin-echo EPI pair for susceptibility-induced distortion correction. Otherwise, motion correction is performed using eddy with the `-rpe_none` option. Eddy options include the user-specified second-level model (default: linear) and the `--repol` flag for outlier replacement.

#### 2.4.4 Registration Quality Assessment

When ML registration is applied, an automated quality assessment computes six metrics between the reference b=0 image and the registered output: Pearson correlation coefficient, normalized cross-correlation (NCC), mean squared error (MSE), mean absolute error (MAE), a simplified structural similarity index (SSIM), and mutual information estimated from a 50-bin joint histogram. An overall quality score is derived from thresholds on correlation (>0.7), NCC (>0.7), SSIM (>0.8), and MI (>0.5). Registrations scoring below 50% are flagged and the pipeline reverts to the traditional method.

#### 2.4.5 Brain Mask Generation and DTI Metric Computation

A brain mask is generated from the preprocessed DWI using dwi2mask (MRtrix3). Bias-field correction is then performed using the N4 algorithm [5] via MRtrix3's dwibiascorrect (with the `ants` backend). The diffusion tensor is estimated using dwi2tensor, and scalar maps — fractional anisotropy (FA), mean diffusivity (MD), axial diffusivity (AD), and radial diffusivity (RD) — are extracted with tensor2metric. The FA-modulated primary eigenvector is additionally computed using the `-vector` and `-modulate fa` options, which scales each eigenvector's magnitude by the local FA value.

Two eigenvector-derived outputs are saved. The signed eigenvector map (`_ev.nii.gz`) retains the directional sign as output by the tensor solver, which is meaningful for downstream tractography and quantitative analyses. A separate directionally-encoded colour (DEC) map (`_dec.nii.gz`) is generated by taking the absolute value of the eigenvector components, producing a map suitable for RGB visualisation in viewers that do not apply absolute-value scaling internally (e.g., FSLeyes, ITK-SNAP). All outputs are exported to NIfTI format and transferred to external storage with file-level verification.

---

### 2.5 Stage 2: Post-Hoc Refinement

#### 2.5.1 Advanced Bias Field Correction and Intensity Normalization

The preprocessed DWI data from Stage 1 are subjected to a second round of refined bias correction using dwibiascorrect with enhanced N4 parameters (b-spline fitting distance: 200; convergence: 50×50×30 iterations at 1e−6 tolerance; shrink factor: 4). If the advanced correction fails, a standard N4 correction is attempted. Individual-level intensity normalization is then applied via dwinormalise (MRtrix3) to harmonize signal intensity across the brain volume.

#### 2.5.2 Enhanced Brain Masking

An enhanced brain mask is constructed by combining up to four independent masking strategies via majority-vote consensus. The strategies include: (1) a DWI-based mask from dwi2mask; (2) an FA-based mask generated by thresholding at FA > 0.1 with morphological dilation and erosion; (3) a BET-based mask [6] derived from the mean b=0 image with a fractional intensity threshold of 0.3; and (4) an ML-enhanced mask using K-means clustering and connected-component analysis with morphological smoothing (opening and closing with a 3D structuring element). The consensus mask is obtained by summing all individual masks and thresholding at a majority vote (≥ ceiling of N/2). The resulting mask undergoes a dilate–erode–dilate morphological cleaning sequence and is validated by checking that the mask volume exceeds 1,000 voxels.

#### 2.5.3 ML Registration Refinement and Residual Distortion Analysis

If ML registration is enabled and dependencies are satisfied, an additional VoxelMorph-based refinement pass is applied to the normalized DWI data. Residual distortion analysis is performed when a T1w image is available, using one of three registration methods (selected in priority order): SynthMorph [8] (mri_synthmorph with the `-m rigid` flag), enhanced ANTs registration, or boundary-based registration (BBR) [20] via FSL's epi_reg. Registration matrices are validated by checking that the rotation component has a determinant between 0.8 and 1.2 and that translation magnitudes do not exceed 1,000 mm. A residual distortion map is computed as the absolute difference between the registered T1w and mean b=0 images, and comprehensive statistics (mean, standard deviation, range, Dice coefficient, Jaccard index, mutual information, and correlation coefficient) are reported.

---

### 2.6 Stage 3: Structural Connectivity Analysis

#### 2.6.1 FreeSurfer Cortical Reconstruction

T1-weighted images are processed with FreeSurfer's recon-all [10] using parallel execution (`-parallel -openmp N`). When ML registration is enabled, the T1w image is first enhanced with N4 bias correction [5] followed by ML-based intensity normalization, which rescales intensity such that the estimated brain tissue peak aligns to a target intensity of 110. Thread counts are dynamically adjusted based on available memory and GPU activity. Up to two recon-all attempts are permitted; if the first fails, threads are reduced and processing retries. Required outputs (aparc+aseg.mgz, pial and white surfaces, recon-all.done flag) are verified, and the complete FreeSurfer directory (~5 GB) is transferred to large-capacity external storage.

#### 2.6.2 Tissue Segmentation and Parcellation

Five-tissue-type (5TT) segmentation images are generated from the FreeSurfer outputs using 5ttgen (MRtrix3) with the `freesurfer` algorithm [14]. Cortical parcellation labels are converted from the FreeSurfer aparc+aseg volume to a MRtrix3-compatible node image using labelconvert with the Desikan–Killiany atlas [16] lookup table (fs_default.txt). The grey matter–white matter interface is extracted using 5tt2gmwmi to generate a seeding mask for anatomically-constrained tractography.

#### 2.6.3 Response Function Estimation and Fiber Orientation Distributions

An automated b-value analysis determines the optimal response estimation strategy. For multi-shell acquisitions, the Dhollander algorithm [13] estimates white matter, grey matter, and CSF response functions, followed by multi-tissue constrained spherical deconvolution (MSMT-CSD) [21] via dwi2fod. For single-shell acquisitions, the Tournier algorithm [12] estimates a single white matter response function, followed by single-tissue CSD. The maximum spherical harmonic order (lmax) is set adaptively: lmax = 8 for acquisitions with ≥45 gradient directions, and lmax = 6 otherwise. Response functions are validated by checking tissue-specific anisotropy properties (e.g., high FA for white matter, low FA for grey matter and CSF). FOD quality is validated by confirming non-empty, reasonably-valued amplitude distributions within the brain mask.

#### 2.6.4 Anatomically-Constrained Tractography

Whole-brain probabilistic tractography is performed using MRtrix3's tckgen with the iFOD2 algorithm [14]. Streamlines are seeded from the grey matter–white matter interface mask and constrained by the 5TT segmentation image (`-act` option with backtracking enabled). Default parameters include a target of 10 million streamlines, minimum/maximum streamline lengths of 10/250 mm, and an angular threshold of 45°. Tractography parameters are adjusted based on data quality scores: high-quality data use a tighter angular threshold (35°) and longer minimum length (15 mm), while lower-quality data use a reduced target (5 million) and more permissive angular threshold (50°). Up to three attempts are made with progressively relaxed parameters if insufficient streamlines are generated.

#### 2.6.5 Track Filtering and Connectome Construction

Streamlines are filtered using SIFT2 [15] (tcksift2), which assigns per-streamline weights to improve biological plausibility. Structural connectomes are constructed using tck2connectome with the Desikan–Killiany parcellation, producing three connectivity matrices: (1) a streamline-count-weighted matrix, (2) an FA-weighted matrix (mean FA along each connection), and (3) a streamline-length-weighted matrix (mean length of connecting streamlines). If SIFT2 weights are available, a fourth SIFT2-weighted connectome is also generated. All connectomes are saved as CSV files.

---

### 2.7 Stage 4: NODDI Microstructural Parameter Estimation

Neurite Orientation Dispersion and Density Imaging (NODDI) [17] parameters are estimated using the Accelerated Microstructure Imaging via Convex Optimization (AMICO) framework [18]. The pipeline preferentially uses post-hoc-refined DWI data from Stage 2 when available, falling back to Stage 1 preprocessed data otherwise. DWI data are loaded via AMICO's acquisition scheme interface, which reads bval/bvec gradient files directly. The NODDI model is initialized and fit using AMICO's default optimization parameters. Three parameter maps are extracted: the neurite density index (NDI), the orientation dispersion index (ODI), and the free-water fraction (FWF, corresponding to the isotropic volume fraction). All maps are saved as NIfTI files and transferred to external storage.

Comprehensive quality validation is performed on the NODDI outputs, including: (a) parameter range validation (checking that NDI, ODI, and FWF fall within the [0, 1] interval, with flagging when <85% of voxels are valid); (b) model fit quality assessment, which identifies suspicious voxel patterns such as simultaneous extreme values of NDI and ODI; (c) acquisition adequacy analysis, which verifies sufficient b=0 volumes (≥1), low-b volumes between 800–1200 s/mm² (≥15), and high-b volumes ≥1800 s/mm² (≥30); and (d) an overall quality score on a 100-point scale derived from parameter validity, acquisition adequacy, and fit quality sub-scores.

---

### 2.8 Quality Control and Reporting

Quality control reports are generated at each processing stage. Stage 1 reports include DTI metric statistics (mean and standard deviation of FA, MD, AD, and RD within the brain mask), signal-to-noise ratio (SNR) estimates computed from b=0 volumes, and FA mosaic images (when FSL's slicer is available). Stage 2 reports add comprehensive quality scores on a 100-point composite scale incorporating FA range reasonableness, mask coverage ratio, registration quality, and processing completeness. Stage 3 reports include tractography quality metrics (streamline count, mean length, SIFT2 application status) and connectome statistics (number of nodes, edges, network density, and connection strength distributions). Stage 4 reports provide detailed NODDI parameter statistics and clinical interpretation guidance.

An HTML summary report is generated at pipeline completion, presenting per-subject pass/fail status for all four stages, configuration parameters, stage timing extracted from the JSONL event log, and output file locations. A machine-readable JSONL event log records timestamped events with subject identifiers and log levels throughout execution.

---

### 2.9 Error Handling and Operational Modes

Pipeline stages follow one of two error-handling contracts. Fatal stages (basic preprocessing, eddy/bias correction) cause the pipeline to abort processing for the affected subject upon failure. Advisory stages (Synb0-DisCo, post-hoc refinement, connectivity analysis, NODDI estimation, and all ML registration functions) log failures but allow processing to continue, either by falling back to traditional methods or by skipping the failed stage. A retry mechanism with exponential backoff (up to 3 attempts, starting at 5-second delays) is applied to file-transfer operations.

The pipeline supports three operational modes: standard execution, dry-run mode (which previews what would run for each subject without executing any processing), and resume mode (which restarts from the last successful checkpoint). Checkpoint files are written after each major stage, enabling fine-grained resumption after interruptions.

---

### 2.10 Software Dependencies

The pipeline integrates the following software packages: MRtrix3 [11] for DWI preprocessing, tensor estimation, tractography, and connectome construction; FSL [19] for eddy-current correction, brain extraction (BET), tissue segmentation (FAST), and boundary-based registration (epi_reg); FreeSurfer [10] for cortical reconstruction (recon-all) and SynthMorph registration; ANTs [9] for N4 bias correction and enhanced deformable registration; TensorFlow [22] and VoxelMorph [7] for ML-based image registration; AMICO [18] for NODDI parameter estimation; and the Synb0-DisCo container [4] for synthetic field map generation. Python dependencies include NumPy, SciPy, NiBabel, and scikit-learn.

---

## References

1. Veraart, J., Novikov, D. S., Christiaens, D., Ades-Aron, B., Sijbers, J., & Fieremans, E. (2016). Denoising of diffusion MRI using random matrix theory. *NeuroImage*, 142, 394–406.

2. Kellner, E., Dhital, B., Kiselev, V. G., & Reisert, M. (2016). Gibbs-ringing artifact removal based on local subvoxel-shifts. *Magnetic Resonance in Medicine*, 76(5), 1574–1581.

3. Andersson, J. L. R., & Sotiropoulos, S. N. (2016). An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. *NeuroImage*, 125, 1063–1078.

4. Schilling, K. G., Blaber, J., Huo, Y., Newton, A., Hansen, C., Nath, V., ... & Landman, B. A. (2020). Synthesized b0 for diffusion distortion correction (Synb0-DisCo). *Magnetic Resonance Imaging*, 64, 62–70.

5. Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A., Yushkevich, P. A., & Gee, J. C. (2010). N4ITK: Improved N3 bias correction. *IEEE Transactions on Medical Imaging*, 29(6), 1310–1320.

6. Smith, S. M. (2002). Fast robust automated brain extraction. *Human Brain Mapping*, 17(3), 143–155.

7. Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019). VoxelMorph: A learning framework for deformable medical image registration. *IEEE Transactions on Medical Imaging*, 38(8), 1788–1800.

8. Hoffmann, M., Billot, B., Greve, D. N., Iglesias, J. E., Fischl, B., & Dalca, A. V. (2022). SynthMorph: Learning contrast-invariant registration without acquired images. *IEEE Transactions on Medical Imaging*, 41(3), 543–558.

9. Avants, B. B., Tustison, N. J., Song, G., Cook, P. A., Klein, A., & Gee, J. C. (2011). A reproducible evaluation of ANTs similarity metric performance in brain image registration. *NeuroImage*, 54(3), 2033–2044.

10. Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774–781.

11. Tournier, J.-D., Smith, R. E., Raffelt, D., Tabbara, R., Dhollander, T., Pietsch, M., ... & Connelly, A. (2019). MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. *NeuroImage*, 202, 116137.

12. Tournier, J.-D., Calamante, F., & Connelly, A. (2007). Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution. *NeuroImage*, 35(4), 1459–1472.

13. Dhollander, T., Mito, R., Raffelt, D., & Connelly, A. (2019). Improved white matter response function estimation for 3-tissue constrained spherical deconvolution. *Proc. Intl. Soc. Mag. Reson. Med.*, 27, 555.

14. Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A. (2012). Anatomically-constrained tractography: Improved diffusion MRI streamlines tractography through effective use of anatomical information. *NeuroImage*, 62(3), 1924–1938.

15. Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A. (2015). SIFT2: Enabling dense quantitative assessment of brain white matter connectivity using streamlines tractography. *NeuroImage*, 119, 338–351.

16. Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B. C., Blacker, D., ... & Killiany, R. J. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. *NeuroImage*, 31(3), 968–980.

17. Zhang, H., Schneider, T., Wheeler-Kingshott, C. A., & Alexander, D. C. (2012). NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain. *NeuroImage*, 61(4), 1000–1016.

18. Daducci, A., Canales-Rodríguez, E. J., Zhang, H., Dyrby, T. B., Alexander, D. C., & Thiran, J.-P. (2015). Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data. *NeuroImage*, 105, 32–44.

19. Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. *NeuroImage*, 62(2), 782–790.

20. Greve, D. N., & Fischl, B. (2009). Accurate and robust brain image alignment using boundary-based registration. *NeuroImage*, 48(1), 63–72.

21. Jeurissen, B., Tournier, J.-D., Dhollander, T., Connelly, A., & Sijbers, J. (2014). Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data. *NeuroImage*, 103, 411–426.

22. Abadi, M., et al. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.
