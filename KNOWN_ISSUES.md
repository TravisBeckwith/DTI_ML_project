# Known Issues — ML DTI Processing Pipeline

**Last updated:** 2026-03-23  
**Pipeline version:** v1.4-ml-enhanced  
**Dataset:** CLS_125

---

## Issue 1 — dcm2niix exports bvec with inverted y-axis

**Status:** Fixed in pipeline (v1.4)  
**Affected files:** Raw BIDS bvec (`sub-XXX_dwi.bvec`)  
**Symptom:** Corpus callosum appears as an "n" shape instead of the correct "u" shape in the DEC/RGB-FA map, indicating the anterior-posterior gradient axis is inverted.  
**Root cause:** dcm2niix exports the y-row of the gradient table with inverted polarity for this scanner/sequence. Confirmed by inspecting the raw bvec: the y-row (row 2) is almost entirely positive, which is physically implausible for a well-distributed gradient scheme. The x and z rows correctly contain a mixture of positive and negative values. MRtrix3's stride-based automatic gradient correction does not touch the y-axis in this case because the NIfTI y-stride is already positive (`-1 2 3 4`), so the error passes through uncorrected without an explicit fix.  
**Fix applied:** An `awk` one-liner negates row 2 of the raw bvec immediately before the primary `mrconvert -fslgrad` import call. All downstream bvec exports derive from this corrected MIF, so the fix propagates automatically through the full pipeline without further changes.  
**Note for external users:** The raw BIDS bvecs in this dataset have an inverted y-axis as exported by dcm2niix. Anyone running FSL, MRtrix3, or any other tool directly on the raw BIDS bvecs (outside this pipeline) must apply the same y-axis negation manually before tensor fitting.

---

## Issue 2 — NIfTI x-stride causes non-deterministic gradient rotation at import

**Status:** Fixed in pipeline (v1.4)  
**Symptom:** Potential gradient axis misassignment depending on how dcm2niix stored the NIfTI (stride pattern `-1 2 3 4` confirmed for this dataset).  
**Root cause:** `mrconvert -fslgrad` rotates FSL image-space bvecs into MRtrix3 scanner space using the NIfTI header strides. A negative x-stride means MRtrix3 must negate the x-gradient row. While MRtrix3 handles this correctly in most cases, the behaviour is sensitive to qform/sform integrity and can be non-deterministic across MRtrix3 versions.  
**Fix applied:** `-strides 1,2,3,4` added to all `mrconvert -fslgrad` import calls. This reorients the voxel grid to all-positive strides before gradient embedding, making the vox-to-scanner rotation trivially identity on the sign components and eliminating the ambiguity entirely. Applied at four sites: raw import, VoxelMorph re-import, post-hoc refinement import, and ML refinement re-import.

---

## Issue 3 — abs() applied to ev.mif caused asymmetric DEC map

**Status:** Fixed in pipeline (v1.4)  
**Symptom:** All corpus callosum fibres appeared to lean right; left-hemisphere CC fibres did not mirror right-hemisphere fibres as expected.  
**Root cause:** An initial attempt to fix viewer clipping of negative eigenvector components applied `mrcalc -abs` directly to `ev.mif`. This forced all x-components positive, destroying the meaningful sign that distinguishes left-hemisphere from right-hemisphere fibre orientations.  
**Fix applied:** `abs()` is no longer applied to `ev.mif`. Instead, a separate `dec.nii.gz` is generated as `abs(ev)` strictly for visualisation in viewers that do not apply abs() automatically (FSLeyes, ITK-SNAP). `ev.nii.gz` retains signed values and should be used for tractography and any quantitative analysis. mrview does not require `dec.nii.gz` as it applies abs() internally when loading a vector image as a DEC map.

---

## Issue 4 — FA values exceeding 1.0 at mask boundary

**Status:** Confirmed benign for this dataset  
**Symptom:** `mrstats` reported a maximum FA of 1.22, which exceeds the mathematical bound of 0–1.  
**Root cause:** Noise at the mask boundary pushes the smallest tensor eigenvalue (λ3) below zero in a small number of voxels. MRtrix3's FA formula does not clamp the result, so degenerate tensors at the WM/CSF boundary can yield FA > 1.  
**Extent:** 0.1% of masked voxels (approximately 822 of 822,625). Mean FA within mask was 0.30, consistent with expected values for this brain region and preprocessing stage.  
**Resolution:** No fix required. The affected voxels are at the mask periphery and do not affect WM metrics. If tighter boundary control is needed in future, consider eroding the mask by 1 voxel before tensor fitting.

---

## Verified correct

- Mean FA within mask: 0.30 (expected range 0.25–0.50 for whole-brain masked mean)  
- FA > 1 voxels: 0.1% of mask (benign boundary effect)  
- NIfTI strides: `-1 2 3 4` (x negative, consistent with dcm2niix output for this scanner)  
- TensorFlow duplicate CUDA factory registration warnings at startup: benign, caused by dual registration of cuDNN/cuBLAS plugins; does not affect computation
