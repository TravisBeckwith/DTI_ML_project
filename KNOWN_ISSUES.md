# Known Issues — ML DTI Processing Pipeline

**Last updated:** 2026-03-23  
**Pipeline version:** v1.4-ml-enhanced  
**Dataset:** CLS_125

---

## Issue 1 — -strides 1,2,3,4 flag introduced y-axis gradient flip

**Status:** Fixed in pipeline (v1.4)  
**Symptom:** Corpus callosum appeared as an "n" shape instead of the correct "u" shape in the DEC/RGB-FA map.  
**Root cause:** Adding `-strides 1,2,3,4` to `mrconvert -fslgrad` calls was intended to canonicalise the voxel-to-scanner orientation. However, the raw NIfTI for this dataset has native strides of `-1 2 3 4` (x-axis negative). Forcing a reorientation to all-positive strides caused MRtrix3 to apply a compensatory transform to the embedded gradient directions that incorrectly negated the y-axis, introducing the "n" shape in the CC.  
**Confirmed via:** dtifit run directly on the raw BIDS NIfTI and unmodified bvec produced a correct "u"-shaped CC, proving the raw bvec is correct and the pipeline was introducing the error.  
**Fix applied:** All `-strides 1,2,3,4` flags removed from every `mrconvert -fslgrad` import site. Plain `-fslgrad` import used throughout, consistent with dtifit behaviour.  
**Note:** The raw BIDS bvec for this dataset is correct as exported by dcm2niix. No gradient table correction is needed.

---

## Issue 2 — ~~dcm2niix bvec y-axis inversion~~ RETRACTED

**Status:** Retracted — was an incorrect diagnosis  
**Original hypothesis:** dcm2niix exported the bvec y-row with inverted polarity, evidenced by the y-row being almost entirely positive and a high asymmetry norm (0.559) from dirstat.  
**Why it was wrong:** The asymmetry norm is a scalar magnitude and is identical before and after negating any axis — it cannot distinguish a flipped axis from a correctly oriented one. The high norm is a property of this particular 32-direction scheme being unipolar (directions distributed on one hemisphere by design, relying on diffusion symmetry). dtifit confirmed the raw bvec is correct.  
**A y-flip awk command was briefly added to the pipeline and has been fully removed.**

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

- Raw BIDS bvec: correct as exported by dcm2niix — confirmed via dtifit visual inspection
- NIfTI strides: `-1 2 3 4` (x-axis negative; handled correctly by MRtrix3 native -fslgrad without stride forcing)
- Mean FA within mask: 0.30 (expected range 0.25–0.50 for whole-brain masked mean)
- FA > 1 voxels: 0.1% of mask (benign boundary effect)
- TensorFlow duplicate CUDA factory registration warnings at startup: benign, caused by dual registration of cuDNN/cuBLAS plugins; does not affect computation
- dirstat asymmetry norm of 0.559: expected for this unipolar 32-direction scheme; not indicative of a gradient error
