# Known Issues — dwiforge

**Last updated:** 2026-03-29  
**Pipeline version:** v1.4  
**Dataset:** CLS_125

---

## Issue 1 — Eigenvector/DEC map orientation (resolved after extended investigation)

**Status:** Fixed in pipeline (v1.4)  
**Symptom:** The corpus callosum appeared as an "n" shape instead of the correct "u" shape in the DEC/RGB-FA map, and/or both hemispheres of the CC appeared to lean in the same direction.

**Root cause (confirmed):** MRtrix3's `dwi2tensor` and `tensor2metric` produce eigenvectors in scanner/RAS space. When exported to NIfTI with native strides `-1 2 3`, the x-component sign is not adjusted for the negative x-stride, so viewers read the left-hemisphere x-component inverted. Multiple approaches to correct this in MRtrix space failed due to coordinate frame ambiguity between image space and scanner space.

**Fix applied:** The eigenvector and DEC map are now produced using FSL's `dtifit`, which works entirely in image space and handles the coordinate frame correctly:
1. Export `dwi_biascorr.mif` to NIfTI
2. Export the eddy-rotated bvec from the MIF using `mrinfo -export_grad_fsl`
3. Negate the y-row of the bvec with awk (confirmed correct via exhaustive testing of all axis combinations)
4. Run `dtifit --wls --no_tensor` to produce `V1`
5. Modulate V1 by FA with `fslmaths FA -mul V1 dec`
6. Load `dec.nii.gz -ot rgb` in FSLeyes — **do not apply abs() before loading**, FSLeyes applies it internally and pre-applying abs() breaks left-hemisphere display

Scalar metrics (FA, MD, AD, RD) are still computed via `dwi2tensor` + `tensor2metric` as these are rotationally invariant and unaffected by axis sign errors.

**Approaches tested and rejected:**
- `-strides 1,2,3,4` on `mrconvert -fslgrad` import — introduced y-axis flip via incorrect stride-based gradient rotation
- y-flip at raw import — eddy then applied its own per-volume motion rotation on top, corrupting the correction
- y+z flip — incorrect, tested and ruled out
- `mrcalc -abs` on `ev.mif` — forced all x-components positive, making both CC hemispheres lean the same direction
- Passing `-fslgrad` to `dwi2tensor` on a MIF with embedded gradients — MRtrix conflict error
- Exporting to NIfTI then passing y-flipped bvec to `dwi2tensor` — applied flip in wrong coordinate frame
- Exporting in MRtrix gradient format and negating y column — same coordinate frame problem
- Applying `fslmaths -abs` to the dec map before loading in FSLeyes — broke left-hemisphere display

**Key diagnostic:** `dtifit` run directly on the raw BIDS NIfTI with the unmodified bvec produced a correct "u"-shaped CC (confirmed visually), proving the raw bvec is correct. The y-flip is required only for the eddy-rotated bvec used for tensor fitting, not at the raw import stage.

**Note for viewers:** `dec.nii.gz` must be loaded as an RGB/vector image (`-ot rgb` in FSLeyes). Loading it as greyscale will appear incorrect. The file contains signed values — FSLeyes handles the sign display correctly internally.

---

## Issue 2 — Cleanup on failure deleted working files, breaking --resume

**Status:** Known limitation  
**Symptom:** When a pipeline run fails, cleanup runs anyway and deletes the work directory. A subsequent `--resume` run then fails immediately because the intermediate files it needs (e.g. `dwi_degibbs.mif`, `dwi_biascorr.mif`) no longer exist, even though their checkpoints are recorded as complete.  
**Workaround:** Always use `--no-cleanup` when debugging or when failure is expected. If cleanup has already run, delete all checkpoints for the subject and start fresh:
```bash
rm -f /path/to/BIDS/derivatives/logs/checkpoints/sub-XXX_checkpoints.txt
./dwiforge.sh -s sub-XXX --no-cleanup
```
**Root cause:** The cleanup function runs unconditionally at pipeline end regardless of exit status. It should only run on successful completion.

---

## Issue 3 — ~~dcm2niix bvec y-axis inversion~~ RETRACTED

**Status:** Retracted — was an incorrect diagnosis  
**Original hypothesis:** dcm2niix exported the bvec y-row with inverted polarity, evidenced by the y-row being almost entirely positive and a high asymmetry norm (0.559) from dirstat.  
**Why it was wrong:** The asymmetry norm is a scalar magnitude and is identical before and after negating any axis — it cannot distinguish a flipped axis from a correctly oriented one. The high norm is a property of this particular 32-direction scheme being unipolar (directions distributed on one hemisphere by design). dtifit confirmed the raw bvec is correct.

---

## Issue 4 — FA values exceeding 1.0 at mask boundary

**Status:** Confirmed benign for this dataset  
**Extent:** 0.1% of masked voxels (approximately 822 of 822,625). Mean FA within mask: 0.30.  
**Root cause:** Noise at the mask boundary pushes λ3 below zero. MRtrix3's FA formula does not clamp the result.  
**Resolution:** No fix required. Affected voxels are at the mask periphery and do not affect WM metrics.

---

## Verified correct

- Raw BIDS bvec: correct as exported by dcm2niix — confirmed via dtifit visual inspection
- NIfTI strides: `-1 2 3 4` (x-axis negative; handled correctly by plain `-fslgrad` import without stride forcing)
- Mean FA within mask: 0.30 (expected range 0.25–0.50 for whole-brain masked mean)
- FA > 1 voxels: 0.1% of mask (benign boundary effect)
- dirstat asymmetry norm of 0.559: expected for this unipolar 32-direction scheme; not indicative of a gradient error
- TensorFlow duplicate CUDA factory registration warnings at startup: benign, does not affect computation
- DEC map: produced by dtifit V1 × FA, loaded as `-ot rgb` in FSLeyes without pre-applying abs()
