# Preparation validation

The preparation workflow is separate from the existing ROM numerical
profiles. These checks characterize the implementation and its limitations;
they do not establish species-specific habitual posture.

## Standalone extraction — 0.1.0

- Built and installed separate `svrom-preparation` 0.1.0 and unmodified
  `svrom-python` 0.7.1 wheels. The preparation wheel contains only the
  `svrom_preparation` package. The analysis wheel has no `svrom.articulation`
  namespace.
- All **15 preparation tests passed against the installed wheels**, including
  independent preparation/SVROM version checks when resuming runs.
- Verified the installed preparation, export-check, and Python-module entry
  points.
- Repeated native landmark transfer and all three gap scenarios for T2/T3.
  The 28 transferred landmarks, five retained pose matrices, energies, interface
  supports, and collision decisions were identical to the pre-extraction run.
  Maximum measured numeric difference was **0.0** in this controlled comparison.
- The installed export checker verified all **96** previously generated patch
  meshes and both whole-bone transforms for all **16** configurations.
- Restored the analysis checkout exactly to commit
  `d420b01864368a848013204a8c263040d537cbc8`; its source is unchanged.

This is a packaging/ownership change. The dependency on SVROM's existing geometry
implementation is explicit and one-way. No geometry implementation is copied
into this package. The original geometric and transfer accuracy results below
still apply; numerical equivalence of extraction does not establish additional
biological accuracy.

## Automated and packaging checks

- Before extraction, the combined prototype suite had **184 passing tests**: the existing SVROM tests with meshrom 0.1.2 and 15 preparation tests.
  The standalone extraction checks are recorded below.
- Wheel construction and installation succeeded; the installed
  `svrom-articulate` entry point loads and displays its commands.
- Preparation tests cover explicit RAS/LPS and FCSV conversion, atlas ZIP input,
  affine landmark interpolation at large coordinate offsets, proper rigid
  transforms, closest-triangle queries, collision/containment, original patch
  identity, zero-pose export, chain composition, and changed-input rejection.
- Two thin crossing solids with no vertices inside the opposing solid are
  correctly rejected by the triangle-intersection check. Open surfaces cannot
  receive a verified articulation status.

## Curvature recovery

A constructed joint has parallel anatomical landmark frames but articular
surfaces inclined by 10 degrees. The fit recovered their opposing-normal
alignment with **0.00122 degrees** error and passed the independent collision
check. Rotation about a single planar interface's normal is underdetermined;
this test does not claim to recover that free degree of freedom. The regression
tolerance is 0.2 degrees.

A separate chain-composition test retains successive 12-degree relative bends
and produces a curved sequence of centers. Export tests verify that the fitted
relative orientation survives loading the resulting SVROM configuration at
zero pose. Neither test imposes a straight centerline on the fitting objective.

## Native SSM transfer

The supplied model contains 28 sparse points, 3,319 dense correspondences, and
242 SSM modes; the default 95% variance setting retained 24 modes. Its NPZ uses
RAS coordinates and its template/Markups use LPS.

For a controlled check, the template was scaled by 8, rotated by XYZ Euler
angles [20, -30, 55] degrees, and translated by [100, -200, 300] mm. The known
transformed sparse landmarks provide the comparison positions. With rustcpd
3.1.0 and the default transfer settings:

| Metric | Value |
|---|---:|
| RMS landmark error | 0.0873004 mm |
| RMS error / target diagonal | 1.460% |
| Maximum landmark error | 0.263600 mm |
| Registered dense surface RMS distance | 0.0091522 mm |

The template sparse points have small nonzero surface offsets, but projecting
the expected points does not explain away the main error. The larger errors
include landmarks 10, 27, and 28. This is an **approximate registration and
transfer**, even on the known template. Close surface fit is not evidence of
equally accurate anatomical correspondence on a new species.

Two diagnostic sensitivity runs further illustrate this limitation: disabling
fine CPD gave 0.0895350 mm RMS error; increasing target sampling from 1,600 to
4,000 points gave 0.918883 mm RMS error in this case. More samples did not
guarantee a better registration. Those alternate settings were not substituted
for the documented defaults. Inspect transferred landmarks and the pose
diagnostics, or supply reviewed MorphoWeave exports before articulation.

Reproduce the controlled default run with an extracted atlas directory:

```bash
python scripts/validate_articulation_transfer.py \
  --atlas /path/to/vertebrae_smooth \
  --output results/known_transform
```

The script writes expected and predicted Markups files, a transformed template,
and per-landmark errors. It does not convert an arbitrary error threshold into
an anatomical accuracy claim.

## Supplied Ouroborus meshes

Landmarks were transferred to all **21** supplied meshes. The three assumed gap
scenarios produced the following adjacent-pair results:

| Outcome | Adjacent pairs |
|---|---:|
| Verified geometric articulation | 8 |
| Coordinate-frame mismatch | 6 |
| Open mesh requires review | 4 |
| No verified candidate within the bounded search | 2 |

The accepted pairs form **T2–L1** and **L7–L10** segments. All retain their
individual fitted rotations. The unsuccessful fits are **L1/L2** and **L4/L5**;
their best scoring attempts had intersections and inadequate facet support.
L2 and L5 also had the largest dense registration residuals, approximately
0.109 and 0.117 mm, making their transferred guides a useful review target.
This observation does not establish the cause of the unsuccessful fits.

Transfer took 143.8 seconds in total. The subsequent full workflow with cached
landmarks took 528.1 seconds on the validation worker. Timing depends on meshes,
hardware, and settings. Dense registration surface RMS distances ranged from
0.0204 to 0.1170 mm; these are not landmark correspondence errors.

Original source meshes, the SSM, and detailed specimen audit records are not
bundled with this report. The workflow records content hashes, transforms,
and individual errors in each local run's outputs and keeps the original input
files unchanged. This document records aggregate validation results.

The export audit checked **96 patch meshes** against the original face/vertex
index maps and verified both whole-bone role transforms. All **16** core/possible
SVROM configurations passed the neutral-pose check with meshrom 0.1.2 and the
exported settings. All 21 original mesh hashes were unchanged. These checks
verify the exported geometries and settings; they do not calibrate those settings
biologically. Repeat the audit on a new run with:

```bash
python scripts/validate_articulation_exports.py \
  --results results/ouroborus --check-rom --backend meshrom
```

No coordinate repair or hole filling is performed. The displaced C4, C6, and
T1 inputs and open L3/L6 surfaces remain for manual correction. A failed fit
means no verified candidate was found under the declared search assumptions;
it does not prove that the anatomical articulation is impossible.

The reported gaps, guide radii, support cutoffs, and search bounds are modeling
assumptions. Proposed core/possible contact regions remain pre-annotations.
Anatomical boundary accuracy and biological neutral posture require evidence
beyond these geometric checks.
