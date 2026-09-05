# Landmark-guided articulation and contact pre-annotation

svrom-preparation 0.2 provides a preparation workflow for an **ordered sequence of
approximately articulated vertebral meshes**. It accepts the existing
MorphoWeave vertebra SSM or already transferred landmarks, fits rigid reference
articulations, and proposes specimen-specific contact patches. It does not need
a trusted neutral-posture reference, a universal patch atlas, or new boundary
annotations. The existing ROM evaluators and their numerical profiles are unchanged.

## Run with the supplied data

After installing SVROM >=0.7.1, from this repository:

```bash
python -m pip install -e '.[ssm]'

svrom-prepare init \
  --atlas '/path/to/vertebrae_smooth(1).zip' \
  --meshes /path/to/Ouroborus \
  --out ouroborus.yaml

svrom-prepare run ouroborus.yaml --output results/ouroborus
```

`--atlas` also accepts the extracted `vertebrae_smooth` directory containing
`manifest.json`. Extract the target meshes into a directory first. `init`
naturally sorts PLY filenames and writes their order explicitly; inspect that
order, units, and coordinate declarations. A manifest may also list OBJ meshes.
Paths inside a manually written manifest are relative to that manifest.

To run landmark transfer first, then continue fitting:

```bash
svrom-prepare run ouroborus.yaml --output results/ouroborus --transfer-only
svrom-prepare run ouroborus.yaml --output results/ouroborus --resume
```

Resume reuses completed landmark transfers after checking an input/settings
signature and both package versions. Fitting is recomputed. Changed inputs or settings require a fresh
output directory, preventing old patch files from being mistaken for new results.

The CLI returns exit code **2** for a partial result requiring review, and **0**
for a completed transfer-only run or complete geometric reference sequence.
Read `report.json` even when some joints cannot be fitted. A failed search means
that this bounded procedure did not find an acceptable articulation; it does
not establish anatomical impossibility.

## Existing MorphoWeave landmarks

The supplied atlas contains **28 sparse landmarks, 3,319 dense correspondences,
and 242 SSM modes**. The bundled `vertebra28` profile interprets these one-based
control-point numbers:

| Numbers | Search role |
|---|---|
| 1–3 / 4–6 | Anterior facet regions on sides A/B |
| 11,17 / 12,18 | Posterior facet regions on sides A/B |
| 19–22 | Anterior centrum rim |
| 16 | Posterior centrum surface guide |
| 9,10,13 | Dorsal direction |

These are **provisional anatomical interpretations of the supplied template**;
the source file contains numbered labels without an anatomical dictionary.
Control-point order, source IDs, and labels are retained during transfer. An
imported landmark file must preserve the atlas's control-point order.

An interface connects a posterior region on the cranial bone to its matching
anterior region on the caudal bone. Side A/B avoid assuming an anatomical
left/right label from the atlas's arbitrary storage orientation. The profile
can be replaced by a YAML mapping with different regions, radii, and interfaces;
missing or accessory articulations need an appropriate profile.

If landmarks have already been transferred in MorphoWeave, add a `landmarks`
path to each bone entry. The native registration dependency is then unnecessary:

```yaml
schema_version: 1
units: mm
mesh_coordinate_system: LPS
guide_profile: vertebra28
bones:
  - name: T2
    mesh: meshes/T2.ply
    landmarks: landmarks/T2.mrk.json
  - name: T3
    mesh: meshes/T3.ply
    landmarks: landmarks/T3.mrk.json
```

Both Markups JSON and FCSV are supported. RAS/LPS conversions are explicit;
legacy FCSV numeric conventions are **0=RAS, 1=LPS**. The supplied atlas's JSON
points are LPS, while its NPZ SSM arrays use Slicer's internal RAS convention.
`ssm_coordinate_system: RAS` handles that difference, including the mode vectors.
The template can have small smoothing differences from the SSM mean; a large
discrepancy is rejected. Target meshes and landmark distances must be in mm.

## Registration and mesh preservation

The headless transfer path uses rustcpd >=3.1:

1. Center and scale the SSM and target into stable working coordinates.
2. Use the native pose lattice to initialize orientation and SSM coefficients.
3. Pass that state into the full SSM registration.
4. Optionally refine the dense correspondences with nonrigid CPD.
5. Transfer sparse landmarks through a 3-D radial-basis interpolant and project
   them onto the original target triangle surface.

The registration primitives, pose settings, and 3-D interpolation approach
follow [SlicerMorphoWeave's landmark-transfer code](https://github.com/agporto/SlicerMorphoWeave/tree/6fd9e448fb3441b71f7fed2bec9434a9b2190d24/MorphoWeaveLandmarkTransfer).
This standalone path uses the native pose estimate directly; it does not run
the GUI's subsequent tiny3d/FPFH stage and does not claim identical GUI outputs.
Externally supplied MorphoWeave exports remain supported.

**The target bone is never deformed, rescaled, smoothed, or remeshed.** Its
original triangle IDs are retained. Closed-mesh winding may be reoriented to
make normals consistent and outward; the number of reoriented faces is reported,
and each face retains the same three original vertex IDs. Input files are never
overwritten. No automatic hole filling or coordinate-frame repair is performed.

Transfer reports record the backend/version, atlas and target hashes, retained
SSM rank, pose ambiguity diagnostics, projection distances, and dense surface
fit. Surface proximity is a geometric diagnostic, not manual-landmark accuracy.
Gross landmark projection errors stop downstream certification. Smaller
anatomical transfer errors remain possible and require inspection.

## Pose and patch estimation

Each landmark group seeds a broad region using distances along the face-adjacency
graph. Region radii are fractions of landmark-defined centrum length, rather
than the full spine's median bounding-box diagonal. Geodesic graph distances
approximate continuous surface geodesics; the guide regions are not claimed to
be anatomical patch boundaries.

The default `complementary_seating` objective queries the closest original
triangles in the opposing guide region. For each source surface point `p` with
outward normal `n`, receiver point `q`, receiver normal `m`, and assumed gap `g`,
the area-weighted loss is:

```text
log(1 + ||q - p - g*n||² / width²) + normal_weight * ||m + n||²
```

The long-tailed distance term provides a fitting gradient even when surfaces
start apart. Opposite normals favor complementary surfaces rather than
same-facing surface registration. A quadratic tangential offset term weakly
centers the projected landmark groups. It uses the projected guide points,
not seed-triangle centroids, which can have different triangulation offsets.
The average opposing regional normals define the tangent plane. These terms
are explicit geometric assumptions, not a cartilage or soft-tissue model.

The score integrates over **fixed guide regions**. Its area normalization does
not change when only a small part fits. Both directions and all configured
interfaces contribute equally, so a large centrum cannot overwhelm the facets.
Closest-surface assignments update at every trial pose; hard patch masks never
define their own optimum. Final scoring uses every guide-region triangle
centroid, while fitting uses deterministic area quadrature.

Each retained pose must also pass three full-region checks on both sides of
every interface: sufficient area-weighted support, sufficient support spread
in the region's tangential directions, and a bounded guide-centering offset.
Support requires proximity, opposing normals, and surfaces facing across the
gap. Spread is the smallest eigenvalue of the supported footprint covariance
normalized by the full region covariance. These diagnostics help reject a
small or narrow rim contact. They are not anatomical contact probabilities.

Rigid optimization uses the original pose, landmark alignment, surface
normal alignment, and feasible legacy fits as initializations. In particular, the normal-based seed can
recover intrinsic articulation angles even when the anatomical frames are
parallel. There is no straight target centerline and no penalty pulling joint
angles toward zero. All rotations must have determinant +1.

The complementary fitter uses bounded SLSQP with exact signed-triangle distance
constraints on sampled whole-bone face centroids and vertices. Final candidate
acceptance independently checks triangle intersections and containment of every
mesh component. If that check finds a missed intersection, its contact points
are attached to both bones as additional clearance constraints and the fitter
refines again. Small bounded opening translations are a final fallback; every
acceptance criterion is recomputed afterward. Sampling does not certify a
global minimum clearance. The independent mesh check certifies only the stated
nonintersection/containment conditions on closed, consistently wound inputs.

An optimizer iteration limit is reported separately from geometric acceptance;
acceptance does not guarantee convergence or global optimality. The old
`apposition` objective retains Powell and its sampled SDF penalty, both for
reproduction and to supply initializations. Its score does not rank the final
complementary candidates. Use a fresh output directory when changing objectives.

Candidate joint poses are assembled into chains by a bounded beam search that
checks nonadjacent bones for collisions. This preserves the fitted relative
rotations and their resulting curvature. A weak robust penalty on differences
between consecutive anatomical joint rotations helps choose between similarly
seated candidates. It does not prescribe zero angles or a centerline. Set
`chain_consistency_weight: 0` to disable it. Failed adjacencies split the sequence;
the code never substitutes a connection across a missing/rejected vertebra.

Patch support is then evaluated over retained, independently checked
articulations across gap scenarios and distinct fitted initializations. Legacy
mode also probes small local pose perturbations. Outputs distinguish
a consistently supported **core** and a broader **possible** region. Each
interface retains its largest supported connected component. Frequencies are
fractions of the explored candidate set, **not calibrated probabilities or
proof of complete anatomical articular-surface boundaries**.

## Assumptions to inspect

The generated manifest exposes all preparation settings. Defaults are starting
assumptions, not species-specific biological calibrations:

| Setting | Default | Meaning |
|---|---:|---|
| `objective` | `complementary_seating` | Surface seating; `apposition` reproduces the earlier objective |
| `gap_fractions` | 0.01, 0.02, 0.04 | Three gap scenarios relative to pair centrum length |
| `gap_width_fraction` | 0.025 | Width of the soft distance band |
| `rotation_bound_deg` | 25 | Bound on each anatomical rotation-vector component |
| `translation_bound_fraction` | 0.30 | Bound on each translation component relative to length |
| `sample_count` | 192 | Area-stratified fitting samples per guide-region direction |
| `max_evaluations` | 260 | Powell evaluations per legacy initialization |
| `refine_candidates` | 3 | Initializations refined per gap scenario |
| `seating_max_iterations` | 60 | SLSQP iterations per initialization/refinement |
| `seating_distance_fraction` | 0.08 | Long-tailed distance-loss width relative to length |
| `seating_normal_weight` / `seating_center_weight` | 0.5 / 20 | Normal and projected-guide centering weights |
| `minimum_seating_coverage` | 0.20 | Minimum weighted supported fraction of each guide region |
| `minimum_seating_spread` | 0.10 | Minimum normalized support covariance eigenvalue |
| `maximum_seating_offset_fraction` | 0.15 | Maximum tangential guide-centering offset relative to length |
| `clearance_fraction` | 0.001 | Sampled clearance constraint margin relative to length |
| `collision_refinement_rounds` | 2 | Additional refinements using detected intersection points |
| `chain_consistency_weight` / `chain_angle_scale_deg` | 0.02 / 10 | Weak neighboring-joint rotation consistency |
| `ensemble_angle_deg` | 2 | Legacy local rotations used for annotation support |
| `core_frequency` / `extension_frequency` | 0.80 / 0.15 | Candidate-set support thresholds |

The gap scenarios are reported separately and their representatives are retained.
Selecting a high-scoring candidate does not estimate cartilage thickness. The
preparation gap assumptions and study ROM thresholds are separate settings.
Use fresh output directories when exploring sensitivity.

## Outputs and SVROM

`landmarks/` contains the transferred points in input mesh coordinates and their
diagnostics. `report.json` contains per-bone, per-joint, and whole-chain status,
relative angle changes, transforms, and reasons for unresolved cases.

Each accepted joint directory contains:

- Original-coordinate whole-bone OBJ exports.
- Core and possible patch OBJ files for each side and interface.
- `patch_labels.npz`: scores, support frequencies, original zero-based face IDs,
  and patch-to-original vertex index maps. RGB colors are not the annotation.
- `joint_report.json`: retained candidates, assumed gaps, checks, areas, and
  the chosen coordinate origin.
- `joint_core.yaml` and/or `joint_possible.yaml`, emitted only when every
  configured interface has a nonempty patch on both sides.

`articulated/` contains rigidly posed whole meshes and landmarks for successfully
assembled segments. They can be opened together in Slicer to inspect curvature.

The exported role-specific transforms **preserve the fitted articulation at
zero pose**. For fixed-local to joint transform `J` and fitted moving-local to
fixed-local transform `M`, they are `J @ fixed_input_to_local` and
`J @ M @ moving_input_to_local`. Independently zeroing both anatomical frames
would remove the fitted relative orientation and must not replace these matrices.

Landmark 16 is not treated as a measured rotation center. A conditioned sphere
fit to the posterior centrum supplies the reference origin when supported by
the local geometry; otherwise the interface-anchor midpoint is used and recorded.
This is a coordinate convention, not a verified biomechanical center of rotation.

The emitted search grids contain the zero pose only. First inspect the
pre-annotations, then set study-specific ROM grids and justified constraints:

```bash
svrom-pose results/ouroborus/joints/JOINT/joint_possible.yaml --mode robust
```

Core and possible exports support a boundary-sensitivity comparison. Keep the
chosen patch definitions fixed throughout each ROM run. The preparation process
never uses the final SVROM coverage threshold to expand its own patch labels.

## Verification

Run `pytest -q tests/test_articulation.py` for coordinate, interpolation,
surface-query, rigid-pose, collision, export, and chain checks. The full suite
also checks that existing ROM behavior remains intact. Native landmark transfer
can be checked against a known transformed copy of the supplied template with
`scripts/validate_articulation_transfer.py`.

See [the recorded validation results](VALIDATION.md), including
measured transfer error and sensitivity to registration settings.

Audit an actual run's original triangle identities and exported transforms with:

```bash
python scripts/validate_articulation_exports.py --results results/ouroborus
```

Add `--check-rom` to prepare cached distance fields and evaluate every exported
SVROM zero pose; `--backend python` or `--backend meshrom` makes the evaluator
choice explicit. This more expensive check writes `export_validation.json` and
returns 2 if an exported zero pose fails those SVROM settings. It does not change
the fitted articulation or relax the settings to obtain a pass.

Geometric recovery tests and original-data runs verify implementation behavior.
They cannot establish the habitual posture of species without posture reference
data. A geometric reference articulation and reviewable pre-annotation are the
intended outputs.
