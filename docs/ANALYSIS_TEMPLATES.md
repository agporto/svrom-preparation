# Size-aware SVROM analysis templates

Preparation can populate a study analysis configuration while retaining each
target's original meshes, fitted articulation, and inferred contact patches.
The template supplies the analysis settings; it does not supply target anatomy
or establish biological neutral posture. Omitting `analysis` preserves the
existing inspection-only, zero-pose export behavior.

## Original two-snake-vertebra example

Add this section to a preparation manifest. Paths are relative to the manifest:

```yaml
analysis:
  template: ../svrom_python/examples/svrom02_cervicalB.yaml
  reference_frame_landmarks: ../svrom_python/data/reference_scene/SVROM02_cervicalB_reference_frames.json
```

Alternatively, create a new manifest with:

```bash
svrom-prepare init \
  --atlas /path/to/vertebrae_smooth \
  --meshes /path/to/Ouroborus \
  --out ouroborus.yaml \
  --analysis-template /path/to/svrom_python/examples/svrom02_cervicalB.yaml \
  --reference-frame-landmarks /path/to/svrom_python/data/reference_scene/SVROM02_cervicalB_reference_frames.json

svrom-prepare run ouroborus.yaml --output results/ouroborus_analysis
svrom-prepare-check --results results/ouroborus_analysis --check-rom --backend python
```

Use a fresh output directory when adding or changing a template. Already
transferred MorphoWeave landmarks can be supplied in the manifest as usual.
This skips registration; the normal workflow still computes articulation fits.

The original reference JSON contains `fixed_landmarks_local` and
`moving_landmarks_local`, each with 3D `cotyle` and `condyle` coordinates. Those
coordinates must use the template's physical units. Their Euclidean distances
are independent of the specimen's stored orientation or translation. The
source snake lengths are **0.3185503461 and 0.3195350770 cm**, with a mean of
**3.1904271156 mm**. These are reference object measurements, not a trusted
posture reference for other species.

For another template, provide the same landmark JSON structure or explicitly
measured centrum lengths in that template's units:

```yaml
analysis:
  template: study.yaml
  reference_centrum_lengths:
    fixed: 0.318550346089484
    moving: 0.319535077029534
```

Supply exactly one reference measurement source. The example numbers above
belong to the centimetre-based snake template; they are not universal defaults.
The template must use SVROM schema 2 and explicit physical units. Supported
units are mm, cm, m, and um, including the singular spelled-out equivalents.

## Scaling rule

Let `Lref` be the mean of the two reference centrum lengths in template units,
`u` the template-unit conversion to millimetres, and `Ltarget` the mean of the
two target centrum lengths in millimetres. For every dimensional setting `d`:

```text
physical_size_ratio = Ltarget / (u * Lref)
d_target_mm = d_template * u * physical_size_ratio
```

The target length is the distance between the guide profile's anterior and
posterior landmark-group centers. In the provisional `vertebra28` profile,
these are landmarks 19–22 and landmark 16. The reference and target endpoint
definitions should be anatomically comparable; errors in transferred guides
also affect this scale estimate. No bounding-box, whole-chain, or pose-dependent
size estimate is substituted. A single isotropic factor is used for each pair;
neither bone is resized or deformed to match the other.

| Setting | Treatment |
|---|---|
| Translation coordinates and step spacing | Scale as lengths |
| `penetration_tolerance`, `maximum_quantile_depth` | Scale as lengths when set |
| `apposition_max_gap` | Scale as a length |
| `sdf_voxel_size`, `sdf_smoothing_sigma_mesh_units` | Scale as lengths when set |
| Explicit Maya `ray_length` | Scale as a length |
| Rotations, angular thresholds | Preserve |
| Coverage/penetrating-area fractions and quantiles | Preserve |
| Sample counts, neighbor counts, SDF axis resolution | Preserve |
| SDF padding fraction, smoothing sigma in voxels, Maya `ray_length_scale` | Preserve |

Defaults from the installed SVROM `RobustSettings` are made explicit before
scaling, so omitted dimensional defaults cannot remain accidentally in source
units. Null values remain null. Source translation ranges are resolved before
scaling, preserving grid membership and order despite floating-point rounding.
The template and reference JSON are content-hashed; changes invalidate resume.
Each joint YAML and report records both unit conversion and physical size ratio.

## Search and downstream interpretation

The original snake template retains RX -10 to 10, RY -30 to 30, and RZ -20 to
20 degrees, all at 1-degree steps. There are **52,521 orientations** and **125
translations**, or **6,565,125 candidate poses per configuration** before any
search short-circuiting. Its translation grid is asymmetric and has no exact
zero-translation point. Preparation preserves that lattice; the separate
`svrom-prepare-check --check-rom` audit evaluates the actual zero pose directly.
Do not infer that the zero pose was searched merely because the rotation is zero.

Axes follow +X posterior, +Y ventral, and +Z completing the right-handed frame.
The original snake ACS uses this convention. Other study templates must use
compatible axis meanings; this exporter does not infer an axis permutation or
reorient a template's search lattice. The target's role transforms retain its
fitted articulation, with the estimated joint origin recorded as before.

The snake's four accessory/facet patch pairs, historical Maya labels, coordinate
matrices, and scene-validation provenance are not copied to target joints.
Only search, robust settings, and the explicit Maya settings are transferred.
Target anatomical interfaces still come from the guide profile and distance
queries. Analysis thresholds never expand the inferred patch boundaries.

Use **`--mode robust`** for these size-normalized analyses. SVROM's Maya
compatibility evaluator retains a hard-coded 10 mesh-unit collision cutoff;
scaling its configurable ray length does not make that entire profile scale
invariant. The analysis repository and either evaluator's math are unchanged.

The source robust settings are provisional. Size normalization preserves their
relative geometric meaning, but it does not calibrate cartilage thickness,
segmentation uncertainty, or biological ROM across species. Known voxel or
measurement uncertainty may justify different physical tolerances. A target
may fail a transferred criterion even when its preparation seating checks pass;
the exporter does not relax criteria or alter its pose to force feasibility.

Reference-pose validation and a few sampled rotations/translations do not
constitute an exhaustive ROM search. See [validation results](VALIDATION.md)
for the measured checks on synthetic geometry and the supplied Ouroborus joints.
