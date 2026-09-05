# svrom-preparation

Prepare ordered vertebral meshes for SVROM analyses using an existing
MorphoWeave statistical shape model or transferred landmarks.

This package owns the preparation sequence: **SSM landmark transfer → rigid
reference articulation → specimen-specific contact patches → SVROM inputs**.
It allows curvature to follow the articulating surfaces and retains the fitted
relative rotations in the exported zero pose. Outputs remain geometric reference
poses and anatomical pre-annotations for review.

Version 0.2 fits complementary opposing surfaces with distance, normal, and
landmark-guided seating terms. Sampled clearance constraints are refined at
detected intersections, and every retained pose must pass the independent
triangle/containment check and distributed-support criteria. A small contact
near one rim is no longer enough to accept a joint. The earlier objective remains
available with `articulation.objective: apposition` for controlled comparisons.

## Installation

Install the SVROM analysis package first. This preparation package has a one-way
dependency on its tested `SurfaceMesh`, closest-triangle queries, and signed
distance fields. SVROM does not import or depend on `svrom-preparation`.

The extraction was tested with SVROM 0.7.1 on this branch:

```bash
git clone --branch feature/rom-reproducibility-surface-contact \
  https://github.com/agporto/svrom_python.git
python -m pip install -e ./svrom_python

git clone https://github.com/agporto/svrom-preparation.git
python -m pip install -e './svrom-preparation[ssm]'
```

The `ssm` extra installs rustcpd for native landmark transfer. When supplying
reviewed MorphoWeave landmarks, install `svrom-preparation` without that extra.
SVROM must be at least version 0.7.1; the geometry routines used here are already
present in that version. No preparation changes to the analysis repository are
required.

## Prepare a specimen

Extract the target vertebra meshes into an ordered PLY directory. The atlas can
be a MorphoWeave ZIP or an extracted atlas directory containing `manifest.json`.

```bash
svrom-prepare init \
  --atlas '/path/to/vertebrae_smooth(1).zip' \
  --meshes /path/to/Ouroborus \
  --out ouroborus.yaml

# Inspect the explicit vertebral order, units, and coordinate declarations.
svrom-prepare run ouroborus.yaml --output results/ouroborus
```

`python -m svrom_preparation` runs the same CLI. `svrom-articulate` is retained
as an alias for the command used during the prototype.

For a two-stage workflow:

```bash
svrom-prepare run ouroborus.yaml --output results/ouroborus --transfer-only
svrom-prepare run ouroborus.yaml --output results/ouroborus --resume
```

Resume requires matching inputs, settings, preparation version, and SVROM
version. For results from the earlier in-repository prototype, create a fresh
manifest/output directory and list its reviewed `landmarks/*.mrk.json` files
as the bone entries' `landmarks` paths. This reuses transferred points explicitly.

## Outputs and analysis

The preparation output contains:

- Transferred landmarks and fit diagnostics.
- Articulated meshes for accepted contiguous segments.
- Core and possible contact-patch OBJ files with original face/vertex mappings.
- SVROM joint YAML files whose zero pose retains the fitted articulation.
- A review report for displaced inputs, open meshes, and unsuccessful fits.

Audit the outputs, then run SVROM independently:

```bash
svrom-prepare-check --results results/ouroborus
svrom-prepare-check --results results/ouroborus --check-rom --backend python
svrom-pose results/ouroborus/joints/JOINT/joint_possible.yaml --mode robust
```

The emitted ROM grids contain zero pose only. Inspect the poses, guides, patches,
and assumed gaps before selecting study-specific ROM grids and thresholds.
Original target files are never overwritten, deformed, or automatically repaired.
An unsuccessful bounded search does not establish anatomical impossibility.

See the [workflow guide](docs/WORKFLOW.md) for the supplied 28-landmark profile,
coordinates, settings, patch interpretation, and output schema. The
[validation report](docs/VALIDATION.md) records measured accuracy and limitations.

## Package layout

| Module | Responsibility |
|---|---|
| `data` | Input meshes, coordinate conventions, landmarks, rigid frames |
| `transfer` | Native SSM registration and sparse landmark transfer |
| `surfaces` | Guide regions, surface support, collision checks, patch ensembles |
| `seating` | Complementary surface loss, distributed support, adaptive clearance constraints |
| `fitting` | Adjacent-joint optimization and chain assembly |
| `workflow` | Manifests, provenance, resumable execution, SVROM exports |
| `validation` | Original-face identity and exported-pose checks |
| `settings` / `cli` | Explicit assumptions and command-line interface |

```python
from svrom_preparation.workflow import create_manifest, run_manifest

create_manifest('atlas.zip', 'meshes', 'specimen.yaml')
report = run_manifest('specimen.yaml', 'results/specimen')
```

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The tests create synthetic meshes at runtime. Specimen meshes, SSM arrays,
transferred coordinates, and detailed private audit records are not bundled
with this repository. They remain local inputs and outputs.
