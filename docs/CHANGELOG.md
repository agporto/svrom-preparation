# Changelog

## 0.2.0

- Added complementary surface seating with long-tailed surface distances,
  opposite normals, and projected landmark centering on fixed guide regions.
- Added distributed support and footprint-spread checks for every interface,
  preventing small rim contacts from receiving an accepted seating status.
- Added bounded SLSQP clearance constraints and adaptive intersection witnesses;
  retained independent full-mesh intersection and containment verification.
- Added a weak neighboring-joint rotation consistency term without prescribing
  a straight centerline, plus per-candidate seating and convergence diagnostics.
- Kept the 0.1 apposition objective available for reproduction and initialization.
- Added rigid-invariance, edge-contact, collision-witness, legacy-mode, and known
  nonplanar complementary-surface regression tests.
- Preserved original mesh/face identity, transferred landmark input, specimen
  patch queries, and fitted zero-pose export. SVROM analysis source is unchanged.

## 0.1.0

- Extracted the tested preparation workflow into the `svrom_preparation`
  namespace and its own installable package/repository.
- Added `svrom-prepare`, its `svrom-articulate` alias, and `svrom-prepare-check`.
- Kept a one-way dependency on SVROM's existing geometry implementation.
- Recorded preparation and analysis package versions independently for resume.
- Retained SSM/landmark inputs, curvature-preserving rigid fits, core/possible
  patch annotation, original face mappings, and SVROM configuration export.
