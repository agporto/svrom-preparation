# Changelog

## 0.1.0

- Extracted the tested preparation workflow into the `svrom_preparation`
  namespace and its own installable package/repository.
- Added `svrom-prepare`, its `svrom-articulate` alias, and `svrom-prepare-check`.
- Kept a one-way dependency on SVROM's existing geometry implementation.
- Recorded preparation and analysis package versions independently for resume.
- Retained SSM/landmark inputs, curvature-preserving rigid fits, core/possible
  patch annotation, original face mappings, and SVROM configuration export.
