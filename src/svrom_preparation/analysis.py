"""Transfer SVROM analysis settings using landmark-defined centrum size.

Only analysis settings are transferred. Reference geometry, anatomical frames,
patch identities, and historical reference labels never become target data.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from math import prod
from pathlib import Path

import numpy as np
import yaml

from svrom.config import AxisGrid
from svrom.constraints.settings import RobustSettings

from .data import sha256


# All physical lengths in RobustSettings, including optional/defaulted fields.
# Counts, fractions, angles, and smoothing sigma expressed in voxels are invariant.
ROBUST_LENGTH_FIELDS = (
    'penetration_tolerance', 'maximum_quantile_depth', 'sdf_voxel_size',
    'sdf_smoothing_sigma_mesh_units', 'apposition_max_gap',
)
_MM_PER_UNIT = {
    'mm': 1., 'millimeter': 1., 'millimetre': 1.,
    'cm': 10., 'centimeter': 10., 'centimetre': 10.,
    'm': 1000., 'meter': 1000., 'metre': 1000.,
    'um': .001, 'micrometer': .001, 'micrometre': .001,
}


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a finite positive length')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive length')
    return value


def resolve_analysis_spec(spec, base):
    """Resolve the optional manifest analysis section without implicit defaults."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError('analysis must be a mapping')
    spec = dict(spec)
    unknown = set(spec) - {'template', 'reference_frame_landmarks', 'reference_centrum_lengths'}
    if unknown:
        raise ValueError(f'unknown analysis settings: {sorted(unknown)}')
    if not spec.get('template'):
        raise ValueError('analysis.template is required')
    sources = ('reference_frame_landmarks', 'reference_centrum_lengths')
    if sum(spec.get(k) is not None for k in sources) != 1:
        raise ValueError('analysis needs exactly one of reference_frame_landmarks or reference_centrum_lengths')
    for key in ('template', 'reference_frame_landmarks'):
        if spec.get(key) is not None:
            path = (Path(base)/Path(spec[key]).expanduser()).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            spec[key] = path
    return spec


@dataclass(frozen=True)
class AnalysisTemplate:
    search: dict
    robust: dict
    maya: dict
    provenance: dict

    @classmethod
    def load(cls, spec, *, base=Path('.')):
        spec = resolve_analysis_spec(spec, base)
        if spec is None:
            return None
        path = spec['template']
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, Mapping) or raw.get('schema_version') != 2:
            raise ValueError('analysis template must use SVROM schema_version 2')
        units = str(raw.get('units', '')).lower()
        if units not in _MM_PER_UNIT:
            raise ValueError('analysis template needs explicit recognized physical units')
        search = deepcopy(raw.get('search'))
        if not isinstance(search, Mapping):
            raise ValueError('analysis template needs a search mapping')
        counts = {}
        for kind, axes in (('rotations', ('rx_deg', 'ry_deg', 'rz_deg')),
                           ('translations', ('tx', 'ty', 'tz'))):
            if not isinstance(search.get(kind), Mapping) or set(search[kind]) != set(axes):
                raise ValueError(f'analysis template needs exactly these {kind}: {axes}')
            counts[kind] = []
            for axis in axes:
                grid = AxisGrid.from_value(search[kind][axis], name=f'{kind}.{axis}')
                counts[kind].append(len(grid.values))
                if kind == 'translations':
                    # Resolve before scaling: do not re-round a scaled step/range.
                    search[kind][axis] = {'values': list(grid.values)}
        robust = asdict(RobustSettings(**dict(raw.get('robust', {}))))
        maya = deepcopy(raw.get('maya_compat', {}))
        if not isinstance(maya, Mapping) or set(maya) - {'ray_length', 'ray_length_scale', 'point_normal_policy'}:
            raise ValueError('unsupported maya_compat template settings')
        for key in ('ray_length', 'ray_length_scale'):
            if maya.get(key) is not None:
                maya[key] = _positive(maya[key], f'maya_compat.{key}')
        if maya.get('point_normal_policy', 'bifrost_geometric_area_corner_angle') not in {
                'bifrost_geometric_area_corner_angle', 'mesh'}:
            raise ValueError('unsupported maya_compat.point_normal_policy')

        if spec.get('reference_frame_landmarks') is not None:
            source = spec['reference_frame_landmarks']
            frame_data = json.loads(source.read_text())
            lengths = {}
            for side in ('fixed', 'moving'):
                landmarks = frame_data[f'{side}_landmarks_local']
                points = np.asarray([landmarks['cotyle'], landmarks['condyle']], dtype=float)
                if points.shape != (2, 3) or not np.all(np.isfinite(points)):
                    raise ValueError(f'{side} reference cotyle/condyle must be finite 3D points')
                lengths[side] = float(np.linalg.norm(points[1]-points[0]))
            reference = {'path': str(source), 'sha256': sha256(source),
                         'measurement': 'cotyle_to_condyle_landmark_distance'}
        else:
            lengths = spec['reference_centrum_lengths']
            reference = {'measurement': 'explicit_centrum_lengths_in_template_units'}
        if not isinstance(lengths, Mapping) or set(lengths) != {'fixed', 'moving'}:
            raise ValueError('reference_centrum_lengths needs fixed and moving lengths in template units')
        lengths = {side: _positive(lengths[side], f'reference centrum length: {side}')
                   for side in ('fixed', 'moving')}
        provenance = {
            'template': {'path': str(path), 'sha256': sha256(path), 'name': raw.get('name', path.stem)},
            'reference': reference, 'reference_units': units,
            'reference_centrum_lengths': lengths,
            'unit_conversion_to_mm': _MM_PER_UNIT[units],
            'scaling_method': 'mean_landmark_centrum_length',
            'orientation_count': prod(counts['rotations']),
            'translation_count': prod(counts['translations']),
            'pose_count': prod(counts['rotations']+counts['translations']),
            'validated_mode': 'robust',
            'axis_convention': '+X posterior, +Y ventral, +Z completes the right-handed frame',
            'threshold_interpretation': 'size-normalized template assumptions, not species calibration',
            'maya_scale_limitation': 'SVROM maya_compat retains a fixed 10 mesh-unit collision cutoff',
        }
        return cls(dict(search), robust, dict(maya), provenance)

    def for_joint(self, fixed_length_mm, moving_length_mm):
        """Return independent settings and an audit record; never scale geometry."""
        lengths = {side: _positive(value, f'target {side} centrum length')
                   for side, value in (('fixed', fixed_length_mm), ('moving', moving_length_mm))}
        audit = deepcopy(self.provenance)
        reference_mean = sum(audit['reference_centrum_lengths'].values())/2
        target_mean = sum(lengths.values())/2
        conversion = audit['unit_conversion_to_mm']
        size_ratio = target_mean/(reference_mean*conversion)
        factor = _positive(conversion*size_ratio, 'length scaling factor')
        search, robust, maya = deepcopy((self.search, self.robust, self.maya))
        for axis, grid in search['translations'].items():
            values = np.asarray(grid['values'])*factor
            grid['values'] = list(AxisGrid.from_value(values.tolist(), name=axis).values)
        scaled_fields = [f'search.translations.{a}' for a in search['translations']]
        for key in ROBUST_LENGTH_FIELDS:
            if robust[key] is not None:
                robust[key] *= factor
                scaled_fields.append(f'robust.{key}')
        # Validate scaled settings as well, including overflow/nonfinite results.
        RobustSettings(**robust)
        if maya.get('ray_length') is not None:
            maya['ray_length'] = _positive(maya['ray_length']*factor, 'maya_compat.ray_length')
            scaled_fields.append('maya_compat.ray_length')
        audit.update({
            'target_units': 'millimeter', 'target_centrum_lengths_mm': lengths,
            'reference_mean_centrum_length_mm': reference_mean*conversion,
            'target_mean_centrum_length_mm': target_mean,
            'physical_size_ratio': size_ratio, 'numeric_length_multiplier': factor,
            'scaled_fields': scaled_fields,
            'zero_translation_in_lattice': all(0. in g['values'] for g in search['translations'].values()),
        })
        return {'search': search, 'robust': robust, 'maya_compat': maya}, audit
