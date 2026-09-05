"""Explicit, serializable assumptions for preparation, separate from ROM."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class TransferSettings:
    rotation_count: int = 193
    refine_count: int = 12
    coarse_count: int = 400
    target_count: int = 1600
    max_iterations: int = 150
    variance_keep: float = 0.95
    lambda_regularization: float = 0.01
    fine_cpd: bool = True
    fine_rank: int = 300
    tps_count: int = 800
    seed: int = 0
    parallel: bool = True
    maximum_projection_fraction: float = 0.08

    def __post_init__(self):
        for name in ('rotation_count', 'refine_count', 'coarse_count', 'target_count',
                     'max_iterations', 'fine_rank', 'tps_count'):
            _positive_int(name, getattr(self, name))
        _positive_int('seed', self.seed, 0)
        if self.tps_count < 4 or self.target_count < 4:
            raise ValueError('tps_count and target_count must be at least four')
        if not np.isfinite(self.variance_keep) or not 0 < self.variance_keep <= 1:
            raise ValueError('variance_keep must be in (0, 1]')
        if not np.isfinite(self.lambda_regularization) or self.lambda_regularization < 0:
            raise ValueError('lambda_regularization must be finite and nonnegative')
        if not np.isfinite(self.maximum_projection_fraction) or self.maximum_projection_fraction <= 0:
            raise ValueError('maximum_projection_fraction must be finite and positive')


@dataclass(frozen=True)
class ArticulationSettings:
    # Fractions of the pair's mean landmark-defined centrum length. These are
    # modeling scenarios, not inferred cartilage thicknesses.
    gap_fractions: tuple[float, ...] = (0.01, 0.02, 0.04)
    gap_width_fraction: float = 0.025
    rotation_bound_deg: float = 25.0
    translation_bound_fraction: float = 0.30
    sample_count: int = 192
    collision_samples: int = 512
    sdf_samples: int = 48
    max_evaluations: int = 260
    refine_candidates: int = 3
    retain_candidates: int = 5
    ensemble_angle_deg: float = 2.0
    ensemble_energy_slack: float = 0.25
    normal_cosine: float = 0.25
    minimum_interface_support: float = 0.015
    patch_score_threshold: float = 0.15
    core_frequency: float = 0.80
    extension_frequency: float = 0.15
    maximum_origin_step_fraction: float = 4.0
    chain_beam_width: int = 4

    def __post_init__(self):
        object.__setattr__(self, 'gap_fractions', tuple(self.gap_fractions))
        if (not self.gap_fractions or not np.isfinite(self.gap_fractions).all()
                or min(self.gap_fractions) <= 0 or len(set(self.gap_fractions)) != len(self.gap_fractions)):
            raise ValueError('gap_fractions must be distinct finite positive values')
        for name in ('sample_count', 'collision_samples', 'max_evaluations',
                     'refine_candidates', 'retain_candidates', 'chain_beam_width'):
            _positive_int(name, getattr(self, name))
        _positive_int('sdf_samples', self.sdf_samples, 16)
        for name in ('gap_width_fraction', 'rotation_bound_deg', 'translation_bound_fraction',
                     'maximum_origin_step_fraction'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('ensemble_angle_deg', 'ensemble_energy_slack'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('normal_cosine', 'minimum_interface_support', 'patch_score_threshold',
                     'core_frequency', 'extension_frequency'):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f'{name} must be in (0, 1)')
        if self.extension_frequency > self.core_frequency:
            raise ValueError('extension_frequency cannot exceed core_frequency')

    def as_dict(self):
        return asdict(self)


def vertebra28_profile():
    """Provisional anatomical interpretation of the supplied numbered atlas.

    Numbers are one-based control-point order, retained during transfer. The
    model does not contain anatomical labels. Side A/B avoid assuming a
    biological left/right sign from an arbitrary atlas orientation.
    """
    return {
        'name': 'vertebra28', 'minimum_landmarks': 28,
        'interpretation': 'provisional; inferred from supplied template geometry',
        'anterior': [19, 20, 21, 22], 'posterior': [16], 'dorsal': [9, 10, 13],
        'side_a': [1, 2, 3], 'side_b': [4, 5, 6],
        'regions': {
            'anterior_a': {'landmarks': [1, 2, 3], 'radius_fraction': 0.35},
            'anterior_b': {'landmarks': [4, 5, 6], 'radius_fraction': 0.35},
            'posterior_a': {'landmarks': [11, 17], 'radius_fraction': 0.35},
            'posterior_b': {'landmarks': [12, 18], 'radius_fraction': 0.35},
            'anterior_centrum': {'landmarks': [19, 20, 21, 22], 'radius_fraction': 0.32},
            'posterior_centrum': {'landmarks': [16], 'radius_fraction': 0.44},
        },
        'interfaces': [
            {'name': 'facet_a', 'fixed': 'posterior_a', 'moving': 'anterior_a'},
            {'name': 'facet_b', 'fixed': 'posterior_b', 'moving': 'anterior_b'},
            {'name': 'centrum', 'fixed': 'posterior_centrum', 'moving': 'anterior_centrum'},
        ],
    }
