"""Specimen-specific reference articulation and contact pre-annotation.

This preparation workflow does not change the existing ROM evaluators. Its
outputs are geometric reference poses and reviewable candidate patches, not
estimates of habitual living posture or calibrated anatomical probabilities.
"""

from .settings import ArticulationSettings, TransferSettings, vertebra28_profile
from ._version import __version__

__all__ = ['ArticulationSettings', 'TransferSettings', 'vertebra28_profile', '__version__']
