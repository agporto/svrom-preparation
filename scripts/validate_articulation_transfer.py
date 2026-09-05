#!/usr/bin/env python3
"""Measure native landmark recovery on a known transformed atlas template.

This controlled identity check has known landmark correspondences. It is not
an estimate of anatomical landmark accuracy on novel species. The template's
small smoothing differences from the SSM mean remain in this test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from svrom_preparation.data import Bone, Landmarks, transform_points, write_json, write_landmarks
from svrom_preparation.settings import TransferSettings
from svrom_preparation.transfer import Atlas, transfer_landmarks
from svrom_preparation.workflow import write_obj


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, required=True,
                        help='Extracted MorphoWeave atlas directory containing manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args(argv)
    root = args.atlas.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be a new or empty directory')
    atlas = Atlas.load(root)
    manifest = json.loads((root/'manifest.json').read_text())
    model_path = (root/manifest['files']['model']).resolve()
    if not model_path.is_relative_to(root):
        parser.error('model must be inside the atlas directory')
    mesh = trimesh.load(model_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        parser.error('template must be a single triangle mesh')
    # The supplied template PLY and sparse points are both LPS; the NPZ SSM
    # frame conversion is tested by Atlas.load rather than silently bypassed.
    matrix = np.eye(4)
    matrix[:3, :3] = 8.*Rotation.from_euler('xyz', [20., -30., 55.], degrees=True).as_matrix()
    matrix[:3, 3] = [100., -200., 300.]
    out.mkdir(parents=True, exist_ok=True)
    target_path = out/'known_transform.obj'
    write_obj(target_path, transform_points(mesh.vertices, matrix), mesh.faces)
    target = Bone.load(target_path)
    settings = TransferSettings()
    predicted, diagnostic = transfer_landmarks(atlas, target, settings)
    expected = Landmarks(transform_points(atlas.sparse.points, matrix), atlas.sparse.labels,
                         atlas.sparse.ids, atlas.sparse.coordinates)
    errors = np.linalg.norm(predicted.points-expected.points, axis=1)
    write_landmarks(out/'expected.mrk.json', expected)
    write_landmarks(out/'predicted.mrk.json', predicted)
    report = {
        'experiment': 'known similarity transform of the supplied template',
        'interpretation': 'controlled transfer recovery, not novel-species anatomical accuracy',
        'template_to_target': matrix, 'settings': asdict(settings),
        'landmark_error_mm': errors, 'rms_error_mm': float(np.sqrt(np.mean(errors**2))),
        'median_error_mm': float(np.median(errors)), 'maximum_error_mm': float(errors.max()),
        'target_diagonal_mm': target.mesh.diagonal,
        'rms_error_fraction_diagonal': float(np.sqrt(np.mean(errors**2))/target.mesh.diagonal),
        'maximum_error_fraction_diagonal': float(errors.max()/target.mesh.diagonal),
        'transfer': diagnostic,
    }
    write_json(out/'validation.json', report)
    print(f'RMS landmark error: {report["rms_error_mm"]:.6g} mm '
          f'({100*report["rms_error_fraction_diagonal"]:.4g}% of target diagonal)')
    print(f'Maximum landmark error: {report["maximum_error_mm"]:.6g} mm')
    print(out/'validation.json')


if __name__ == '__main__':
    main()
