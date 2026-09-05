#!/usr/bin/env python3
"""Audit original triangle identity and fitted zero poses in a completed run.

Optionally prepare SVROM distance fields and evaluate all exported neutral
poses. This can take several minutes; fields are cached per joint. Validation
does not certify anatomical landmark or patch-boundary accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import trimesh

from svrom_preparation.data import (Bone, inverse_rigid, sha256, transform_points,
                                     validate_rigid, write_json)
from svrom_preparation.surfaces import collision_check
from svrom.config import load_joint_config
from svrom.model import Pose


def validate_results(root, *, check_rom=False, backend='auto'):
    root = Path(root).resolve()
    metadata = json.loads((root/'run_metadata.json').read_text())
    report = json.loads((root/'report.json').read_text())
    bones = {}
    for spec, fingerprint in zip(metadata['manifest']['bones'], metadata['inputs']):
        if sha256(spec['mesh']) != fingerprint['mesh']:
            raise ValueError(f'input mesh changed: {spec["name"]}')
        bones[spec['name']] = Bone.load(spec['mesh'], name=spec['name'])
    rows = []
    for pair in report['pairs']:
        if not pair.get('directory'): continue
        directory = root/pair['directory']
        joint = json.loads((directory/'joint_report.json').read_text())
        fixed, moving = bones[pair['fixed']], bones[pair['moving']]
        matrix = np.asarray(joint['candidates'][joint['selected_candidate']]['moving_local_to_fixed_local'])
        validate_rigid(matrix)
        if not collision_check(fixed, moving, matrix)['verified']:
            raise ValueError(f'selected pose failed collision verification: {directory.name}')
        transforms = {s: np.asarray(joint['input_to_joint'][s]) for s in ('fixed', 'moving')}
        for value in transforms.values(): validate_rigid(value)
        decoded = (inverse_rigid(fixed.local_to_input) @ inverse_rigid(transforms['fixed'])
                   @ transforms['moving'] @ moving.local_to_input)
        np.testing.assert_allclose(decoded, matrix, atol=2e-12, rtol=0)
        count = 0
        with np.load(directory/'patch_labels.npz', allow_pickle=False) as archive:
            for interface in metadata['profile']['interfaces']:
                for side, bone in (('fixed', fixed), ('moving', moving)):
                    for kind in ('core', 'possible'):
                        key = f'{interface["name"]}__{side}__{kind}'
                        faces = archive[key+'_face_ids']
                        if not len(faces): continue
                        vertices = archive[key+'_original_vertex_ids']
                        patch = trimesh.load(directory/f'{interface["name"]}_{side}_{kind}.obj', process=False)
                        np.testing.assert_array_equal(vertices[np.asarray(patch.faces)], bone.mesh.faces[faces])
                        np.testing.assert_array_equal(patch.vertices, (bone.mesh.vertices+bone.origin)[vertices])
                        count += 1
        configurations = []
        for filename in pair['configurations']:
            config = load_joint_config(directory/filename)
            geometry = config.load_geometry()
            for side, bone, surface in (('fixed', fixed, geometry.fixed_bone),
                                        ('moving', moving, geometry.moving_bone)):
                # Whole-bone OBJ export discards only unreferenced vertices.
                referenced = np.unique(bone.mesh.faces)
                expected = transform_points((bone.mesh.vertices+bone.origin)[referenced], transforms[side])
                np.testing.assert_allclose(surface.vertices, expected, atol=2e-12, rtol=0)
            result = {'configuration': filename, 'geometry_roundtrip_verified': True}
            if check_rom:
                oracle = config.build_robust_oracle(backend=backend, cache_directory=directory/'audit_sdf')
                pose = oracle.evaluate_pose(Pose())
                result.update({'neutral_feasible': pose.feasible, 'collision': pose.collision,
                               'patch_passes': pose.patch_passes, 'metrics': pose.metrics})
            configurations.append(result)
        rows.append({'fixed': fixed.name, 'moving': moving.name,
                     'transform_error_max': float(np.max(np.abs(decoded-matrix))),
                     'patch_exports_checked': count, 'configurations': configurations})
        print(f'Checked {directory.name}', flush=True)
    return {'input_files_unchanged': len(bones), 'rom_checked': check_rom,
            'pair_status_counts': dict(Counter(p['status'] for p in report['pairs'])), 'joints': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--check-rom', action='store_true')
    parser.add_argument('--backend', choices=('auto', 'python', 'meshrom'), default='auto')
    args = parser.parse_args(argv)
    report = validate_results(args.results, check_rom=args.check_rom, backend=args.backend)
    output = args.results.resolve()/'export_validation.json'
    write_json(output, report)
    print(output)
    return 2 if any(c.get('neutral_feasible') is False for j in report['joints'] for c in j['configurations']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
