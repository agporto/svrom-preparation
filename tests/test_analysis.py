"""Dimensional correctness and downstream invariance of analysis templates."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from svrom.config import load_joint_config
from svrom.model import Pose
from svrom_preparation.analysis import AnalysisTemplate, ROBUST_LENGTH_FIELDS
from svrom_preparation.data import Bone, Landmarks, rigid_matrix, write_landmarks
from svrom_preparation.fitting import Candidate, PairFit
from svrom_preparation.settings import ArticulationSettings
from svrom_preparation.surfaces import prepare_regions
from svrom_preparation.workflow import export_joint, run_manifest, write_obj

from test_articulation import BASE_LM, PROFILE, make_bone


def template_files(tmp_path, *, unit='cm', multiplier=1.):
    """Reference lengths: 0.2 and 0.3 cm; a deliberately asymmetric grid."""
    raw = {
        'schema_version': 2, 'name': 'reference', 'units': unit,
        'search': {'rotations': {a: {'start': -2., 'stop': 2., 'step': 1.}
                                 for a in ('rx_deg', 'ry_deg', 'rz_deg')},
                   'translations': {'tx': {'values': [-.01, .003, .02]},
                                    'ty': [-.01, 0., .01],
                                    'tz': {'start': -.01, 'stop': .01, 'step': .01}}},
        'robust': {'penetration_tolerance': .001, 'apposition_max_gap': .025,
                   'maximum_quantile_depth': .003, 'sdf_voxel_size': .007,
                   'sdf_smoothing_sigma_mesh_units': .002,
                   'sdf_longest_axis_samples': 32,
                   'apposition_method': 'directional',
                   'minimum_bidirectional_coverage': .05},
        'maya_compat': {'ray_length': 5., 'ray_length_scale': 100.},
        # Deliberately missing paths: these must not be loaded or transferred.
        'meshes': {'fixed': {'bone': 'snake_only.ply'}},
        'coordinate_frames': {'fixed': {'name': 'snake_frame'}},
        'patch_pairs': [{'name': 'snake_patch'}],
        'reference_labels': 'historical_snake_labels.csv',
        'source_provenance': {'historical_live_runtime_oracle': True},
    }
    from svrom.config import AxisGrid
    for axis, grid in raw['search']['translations'].items():
        raw['search']['translations'][axis] = {'values': [v*multiplier for v in AxisGrid.from_value(grid, name=axis).values]}
    for key in ROBUST_LENGTH_FIELDS:
        raw['robust'][key] *= multiplier
    raw['maya_compat']['ray_length'] *= multiplier
    path = tmp_path/f'template_{unit}.yaml'
    path.write_text(yaml.safe_dump(raw))
    points = {f'{side}_landmarks_local': {'cotyle': [0., 0., 0.], 'condyle': [0., 0., length*multiplier]}
              for side, length in (('fixed', .2), ('moving', .3))}
    ref = tmp_path/f'frames_{unit}.json'
    ref.write_text(json.dumps(points))
    return {'template': path, 'reference_frame_landmarks': ref}


def test_units_and_body_size_are_separate_and_no_settings_leak(tmp_path):
    spec = template_files(tmp_path)
    template = AnalysisTemplate.load(spec)
    raw_before = deepcopy(template)
    # The same physical 2/3 mm pair requires only cm -> mm conversion.
    output, audit = template.for_joint(2., 3.)
    assert audit['physical_size_ratio'] == pytest.approx(1.)
    assert audit['unit_conversion_to_mm'] == 10.
    assert audit['numeric_length_multiplier'] == pytest.approx(10.)
    assert output['robust']['penetration_tolerance'] == pytest.approx(.01)
    assert output['robust']['apposition_max_gap'] == pytest.approx(.25)
    np.testing.assert_allclose(output['search']['translations']['tx']['values'], [-.1, .03, .2])
    assert not audit['zero_translation_in_lattice']
    assert audit['pose_count'] == 125*27
    twice, doubled = template.for_joint(4., 6.)
    assert doubled['physical_size_ratio'] == pytest.approx(2.)
    for key, value in output['robust'].items():
        assert twice['robust'][key] == (pytest.approx(2*value) if key in ROBUST_LENGTH_FIELDS else value)
    assert twice['maya_compat']['ray_length'] == pytest.approx(100.)
    assert twice['maya_compat']['ray_length_scale'] == 100.
    assert twice['search']['rotations'] == template.search['rotations']
    assert set(twice) == {'search', 'robust', 'maya_compat'}
    assert template == raw_before
    # Re-expressing the complete source in mm gives identical target settings.
    in_mm = AnalysisTemplate.load(template_files(tmp_path, unit='mm', multiplier=10.))
    same, mm_audit = in_mm.for_joint(2., 3.)
    assert mm_audit['physical_size_ratio'] == pytest.approx(1.)
    for axis in ('tx', 'ty', 'tz'):
        np.testing.assert_allclose(same['search']['translations'][axis]['values'], output['search']['translations'][axis]['values'], atol=1e-16)
    for key in output['robust']:
        assert same['robust'][key] == output['robust'][key]


def test_reference_length_is_rigid_invariant_and_defaults_scale(tmp_path):
    spec = template_files(tmp_path)
    expected = AnalysisTemplate.load(spec)
    ref = json.loads(spec['reference_frame_landmarks'].read_text())
    rotation = Rotation.from_euler('xyz', [21, 13, -44], degrees=True).as_matrix()
    for landmarks in ref.values():
        for name, point in landmarks.items():
            landmarks[name] = (rotation @ point + [200., -100., 400.]).tolist()
    spec['reference_frame_landmarks'].write_text(json.dumps(ref))
    actual = AnalysisTemplate.load(spec)
    for side in ('fixed', 'moving'):
        assert actual.provenance['reference_centrum_lengths'][side] == pytest.approx(expected.provenance['reference_centrum_lengths'][side], abs=1e-13)
    raw = yaml.safe_load(spec['template'].read_text())
    del raw['robust']['apposition_max_gap']
    del raw['robust']['sdf_voxel_size']
    spec['template'].write_text(yaml.safe_dump(raw))
    analysis = AnalysisTemplate.load(spec)
    settings, _ = analysis.for_joint(2., 3.)
    assert settings['robust']['apposition_max_gap'] == pytest.approx(.3)  # SVROM default .03 cm
    assert settings['robust']['sdf_voxel_size'] is None


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf'), True])
def test_bad_reference_and_target_lengths_rejected(tmp_path, invalid):
    spec = template_files(tmp_path)
    template = AnalysisTemplate.load(spec)
    with pytest.raises(ValueError, match='finite positive'):
        template.for_joint(invalid, 1.)
    spec.pop('reference_frame_landmarks')
    spec['reference_centrum_lengths'] = {'fixed': invalid, 'moving': 1.}
    with pytest.raises(ValueError, match='finite positive'):
        AnalysisTemplate.load(spec)


def scaled_joint(tmp_path, factor):
    bones = []
    for side, offset in (('fixed', np.zeros(3)), ('moving', np.array([0., 0., 1.04]))):
        mesh = trimesh.creation.box(extents=(2., 2., 1.))
        for _ in range(2):
            mesh = mesh.subdivide()
        if side == 'moving':
            # Give the two opposed planes matching centroid quadrature for the
            # directional (centroid-based) SVROM profile, with outward winding.
            mesh.vertices[:, 2] *= -1
            mesh.faces = mesh.faces[:, [0, 2, 1]]
        path = tmp_path/f'{side}_{factor}.obj'
        write_obj(path, (mesh.vertices+offset)*factor, mesh.faces)
        bone = Bone.load(path)
        labels = tuple(str(i) for i in range(5))
        bone.set_landmarks(Landmarks((BASE_LM+offset)*factor, labels, labels), PROFILE)
        prepare_regions(bone, PROFILE)
        bones.append(bone)
    matrix = rigid_matrix(translation=bones[1].origin-bones[0].origin)
    candidate = Candidate(matrix, .04*factor, 0., np.ones(1), {'verified': True}, 'constructed')
    fit = PairFit(bones[0].name, bones[1].name, 'verified_geometric_reference', [candidate], {})
    return bones, fit


def test_export_preserves_geometry_and_robust_decisions_under_scaling(tmp_path):
    template = AnalysisTemplate.load(template_files(tmp_path))
    settings = ArticulationSettings()
    evaluations = []
    for factor in (1., .1, 10.):
        (fixed, moving), fit = scaled_joint(tmp_path, factor)
        destination = tmp_path/f'joint_{factor}'
        export_joint(destination, fixed, moving, fit, 0, settings, PROFILE, analysis_template=template)
        config = load_joint_config(destination/'joint_possible.yaml')
        assert config.reference_labels_path is None
        assert [p.name for p in config.patch_path_pairs] == ['centrum']
        assert config.fixed_frame.name == 'fitted_fixed'
        assert 'historical_live_runtime_oracle' not in config.source_provenance
        if factor == 1.:
            old = tmp_path/'inspection_only'
            export_joint(old, fixed, moving, fit, 0, settings, PROFILE)
            before = load_joint_config(old/'joint_possible.yaml').load_geometry()
            after = config.load_geometry()
            np.testing.assert_array_equal(before.fixed_bone.vertices, after.fixed_bone.vertices)
            np.testing.assert_array_equal(before.moving_bone.vertices, after.moving_bone.vertices)
            for path in old.glob('*.obj'):
                assert path.read_bytes() == (destination/path.name).read_bytes()
        oracle = config.build_robust_oracle(backend='python')
        # A valid reference, collision, loss of contact, and rotated pose.
        poses = (Pose(), Pose(tx=-.2*factor), Pose(tx=.3*factor), Pose(rz_deg=2., tx=.01*factor))
        evaluations.append([oracle.evaluate_pose(pose) for pose in poses])
    assert evaluations[0][0].feasible
    assert evaluations[0][1].collision
    assert not evaluations[0][2].feasible and not evaluations[0][2].collision
    for factor, results in zip((.1, 10.), evaluations[1:]):
        for expected, result in zip(evaluations[0], results):
            assert (result.collision, result.overlap, result.patch_passes) == (expected.collision, expected.overlap, expected.patch_passes)
            assert result.metrics.keys() == expected.metrics.keys()
            for key in result.metrics:
                value = result.metrics[key]
                if 'depth' in key or 'signed_distance' in key:
                    value /= factor
                assert value == pytest.approx(expected.metrics[key], abs=2e-6), key


def test_template_and_reference_edits_invalidate_resume(tmp_path):
    spec = template_files(tmp_path)
    bone = make_bone(tmp_path, 'single')
    landmark_path = tmp_path/'single.mrk.json'
    write_landmarks(landmark_path, bone.landmarks.moved(bone.local_to_input))
    manifest = tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'schema_version': 1, 'guide_profile': PROFILE,
        'analysis': {key: str(path.name) for key, path in spec.items()},
        'bones': [{'name': bone.name, 'mesh': str(bone.path), 'landmarks': str(landmark_path)}]}))
    for key in ('template', 'reference_frame_landmarks'):
        out = tmp_path/key
        run_manifest(manifest, out, transfer_only=True, progress=lambda _: None)
        original = spec[key].read_text()
        spec[key].write_text(original+'\n')
        with pytest.raises(ValueError, match='inputs/settings changed'):
            run_manifest(manifest, out, resume=True, transfer_only=True, progress=lambda _: None)
        spec[key].write_text(original)
