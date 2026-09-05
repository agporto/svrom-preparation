"""Numerical and behavioral checks for the separate preparation workflow."""
import json
import zipfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from svrom_preparation.data import (Bone, Landmarks, inverse_rigid, read_landmarks,
    rigid_matrix, transform_points, validate_rigid, write_landmarks)
from svrom_preparation.fitting import Candidate, PairFit, assemble_chain, fit_pair
from svrom_preparation.settings import ArticulationSettings
from svrom_preparation.seating import SeatingEvaluator
from svrom_preparation.surfaces import (Region, apposition_scores, collision_check,
    patch_ensemble, prepare_regions)
from svrom_preparation.transfer import Atlas, warp_landmarks
from svrom_preparation.workflow import export_joint, run_manifest, write_obj
from svrom.config import load_joint_config
from svrom.geometry import SurfaceMesh


PROFILE = {
    'minimum_landmarks': 5, 'anterior': [1], 'posterior': [2], 'dorsal': [3],
    'side_a': [4], 'side_b': [5],
    'regions': {'front': {'landmarks': [1], 'radius_fraction': 0.85},
                'back': {'landmarks': [2], 'radius_fraction': 0.85}},
    'interfaces': [{'name': 'centrum', 'fixed': 'back', 'moving': 'front'}],
}
BASE_LM = np.array([[0, 0, -.5], [0, 0, .5], [0, 1, 0], [-.8, 0, 0], [.8, 0, 0]])


def make_bone(tmp_path, name, *, rotation=None, origin=(0, 0, 0), subdivisions=0, extents=(2, 2, 1)):
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(subdivisions): mesh = mesh.subdivide()
    matrix = rigid_matrix(rotation, origin)
    vertices = transform_points(mesh.vertices, matrix)
    path = tmp_path/f'{name}.obj'
    write_obj(path, vertices, mesh.faces)
    bone = Bone.load(path)
    lm = Landmarks(transform_points(BASE_LM, matrix), tuple(f'p{i}' for i in range(5)), tuple(str(i+1) for i in range(5)))
    bone.set_landmarks(lm, PROFILE)
    prepare_regions(bone, PROFILE)
    return bone


def test_markups_coordinate_roundtrip_and_numeric_fcsv(tmp_path):
    path = tmp_path/'landmarks.fcsv'
    path.write_text('# CoordinateSystem = 1\n# columns = id,x,y,z,label\na,1,2,3,"facet, right"\n')
    lm = read_landmarks(path, coordinates='RAS')
    np.testing.assert_array_equal(lm.points, [[-1, -2, 3]])
    assert lm.labels == ('facet, right',)
    out = tmp_path/'landmarks.mrk.json'
    write_landmarks(out, lm)
    restored = read_landmarks(out, coordinates='LPS')
    np.testing.assert_array_equal(restored.points, [[1, 2, 3]])
    assert restored.ids == ('a',)


def test_rigid_frames_and_global_motion(tmp_path):
    a = make_bone(tmp_path, 'a')
    r = Rotation.from_euler('xyz', [25, -31, 72], degrees=True).as_matrix()
    t = np.array([300, -40, 500])
    b = make_bone(tmp_path, 'b', rotation=r, origin=t)
    assert np.linalg.det(a.frame[:3, :3]) == pytest.approx(1.)
    np.testing.assert_allclose(b.frame[:3, :3], r @ a.frame[:3, :3], atol=2e-13)
    assert a.length == pytest.approx(b.length)
    m = rigid_matrix(r, t)
    np.testing.assert_allclose(inverse_rigid(m) @ m, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError, match='proper rigid'):
        validate_rigid(np.diag([-1, 1, 1, 1]))


def test_tps_preserves_affine_landmarks_at_large_offsets():
    rng = np.random.default_rng(10)
    dense = rng.normal(size=(90, 3))+[1e5, -2e5, 3e5]
    points = rng.normal(size=(12, 3))+dense.mean(0)
    a = np.array([[1.1, .2, .1], [0, .7, -.1], [.1, .2, .9]])
    t = np.array([12, 35, -20])
    out = warp_landmarks(points, dense, dense @ a.T+t, max_controls=70)
    np.testing.assert_allclose(out, points @ a.T+t, rtol=0, atol=2e-9)


def test_atlas_converts_modes_and_catches_frame_mismatch(tmp_path):
    rng = np.random.default_rng(2)
    points = rng.normal(size=(12, 3))
    labels = tuple(str(i) for i in range(12))
    lm = Landmarks(points, labels, labels, 'LPS')
    write_landmarks(tmp_path/'dense.json', lm)
    write_landmarks(tmp_path/'sparse.json', lm)
    modes = rng.normal(size=(12, 3, 2))
    np.savez(tmp_path/'ssm.npz', mean_shape=points*[-1, -1, 1],
             modes=modes*np.array([-1, -1, 1])[None, :, None], eigenvalues=[2., 1.])
    (tmp_path/'manifest.json').write_text(json.dumps({'files': {'dense': 'dense.json', 'sparse': 'sparse.json', 'ssm': 'ssm.npz'}}))
    a = Atlas.load(tmp_path)
    np.testing.assert_array_equal(a.mean, points)
    np.testing.assert_array_equal(a.modes, modes)
    archive = tmp_path/'atlas.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in ('manifest.json', 'dense.json', 'sparse.json', 'ssm.npz'):
            z.write(tmp_path/name, f'atlas/{name}')
    from_zip = Atlas.load(archive)
    np.testing.assert_array_equal(from_zip.modes, modes)
    assert from_zip.fingerprint == a.fingerprint
    with pytest.raises(ValueError, match='disagree'):
        Atlas.load(tmp_path, ssm_coordinates='LPS')


def plane_receiver(z=.05):
    v = np.array([[-4., -4., z], [4., -4., z], [4., 4., z], [-4., 4., z]])
    faces = np.array([[0, 2, 1], [0, 3, 2]])
    mesh = SurfaceMesh.from_arrays(name='plane', vertices=v, faces=faces,
                                  vertex_normals=np.tile([0, 0, -1.], (4, 1)))
    region = Region('plane', np.arange(2), np.ones(2), np.ones(2), np.array([0]), 1.)
    return mesh, region


def test_triangle_queries_distance_facing_and_rigid_invariance():
    receiver, region = plane_receiver()
    points = np.array([[0., 0., 0.], [.2, -.3, 0.]])
    normals = np.tile([0., 0., 1.], (2, 1))
    scores = apposition_scores(points, normals, receiver, region, .05, .01, .25)
    np.testing.assert_allclose(scores, 1., atol=1e-12)
    assert np.min(np.linalg.norm(receiver.vertices, axis=1)) > 5
    np.testing.assert_array_equal(apposition_scores(points, -normals, receiver, region, .05, .01, .25), 0.)
    assert np.max(apposition_scores(points-[0, 0, .5], normals, receiver, region, .05, .01, .25)) < 1e-100
    m = rigid_matrix(Rotation.from_euler('xyz', [20, 50, -70], degrees=True).as_matrix(), [12, -5, 9])
    moved = SurfaceMesh.from_arrays(name='moved', vertices=transform_points(receiver.vertices, m),
                                    faces=receiver.faces, vertex_normals=receiver.vertex_normals @ m[:3, :3].T)
    np.testing.assert_allclose(apposition_scores(transform_points(points, m), normals @ m[:3, :3].T,
                                                moved, region, .05, .01, .25), scores, atol=1e-12)


def test_collision_checks_crossing_containment_and_open_mesh(tmp_path):
    a = make_bone(tmp_path, 'a', extents=(4, .2, .2))
    b = make_bone(tmp_path, 'b', extents=(.2, 4, .2))
    # Thin crossing beams have no vertex of either beam inside the other.
    check = collision_check(a, b, np.eye(4))
    assert check['intersections'] and not check['verified']
    small = make_bone(tmp_path, 'small', extents=(.1, .1, .1))
    check = collision_check(a, small, np.eye(4))
    assert check['containment'] and not check['verified']
    check = collision_check(a, b, rigid_matrix(translation=[0, 0, 2]))
    assert check['verified']
    b.watertight = False
    assert not collision_check(a, b, rigid_matrix(translation=[0, 0, 2]))['verified']


def test_coordinate_mismatch_and_open_mesh_never_certified(tmp_path):
    a = make_bone(tmp_path, 'a')
    b = make_bone(tmp_path, 'b', origin=[500, 0, 0])
    fit = fit_pair(a, b, PROFILE, ArticulationSettings())
    assert fit.status == 'coordinate_frame_mismatch' and not fit.candidates
    b.origin[:] = [0, 0, 1.1]
    b.watertight = False
    assert fit_pair(a, b, PROFILE, ArticulationSettings()).status == 'mesh_requires_review'


def test_known_tilt_recovery_and_fixed_patch_export(tmp_path):
    a = make_bone(tmp_path, 'a', subdivisions=3)
    tilt = Rotation.from_euler('x', 9., degrees=True).as_matrix()
    b = make_bone(tmp_path, 'b', rotation=tilt, origin=[0, 0, 1.12], subdivisions=3)
    settings = ArticulationSettings(gap_fractions=(.04,), gap_width_fraction=.03,
                                    max_evaluations=450, refine_candidates=2, sample_count=160,
                                    sdf_samples=48, retain_candidates=3)
    fit = fit_pair(a, b, PROFILE, settings)
    assert fit.candidates, fit.report
    c = fit.candidates[0]
    normal = c.matrix[:3, :3] @ tilt @ np.array([0., 0., -1.])
    angle = np.rad2deg(np.arccos(np.clip(-normal[2], -1, 1)))
    assert angle < 2.0, (angle, fit.report)
    assert c.collision['verified']
    patches = patch_ensemble(a, b, PROFILE['interfaces'], fit.candidates, settings)
    for side in ('fixed', 'moving'):
        p = patches['centrum'][side]
        assert len(p['possible_face_ids'])
        assert set(p['core_face_ids']) <= set(p['possible_face_ids'])
    report = export_joint(tmp_path/'joint', a, b, fit, 0, settings, PROFILE)
    assert 'joint_possible.yaml' in report['configurations']
    config = load_joint_config(tmp_path/'joint/joint_possible.yaml')
    geometry = config.load_geometry()
    # Independently check that zero-pose export retains the fitted relation.
    f = report['input_to_joint']['fixed']
    expected = transform_points(transform_points(b.mesh.vertices, c.matrix)+a.origin, f)
    np.testing.assert_allclose(geometry.moving_bone.vertices, expected, atol=2e-13)
    archive = np.load(tmp_path/'joint/patch_labels.npz')
    ids = archive['centrum__fixed__possible_face_ids']
    assert ids.min() >= 0 and ids.max() < len(a.mesh.faces)


def test_curved_chain_retains_relative_rotations(tmp_path):
    bones = [make_bone(tmp_path, str(i), extents=(.2, .2, .2)) for i in range(4)]
    r = Rotation.from_euler('y', 12., degrees=True).as_matrix()
    m = rigid_matrix(r, [0, 0, 1.])
    c = Candidate(m, .02, 1., np.ones(1), {'verified': True}, 'constructed')
    fits = [PairFit(bones[i].name, bones[i+1].name, 'verified_geometric_reference', [c], {}) for i in range(3)]
    segments = assemble_chain(bones, fits, ArticulationSettings())
    assert len(segments) == 1 and segments[0]['status'] == 'verified'
    frames = segments[0]['local_to_world']
    for f, g in zip(frames[:-1], frames[1:]): np.testing.assert_allclose(inverse_rigid(f) @ g, m, atol=1e-14)
    centers = np.stack([t[:3, 3] for t in frames])
    assert np.linalg.norm(np.cross(centers[1]-centers[0], centers[2]-centers[1])) > .1


def test_intrinsic_curvature_with_parallel_landmark_frames(tmp_path):
    """The optimum must follow oblique surfaces, not zero landmark angles."""
    fixed = make_bone(tmp_path, 'fixed', subdivisions=3)
    mesh = trimesh.creation.box(extents=(2, 2, 1))
    for _ in range(3): mesh = mesh.subdivide()
    angle = np.deg2rad(10.)
    mesh.vertices[:, 2] += np.tan(angle)*mesh.vertices[:, 1]
    offset = np.array([0., 0., 1.15])
    path = tmp_path/'oblique.obj'
    write_obj(path, mesh.vertices+offset, mesh.faces)
    moving = Bone.load(path)
    labels = tuple(str(i) for i in range(5))
    moving.set_landmarks(Landmarks(BASE_LM+offset, labels, labels), PROFILE)
    prepare_regions(moving, PROFILE)
    np.testing.assert_allclose(fixed.frame[:3, :3], moving.frame[:3, :3], atol=1e-14)
    settings = ArticulationSettings(gap_fractions=(.04,), gap_width_fraction=.03,
                                   max_evaluations=500, refine_candidates=3)
    fit = fit_pair(fixed, moving, PROFILE, settings)
    assert fit.candidates, fit.report
    matrix = fit.candidates[0].matrix
    actual = matrix[:3, :3] @ np.array([0., np.sin(angle), -np.cos(angle)])
    error = np.rad2deg(np.arccos(np.clip(-actual[2], -1., 1.)))
    assert error < .2, (error, fit.report)
    assert np.rad2deg(Rotation.from_matrix(matrix[:3, :3]).magnitude()) > 8.
    assert fit.candidates[0].collision['verified']


@pytest.mark.parametrize('kwargs', [{'gap_fractions': [0.]}, {'normal_cosine': 1.},
                                   {'sample_count': 1.2}, {'gap_width_fraction': float('nan')}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError): ArticulationSettings(**kwargs)


def test_resume_rejects_changed_inputs_and_preserves_originals(tmp_path, monkeypatch):
    a = make_bone(tmp_path, 'a')
    b = make_bone(tmp_path, 'b', origin=[500, 0, 0])
    import yaml
    entries = []
    for bone in (a, b):
        lm_path = tmp_path/f'{bone.name}.mrk.json'
        write_landmarks(lm_path, bone.landmarks.moved(bone.local_to_input))
        entries.append({'name': bone.name, 'mesh': str(bone.path), 'landmarks': str(lm_path)})
    manifest = tmp_path/'run.yaml'
    manifest.write_text(yaml.safe_dump({'schema_version': 1, 'bones': entries, 'guide_profile': PROFILE}))
    original = a.path.read_bytes()
    report = run_manifest(manifest, tmp_path/'out', progress=lambda _: None)
    assert report['pairs'][0]['status'] == 'coordinate_frame_mismatch'
    assert report['status'] == 'partial_result_requires_review'
    assert a.path.read_bytes() == original
    from svrom_preparation import workflow
    metadata = json.loads((tmp_path/'out/run_metadata.json').read_text())
    assert metadata['preparation_version'] == workflow.__version__
    assert metadata['svrom_version'] == workflow.svrom_version
    for attribute, message in (('__version__', 'preparation version changed'),
                               ('svrom_version', 'SVROM version changed')):
        with monkeypatch.context() as patch:
            patch.setattr(workflow, attribute, '999.0.0')
            with pytest.raises(ValueError, match=message):
                run_manifest(manifest, tmp_path/'out', resume=True, progress=lambda _: None)
    entries[1]['name'] = 'renamed'
    manifest.write_text(yaml.safe_dump({'schema_version': 1, 'bones': entries, 'guide_profile': PROFILE}))
    with pytest.raises(ValueError, match='changed'):
        run_manifest(manifest, tmp_path/'out', resume=True, progress=lambda _: None)


def test_seating_rejects_edge_contact_with_the_same_minimum_gap(tmp_path):
    a = make_bone(tmp_path, 'seat_a', subdivisions=3)
    b = make_bone(tmp_path, 'seat_b', origin=[0, 0, 1.04], subdivisions=3)
    settings = ArticulationSettings()
    evaluator = SeatingEvaluator(a, b, PROFILE, settings)
    centered = rigid_matrix(translation=b.origin-a.origin)
    edge = centered.copy(); edge[0, 3] += 1.4
    good_energy, good = evaluator.evaluate(centered, .04, full=True, metrics=True)
    bad_energy, bad = evaluator.evaluate(edge, .04, full=True, metrics=True)
    # Both assemblies are collision-free and parallel with a 0.04 mm plane gap.
    assert collision_check(a, b, centered)['verified']
    assert collision_check(a, b, edge)['verified']
    assert good['passes']
    assert not bad['passes']
    assert bad_energy > good_energy + 1.
    assert any('seating offset' in reason for reason in bad['review_reasons'])


def test_seating_objective_and_clearance_are_rigid_invariant(tmp_path):
    a = make_bone(tmp_path, 'inv_a', subdivisions=2)
    b = make_bone(tmp_path, 'inv_b', origin=[0, 0, 1.08], subdivisions=2)
    settings = ArticulationSettings()
    q = SeatingEvaluator(a, b, PROFILE, settings)
    matrix = rigid_matrix(translation=b.origin-a.origin)
    expected = q.evaluate(matrix, .04, full=True)
    constraints = q.constraints(matrix)
    r = Rotation.from_euler('xyz', [27, -38, 51], degrees=True).as_matrix()
    t = np.array([1000, -350, 500])
    c = make_bone(tmp_path, 'inv_c', rotation=r, origin=t, subdivisions=2)
    d = make_bone(tmp_path, 'inv_d', rotation=r, origin=r @ np.array([0, 0, 1.08])+t, subdivisions=2)
    other = SeatingEvaluator(c, d, PROFILE, settings)
    moved = rigid_matrix(translation=d.origin-c.origin)
    assert other.evaluate(moved, .04, full=True) == pytest.approx(expected, abs=2e-10)
    np.testing.assert_allclose(other.constraints(moved), constraints, atol=2e-10, rtol=0)


def test_intersection_witnesses_close_a_sampling_blind_spot(tmp_path):
    a = make_bone(tmp_path, 'witness_a', extents=(4, .2, .2))
    b = make_bone(tmp_path, 'witness_b', extents=(.2, 4, .2))
    q = SeatingEvaluator(a, b, PROFILE, ArticulationSettings())
    assert collision_check(a, b, np.eye(4))['intersections']
    before = {name: len(points) for name, points in q.collision_points.items()}
    count = q.add_collision_witnesses(np.eye(4))
    assert count > 0
    assert all(len(points) == before[name]+count for name, points in q.collision_points.items())
    assert q.constraints(np.eye(4)).min() < -.0005


def test_apposition_objective_remains_available(tmp_path):
    a = make_bone(tmp_path, 'legacy_a', subdivisions=2)
    b = make_bone(tmp_path, 'legacy_b', origin=[0, 0, 1.04], subdivisions=2)
    fit = fit_pair(a, b, PROFILE, ArticulationSettings(objective='apposition', gap_fractions=(.04,)))
    assert fit.candidates
    assert all(not candidate.seating for candidate in fit.candidates)
