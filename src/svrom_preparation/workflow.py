"""Ordered specimen workflow, resumable transfer, review reports, SVROM export."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import yaml

from ._version import __version__
from svrom._version import __version__ as svrom_version
from .analysis import AnalysisTemplate, resolve_analysis_spec
from .data import (Bone, inverse_rigid, read_landmarks, relative_angles, rigid_matrix,
                   safe_name, sha256, transform_points, write_json, write_landmarks)
from .fitting import PairFit, assemble_chain, fit_pair
from .settings import ArticulationSettings, TransferSettings, vertebra28_profile
from .surfaces import patch_ensemble, prepare_regions
from .transfer import Atlas, transfer_landmarks


def load_manifest(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('articulation manifest schema_version must be 1')
    bones = data.get('bones', [])
    if not bones: raise ValueError('manifest must contain an ordered bones list')
    names = [safe_name(b['name']) for b in bones]
    if len(set(names)) != len(names): raise ValueError('bone names must be unique after filename normalization')
    if str(data.get('units', 'mm')).lower() not in {'mm', 'millimeter', 'millimetre'}:
        raise ValueError('this workflow requires meshes in millimeters; rescale explicitly before running')
    for b, name in zip(bones, names):
        b['name'] = name
        for key in ('mesh', 'landmarks'):
            if b.get(key):
                p = Path(b[key])
                b[key] = (path.parent/p).resolve() if not p.is_absolute() else p.resolve()
                if not b[key].is_file(): raise FileNotFoundError(b[key])
    atlas = data.get('atlas')
    if atlas:
        p = Path(atlas)
        data['atlas'] = (path.parent/p).resolve() if not p.is_absolute() else p.resolve()
    if not atlas and any(not b.get('landmarks') for b in bones):
        raise ValueError('provide an atlas or landmark files for every bone')
    if data.get('analysis') is not None:
        data['analysis'] = resolve_analysis_spec(data['analysis'], path.parent)
    profile = data.get('guide_profile', 'vertebra28')
    if isinstance(profile, str):
        if profile != 'vertebra28': raise ValueError('unknown guide profile')
        profile = vertebra28_profile()
    if not profile.get('interfaces'): raise ValueError('guide profile needs at least one interface')
    names = [x['name'] for x in profile['interfaces']]
    if len(set(names)) != len(names): raise ValueError('interface names must be unique')
    for name in list(profile['regions'])+names:
        if safe_name(name) != name: raise ValueError('region/interface names must be filename-safe')
    for i in profile['interfaces']:
        if i['fixed'] not in profile['regions'] or i['moving'] not in profile['regions']:
            raise ValueError('interface references an undefined guide region')
    return data, profile, TransferSettings(**data.get('transfer', {})), ArticulationSettings(**data.get('articulation', {}))


def write_obj(path, vertices, faces):
    """Compact OBJ with 17-digit coordinates; face identity is saved separately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.unique(np.asarray(faces).ravel())
    lookup = np.full(len(vertices), -1, dtype=int)
    lookup[ids] = np.arange(len(ids))
    with path.open('w') as f:
        f.write('# SVROM articulation; coordinates retained at float64 precision\n')
        for p in np.asarray(vertices)[ids]: f.write('v '+' '.join(format(float(x), '.17g') for x in p)+'\n')
        for face in lookup[faces]+1: f.write('f '+' '.join(map(str, face))+'\n')
    return ids


def estimate_joint_origin(fixed, moving, matrix, profile):
    """Estimate a coordinate origin from posterior centrum geometry.

    A sphere fit is accepted only for a well-conditioned, approximately
    spherical region of plausible size. Otherwise use the midpoint of the
    two interface anchors and label it as a coordinate convention. Neither
    estimate is a measured instantaneous axis/center of rotation.
    """
    posterior = fixed.landmarks.center(profile['posterior'])
    anterior = transform_points(moving.landmarks.center(profile['anterior'])[None], matrix)[0]
    fallback = (posterior+anterior)/2
    info = {'method': 'interface_anchor_midpoint', 'biomechanical_rotation_center_verified': False}
    spec = next((x for x in profile['interfaces'] if x['name'] == 'centrum'), None)
    if spec is None: return fallback, info
    region = fixed.regions[spec['fixed']]
    points = fixed.mesh.face_centroids[region.face_ids]
    origin = np.average(points, axis=0, weights=fixed.mesh.face_areas[region.face_ids])
    p = (points-origin)/fixed.length
    a = np.c_[2*p, np.ones(len(p))]
    if len(p) < 8 or np.linalg.cond(a) > 100: return fallback, info
    solution, _, rank, _ = np.linalg.lstsq(a, np.sum(p*p, axis=1), rcond=None)
    radius2 = float(solution[3]+solution[:3] @ solution[:3])
    if rank < 4 or radius2 <= 0: return fallback, info
    radius = float(np.sqrt(radius2)*fixed.length)
    center = solution[:3]*fixed.length+origin
    residual = float(np.sqrt(np.mean((np.linalg.norm(points-center, axis=1)-radius)**2)))
    info.update({'sphere_relative_rms': residual/radius, 'sphere_radius': radius})
    if (0.08 < radius/fixed.length < 0.8 and residual/radius < 0.08
            and np.linalg.norm(center-posterior) < fixed.length):
        info['method'] = 'posterior_centrum_sphere_fit'
        return center, info
    return fallback, info


def export_joint(directory, fixed, moving, fit, selected, settings, profile, *, analysis_template=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = fit.candidates[selected]
    patches = patch_ensemble(fixed, moving, profile['interfaces'], fit.candidates, settings)
    origin, origin_info = estimate_joint_origin(fixed, moving, candidate.matrix, profile)
    r = fixed.frame[:3, :3].T
    fixed_local_to_joint = rigid_matrix(r, -r @ origin)
    matrices = {
        'fixed': fixed_local_to_joint @ inverse_rigid(fixed.local_to_input),
        'moving': fixed_local_to_joint @ candidate.matrix @ inverse_rigid(moving.local_to_input),
    }
    archive, patch_report, paths = {}, {}, {'core': [], 'possible': []}
    for interface in profile['interfaces']:
        name = interface['name']
        patch_report[name] = {}
        for side, bone in (('fixed', fixed), ('moving', moving)):
            value = patches[name][side]
            patch_report[name][side] = {k: v for k, v in value.items() if k.endswith('_area')}
            for k in ('mean_score', 'support_frequency', 'core_face_ids', 'possible_face_ids'):
                archive[f'{name}__{side}__{k}'] = value[k]
            for kind in ('core', 'possible'):
                ids = value[f'{kind}_face_ids']
                if len(ids):
                    vertex_ids = write_obj(directory/f'{name}_{side}_{kind}.obj', bone.mesh.vertices+bone.origin, bone.mesh.faces[ids])
                    archive[f'{name}__{side}__{kind}_original_vertex_ids'] = vertex_ids
        for kind in ('core', 'possible'):
            if all(len(patches[name][s][f'{kind}_face_ids']) for s in ('fixed', 'moving')):
                paths[kind].append({'name': name, 'fixed': f'{name}_fixed_{kind}.obj', 'moving': f'{name}_moving_{kind}.obj'})
    np.savez_compressed(directory/'patch_labels.npz', **archive)
    # Retain input coordinates. Distinct role transforms preserve the fitted
    # articulation exactly at zero relative pose; no independent frame reset.
    for side, bone in (('fixed', fixed), ('moving', moving)):
        write_obj(directory/f'{side}_bone.obj', bone.mesh.vertices+bone.origin, bone.mesh.faces)
    scale = (fixed.length+moving.length)/2
    config = {
        'schema_version': 2, 'name': f'{fixed.name}__{moving.name}', 'units': 'millimeter',
        'neutral_articulation': {'convention': 'align_frames', 'fixed_frame': 'fitted_fixed', 'moving_frame': 'fitted_moving'},
        'coordinate_frames': {s: {'name': f'fitted_{s}', 'input_to_joint_matrix': m.tolist()} for s, m in matrices.items()},
        'meshes': {s: {'bone': f'{s}_bone.obj', 'coordinate_state': 'raw_scan'} for s in ('fixed', 'moving')},
        # Inspection defaults apply only when no analysis template is supplied.
        'search': {'rotations': {a: [0.] for a in ('rx_deg', 'ry_deg', 'rz_deg')},
                   'translations': {a: [0.] for a in ('tx', 'ty', 'tz')}},
        'robust': {'penetration_tolerance': 0.005*scale, 'maximum_penetrating_area_fraction': 0.002,
                   'sdf_longest_axis_samples': 160, 'sdf_smoothing_sigma_voxels': 0.,
                   'apposition_method': 'directional_surface', 'apposition_normal_method': 'area_weighted_vertex',
                   'apposition_max_gap': candidate.gap+2*settings.gap_width_fraction*scale,
                   'apposition_max_normal_angle_deg': 80., 'apposition_sender_cone_angle_deg': 75.,
                   'minimum_bidirectional_coverage': 0.01},
        'source_provenance': {'preannotation': True, 'requires_anatomical_review': True,
                              'neutral_definition': 'geometric reference articulation',
                              'profile': profile, 'joint_origin': origin_info,
                              'input_hashes': {s: sha256(b.path) if b.path else None for s, b in (('fixed', fixed), ('moving', moving))}},
    }
    analysis_audit = None
    if analysis_template is not None:
        analysis_settings, analysis_audit = analysis_template.for_joint(fixed.length, moving.length)
        config.update(analysis_settings)
        config['source_provenance']['analysis_template'] = analysis_audit
    configurations = []
    for kind in ('core', 'possible'):
        if len(paths[kind]) == len(profile['interfaces']):
            config['patch_pairs'] = paths[kind]
            filename = f'joint_{kind}.yaml'
            (directory/filename).write_text(yaml.safe_dump(config, sort_keys=False))
            configurations.append(filename)
    report = {'status': fit.status, 'selected_candidate': selected,
              'selected_relative_angles_deg': relative_angles(fixed, moving, candidate.matrix),
              'joint_origin_fixed_local': origin, 'origin_diagnostics': origin_info,
              'input_to_joint': matrices, 'patches': patch_report, 'configurations': configurations,
              'analysis_template': analysis_audit,
              'patch_interpretation': 'support frequency over sampled articulations, not calibrated anatomical probability',
              'candidates': [c.as_dict() for c in fit.candidates], 'fit': fit.report}
    if not configurations: report['annotation_status'] = 'incomplete_patch_support_requires_review'
    else: report['annotation_status'] = 'preannotation_requires_anatomical_review'
    write_json(directory/'joint_report.json', report)
    return report


def run_manifest(manifest_path, output_directory, *, resume=False, transfer_only=False, progress=print):
    started = time.perf_counter()
    data, profile, transfer_settings, settings = load_manifest(manifest_path)
    analysis_template = AnalysisTemplate.load(data.get('analysis'))
    out = Path(output_directory).resolve()
    coordinates = data.get('mesh_coordinate_system', 'LPS')
    atlas = Atlas.load(data['atlas'], coordinates=coordinates,
                       ssm_coordinates=data.get('ssm_coordinate_system', 'RAS')) if data.get('atlas') else None
    fingerprints = [{k: sha256(b[k]) for k in ('mesh', 'landmarks') if b.get(k)} for b in data['bones']]
    identity = {'manifest': data, 'profile': profile, 'inputs': fingerprints, 'atlas': atlas.fingerprint if atlas else None}
    if analysis_template is not None:
        identity['analysis_template'] = analysis_template.provenance
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    meta_path = out/'run_metadata.json'
    if out.exists() and any(out.iterdir()):
        if not resume or not meta_path.exists(): raise FileExistsError('use a fresh output directory or --resume for an existing matching run')
        previous = json.loads(meta_path.read_text())
        if previous['input_signature'] != signature:
            raise ValueError('inputs/settings changed; choose a fresh output directory to avoid stale exports')
        if previous.get('preparation_version') != __version__:
            raise ValueError('preparation version changed; choose a fresh output directory')
        if previous.get('svrom_version') != svrom_version:
            raise ValueError('SVROM version changed; choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, {'input_signature': signature, 'preparation_version': __version__,
                          'svrom_version': svrom_version, 'python': platform.python_version(),
                          'manifest': data, 'profile': profile, 'articulation_settings': settings.as_dict(),
                          'transfer_settings': asdict(transfer_settings), 'inputs': fingerprints,
                          'analysis_template': analysis_template.provenance if analysis_template else None})
    bones, bone_reports = [], []
    for index, spec in enumerate(data['bones']):
        progress(f'[{index+1}/{len(data["bones"])}] Landmarks: {spec["name"]}')
        bone = None
        try:
            bone = Bone.load(spec['mesh'], name=spec['name'])
            destination = out/'landmarks'/f'{bone.name}.mrk.json'
            report_path = out/'landmarks'/f'{bone.name}.json'
            if spec.get('landmarks'):
                lm = read_landmarks(spec['landmarks'], coordinates=coordinates)
                projection = bone.mesh.project_surface(lm.points-bone.origin)
                report = {'backend': 'provided_landmarks', 'landmark_projection_distances': projection.distance,
                          'needs_review': bool(projection.distance.max()/bone.mesh.diagonal > transfer_settings.maximum_projection_fraction)}
            elif resume and destination.exists() and report_path.exists():
                lm, report = read_landmarks(destination, coordinates=coordinates), json.loads(report_path.read_text())
            else:
                lm, report = transfer_landmarks(atlas, bone, transfer_settings)
            bone.transfer_report = report
            bone.set_landmarks(lm, profile)
            prepare_regions(bone, profile)
            write_landmarks(destination, lm)
            write_json(report_path, report)
            bone_reports.append({'name': bone.name, 'status': 'landmarks_ready', 'watertight': bone.watertight,
                                 'winding_consistent': bone.winding_consistent, 'length_scale': bone.length,
                                 'reoriented_faces': bone.reoriented_faces,
                                 'transfer_needs_review': report.get('needs_review', False)})
        except (ValueError, RuntimeError, OSError) as exc:
            bone_reports.append({'name': spec['name'], 'status': 'input_or_transfer_error', 'error': str(exc)})
        bones.append(bone)
        write_json(out/'progress.json', {'bones': bone_reports})
    report = {'bones': bone_reports, 'pairs': [], 'segments': [],
              'status': 'transfer_complete' if all(x['status'] == 'landmarks_ready' for x in bone_reports)
                        else 'partial_result_requires_review'}
    if not transfer_only:
        fits = []
        for i in range(len(bones)-1):
            a, b = bones[i:i+2]
            progress(f'[{i+1}/{len(bones)-1}] Articulation: {data["bones"][i]["name"]} / {data["bones"][i+1]["name"]}')
            if any(x is None or x.frame is None or not x.regions for x in (a, b)):
                fit = PairFit(data['bones'][i]['name'], data['bones'][i+1]['name'], 'input_or_transfer_error', [], {})
            else:
                fit = fit_pair(a, b, profile, settings)
            fits.append(fit)
            progress(f'  {fit.status}; {len(fit.candidates)} retained candidates')
            write_json(out/'progress.json', {'bones': bone_reports,
                                            'pairs': [{'fixed': f.fixed, 'moving': f.moving, 'status': f.status, 'report': f.report} for f in fits]})
        segments = assemble_chain(bones, fits, settings) if all(b is not None for b in bones) else []
        selections = {}
        world_transforms = {}
        for seg in segments:
            if seg['status'] != 'verified': continue
            for offset, t in enumerate(seg['local_to_world']): world_transforms[seg['start']+offset] = t
            for offset, k in enumerate(seg['candidate_indices']): selections[seg['start']+offset] = k
        for i, fit in enumerate(fits):
            entry = {'fixed': fit.fixed, 'moving': fit.moving, 'status': fit.status, 'fit': fit.report}
            if fit.candidates:
                destination = out/'joints'/f'{fit.fixed}__{fit.moving}'
                summary = export_joint(destination, bones[i], bones[i+1], fit, selections.get(i, 0), settings, profile,
                                       analysis_template=analysis_template)
                entry.update({'directory': str(destination.relative_to(out)), 'configurations': summary['configurations'],
                              'annotation_status': summary['annotation_status'], 'selected_candidate': selections.get(i, 0)})
            report['pairs'].append(entry)
        for i, t in world_transforms.items():
            b = bones[i]
            write_obj(out/'articulated'/f'{b.name}.obj', transform_points(b.mesh.vertices, t), b.mesh.faces)
            write_landmarks(out/'articulated'/f'{b.name}.mrk.json', b.landmarks.moved(t))
        report['segments'] = segments
        report['input_to_world'] = {bones[i].name: t @ inverse_rigid(bones[i].local_to_input) for i, t in world_transforms.items()}
        complete = (bool(fits) and all(f.candidates for f in fits)
                    and len(segments) == 1 and segments[0]['status'] == 'verified')
        report['status'] = 'complete_geometric_reference' if complete else 'partial_result_requires_review'
    report['elapsed_seconds'] = time.perf_counter()-started
    write_json(out/'report.json', report)
    progress(f'Wrote {out / "report.json"}: {report["status"]}')
    return report


def create_manifest(atlas, meshes, destination, *, analysis_template=None, reference_frame_landmarks=None):
    paths = sorted(Path(meshes).glob('*.ply'), key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)])
    if not paths: raise ValueError('mesh directory contains no PLY files')
    destination = Path(destination).resolve()
    data = {'schema_version': 1, 'name': Path(meshes).name, 'units': 'mm',
            'mesh_coordinate_system': 'LPS', 'ssm_coordinate_system': 'RAS',
            'atlas': str(Path(atlas).resolve()), 'guide_profile': 'vertebra28',
            'bones': [{'name': p.stem, 'mesh': str(p.resolve())} for p in paths],
            'transfer': asdict(TransferSettings()), 'articulation': settings_to_yaml(ArticulationSettings())}
    if analysis_template is not None or reference_frame_landmarks is not None:
        spec = resolve_analysis_spec({'template': analysis_template,
                                      'reference_frame_landmarks': reference_frame_landmarks}, Path.cwd())
        AnalysisTemplate.load(spec)  # Fail before writing a manifest with invalid settings.
        data['analysis'] = {key: str(value) for key, value in spec.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    destination.write_text(yaml.safe_dump(data, sort_keys=False))
    return destination


def settings_to_yaml(settings):
    data = asdict(settings)
    data['gap_fractions'] = list(data['gap_fractions'])
    return data
