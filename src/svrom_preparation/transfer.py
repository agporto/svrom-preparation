"""Headless MorphoWeave-style SSM registration and landmark transfer.

Native rustcpd pose initialization -> initialized full SSM fit -> optional
fine CPD -> 3-D radial-basis interpolation of dense displacements. This uses
the native registration primitives also used by SlicerMorphoWeave; it does
not require Slicer or claim bitwise equivalence to its GUI's FPFH stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import tempfile
import zipfile

import numpy as np
from scipy.interpolate import RBFInterpolator

from .data import Landmarks, convert_coordinates, coordinate_system, read_landmarks, sha256
from .settings import TransferSettings


def farthest_indices(points, count):
    points = np.asarray(points, float)
    count = min(int(count), len(points))
    if count == len(points): return np.arange(count)
    chosen = np.empty(count, dtype=int)
    chosen[0] = np.argmax(np.sum((points-points.mean(0))**2, axis=1))
    distance = np.full(len(points), np.inf)
    for j in range(1, count):
        distance = np.minimum(distance, np.sum((points-points[chosen[j-1]])**2, axis=1))
        distance[chosen[:j]] = -1
        chosen[j] = np.argmax(distance)
    return chosen


def warp_landmarks(points, source, destination, *, max_controls=800):
    """3-D U(r)=r interpolant with an affine tail; exact for affine maps.

    SciPy's linear kernel is -r, which gives the same interpolant at zero
    smoothing. Normalize the source for numerical stability. No 2-D r²log(r)
    thin-plate kernel is used for these three-dimensional points.
    """
    source, destination = np.asarray(source, float), np.asarray(destination, float)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('dense correspondence arrays must have matching Nx3 shapes')
    ids = farthest_indices(source, max_controls)
    center = source.mean(0)
    scale = float(np.linalg.norm(np.ptp(source, axis=0)))
    if scale <= 0: raise ValueError('degenerate dense correspondences')
    src = (source[ids]-center)/scale
    if np.linalg.matrix_rank(np.c_[np.ones(len(src)), src]) < 4:
        raise ValueError('landmark interpolation requires noncoplanar controls')
    interpolator = RBFInterpolator(src, destination[ids]-source[ids], kernel='linear', degree=1)
    return np.asarray(points) + interpolator((np.asarray(points)-center)/scale)


@dataclass
class Atlas:
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    dense: np.ndarray
    sparse: Landmarks
    fingerprint: dict

    @classmethod
    def load(cls, directory, *, coordinates='LPS', ssm_coordinates='RAS'):
        directory = Path(directory).resolve()
        if directory.is_file() and zipfile.is_zipfile(directory):
            with tempfile.TemporaryDirectory(prefix='svrom_atlas_') as temp:
                root = Path(temp).resolve()
                with zipfile.ZipFile(directory) as archive:
                    for member in archive.infolist():
                        if not (root/member.filename).resolve().is_relative_to(root):
                            raise ValueError('atlas archive contains an unsafe path')
                    archive.extractall(root)
                manifests = [p for p in root.rglob('manifest.json') if '__MACOSX' not in p.parts]
                if len(manifests) != 1: raise ValueError('atlas ZIP must contain exactly one model manifest')
                return cls.load(manifests[0].parent, coordinates=coordinates, ssm_coordinates=ssm_coordinates)
        manifest = json.loads((directory/'manifest.json').read_text())
        paths = {k: (directory/v).resolve() for k, v in manifest['files'].items()}
        if any(not p.is_relative_to(directory) for p in paths.values()):
            raise ValueError('atlas manifest entries must remain inside the atlas directory')
        dense = read_landmarks(paths['dense'], coordinates=coordinates).points
        sparse = read_landmarks(paths['sparse'], coordinates=coordinates)
        with np.load(paths['ssm'], allow_pickle=False) as z:
            mean = np.array(z['mean_shape'], dtype=float)
            modes = np.array(z['modes'], dtype=float)
            eig = np.array(z['eigenvalues'], dtype=float)
        if (mean.shape != dense.shape or modes.shape != (*mean.shape, len(eig))
                or not all(np.isfinite(x).all() for x in (mean, modes, eig))
                or np.any(eig < 0) or not np.any(eig > 0)):
            raise ValueError('invalid SSM shapes, eigenvalues, or dense correspondence count')
        mean = convert_coordinates(mean, ssm_coordinates, coordinates)
        if coordinate_system(ssm_coordinates) != coordinate_system(coordinates):
            modes[:, :2, :] *= -1
        # Smoothed template correspondences may differ slightly from the SSM
        # mean; allow that, but catch an undeclared RAS/LPS mismatch.
        discrepancy = float(np.sqrt(np.mean(np.sum((dense-mean)**2, axis=1))) /
                            np.linalg.norm(np.ptp(mean, axis=0)))
        if discrepancy > 0.10:
            raise ValueError('SSM and dense points disagree; check ssm_coordinate_system and correspondence order')
        return cls(mean, modes, eig, dense, sparse,
                   {'files': {k: sha256(p) for k, p in paths.items()},
                    'ssm_coordinate_system': coordinate_system(ssm_coordinates),
                    'working_coordinate_system': coordinate_system(coordinates),
                    'mean_dense_relative_rms': discrepancy})


def transfer_landmarks(atlas, bone, settings=TransferSettings()):
    try:
        import rustcpd
    except ImportError as exc:
        raise RuntimeError('Install svrom-preparation[ssm] for SSM landmark transfer, or supply MorphoWeave landmark exports') from exc
    started = time.perf_counter()
    raw = bone.mesh
    # Deterministic area-stratified target points, with a fixed barycentric
    # low-discrepancy sequence so repeated face selections are distinct.
    count = settings.target_count
    cdf = np.cumsum(raw.face_areas)
    ids = np.searchsorted(cdf, (np.arange(count)+0.5)*cdf[-1]/count)
    u = np.mod((np.arange(count)+0.5)*0.7548776662466927, 1.)
    v = np.mod((np.arange(count)+0.5)*0.5698402909980532, 1.)
    r = np.sqrt(u)
    bary = np.c_[1-r, r*(1-v), r*v]
    target = np.einsum('ij,ijk->ik', bary, raw.vertices[raw.faces[ids]])
    target_center = target.mean(0)
    target_scale = np.linalg.norm(np.ptp(target, axis=0))/20.0
    source_center = atlas.mean.mean(0)
    source_scale = np.linalg.norm(np.ptp(atlas.mean, axis=0))/20.0
    source = (atlas.mean-source_center)/source_scale
    target_work = (target-target_center)/target_scale
    positive = atlas.eigenvalues > np.max(atlas.eigenvalues)*1e-12
    eig = atlas.eigenvalues[positive]
    rank = min(len(eig), int(np.searchsorted(np.cumsum(eig), settings.variance_keep*eig.sum()))+1)
    eig = eig[:rank]
    modes = np.ascontiguousarray(atlas.modes[:, :, positive][:, :, :rank]/source_scale).reshape(source.size, rank)
    pose = rustcpd.pose_initialize(
        source, target_work, modes, eig, rotation_count=settings.rotation_count,
        coarse_source_count=min(settings.coarse_count, len(source)),
        coarse_target_count=min(settings.coarse_count, len(target_work)),
        coarse_rank=min(12, rank), coarse_iterations=8, coarse_screen_iterations=8,
        coarse_survivor_count=settings.rotation_count, coarse_score_mode='trajectory',
        refine_count=min(settings.refine_count, settings.rotation_count),
        refine_target_count=len(target_work), refine_iterations=30,
        lambda_regularization=5.0, outlier_weight=0.15, with_scale=True,
        seed=settings.seed, parallel=settings.parallel,
    )
    fitted = rustcpd.register_atlas(
        target_work, source, modes, eig,
        initial_coefficients=np.asarray(pose.coefficients), initial_rotation=np.asarray(pose.rotation),
        initial_scale=float(pose.scale), initial_translation=np.asarray(pose.translation),
        lambda_regularization=settings.lambda_regularization, normalize=True,
        optimize_similarity=True, with_scale=True, outlier_weight=0.10,
        max_iterations=settings.max_iterations, tolerance=1e-6,
        k=min(10, len(target_work)), kdtree_radius_scale=10., parallel=settings.parallel,
    )
    registered = np.asarray(fitted.points)
    if settings.fine_cpd:
        fine = rustcpd.register_deformable(
            target_work, registered, alpha=2., beta=2.,
            low_rank=min(settings.fine_rank, len(registered)),
            max_iterations=settings.max_iterations, tolerance=1e-6,
            k=min(10, len(target_work)), parallel=settings.parallel,
        )
        registered = np.asarray(fine.points)
    registered = registered*target_scale + target_center
    predicted = warp_landmarks(atlas.sparse.points, atlas.dense, registered, max_controls=settings.tps_count)
    projection = raw.project_surface(predicted)
    dense_projection = raw.project_surface(registered)
    if not np.isfinite(predicted).all() or not np.isfinite(projection.point).all():
        raise RuntimeError('registration produced nonfinite landmarks')
    rotation = np.asarray(pose.rotation)
    if not np.isclose(np.linalg.det(rotation), 1., atol=1e-7):
        raise RuntimeError('registration returned a reflected pose')
    output = Landmarks(projection.point+bone.origin, atlas.sparse.labels, atlas.sparse.ids, atlas.sparse.coordinates)
    report = {
        'backend': 'rustcpd', 'backend_version': rustcpd.__version__, 'settings': asdict(settings),
        'atlas': atlas.fingerprint, 'mesh_sha256': sha256(bone.path) if bone.path else None,
        'rank': rank, 'elapsed_seconds': time.perf_counter()-started,
        'pose_score_margin': float(pose.score_margin),
        'pose_effective_hypotheses': float(pose.effective_hypotheses),
        'rotation_determinant': float(np.linalg.det(rotation)),
        'dense_surface_rms': float(np.sqrt(np.mean(dense_projection.distance**2))),
        'landmark_projection_distances': projection.distance.tolist(),
        'maximum_landmark_projection_fraction': float(projection.distance.max()/raw.diagonal),
        'projection_face_ids': projection.cell_id.tolist(),
        'needs_review': bool(projection.distance.max()/raw.diagonal > settings.maximum_projection_fraction),
        'interpretation': 'surface-fit diagnostics; not landmark accuracy against manual ground truth',
    }
    return output, report
