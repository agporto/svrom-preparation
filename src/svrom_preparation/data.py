"""Mesh identity, coordinate-safe landmarks, and right-handed local frames."""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from svrom.geometry import SurfaceMesh


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def coordinate_system(value):
    # Slicer legacy FCSV: 0=RAS, 1=LPS.
    name = str(value).upper()
    name = {'0': 'RAS', '1': 'LPS'}.get(name, name)
    if name not in {'RAS', 'LPS'}:
        raise ValueError(f'unsupported coordinate system: {value}')
    return name


def convert_coordinates(points, source, destination):
    out = np.asarray(points, dtype=float).copy()
    if coordinate_system(source) != coordinate_system(destination):
        out[..., :2] *= -1
    return out


def transform_points(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def rigid_matrix(rotation=None, translation=None):
    m = np.eye(4)
    if rotation is not None:
        m[:3, :3] = rotation
    if translation is not None:
        m[:3, 3] = translation
    validate_rigid(m)
    return m


def validate_rigid(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('transform must be a finite 4x4 matrix')
    if (not np.allclose(m[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-8, rtol=0)):
        raise ValueError('transform must be a proper rigid rotation (determinant +1)')


def inverse_rigid(matrix):
    validate_rigid(matrix)
    r = matrix[:3, :3].T
    return rigid_matrix(r, -r @ matrix[:3, 3])


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError('landmarks do not define a nondegenerate anatomical frame')
    return vector / norm


@dataclass
class Landmarks:
    points: np.ndarray
    labels: tuple[str, ...]
    ids: tuple[str, ...]
    coordinates: str = 'LPS'

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.coordinates = coordinate_system(self.coordinates)
        if (self.points.shape != (len(self.labels), 3) or len(self.ids) != len(self.labels)
                or not np.isfinite(self.points).all() or not len(self.points)):
            raise ValueError('landmarks require matching labels, IDs, and finite Nx3 positions')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('landmark IDs must be unique')

    def select(self, numbers):
        indices = np.asarray(numbers)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 1 or indices.max() > len(self.points)):
            raise ValueError('guide numbers must be valid one-based control-point indices')
        return self.points[indices - 1]

    def center(self, numbers):
        return self.select(numbers).mean(axis=0)

    def moved(self, matrix):
        return Landmarks(transform_points(self.points, matrix), self.labels, self.ids, self.coordinates)


def read_landmarks(path, *, coordinates='LPS'):
    path = Path(path)
    if path.suffix.lower() == '.fcsv':
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        system = 'RAS'
        columns = None
        for line in lines:
            if line.startswith('#') and 'CoordinateSystem' in line:
                system = coordinate_system(line.split('=', 1)[1].strip())
            if line.startswith('#') and 'columns' in line:
                columns = [x.strip() for x in line.split('=', 1)[1].split(',')]
        if columns is None:
            raise ValueError('FCSV must declare its columns')
        rows = list(csv.DictReader((x for x in lines if x and not x.startswith('#')), fieldnames=columns))
        points = [[float(r[a]) for a in ('x', 'y', 'z')] for r in rows]
        labels = tuple(r.get('label', str(i + 1)) for i, r in enumerate(rows))
        ids = tuple(r.get('id', str(i + 1)) for i, r in enumerate(rows))
    else:
        doc = json.loads(path.read_text())
        if len(doc.get('markups', [])) != 1:
            raise ValueError('expected exactly one markup set')
        mark = doc['markups'][0]
        if mark.get('coordinateUnits', 'mm') not in {'mm', 'millimeter', 'millimetre'}:
            raise ValueError('landmarks must be in millimeters')
        system = coordinate_system(mark.get('coordinateSystem', 'RAS'))
        rows = mark['controlPoints']
        if any(r.get('positionStatus', 'defined') != 'defined' for r in rows):
            raise ValueError('all required landmark positions must be defined')
        points = [r['position'] for r in rows]
        labels = tuple(r.get('label', str(i + 1)) for i, r in enumerate(rows))
        ids = tuple(r.get('id', str(i + 1)) for i, r in enumerate(rows))
    return Landmarks(convert_coordinates(points, system, coordinates), labels, ids, coordinates)


def write_landmarks(path, landmarks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {'markups': [{'type': 'Fiducial', 'coordinateSystem': landmarks.coordinates,
                       'coordinateUnits': 'mm', 'controlPoints': [
                           {'id': i, 'label': label, 'position': p.tolist(), 'positionStatus': 'defined'}
                           for i, label, p in zip(landmarks.ids, landmarks.labels, landmarks.points)]}]}
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')


def safe_name(name):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))
    if value in {'', '.', '..'}:
        raise ValueError('invalid specimen/bone name')
    return value


@dataclass
class Bone:
    name: str
    mesh: SurfaceMesh
    origin: np.ndarray
    landmarks: Landmarks | None = None
    path: Path | None = None
    watertight: bool = True
    winding_consistent: bool = True
    reoriented_faces: int = 0
    component_vertices: np.ndarray = field(default_factory=lambda: np.array([0]))
    regions: dict = field(default_factory=dict)
    sdf: object = None
    frame: np.ndarray | None = None
    length: float = 1.0
    transfer_report: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, name=None):
        path = Path(path).resolve()
        raw = trimesh.load(path, process=False)
        if not isinstance(raw, trimesh.Trimesh):
            raise ValueError('expected one triangular mesh')
        v, f = np.asarray(raw.vertices, float), np.asarray(raw.faces, np.int64)
        if not np.isfinite(v).all() or not len(f) or np.any(raw.area_faces <= 0):
            raise ValueError('mesh must contain finite vertices and nondegenerate triangles')
        # Center before VTK construction, avoiding float32 loss at scan offsets.
        origin = np.average(raw.triangles_center, axis=0, weights=raw.area_faces)
        original_faces = f.copy()
        if raw.is_watertight:
            raw.fix_normals(multibody=True)
            f = np.asarray(raw.faces, np.int64)
            if not np.array_equal(np.sort(f, axis=1), np.sort(original_faces, axis=1)):
                raise RuntimeError('normal repair changed face identity')
        reoriented = int(np.count_nonzero(np.any(original_faces != f, axis=1)))
        components = trimesh.graph.connected_components(raw.edges_unique, nodes=np.arange(len(v)), min_len=1)
        representatives = np.array([int(c[0]) for c in components], dtype=int)
        surface = SurfaceMesh.from_arrays(name=name or path.stem, vertices=v-origin, faces=f)
        # Use float64 VTK points for this new workflow without changing the
        # established SurfaceMesh/ROM numerical profiles.
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(v-origin), deep=True))
        surface.vtk_polydata.SetPoints(points)
        surface.vtk_polydata.Modified()
        surface.vtk_locator.BuildLocator()
        return cls(name=safe_name(name or path.stem),
                   mesh=surface,
                   origin=origin, path=path, watertight=bool(raw.is_watertight),
                   winding_consistent=bool(raw.is_winding_consistent), component_vertices=representatives,
                   reoriented_faces=reoriented)

    def set_landmarks(self, landmarks, profile):
        if len(landmarks.points) < int(profile.get('minimum_landmarks', 1)):
            raise ValueError(f'{self.name}: too few landmarks for profile')
        self.landmarks = landmarks.moved(rigid_matrix(translation=-self.origin))
        lm = self.landmarks
        a, p = lm.center(profile['anterior']), lm.center(profile['posterior'])
        self.length = float(np.linalg.norm(p-a))
        # +X posterior, +Y ventral, +Z completes a right-handed frame.
        x = unit(p-a)
        up = lm.center(profile['dorsal']) - (a+p)/2
        y = -unit(up - (up @ x)*x)
        z = unit(np.cross(x, y))
        y = unit(np.cross(z, x))
        self.frame = rigid_matrix(np.column_stack([x, y, z]), (a+p)/2)
        side_vector = lm.center(profile['side_b']) - lm.center(profile['side_a'])
        if abs(unit(side_vector) @ z) < 0.5:
            raise ValueError(f'{self.name}: transferred side landmarks disagree with the anatomical frame')

    @property
    def local_to_input(self):
        return rigid_matrix(translation=self.origin)


def relative_angles(fixed, moving, transform):
    r = fixed.frame[:3, :3].T @ transform[:3, :3] @ moving.frame[:3, :3]
    return Rotation.from_matrix(r).as_euler('xyz', degrees=True)
