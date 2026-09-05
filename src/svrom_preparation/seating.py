"""Complementary surface seating with adaptive nonpenetration constraints.

Regions are fixed, specimen-specific search neighborhoods, not universal patch
annotations. Every sampled part contributes to a long-tailed distance loss;
opposite normals and tangential centering discourage a convenient rim contact.
Final feasibility always comes from the independent triangle/containment check.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svrom.geometry import SurfaceMesh
from .data import inverse_rigid, transform_points
from .surfaces import area_samples, _vtk_matrix


def _unit(v):
    return v / max(float(np.linalg.norm(v)), np.finfo(float).eps)


@dataclass
class SeatingRegion:
    surface: SurfaceMesh
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    anchor: np.ndarray


class SeatingEvaluator:
    """Cached exact triangle geometry for a pair and its gap scenarios."""

    def __init__(self, fixed, moving, profile, settings):
        self.fixed, self.moving = fixed, moving
        self.interfaces, self.settings = profile['interfaces'], settings
        self.scale = (fixed.length + moving.length) / 2
        self.regions, self.collision_points, self.implicit = {}, {}, {}
        self._last_matrix, self._last_clearances = None, None
        for bone in (fixed, moving):
            for key, region in bone.regions.items():
                # Seed-face centroids can be offset differently by triangulation
                # on the two sides. Project the guide points themselves for the
                # weak centering term; retain area centers for footprint spread.
                guide_points = bone.landmarks.select(profile['regions'][key]['landmarks'])
                anchor = bone.mesh.project_surface(guide_points).point.mean(axis=0)
                # Original triangles, with original normal interpolation.
                surface = SurfaceMesh.from_arrays(
                    name=key, vertices=bone.mesh.vertices,
                    faces=bone.mesh.faces[region.face_ids],
                    vertex_normals=bone.mesh.vertex_normals)
                for full in (False, True):
                    ids, areas = area_samples(bone.mesh, region.face_ids,
                                              None if full else settings.sample_count)
                    points = bone.mesh.face_centroids[ids]
                    normals = bone.mesh.vertex_normals[bone.mesh.faces[ids]].mean(axis=1)
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
                    weights = areas * region.lookup[ids]
                    weights /= weights.sum()
                    center = np.sum(points * weights[:, None], axis=0)
                    normal = _unit(np.sum(normals * weights[:, None], axis=0))
                    self.regions[bone.name, key, full] = SeatingRegion(
                        surface, points, normals, weights, center, normal, anchor)
            ids, _ = area_samples(bone.mesh, np.arange(len(bone.mesh.faces)), settings.collision_samples)
            vertices = np.unique(bone.mesh.faces[ids])
            self.collision_points[bone.name] = np.concatenate(
                [bone.mesh.face_centroids[ids], bone.mesh.vertices[vertices]])
            implicit = vtk.vtkImplicitPolyDataDistance()
            implicit.SetInput(bone.mesh.vtk_polydata)
            self.implicit[bone.name] = implicit

    def _directions(self, matrix, item):
        return ((self.fixed, item['fixed'], self.moving, item['moving'], inverse_rigid(matrix)),
                (self.moving, item['moving'], self.fixed, item['fixed'], matrix))

    def _direction(self, sender, skey, receiver, rkey, matrix, gap, full, metrics):
        a = self.regions[sender.name, skey, full]
        b = self.regions[receiver.name, rkey, full]
        points = transform_points(a.points, matrix)
        normals = a.normals @ matrix[:3, :3].T
        hit = b.surface.project_surface(points)
        receiver_normals = np.einsum('ij,ijk->ik', hit.barycentric,
            b.surface.vertex_normals[b.surface.faces[hit.cell_id]])
        receiver_normals /= np.maximum(np.linalg.norm(receiver_normals, axis=1, keepdims=True), 1e-15)
        delta = hit.point - points
        offset_error = np.sum((delta - gap * normals)**2, axis=1)
        normal_error = np.sum((receiver_normals + normals)**2, axis=1)
        width = self.settings.seating_distance_fraction * self.scale
        loss = float(a.weights @ (np.log1p(offset_error / width**2)
                     + self.settings.seating_normal_weight * normal_error))
        axis = _unit(b.normal - a.normal @ matrix[:3, :3].T)
        shift = b.anchor - transform_points(a.anchor[None], matrix)[0]
        tangent = shift - np.dot(shift, axis) * axis
        tangent_fraction = float(np.linalg.norm(tangent) / self.scale)
        loss += self.settings.seating_center_weight * tangent_fraction**2
        if not metrics:
            return loss, None
        direction = delta / np.maximum(hit.distance[:, None], 1e-15)
        opposition = -np.einsum('ij,ij->i', normals, receiver_normals)
        facing = ((np.einsum('ij,ij->i', normals, direction) > 0)
                  & (np.einsum('ij,ij->i', receiver_normals, direction) < 0))
        supported = ((hit.distance <= gap + 2*self.settings.gap_width_fraction*self.scale)
                     & (opposition >= self.settings.normal_cosine) & facing)
        coverage = float(a.weights @ supported)
        # Spread along both principal tangential directions, normalized against
        # the full fixed neighborhood. This does not reward a point or thin rim.
        tangent_points = a.points-a.center
        tangent_points -= (tangent_points @ a.normal)[:, None]*a.normal
        covariance = (tangent_points*a.weights[:, None]).T @ tangent_points
        values, vectors = np.linalg.eigh(covariance)
        active = values > max(values[-1], self.scale**2)*1e-10
        active_ids = np.flatnonzero(active)[-2:]
        spread = 0.
        if coverage > 0 and len(active_ids):
            basis = vectors[:, active_ids] / np.sqrt(values[active_ids])[None, :]
            p = (a.points-a.center) @ basis
            w = a.weights*supported/coverage
            centered = p - np.sum(p*w[:, None], axis=0)
            spread = float(np.clip(np.linalg.eigvalsh((centered*w[:, None]).T @ centered).min(), 0, 1))
        return loss, {'coverage': coverage, 'spread': spread,
                      'tangential_offset_fraction': tangent_fraction,
                      'surface_rms_mm': float(np.sqrt(a.weights @ hit.distance**2)),
                      'offset_rms_mm': float(np.sqrt(a.weights @ offset_error)),
                      'normal_error': float(a.weights @ normal_error)}

    def evaluate(self, matrix, gap, *, full=False, metrics=False):
        losses, details = [], {}
        for item in self.interfaces:
            directions = [self._direction(*args, gap, full, metrics)
                          for args in self._directions(matrix, item)]
            losses.append(sum(x[0] for x in directions)/2)
            if metrics:
                a, b = [x[1] for x in directions]
                details[item['name']] = {
                    'fixed': a, 'moving': b,
                    'minimum_coverage': min(a['coverage'], b['coverage']),
                    'minimum_spread': min(a['spread'], b['spread']),
                    'maximum_tangential_offset_fraction': max(a['tangential_offset_fraction'], b['tangential_offset_fraction'])}
        energy = float(np.mean(losses))
        if not metrics:
            return energy
        reasons = []
        for name, d in details.items():
            if d['minimum_coverage'] < self.settings.minimum_seating_coverage:
                reasons.append(f'{name}: insufficient distributed contact')
            if d['minimum_spread'] < self.settings.minimum_seating_spread:
                reasons.append(f'{name}: contact concentrated in a narrow footprint')
            if d['maximum_tangential_offset_fraction'] > self.settings.maximum_seating_offset_fraction:
                reasons.append(f'{name}: excessive tangential seating offset')
        return energy, {'interfaces': details, 'passes': not reasons, 'review_reasons': reasons}

    def clearances(self, matrix):
        """Exact signed triangle distances, positive outside each closed mesh."""
        if self._last_matrix is not None and np.array_equal(matrix, self._last_matrix):
            return self._last_clearances
        distances = []
        for sender, receiver, transform in ((self.fixed, self.moving, inverse_rigid(matrix)),
                                            (self.moving, self.fixed, matrix)):
            points = transform_points(self.collision_points[sender.name], transform)
            inputs = numpy_to_vtk(np.ascontiguousarray(points), deep=False)
            output = vtk.vtkDoubleArray()
            output.SetNumberOfTuples(len(points))
            self.implicit[receiver.name].EvaluateFunction(inputs, output)
            distances.append(vtk_to_numpy(output).copy()/self.scale)
        self._last_matrix, self._last_clearances = matrix.copy(), distances
        return distances

    def constraints(self, matrix):
        # Small fixed groups retain multiple active contacts for the constrained
        # solver. Grouping changes no inequality: every point must be outside.
        return np.array([part.min() for d in self.clearances(matrix)
                         for part in np.array_split(d, min(32, len(d)))]) - self.settings.clearance_fraction

    def add_collision_witnesses(self, matrix):
        """Attach intersection locations to both meshes for the next refinement."""
        detector = vtk.vtkCollisionDetectionFilter()
        detector.SetInputData(0, self.fixed.mesh.vtk_polydata)
        detector.SetInputData(1, self.moving.mesh.vtk_polydata)
        detector.SetMatrix(0, _vtk_matrix(np.eye(4)))
        detector.SetMatrix(1, _vtk_matrix(matrix))
        detector.SetCollisionModeToAllContacts()
        detector.SetBoxTolerance(0.)
        detector.SetCellTolerance(0.)
        detector.GenerateScalarsOff()
        detector.Update()
        points = detector.GetContactsOutput().GetPoints()
        if points is None or not points.GetNumberOfPoints():
            return 0
        fixed_points = np.unique(vtk_to_numpy(points.GetData()), axis=0)
        # Limit redundant contacts deterministically, never the final check.
        if len(fixed_points) > 128:
            fixed_points = fixed_points[np.linspace(0, len(fixed_points)-1, 128, dtype=int)]
        for bone, pts in ((self.fixed, fixed_points),
                          (self.moving, transform_points(fixed_points, inverse_rigid(matrix)))):
            self.collision_points[bone.name] = np.concatenate([self.collision_points[bone.name], pts])
        self._last_matrix = None
        return len(fixed_points)
