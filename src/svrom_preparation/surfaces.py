"""Geodesic guide regions, exact triangle queries, and collision validation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import trimesh
import vtk

from svrom.sdf import SignedDistanceGrid
from .data import inverse_rigid, transform_points


@dataclass
class Region:
    name: str
    face_ids: np.ndarray
    weights: np.ndarray
    lookup: np.ndarray
    seed_faces: np.ndarray
    radius: float


def face_graph(mesh):
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    adjacency = tm.face_adjacency
    distance = np.linalg.norm(mesh.face_centroids[adjacency[:, 0]]-mesh.face_centroids[adjacency[:, 1]], axis=1)
    distance = np.maximum(distance, np.finfo(float).eps*mesh.diagonal)
    return coo_matrix((np.r_[distance, distance],
                       (np.r_[adjacency[:, 0], adjacency[:, 1]],
                        np.r_[adjacency[:, 1], adjacency[:, 0]])),
                      shape=(len(mesh.faces), len(mesh.faces))).tocsr()


def prepare_regions(bone, profile):
    graph = face_graph(bone.mesh)
    bone.regions.clear()
    for name, spec in profile['regions'].items():
        radius = float(spec['radius_fraction'])*bone.length
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('region radii must be finite and positive')
        seeds = bone.landmarks.select(spec['landmarks'])
        nearest = bone.mesh.project_surface(seeds)
        seed_faces = np.unique(nearest.cell_id)
        distance = dijkstra(graph, directed=False, indices=seed_faces, min_only=True, limit=radius)
        ids = np.flatnonzero(distance <= radius)
        if not len(ids): raise ValueError(f'empty guide region: {bone.name}/{name}')
        lookup = np.zeros(len(bone.mesh.faces))
        weights = np.exp(-0.5*(distance[ids]/radius)**2)
        lookup[ids] = weights
        bone.regions[name] = Region(name, ids, weights, lookup, seed_faces, radius)
    bone._face_graph = graph


def area_samples(mesh, face_ids, count):
    """Deterministic area-CDF quadrature, retaining the fixed region's area.

    None uses every triangle centroid. Subsampling uses equal-area bins,
    coalescing duplicate selected faces without losing their quadrature mass.
    """
    ids = np.asarray(face_ids, dtype=int)
    areas = mesh.face_areas[ids]
    if count is None or len(ids) <= count:
        return ids, areas.copy()
    cdf = np.cumsum(areas)
    selected = ids[np.searchsorted(cdf, (np.arange(count)+0.5)*cdf[-1]/count)]
    ids, repetitions = np.unique(selected, return_counts=True)
    return ids, repetitions*(cdf[-1]/count)


def apposition_scores(points, normals, receiver, receiver_region, gap, width, normal_cosine):
    """Continuous distance/facing support, evaluated in the receiver frame.

    Closest points are on the full original mesh; hits outside the appropriate
    anatomical neighborhood contribute zero. Normal agreement and facing
    are independent of the gap kernel. Values are scores, not probabilities.
    """
    projection = receiver.project_surface(points)
    rnormals = np.einsum('ij,ijk->ik', projection.barycentric,
                        receiver.vertex_normals[receiver.faces[projection.cell_id]])
    rnormals /= np.maximum(np.linalg.norm(rnormals, axis=1, keepdims=True), np.finfo(float).eps)
    opposition = -np.einsum('ij,ij->i', normals, rnormals)
    opposition = np.clip((opposition-normal_cosine)/(1-normal_cosine), 0, 1)
    delta = projection.point-points
    direction = delta/np.maximum(projection.distance[:, None], 1e-14*receiver.diagonal)
    sender_facing = np.clip(np.einsum('ij,ij->i', normals, direction), 0, 1)
    receiver_facing = np.clip(-np.einsum('ij,ij->i', rnormals, direction), 0, 1)
    coincident = projection.distance <= 1e-12*receiver.diagonal
    sender_facing[coincident] = receiver_facing[coincident] = 1.
    distance_score = np.exp(-0.5*((projection.distance-gap)/width)**2)
    score = distance_score*opposition*np.sqrt(sender_facing*receiver_facing)
    score *= receiver_region.lookup[projection.cell_id]
    return score


def direction_scores(sender, receiver, sregion, rregion, sender_to_receiver, gap, width, settings, count):
    ids, areas = area_samples(sender.mesh, sregion.face_ids, count)
    points = transform_points(sender.mesh.face_centroids[ids], sender_to_receiver)
    normals = sender.mesh.vertex_normals[sender.mesh.faces[ids]].mean(axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), np.finfo(float).eps)
    normals = normals @ sender_to_receiver[:3, :3].T
    scores = apposition_scores(points, normals, receiver.mesh, rregion, gap, width, settings.normal_cosine)
    weights = areas*sregion.lookup[ids]
    support = float(np.dot(scores, weights)/weights.sum())
    return support, ids, scores


def pair_supports(fixed, moving, interfaces, matrix, gap, width, settings, count=None):
    inv = inverse_rigid(matrix)
    supports, details = [], {}
    for item in interfaces:
        a, b = fixed.regions[item['fixed']], moving.regions[item['moving']]
        sa, ia, qa = direction_scores(fixed, moving, a, b, inv, gap, width, settings, count)
        sb, ib, qb = direction_scores(moving, fixed, b, a, matrix, gap, width, settings, count)
        supports.append(min(sa, sb))
        details[item['name']] = {'fixed': (ia, qa), 'moving': (ib, qb), 'supports': [sa, sb]}
    return np.asarray(supports), details


def prepare_collision(bone, settings):
    if bone.sdf is None:
        bone.sdf = SignedDistanceGrid.from_surface(bone.mesh, longest_axis_samples=settings.sdf_samples,
                                                  padding_fraction=0.20, smoothing_sigma_voxels=0.)


def collision_penalty(fixed, moving, matrix, scale, settings):
    depths, masses = [], []
    for sender, receiver, transform in ((moving, fixed, matrix), (fixed, moving, inverse_rigid(matrix))):
        ids, areas = area_samples(sender.mesh, np.arange(len(sender.mesh.faces)), settings.collision_samples)
        points = transform_points(sender.mesh.face_centroids[ids], transform)
        signed = receiver.sdf.query(points)
        depths.append(np.maximum(-signed, 0.)/scale)
        masses.append(areas/areas.sum()/2)
    depth, mass = np.concatenate(depths), np.concatenate(masses)
    return float(100.*depth.max() + 20000.*np.dot(mass, depth**2))


def _vtk_matrix(matrix):
    out = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4): out.SetElement(i, j, float(matrix[i, j]))
    return out


def collision_check(fixed, moving, matrix):
    """Triangle-intersection plus component-containment check, without an SDF.

    The optimization SDF is deliberately not the final acceptance test. Open
    or inconsistently wound meshes cannot receive a verified status. Touching
    triangles count as contact; positive gap scenarios keep them separated.
    """
    closed = fixed.watertight and moving.watertight and fixed.winding_consistent and moving.winding_consistent
    vf, vm = fixed.mesh.vertices, transform_points(moving.mesh.vertices, matrix)
    disjoint = bool(np.any(vf.max(0) < vm.min(0)) or np.any(vm.max(0) < vf.min(0)))
    if disjoint:
        return {'verified': bool(closed), 'intersections': False, 'containment': False,
                'closed_surfaces': bool(closed)}
    detector = vtk.vtkCollisionDetectionFilter()
    detector.SetInputData(0, fixed.mesh.vtk_polydata)
    detector.SetInputData(1, moving.mesh.vtk_polydata)
    detector.SetMatrix(0, _vtk_matrix(np.eye(4)))
    detector.SetMatrix(1, _vtk_matrix(matrix))
    detector.SetCollisionModeToFirstContact()
    detector.SetBoxTolerance(0.)
    detector.SetCellTolerance(0.)
    detector.GenerateScalarsOff()
    detector.Update()
    intersections = bool(detector.GetNumberOfContacts())
    containment = False
    if closed and not intersections:
        for receiver, points in (
            (fixed, vm[moving.component_vertices]),
            (moving, transform_points(vf[fixed.component_vertices], inverse_rigid(matrix))),
        ):
            enclosed = vtk.vtkSelectEnclosedPoints()
            enclosed.SetTolerance(1e-9)
            enclosed.Initialize(receiver.mesh.vtk_polydata)
            containment |= any(bool(enclosed.IsInsideSurface(*map(float, p))) for p in points)
            enclosed.Complete()
    return {'verified': bool(closed and not intersections and not containment),
            'intersections': intersections, 'containment': bool(containment), 'closed_surfaces': bool(closed)}


def connected_patch(bone, mask):
    """Keep the largest supported connected component within one interface."""
    ids = np.flatnonzero(mask)
    output = np.zeros(len(mask), dtype=bool)
    if not len(ids): return output
    _, labels = connected_components(bone._face_graph[ids][:, ids], directed=False)
    areas = np.bincount(labels, weights=bone.mesh.face_areas[ids])
    output[ids[labels == areas.argmax()]] = True
    return output


def patch_ensemble(fixed, moving, interfaces, candidates, settings):
    if not candidates: raise ValueError('patch annotation requires retained verified articulations')
    accum = {i['name']: {side: [] for side in ('fixed', 'moving')} for i in interfaces}
    scale = (fixed.length+moving.length)/2
    for candidate in candidates:
        _, details = pair_supports(fixed, moving, interfaces, candidate.matrix, candidate.gap,
                                  settings.gap_width_fraction*scale, settings, None)
        for name, value in details.items():
            for side, bone in (('fixed', fixed), ('moving', moving)):
                ids, scores = value[side]
                full = np.zeros(len(bone.mesh.faces))
                full[ids] = scores
                accum[name][side].append(full)
    result = {}
    for name, sides in accum.items():
        result[name] = {}
        for side, bone in (('fixed', fixed), ('moving', moving)):
            scores = np.stack(sides[side])
            frequency = np.mean(scores >= settings.patch_score_threshold, axis=0)
            possible = connected_patch(bone, frequency >= settings.extension_frequency)
            core = connected_patch(bone, (frequency >= settings.core_frequency) & possible)
            result[name][side] = {
                'mean_score': scores.mean(0), 'support_frequency': frequency,
                'core_face_ids': np.flatnonzero(core), 'possible_face_ids': np.flatnonzero(possible),
                'core_area': float(bone.mesh.face_areas[core].sum()),
                'possible_area': float(bone.mesh.face_areas[possible].sum()),
            }
    return result
