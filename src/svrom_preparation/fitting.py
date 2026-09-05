"""Bounded rigid articulation with fixed anatomical search neighborhoods."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .data import inverse_rigid, relative_angles, rigid_matrix, transform_points, validate_rigid
from .surfaces import collision_check, collision_penalty, pair_supports, prepare_collision


@dataclass
class Candidate:
    matrix: np.ndarray
    gap: float
    energy: float
    supports: np.ndarray
    collision: dict
    source: str
    optimization: dict = field(default_factory=dict)
    seating: dict = field(default_factory=dict)

    def as_dict(self):
        return {'moving_local_to_fixed_local': self.matrix, 'gap': self.gap,
                'energy': self.energy, 'interface_supports': self.supports,
                'collision': self.collision, 'source': self.source, 'optimization': self.optimization,
                'seating': self.seating}


@dataclass
class PairFit:
    fixed: str
    moving: str
    status: str
    candidates: list[Candidate]
    report: dict


class PairObjective:
    """Soft contact assignments are refreshed at each pose evaluation.

    Anatomical neighborhoods and their quadrature denominators stay fixed,
    preventing a patch from shrinking to one convenient point during fitting.
    """

    def __init__(self, fixed, moving, profile, settings, gap):
        self.fixed, self.moving = fixed, moving
        self.interfaces, self.settings = profile['interfaces'], settings
        self.scale = (fixed.length+moving.length)/2
        self.gap = float(gap)
        self.width = settings.gap_width_fraction*self.scale
        self.pivot = moving.landmarks.center(profile['anterior'])
        self.posterior = fixed.landmarks.center(profile['posterior'])
        self.base = rigid_matrix(translation=moving.origin-fixed.origin)
        self.anchor = transform_points(self.pivot[None], self.base)[0]
        # Rotation and translation coordinates are expressed in the fixed
        # anatomical frame, so global input rotation does not alter the search.
        self.axes = fixed.frame[:3, :3]
        r = np.deg2rad(settings.rotation_bound_deg)
        t = settings.translation_bound_fraction
        self.bounds = [(-r, r)]*3 + [(-t, t)]*3
        self.evaluations = 0

    def matrix(self, parameters):
        r = self.axes @ Rotation.from_rotvec(parameters[:3]).as_matrix() @ self.axes.T
        translation = self.anchor + self.scale*(self.axes @ parameters[3:]) - r @ self.pivot
        return rigid_matrix(r, translation)

    def parameters(self, matrix):
        rotation = Rotation.from_matrix(self.axes.T @ matrix[:3, :3] @ self.axes).as_rotvec()
        anchor = transform_points(self.pivot[None], matrix)[0]
        translation = self.axes.T @ (anchor-self.anchor)/self.scale
        return np.r_[rotation, translation]

    def evaluate_matrix(self, matrix, full=False):
        supports, _ = pair_supports(self.fixed, self.moving, self.interfaces, matrix,
                                    self.gap, self.width, self.settings,
                                    None if full else self.settings.sample_count)
        apposition = -float(np.mean(np.log(np.maximum(supports, 1e-6))))
        collision = collision_penalty(self.fixed, self.moving, matrix, self.scale, self.settings)
        return apposition+collision, supports

    def __call__(self, parameters):
        self.evaluations += 1
        return self.evaluate_matrix(self.matrix(parameters))[0]

    def candidate(self, matrix, source, optimization=None):
        energy, supports = self.evaluate_matrix(matrix, full=True)
        check = collision_check(self.fixed, self.moving, matrix)
        return Candidate(matrix, self.gap, energy, supports, check, source, optimization or {})


def _distinct(candidate, retained, scale, angle=0.3, translation=0.002):
    for old in retained:
        rotation = Rotation.from_matrix(candidate.matrix[:3, :3] @ old.matrix[:3, :3].T).magnitude()
        distance = np.linalg.norm(candidate.matrix[:3, 3]-old.matrix[:3, 3])/scale
        if rotation < np.deg2rad(angle) and distance < translation and np.isclose(candidate.gap, old.gap):
            return False
    return True


def fit_pair(fixed, moving, profile, settings):
    started = time.perf_counter()
    scale = (fixed.length+moving.length)/2
    original_step = float(np.linalg.norm(moving.origin-fixed.origin))
    info = {'length_scale': scale, 'original_origin_step': original_step,
            'interfaces': [i['name'] for i in profile['interfaces']]}
    if original_step/scale > settings.maximum_origin_step_fraction:
        return PairFit(fixed.name, moving.name, 'coordinate_frame_mismatch', [], info)
    if not (fixed.watertight and moving.watertight and fixed.winding_consistent and moving.winding_consistent):
        return PairFit(fixed.name, moving.name, 'mesh_requires_review', [], info)
    if fixed.transfer_report.get('needs_review') or moving.transfer_report.get('needs_review'):
        return PairFit(fixed.name, moving.name, 'landmark_transfer_requires_review', [], info)
    if settings.objective == 'complementary_seating':
        return _fit_complementary(fixed, moving, profile, settings, info, started)
    prepare_collision(fixed, settings)
    prepare_collision(moving, settings)
    all_candidates, gap_reports = [], []
    for fraction in settings.gap_fractions:
        objective = PairObjective(fixed, moving, profile, settings, fraction*scale)
        seeds = [np.zeros(6)]
        rotations = [np.eye(3), fixed.frame[:3, :3] @ moving.frame[:3, :3].T]
        # Apposed surface normals supply an initialization that can contain
        # intrinsic joint curvature even when the landmark frames are parallel.
        # A single planar interface leaves rotation about its normal free;
        # align_vectors then uses the shortest proper rotation as a convention.
        normal_fixed, normal_moving = [], []
        for item in profile['interfaces']:
            directions = []
            for bone, key in ((fixed, item['fixed']), (moving, item['moving'])):
                region = bone.regions[key]
                weights = bone.mesh.face_areas[region.face_ids]*region.weights
                normal = np.average(bone.mesh.face_normals[region.face_ids], axis=0, weights=weights)
                length = np.linalg.norm(normal)
                directions.append(normal/length if length > 1e-6 else None)
            if all(n is not None for n in directions):
                normal_fixed.append(-directions[0]); normal_moving.append(directions[1])
        if normal_fixed:
            nf, nm = np.asarray(normal_fixed), np.asarray(normal_moving)
            if min(np.linalg.matrix_rank(nf, tol=1e-5), np.linalg.matrix_rank(nm, tol=1e-5)) < 2:
                nf, nm = nf[:1], nm[:1]
            rotations.append(Rotation.align_vectors(nf, nm)[0].as_matrix())
        for axis in range(3):
            for sign in (-1, 1):
                local = Rotation.from_rotvec(np.eye(3)[axis]*sign*np.deg2rad(min(8., settings.rotation_bound_deg))).as_matrix()
                rotations.append(objective.axes @ local @ objective.axes.T)
        # Landmark frames provide an additional initialization, not a target
        # orientation in the objective. Keep both the as-scanned orientation
        # and frame-aligned orientation, with centrum interface anchors near
        # the requested gap. This avoids a local distance trap at a tilted edge.
        for r in rotations:
            t = objective.posterior + objective.axes[:, 0]*objective.gap-r @ objective.pivot
            x = objective.parameters(rigid_matrix(r, t))
            if all(lo <= v <= hi for v, (lo, hi) in zip(x, objective.bounds)):
                seeds.append(x)
        # Always include the unmodified input pose and opening translations.
        for offset in (0.03, 0.08, -0.03):
            x = np.zeros(6); x[3] = np.clip(offset, *objective.bounds[3]); seeds.append(x)
        for axis in range(3):
            for sign in (-1, 1):
                x = np.zeros(6)
                x[axis] = sign*np.deg2rad(min(8., settings.rotation_bound_deg))
                seeds.append(x)
        ranked = sorted(((objective(s), j, s) for j, s in enumerate(seeds)), key=lambda x: (x[0], x[1]))
        proposals = [(objective.matrix(s), 'input_or_seed', {}) for _, _, s in ranked]
        for _, _, seed in ranked[:settings.refine_candidates]:
            result = minimize(objective, seed, method='Powell', bounds=objective.bounds,
                              options={'maxfev': settings.max_evaluations, 'xtol': 2e-4, 'ftol': 2e-4})
            proposals.append((objective.matrix(result.x), 'optimized',
                              {'success': bool(result.success), 'message': str(result.message),
                               'evaluations': int(result.nfev)}))
        # Full face-centroid scoring and triangle/containment validation only
        # after optimization. An optimizer budget limit is reported honestly.
        candidates = [objective.candidate(m, source, opt) for m, source, opt in proposals]
        candidates.sort(key=lambda c: c.energy)
        # Coarse SDF optimization can finish with a small unresolved surface
        # intersection. Probe bounded opening translations, accepting only
        # poses that pass the independent triangle/containment check.
        for c in [x for x in candidates if x.source == 'optimized' and not x.collision['verified']][:2]:
            for fraction_step in (0.002, 0.005, 0.01, 0.02, 0.04):
                matrix = c.matrix.copy()
                matrix[:3, 3] += objective.axes[:, 0]*(fraction_step*scale)
                x = objective.parameters(matrix)
                if not all(lo <= v <= hi for v, (lo, hi) in zip(x, objective.bounds)): continue
                repaired = objective.candidate(matrix, 'bounded_clearance_probe')
                if repaired.collision['verified']:
                    candidates.append(repaired)
                    break
        candidates.sort(key=lambda c: c.energy)
        valid = [c for c in candidates if c.collision['verified'] and
                 np.min(c.supports) >= settings.minimum_interface_support]
        gap_reports.append({'gap_fraction': fraction, 'gap': fraction*scale,
                            'evaluations': objective.evaluations, 'verified_candidates': len(valid),
                            'best_verified_energy': valid[0].energy if valid else None,
                            'best_attempt': candidates[0].as_dict()})
        if valid:
            best = valid[0]
            # A bounded local ensemble supports pre-annotation beyond a
            # single footprint. Final SVROM feasibility is never used here.
            for axis in range(3):
                for sign in (-1, 1):
                    delta = Rotation.from_rotvec(np.eye(3)[axis]*np.deg2rad(settings.ensemble_angle_deg)*sign).as_matrix()
                    delta = objective.axes @ delta @ objective.axes.T
                    anchor = transform_points(objective.pivot[None], best.matrix)[0]
                    r = delta @ best.matrix[:3, :3]
                    m = rigid_matrix(r, anchor-r @ objective.pivot)
                    # Keep local perturbations inside the original search box.
                    rv = Rotation.from_matrix(objective.axes.T @ r @ objective.axes).as_rotvec()
                    if np.any(np.abs(rv) > np.deg2rad(settings.rotation_bound_deg)+1e-10): continue
                    c = objective.candidate(m, 'local_ensemble')
                    if (c.collision['verified'] and np.min(c.supports) >= settings.minimum_interface_support
                            and c.energy <= best.energy+settings.ensemble_energy_slack):
                        valid.append(c)
            for c in sorted(valid, key=lambda c: c.energy):
                if c.energy <= best.energy+settings.ensemble_energy_slack and _distinct(c, all_candidates, scale):
                    all_candidates.append(c)
    all_candidates.sort(key=lambda c: c.energy)
    # Preserve a representative from every gap scenario before filling the
    # remaining candidate budget; these are sensitivity scenarios, not an
    # inferred posterior distribution of cartilage gaps.
    retained = []
    for fraction in settings.gap_fractions:
        group = [c for c in all_candidates if np.isclose(c.gap, fraction*scale)]
        if group: retained.append(group[0])
    for c in all_candidates:
        if len(retained) >= max(settings.retain_candidates, len(settings.gap_fractions)): break
        if all(c is not old for old in retained): retained.append(c)
    retained.sort(key=lambda c: c.energy)
    info.update({'gap_scenarios': gap_reports, 'elapsed_seconds': time.perf_counter()-started,
                 'retained_candidates': len(retained)})
    if retained:
        info['input_relative_angles_deg'] = relative_angles(fixed, moving, rigid_matrix(translation=moving.origin-fixed.origin))
        info['fitted_relative_angles_deg'] = relative_angles(fixed, moving, retained[0].matrix)
        info['relative_rotation_change_deg'] = float(np.rad2deg(Rotation.from_matrix(retained[0].matrix[:3, :3]).magnitude()))
    return PairFit(fixed.name, moving.name, 'verified_geometric_reference' if retained else 'no_verified_articulation', retained, info)


def _fit_complementary(fixed, moving, profile, settings, info, started):
    from .seating import SeatingEvaluator

    # Retain the previous algorithm as a source of feasible initializations.
    # Its scores never decide the final complementary-seating ranking.
    baseline = fit_pair(fixed, moving, profile, replace(settings, objective='apposition'))
    evaluator = SeatingEvaluator(fixed, moving, profile, settings)
    scale = evaluator.scale
    proposed, gap_reports, warm = [], [], None
    gaps = list(settings.gap_fractions)
    middle = len(gaps)//2
    order = [gaps[middle]] + [g for i, g in enumerate(gaps) if i != middle]
    input_matrix = rigid_matrix(translation=moving.origin-fixed.origin)

    def candidate(matrix, gap, source, optimization=None):
        energy, diagnostics = evaluator.evaluate(matrix, gap, full=True, metrics=True)
        supports, _ = pair_supports(fixed, moving, profile['interfaces'], matrix,
            gap, settings.gap_width_fraction*scale, settings, None)
        if np.min(supports) < settings.minimum_interface_support:
            diagnostics['passes'] = False
            diagnostics['review_reasons'].append('insufficient gap/facing support')
        return Candidate(matrix, gap, energy, supports, collision_check(fixed, moving, matrix),
                         source, optimization or {}, diagnostics)

    for fraction in order:
        gap = fraction*scale
        coordinates = PairObjective(fixed, moving, profile, settings, gap)
        seeds = [c.matrix for c in baseline.candidates if np.isclose(c.gap, gap)]
        if baseline.candidates:
            seeds.append(baseline.candidates[0].matrix)
        if warm is not None:
            seeds.insert(0, warm)
        seeds.append(input_matrix)
        rotations = [fixed.frame[:3, :3] @ moving.frame[:3, :3].T]
        nf, nm = [], []
        for item in profile['interfaces']:
            nf.append(-evaluator.regions[fixed.name, item['fixed'], False].normal)
            nm.append(evaluator.regions[moving.name, item['moving'], False].normal)
        nf, nm = np.asarray(nf), np.asarray(nm)
        if min(np.linalg.matrix_rank(nf, tol=1e-5), np.linalg.matrix_rank(nm, tol=1e-5)) < 2:
            nf, nm = nf[:1], nm[:1]
        if np.linalg.norm(nf) > 1e-10 and np.linalg.norm(nm) > 1e-10:
            rotations.append(Rotation.align_vectors(nf, nm)[0].as_matrix())
        for rotation in rotations:
            translation = coordinates.posterior + coordinates.axes[:, 0]*gap - rotation @ coordinates.pivot
            seeds.append(rigid_matrix(rotation, translation))
        unique = []
        for matrix in seeds:
            x = coordinates.parameters(matrix)
            if (all(lo <= v <= hi for v, (lo, hi) in zip(x, coordinates.bounds))
                    and not any(np.allclose(matrix, previous, atol=1e-10, rtol=0) for previous in unique)):
                unique.append(matrix)
        ranked = sorted(enumerate(unique), key=lambda item: (
            evaluator.evaluate(item[1], gap) + 20*max(0., -float(evaluator.constraints(item[1]).min())), item[0]))
        candidates = [candidate(m, gap, 'initialization') for m in unique]
        selected = []
        for _, matrix in ranked:
            if all(Rotation.from_matrix(matrix[:3, :3] @ m[:3, :3].T).magnitude() > np.deg2rad(1.)
                   or np.linalg.norm(matrix[:3, 3]-m[:3, 3]) > .01*scale for m in selected):
                selected.append(matrix)
            if len(selected) >= settings.refine_candidates:
                break
        optimization_reports = []
        for seed in selected:
            x = coordinates.parameters(seed)
            evaluations, iterations, witnesses = 0, 0, 0
            for refinement in range(settings.collision_refinement_rounds+1):
                result = minimize(
                    lambda p: evaluator.evaluate(coordinates.matrix(p), gap), x,
                    method='SLSQP', bounds=coordinates.bounds,
                    constraints={'type': 'ineq', 'fun': lambda p: evaluator.constraints(coordinates.matrix(p))},
                    options={'maxiter': settings.seating_max_iterations, 'ftol': 2e-6, 'eps': 2e-6})
                x = result.x
                evaluations += int(result.nfev)
                iterations += int(result.nit)
                matrix = coordinates.matrix(x)
                opt = {'method': 'SLSQP', 'success': bool(result.success), 'message': str(result.message),
                       'evaluations': evaluations, 'iterations': iterations, 'refinement_rounds': refinement,
                       'collision_witnesses': witnesses,
                       'minimum_constraint': float(evaluator.constraints(matrix).min())}
                value = candidate(matrix, gap, 'complementary_seating', opt)
                candidates.append(value)
                if value.collision['verified']:
                    break
                if refinement == settings.collision_refinement_rounds:
                    break
                added = evaluator.add_collision_witnesses(matrix)
                witnesses += added
                if not added:
                    break
            optimization_reports.append(opt)
            # A final tiny opening is allowed only within the original bounds;
            # all seating criteria are recomputed after this adjustment.
            if not value.collision['verified']:
                for opening in (.0005, .001, .002, .005, .01):
                    adjusted = matrix.copy()
                    adjusted[:3, 3] += coordinates.axes[:, 0]*(opening*scale)
                    p = coordinates.parameters(adjusted)
                    if not all(lo <= v <= hi for v, (lo, hi) in zip(p, coordinates.bounds)):
                        continue
                    check = collision_check(fixed, moving, adjusted)
                    if check['verified']:
                        opt = dict(opt, axial_clearance_adjustment_mm=opening*scale)
                        candidates.append(candidate(adjusted, gap, 'seating_clearance_adjustment', opt))
                        break
        candidates.sort(key=lambda c: c.energy)
        valid = [c for c in candidates if c.collision['verified'] and c.seating['passes']]
        if valid:
            warm = valid[0].matrix
            for c in valid:
                if c.energy <= valid[0].energy+settings.ensemble_energy_slack and _distinct(c, proposed, scale):
                    proposed.append(c)
        clear = [c for c in candidates if c.collision['verified']]
        gap_reports.append({'gap_fraction': fraction, 'gap': gap,
                            'verified_candidates': len(valid),
                            'best_verified_energy': valid[0].energy if valid else None,
                            'best_attempt': candidates[0].as_dict() if candidates else None,
                            'best_collision_free_attempt': clear[0].as_dict() if clear else None,
                            'optimizations': optimization_reports})
    proposed.sort(key=lambda c: c.energy)
    retained = []
    for fraction in settings.gap_fractions:
        group = [c for c in proposed if np.isclose(c.gap, fraction*scale)]
        if group:
            retained.append(group[0])
    for c in proposed:
        if len(retained) >= max(settings.retain_candidates, len(settings.gap_fractions)):
            break
        if all(c is not old for old in retained):
            retained.append(c)
    retained.sort(key=lambda c: c.energy)
    info.update({'objective': settings.objective, 'gap_scenarios': gap_reports,
                 'retained_candidates': len(retained), 'elapsed_seconds': time.perf_counter()-started,
                 'baseline_status': baseline.status,
                 'input_relative_angles_deg': relative_angles(fixed, moving, input_matrix)})
    if retained:
        info['fitted_relative_angles_deg'] = relative_angles(fixed, moving, retained[0].matrix)
        info['relative_rotation_change_deg'] = float(np.rad2deg(Rotation.from_matrix(retained[0].matrix[:3, :3]).magnitude()))
    return PairFit(fixed.name, moving.name,
                   'verified_geometric_reference' if retained else 'no_verified_seating', retained, info)


def assemble_chain(bones, pair_fits, settings):
    """Select compatible pair candidates with global nonadjacent collision checks.

    With pairwise energies an open chain otherwise separates into independent
    joint fits. A bounded beam retains alternative combinations where global
    collisions couple those choices. Failed adjacencies split the sequence;
    missing vertebrae are never silently bridged.
    """
    segments, start = [], 0
    while start < len(bones):
        stop = start
        while stop < len(pair_fits) and pair_fits[stop].candidates: stop += 1
        beam = [(0., [bones[start].local_to_input], [])]
        exhausted = False
        for j in range(start, stop):
            extensions = []
            for energy, transforms, choices in beam:
                for k, c in enumerate(pair_fits[j].candidates):
                    world = transforms[-1] @ c.matrix
                    if all(collision_check(bones[start+i], bones[j+1], inverse_rigid(t) @ world)['verified']
                           for i, t in enumerate(transforms[:-1])):
                        consistency = 0.
                        if (j > start and settings.objective == 'complementary_seating'
                                and settings.chain_consistency_weight > 0):
                            previous = pair_fits[j-1].candidates[choices[-1]]
                            a = bones[j-1].frame[:3, :3].T @ previous.matrix[:3, :3] @ bones[j].frame[:3, :3]
                            b = bones[j].frame[:3, :3].T @ c.matrix[:3, :3] @ bones[j+1].frame[:3, :3]
                            angle = Rotation.from_matrix(b @ a.T).magnitude()
                            consistency = settings.chain_consistency_weight*np.log1p(
                                (angle/np.deg2rad(settings.chain_angle_scale_deg))**2)
                        extensions.append((energy+c.energy+consistency, transforms+[world], choices+[k]))
            if not extensions:
                exhausted = True
                break
            extensions.sort(key=lambda x: x[0])
            beam = extensions[:settings.chain_beam_width]
        energy, transforms, choices = beam[0]
        # A failed beam must not be presented as a complete collision-free chain.
        status = 'global_collision_conflict' if exhausted else ('isolated_bone' if stop == start else 'verified')
        segments.append({'start': start, 'stop': stop, 'status': status,
                         'energy': energy, 'local_to_world': transforms, 'candidate_indices': choices,
                         'pair_energy': sum(pair_fits[start+i].candidates[k].energy for i, k in enumerate(choices))})
        start = stop+1
    return segments
