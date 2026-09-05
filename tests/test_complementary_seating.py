"""Known nonplanar complementary surfaces, distinct from flat apposition tests."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from test_articulation import PROFILE
from svrom_preparation.data import (Bone, Landmarks, inverse_rigid, rigid_matrix, transform_points)
from svrom_preparation.workflow import write_obj
from svrom_preparation.surfaces import prepare_regions, collision_check
from svrom_preparation.fitting import fit_pair
from svrom_preparation.settings import ArticulationSettings
from svrom_preparation.seating import SeatingEvaluator

def height(x,y):
    return .32*np.exp(-((x-.15)/.65)**2-((y+.10)/.35)**2)+.12*np.exp(-((x+.5)/.35)**2-((y-.3)/.3)**2)

def key_bone(path,name,upper,perturb):
    n=19; xx,yy=np.meshgrid(np.linspace(-1.2,1.2,n),np.linspace(-1.2,1.2,n));z=height(xx,yy)
    if upper:
        low=z+.04;high=np.ones_like(z)*1.2
        lm=np.array([[.04,.02,height(.04,.02)+.04],[.04,.02,height(.04,.02)+1.04],[0,1,.7],[-.8,0,.7],[.8,0,.7]])
    else:
        low=np.ones_like(z)*-.8;high=z
        lm=np.array([[.04,.02,height(.04,.02)-1],[.04,.02,height(.04,.02)],[0,1,0],[-.8,0,0],[.8,0,0]])
    v=np.r_[np.c_[xx.ravel(),yy.ravel(),low.ravel()],np.c_[xx.ravel(),yy.ravel(),high.ravel()]]
    faces=[]; m=n*n
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;b=a+1;c=a+n;d=c+1
            faces += [[a,c,b],[b,c,d],[a+m,b+m,c+m],[b+m,d+m,c+m]]
    boundary=list(range(n))+[j*n+n-1 for j in range(1,n)]+list(range(m-2,m-n-1,-1))+[j*n for j in range(n-2,0,-1)]
    for a,b in zip(boundary,boundary[1:]+boundary[:1]):
        faces += [[a,b,a+m],[b,b+m,a+m]]
    mesh=trimesh.Trimesh(v,faces,process=False);mesh.fix_normals()
    write_obj(path/f'{name}.obj',transform_points(v,perturb),mesh.faces)
    bone=Bone.load(path/f'{name}.obj');bone.set_landmarks(Landmarks(transform_points(lm,perturb),tuple(map(str,range(5))),tuple(map(str,range(5)))),PROFILE)
    prepare_regions(bone,PROFILE);return bone


def test_known_nonplanar_key_and_socket_recovery(tmp_path):
    # A nonsymmetric, two-bump surface and its matching recess. Guide lengths
    # match so differences in regional extent do not redefine the ground truth.
    # The true gap is vertical; the loss uses normal offsets, allowing a small
    # geometric-model discrepancy even with perfect source coordinates.
    perturb = rigid_matrix(Rotation.from_euler('xyz', [4, -3, 12], degrees=True).as_matrix(), [.12, -.08, .03])
    fixed = key_bone(tmp_path, 'socket', False, np.eye(4))
    moving = key_bone(tmp_path, 'key', True, perturb)
    true_matrix = inverse_rigid(fixed.local_to_input) @ inverse_rigid(perturb) @ moving.local_to_input
    settings = ArticulationSettings(gap_fractions=(.04,), refine_candidates=2,
        seating_max_iterations=50, max_evaluations=350, sample_count=160)
    evaluator = SeatingEvaluator(fixed, moving, PROFILE, settings)
    exact_energy, metrics = evaluator.evaluate(true_matrix, .04, full=True, metrics=True)
    assert metrics['passes'] and collision_check(fixed, moving, true_matrix)['verified']
    supplied = rigid_matrix(translation=moving.origin-fixed.origin)
    assert evaluator.evaluate(supplied, .04, full=True) > exact_energy + .1
    fit = fit_pair(fixed, moving, PROFILE, settings)
    assert fit.candidates, fit.report
    selected = fit.candidates[0]
    error = inverse_rigid(true_matrix) @ selected.matrix
    assert np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude()) < 1.
    assert np.linalg.norm(error[:3, 3]) < .01
    assert selected.seating['passes'] and selected.collision['verified']
