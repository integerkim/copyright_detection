"""
motion_canonical.py
-------------------
Swing-only canonicalization of SMPL poses (applied to BOTH query and DB before the metric).

Why: mocap-retargeted SMPL and video-estimated SMPL (HybrIK) can reach the same joint positions
with very different local rotations (bone twist is invisible in joint positions). The rotation-based
metric then sees "different motion" for the same dance. We recompute every local rotation from the
joint positions as the minimal (swing) rotation that points the rest bone to the posed bone, so both
sources share one twist-free representation. Root orientation and translation are kept as-is.

Pure numpy / scipy (no SMPL model needed): rest joints of the neutral SMPL (betas = 0) are embedded.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
J_REST = np.array([
    [-0.00180, -0.22333, 0.02822], [0.06772, -0.31474, 0.02140], [-0.06947, -0.31386, 0.02390],
    [-0.00433, -0.11437, 0.00152], [0.10200, -0.68994, 0.01691], [-0.10776, -0.69642, 0.01505],
    [0.00116, 0.02081, 0.00262], [0.08841, -1.08790, -0.02679], [-0.09198, -1.09484, -0.02726],
    [0.00262, 0.07373, 0.02804], [0.11476, -1.14369, 0.09250], [-0.11735, -1.14298, 0.09609],
    [-0.00016, 0.28760, -0.01482], [0.08146, 0.19548, -0.00605], [-0.07914, 0.19257, -0.01058],
    [0.00499, 0.35257, 0.03653], [0.17244, 0.22595, -0.01492], [-0.17516, 0.22512, -0.01972],
    [0.43205, 0.21318, -0.04237], [-0.42890, 0.21179, -0.04112], [0.68128, 0.22216, -0.04355],
    [-0.68420, 0.21956, -0.04668], [0.76533, 0.21400, -0.05849], [-0.76882, 0.21344, -0.05699],
], dtype=np.float64)

# child used to define each joint's bone direction (joints without a child keep identity)
CHILD = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15,
         13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23}


def forward_kinematics(poses):
    """poses [T,24,3] axis-angle -> joint positions [T,24,3] in the body (root-aligned) frame."""
    T = poses.shape[0]
    Rl = R.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    G = np.zeros((T, 24, 3, 3))
    P = np.zeros((T, 24, 3))
    G[:, 0] = np.eye(3)  # root orientation removed
    P[:, 0] = J_REST[0]
    for j in range(1, 24):
        p = PARENTS[j]
        G[:, j] = G[:, p] @ Rl[:, j]
        P[:, j] = P[:, p] + np.einsum("tij,j->ti", G[:, p], J_REST[j] - J_REST[p])
    return P


def _swing(a, b):
    """Minimal rotation (matrices [T,3,3]) taking unit vectors a[T,3] to b[T,3]."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=-1, keepdims=True)
    c = np.sum(a * b, axis=-1, keepdims=True)
    ang = np.arctan2(s, c)
    axis = np.where(s > 1e-8, axis / (s + 1e-12), np.array([1.0, 0.0, 0.0]))
    return R.from_rotvec(axis * ang).as_matrix()


def canonicalize(poses):
    """poses [T,24,3] -> twist-free poses [T,24,3] (same joint positions, root orientation kept)."""
    poses = np.asarray(poses, dtype=np.float64)
    T = poses.shape[0]
    P = forward_kinematics(poses)
    G = np.zeros((T, 24, 3, 3))
    G[:, 0] = np.eye(3)
    out = np.zeros_like(poses)
    out[:, 0] = poses[:, 0]
    for j in range(1, 24):
        Gp = G[:, PARENTS[j]]
        if j in CHILD:
            c = CHILD[j]
            rest = np.broadcast_to(J_REST[c] - J_REST[j], (T, 3))
            cur = np.einsum("tji,tj->ti", Gp, P[:, c] - P[:, j])  # posed bone in parent frame
            Rl = _swing(rest, cur)
        else:
            Rl = np.tile(np.eye(3), (T, 1, 1))
        G[:, j] = Gp @ Rl
        out[:, j] = R.from_matrix(Rl).as_rotvec()
    return out.astype(np.float32)
