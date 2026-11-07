"""
motion_metric.py
----------------
Motion similarity metric (PSC / AC / DC -> S_FINAL).
Identical to the metric used in the APSIPA paper experiments (newmetric.py),
with file-path config / plotting removed so it can be imported as a module.

Input : two SMPL motion sequences, poses_axis_angle [T, 24, 3] (30 fps)
Output: S_PSC, S_AC, S_DC, S_FINAL in [0, 1]
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

USE_DTW = True
MIRROR_AWARE = True

# Final weights
# These should later be selected by validation/grid search.
LAMBDA_PSC = 0.50
LAMBDA_AC  = 0.30
LAMBDA_DC  = 0.20

# DTW / local correlation mixing
ALPHA_DTW  = 0.20
ALPHA_CORR = 0.80

SIGMA_PSC = 2.0
SIGMA_AC  = 2.0
SIGMA_DC  = 2.0

EPS = 1e-6

# ============================================================
# SMPL JOINT INDEX
# ============================================================
# SMPL 24 joints:
# 0 pelvis/root
# 1 left_hip, 2 right_hip, 3 spine1
# 4 left_knee, 5 right_knee, 6 spine2
# 7 left_ankle, 8 right_ankle, 9 spine3
# 10 left_foot, 11 right_foot, 12 neck
# 13 left_collar, 14 right_collar, 15 head
# 16 left_shoulder, 17 right_shoulder
# 18 left_elbow, 19 right_elbow
# 20 left_wrist, 21 right_wrist
# 22 left_hand, 23 right_hand

LR_PAIRS = [
    (1, 2), (4, 5), (7, 8), (10, 11),
    (13, 14), (16, 17), (18, 19),
    (20, 21), (22, 23)
]

LEFT_ARM  = [13, 16, 18, 20, 22]
RIGHT_ARM = [14, 17, 19, 21, 23]
LEFT_LEG  = [1, 4, 7, 10]
RIGHT_LEG = [2, 5, 8, 11]

UPPER_BODY = [3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_BODY = [1, 2, 4, 5, 7, 8, 10, 11]
ARMS = LEFT_ARM + RIGHT_ARM
LEGS = LEFT_LEG + RIGHT_LEG

# PSC is intentionally upper-body focused because hand/arm gestures often
# carry distinctive choreography signatures.
UPPER_PSC_JOINTS = [
    3, 6, 9,        # spine
    12, 15,         # neck, head
    13, 14,         # collar
    16, 17,         # shoulders
    18, 19,         # elbows
    20, 21,         # wrists
    22, 23          # hands
]

# ============================================================
# UTILS
# ============================================================

def load_motion(path):
    data = np.load(path, allow_pickle=True)
    poses = data["poses_axis_angle"].astype(np.float32)  # [T, 24, 3]
    trans = data["trans"].astype(np.float32)             # kept for compatibility/mirroring
    return poses, trans


def normalize_pair(f_ref, f_query):
    """
    Pairwise normalization is acceptable for pairwise similarity inspection.
    If a global threshold is claimed in the paper, replace this with
    dataset-level normalization estimated on a validation split.
    """
    all_feat = np.concatenate([f_ref, f_query], axis=0)
    mu = all_feat.mean(axis=0, keepdims=True)
    std = all_feat.std(axis=0, keepdims=True) + EPS
    return (f_ref - mu) / std, (f_query - mu) / std


def rotation_angle_from_matrix(Rmat):
    trace = np.trace(Rmat, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos)


def rotation_distance_rotvec(a, b):
    """
    Geodesic distance between two axis-angle rotations.
    a, b: [..., 3]
    return: [...]
    """
    Ra = R.from_rotvec(a.reshape(-1, 3)).as_matrix()
    Rb = R.from_rotvec(b.reshape(-1, 3)).as_matrix()
    Rrel = np.matmul(np.transpose(Ra, (0, 2, 1)), Rb)
    ang = rotation_angle_from_matrix(Rrel)
    return ang.reshape(a.shape[:-1])


def rotvec_to_6d(poses):
    """
    Convert axis-angle rotations to continuous 6D rotation representation.
    This reduces axis-angle discontinuity for frame-wise pose signatures.
    poses: [T, J, 3]
    return: [T, J, 6]
    """
    T, J, _ = poses.shape
    mat = R.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(T, J, 3, 3)
    return mat[:, :, :, :2].reshape(T, J, 6).astype(np.float32)


def mirror_motion(poses, trans):
    poses_m = poses.copy()
    trans_m = trans.copy()

    M = np.diag([-1.0, 1.0, 1.0])

    rot = R.from_rotvec(poses_m.reshape(-1, 3)).as_matrix()
    rot = M @ rot @ M
    poses_m = R.from_matrix(rot).as_rotvec().reshape(poses.shape)

    for l, r in LR_PAIRS:
        poses_m[:, [l, r], :] = poses_m[:, [r, l], :]

    trans_m[:, 0] *= -1.0
    return poses_m.astype(np.float32), trans_m.astype(np.float32)


def group_mean(x, joints):
    return x[:, joints].mean(axis=1, keepdims=True)


def group_sum(x, joints):
    return x[:, joints].sum(axis=1, keepdims=True)


def safe_ratio(a, b):
    return a / (b + EPS)


def temporal_local_corr_1d(a, b, window=25):
    """
    Local correlation between two 1D temporal signals.
    Returns a per-frame curve in [0, 1].
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    T = len(a)
    out = np.zeros((T, 1), dtype=np.float32)

    for t in range(T):
        s = max(0, t - window)
        e = min(T, t + window + 1)

        aa = a[s:e]
        bb = b[s:e]

        aa = (aa - aa.mean()) / (aa.std() + EPS)
        bb = (bb - bb.mean()) / (bb.std() + EPS)

        corr = float(np.mean(aa * bb))
        out[t, 0] = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)

    return out

# ============================================================
# PSC: POSE SIGNATURE CONSISTENCY
# ============================================================

def pose_signature_feature(poses):
    """
    PSC: "What static pose configuration is performed?"

    Important:
    - No velocity.
    - No acceleration.
    - No temporal dynamics.
    - Uses continuous 6D rotation representation instead of raw axis-angle.

    This term is intended to capture pose identity/signature only.
    """
    body_6d = rotvec_to_6d(poses[:, UPPER_PSC_JOINTS, :])  # [T, J, 6]
    T = body_6d.shape[0]

    # Static upper-body pose configuration.
    pose_config = body_6d.reshape(T, -1)

    # Static articulation amount per joint.
    # This keeps pose magnitude but does not include transition speed.
    pose_mag = np.linalg.norm(poses[:, UPPER_PSC_JOINTS, :], axis=-1)

    feat = np.concatenate([
        pose_config,
        pose_mag,
    ], axis=1)

    return feat.astype(np.float32)

# ============================================================
# AC: ARTICULATION / COORDINATION CONSISTENCY
# ============================================================

def articulation_feature(poses):
    """
    AC: "How are body parts coordinated?"

    This term avoids raw pose configuration as much as possible.
    It uses relationship-level and group-level coordination features:
    - left-right symmetry
    - upper-lower coupling
    - arm-leg energy ratio
    - body-part activation pattern
    - local phase/correlation-like coupling over time

    The feature is based on angular motion energy rather than raw pose.
    """
    T = poses.shape[0]
    body = poses[:, 1:, :]  # exclude root for motion energy

    # Angular velocity as geodesic distance between consecutive rotations.
    vel = np.zeros((T, 24), dtype=np.float32)
    if T > 1:
        vel[1:, 1:] = rotation_distance_rotvec(body[1:], body[:-1])

    # Per-frame group energy.
    e_left_arm  = group_sum(vel, LEFT_ARM)
    e_right_arm = group_sum(vel, RIGHT_ARM)
    e_left_leg  = group_sum(vel, LEFT_LEG)
    e_right_leg = group_sum(vel, RIGHT_LEG)
    e_upper     = group_sum(vel, UPPER_BODY)
    e_lower     = group_sum(vel, LOWER_BODY)
    e_arms      = group_sum(vel, ARMS)
    e_legs      = group_sum(vel, LEGS)
    e_total     = vel[:, 1:].sum(axis=1, keepdims=True) + EPS

    # Symmetry / imbalance patterns.
    lr_arm_diff = np.abs(e_left_arm - e_right_arm)
    lr_leg_diff = np.abs(e_left_leg - e_right_leg)

    lr_arm_ratio = np.abs(e_left_arm - e_right_arm) / (e_left_arm + e_right_arm + EPS)
    lr_leg_ratio = np.abs(e_left_leg - e_right_leg) / (e_left_leg + e_right_leg + EPS)

    # Upper/lower and arm/leg activation ratios.
    upper_ratio = safe_ratio(e_upper, e_total)
    lower_ratio = safe_ratio(e_lower, e_total)
    arm_ratio   = safe_ratio(e_arms, e_total)
    leg_ratio   = safe_ratio(e_legs, e_total)

    upper_lower_balance = np.abs(e_upper - e_lower) / (e_upper + e_lower + EPS)
    arm_leg_balance     = np.abs(e_arms - e_legs) / (e_arms + e_legs + EPS)

    # Local temporal coupling within the same motion.
    # These values describe how body parts co-activate over time.
    c_left_right_arm = temporal_local_corr_1d(e_left_arm, e_right_arm)
    c_left_right_leg = temporal_local_corr_1d(e_left_leg, e_right_leg)
    c_upper_lower    = temporal_local_corr_1d(e_upper, e_lower)
    c_arm_leg        = temporal_local_corr_1d(e_arms, e_legs)

    feat = np.concatenate([
        lr_arm_diff,
        lr_leg_diff,
        lr_arm_ratio,
        lr_leg_ratio,
        upper_ratio,
        lower_ratio,
        arm_ratio,
        leg_ratio,
        upper_lower_balance,
        arm_leg_balance,
        c_left_right_arm,
        c_left_right_leg,
        c_upper_lower,
        c_arm_leg,
    ], axis=1)

    return feat.astype(np.float32)

# ============================================================
# DC: DYNAMIC FLOW CONSISTENCY
# ============================================================

def dynamic_flow_feature(poses):
    """
    DC: "How does the motion evolve over time?"

    This term focuses on group-level temporal flow:
    - angular velocity energy
    - angular acceleration energy
    - motion intensity curve
    - upper/lower flow
    - temporal derivative of motion energy

    It intentionally avoids high-dimensional raw joint pose configuration.
    """
    T = poses.shape[0]
    body = poses[:, 1:, :]  # [T, 23, 3]

    pose_vel = np.zeros((T, 23), dtype=np.float32)
    if T > 1:
        pose_vel[1:] = rotation_distance_rotvec(body[1:], body[:-1])

    pose_acc = np.zeros((T, 23), dtype=np.float32)
    if T > 2:
        pose_acc[2:] = np.abs(pose_vel[2:] - pose_vel[1:-1])

    # Convert SMPL indices to body array indices after root removal.
    upper_idx = [j - 1 for j in UPPER_BODY]
    lower_idx = [j - 1 for j in LOWER_BODY]
    arm_idx   = [j - 1 for j in ARMS]
    leg_idx   = [j - 1 for j in LEGS]

    mean_vel = pose_vel.mean(axis=1, keepdims=True)
    max_vel  = pose_vel.max(axis=1, keepdims=True)
    mean_acc = pose_acc.mean(axis=1, keepdims=True)
    max_acc  = pose_acc.max(axis=1, keepdims=True)

    kinetic = np.mean(pose_vel ** 2, axis=1, keepdims=True)

    upper_vel = pose_vel[:, upper_idx].mean(axis=1, keepdims=True)
    lower_vel = pose_vel[:, lower_idx].mean(axis=1, keepdims=True)
    arm_vel   = pose_vel[:, arm_idx].mean(axis=1, keepdims=True)
    leg_vel   = pose_vel[:, leg_idx].mean(axis=1, keepdims=True)

    upper_acc = pose_acc[:, upper_idx].mean(axis=1, keepdims=True)
    lower_acc = pose_acc[:, lower_idx].mean(axis=1, keepdims=True)
    arm_acc   = pose_acc[:, arm_idx].mean(axis=1, keepdims=True)
    leg_acc   = pose_acc[:, leg_idx].mean(axis=1, keepdims=True)

    # Energy derivative: captures rising/falling movement intensity.
    kinetic_diff = np.zeros_like(kinetic)
    if T > 1:
        kinetic_diff[1:] = np.abs(kinetic[1:] - kinetic[:-1])

    # Peak-like flow cue.
    vel_peak_ratio = safe_ratio(max_vel, mean_vel + EPS)
    acc_peak_ratio = safe_ratio(max_acc, mean_acc + EPS)

    feat = np.concatenate([
        mean_vel,
        max_vel,
        mean_acc,
        max_acc,
        kinetic,
        kinetic_diff,
        upper_vel,
        lower_vel,
        arm_vel,
        leg_vel,
        upper_acc,
        lower_acc,
        arm_acc,
        leg_acc,
        vel_peak_ratio,
        acc_peak_ratio,
    ], axis=1)

    return feat.astype(np.float32)

# ============================================================
# DTW + CORRELATION SIMILARITY
# ============================================================

def dtw_path(f1, f2):
    D = cdist(f1, f2, metric="euclidean")
    n, m = D.shape

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = D[i - 1, j - 1] + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        step = np.argmin([
            cost[i - 1, j],
            cost[i, j - 1],
            cost[i - 1, j - 1],
        ])

        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return path


def direct_path(f1, f2):
    T = min(len(f1), len(f2))
    return [(i, i) for i in range(T)]


def dtw_similarity(f_ref, f_query, path, sigma):
    dists = np.array([
        np.linalg.norm(f_ref[i] - f_query[j]) / np.sqrt(f_ref.shape[1])
        for i, j in path
    ], dtype=np.float32)

    sims = np.exp(-((dists / sigma) ** 2))
    return float(np.mean(sims)), sims.astype(np.float32), dists.astype(np.float32)


def sequence_correlation_similarity(f_ref, f_query):
    T = min(len(f_ref), len(f_query))
    a = f_ref[:T]
    b = f_query[:T]

    a = (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + EPS)
    b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + EPS)

    corr_per_dim = np.mean(a * b, axis=0)
    corr = np.mean(corr_per_dim)

    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def local_correlation_curve(f_ref, f_query, path, window=25):
    aligned_ref = np.array([f_ref[i] for i, j in path])
    aligned_q = np.array([f_query[j] for i, j in path])

    N = len(aligned_ref)
    curve = np.zeros(N, dtype=np.float32)

    for t in range(N):
        s = max(0, t - window)
        e = min(N, t + window + 1)

        a = aligned_ref[s:e]
        b = aligned_q[s:e]

        a = (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + EPS)
        b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + EPS)

        corr = np.mean(a * b)
        curve[t] = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)

    return curve


def compute_common_path(psc_r, ac_r, dc_r, psc_q, ac_q, dc_q):
    psc_rn, psc_qn = normalize_pair(psc_r, psc_q)
    ac_rn, ac_qn = normalize_pair(ac_r, ac_q)
    dc_rn, dc_qn = normalize_pair(dc_r, dc_q)

    combo_r = np.concatenate([psc_rn, ac_rn, dc_rn], axis=1)
    combo_q = np.concatenate([psc_qn, ac_qn, dc_qn], axis=1)

    return dtw_path(combo_r, combo_q) if USE_DTW else direct_path(combo_r, combo_q)


def term_similarity(f_ref, f_query, path, sigma):
    f_ref_n, f_query_n = normalize_pair(f_ref, f_query)

    S_dtw, curve_dtw, dists = dtw_similarity(f_ref_n, f_query_n, path, sigma)
    S_corr = sequence_correlation_similarity(f_ref_n, f_query_n)
    curve_corr = local_correlation_curve(f_ref_n, f_query_n, path)

    curve = ALPHA_DTW * curve_dtw + ALPHA_CORR * curve_corr
    score = float(np.mean(curve))

    return score, curve.astype(np.float32), dists, S_dtw, S_corr


def pair_term_correlation(curves):
    """
    Correlation among PSC/AC/DC curves.
    Used for complementarity/orthogonality analysis.
    """
    names = ["PSC", "AC", "DC"]
    mat = np.eye(3, dtype=np.float32)

    for i in range(3):
        for j in range(i + 1, 3):
            a = curves[names[i]]
            b = curves[names[j]]
            T = min(len(a), len(b))
            aa = (a[:T] - a[:T].mean()) / (a[:T].std() + EPS)
            bb = (b[:T] - b[:T].mean()) / (b[:T].std() + EPS)
            corr = float(np.mean(aa * bb))
            mat[i, j] = mat[j, i] = corr

    return names, mat

# ============================================================
# MAIN SIMILARITY
# ============================================================

def compute_once(poses_r, poses_q):
    psc_r = pose_signature_feature(poses_r)
    ac_r  = articulation_feature(poses_r)
    dc_r  = dynamic_flow_feature(poses_r)

    psc_q = pose_signature_feature(poses_q)
    ac_q  = articulation_feature(poses_q)
    dc_q  = dynamic_flow_feature(poses_q)

    path = compute_common_path(psc_r, ac_r, dc_r, psc_q, ac_q, dc_q)

    S_PSC, curve_psc, dist_psc, S_PSC_DTW, S_PSC_CORR = term_similarity(
        psc_r, psc_q, path, SIGMA_PSC
    )

    S_AC, curve_ac, dist_ac, S_AC_DTW, S_AC_CORR = term_similarity(
        ac_r, ac_q, path, SIGMA_AC
    )

    S_DC, curve_dc, dist_dc, S_DC_DTW, S_DC_CORR = term_similarity(
        dc_r, dc_q, path, SIGMA_DC
    )

    curve_final = (
        LAMBDA_PSC * curve_psc +
        LAMBDA_AC  * curve_ac +
        LAMBDA_DC  * curve_dc
    ).astype(np.float32)

    S_FINAL = float(np.mean(curve_final))

    term_names, term_corr = pair_term_correlation({
        "PSC": curve_psc,
        "AC": curve_ac,
        "DC": curve_dc,
    })

    return {
        "S_PSC": S_PSC,
        "S_AC": S_AC,
        "S_DC": S_DC,
        "S_FINAL": S_FINAL,

        "curve_psc": curve_psc,
        "curve_ac": curve_ac,
        "curve_dc": curve_dc,
        "curve_final": curve_final,

        "dist_psc": dist_psc,
        "dist_ac": dist_ac,
        "dist_dc": dist_dc,

        "dtw_corr": {
            "PSC": (S_PSC_DTW, S_PSC_CORR),
            "AC":  (S_AC_DTW, S_AC_CORR),
            "DC":  (S_DC_DTW, S_DC_CORR),
        },

        "term_corr_names": term_names,
        "term_corr_matrix": term_corr,
        "path_length": len(path),

        "feature_dims": {
            "PSC": psc_r.shape[1],
            "AC": ac_r.shape[1],
            "DC": dc_r.shape[1],
        }
    }


def compute_similarity(ref_path, query_path):
    poses_r, trans_r = load_motion(ref_path)
    poses_q, trans_q = load_motion(query_path)

    normal = compute_once(poses_r, poses_q)
    normal["mirror_used"] = False

    if not MIRROR_AWARE:
        return normal

    poses_q_m, _ = mirror_motion(poses_q, trans_q)
    mirrored = compute_once(poses_r, poses_q_m)
    mirrored["mirror_used"] = True


    return mirrored if mirrored["S_FINAL"] > normal["S_FINAL"] else normal
