"""
얼굴 Geometry 피처 5종 추출, 유사도 계산
------------------------------------------------------
피처
  F1 눈 사이 간격   : |p39 - p42| / W
  F2 눈꼬리 기울기  : 롤 보정 후 (내안각 -> 외안각) 선의 상승 각도, 양쪽 평균 [deg]
  F3 코 길이        : |p27 - p33| / W
  F4 코 폭          : |p31 - p35| / W
  F5 턱 끝 각도     : ∠(p6, p8, p10) [deg]
  W (정규화 기준)   : |p0 - p16|  (얼굴 상부 폭)

비교 방식
  정면 사진만 쓰지 않고, reference와 target에서 얼굴 각도가 같은 프레임끼리 비교한다.
  같은 각도에서는 원근에 의한 왜곡이 양쪽에 똑같이 생기므로 정면이 아니어도 피처를 비교할 수 있다.
  결과는 results/<날짜_시간>/ 아래 run.log, result.json, figures/ 로 저장된다.
"""
import os
import json
import logging
from collections import defaultdict
from datetime import datetime
import numpy as np
import cv2

# ---------------------------------------------------------------- 랜드마크 인덱스
R_OUT, R_IN, L_IN, L_OUT = 36, 39, 42, 45      # 오른눈 끝/앞머리, 왼눈 앞머리/끝 (피사체 기준)
NASION, SUBNASALE = 27, 33                      # 콧대 시작점, 코 밑
ALAR_R, ALAR_L = 31, 35                         # 콧망울 바깥쪽
JAW_R, MENTON, JAW_L = 6, 8, 10                 # 턱선 좌우 점, 턱 끝
FACE_R, FACE_L = 0, 16                          # 얼굴 상부 폭
LIP_IN_UP, LIP_IN_LO = 62, 66                   # 입 안쪽 위/아래 (입 다묾 판정용)

FEATURES = ["intercanthal", "canthal_tilt", "nose_length", "nose_width", "chin_angle"]


# ================================================================ 1. 랜드마크 추출
class LandmarkDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        import dlib  # 지연 import: 피처 함수만 쓸 때는 dlib 불필요
        self.det = dlib.get_frontal_face_detector()
        self.pred = dlib.shape_predictor(predictor_path)

    def __call__(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det(gray, 1)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: r.width() * r.height())  # 가장 큰 얼굴
        shape = self.pred(gray, face)
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)


# ================================================================ 2. 기하 유틸
def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_at(a, b, c):
    """꼭짓점 b에서 벡터 (a-b), (c-b)가 이루는 각 [deg]"""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def roll_angle(lm):
    """두 눈 앞머리를 잇는 선의 기울기 = 머리 롤 [rad]"""
    d = lm[L_IN] - lm[R_IN]
    return float(np.arctan2(d[1], d[0]))


def roll_align(lm):
    """두 눈 앞머리 선이 수평이 되도록 회전 (원점 = 두 앞머리의 중점)"""
    phi = roll_angle(lm)
    c, s = np.cos(-phi), np.sin(-phi)
    R = np.array([[c, -s], [s, c]])
    center = (lm[R_IN] + lm[L_IN]) / 2
    return (lm - center) @ R.T


# ================================================================ 3. 피처 정의
def extract_features(lm):
    """68x2 랜드마크 -> 피처 dict (이미지 좌표계, y축 아래 방향)"""
    W = dist(lm[FACE_R], lm[FACE_L])

    # F2: 롤 보정 후, 바깥 눈꼬리가 앞머리보다 위(y 작음)에 있으면 +
    a = roll_align(lm)
    def tilt(inner, outer):
        dx = abs(a[outer, 0] - a[inner, 0])
        dy = a[inner, 1] - a[outer, 1]          # y축이 아래 방향이므로 부호 반전
        return np.degrees(np.arctan2(dy, dx))
    tilt_r, tilt_l = tilt(R_IN, R_OUT), tilt(L_IN, L_OUT)

    return {
        "intercanthal": dist(lm[R_IN], lm[L_IN]) / W,
        "canthal_tilt": float((tilt_r + tilt_l) / 2),
        "nose_length":  dist(lm[NASION], lm[SUBNASALE]) / W,
        "nose_width":   dist(lm[ALAR_R], lm[ALAR_L]) / W,
        "chin_angle":   angle_at(lm[JAW_R], lm[MENTON], lm[JAW_L]),
        # 참고용 (설명·디버깅)
        "_W_px": W, "_tilt_r": float(tilt_r), "_tilt_l": float(tilt_l),
    }


# ================================================================ 4. 프레임 품질 판정
# 일반 성인 얼굴 3D 모델 (단위 임의, x 오른쪽 / y 아래 / z 카메라에서 멀어지는 방향)
_MODEL_3D = np.array([
    [0.0,     0.0,   0.0],    # 30 코끝
    [0.0,   330.0,  65.0],    # 8  턱 끝
    [-225.0, -170.0, 135.0],  # 36 오른눈 끝
    [225.0, -170.0, 135.0],   # 45 왼눈 끝
    [-150.0,  150.0, 125.0],  # 48 오른 입꼬리
    [150.0,  150.0, 125.0],   # 54 왼 입꼬리
])
_MODEL_IDX = [30, 8, 36, 45, 48, 54]


def head_pose(lm, img_shape):
    """solvePnP로 (pitch, yaw, roll) [deg] 추정"""
    h, w = img_shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, lm[_MODEL_IDX], K, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
    return float(pitch), float(yaw), float(roll)


def mouth_open_ratio(lm):
    return dist(lm[LIP_IN_UP], lm[LIP_IN_LO]) / dist(lm[NASION], lm[SUBNASALE])


def frame_ok(lm, img_shape, max_yaw=15, max_pitch=15, max_mouth=0.10):
    pose = head_pose(lm, img_shape)
    if pose is None:
        return False, None
    pitch, yaw, _ = pose
    ok = abs(yaw) <= max_yaw and abs(pitch) <= max_pitch and mouth_open_ratio(lm) <= max_mouth
    return ok, pose



# ================================================================ 5. 프레임 수집 (이미지 · 영상 공통)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

POSE_BIN = 10              # 영상끼리 비교할 때 각도 구간 폭 [deg]
POSE_TOL = 10              # 이미지가 끼면, 이미지 각도 ± 이 범위의 프레임과 비교 [deg]
MAX_YAW, MAX_PITCH = 45, 45          # 이 범위를 넘는 프레임은 랜드마크 신뢰도가 낮아 버림
MOUTH_OPEN_MAX = 0.10                # 입을 벌린 프레임은 턱 끝 각도만 제외
FEATURE_MAX_YAW = {                  # 피처별로 사용할 수 있는 최대 |yaw| [deg]
    "intercanthal": 15,              # 고개가 돌아가면 눈 사이 간격은 제외
    "canthal_tilt": 45, "nose_length": 45, "nose_width": 45, "chin_angle": 45,
}
MIN_FRAMES = 1             # 영상에서 한 각도 구간을 비교에 쓰기 위한 최소 프레임 수

log = logging.getLogger("face_geometry")


def pose_bin(yaw, pitch, width=POSE_BIN):
    return int(round(yaw / width)) * width, 0


def frame_record(frame, lm, t):
    """프레임 1장 -> 포즈 + 피처 기록 (각도가 너무 크면 None)"""
    pose = head_pose(lm, frame.shape)
    if pose is None:
        return None
    pitch, yaw, roll = pose
    if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
        return None
    f = extract_features(lm)
    mo = mouth_open_ratio(lm)
    feats = {k: f[k] for k in FEATURES}
    if mo > MOUTH_OPEN_MAX:
        feats["chin_angle"] = float("nan")
    return {"t": t, "yaw": yaw, "pitch": pitch, "roll": roll, "mouth_open": mo,
            "bin": pose_bin(yaw, pitch), "feats": feats, "lm": lm}


def collect(path, detector, step=5):
    """이미지 또는 영상 -> 유효 프레임 기록 리스트"""
    kind = "image" if os.path.splitext(path)[1].lower() in IMG_EXT else "video"
    records, n_sampled, n_face = [], 0, 0
    if kind == "image":
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"{path} (경로에 한글이 있으면 OpenCV가 못 읽을 수 있음)")
        frames = [(0, img)]
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        def gen():
            idx = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield idx, fr
                idx += 1
            cap.release()
        frames = gen()
    for t, fr in frames:
        n_sampled += 1
        lm = detector(fr)
        if lm is None:
            continue
        n_face += 1
        r = frame_record(fr, lm, t)
        if r is not None:
            records.append(r)
    src = {"path": path, "kind": kind, "records": records,
           "n_sampled": n_sampled, "n_face": n_face, "n_valid": len(records)}
    if records:
        yaws = [r["yaw"] for r in records]
        pits = [r["pitch"] for r in records]
        log.info(f"[{os.path.basename(path)}] {kind}: 샘플 {n_sampled} / 얼굴 검출 {n_face} / 유효 {len(records)}"
                 f"  (yaw {min(yaws):+.1f}° ~ {max(yaws):+.1f}°, pitch {min(pits):+.1f}° ~ {max(pits):+.1f}°)")
    else:
        log.warning(f"[{os.path.basename(path)}] {kind}: 유효 프레임 없음 (샘플 {n_sampled}, 얼굴 검출 {n_face})")
    return src


def load_frame(src, t):
    if src["kind"] == "image":
        return cv2.imread(src["path"])
    cap = cv2.VideoCapture(src["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


# ================================================================ 6. 같은 각도끼리 비교 · 판단
Z_TOL = 1.5                # z가 이 값일 때 s ≈ 0.61 (판정 기준과 맞춤)
Z_MATCH, Z_SIMILAR = 1.5, 3.0


def matched_groups(ref, tgt):
    """
    reference / target에서 같은 얼굴 각도끼리 묶은 그룹 목록
      - 영상 vs 영상 : (yaw, pitch)를 POSE_BIN 폭 구간으로 나눠 같은 구간끼리
      - 이미지가 낀 경우 : 이미지 각도 ± POSE_TOL 안의 프레임끼리
    반환: [(yaw중심, pitch중심, ref기록들, tgt기록들), ...]
    """
    groups = []
    if ref["kind"] == "video" and tgt["kind"] == "video":
        rb, tb = defaultdict(list), defaultdict(list)
        for r in ref["records"]:
            rb[r["bin"]].append(r)
        for r in tgt["records"]:
            tb[r["bin"]].append(r)
        for b in sorted(set(rb) & set(tb)):
            if len(rb[b]) >= MIN_FRAMES and len(tb[b]) >= MIN_FRAMES:
                groups.append((b[0], b[1], rb[b], tb[b]))
    else:
        anchor_src, other_src = (ref, tgt) if ref["kind"] == "image" else (tgt, ref)
        for a in anchor_src["records"]:
            near = [r for r in other_src["records"]
                    if abs(r["yaw"] - a["yaw"]) <= POSE_TOL and abs(r["pitch"] - a["pitch"]) <= 25]
            need = 1 if other_src["kind"] == "image" else MIN_FRAMES
            if len(near) >= need:
                rr, tt = ([a], near) if anchor_src is ref else (near, [a])
                groups.append((a["yaw"], a["pitch"], rr, tt))
    return groups


def _median(records, k):
    v = np.array([r["feats"][k] for r in records], dtype=float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if v.size else float("nan")


def compare(ref, tgt, sigma, weight, tau):
    """같은 각도 그룹마다 피처를 비교하고, 프레임 수로 가중평균해 최종 점수 산출"""
    groups = matched_groups(ref, tgt)
    result = {"reference": ref["path"], "target": tgt["path"],
              "groups": [], "features": {}, "excluded": {}, "score": None, "decision": None}
    if not groups:
        result["decision"] = "판단 불가"
        result["reason"] = "두 입력에 공통된 얼굴 각도 구간이 없음"
        return result

    for yaw, pitch, rr, tt in groups:
        result["groups"].append({"yaw": round(yaw, 1) + 0.0, "pitch": round(pitch, 1) + 0.0,
                                 "n_ref": len(rr), "n_tgt": len(tt)})

    for k in FEATURES:
        rows = []
        for yaw, pitch, rr, tt in groups:
            if abs(yaw) > FEATURE_MAX_YAW[k]:
                continue
            a, b = _median(rr, k), _median(tt, k)
            if np.isnan(a) or np.isnan(b):
                continue
            z = abs(a - b) / sigma[k]
            rows.append({"yaw": round(yaw, 1) + 0.0, "pitch": round(pitch, 1) + 0.0, "ref": a, "tgt": b,
                         "z": z, "s": float(np.exp(-0.5 * (z / Z_TOL) ** 2)), "n": min(len(rr), len(tt))})
        if not rows:
            reason = ("고개 각도가 사용 범위(|yaw| ≤ %d°)를 벗어남" % FEATURE_MAX_YAW[k]
                      if all(abs(g[0]) > FEATURE_MAX_YAW[k] for g in groups) else "측정 가능한 프레임 없음(입 벌림 등)")
            result["excluded"][k] = reason
            continue
        n = sum(r["n"] for r in rows)
        result["features"][k] = {"z": sum(r["z"] * r["n"] for r in rows) / n,
                                 "s": sum(r["s"] * r["n"] for r in rows) / n,
                                 "n_groups": len(rows), "per_group": rows}

    keys = list(result["features"])
    if not keys:
        result["decision"] = "판단 불가"
        result["reason"] = "비교 가능한 피처가 없음"
        return result
    w = weight or {k: 1.0 for k in FEATURES}
    score = sum(w[k] * result["features"][k]["s"] for k in keys) / sum(w[k] for k in keys)
    result["score"], result["tau"] = score, tau
    result["decision"] = "도용 의심" if score >= tau else "도용 근거 부족"
    return result


NAMES = {"intercanthal": "눈 사이 간격", "canthal_tilt": "눈꼬리 기울기",
         "nose_length": "코 길이", "nose_width": "코 폭", "chin_angle": "턱 끝 각도"}


def report(result):
    """결과 dict -> 사람이 읽는 설명 문자열"""
    L = [f"Reference : {result['reference']}", f"Target    : {result['target']}"]
    if result["score"] is None:
        L.append(f"결과: {result['decision']} ({result.get('reason', '')})")
        return "\n".join(L)
    g = ", ".join(f"yaw {x['yaw']:+.0f}°/pitch {x['pitch']:+.0f}° ({x['n_ref']}:{x['n_tgt']})" for x in result["groups"])
    L.append(f"비교한 각도 그룹 {len(result['groups'])}개: {g}")
    L.append(f"Geometry 유사도 {result['score']:.3f} (임계값 {result['tau']:.3f}) -> {result['decision']}")
    for k, v in sorted(result["features"].items(), key=lambda kv: -kv[1]["s"]):
        verdict = "일치" if v["z"] <= Z_MATCH else ("유사" if v["z"] <= Z_SIMILAR else "차이")
        L.append(f"  {NAMES[k]:<8} z={v['z']:.2f}  s={v['s']:.3f}  [{verdict}]  (각도 그룹 {v['n_groups']}개)")
        for r in v["per_group"]:
            L.append(f"      yaw {r['yaw']:+.0f}° pitch {r['pitch']:+.0f}° : "
                     f"ref {r['ref']:.4f} / tgt {r['tgt']:.4f}  z={r['z']:.2f}")
    for k, why in result["excluded"].items():
        L.append(f"  {NAMES[k]:<8} 제외 ({why})")
    return "\n".join(L)


def save_figures(ref, tgt, result, out_dir):
    """가장 프레임이 많은 각도 그룹에서 각 입력의 대표 프레임을 그림으로 저장"""
    groups = matched_groups(ref, tgt)
    if not groups:
        return
    yaw, pitch, rr, tt = max(groups, key=lambda g: min(len(g[2]), len(g[3])))
    for name, src, recs in [("reference", ref, rr), ("target", tgt, tt)]:
        best = min(recs, key=lambda r: abs(r["yaw"] - yaw) + abs(r["pitch"] - pitch))
        frame = load_frame(src, best["t"])
        if frame is not None:
            draw_all(frame, best["lm"], out_dir=out_dir, prefix=f"{name}_yaw{yaw:+.0f}")


def estimate_sigma_and_weight(by_identity):
    """
    by_identity: {사람ID: [피처 dict, 피처 dict, ...]}  (한 사람당 영상 2개 이상)
    sigma_k  = 개인 내 표준편차 (pooled within-identity std)
    weight_k = Fisher 비율 = 개인 간 분산 / 개인 내 분산
    """
    sigma, weight = {}, {}
    for k in FEATURES:
        groups = [np.array([f[k] for f in fs]) for fs in by_identity.values() if len(fs) >= 2]
        within = sum(((g - g.mean()) ** 2).sum() for g in groups) / sum(len(g) - 1 for g in groups)
        between = np.var([g.mean() for g in groups], ddof=1)
        sigma[k] = float(np.sqrt(within))
        weight[k] = float(between / within)
    return sigma, weight


def threshold_at_far(genuine, impostor, far=0.01):
    """타인 쌍 점수의 (1-FAR) 분위수를 임계값으로, 그때의 FRR도 반환"""
    tau = float(np.quantile(impostor, 1 - far))
    frr = float(np.mean(np.asarray(genuine) < tau))
    return tau, frr



# ================================================================ 7. 시각화 (논문 그림용)
COLORS = {"intercanthal": (0, 200, 255), "canthal_tilt": (255, 120, 0),
          "nose_length": (0, 220, 0), "nose_width": (200, 0, 255), "chin_angle": (0, 0, 255),
          "W": (160, 160, 160)}


def _p(pt):
    return tuple(int(round(v)) for v in pt)


def draw_feature(img, lm, key, f=None, thick=2, row=0):
    out = img.copy()
    f = f or extract_features(lm)
    c = COLORS[key]
    t = max(1, int(round(f["_W_px"] / 250 * thick)))   # 얼굴 크기에 맞춘 선 두께
    fs = f["_W_px"] / 600                                  # 글자 크기

    if key == "intercanthal":
        cv2.line(out, _p(lm[R_IN]), _p(lm[L_IN]), c, t)
        pts, label = [R_IN, L_IN], f"intercanthal/W = {f[key]:.3f}"
    elif key == "nose_length":
        cv2.line(out, _p(lm[NASION]), _p(lm[SUBNASALE]), c, t)
        pts, label = [NASION, SUBNASALE], f"nose length/W = {f[key]:.3f}"
    elif key == "nose_width":
        cv2.line(out, _p(lm[ALAR_R]), _p(lm[ALAR_L]), c, t)
        pts, label = [ALAR_R, ALAR_L], f"nose width/W = {f[key]:.3f}"
    elif key == "chin_angle":
        cv2.line(out, _p(lm[MENTON]), _p(lm[JAW_R]), c, t)
        cv2.line(out, _p(lm[MENTON]), _p(lm[JAW_L]), c, t)
        r = int(dist(lm[MENTON], lm[JAW_R]) * 0.4)
        a1 = np.degrees(np.arctan2(*(lm[JAW_R] - lm[MENTON])[::-1]))
        a2 = np.degrees(np.arctan2(*(lm[JAW_L] - lm[MENTON])[::-1]))
        cv2.ellipse(out, _p(lm[MENTON]), (r, r), 0, min(a1, a2), max(a1, a2), c, t)
        pts, label = [JAW_R, MENTON, JAW_L], f"chin angle = {f[key]:.1f} deg"
    elif key == "canthal_tilt":
        phi = roll_angle(lm)
        u = np.array([np.cos(phi), np.sin(phi)])           # 얼굴 기준 수평 방향
        for inner, outer in [(R_IN, R_OUT), (L_IN, L_OUT)]:
            L = dist(lm[inner], lm[outer]) * 1.2
            sgn = np.sign(np.dot(lm[outer] - lm[inner], u))
            cv2.line(out, _p(lm[inner]), _p(lm[inner] + sgn * u * L), COLORS["W"], max(1, t // 2))
            cv2.line(out, _p(lm[inner]), _p(lm[outer]), c, t)
        pts = [R_IN, R_OUT, L_IN, L_OUT]
        label = f"canthal tilt = {f[key]:+.1f} deg (R {f['_tilt_r']:+.1f}, L {f['_tilt_l']:+.1f})"
    else:
        raise ValueError(key)

    for i in pts:
        cv2.circle(out, _p(lm[i]), t + 2, c, -1)
    cv2.putText(out, label, (10, int(30 * fs * (row + 1.5)) + 10), cv2.FONT_HERSHEY_SIMPLEX, fs, c, max(1, t // 2), cv2.LINE_AA)
    return out


def draw_all(img, lm, out_dir="figures", prefix="face"):
    os.makedirs(out_dir, exist_ok=True)
    f = extract_features(lm)
    combined = img.copy()
    t = max(1, int(round(f["_W_px"] / 250)))
    cv2.line(combined, _p(lm[FACE_R]), _p(lm[FACE_L]), COLORS["W"], t)   # 정규화 기준 W
    for i, k in enumerate(FEATURES):
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{k}.png"), draw_feature(img, lm, k, f))
        combined = draw_feature(combined, lm, k, f, row=i)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_all.png"), combined)
    return f


# ================================================================ 8. 불변성 검증 실험 (논문의 '왜')
def similarity_transform(img, scale=1.0, rot_deg=0.0, tx=0, ty=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, scale)
    M[:, 2] += (tx, ty)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def invariance_test(img, detector,
                    settings=((1.0, 0, 0, 0), (0.6, 0, 0, 0), (1.4, 0, 0, 0),
                              (1.0, 15, 0, 0), (1.0, -15, 0, 0), (1.0, 0, 60, -40))):
    """같은 얼굴을 크기·회전·이동시켜도 피처 값이 유지되는지 확인"""
    log.info(f"{'scale':>5} {'rot':>5} {'tx':>4} {'ty':>4} | " + " ".join(f"{k:>12}" for k in FEATURES))
    rows = []
    for s, r, tx, ty in settings:
        lm = detector(similarity_transform(img, s, r, tx, ty))
        if lm is None:
            log.info(f"{s:>5} {r:>5} {tx:>4} {ty:>4} | face not found")
            continue
        f = extract_features(lm)
        rows.append([f[k] for k in FEATURES])
        log.info(f"{s:>5} {r:>5} {tx:>4} {ty:>4} | " + " ".join(f"{f[k]:>12.4f}" for k in FEATURES))
    arr = np.array(rows)
    log.info("std    " + " " * 15 + " ".join(f"{v:>12.4f}" for v in arr.std(axis=0)))
    return arr


# ================================================================ 실행 · 로그
def load_stats(path=None):
    if path:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        return st["sigma"], st.get("weight") or {k: 1.0 for k in FEATURES}, st["tau"]
    # 예시값: 반드시 동일인/타인 데이터로 다시 추정할 것
    sigma = {"intercanthal": 0.01, "canthal_tilt": 1.5, "nose_length": 0.015,
             "nose_width": 0.01, "chin_angle": 3.0}
    return sigma, {k: 1.0 for k in FEATURES}, 0.6


def setup_run(out_root="results"):
    """results/<날짜_시간>/ 폴더를 만들고, 콘솔 + run.log 파일에 동시에 기록"""
    run_dir = os.path.join(out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return run_dir


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_compare(ref_path, tgt_path, detector, stats=None, step=5, out_root="results"):
    run_dir = setup_run(out_root)
    log.info(f"비교 시작 (각도 구간 {POSE_BIN}°, 이미지 허용 오차 ±{POSE_TOL}°, 영상 샘플 간격 {step}프레임)")
    ref, tgt = collect(ref_path, detector, step), collect(tgt_path, detector, step)
    sigma, weight, tau = load_stats(stats)
    if not stats:
        log.warning("sigma/tau가 예시값입니다. --stats로 실제 추정값을 넣어야 판단 근거가 성립합니다.")
    result = compare(ref, tgt, sigma, weight, tau)
    log.info("결과\n" + report(result))
    save_figures(ref, tgt, result, os.path.join(run_dir, "figures"))
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable({**result, "sigma": sigma, "weight": weight, "settings": {
            "POSE_BIN": POSE_BIN, "POSE_TOL": POSE_TOL, "MIN_FRAMES": MIN_FRAMES, "Z_TOL": Z_TOL,
            "FEATURE_MAX_YAW": FEATURE_MAX_YAW, "step": step}}), f, ensure_ascii=False, indent=2)
    log.info(f"저장 위치: {run_dir} (run.log, result.json, figures/)")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("REFERENCE", "TARGET"),
                    help="이미지/영상 2개 비교 (섞어서도 가능)")
    ap.add_argument("--images", nargs=2, metavar=("REFERENCE", "TARGET"), help="--compare와 동일")
    ap.add_argument("--videos", nargs=2, metavar=("REFERENCE", "TARGET"), help="--compare와 동일")
    ap.add_argument("--image", help="얼굴 이미지 1장 -> 피처 시각화 + 불변성 실험")
    ap.add_argument("--predictor", default="shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--stats", help="sigma/weight/tau JSON (없으면 예시값 사용)")
    ap.add_argument("--step", type=int, default=5, help="영상에서 몇 프레임마다 분석할지")
    ap.add_argument("--out", default="results", help="결과 저장 폴더")
    args = ap.parse_args()
    det = LandmarkDetector(args.predictor)

    pair = args.compare or args.images or args.videos
    if pair:
        run_compare(*pair, det, args.stats, args.step, args.out)

    if args.image:
        run_dir = setup_run(args.out)
        img = cv2.imread(args.image)
        lm = det(img)
        if lm is None:
            raise SystemExit("얼굴을 찾지 못했습니다")
        f = draw_all(img, lm, out_dir=os.path.join(run_dir, "figures"))
        log.info(f"{ {k: round(f[k], 4) for k in FEATURES} }  pose(pitch, yaw, roll): {head_pose(lm, img.shape)}")
        invariance_test(img, det)
