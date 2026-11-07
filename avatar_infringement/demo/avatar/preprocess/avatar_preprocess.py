"""
Avatar modality - preprocess
============================
영상(mp4) -> 프레임별 [사람 마스크 + 2D 관절] -> 체형/외형 hand-crafted 특징 -> JSON

출력 JSON 구조
  {
    "video": ..., "n_frames_total": ..., "n_frames_valid": ...,
    "frames": [ {frame_idx, features{...}, coverage{...}}, ... ],
    "video_features": {feature_name: value, ...},   # 프레임 대표값(중앙값/상위 분위수)
    "video_coverage": {...}                         # 측정 부위가 맨살(skin)인지 옷(cloth)인지
  }

사용 예
  python avatar_preprocess.py --video ../../../../suspect_data/a.mp4 --out_dir ../../../../cache/avatar_standalone
  python avatar_preprocess.py --video_dir ../../../../copyright_database/avatar/SSH --out_dir ../../../../cache/avatar_standalone/db/SSH
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

# ---------------------------------------------------------------- 상수
KP = dict(nose=0, leye=1, reye=2, lear=3, rear=4, lsho=5, rsho=6, lelb=7, relb=8,
          lwri=9, rwri=10, lhip=11, rhip=12, lknee=13, rknee=14, lank=15, rank=16)
BODY_KP = ['nose', 'lsho', 'rsho', 'lhip', 'rhip', 'lknee', 'rknee', 'lank', 'rank']
KP_CONF = 0.5          # 관절 신뢰도 기준
TEX_H = 512            # 질감(LBP) 계산 시 사람 키를 이 픽셀로 정규화
LBP_P, LBP_R = 8, 2    # uniform LBP -> P+2 bins

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, '..', '..', '..', '..', 'weights', 'yolo')


# ---------------------------------------------------------------- 모델
class PersonParser:
    """YOLO11 pose + seg 로 가장 큰 사람 1명의 마스크와 COCO-17 관절을 얻는다."""

    def __init__(self, weights_dir=DEFAULT_WEIGHTS, device='cpu', imgsz=640):
        from ultralytics import YOLO
        def w(name):
            p = os.path.join(weights_dir, name)
            return p if os.path.exists(p) else name   # 없으면 ultralytics가 자동 다운로드
        self.pose = YOLO(w('yolo11m-pose.pt'))
        self.seg = YOLO(w('yolo11m-seg.pt'))
        self.device, self.imgsz = device, imgsz

    def __call__(self, frame):
        H, W = frame.shape[:2]
        rs = self.seg.predict(frame, imgsz=self.imgsz, classes=[0], retina_masks=True,
                              device=self.device, verbose=False)[0]
        if rs.masks is None or len(rs.masks) == 0:
            return None
        masks = rs.masks.data.cpu().numpy() > 0.5
        areas = masks.reshape(len(masks), -1).sum(1)
        mask = masks[int(areas.argmax())]
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(mask)
        mbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)

        rp = self.pose.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if rp.keypoints is None or len(rp.keypoints) == 0:
            return None
        boxes = rp.boxes.xyxy.cpu().numpy()
        iou = [_iou(b, mbox) for b in boxes]
        i = int(np.argmax(iou))
        if iou[i] < 0.3:
            return None
        kxy = rp.keypoints.xy.cpu().numpy()[i]
        kcf = rp.keypoints.conf.cpu().numpy()[i]
        return dict(mask=mask, kxy=kxy, kconf=kcf)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


# ---------------------------------------------------------------- 기하 유틸
def P(kxy, name):
    return kxy[KP[name]].astype(float)


def mid(a, b):
    return (a + b) / 2.0


def dist(a, b):
    return float(np.linalg.norm(a - b))


def march(mask, p, d, max_len, stop_fn=None):
    """p 에서 단위벡터 d 방향으로 마스크 밖으로 나갈 때까지 걸은 거리(px)."""
    H, W = mask.shape
    miss = 0
    last_in = 0.0
    for s in np.arange(0, max_len, 1.0):
        q = p + d * s
        x, y = int(round(q[0])), int(round(q[1]))
        if x < 0 or y < 0 or x >= W or y >= H:
            break
        if stop_fn is not None and stop_fn(q):
            break
        if mask[y, x]:
            last_in, miss = s, 0
        else:
            miss += 1
            if miss > 2:       # 2px 이하 구멍은 무시
                break
    return last_in


def width_across(mask, a, b, t, max_len, inner_stop=None):
    """선분 a->b 의 비율 t 지점에서, 선분에 수직인 방향의 마스크 폭(px)."""
    v = b - a
    n = np.linalg.norm(v)
    if n < 1:
        return np.nan
    u = v / n
    perp = np.array([-u[1], u[0]])
    p = a + v * t
    w1 = march(mask, p, perp, max_len, inner_stop)
    w2 = march(mask, p, -perp, max_len, inner_stop)
    return w1 + w2, p, perp


def point_seg_dist(q, a, b):
    v = b - a
    t = np.clip(np.dot(q - a, v) / (np.dot(v, v) + 1e-9), 0, 1)
    return float(np.linalg.norm(q - (a + t * v)))


# ---------------------------------------------------------------- 피부 모델
class SkinModel:
    """얼굴(코 주변) 픽셀로 그 사람의 피부색 분포(YCrCb의 Cr,Cb)를 추정."""

    def __init__(self, ycc, kxy, kconf, mask):
        self.ok = False
        if min(kconf[KP['nose']], kconf[KP['leye']], kconf[KP['reye']]) < KP_CONF:
            return
        nose, le, re = P(kxy, 'nose'), P(kxy, 'leye'), P(kxy, 'reye')
        r = max(3, int(0.35 * dist(le, re)))
        x, y = int(nose[0]), int(nose[1] + 0.3 * r)
        patch = ycc[max(0, y - r):y + r, max(0, x - r):x + r]
        pm = mask[max(0, y - r):y + r, max(0, x - r):x + r]
        px = patch[pm][:, 1:3].astype(float)
        if len(px) < 20:
            return
        self.mu = np.median(px, 0)
        cov = np.cov(px.T) + np.eye(2) * 9.0     # 조명 차이를 고려해 분산을 넓힘
        self.icov = np.linalg.inv(cov)
        self.ok = True

    def prob_map(self, ycc):
        d = ycc[..., 1:3].astype(float) - self.mu
        m = np.einsum('...i,ij,...j->...', d, self.icov, d)
        return m < 16.0         # 마할라노비스 거리 4 이내 = 피부


# ---------------------------------------------------------------- 한 프레임 특징
def frame_features(frame, det):
    mask, kxy, kc = det['mask'], det['kxy'], det['kconf']
    Hh, Ww = mask.shape
    feats, cov, dbg = {}, {}, {'lines': []}

    # 1) 프레임 유효성: 주요 관절 + 전신 + 직립
    if any(kc[KP[k]] < KP_CONF for k in BODY_KP):
        return None
    ys, xs = np.nonzero(mask)
    if ys.min() <= 1 or ys.max() >= Hh - 2:
        return None                                   # 머리/발이 잘림
    sho_c = mid(P(kxy, 'lsho'), P(kxy, 'rsho'))
    hip_c = mid(P(kxy, 'lhip'), P(kxy, 'rhip'))
    torso_v = hip_c - sho_c
    if abs(np.degrees(np.arctan2(torso_v[0], torso_v[1]))) > 20:
        return None                                   # 몸통이 많이 기울어짐

    # 2) 키 H : 머리 밴드의 마스크 최상단 ~ 발목 주변 마스크 최하단
    sw = dist(P(kxy, 'lsho'), P(kxy, 'rsho'))
    head_x = P(kxy, 'nose')[0]
    band = max(4, int(0.35 * sw))
    col = mask[:, max(0, int(head_x - band)):int(head_x + band)]
    rows = np.nonzero(col.any(1))[0]
    rows = rows[rows < P(kxy, 'nose')[1]]
    if len(rows) == 0:
        return None
    head_top = float(rows.min())
    foot = 0.0
    for a in ('lank', 'rank'):
        ax = P(kxy, a)[0]
        c = mask[:, max(0, int(ax - band)):int(ax + band)]
        r = np.nonzero(c.any(1))[0]
        if len(r):
            foot = max(foot, float(r.max()))
    Hpx = foot - head_top
    if Hpx < 100:
        return None
    feats['_H_px'] = Hpx

    # 3) 머리 길이: 정수리 ~ 목의 가장 좁은 지점(턱 아래)
    y0, y1 = int(P(kxy, 'nose')[1]), int(sho_c[1])
    best, neck_y = 1e9, None
    for y in range(y0, max(y0 + 1, y1)):
        xs_row = np.nonzero(mask[y, max(0, int(head_x - 2 * band)):int(head_x + 2 * band)])[0]
        if len(xs_row) == 0:
            continue
        w = xs_row.max() - xs_row.min()
        if w < best:
            best, neck_y = w, y
    if neck_y is not None:
        head_len = neck_y - head_top
        feats['head_ratio'] = head_len / Hpx           # 1/등신
        feats['head_count'] = Hpx / max(head_len, 1)   # 등신

    # 4) 길이 비율 (L/R 평균, 원근으로 짧아지는 것은 영상 단위에서 상위 분위수로 보정)
    def seg_len(a, b):
        return np.mean([dist(P(kxy, s + a), P(kxy, s + b)) for s in ('l', 'r')])
    feats['torso_ratio'] = dist(sho_c, hip_c) / Hpx
    feats['thigh_ratio'] = seg_len('hip', 'knee') / Hpx
    feats['shin_ratio'] = seg_len('knee', 'ank') / Hpx
    feats['leg_ratio'] = feats['thigh_ratio'] + feats['shin_ratio']
    arm_ok = all(kc[KP[k]] >= KP_CONF for k in ('lelb', 'relb', 'lwri', 'rwri'))
    if arm_ok:
        feats['upperarm_ratio'] = seg_len('sho', 'elb') / Hpx
        feats['forearm_ratio'] = seg_len('elb', 'wri') / Hpx
    hw = dist(P(kxy, 'lhip'), P(kxy, 'rhip'))
    frontal = sw / dist(sho_c, hip_c)          # 정면일수록 큼
    feats['_frontal'] = frontal
    feats['shoulder_hip_ratio'] = sw / max(hw, 1)

    # 5) 피부 모델
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = SkinModel(ycc, kxy, kc, mask)
    skin_map = skin.prob_map(ycc) & mask if skin.ok else None

    def coverage_at(p, r):
        if skin_map is None:
            return None
        x, y, r = int(p[0]), int(p[1]), max(2, int(r))
        m = mask[max(0, y - r):y + r, max(0, x - r):x + r]
        s = skin_map[max(0, y - r):y + r, max(0, x - r):x + r]
        if m.sum() < 5:
            return None
        return float(s.sum() / m.sum())

    # 6) 두께: 뼈 방향에 수직으로 마스크 폭 (다른 신체 부위와 붙은 경우 제외)
    torso_poly = [P(kxy, 'lsho'), P(kxy, 'rsho'), P(kxy, 'rhip'), P(kxy, 'lhip')]

    def near_torso(q, margin):
        # 몸통 사각형 안 또는 가까우면 True
        inside = cv2.pointPolygonTest(np.array(torso_poly, np.float32), (float(q[0]), float(q[1])), True)
        return inside > -margin

    def thickness(side, a, b, name, other_axis=None, arm=False):
        A, B = P(kxy, side + a), P(kxy, side + b)
        if kc[KP[side + a]] < KP_CONF or kc[KP[side + b]] < KP_CONF:
            return
        L = dist(A, B)
        if L < 0.05 * Hpx:
            return
        if arm:
            # 팔이 몸통에 붙어 있으면 폭이 합쳐지므로 제외
            if near_torso(mid(A, B), 0.04 * Hpx):
                return
        stop = None
        if other_axis is not None:
            oa, ob = other_axis
            def stop(q, oa=oa, ob=ob, A=A, B=B):
                # 반대쪽 다리 축에 더 가까워지면 멈춤 (두 다리가 붙어도 분리)
                return point_seg_dist(q, oa, ob) < point_seg_dist(q, A, B)
        ws = []
        for t in (0.4, 0.5, 0.6):
            w, p, perp = width_across(mask, A, B, t, 0.25 * Hpx, stop)
            ws.append(w)
        w = float(np.median(ws))
        if not np.isfinite(w) or w <= 0 or w > 0.8 * L:
            return
        feats.setdefault(name, []).append(w / Hpx)
        c = coverage_at(A + (B - A) * 0.5, 0.01 * Hpx)
        if c is not None:
            cov.setdefault(name, []).append(c)
        w0, p0, pp = width_across(mask, A, B, 0.5, 0.25 * Hpx, stop)
        dbg['lines'].append((p0, pp, w0))

    for s in ('l', 'r'):
        o = 'r' if s == 'l' else 'l'
        thickness(s, 'sho', 'elb', 'upperarm_thick', arm=True)
        thickness(s, 'elb', 'wri', 'forearm_thick', arm=True)
        thickness(s, 'hip', 'knee', 'thigh_thick', other_axis=(P(kxy, o + 'hip'), P(kxy, o + 'knee')))
        thickness(s, 'knee', 'ank', 'calf_thick', other_axis=(P(kxy, o + 'knee'), P(kxy, o + 'ank')))
    for k in [k for k in feats if isinstance(feats[k], list)]:
        feats[k] = float(np.mean(feats[k]))
    for k in list(cov):
        cov[k] = float(np.mean(cov[k]))

    # 허리 / 엉덩이 : 몸통 축에 수직, 정면 프레임 & 손·팔꿈치가 몸통에서 떨어져 있을 때만
    arms_free = arm_ok and all(not near_torso(P(kxy, k), 0.03 * Hpx) for k in ('lelb', 'relb', 'lwri', 'rwri'))
    if arms_free:
        for name, t in (('waist_thick', 0.7), ('hip_thick', 1.0)):
            w, p, perp = width_across(mask, sho_c, hip_c, t, 0.35 * Hpx)
            if np.isfinite(w) and w > 0:
                feats[name] = w / Hpx
                dbg['lines'].append((p, perp, w))
                c = coverage_at(p, 0.01 * Hpx)
                if c is not None:
                    cov[name] = c

    # 7) 옷 형태: 소매 길이(어깨->손목 중 옷이 덮는 비율), 하의 길이(골반->발목)
    def covered_len(chain):
        if skin_map is None:
            return None
        pts = []
        for a, b in zip(chain[:-1], chain[1:]):
            A, B = P(kxy, a), P(kxy, b)
            for t in np.linspace(0, 1, 15, endpoint=False):
                pts.append(A + (B - A) * t)
        pts.append(P(kxy, chain[-1]))
        prof = [coverage_at(p, 0.008 * Hpx) for p in pts]
        prof = [1.0 if v is None else v for v in prof]
        # 처음으로 피부 비율이 50%를 넘는 지점 = 옷이 끝나는 위치
        for i, v in enumerate(prof):
            if v > 0.5 and all(x > 0.5 for x in prof[i:i + 3]):
                return i / (len(prof) - 1)
        return 1.0

    if arm_ok and skin_map is not None:
        v = [covered_len([s + 'sho', s + 'elb', s + 'wri']) for s in ('l', 'r')]
        feats['sleeve_len'] = float(np.mean(v))
    if skin_map is not None:
        v = [covered_len([s + 'hip', s + 'knee', s + 'ank']) for s in ('l', 'r')]
        feats['pants_len'] = float(np.mean(v))

    # 8) 외형 영역: 상의 / 하의 / 피부(팔·다리) / 머리카락   (얼굴은 제외)
    regions = {}
    tp = np.array(torso_poly, np.int32)
    torso_m = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(torso_m, [tp], 1)
    notskin = ~skin_map if skin_map is not None else np.ones_like(mask)
    regions['upper'] = (torso_m > 0) & mask & notskin
    leg_m = np.zeros_like(mask, np.uint8)
    lw = max(2, int(0.035 * Hpx))
    for s in ('l', 'r'):
        for a, b in (('hip', 'knee'), ('knee', 'ank')):
            cv2.line(leg_m, tuple(P(kxy, s + a).astype(int)), tuple(P(kxy, s + b).astype(int)), 1, lw)
    regions['lower'] = (leg_m > 0) & mask & notskin
    if skin_map is not None:
        limb_m = leg_m.copy()
        if arm_ok:
            for s in ('l', 'r'):
                for a, b in (('sho', 'elb'), ('elb', 'wri')):
                    cv2.line(limb_m, tuple(P(kxy, s + a).astype(int)), tuple(P(kxy, s + b).astype(int)), 1, lw)
        regions['skin'] = (limb_m > 0) & skin_map
    eye_y = min(P(kxy, 'leye')[1], P(kxy, 'reye')[1])
    hair_m = np.zeros_like(mask)
    hair_m[int(head_top):int(eye_y), max(0, int(head_x - 1.5 * band)):int(head_x + 1.5 * band)] = True
    regions['hair'] = hair_m & mask & notskin

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    lab[..., 0] *= 100 / 255.0
    lab[..., 1:] -= 128
    scale = TEX_H / Hpx
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for rname, rm in regions.items():
        if rm.sum() < 200:
            continue
        px = lab[rm]
        feats[f'{rname}_lab'] = np.median(px, 0).tolist()
        if len(px) > 50:
            sub = px[np.random.default_rng(0).choice(len(px), min(len(px), 3000), replace=False)]
            km = KMeans(3, n_init=3, random_state=0).fit(sub)
            order = np.argsort(-np.bincount(km.labels_, minlength=3))
            feats[f'{rname}_domcolors'] = km.cluster_centers_[order].tolist()
        if rname in ('upper', 'lower'):
            ys_, xs_ = np.nonzero(rm)
            y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            g = cv2.resize(gray[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            m = cv2.resize(rm[y0:y1, x0:x1].astype(np.uint8), (g.shape[1], g.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
            m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if m.sum() > 100:
                lbp = local_binary_pattern(g, LBP_P, LBP_R, 'uniform')
                h = np.bincount(lbp[m].astype(int), minlength=LBP_P + 2).astype(float)
                feats[f'{rname}_lbp'] = (h / h.sum()).tolist()

    # 9) 신원(ID) 단서용 원시 측정: 머리 모양 / 맨살 팔다리 폭 곡선 / 덩치
    idm = {}
    # 9-1) 머리 모양: 얼굴이 정면에 가깝고(코가 두 눈 사이) 손이 머리 근처에 없을 때만
    le, re_, nose = P(kxy, 'leye'), P(kxy, 'reye'), P(kxy, 'nose')
    if (neck_y is not None and min(kc[KP['leye']], kc[KP['reye']], kc[KP['nose']]) >= KP_CONF
            and min(le[0], re_[0]) < nose[0] < max(le[0], re_[0])):
        y0, y1 = int(head_top), int(neck_y)
        x0, x1 = max(0, int(head_x - 2.5 * band)), min(Ww, int(head_x + 2.5 * band))
        hands_near = any(kc[KP[w_]] >= KP_CONF and x0 - band < P(kxy, w_)[0] < x1 + band
                         and P(kxy, w_)[1] < neck_y + band for w_ in ('lwri', 'rwri'))
        if not hands_near and y1 - y0 > 10:
            hm = mask[y0:y1, x0:x1]
            head_w = float(hm.sum(1).max())
            idm['head'] = dict(head_w=head_w, head_h=float(y1 - y0), neck_w=float(best), sh_w=float(sw),
                               eye_y=float((le[1] + re_[1]) / 2 - head_top))
            if skin_map is not None and head_w > 5:
                sm = skin_map[y0:y1, x0:x1]
                hair = hm & ~sm
                cx, cw = int(nose[0] - x0), max(2, int(0.1 * head_w))
                col_s = sm[:, max(0, cx - cw):cx + cw].mean(1)
                idm['head']['bang'] = float(np.argmax(col_s > 0.5)) if (col_s > 0.5).any() else float(len(col_s))
                cols = np.nonzero(hm.any(0))[0]
                side = []
                if len(cols):
                    L_, R_ = cols.min(), cols.max()
                    wv = max(1, R_ - L_)
                    for c0 in (L_ + 0.05 * wv, R_ - 0.15 * wv):
                        c0 = int(c0)
                        rows_h = np.nonzero(hair[:, c0:c0 + max(2, int(0.1 * wv))].any(1))[0]
                        if len(rows_h):
                            side.append(float(rows_h.max()))
                if side:
                    idm['head']['side_hair'] = float(np.mean(side))
                idm['head']['hair_area'] = float(hair.sum() / max(hm.sum(), 1))
    # 9-2) 맨살 팔다리 폭 곡선: 아래팔(팔꿈치->손목), 정강이(무릎->발목)를 9개 지점에서
    limb = {}
    for s_ in ('l', 'r'):
        o_ = 'r' if s_ == 'l' else 'l'
        for a_, b_, nm in (('elb', 'wri', 'forearm'), ('knee', 'ank', 'shin')):
            if kc[KP[s_ + a_]] < KP_CONF or kc[KP[s_ + b_]] < KP_CONF:
                continue
            A_, B_ = P(kxy, s_ + a_), P(kxy, s_ + b_)
            L_ = dist(A_, B_)
            if L_ < 0.05 * Hpx or (nm == 'forearm' and near_torso(mid(A_, B_), 0.04 * Hpx)):
                continue
            stop_ = None
            if nm == 'shin':
                oa, ob = P(kxy, o_ + 'knee'), P(kxy, o_ + 'ank')
                def stop_(q, oa=oa, ob=ob, A_=A_, B_=B_):
                    return point_seg_dist(q, oa, ob) < point_seg_dist(q, A_, B_)
            ws, cs = [], []
            for t in np.linspace(0.1, 0.9, 9):
                w_, p_, _ = width_across(mask, A_, B_, t, 0.2 * Hpx, stop_)
                c_ = coverage_at(p_, 0.008 * Hpx)
                ws.append(float(w_))
                cs.append(-1.0 if c_ is None else c_)
            limb[s_ + '_' + nm] = dict(w=ws, cov=cs, L=float(L_))
    idm['limb'] = limb
    # 9-3) 덩치: 몸통 축에 수직인 폭. 팔이 몸통에서 떨어져 있고 팔꿈치가 어깨보다 충분히 아래일 때만
    elbows_down = arm_ok and all(P(kxy, e)[1] > sho_c[1] + 0.08 * Hpx for e in ('lelb', 'relb'))
    if arms_free and elbows_down:
        build = {}
        for nm, t in (('shoulder', 0.05), ('chest', 0.3), ('waist', 0.7), ('hip', 1.0)):
            w_, p_, _ = width_across(mask, sho_c, hip_c, t, 0.35 * Hpx)
            c_ = coverage_at(p_, 0.01 * Hpx)
            build[nm] = dict(w=float(w_), cov=-1.0 if c_ is None else c_)
        if build['shoulder']['w'] < 2.5 * sw:
            build['torso_len'] = float(dist(sho_c, hip_c))
            build['sh_kp'] = float(sw)
            idm['build'] = build
    dbg.update(mask=mask, kxy=kxy, head_top=head_top, foot=foot, regions=regions)
    # 기하 불변량 계산용 원시 좌표 (metric 단계에서 사용)
    feet = {}
    for s_ in ('l', 'r'):
        ax = P(kxy, s_ + 'ank')[0]
        c = mask[:, max(0, int(ax - band)):int(ax + band)]
        r_ = np.nonzero(c.any(1))[0]
        feet[s_] = [float(ax), float(r_.max())] if len(r_) else None
    dbg['raw'] = dict(kxy=kxy.tolist(), kconf=kc.tolist(), head_top=[float(head_x), head_top],
                      neck_y=None if neck_y is None else float(neck_y), feet=feet,
                      H_px=Hpx, img_wh=[int(Ww), int(Hh)], id=idm)
    return feats, cov, dbg


# ---------------------------------------------------------------- 영상 단위 집계
LENGTH_KEYS = ['torso_ratio', 'thigh_ratio', 'shin_ratio', 'leg_ratio', 'upperarm_ratio', 'forearm_ratio']


def aggregate(frames):
    """프레임별 특징 -> 영상 대표값.
    - 길이: 원근 때문에 짧게만 보이므로 85번째 분위수
    - 어깨/골반 비, 허리·엉덩이: 정면에 가까운 상위 50% 프레임의 중앙값
    - 그 외 스칼라: 중앙값,  색(Lab): 프레임 중앙값,  LBP: 평균 히스토그램
    """
    out, cov = {}, {}
    keys = set(k for f in frames for k in f['features'] if not k.startswith('_'))
    fr = np.array([f['features']['_frontal'] for f in frames])
    front_sel = fr >= np.median(fr)
    for k in sorted(keys):
        vals = [(i, f['features'][k]) for i, f in enumerate(frames) if k in f['features']]
        if len(vals) < 3:
            continue
        idx = [i for i, _ in vals]
        v = [x for _, x in vals]
        if isinstance(v[0], list) and isinstance(v[0][0], list):          # domcolors
            out[k] = np.median(np.array(v), 0).tolist()
        elif isinstance(v[0], list):
            arr = np.array(v)
            out[k] = (arr.mean(0) if k.endswith('_lbp') else np.median(arr, 0)).tolist()
        elif k in LENGTH_KEYS:
            out[k] = float(np.percentile(v, 85))
        elif k in ('shoulder_hip_ratio', 'waist_thick', 'hip_thick'):
            sel = [x for i, x in vals if front_sel[i]]
            out[k] = float(np.median(sel if len(sel) >= 3 else v))
        else:
            out[k] = float(np.median(v))
        out[k + '_n'] = len(v)
    ckeys = set(k for f in frames for k in f['coverage'])
    for k in ckeys:
        v = [f['coverage'][k] for f in frames if k in f['coverage']]
        if v:
            cov[k] = 'skin' if np.median(v) > 0.5 else 'cloth'
    return out, cov


# ---------------------------------------------------------------- 디버그 이미지
def draw_debug(frame, dbg, path):
    img = frame.copy()
    over = img.copy()
    colors = dict(upper=(0, 0, 255), lower=(255, 0, 0), skin=(0, 200, 255), hair=(255, 0, 255))
    for r, m in dbg['regions'].items():
        over[m] = colors[r]
    img = cv2.addWeighted(img, 0.55, over, 0.45, 0)
    cnts, _ = cv2.findContours(dbg['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    for p in dbg['kxy']:
        cv2.circle(img, tuple(p.astype(int)), 5, (255, 255, 255), -1)
    for p, perp, w in dbg['lines']:
        a = (p + perp * w / 2).astype(int)
        b = (p - perp * w / 2).astype(int)
        cv2.line(img, tuple(a), tuple(b), (0, 255, 255), 3)
    cv2.line(img, (0, int(dbg['head_top'])), (img.shape[1], int(dbg['head_top'])), (255, 255, 0), 1)
    cv2.line(img, (0, int(dbg['foot'])), (img.shape[1], int(dbg['foot'])), (255, 255, 0), 1)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------- 실행
def process_video(path, parser, out_dir, stride=2, max_frames=None, debug=True):
    cap = cv2.VideoCapture(path)
    frames, idx, n_total = [], 0, 0
    best_dbg = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        if idx % stride == 0:
            det = parser(frame)
            if det is not None:
                r = frame_features(frame, det)
                if r is not None:
                    f, c, dbg = r
                    frames.append(dict(frame_idx=idx, features=f, coverage=c, raw=dbg['raw']))
                    if debug and (best_dbg is None or f['_frontal'] > best_dbg[0]):
                        best_dbg = (f['_frontal'], frame.copy(), dbg)
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    name = os.path.splitext(os.path.basename(path))[0]
    res = dict(video=os.path.abspath(path), name=name, n_frames_total=n_total,
               n_frames_sampled=(min(idx, n_total) + stride - 1) // stride, n_frames_valid=len(frames))
    if len(frames) >= 3:
        vf, vc = aggregate(frames)
        res['video_features'], res['video_coverage'] = vf, vc
    else:
        res['video_features'], res['video_coverage'] = {}, {}
    res['frames'] = frames
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1, default=float)
    if debug and best_dbg is not None:
        os.makedirs(os.path.join(out_dir, 'debug'), exist_ok=True)
        draw_debug(best_dbg[1], best_dbg[2], os.path.join(out_dir, 'debug', name + '.jpg'))
    print(f'[preprocess] {name}: valid {len(frames)} / sampled {res["n_frames_sampled"]}')
    return res


def main():
    ap = argparse.ArgumentParser(description='Avatar modality preprocess (체형/외형 특징 추출)')
    ap.add_argument('--video', nargs='*', default=[], help='입력 영상 경로(여러 개 가능)')
    ap.add_argument('--video_dir', default=None, help='폴더 안의 *.mp4 전체 처리')
    ap.add_argument('--pattern', default='*.mp4')
    ap.add_argument('--out_dir', default=os.path.join(HERE, '..', '..', '..', '..', 'cache', 'avatar_standalone'))
    ap.add_argument('--weights_dir', default=DEFAULT_WEIGHTS)
    ap.add_argument('--stride', type=int, default=2, help='프레임 간격 (2 = 24fps 영상에서 12fps)')
    ap.add_argument('--device', default='cpu', help="'cpu' 또는 '0'(GPU)")
    args = ap.parse_args()
    vids = list(args.video)
    if args.video_dir:
        vids += sorted(glob.glob(os.path.join(args.video_dir, args.pattern)))
    if not vids:
        ap.error('--video 또는 --video_dir 를 지정하세요')
    parser = PersonParser(args.weights_dir, args.device)
    for v in vids:
        process_video(v, parser, args.out_dir, args.stride)


if __name__ == '__main__':
    main()
