"""
Avatar modality - 체형 (기하 불변량)
===================================
카메라 높이·각도·거리가 달라도 값이 (거의) 변하지 않는 2D 체형 특징.

1) 같은 직선 위 길이의 비  (약원근에서 불변)
     - 팔을 폈을 때  위팔 / 아래팔
     - 다리를 폈을 때 허벅지 / 정강이
     - 똑바로 섰을 때 몸 세로축(정수리-어깨중심-골반중심-무릎중심-발목중심)의 구간 비
2) 평행 선분 길이의 비  (약원근에서 불변)
     - 어깨선과 골반선이 평행할 때  어깨 너비 / 골반 너비
3) 복비(cross-ratio)  (완전 원근에서도 불변, 한 직선 위 4점)
     - CR(정수리, 어깨중심, 골반중심, 발목중심),  CR(어깨중심, 골반중심, 무릎중심, 발목중심)
4) 실루엣 두께 비  (불변량이 아니므로 몸 방향 구간별로만 비교)
     - 허리/엉덩이, 아래팔/위팔, 종아리/허벅지, 허벅지/엉덩이  (같은 프레임 안의 비율)

각 특징은 전제 조건(일직선, 평행, 충분히 화면과 평행)을 만족하는 프레임에서만 계산한다.
"""
import numpy as np

KP = dict(nose=0, leye=1, reye=2, lear=3, rear=4, lsho=5, rsho=6, lelb=7, relb=8,
          lwri=9, rwri=10, lhip=11, rhip=12, lknee=13, rknee=14, lank=15, rank=16)
CONF = 0.5
STRAIGHT_DEG = 170.0      # 팔다리가 펴졌다고 보는 관절 각도
PARALLEL_DEG = 5.0        # 어깨선-골반선 평행 허용 각도
MIN_LEN_FRAC = 0.7        # 영상 내 최대 길이 대비 이 비율 이상일 때만 (카메라 방향으로 누운 구간 제외)
COLLINEAR_DEV = 0.03      # 세로축 점들이 직선에서 벗어난 정도 (축 길이 대비)
MIN_FRAMES = 3

# key, 한글 이름, 종류, 가중치, 최소 허용오차(상대)
INVARIANT_FEATURES = [
    # 가중치: SSH 카메라 5대에서 확인한 시점 안정성 기준 (복비가 가장 안정적: 카메라 간 흔들림 < 1%)
    ('cr_upper',         '복비(정수리·어깨·골반·발목)',        'cross',   2.0, 0.02),
    ('cr_lower',         '복비(어깨·골반·무릎·발목)',          'cross',   2.0, 0.02),
    ('arm_upper_fore',   '위팔/아래팔 (팔 편 상태)',          'line',    1.0, 0.03),
    ('chain_torso',      '몸통 / 전신 세로축',                 'line',    1.0, 0.03),
    ('shoulder_hip',     '어깨 너비/골반 너비 (평행 상태)',    'parallel', 0.5, 0.03),
    ('leg_thigh_shin',   '허벅지/정강이 (다리 편 상태)',      'line',    0.5, 0.03),
    ('chain_head',       '머리+목 / 전신 세로축',              'line',    0.5, 0.03),
    ('chain_leg',        '다리 / 전신 세로축',                 'line',    0.5, 0.02),
]
THICK_RATIOS = [
    ('thk_waist_hip',     '허리/엉덩이 너비',   'waist_thick',   'hip_thick'),
    ('thk_fore_upper',    '아래팔/위팔 두께',   'forearm_thick', 'upperarm_thick'),
    ('thk_calf_thigh',    '종아리/허벅지 두께', 'calf_thick',    'thigh_thick'),
    ('thk_thigh_hip',     '허벅지/엉덩이 너비', 'thigh_thick',   'hip_thick'),
]
THICK_FLOOR = 0.05
YAW_BINS = (('정면', 0.85, 9.0), ('비스듬', 0.55, 0.85), ('측면', 0.0, 0.55))


# ---------------------------------------------------------------- 기하 유틸
def _p(k, n):
    return np.asarray(k[KP[n]], float)


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _line_params(pts):
    """점들을 주축에 투영한 1D 좌표와, 주축에서 벗어난 최대 거리(축 길이 대비)."""
    P = np.asarray(pts, float)
    c = P.mean(0)
    u = np.linalg.svd(P - c)[2][0]
    t = (P - c) @ u
    off = np.abs((P - c) @ np.array([-u[1], u[0]]))
    span = t.max() - t.min()
    if t[-1] < t[0]:
        t = -t
    return t, float(off.max() / (span + 1e-9)), span


def cross_ratio(t):
    a, b, c, d = t
    return float(((c - a) * (d - b)) / ((c - b) * (d - a) + 1e-12))


# ---------------------------------------------------------------- 프레임별 측정
def frame_measures(raw, feats, cov):
    k, cf = np.asarray(raw['kxy'], float), np.asarray(raw['kconf'], float)
    ok = lambda *ns: all(cf[KP[n]] >= CONF for n in ns)
    m = {}
    # 1) 팔 / 다리 (한쪽씩)
    for s in ('l', 'r'):
        if ok(s + 'sho', s + 'elb', s + 'wri'):
            A, B, C = _p(k, s + 'sho'), _p(k, s + 'elb'), _p(k, s + 'wri')
            if _angle(A, B, C) >= STRAIGHT_DEG:
                m.setdefault('arm_upper_fore', []).append(
                    (np.linalg.norm(A - B) / np.linalg.norm(B - C), np.linalg.norm(A - B), np.linalg.norm(B - C), 'arm' + s))
        if ok(s + 'hip', s + 'knee', s + 'ank'):
            A, B, C = _p(k, s + 'hip'), _p(k, s + 'knee'), _p(k, s + 'ank')
            if _angle(A, B, C) >= STRAIGHT_DEG:
                m.setdefault('leg_thigh_shin', []).append(
                    (np.linalg.norm(A - B) / np.linalg.norm(B - C), np.linalg.norm(A - B), np.linalg.norm(B - C), 'leg' + s))
    # 2) 어깨/골반 (평행)
    if ok('lsho', 'rsho', 'lhip', 'rhip'):
        vs, vh = _p(k, 'rsho') - _p(k, 'lsho'), _p(k, 'rhip') - _p(k, 'lhip')
        ang = abs(np.degrees(np.arctan2(vs[1], vs[0]) - np.arctan2(vh[1], vh[0])))
        ang = min(ang, 360 - ang)
        if ang <= PARALLEL_DEG:
            sw, hw = np.linalg.norm(vs), np.linalg.norm(vh)
            m['shoulder_hip'] = [(sw / hw, sw, sw, 'sw')]
    # 3) 몸 세로축 (정수리-어깨중심-골반중심-무릎중심-발목중심)
    if ok('lsho', 'rsho', 'lhip', 'rhip', 'lknee', 'rknee', 'lank', 'rank'):
        legs_straight = all(_angle(_p(k, s + 'hip'), _p(k, s + 'knee'), _p(k, s + 'ank')) >= STRAIGHT_DEG
                            for s in ('l', 'r'))
        if legs_straight:
            T = np.asarray(raw['head_top'], float)
            S = (_p(k, 'lsho') + _p(k, 'rsho')) / 2
            H = (_p(k, 'lhip') + _p(k, 'rhip')) / 2
            K = (_p(k, 'lknee') + _p(k, 'rknee')) / 2
            A = (_p(k, 'lank') + _p(k, 'rank')) / 2
            t, dev, span = _line_params([T, S, H, K, A])
            if dev <= COLLINEAR_DEV and np.all(np.diff(t) > 0):
                L = t[4] - t[0]
                m['chain_head'] = [((t[1] - t[0]) / L, L, L, 'chain')]
                m['chain_torso'] = [((t[2] - t[1]) / L, L, L, 'chain')]
                m['chain_leg'] = [((t[4] - t[2]) / L, L, L, 'chain')]
                m['cr_upper'] = [(cross_ratio(t[[0, 1, 2, 4]]), L, L, 'chain')]
                m['cr_lower'] = [(cross_ratio(t[[1, 2, 3, 4]]), L, L, 'chain')]
    # 4) 두께 비 (같은 프레임) + 몸 방향 + 맨살/옷
    sw = np.linalg.norm(_p(k, 'rsho') - _p(k, 'lsho'))
    S = (_p(k, 'lsho') + _p(k, 'rsho')) / 2
    H = (_p(k, 'lhip') + _p(k, 'rhip')) / 2
    th = {}
    for key, _, a, b in THICK_RATIOS:
        if a in feats and b in feats and feats[b] > 0:
            ca = 'skin' if cov.get(a, 0) > 0.5 else 'cloth'
            cb = 'skin' if cov.get(b, 0) > 0.5 else 'cloth'
            th[key] = (feats[a] / feats[b], f'{ca}/{cb}')
    return m, th, sw / (np.linalg.norm(H - S) + 1e-9)


def video_body(result):
    """영상 1개 -> 불변량 프레임 값 목록 + 두께 비(몸 방향 구간별)."""
    frames = [f for f in result.get('frames', []) if 'raw' in f]
    per = [frame_measures(f['raw'], f['features'], f.get('coverage', {})) for f in frames]
    # 카메라 방향으로 누운 구간 제외: 영상 내 최대 길이 대비 MIN_LEN_FRAC 이상
    maxlen = {}
    for m, _, _ in per:
        for key, items in m.items():
            for v, l1, l2, tag in items:
                maxlen[(tag, 1)] = max(maxlen.get((tag, 1), 0), l1)
                maxlen[(tag, 2)] = max(maxlen.get((tag, 2), 0), l2)
    inv = {}
    for m, _, _ in per:
        for key, items in m.items():
            for v, l1, l2, tag in items:
                if l1 >= MIN_LEN_FRAC * maxlen[(tag, 1)] and l2 >= MIN_LEN_FRAC * maxlen[(tag, 2)]:
                    inv.setdefault(key, []).append(v)
    # 몸 방향 구간: 어깨너비/몸통길이 를 영상 내 최대값으로 나눈 값
    fr = np.array([x[2] for x in per]) if per else np.array([])
    fmax = np.percentile(fr, 95) if len(fr) else 1
    thick = {}
    for (m, th, f) in per:
        r = f / fmax
        b = next(name for name, lo, hi in YAW_BINS if lo <= r < hi)
        for key, (v, sig) in th.items():
            thick.setdefault(f'{key}@{b}', []).append((v, sig))
    return inv, thick


def summarize(inv, thick):
    out = {}
    for k, v in inv.items():
        if len(v) >= MIN_FRAMES:
            out[k] = dict(value=float(np.median(v)), n=len(v))
    for k, v in thick.items():
        if len(v) >= MIN_FRAMES:
            sigs = [s for _, s in v]
            sig = max(set(sigs), key=sigs.count)
            vals = [x for x, s in v if s == sig]
            if len(vals) >= MIN_FRAMES:
                out[k] = dict(value=float(np.median(vals)), n=len(vals), coverage=sig)
    return out


# ---------------------------------------------------------------- DB 프로필 / 비교
def build_body_profile(db_results, inflate=1.5):
    per_video = [summarize(*video_body(r)) for r in db_results]
    pooled_inv, pooled_th = {}, {}
    for r in db_results:
        inv, th = video_body(r)
        for k, v in inv.items():
            pooled_inv.setdefault(k, []).extend(v)
        for k, v in th.items():
            pooled_th.setdefault(k, []).extend(v)
    prof = summarize(pooled_inv, pooled_th)
    floors = {k: f for k, _, _, _, f in INVARIANT_FEATURES}
    for k, d in prof.items():
        vals = [pv[k]['value'] for pv in per_video if k in pv]
        spread = float(np.sqrt(np.mean((np.array(vals) - d['value']) ** 2))) if len(vals) > 1 else 0.0
        floor = floors.get(k, THICK_FLOOR) * abs(d['value'])
        d['sigma'] = max(spread * inflate, floor)
        d['per_video'] = vals
    return prof, per_video


def label_of(key):
    base, _, b = key.partition('@')
    for k, lab, *_ in INVARIANT_FEATURES:
        if k == base:
            return lab
    for k, lab, *_ in THICK_RATIOS:
        if k == base:
            return f'{lab} [{b}]'
    return key


def compare_body(inp_summary, prof):
    rows = []
    weights = {k: w for k, _, _, w, _ in INVARIANT_FEATURES}
    keys = [k for k, *_ in INVARIANT_FEATURES] + sorted(k for k in prof if k.startswith('thk_'))
    for k in keys:
        row = dict(key=k, label=label_of(k), group='body', weight=weights.get(k, 0.5))
        if k not in prof or k not in inp_summary:
            row.update(status='측정 불가 (조건 맞는 프레임 부족)', score=None)
        elif 'coverage' in prof[k] and prof[k]['coverage'] != inp_summary[k].get('coverage'):
            row.update(status=f'제외(옷/맨살 조건 다름 DB:{prof[k]["coverage"]} / 입력:{inp_summary[k].get("coverage")})',
                       score=None)
        else:
            d = abs(inp_summary[k]['value'] - prof[k]['value'])
            s = float(np.exp(-0.5 * (d / prof[k]['sigma']) ** 2))
            row.update(db=prof[k]['value'], input=inp_summary[k]['value'], sigma=prof[k]['sigma'],
                       n_db=prof[k]['n'], n_in=inp_summary[k]['n'], score=s * 100, status='비교')
        rows.append(row)
    return rows
