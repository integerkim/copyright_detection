"""
Avatar modality - 신원(ID) 단서
==============================
옷이 바뀌거나 시점이 바뀌어도 사람에게 남는 hand-crafted 특징 3종.

1) 머리 모양 (얼굴 제외, 얼굴이 정면에 가까운 프레임만)
     머리 폭/어깨 폭, 머리 세로/가로, 목 폭/머리 폭, 앞머리 길이, 눈 위치, 옆머리 길이, 머리카락 면적 비
2) 맨살 팔다리 폭 곡선 (맨살일 때만, 팔다리는 원기둥에 가까워 시점에 강함)
     정강이: 발목/종아리 최대 폭, 종아리 최대 폭 위치, 종아리 폭/정강이 길이
     아래팔: 손목/아래팔 최대 폭, 아래팔 폭/아래팔 길이
3) 덩치 (정면 프레임, 팔이 몸통에서 떨어져 있을 때만)
     어깨 바깥 폭/몸통 길이, 허리/어깨, 엉덩이/어깨, 허리/엉덩이   (옷 위 측정이면 옷 핏의 영향을 받음)
"""
import numpy as np

T = np.linspace(0.1, 0.9, 9)
FRONT = 0.85            # 몸 방향: 어깨폭/몸통길이 가 영상 내 95분위의 85% 이상 = 정면
MIN_FRAMES = 3
LEN_FRAC = 0.85         # 길이로 나누는 특징은 영상 내 최대 길이의 85% 이상인 프레임만

# key, 한글 이름, 그룹, 최소 허용오차(상대)
ID_FEATURES = [
    ('h_w_sh',       '머리 폭 / 어깨 폭',              'head',  0.04),
    ('h_aspect',     '머리 세로 / 가로',               'head',  0.04),
    ('h_neck',       '목 폭 / 머리 폭',                'head',  0.04),
    ('h_bang',       '앞머리 길이 / 머리 세로',        'head',  0.06),
    ('h_eye',        '정수리~눈 / 머리 세로',          'head',  0.04),
    ('h_side',       '옆머리 끝 / 머리 세로',          'head',  0.05),
    ('h_hair_area',  '머리카락 면적 / 머리 면적',      'head',  0.06),
    ('s_ankle_calf', '발목 / 종아리 최대 폭',          'limb',  0.04),
    ('s_calf_pos',   '종아리 최대 폭 위치',            'limb',  0.05),
    ('s_calf_len',   '종아리 폭 / 정강이 길이',        'limb',  0.04),
    ('f_wrist_fore', '손목 / 아래팔 최대 폭',          'limb',  0.04),
    ('f_fore_len',   '아래팔 폭 / 아래팔 길이',        'limb',  0.04),
    ('b_sh_torso',   '어깨 바깥 폭 / 몸통 길이',       'build', 0.04),
    ('b_waist_sh',   '허리 / 어깨 바깥 폭',            'build', 0.04),
    ('b_hip_sh',     '엉덩이 / 어깨 바깥 폭',          'build', 0.04),
    ('b_waist_hip',  '허리 / 엉덩이',                  'build', 0.04),
]
# Fisher 비율 (σ_between² / σ_within²): 5명(SSH, YHS, 비니 인물, hf09 인물, 반바지 인물) 영상 21개로 추정한 값.
# 1 미만(사람 간 차이 < 사람 안 흔들림)인 특징은 가중치 0 으로 두고 참고로만 표시한다. DB 인원이 늘면 다시 추정할 것.
ID_WEIGHTS = {'h_w_sh': 2.55, 'h_neck': 1.48, 'h_bang': 2.29, 'h_side': 2.27, 'h_hair_area': 2.27,
              'f_wrist_fore': 4.60, 'b_hip_sh': 1.13}
GROUP_NAME = dict(head='머리 모양', limb='맨살 팔다리 폭 곡선', build='덩치')


def _frontal(raw):
    k = np.asarray(raw['kxy'], float)
    sw = np.linalg.norm(k[5] - k[6])
    torso = np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)
    return sw / (torso + 1e-9)


def _limb_feats(d, nm):
    w, c = np.asarray(d['w'], float), np.asarray(d['cov'], float)
    skin = c > 0.5
    out = {}
    if nm == 'shin':
        # 종아리(0.1~0.6)와 발목(0.7~0.9) 구간이 모두 맨살이어야 함
        calf_i = [i for i in range(1, 6) if skin[i]]
        ank_i = [i for i in range(6, 9) if skin[i]]
        if len(calf_i) >= 3 and len(ank_i) == 3:
            j = calf_i[int(np.argmax(w[calf_i]))]
            cmax = w[j]
            out['s_ankle_calf'] = float(w[ank_i].min() / cmax)
            out['s_calf_pos'] = float(T[j])
            out['_s_calf_len'] = (float(cmax / d['L']), d['L'])
    else:
        if skin[3:9].all():
            fmax = w[3:6].max()
            out['f_wrist_fore'] = float(w[6:9].min() / fmax)
            out['_f_fore_len'] = (float(fmax / d['L']), d['L'])
    return out


def video_id(result):
    """영상 1개 -> 특징별 프레임 값 목록."""
    frames = [f for f in result.get('frames', []) if 'raw' in f and 'id' in f['raw']]
    if not frames:
        return {}
    # 맨살 판정이 베이지 계열 옷을 피부로 오인하는 경우가 있어, 영상 단위 옷 형태로 한 번 더 거른다
    vf = result.get('video_features', {})
    shin_ok = vf.get('pants_len', 1.0) < 0.75       # 반바지일 때만 정강이 맨살 특징 사용
    fore_ok = vf.get('sleeve_len', 1.0) < 0.75      # 반팔(또는 걷은 소매)일 때만 아래팔 특징 사용
    fr = np.array([_frontal(f['raw']) for f in frames])
    fmax = np.percentile(fr, 95)
    vals, lens = {}, {}
    for f, r in zip(frames, fr):
        idm = f['raw']['id']
        front = r / fmax >= FRONT
        h = idm.get('head')
        if front and h and h['head_w'] > 0:
            vals.setdefault('h_w_sh', []).append(h['head_w'] / h['sh_w'])
            vals.setdefault('h_aspect', []).append(h['head_h'] / h['head_w'])
            vals.setdefault('h_neck', []).append(h['neck_w'] / h['head_w'])
            vals.setdefault('h_eye', []).append(h['eye_y'] / h['head_h'])
            if 'bang' in h:
                vals.setdefault('h_bang', []).append(h['bang'] / h['head_h'])
            if 'side_hair' in h:
                vals.setdefault('h_side', []).append(h['side_hair'] / h['head_h'])
            if 'hair_area' in h:
                vals.setdefault('h_hair_area', []).append(h['hair_area'])
        for key, d in idm.get('limb', {}).items():
            side, nm = key.split('_')
            if (nm == 'shin' and not shin_ok) or (nm == 'forearm' and not fore_ok):
                continue
            for k, v in _limb_feats(d, nm).items():
                if k.startswith('_'):
                    lens.setdefault((k, side), []).append(v)
                else:
                    vals.setdefault(k, []).append(v)
        b = idm.get('build')
        if front and b and b['shoulder']['w'] > 0 and b['hip']['w'] > 0:
            vals.setdefault('b_sh_torso', []).append(b['shoulder']['w'] / b['torso_len'])
            vals.setdefault('b_waist_sh', []).append(b['waist']['w'] / b['shoulder']['w'])
            vals.setdefault('b_hip_sh', []).append(b['hip']['w'] / b['shoulder']['w'])
            vals.setdefault('b_waist_hip', []).append(b['waist']['w'] / b['hip']['w'])
    # 길이로 나누는 특징: 화면과 평행에 가까운 프레임(길이가 최대의 85% 이상)만
    for (k, side), items in lens.items():
        Lmax = max(L for _, L in items)
        vals.setdefault(k[1:], []).extend(v for v, L in items if L >= LEN_FRAC * Lmax)
    return vals


def summarize(vals):
    return {k: dict(value=float(np.median(v)), n=len(v)) for k, v in vals.items() if len(v) >= MIN_FRAMES}


def build_id_profile(db_results, inflate=1.5):
    per_video = [summarize(video_id(r)) for r in db_results]
    pooled = {}
    for r in db_results:
        for k, v in video_id(r).items():
            pooled.setdefault(k, []).extend(v)
    prof = summarize(pooled)
    floors = {k: f for k, _, _, f in ID_FEATURES}
    for k, d in prof.items():
        vs = [pv[k]['value'] for pv in per_video if k in pv]
        spread = float(np.sqrt(np.mean((np.array(vs) - d['value']) ** 2))) if len(vs) > 1 else 0.0
        d['sigma'] = max(spread * inflate, floors[k] * abs(d['value']))
    return prof


def compare_id(inp_summary, prof, weights=None):
    rows = []
    for k, lab, g, _ in ID_FEATURES:
        row = dict(key=k, label=lab, group='id_' + g, weight=(ID_WEIGHTS if weights is None else weights).get(k, 0.0))
        if k not in prof or k not in inp_summary:
            row.update(status='측정 불가 (조건 맞는 프레임 부족)', score=None)
        else:
            d = abs(inp_summary[k]['value'] - prof[k]['value'])
            z = d / prof[k]['sigma']
            row.update(db=prof[k]['value'], input=inp_summary[k]['value'], sigma=prof[k]['sigma'], z=z,
                       score=float(np.exp(-0.5 * z ** 2)) * 100, status='비교',
                       verdict='일치' if z <= 1 else ('유사' if z <= 2 else '차이'))
        rows.append(row)
    return rows


def fisher_table(persons):
    """persons: {이름: [영상 결과, ...]} -> 특징별 σ_within, σ_between, Fisher 비율, 사람별 평균."""
    summ = {p: [summarize(video_id(r)) for r in rs] for p, rs in persons.items()}
    table = []
    for k, lab, g, _ in ID_FEATURES:
        means, wv, per = [], [], {}
        for p, ss in summ.items():
            v = [s[k]['value'] for s in ss if k in s]
            per[p] = v
            if v:
                means.append(np.mean(v))
            if len(v) >= 2:
                wv.append(np.var(v, ddof=1))
        if len(means) < 2 or not wv:
            table.append(dict(key=k, label=lab, group=g, per=per, fisher=None))
            continue
        sw, sb = float(np.sqrt(np.mean(wv))), float(np.std(means, ddof=1))
        table.append(dict(key=k, label=lab, group=g, per=per, s_within=sw, s_between=sb,
                          fisher=sb ** 2 / (sw ** 2 + 1e-12)))
    return table
