"""
Avatar modality - metric
========================
preprocess 가 만든 영상 단위 특징(video_features)을

  1) DB 프로필(여러 카메라 영상의 평균 + 같은 사람 안에서의 흔들림 σ)로 묶고
  2) 입력 영상과 특징별로 비교해 0~1 유사도로 바꾼 뒤
  3) 체형 점수 / 외형 점수 / 최종 점수 (0~100) 와 판정을 만든다.

체형: avatar_body.py 의 기하 불변량(같은 직선 위 비, 평행 선분 비, 복비) + 몸 방향 구간별 두께 비
외형: 부위별 색(Lab), 질감(LBP), 옷 형태(소매/하의 길이)

특징별 유사도:  s = exp(-0.5 * (d / σ)^2)
   d : 입력과 DB의 차이 (스칼라는 |Δ|, 색은 ΔE, 질감은 χ² 거리)
   σ : max(DB 내부 흔들림, 최소 허용오차)  -> 같은 사람의 카메라/프레임 차이 수준이면 1에 가깝다
"""
import numpy as np

import avatar_body as B
import avatar_id as I

# ---------------------------------------------------------------- 특징 정의
# key, 한글 이름, 그룹, 가중치, 최소 허용오차(상대값 rel 또는 절대값 abs)
# 체형 특징은 avatar_body.py (기하 불변량) 에서 정의
BODY_FEATURES = []
APPEAR_FEATURES = [
    ('upper_color', '상의 색',        'color',   2.0, dict(abs=6.0)),
    ('lower_color', '하의 색',        'color',   2.0, dict(abs=6.0)),
    ('skin_color',  '피부 색',        'color',   1.0, dict(abs=5.0)),
    ('hair_color',  '머리카락 색',    'color',   1.0, dict(abs=6.0)),
    ('upper_lbp',   '상의 질감(LBP)', 'texture', 1.0, dict(abs=0.02)),
    ('lower_lbp',   '하의 질감(LBP)', 'texture', 1.0, dict(abs=0.02)),
    ('sleeve_len',  '소매 길이',      'shape',   1.5, dict(abs=0.10)),
    ('pants_len',   '하의 길이',      'shape',   1.5, dict(abs=0.10)),
]
SIGMA_INFLATE = 1.5   # 생성 영상은 실사보다 흔들림이 크므로 DB σ 를 넉넉하게

W_ID, W_APPEAR = 0.5, 0.5
TAU_ID, TAU_APPEAR = 70.0, 75.0   # 잠정값: 5명 데이터 기준. DB 인원이 늘면 타인 점수 분포(FAR)로 다시 정할 것


# ---------------------------------------------------------------- 거리 함수
def lab_de(a, b):
    """밝기(L)는 조명 영향이 커서 절반만 반영한 ΔE."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return float(np.sqrt((0.5 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def domcolor_de(A, B):
    """대표색 3개끼리 가장 가까운 색과의 ΔE 평균 (양방향)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ab = np.mean([min(lab_de(a, b) for b in B) for a in A])
    ba = np.mean([min(lab_de(b, a) for a in A) for b in B])
    return float((ab + ba) / 2)


def color_dist(fa, fb, region):
    """영역 색 거리 = 중앙값 ΔE 와 대표색 ΔE 의 평균."""
    d = [lab_de(fa[f'{region}_lab'], fb[f'{region}_lab'])]
    if f'{region}_domcolors' in fa and f'{region}_domcolors' in fb:
        d.append(domcolor_de(fa[f'{region}_domcolors'], fb[f'{region}_domcolors']))
    return float(np.mean(d))


def chi2(h1, h2):
    h1, h2 = np.asarray(h1, float), np.asarray(h2, float)
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-9)))


def has(f, key, kind):
    if kind == 'color':
        return f'{key.split("_")[0]}_lab' in f
    return key in f


def feat_dist(fa, fb, key, kind):
    if kind == 'color':
        return color_dist(fa, fb, key.split('_')[0])
    if kind == 'texture':
        return chi2(fa[key], fb[key])
    return abs(fa[key] - fb[key])


# ---------------------------------------------------------------- DB 프로필
def mean_feature(vals, key, kind):
    if kind == 'color':
        r = key.split('_')[0]
        out = {f'{r}_lab': np.mean([v[f'{r}_lab'] for v in vals], 0).tolist()}
        dc = [v[f'{r}_domcolors'] for v in vals if f'{r}_domcolors' in v]
        if dc:
            out[f'{r}_domcolors'] = np.median(np.array(dc), 0).tolist()
        return out
    if kind == 'texture':
        return {key: np.mean([v[key] for v in vals], 0).tolist()}
    return {key: float(np.mean([v[key] for v in vals]))}


def build_profile(video_results, name='DB'):
    """여러 DB 영상(카메라)의 video_features -> 평균 특징 + 특징별 σ + 맨살/옷 정보."""
    feats = [r['video_features'] for r in video_results if r.get('video_features')]
    covs = [r.get('video_coverage', {}) for r in video_results if r.get('video_features')]
    prof = dict(name=name, n_videos=len(feats), features={}, sigma={}, coverage={},
                videos=[r['name'] for r in video_results])
    prof['body'], prof['body_per_video'] = B.build_body_profile(
        [r for r in video_results if r.get('video_features')])
    prof['id'] = I.build_id_profile([r for r in video_results if r.get('video_features')])
    for key, _, kind, _, floor in APPEAR_FEATURES:
        vals = [f for f in feats if has(f, key, kind)]
        if len(vals) == 0:
            continue
        m = mean_feature(vals, key, kind)
        prof['features'].update(m)
        # 같은 사람 안의 흔들림: 각 카메라 영상이 평균에서 떨어진 거리의 RMS
        d = [feat_dist(v, m, key, kind) for v in vals]
        spread = float(np.sqrt(np.mean(np.square(d)))) if len(d) > 1 else 0.0
        if 'rel' in floor:
            base = abs(m[key]) * floor['rel']
        else:
            base = floor['abs']
        prof['sigma'][key] = max(spread * SIGMA_INFLATE, base)
        c = [cv[key] for cv in covs if key in cv]
        if c:
            prof['coverage'][key] = max(set(c), key=c.count)
    return prof


# ---------------------------------------------------------------- 비교
def compare(input_result, prof):
    fi = input_result['video_features']
    ci = input_result.get('video_coverage', {})
    fd = prof['features']
    rows = B.compare_body(B.summarize(*B.video_body(input_result)), prof['body'])
    rows += I.compare_id(I.summarize(I.video_id(input_result)), prof['id'])
    for key, label, kind, w, _ in APPEAR_FEATURES:
        group = 'appearance'
        row = dict(key=key, label=label, group=group, kind=kind, weight=w)
        if key not in prof['sigma'] or not has(fi, key, kind):
            row.update(status='측정 불가', score=None)
            rows.append(row)
            continue
        # 두께는 같은 조건(둘 다 맨살 또는 둘 다 옷 위)일 때만 체형 비교에 사용
        if kind == 'thick':
            cd, cin = prof['coverage'].get(key), ci.get(key)
            row['coverage'] = f'DB:{cd} / 입력:{cin}'
            if cd is not None and cin is not None and cd != cin:
                row.update(status='제외(옷/맨살 조건 다름)', score=None)
                rows.append(row)
                continue
        d = feat_dist(fi, fd, key, kind)
        sig = prof['sigma'][key]
        s = float(np.exp(-0.5 * (d / sig) ** 2))
        if kind in ('ratio', 'thick', 'shape'):
            row.update(db=fd[key], input=fi[key])
        elif kind == 'color':
            r = key.split('_')[0]
            row.update(db=fd[f'{r}_lab'], input=fi[f'{r}_lab'])
        row.update(dist=d, sigma=sig, score=s * 100, status='비교')
        rows.append(row)

    def group_score(pred):
        rs = [r for r in rows if pred(r['group']) and r['score'] is not None and r['weight'] > 0]
        if not rs:
            return None
        return float(sum(r['score'] * r['weight'] for r in rs) / sum(r['weight'] for r in rs))

    sb = group_score(lambda g: g == 'body')                # 체형 불변량: 참고용 (판정 미반영)
    sa = group_score(lambda g: g == 'appearance')          # 외형 (DB 자산)
    si = group_score(lambda g: g.startswith('id_'))        # 신원 단서 (머리 모양, 맨살 폭, 덩치)
    parts = [(w, s) for w, s in ((W_ID, si), (W_APPEAR, sa)) if s is not None]
    total = sum(w * s for w, s in parts) / sum(w for w, _ in parts) if parts else None
    return dict(identity=si, appearance=sa, body=sb, total=total, decision=decide(si, sa), rows=rows,
                clothing=clothing_labels(fi), db_clothing=clothing_labels(fd))


def decide(si, sa):
    if si is None or sa is None:
        return '판단 보류 (측정 가능한 프레임 부족)'
    hi, ha = si >= TAU_ID, sa >= TAU_APPEAR
    if hi and ha:
        return 'DB 도용 의심 (신원 단서·외형 모두 일치)'
    if hi and not ha:
        return 'ID 도용 의심 (신원 단서 일치·옷 다름 → Face 점수로 교차 확인)'
    if ha and not hi:
        return '외형 모방 의심 (옷은 유사, 신원 단서 불일치)'
    return '도용 아님 (신원 단서·외형 모두 불일치)'


def clothing_labels(f):
    out = {}
    if 'sleeve_len' in f:
        out['상의'] = '긴팔' if f['sleeve_len'] > 0.75 else ('반팔' if f['sleeve_len'] > 0.2 else '민소매')
    if 'pants_len' in f:
        out['하의'] = '긴바지' if f['pants_len'] > 0.8 else ('반바지' if f['pants_len'] > 0.25 else '짧은 하의')
    return out
