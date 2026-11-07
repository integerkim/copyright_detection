"""
Avatar modality - main
======================
입력 영상 1개를 DB(촬영 영상들)와 비교해 체형/외형/최종 도용 점수(0~100)를 낸다.

  python avatar_main.py                               # 기본값: suspect_data/ 의 모든 mp4  vs  copyright_database/avatar/KDG
  python avatar_main.py --input ../../../../suspect_data/a.mp4 --db_dir ../../../../copyright_database/avatar/KDG --db_name KDG
  python avatar_main.py --self_check                  # DB 카메라 1개씩 빼서 '같은 사람' 기준 점수 확인

특징 JSON 이 없으면 preprocess 를 자동으로 실행한다.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'preprocess'))
import avatar_metric as M  # noqa: E402

FEAT_DIR = os.path.join(ROOT, '..', '..', 'cache', 'avatar_standalone')


def load_or_extract(videos, out_dir, stride, device, force=False):
    res, todo = [], []
    for v in videos:
        js = os.path.join(out_dir, os.path.splitext(os.path.basename(v))[0] + '.json')
        if os.path.exists(js) and not force:
            res.append(json.load(open(js, encoding='utf-8')))
        else:
            todo.append(v)
    if todo:
        import avatar_preprocess as AP
        parser = AP.PersonParser(device=device)
        for v in todo:
            res.append(AP.process_video(v, parser, out_dir, stride))
    order = {os.path.splitext(os.path.basename(v))[0]: i for i, v in enumerate(videos)}
    return sorted(res, key=lambda r: order.get(r['name'], 0))


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, list):
        return '(' + ', '.join(f'{x:.1f}' for x in v) + ')'
    return f'{v:.3f}'


def report(title, inp, prof, r):
    L = []
    L.append('=' * 78)
    L.append(f'[Avatar] {title}')
    L.append(f'  입력: {inp["name"]}  (유효 프레임 {inp["n_frames_valid"]}/{inp["n_frames_sampled"]})')
    L.append(f'  DB  : 가장 잘 맞는 DB 시점 = {prof["name"]}')
    if r.get('view_scores'):
        L.append('  시점별 점수 (신원 / 외형 / 최종): ' + ',  '.join(
            f'{n.split("/")[-1].split("_")[0]} {"-" if a is None else f"{a:.0f}"}/{"-" if b is None else f"{b:.0f}"}/{"-" if c is None else f"{c:.0f}"}'
            for n, a, b, c in r['view_scores']))
    L.append('-' * 78)
    s = lambda x: '-' if x is None else f'{x:5.1f}'
    L.append(f'  최종 도용 점수 (S_avatar) : {s(r["total"])} / 100')
    L.append(f'    - 신원 단서 (S_id)      : {s(r["identity"])} / 100   (머리 모양·맨살 폭·덩치, Fisher 가중)')
    L.append(f'    - 외형 점수 (S_appear)  : {s(r["appearance"])} / 100   (옷 색·질감·형태 = DB 자산)')
    L.append(f'    - [참고] 체형 불변량    : {s(r["body"])} / 100   (사람 간 차이가 작아 판정에 미반영)')
    L.append(f'  판정 : {r["decision"]}')
    L.append(f'  옷 형태  입력 {r["clothing"]}  /  DB {r["db_clothing"]}')
    L.append('-' * 78)
    L.append(f'  {"특징":<22}{"DB":>20}{"입력":>20}{"점수":>8}  비고')
    for g, gname in (('appearance', '외형 (DB 자산)'), ('id_head', '신원: 머리 모양'), ('id_limb', '신원: 맨살 팔다리 폭 곡선'),
                     ('id_build', '신원: 덩치'), ('body', '[참고] 체형 불변량')):
        L.append(f'  <{gname}>')
        for row in [x for x in r['rows'] if x['group'] == g]:
            sc = '-' if row['score'] is None else f'{row["score"]:.1f}'
            note = row['status'] if row['status'] != '비교' else ''
            if g.startswith('id_') and row['status'] == '비교':
                note = f'{row["verdict"]} (z={row["z"]:.1f})' + ('' if row['weight'] > 0 else ', 참고: Fisher<1 가중치 0')
            if 'coverage' in row:
                note = (note + ' ' + row['coverage']).strip()
            L.append(f'  {row["label"]:<22}{fmt(row.get("db")):>20}{fmt(row.get("input")):>20}{sc:>8}  {note}')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='Avatar modality: 체형/외형 기반 도용 점수')
    ap.add_argument('--input', nargs='*', default=sorted(glob.glob(os.path.join(ROOT, '..', '..', 'suspect_data', '*.mp4'))))
    ap.add_argument('--db_dir', default=os.path.join(ROOT, '..', '..', 'copyright_database', 'avatar', 'KDG'))
    ap.add_argument('--db_name', default='KDG')
    ap.add_argument('--pattern', default='*_24fps_5s.mp4')
    ap.add_argument('--result', default=os.path.join(ROOT, '..', '..', 'results', 'avatar_standalone', 'result.txt'))
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--db_view', default='cam00',
                    help="비교할 DB 카메라 (예: cam00). 입력과 같은 시점끼리 비교한다. 'auto' 면 점수가 가장 높은 시점을 고른다")
    ap.add_argument('--self_check', action='store_true', help='DB 카메라 leave-one-out 기준 점수도 출력 (auto 방식)')
    ap.add_argument('--force', action='store_true', help='특징을 다시 추출')
    args = ap.parse_args()

    db_videos = sorted(glob.glob(os.path.join(args.db_dir, args.pattern)))
    if not db_videos:
        ap.error(f'DB 영상이 없습니다: {args.db_dir}/{args.pattern}')
    db_res = load_or_extract(db_videos, os.path.join(FEAT_DIR, 'db', args.db_name), args.stride, args.device, args.force)
    prof = M.build_profile(db_res, args.db_name)
    os.makedirs(os.path.join(FEAT_DIR, 'db'), exist_ok=True)
    with open(os.path.join(FEAT_DIR, 'db', f'{args.db_name}_profile.json'), 'w', encoding='utf-8') as fp:
        json.dump(prof, fp, ensure_ascii=False, indent=1)

    # 시점별 DB 프로필: DB 카메라 영상마다 따로 프로필을 만들고, 입력과 가장 잘 맞는 시점을 고른다
    # (PPT의 max_k: 가장 닮은 DB 프레임/시점 기준. 카메라 간 흔들림이 σ를 키워 타인도 통과하는 문제를 막음)
    view_profs = [M.build_profile([r], f'{args.db_name}/{r["name"]}') for r in db_res]

    def best_match(inp, profs):
        cands = [(p, M.compare(inp, p)) for p in profs]
        cands = [c for c in cands if c[1]['total'] is not None] or cands
        cands.sort(key=lambda c: -(c[1]['total'] or 0))
        p, r = cands[0]
        r['matched_view'] = p['name']
        r['view_scores'] = [(c[0]['name'], c[1]['identity'], c[1]['appearance'], c[1]['total']) for c in cands]
        return p, r

    texts, summary, js = [], [], {}
    for v in args.input:
        name = os.path.splitext(os.path.basename(v))[0]
        sub = os.path.basename(os.path.dirname(os.path.abspath(v)))
        inp = load_or_extract([v], os.path.join(FEAT_DIR, 'input', sub), args.stride, args.device, args.force)[0]
        inp['name'] = f'{sub}/{name}'
        if args.db_view == 'auto':
            p_best, r = best_match(inp, view_profs)
        else:
            sel = [vp for vp in view_profs if vp['name'].split('/')[-1].startswith(args.db_view)]
            if not sel:
                ap.error(f'DB 에 {args.db_view} 시점이 없습니다')
            p_best, r = best_match(inp, sel)
        texts.append(report('입력 영상 vs DB', inp, p_best, r))
        summary.append((inp['name'], r))
        js[inp['name']] = r

    if args.self_check:
        for i, held in enumerate(db_res):
            rest = [vp for j, vp in enumerate(view_profs) if j != i]
            _, r = best_match(held, rest)
            summary.append((f'[참고·시점 제외] {held["name"][:5]}', r))
            js[f'self/{held["name"]}'] = r

    s = lambda x: '  -  ' if x is None else f'{x:5.1f}'
    head = ['=' * 78, '[Avatar] 요약  (점수 0~100, 잠정 기준: 신원 ≥ %.0f, 외형 ≥ %.0f)' % (M.TAU_ID, M.TAU_APPEAR),
            f'  {"입력":<34}{"일치 DB 시점":<22}{"신원":>6}{"외형":>6}{"최종":>6}  판정']
    for n, r in summary:
        head.append(f'  {n:<34}{r.get("matched_view", "-"):<22}{s(r["identity"]):>6}{s(r["appearance"]):>6}{s(r["total"]):>6}  {r["decision"]}')
    out = '\n'.join(head) + '\n\n' + '\n\n'.join(texts) + '\n'
    os.makedirs(os.path.dirname(args.result), exist_ok=True)
    with open(args.result, 'w', encoding='utf-8') as fp:
        fp.write(out)
    with open(os.path.splitext(args.result)[0] + '.json', 'w', encoding='utf-8') as fp:
        json.dump(js, fp, ensure_ascii=False, indent=1, default=float)
    print(out)


if __name__ == '__main__':
    main()
