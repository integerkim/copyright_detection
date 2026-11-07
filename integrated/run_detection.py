"""
통합 저작권 도용 탐지: 비디오 -> 모달리티별 탐지 (copyright_database/ 와 비교) -> 정규화/판정 -> 설명.

  python integrated/run_detection.py                       # suspect_data/ 의 영상을 모두 검사
  python integrated/run_detection.py --video a.mp4 b.mp4 --skip motion --no_llm

결과: results/summary.txt     검사한 영상 전체 요약 (영상별 판정 한 줄)
      results/<영상 이름>/
  report.txt   사람이 읽는 요약 (판정, 모달리티별 점수, 설명)
  report.json  전체 수치
  motion/ face/ avatar/ audio/   모달리티별 원본 결과
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yaml  # noqa: E402

import explain as X  # noqa: E402
import fusion as FU  # noqa: E402
import modalities as MO  # noqa: E402

ORDER = ("motion", "face", "avatar", "audio")


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def render(video, mods, fused, expl, expl_src, elapsed):
    L = ["=" * 78, "저작권 도용 탐지 결과", "=" * 78,
         f"입력 영상 : {video}",
         f"최종 판정 : {fused['verdict']}" + (f"   (도용 모달리티: {', '.join(FU.NAMES[k] for k in fused['suspect_modalities'])})"
                                            if fused["suspect_modalities"] else ""),
         f"최종 점수 : {_f(fused['final_score'])} / 100   (영상 모달리티 증거 점수의 최댓값, 50 이상 = 도용 의심)",
         f"패턴     : {fused['pattern']}"]
    if fused.get("cross_check"):
        L.append(f"교차 확인 : {fused['cross_check']}")
    a = fused.get("audio")
    if a is not None:
        L.append(f"오디오    : {a['level'] if a['available'] else '판정 불가'}"
                 + (f" (증거 점수 {_f(a['evidence'])})" if a["available"] else f" - {a['note']}") + "  [별도 판정]")
    L += ["", "-" * 78, f"{'모달리티':<8}{'증거':>7}{'원점수':>8}  {'판정':<10}가장 닮은 DB", "-" * 78]
    for k in ORDER:
        m = mods.get(k)
        if m is None:
            continue
        if not m["available"]:
            L.append(f"{FU.NAMES[k]:<8}{'-':>7}{'-':>8}  {'판정 불가':<10}{m['note']}")
            continue
        L.append(f"{FU.NAMES[k]:<8}{_f(m['weighted_evidence']):>7}{_f(m['raw_score']):>8}  {m['level']:<10}{m['matched']}")
        L.append("          세부: " + ", ".join(f"{n} {_f(v)}" for n, v in m["sub_scores"].items() if v is not None))
        if k == "motion":
            ex = m["extra"]
            L.append(f"          일치 구간 {ex['segment_sec'][0]:.1f}-{ex['segment_sec'][1]:.1f}s, "
                     f"좌우반전 {'예' if ex['mirrored'] else '아니오'}, 2위(다른 모션) {_f(ex['second_best'])}, "
                     f"1위-2위 {_f(ex['margin_over_second'])}")
        if k in ("avatar", "audio", "face"):
            L.append(f"          모듈 판정: {m['module_decision']}")
    L += ["", "-" * 78, f"설명 ({expl_src})", "-" * 78, expl, "",
          "* 점수는 기술적 유사도 지표이며 법적 저작권 침해를 확정하지 않습니다.",
          f"* 처리 시간 {elapsed:.0f}s. 모달리티별 원본 결과는 각 하위 폴더에 있습니다."]
    return "\n".join(L) + "\n"


def detect(video, cfg, skip=(), use_llm=True, explainer=None, out_root=None, reuse=False):
    t0 = time.time()
    name = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(out_root or MO.P(cfg["output_root"]), name)
    os.makedirs(out_dir, exist_ok=True)
    gpu = cfg.get("gpu", 0)

    mods = {}
    prev = os.path.join(out_dir, "report.json")
    base_elapsed = 0.0
    if reuse and os.path.exists(prev):   # 모달리티 결과 재사용: 정규화/판정/설명만 다시 계산
        old_js = json.load(open(prev, encoding="utf-8"))
        old = old_js["modalities"]
        base_elapsed = old_js.get("modality_elapsed_sec", old_js.get("elapsed_sec", 0.0))
        mods = {k: v for k, v in old.items() if k not in skip and cfg[k].get("enabled", True)}
        if mods.get("motion", {}).get("available"):
            mods["motion"] = MO.motion_from_json(mods["motion"]["result_dir"])
        print(f"[reuse] {prev}")
    for k in ORDER:
        if k in mods or k in skip or not cfg[k].get("enabled", True):
            continue
        t = time.time()
        print(f"[{k}] running ...", flush=True)
        mods[k] = MO.run_safe(k, video, out_dir, cfg[k], gpu)
        m = mods[k]
        print(f"[{k}] {'score ' + _f(m['raw_score']) if m['available'] else m['note']}  ({time.time() - t:.0f}s)", flush=True)

    modality_elapsed = base_elapsed + (time.time() - t0)   # 재사용 시: 처음 모달리티를 돌린 시간
    fused = FU.fuse(mods, cfg)
    fact_lines = X.facts(mods, fused)
    expl_tmpl = X.template_explanation(mods, fused)
    expl, src, llm_note = expl_tmpl, "규칙 기반", ""
    if use_llm and explainer is not None and fused["verdict"] != "판정 불가":
        text, ok, why = explainer.explain(expl_tmpl, fused["verdict"])
        if ok:
            expl, src = text, f"LLM: {explainer.name}"
        else:
            llm_note = f"LLM 출력 검증 실패({why}) -> 규칙 기반 설명 사용"
            src = "규칙 기반 (" + llm_note + ")"

    elapsed = base_elapsed + (time.time() - t0)
    txt = render(video, mods, fused, expl, src, elapsed)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(input=video, fused=fused, modalities=mods, facts=fact_lines,
                       explanation=expl, explanation_source=src, template_explanation=expl_tmpl,
                       llm_note=llm_note, elapsed_sec=elapsed, modality_elapsed_sec=modality_elapsed),
                  f, ensure_ascii=False, indent=1, default=float)
    print(txt)
    return out_dir


def _pad(s, w):
    """한글(전각) 폭을 2칸으로 계산해 왼쪽 정렬"""
    import unicodedata
    n = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, w - n)


def summarize(rows, out_root):
    """여러 영상을 검사했을 때 영상별 판정을 한 표로 정리 (results/summary.txt, summary.json)"""
    L = [f"{_pad('영상', 30)} {_pad('최종 판정', 15)} {'점수':>4}  도용 모달리티 / 오디오(별도)", "-" * 90]
    for r in rows:
        f = r["fused"]
        mods = ", ".join(FU.NAMES[k] for k in f["suspect_modalities"]) or "-"
        au = f.get("audio") or {}
        au_txt = f"오디오 {au['level']} ({_f(au['evidence'])})" if au.get("available") else "오디오 없음"
        L.append(f"{_pad(r['name'], 30)} {_pad(f['verdict'], 15)} {_f(f['final_score']):>6}  {mods} / {au_txt}")
    txt = "\n".join(L) + "\n"
    open(os.path.join(out_root, "summary.txt"), "w", encoding="utf-8").write(txt)
    json.dump([dict(name=r["name"], input=r["input"], verdict=r["fused"]["verdict"],
                    final_score=r["fused"]["final_score"], suspect_modalities=r["fused"]["suspect_modalities"],
                    report=os.path.join(r["out_dir"], "report.txt")) for r in rows],
              open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print("=" * 90 + "\n전체 요약 (" + os.path.join(out_root, "summary.txt") + ")\n" + txt)


def main():
    ap = argparse.ArgumentParser(description="통합 저작권 도용 탐지")
    ap.add_argument("--video", nargs="*", default=None,
                    help="검사할 영상 또는 폴더 (생략하면 config 의 suspect_dir = suspect_data/ 전체)")
    ap.add_argument("--config", default=os.path.join(MO.PROJECT, "config.yaml"))
    ap.add_argument("--out", default=None, help="결과 루트 폴더 (기본: config의 output_root)")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(ORDER))
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="이전 report.json 의 모달리티 결과를 재사용 (config 의 정규화/판정 값만 바꿨을 때)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.gpu is not None:
        cfg["gpu"] = a.gpu
    MO.CACHE = MO.P(cfg.get("cache_dir", "cache"))
    out_root = os.path.abspath(a.out) if a.out else MO.P(cfg["output_root"])

    videos = []
    for v in (a.video or [MO.P(cfg.get("suspect_dir", "suspect_data"))]):
        videos += MO._files(os.path.abspath(v), MO.VIDEO_EXT, recursive=False) if os.path.isdir(v) else [os.path.abspath(v)]
    if not videos:
        sys.exit(f"검사할 영상이 없습니다: {a.video or cfg.get('suspect_dir')}")
    print(f"검사할 영상 {len(videos)}개: " + ", ".join(os.path.basename(v) for v in videos), flush=True)

    explainer = None
    if not a.no_llm and cfg["llm"].get("enabled", True):
        md = MO.P(cfg["llm"]["model_dir"])
        if os.path.isdir(md):
            explainer = X.Explainer(md, cfg["llm"].get("max_new_tokens", 450),
                                    device="cpu" if cfg["gpu"] < 0 else f"cuda:{cfg['gpu']}")
        else:
            print(f"[warn] LLM 모델 폴더 없음: {md} -> 규칙 기반 설명만 사용")
    rows = []
    for i, v in enumerate(videos, 1):
        print(f"\n{'=' * 90}\n[{i}/{len(videos)}] {v}", flush=True)
        od = detect(v, cfg, set(a.skip), explainer is not None, explainer, out_root, a.reuse)
        js = json.load(open(os.path.join(od, "report.json"), encoding="utf-8"))
        rows.append(dict(name=os.path.basename(od), input=v, out_dir=od, fused=js["fused"]))
    summarize(rows, out_root)


if __name__ == "__main__":
    main()
