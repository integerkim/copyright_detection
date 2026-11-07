"""
모달리티별 원점수 -> 증거 점수(0~100) -> 최종 판정.

원점수는 모달리티마다 척도가 달라 직접 비교하거나 평균내면 안 된다
(예: motion은 관련 없는 DB 1위도 ~60점, audio는 60점이면 이미 '검토' 구간).
그래서 각 원점수를 모달리티별 기준점으로 구간 선형 정규화한다:
  floor -> 0,  review -> 35 (검토 필요 경계, 선택),  tau -> 50 (도용 의심 경계),  high -> 100
각 모듈이 자체 판정 기준을 가지고 있으면 (avatar 70/75, audio 40/70) 그 기준을 그대로 기준점으로 쓴다.

최종 판정 (vision: motion / face / avatar)
  - 가중 증거 점수 = weight_m * evidence_m
  - 최종 점수     = 가중 증거 점수의 최댓값  (한 모달리티만 확실히 베껴도 도용)
  - 도용 모달리티 = 가중 증거 점수 >= suspect 인 모달리티
  - 판정          : 도용 의심 / 검토 필요 / 도용 근거 부족
오디오는 별도로 판정한다 (include_audio_in_final 이면 최종에도 포함).
"""
import numpy as np

VISION = ("motion", "face", "avatar")
NAMES = {"motion": "모션", "face": "얼굴", "avatar": "아바타", "audio": "오디오"}


def normalize(raw, floor, tau, high, review=None, review_level=35.0):
    """구간 선형: floor->0, (review->review_level), tau->50, high->100"""
    if raw is None:
        return None
    xs, ys = [floor], [0.0]
    if review is not None:
        xs.append(review); ys.append(review_level)
    xs += [tau, high]; ys += [50.0, 100.0]
    return float(np.interp(raw, xs, ys))


def evidence(m, ncfg, rl=35.0):
    """모달리티 결과 m 에 evidence(0~100) 를 붙인다. rl = '검토 필요' 경계 점수"""
    if not m["available"]:
        m["evidence"] = None
        return m
    k = m["modality"]
    if k == "avatar":
        # 신원 단서 / 외형 중 강한 쪽 (avatar 모듈 판정표와 같은 논리: 둘 중 하나만 넘어도 의심)
        e_id = normalize(m["sub_scores"]["신원 단서 S_id"], **ncfg["avatar_id"], review_level=rl)
        e_ap = normalize(m["sub_scores"]["외형 S_appear"], **ncfg["avatar_appear"], review_level=rl)
        m["evidence_parts"] = {"신원 단서": e_id, "외형": e_ap}
        m["evidence"] = max(x for x in (e_id, e_ap) if x is not None)
    elif k == "motion":
        # Overall 자체가 높거나, 1위가 2위(다른 DB 모션)보다 두드러지면 근거로 본다
        e_ov = normalize(m["raw_score"], **ncfg["motion"], review_level=rl)
        mg = m.get("extra", {}).get("margin_over_second")
        e_mg = normalize(mg, **ncfg["motion_margin"], review_level=rl) if mg is not None else None
        m["evidence_parts"] = {"Overall": e_ov, "1위-2위 차이": e_mg}
        m["evidence"] = max(x for x in (e_ov, e_mg) if x is not None)
    else:
        m["evidence"] = normalize(m["raw_score"], **ncfg[k], review_level=rl)
    return m


def level(e, fcfg):
    if e is None:
        return "판정 불가"
    if e >= fcfg["suspect"]:
        return "도용 의심"
    if e >= fcfg["review"]:
        return "검토 필요"
    return "도용 근거 부족"


def fuse(mods, cfg):
    ncfg, fcfg = cfg["normalize"], cfg["fusion"]
    for m in mods.values():
        evidence(m, ncfg, fcfg["review"])
        w = fcfg["weights"].get(m["modality"], 1.0)
        m["weighted_evidence"] = None if m["evidence"] is None else min(100.0, w * m["evidence"])
        m["level"] = level(m["weighted_evidence"], fcfg)

    keys = list(VISION) + (["audio"] if fcfg.get("include_audio_in_final") else [])
    avail = [k for k in keys if k in mods and mods[k]["weighted_evidence"] is not None]
    if not avail:
        return dict(final_score=None, verdict="판정 불가", suspect_modalities=[], review_modalities=[],
                    dominant=None, pattern="판정 가능한 모달리티 없음", audio=_audio(mods, fcfg))

    final = max(mods[k]["weighted_evidence"] for k in avail)
    dominant = max(avail, key=lambda k: mods[k]["weighted_evidence"])
    sus = [k for k in avail if mods[k]["weighted_evidence"] >= fcfg["suspect"]]
    rev = [k for k in avail if fcfg["review"] <= mods[k]["weighted_evidence"] < fcfg["suspect"]]
    verdict = level(final, fcfg)

    # 패턴 서술 (LLM / 템플릿 설명의 뼈대)
    if len(sus) == len(avail) and len(avail) >= 2:
        pattern = "판정 가능한 모든 모달리티가 높음"
    elif len(sus) >= 2:
        pattern = f"{', '.join(NAMES[k] for k in sus)} 모달리티가 함께 높음"
    elif len(sus) == 1:
        pattern = f"{NAMES[sus[0]]} 모달리티만 특히 높음"
    elif rev:
        pattern = f"{', '.join(NAMES[k] for k in rev)} 모달리티가 경계 수준"
    else:
        pattern = "전체적으로 점수가 낮음"

    # 같은 보호 인물을 여러 모달리티가 가리키는지 (face / avatar 는 인물 이름이 같음)
    same = None
    if all(k in sus for k in ("face", "avatar")) and mods["face"]["matched"] == mods["avatar"]["matched"]:
        same = f"얼굴과 아바타가 모두 같은 보호 인물({mods['face']['matched']})을 가리킵니다"

    return dict(final_score=final, verdict=verdict, suspect_modalities=sus, review_modalities=rev,
                dominant=dominant, pattern=pattern, cross_check=same,
                unavailable=[k for k in keys if k in mods and mods[k]["weighted_evidence"] is None],
                audio=_audio(mods, fcfg))


def _audio(mods, fcfg):
    a = mods.get("audio")
    if a is None:
        return None
    return dict(available=a["available"], evidence=a.get("weighted_evidence"), level=a.get("level"),
                note=a["note"])
