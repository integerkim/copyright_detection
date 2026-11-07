"""
결과 설명 생성.
  - template_explanation : 규칙 기반 설명 (항상 생성, LLM 실패 시 대체)
  - LLM 설명             : 작은 LLM(config.yaml 의 llm.model_dir)이 템플릿 설명을 자연스러운 한국어로 다듬음

LLM은 판정을 하지 않는다. 코드가 계산한 점수와 판정을 문장으로 풀어줄 뿐이다.
출력에 입력에 없는 숫자가 있거나 판정 문구가 빠지면 버리고 템플릿 설명을 쓴다.
"""
import os
import re

from fusion import NAMES, VISION


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def facts(mods, fused):
    """LLM과 템플릿이 공통으로 쓰는 사실 목록 (문자열 줄)"""
    L = [f"최종 판정(영상 기반): {fused['verdict']}",
         f"최종 점수: {_f(fused['final_score'])} / 100 (모달리티별 증거 점수의 최댓값, 50 이상이면 도용 의심)",
         f"점수 패턴: {fused['pattern']}"]
    if fused["suspect_modalities"]:
        L.append("도용 모달리티: " + ", ".join(NAMES[k] for k in fused["suspect_modalities"]))
    if fused.get("cross_check"):
        L.append(f"교차 확인: {fused['cross_check']}")
    for k in list(VISION) + ["audio"]:
        m = mods.get(k)
        if m is None:
            continue
        if not m["available"]:
            L.append(f"[{NAMES[k]}] 판정 불가 - {m['note']}")
            continue
        subs = ", ".join(f"{n} {_f(v)}" for n, v in m["sub_scores"].items() if v is not None)
        line = (f"[{NAMES[k]}] 증거 점수 {_f(m['weighted_evidence'])} ({m['level']}), 원점수 {_f(m['raw_score'])}, "
                f"가장 닮은 DB: {m['matched']}, 세부: {subs}")
        ex = m.get("extra", {})
        if k == "motion":
            line += f", 2위 DB 점수 {_f(ex.get('second_best'))}"
            if ex.get("mirrored"):
                line += ", 좌우 반전된 동작으로 일치"
        if k == "avatar":
            line += f", 모듈 판정: {m['module_decision']}"
        if k == "audio":
            line += f", 모듈 판정: {m['module_decision']}"
            if ex.get("direct_copy_alert"):
                line += ", 원음 직접 재사용 경보"
        L.append(line)
    if mods.get("audio") is not None:
        L.append("참고: 오디오는 영상 판정과 별도로 판정함")
    return L


def _top_subs(m, n=2):
    subs = [(k, v) for k, v in m["sub_scores"].items() if v is not None and "%" not in k]
    subs.sort(key=lambda x: -x[1])
    return ", ".join(f"{k} {_f(v)}" for k, v in subs[:n])


def _why(m):
    """모달리티별 근거 한 문장"""
    k, ex = m["modality"], m.get("extra", {})
    if k == "motion":
        s = (f"모션은 DB의 '{m['matched']}' 동작과 Overall {_f(m['raw_score'])}점으로 가장 비슷했고 "
             f"(2위 {_f(ex.get('second_best'))}점), 세부로는 {_top_subs(m)} 순으로 높았습니다")
        if ex.get("mirrored"):
            s += ". 좌우를 뒤집은 동작으로 일치했습니다"
        return s
    if k == "face":
        return (f"얼굴은 DB 인물 '{m['matched']}'와 신원 유사도 {_f(m['raw_score'])}점이었고, "
                f"프레임 {_f(m['sub_scores'].get('프레임 일관성(%)'), 0)}%에서 같은 사람 수준으로 일치했습니다")
    if k == "avatar":
        return (f"아바타는 DB 인물 '{m['matched']}'와 신원 단서 {_f(m['sub_scores']['신원 단서 S_id'])}점, "
                f"외형 {_f(m['sub_scores']['외형 S_appear'])}점이었습니다 (모듈 판정: {m['module_decision']})")
    if k == "audio":
        return (f"오디오는 보호 음원 '{m['matched']}'와 비교해 종합 {_f(m['raw_score'])}점이었고, "
                f"{_top_subs(m)} 항목이 가장 높았습니다")
    return ""


def template_explanation(mods, fused):
    """정확한 사실만 담은 규칙 기반 설명. LLM은 이 문장을 다듬기만 한다."""
    v = fused["verdict"]
    if v == "판정 불가":
        return "판정 가능한 모달리티가 없어 도용 여부를 판단할 수 없습니다."
    S = []
    sus, rev = fused["suspect_modalities"], fused["review_modalities"]
    if sus:
        names = ", ".join(NAMES[k] for k in sus)
        S.append(f"최종 판정은 '{v}'입니다. {fused['pattern']}이며, 도용 모달리티는 {names}입니다 "
                 f"(최종 점수 {_f(fused['final_score'])}).")
    elif rev:
        S.append(f"최종 판정은 '{v}'입니다. {fused['pattern']}이라 사람이 직접 확인할 필요가 있습니다 "
                 f"(최종 점수 {_f(fused['final_score'])}).")
    else:
        S.append(f"최종 판정은 '{v}'입니다. {fused['pattern']}으로, 어느 모달리티도 DB와 뚜렷하게 일치하지 않았습니다 "
                 f"(최종 점수 {_f(fused['final_score'])}).")
    for k in sus + rev:
        S.append(_why(mods[k]) + ".")
    if fused.get("cross_check"):
        S.append(fused["cross_check"] + ".")
    low = [k for k in VISION if k in mods and mods[k]["available"] and mods[k]["level"] == "도용 근거 부족"]
    if low and (sus or rev):
        S.append(", ".join(NAMES[k] for k in low) + " 모달리티는 DB와 뚜렷하게 일치하지 않았습니다.")
    for k in [k for k in VISION if k in mods and not mods[k]["available"]]:
        S.append(f"{NAMES[k]} 모달리티는 판정하지 못했습니다 ({mods[k]['note']}).")
    a = mods.get("audio")
    if a is not None:
        if a["available"]:
            S.append(f"오디오는 영상과 별도로 판정했으며 결과는 '{a['level']}'입니다. " + _why(a) + ".")
        else:
            S.append(f"오디오는 판정하지 않았습니다 ({a['note']}).")
    S.append("이 결과는 기술적 유사도 지표이며 법적 저작권 침해를 확정하지 않습니다.")
    return " ".join(S)


SYSTEM = (
    "너는 한국어 문장 다듬기 도우미다. 주어진 저작권 도용 탐지 결과 설명을 읽기 쉽게 다시 쓴다. "
    "내용을 추가하거나 빼지 않고, 판정과 숫자는 원문 그대로 쓴다. 원문에 없는 추론이나 인과관계를 만들지 않는다."
)

USER_TMPL = (
    "다음 설명을 같은 내용으로, 보고서에 들어갈 자연스러운 한국어 문단으로 다듬어라.\n"
    "- 첫 문장에 최종 판정 '{verdict}'을 그대로 쓴다.\n"
    "- 모든 숫자와 이름(DB 이름, 인물 이름)은 원문 그대로 쓴다.\n"
    "- 원문에 있는 사실만 쓴다. 목록·제목 없이 한 문단, '~습니다' 체.\n\n"
    "원문:\n{text}"
)


def _numbers(text):
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)]


def validate(text, fact_lines, verdict):
    if verdict not in text:
        return False, f"판정 문구 '{verdict}' 누락"
    allowed = _numbers("\n".join(fact_lines))
    for x in _numbers(text):
        if x <= 10 and float(x).is_integer():   # 개수 표현 (2개, 3가지 등)
            continue
        if not any(abs(x - a) <= 0.55 for a in allowed):
            return False, f"입력에 없는 숫자 {x}"
    if re.search(r"[一-鿿]", text):     # 한자 섞임 (소형 모델에서 가끔 발생)
        return False, "한자 섞임"
    return True, ""


class Explainer:
    def __init__(self, model_dir, max_new_tokens=450, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
        self.max_new_tokens, self.device = max_new_tokens, device

    def generate(self, source_text, verdict, sample=False):
        import torch
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(text=source_text, verdict=verdict)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        kw = dict(do_sample=True, temperature=0.6, top_p=0.9) if sample else dict(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens, repetition_penalty=1.05,
                                      pad_token_id=self.tok.eos_token_id, **kw)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def explain(self, source_text, verdict, tries=3):
        """source_text: template_explanation() 결과. 검증 실패 시 (마지막 출력, False, 사유)"""
        why = ""
        for i in range(tries):
            text = self.generate(source_text, verdict, sample=i > 0)
            ok, why = validate(text, [source_text], verdict)
            if ok:
                return text, True, ""
        return text, False, why
