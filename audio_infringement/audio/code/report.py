from __future__ import annotations
import csv,json
from pathlib import Path


def write_report(result:dict,output:Path)->None:
    output.mkdir(parents=True,exist_ok=True); evidence=result["evidence"]; metrics=evidence["metric_scores"]
    (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["[최종 평가]",f"track_score: {result['track_score']}/100",f"decision: {result['decision']}",f"direct_copy_alert: {result['direct_copy_alert']}","",
           "[Low-level 관측 요약]"]
    for side,data in evidence["low_level"].items():lines.append(f"{side}: BPM={data['bpm']}, onset={data['onset_count']}, beats={data['beat_count']}, file={data['low_level_file']}")
    lines += ["","[High-level 패턴 점수]"]
    for name,data in evidence["high_level"].items():lines.append(f"{name}: {data.get('score','not_weighted')} ({data['representation']})")
    lines += ["",f"cross-feature consensus: {evidence['cross_feature_consensus']}",
              f"directionality: {evidence['directionality']}"]
    (output/"result.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with (output/"feature_scores.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["feature","score","coefficient","contribution"])
        keymap={"tonal_harmony":"cover","rhythm_groove":"rhythm","timbre_trajectory":"acoustic_sequence","direct_fingerprint":"fingerprint"}
        for name,score in metrics.items():w.writerow([name,score,evidence["coefficients"][keymap[name]],evidence["contributions"][name]])
