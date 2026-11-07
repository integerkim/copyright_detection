from __future__ import annotations
import argparse,json,re
from pathlib import Path
import yaml

from audio.preprocess.audio_preprocess import extract_audio
from audio.code.cover_similarity import combine_evidence,cover_song_evidence
from audio.code.evidence_feature import (acoustic_sequence_similarity,build_high_level,
    extract_low_level,fingerprint_match,load_audio,rhythm_similarity,save_feature_levels)
from audio.code.report import write_report


def source_tag(index:int,source:Path)->str:
    safe=re.sub(r"[^A-Za-z0-9._-]+","_",source.stem).strip("._") or "input"
    return f"query_{index:04d}_{safe}"


def analyze(source:Path,reference:Path,config:dict,work_dir:Path,tag:str,verbose:bool=False):
    log=print if verbose else lambda *a,**k:None; sr=config["audio"]["sample_rate"]
    query_wav=work_dir/"extracted_audio"/f"{tag}.wav"
    reference_wav=work_dir/"extracted_audio"/f"reference_{reference.stem}.wav"
    log("  [1/6] preprocess: FFmpeg로 분석용 WAV 생성")
    extract_audio(source,query_wav,sr,config["audio"]["channels"]); extract_audio(reference,reference_wav,sr,config["audio"]["channels"])
    ref_audio,analysis_sr=load_audio(reference_wav,22050); qry_audio,_=load_audio(query_wav,22050)
    log("  [2/6] low-level: onset/beat/BPM/MFCC 추출")
    ref_low=extract_low_level(ref_audio,analysis_sr); qry_low=extract_low_level(qry_audio,analysis_sr)
    log("  [3/6] high-level: groove/timbre 시간 패턴 생성")
    ref_high=build_high_level(ref_low); qry_high=build_high_level(qry_low)
    ref_files=save_feature_levels(ref_low,ref_high,work_dir/"features"/f"reference_{reference.stem}")
    qry_files=save_feature_levels(qry_low,qry_high,work_dir/"features"/tag)
    log("  [4/6] metric: tonal/rhythm/timbre/fingerprint 정렬 점수")
    cover=cover_song_evidence(reference_wav,query_wav)
    rhythm=rhythm_similarity(ref_high,qry_high)
    acoustic=acoustic_sequence_similarity(ref_high,qry_high)
    fingerprint=fingerprint_match(ref_audio,qry_audio)
    composite=combine_evidence(cover,rhythm,acoustic,fingerprint["score"],config["evidence_aggregation"])
    contributions={name:round(composite[f"{key}_score"]*weight,2) for name,key,weight in [
        ("tonal_harmony","cover",config["evidence_aggregation"]["cover"]),
        ("rhythm_groove","rhythm",config["evidence_aggregation"]["rhythm"]),
        ("timbre_trajectory","acoustic_sequence",config["evidence_aggregation"]["acoustic_sequence"]),
        ("direct_fingerprint","fingerprint",config["evidence_aggregation"]["fingerprint"])]}
    log("  [5/6] coefficient: 영역별 점수×가중치 합산 및 방향성 보정")
    evidence={
        "low_level":{"reference":{**ref_files,"bpm":round(float(ref_low["bpm"][0]),2),"onset_count":len(ref_low["onset_times"]),"beat_count":len(ref_low["beat_times"])},
                     "query":{**qry_files,"bpm":round(float(qry_low["bpm"][0]),2),"onset_count":len(qry_low["onset_times"]),"beat_count":len(qry_low["beat_times"])}},
        "high_level":{"tonal_harmony":{"representation":"OTI binary HPCP progression + Smith-Waterman","score":composite["cover_score"],"distance":cover["distance"]},
                      "rhythm_groove":{**rhythm,"representation":"16-bin beat-synchronous onset sequence","score":composite["rhythm_score"]},
                      "timbre_trajectory":{"representation":"MFCC frame trajectory + subsequence DTW","score":composite["acoustic_sequence_score"],"cost":acoustic["mfcc_cost"]},
                      "direct_fingerprint":{**fingerprint,"representation":"spectral frame hash + dominant time-offset histogram","score":composite["fingerprint_score"]}},
        "metric_scores":{"tonal_harmony":composite["cover_score"],"rhythm_groove":composite["rhythm_score"],
                         "timbre_trajectory":composite["acoustic_sequence_score"],"direct_fingerprint":composite["fingerprint_score"]},
        "coefficients":config["evidence_aggregation"],"contributions":contributions,
        "directionality":composite["directionality"],"direction_penalty":composite["direction_penalty"],
        "cross_feature_consensus":composite["consensus_score"],"consensus_penalty":composite["consensus_penalty"],
        "raw_weighted_score":composite["raw_score"],"track_score":composite["score"],"cover_details":cover,
    }
    return evidence


def main(argv=None):
    p=argparse.ArgumentParser(description="Audio reproduction evidence pipeline")
    p.add_argument("--query",type=Path,required=True); p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("audio/audio_config.yaml")); p.add_argument("--output-dir",type=Path,default=Path("result"))
    p.add_argument("--verbose",action="store_true"); p.add_argument("--no-json-stdout",action="store_true"); args=p.parse_args(argv)
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evidence=analyze(args.query,args.reference,config,args.output_dir/"intermediate",source_tag(0,args.query),args.verbose)
    score=evidence["track_score"]; thresholds=config["thresholds"]
    decision="high_suspicion" if score>=thresholds["track_high_suspicion"] else "review" if score>=thresholds["track_review"] else "low_evidence"
    direct=evidence["metric_scores"]["direct_fingerprint"]>=100*thresholds["fingerprint_alert"]
    result={"input":str(args.query),"reference":str(args.reference),"track_score":score,"decision":decision,
            "direct_copy_alert":direct,"evidence":evidence}
    log=print if args.verbose else lambda *a,**k:None; log("  [6/6] report: JSON/TXT/CSV 생성")
    write_report(result,args.output_dir)
    if not args.no_json_stdout:print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=="__main__":main()
