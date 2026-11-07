"""
모달리티별 실행 어댑터.
각 함수는 학생별 모듈을 그대로 호출하고, 결과를 아래 공통 형식으로 돌려준다.

  {
    "modality": "motion" | "face" | "avatar" | "audio",
    "available": bool,          # 판정 가능 여부 (사람/얼굴/오디오 없음 등이면 False)
    "raw_score": float | None,  # 모듈 원점수 (0~100)
    "sub_scores": {이름: 점수},  # 세부 metric
    "matched": str | None,      # 가장 닮은 DB 항목
    "module_decision": str,     # 모듈 자체 판정 문구
    "note": str,                # 판정 불가 사유 등
    "result_dir": str,
  }
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = os.path.join(PROJECT, "cache")   # DB 특징 캐시 루트 (run_detection 이 config 의 cache_dir 로 덮어씀)
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")


def P(rel):
    return rel if os.path.isabs(rel) else os.path.join(PROJECT, rel)


def _env():
    env = os.environ.copy()
    env["TORCH_HOME"] = P("weights/torch")              # torchvision detector 가중치
    env["YOLO_CONFIG_DIR"] = P("weights/yolo")
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    return env


def _empty(modality, note, result_dir=None):
    return dict(modality=modality, available=False, raw_score=None, sub_scores={}, matched=None,
                module_decision="판정 불가", note=note, result_dir=result_dir)


def _files(root, exts, recursive=True):
    """root 아래에서 확장자가 exts 인 파일 (숨김 폴더 제외)"""
    pat = os.path.join(root, "**", "*") if recursive else os.path.join(root, "*")
    return sorted(f for f in glob.glob(pat, recursive=recursive)
                  if f.lower().endswith(exts) and os.path.isfile(f) and "/." not in f[len(root):])


def _stamp(files):
    """캐시 무효화 키: 파일 이름, 크기, 수정 시각"""
    return "\n".join(f"{os.path.basename(f)}:{os.path.getsize(f)}:{int(os.path.getmtime(f))}" for f in files)


def _ffmpeg(name="ffmpeg"):
    sib = os.path.join(os.path.dirname(sys.executable), name)
    return sib if os.path.exists(sib) else (shutil.which(name) or name)


# ====================================================================== Motion
def sync_motion_db(cfg, gpu=0, log=None):
    """DB 폴더에 넣은 영상을 SMPL npz 로 변환해 캐시에 둔다 (이미 변환된 것은 건너뜀). 캐시 폴더 반환"""
    cache = os.path.join(CACHE, "db", "motion")
    os.makedirs(cache, exist_ok=True)
    pre = P("motion_infringement/motion/preprocess/motion_preprocess.py")
    for v in _files(P(cfg["db_dir"]), VIDEO_EXT):
        npz = os.path.join(cache, os.path.splitext(os.path.basename(v))[0] + "-smpl_pose_traj.npz")
        if os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(v):
            continue
        print(f"[motion] DB 영상 -> SMPL 변환: {os.path.relpath(v, PROJECT)}", flush=True)
        r = subprocess.run([sys.executable, pre, "--video", v, "--out", npz, "--hybrik_root", P(cfg["hybrik_root"]),
                            "--gpu", str(max(gpu, 0))], cwd=os.path.dirname(pre), env=_env(),
                           stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[motion] 변환 실패 (사람 미검출 등), DB에서 제외: {v}", flush=True)
    return cache


def run_motion(video, out_dir, cfg, gpu=0):
    rd = os.path.join(out_dir, "motion")
    os.makedirs(rd, exist_ok=True)
    root = P("motion_infringement")
    with open(os.path.join(rd, "db_sync.log"), "w") as log:
        db_cache = sync_motion_db(cfg, gpu, log)
    cmd = [sys.executable, os.path.join(root, "motion", "code", "motion_main.py"),
           "--video", os.path.abspath(video), "--result_dir", rd,
           "--db_dir", P(cfg["db_dir"]), db_cache, "--hybrik_root", P(cfg["hybrik_root"]),
           "--workers", str(cfg.get("workers", 32)), "--gpu", str(max(gpu, 0))]
    with open(os.path.join(rd, "run.log"), "w") as log:
        proc = subprocess.run(cmd, cwd=os.path.join(root, "motion", "code"), env=_env(),
                              stdout=log, stderr=subprocess.STDOUT)
    js = os.path.join(rd, "result.json")
    if proc.returncode != 0 or not os.path.exists(js):
        tail = open(os.path.join(rd, "run.log")).read()[-600:]
        note = "영상에서 사람을 찾지 못함" if "No person" in tail else f"motion 실행 실패: {tail.strip().splitlines()[-1] if tail.strip() else ''}"
        return _empty("motion", note, rd)
    return motion_from_json(rd)


def _motion_key(t):
    """같은 곡·파트·퍼포머 (중복 저장된 DB 파일 포함) 를 하나로 보는 키"""
    if t["part"] != "-":
        return (t["song"], t["part"], t["performer"])
    return (t["db_file"].split("-smpl_pose_traj")[0],)


def motion_from_json(rd):
    r = json.load(open(os.path.join(rd, "result.json")))
    b = r["best"]
    # 2위: 1위와 다른 DB 모션 중 최고 점수 (같은 모션의 중복 파일은 건너뜀)
    others = [t for t in r["top"][1:] if _motion_key(t) != _motion_key(b)]
    second = others[0]["overall"] if others else None
    name = b["db_file"].split("-smpl_pose_traj")[0]
    if b["part"] != "-":
        name = f"{b['song']} / {b['part']} / {b['performer']}"
    return dict(
        modality="motion", available=True, raw_score=b["overall"],
        sub_scores={"PSC (포즈 형태)": b["PSC"], "AC (관절 협응)": b["AC"], "DC (동작 흐름)": b["DC"]},
        matched=name,
        extra={"mirrored": b["mirror"], "segment_sec": b["segment_sec"], "second_best": second,
               "margin_over_second": None if second is None else b["overall"] - second,
               "db_size": r["db_size"]},
        module_decision=f"가장 유사한 DB 모션: {name} (Overall {b['overall']:.1f})",
        note="", result_dir=rd)


# ====================================================================== Face
def run_face(video, out_dir, cfg, gpu=0):
    rd = os.path.join(out_dir, "face")
    sys.path.insert(0, P("face_infringement"))
    import face_identity as F
    r = F.run(os.path.abspath(video), P(cfg["db_dir"]), rd, gpu=gpu, weights_dir=P("weights/insightface"),
              cache_dir=os.path.join(CACHE, "db", "face"))
    b = r["best"]
    if b is None:
        return _empty("face", f"유효한 얼굴 프레임 없음 (검출 {r['n_detected']}/{r['n_sampled']})", rd)
    subs = {"신원 유사도": b["score"], "정면 얼굴": b["score_frontal"], "측면 얼굴": b["score_side"],
            "프레임 일관성(%)": 100 * b["consistency"]}
    second = r["ranking"][1] if len(r["ranking"]) > 1 else None
    return dict(
        modality="face", available=True, raw_score=b["score"], sub_scores=subs, matched=b["person"],
        extra={"cos_identity": b["cos_identity"], "n_valid_frames": r["n_valid"],
               "second_best": None if second is None else f"{second['person']} ({second['score']:.1f})"},
        module_decision=r["decision"], note="", result_dir=rd)


# ====================================================================== Avatar
def run_avatar(video, out_dir, cfg, gpu=0):
    rd = os.path.join(out_dir, "avatar")
    os.makedirs(rd, exist_ok=True)
    demo = P("avatar_infringement/demo")
    for sub in ("avatar/code", "avatar/preprocess"):
        sys.path.insert(0, os.path.join(demo, sub))
    import avatar_main as AM
    import avatar_metric as AMet
    device = str(gpu) if gpu >= 0 else "cpu"
    feat_root = os.path.join(CACHE, "db", "avatar")

    inp = AM.load_or_extract([os.path.abspath(video)], os.path.join(rd, "features"), 3, device)[0]
    if not inp.get("n_frames_valid"):
        return _empty("avatar", "전신이 보이는 프레임이 없음", rd)

    cands = []
    for db_dir in sorted(glob.glob(os.path.join(P(cfg["db_root"]), "*"))):
        if not os.path.isdir(db_dir):
            continue
        db_name = os.path.basename(db_dir).replace("_Vid", "")
        vids = sorted(glob.glob(os.path.join(db_dir, cfg.get("pattern", "*.mp4"))))
        if not vids:
            continue
        cdir = os.path.join(feat_root, db_name)
        key_file, key = os.path.join(cdir, "_key.txt"), _stamp(vids)
        stale = not os.path.exists(key_file) or open(key_file).read() != key
        db_res = AM.load_or_extract(vids, cdir, 3, device, force=stale)
        if stale:
            open(key_file, "w").write(key)
        views = [AMet.build_profile([x], f"{db_name}/{x['name']}") for x in db_res]
        if cfg.get("db_view", "auto") != "auto":
            sel = [v for v in views if v["name"].split("/")[-1].startswith(cfg["db_view"])]
            views = sel or views
        for v in views:
            c = AMet.compare(inp, v)
            if c["total"] is not None:
                cands.append((v["name"], c))
    if not cands:
        return _empty("avatar", "DB와 비교 가능한 특징이 부족함", rd)
    cands.sort(key=lambda x: -x[1]["total"])
    name, c = cands[0]
    rows = [dict(label=x["label"], group=x["group"], score=x["score"], status=x.get("verdict") or x["status"])
            for x in c["rows"] if x["score"] is not None]
    json.dump(dict(matched=name, result=c, all=[(n, x["identity"], x["appearance"], x["total"]) for n, x in cands]),
              open(os.path.join(rd, "result.json"), "w"), ensure_ascii=False, indent=1, default=float)
    return dict(
        modality="avatar", available=True, raw_score=c["total"],
        sub_scores={"신원 단서 S_id": c["identity"], "외형 S_appear": c["appearance"]},
        matched=name.split("/")[0], extra={"matched_view": name, "clothing": c["clothing"],
                                            "db_clothing": c["db_clothing"],
                                            "top_features": sorted(rows, key=lambda x: -x["score"])[:4],
                                            "low_features": sorted(rows, key=lambda x: x["score"])[:3]},
        module_decision=c["decision"], note="", result_dir=rd)


# ====================================================================== Audio
def extract_music(video, out_dir, min_rms_db=-45.0):
    """입력 영상의 오디오 트랙을 분석용 WAV로 분리. (wav 경로, 사유) 반환"""
    probe = subprocess.run([_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "a",
                            "-show_entries", "stream=codec_name", "-of", "csv=p=0", video],
                           capture_output=True, text=True)
    if not probe.stdout.strip():
        return None, "입력 영상에 오디오 트랙이 없음"
    wav = os.path.join(out_dir, "input_audio.wav")
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", video, "-vn", "-ac", "2", "-ar", "44100",
                        "-c:a", "pcm_s16le", wav], capture_output=True, text=True)
    if r.returncode != 0:
        return None, f"오디오 분리 실패: {r.stderr.strip()[:200]}"
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(wav, always_2d=True)
    if len(x) < sr:  # 1초 미만
        return None, "오디오가 너무 짧음 (1초 미만)"
    rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    if rms_db < min_rms_db:
        return None, f"오디오 트랙이 거의 무음 (RMS {rms_db:.1f} dB)"
    return wav, f"오디오 분리 완료 ({len(x) / sr:.1f}초, RMS {rms_db:.1f} dB)"


def run_audio(video, out_dir, cfg, gpu=0):
    rd = os.path.join(out_dir, "audio")
    wav, why = extract_music(os.path.abspath(video), rd, cfg.get("min_rms_db", -45))
    if wav is None:
        return _empty("audio", why, rd)
    refs = _files(P(cfg["db_dir"]), VIDEO_EXT + AUDIO_EXT) if cfg.get("db_dir") else []
    refs += sorted({f for pat in cfg.get("references", []) for f in glob.glob(P(pat))})
    if not refs:
        return _empty("audio", "등록된 보호 음원이 없음", rd)
    root = P("audio_infringement")
    best = None
    for ref in refs:
        od = os.path.join(rd, os.path.splitext(os.path.basename(ref))[0])
        with open(os.path.join(rd, "run.log"), "a") as log:
            subprocess.run([sys.executable, "-m", "audio.code.audio_main", "--query", wav, "--reference", ref,
                            "--config", os.path.join(root, "audio", "audio_config.yaml"),
                            "--output-dir", od, "--no-json-stdout"],
                           cwd=root, env=_env(), stdout=log, stderr=subprocess.STDOUT)
        js = os.path.join(od, "result.json")
        if os.path.exists(js):
            r = json.load(open(js, encoding="utf-8"))
            r["_ref"] = ref
            if best is None or r["track_score"] > best["track_score"]:
                best = r
    if best is None:
        return _empty("audio", "audio 실행 실패 (run.log 참고)", rd)
    ms = best["evidence"]["metric_scores"]
    return dict(
        modality="audio", available=True, raw_score=float(best["track_score"]),
        sub_scores={"화성 진행": ms["tonal_harmony"], "리듬·그루브": ms["rhythm_groove"],
                    "음색 궤적": ms["timbre_trajectory"], "직접 재사용(fingerprint)": ms["direct_fingerprint"]},
        matched=os.path.basename(best["_ref"]),
        extra={"decision_code": best["decision"], "direct_copy_alert": best["direct_copy_alert"],
               "directionality": best["evidence"]["directionality"], "extract": why},
        module_decision={"high_suspicion": "높은 유사도", "review": "검토 필요", "low_evidence": "근거 부족"}[best["decision"]],
        note=why, result_dir=rd)


RUNNERS = {"motion": run_motion, "face": run_face, "avatar": run_avatar, "audio": run_audio}


def run_safe(name, video, out_dir, cfg, gpu):
    try:
        return RUNNERS[name](video, out_dir, cfg, gpu)
    except Exception as e:  # 한 모달리티가 실패해도 나머지는 진행
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        with open(os.path.join(out_dir, name, "error.log"), "w") as f:
            f.write(traceback.format_exc())
        return _empty(name, f"{name} 실행 오류: {type(e).__name__}: {e}", os.path.join(out_dir, name))
