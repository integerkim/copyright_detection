"""
Face 모달리티: 얼굴 신원(identity) 기반 도용 탐지
==================================================
입력 영상의 얼굴이 DB에 등록된 보호 인물(얼굴)과 같은 사람인지를 판단한다.

방법
  1) 프레임 샘플링 -> InsightFace SCRFD로 얼굴 검출 (가장 큰 얼굴 = 주 인물)
  2) 품질 필터: 검출 신뢰도, 얼굴 크기, 고개 각도(yaw)
  3) ArcFace(R50, buffalo_l) 512차원 임베딩 추출 (5점 랜드마크로 정렬)
  4) DB 인물마다 비교
       - template  : 입력 평균 임베딩 vs DB 평균 임베딩 코사인
       - frame     : 입력 프레임마다 가장 닮은 DB 프레임과의 코사인 -> 중앙값
       - pose별    : 정면(|yaw|<20°) / 측면 프레임 각각의 frame 유사도
       - 일관성    : 입력 프레임 중 코사인이 임계값을 넘은 비율
     identity cosine = 0.5 * template + 0.5 * frame
  5) 0~100 점수: 코사인을 구간 선형으로 변환 (COS_FLOOR -> 0, COS_TAU -> 50, COS_HIGH -> 100)

임계값 근거 (calibration.txt): 데모 영상 13개(인물 7명, 카메라 시점 다양)에서
  다른 사람 쌍 61개: cos 평균 0.145, 최대 0.276
  같은 사람 쌍 12개 (다른 카메라 시점 포함): cos 최소 0.777
  생성 영상(kdg_vid_df) vs KDG 원본: 0.62 ~ 0.77
인물 수가 적어 잠정값이다. DB 인물이 늘면 다른 사람 분포의 상위값(FAR)으로 다시 정할 것.

사용법
  python face_identity.py --input video.mp4 [--db_dir ../copyright_database/face] --out result_dir
  DB 구조: copyright_database/face/<인물이름>/*.mp4|*.jpg|*.png
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_DB = os.path.join(PROJECT, "copyright_database", "face")
DEFAULT_WEIGHTS = os.path.join(PROJECT, "weights", "insightface")

# ---- 품질 필터
MIN_DET = 0.60        # SCRFD 검출 신뢰도
MIN_FACE_PX = 24      # 얼굴 bbox 짧은 변 (원본 해상도 기준)
MAX_YAW = 60.0        # 이보다 옆으로 돌아간 얼굴은 제외
FRONTAL_YAW = 20.0    # 정면 판정

# ---- 코사인 -> 점수 (잠정값, calibration.txt 참고)
COS_FLOOR = 0.15      # 이하 = 0점 (다른 사람 평균 0.145)
COS_TAU = 0.40        # = 50점, 도용 의심 경계 (다른 사람 최대 0.276 보다 여유를 둠)
COS_HIGH = 0.70       # 이상 = 100점 (같은 사람 최소 0.777, 생성 영상 vs 원본 0.62~0.77)

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def cos_to_score(c):
    if c is None:
        return None
    if c <= COS_TAU:
        s = 50.0 * (c - COS_FLOOR) / (COS_TAU - COS_FLOOR)
    else:
        s = 50.0 + 50.0 * (c - COS_TAU) / (COS_HIGH - COS_TAU)
    return float(np.clip(s, 0.0, 100.0))


class FaceEncoder:
    def __init__(self, weights_dir=DEFAULT_WEIGHTS, gpu=0, det_size=960):
        try:
            import torch  # noqa: F401  (onnxruntime-gpu 가 torch 에 포함된 CUDA/cuDNN 라이브러리를 찾도록 먼저 로드)
        except ImportError:
            pass
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=weights_dir, providers=providers,
                                allowed_modules=["detection", "recognition", "landmark_3d_68"])
        self.app.prepare(ctx_id=gpu, det_size=(det_size, det_size), det_thresh=0.5)

    def encode_image(self, img):
        faces = self.app.get(img)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        w, h = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
        pitch, yaw, roll = (f.pose.tolist() if getattr(f, "pose", None) is not None else (0.0, 0.0, 0.0))
        return dict(emb=f.normed_embedding.astype(np.float32), det=float(f.det_score),
                    size=float(min(w, h)), yaw=float(yaw), pitch=float(pitch))


def read_frames(path, max_frames=60):
    """영상에서 균등 간격으로 최대 max_frames 장, 이미지는 1장."""
    if path.lower().endswith(IMAGE_EXT):
        img = cv2.imread(path)
        return [(0.0, img)] if img is not None else []
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = set(np.linspace(0, max(n - 1, 0), min(max_frames, max(n, 1))).astype(int).tolist())
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            out.append((i / fps, fr))
        i += 1
    cap.release()
    return out


def extract(path, enc, max_frames=60):
    frames = read_frames(path, max_frames)
    recs, n_det = [], 0
    for t, img in frames:
        r = enc.encode_image(img)
        if r is None:
            continue
        n_det += 1
        if r["det"] < MIN_DET or r["size"] < MIN_FACE_PX or abs(r["yaw"]) > MAX_YAW:
            continue
        r["t"] = float(t)
        recs.append(r)
    return dict(path=path, n_sampled=len(frames), n_detected=n_det, n_valid=len(recs),
                emb=np.stack([r["emb"] for r in recs]) if recs else np.zeros((0, 512), np.float32),
                yaw=np.array([r["yaw"] for r in recs]), size=np.array([r["size"] for r in recs]),
                t=np.array([r["t"] for r in recs]))


def _template(emb):
    m = emb.mean(0)
    return m / (np.linalg.norm(m) + 1e-8)


def compare(q, ref):
    """q, ref: extract() 결과 (ref는 같은 인물의 여러 파일을 합친 것)"""
    if len(q["emb"]) == 0 or len(ref["emb"]) == 0:
        return None
    sim = q["emb"] @ ref["emb"].T                 # [Nq, Nr]
    best = sim.max(1)                             # 프레임별 가장 닮은 DB 프레임
    c_tmpl = float(_template(q["emb"]) @ _template(ref["emb"]))
    c_frame = float(np.median(best))
    c_id = 0.5 * c_tmpl + 0.5 * c_frame
    frontal = np.abs(q["yaw"]) < FRONTAL_YAW
    pose = {
        "frontal": float(np.median(best[frontal])) if frontal.any() else None,
        "side": float(np.median(best[~frontal])) if (~frontal).any() else None,
    }
    return dict(
        cos_identity=c_id, cos_template=c_tmpl, cos_frame=c_frame,
        cos_frontal=pose["frontal"], cos_side=pose["side"],
        n_frontal=int(frontal.sum()), n_side=int((~frontal).sum()),
        consistency=float((best >= COS_TAU).mean()),
        score=cos_to_score(c_id),
        score_frontal=cos_to_score(pose["frontal"]), score_side=cos_to_score(pose["side"]),
    )


def load_db(db_dir, enc, cache_dir=None):
    db = {}
    for person in sorted(os.listdir(db_dir)):
        pdir = os.path.join(db_dir, person)
        if not os.path.isdir(pdir):
            continue
        files = sorted(f for f in glob.glob(os.path.join(pdir, "*")) if f.lower().endswith(VIDEO_EXT + IMAGE_EXT))
        if not files:
            continue
        cache = os.path.join(cache_dir, f"{person}.npz") if cache_dir else None
        key = "|".join(f"{os.path.basename(f)}:{os.path.getsize(f)}:{int(os.path.getmtime(f))}" for f in files)
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            if str(z["key"]) == key:
                db[person] = dict(emb=z["emb"], yaw=z["yaw"], files=files)
                continue
        parts = [extract(f, enc) for f in files]
        emb = np.concatenate([p["emb"] for p in parts])
        yaw = np.concatenate([p["yaw"] for p in parts])
        db[person] = dict(emb=emb, yaw=yaw, files=files)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, emb=emb, yaw=yaw, key=key)
    return db


def run(input_path, db_dir=DEFAULT_DB, out_dir=None, gpu=0, weights_dir=DEFAULT_WEIGHTS, cache_dir=None):
    enc = FaceEncoder(weights_dir, gpu)
    db = load_db(db_dir, enc, cache_dir=cache_dir or os.path.join(PROJECT, "cache", "db", "face"))
    q = extract(input_path, enc)
    rows = []
    for person, ref in db.items():
        r = compare(q, ref)
        if r is not None:
            rows.append(dict(person=person, **r))
    rows.sort(key=lambda r: -r["cos_identity"])
    best = rows[0] if rows else None
    if best is None:
        decision = "판정 불가 (유효한 얼굴 없음)"
    elif best["score"] >= 50:
        decision = f"도용 의심 ({best['person']} 얼굴과 동일 인물 수준)"
    else:
        decision = "도용 근거 부족"
    result = dict(
        modality="face", input=input_path,
        n_sampled=q["n_sampled"], n_detected=q["n_detected"], n_valid=q["n_valid"],
        best=best, ranking=rows, decision=decision,
        score=None if best is None else best["score"],
        thresholds=dict(cos_floor=COS_FLOOR, cos_tau=COS_TAU, cos_high=COS_HIGH),
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(format_text(result))
    return result


def format_text(r):
    L = ["==== Face Identity ====", f"입력: {os.path.basename(r['input'])}",
         f"얼굴 프레임: 유효 {r['n_valid']} / 검출 {r['n_detected']} / 샘플 {r['n_sampled']}"]
    b = r["best"]
    if b is None:
        L.append("유효한 얼굴이 없어 판정 불가")
        return "\n".join(L) + "\n"
    f = lambda x: "-" if x is None else f"{x:.3f}"
    g = lambda x: "-" if x is None else f"{x:.1f}"
    L += [f"가장 닮은 DB 인물: {b['person']}",
          f"Face 점수: {b['score']:.1f} / 100  (identity cos {b['cos_identity']:.3f}, 임계 {r['thresholds']['cos_tau']})",
          f"  template cos {f(b['cos_template'])} / frame cos {f(b['cos_frame'])}",
          f"  정면 {g(b['score_frontal'])}점 ({b['n_frontal']}프레임) / 측면 {g(b['score_side'])}점 ({b['n_side']}프레임)",
          f"  일관성: 프레임 {100 * b['consistency']:.0f}% 가 임계 초과",
          f"판정: {r['decision']}", "", "DB 인물별 순위:"]
    for i, x in enumerate(r["ranking"], 1):
        L.append(f"  {i}. {x['person']:<8} score {x['score']:5.1f}  cos {x['cos_identity']:.3f}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face modality: 얼굴 신원 기반 도용 탐지")
    ap.add_argument("--input", required=True)
    ap.add_argument("--db_dir", default=DEFAULT_DB)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "identity"))
    ap.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
    a = ap.parse_args()
    res = run(a.input, a.db_dir, a.out, a.gpu)
    print(format_text(res))
