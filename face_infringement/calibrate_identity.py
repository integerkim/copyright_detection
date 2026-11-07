"""Face 임계값 점검: 영상/사진 쌍의 ArcFace 코사인 분포를 출력한다 (face_identity.py 의 COS_FLOOR/TAU/HIGH 근거).

  python face_infringement/calibrate_identity.py                       # copyright_database/face 의 파일끼리 비교
  python face_infringement/calibrate_identity.py --extra OTHER=a.mp4 X=b.mp4   # DB 밖 영상도 추가 (라벨=경로)

같은 라벨(인물 폴더 이름)끼리는 SAME, 다른 라벨끼리는 diff 로 집계한다.
기준: diff 의 최댓값 < COS_TAU(0.40) < SAME 의 최솟값 이면 현재 임계값이 맞다.
DB 인물이 늘면 이 스크립트로 다시 확인할 것. 개발 당시 측정 결과는 calibration.txt.
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import torch  # noqa: E402,F401  (onnxruntime 이 CUDA 라이브러리를 찾도록 먼저 import)
import face_identity as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default=os.path.join(PROJECT, "copyright_database", "face"))
    ap.add_argument("--extra", nargs="*", default=[], help="라벨=경로 (DB 밖 영상/사진)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    items = {}
    for pdir in sorted(glob.glob(os.path.join(a.db_dir, "*"))):
        if os.path.isdir(pdir):
            for f in sorted(glob.glob(os.path.join(pdir, "*"))):
                if f.lower().endswith(F.VIDEO_EXT + F.IMAGE_EXT):
                    items[f"{os.path.basename(pdir)}/{os.path.basename(f)}"] = (os.path.basename(pdir), f)
    for e in a.extra:
        label, path = e.split("=", 1)
        items[f"{label}/{os.path.basename(path)}"] = (label, path)
    if len(items) < 2:
        sys.exit("비교할 파일이 2개 이상 필요합니다")

    enc = F.FaceEncoder(os.path.join(PROJECT, "weights", "insightface"), a.gpu)
    E = {}
    for k, (pid, p) in items.items():
        E[k] = F.extract(p, enc)
        E[k]["pid"] = pid
        e = E[k]
        print(f"{k:30s} valid {e['n_valid']:2d}/{e['n_detected']:2d}/{e['n_sampled']:2d}")

    same, diff = [], []
    print()
    for x, y in itertools.combinations(E, 2):
        r = F.compare(E[x], E[y])
        if r is None:
            continue
        tag = "SAME" if E[x]["pid"] == E[y]["pid"] else "diff"
        (same if tag == "SAME" else diff).append(r["cos_identity"])
        print(f"{tag:5s} {x:30s} {y:30s} cos {r['cos_identity']:.3f}")
    print()
    if same:
        print(f"SAME n={len(same)}  min {min(same):.3f}  mean {np.mean(same):.3f}")
    if diff:
        print(f"diff n={len(diff)}  max {max(diff):.3f}  mean {np.mean(diff):.3f}")
    print(f"현재 기준: COS_FLOOR {F.COS_FLOOR}  COS_TAU {F.COS_TAU} (50점)  COS_HIGH {F.COS_HIGH}")


if __name__ == "__main__":
    main()
