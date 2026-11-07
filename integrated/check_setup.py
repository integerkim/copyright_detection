"""설치 확인: 패키지 import, GPU, 가중치/DB 파일 존재 여부를 점검한다.  python integrated/check_setup.py"""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ok = True


def check(name, cond, hint=""):
    global ok
    print(f"  [{'OK' if cond else 'MISSING'}] {name}" + ("" if cond else f"  -> {hint}"))
    ok &= bool(cond)


print("패키지")
import numpy as np  # noqa: E402
for n, t in [("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object),
             ("unicode", str), ("str", str)]:
    if not hasattr(np, n):
        setattr(np, n, t)
for mod in ["torch", "torchvision", "pytorch3d", "cv2", "ultralytics", "insightface", "onnxruntime",
            "librosa", "essentia", "soundfile", "transformers", "chumpy", "yaml"]:
    try:
        __import__(mod)
        check(mod, True)
    except Exception as e:  # noqa: BLE001
        check(mod, False, f"{type(e).__name__}: {e}")
check(f"numpy == 1.26.x (현재 {np.__version__})", np.__version__.startswith("1.26"), "pip install numpy==1.26.4")
try:
    import torch
    check(f"CUDA GPU ({torch.cuda.device_count()}개)", torch.cuda.is_available(), "GPU 없음: config.yaml 의 gpu: -1 (매우 느림)")
except Exception:  # noqa: BLE001
    pass

print("가중치")
W = {
    "HybrIK checkpoint": ("third_party/HybrIK/pretrained_models/hybrik_hrnet.pth", "./setup.sh --download"),
    "SMPL neutral (HybrIK)": ("third_party/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                              "SMPL 공식 사이트에서 라이선스 동의 후 받기 (HybrIK README model_files)"),
    "SMPL-X npz (HybrIK import 시 필요)": ("third_party/HybrIK/model_files/smplx/SMPLX_NEUTRAL.npz", "SMPL-X 공식 사이트"),
    "torchvision Faster R-CNN": ("weights/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", "./setup.sh --download"),
    "YOLO11 pose": ("weights/yolo/yolo11m-pose.pt", "./setup.sh --download"),
    "YOLO11 seg": ("weights/yolo/yolo11m-seg.pt", "./setup.sh --download"),
    "InsightFace buffalo_l": ("weights/insightface/models/buffalo_l/w600k_r50.onnx", "./setup.sh --download"),
    "Qwen2.5-7B-Instruct": ("weights/llm/Qwen2.5-7B-Instruct/config.json", "./setup.sh --download (없으면 규칙 기반 설명만)"),
}
for k, (p, h) in W.items():
    check(k, os.path.exists(os.path.join(ROOT, p)), h)

print("보호 DB (copyright_database/, 넣는 법: copyright_database/README.md)")
D = os.path.join(ROOT, "copyright_database")
n = lambda pat: len(glob.glob(os.path.join(D, pat), recursive=True))  # noqa: E731
V = ("mp4", "mov", "avi", "mkv", "webm", "m4v")
A = V + ("wav", "mp3", "flac", "m4a", "aac", "ogg")
n_motion = n("motion/**/*.npz") + sum(n(f"motion/**/*.{e}") for e in V)
check(f"motion: npz/영상 {n_motion}개", n_motion > 0, "copyright_database/motion/ 에 npz 또는 영상 추가")
check(f"face: 인물 {len([d for d in glob.glob(os.path.join(D, 'face/*')) if os.path.isdir(d)])}명", n("face/*/*") > 0,
      "copyright_database/face/<인물>/ 에 영상/사진 추가")
check(f"avatar: 인물 {len([d for d in glob.glob(os.path.join(D, 'avatar/*')) if os.path.isdir(d)])}명", n("avatar/*/*.mp4") > 0,
      "copyright_database/avatar/<인물>/ 에 영상 추가")
n_audio = sum(n(f"audio/**/*.{e}") for e in A)
check(f"audio: 보호 음원 {n_audio}개", n_audio > 0, "copyright_database/audio/ 에 음원/영상 추가")
n_sus = sum(len(glob.glob(os.path.join(ROOT, "suspect_data", f"*.{e}"))) for e in V)
print(f"  [INFO] suspect_data/ 검사 대상 영상 {n_sus}개")

print("\n" + ("모든 항목 정상. ./run.sh 로 suspect_data/ 의 영상을 검사해 보세요." if ok else "MISSING 항목을 해결하세요."))
sys.exit(0 if ok else 1)
