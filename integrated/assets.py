"""모달리티별로 필요한 가중치/모델 파일 목록 (check_setup.py 와 run_detection.py 의 사전 점검이 같이 쓴다)."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DL = "./setup.sh --download"
SMPL = "smplify.is.tue.mpg.de 에서 SMPLify zip 을 받아 models/ 안의 같은 이름 파일을 이 위치에 넣기 (README_RUN.md 1절)"

# 모달리티 -> [(이름, 상대 경로, 해결 방법)]
REQUIRED = {
    "motion": [
        ("HybrIK checkpoint", "third_party/HybrIK/pretrained_models/hybrik_hrnet.pth", DL),
        ("SMPL neutral", "third_party/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl", SMPL),
        ("H36M J_regressor", "third_party/HybrIK/model_files/J_regressor_h36m.npy", "저장소에 포함된 파일 (git checkout 확인)"),
        ("H36M mean beta", "third_party/HybrIK/model_files/h36m_mean_beta.npy", "저장소에 포함된 파일 (git checkout 확인)"),
        ("torchvision Faster R-CNN", "weights/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", DL),
    ],
    "face": [
        ("InsightFace buffalo_l", "weights/insightface/models/buffalo_l/w600k_r50.onnx", DL),
    ],
    "avatar": [
        ("YOLO11 pose", "weights/yolo/yolo11m-pose.pt", DL),
        ("YOLO11 seg", "weights/yolo/yolo11m-seg.pt", DL),
    ],
    "audio": [],
}
OPTIONAL = [
    ("Qwen2.5-7B-Instruct (설명 문장용, 없으면 규칙 기반 설명)", "weights/llm/Qwen2.5-7B-Instruct/config.json", DL),
]
MIN_BYTES = 1000  # 다운로드 실패로 남은 빈 파일을 걸러내기 위한 최소 크기 (h36m_mean_beta.npy 는 208B 라 예외)


def present(rel):
    p = os.path.join(ROOT, rel)
    return os.path.isfile(p) and (os.path.getsize(p) >= MIN_BYTES or rel.endswith(".npy"))


def missing(modalities):
    """주어진 모달리티에 필요한데 없는 파일 [(모달리티, 이름, 경로, 해결 방법)]"""
    return [(m, n, p, h) for m in modalities for n, p, h in REQUIRED.get(m, []) if not present(p)]
