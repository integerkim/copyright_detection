#!/usr/bin/env bash
# 환경 생성 + 가중치 확인. 사용법: ./setup.sh [--env 이름] [--download]
#   --download : weights/ 또는 HybrIK 가중치가 없을 때 인터넷에서 받는다
#               (SMPL / SMPL-X 모델 파일은 라이선스상 자동 다운로드하지 않음 -> 아래 안내 참고)
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=copyright; DOWNLOAD=0
while [[ $# -gt 0 ]]; do case "$1" in
  --env) ENV_NAME="$2"; shift 2;; --download) DOWNLOAD=1; shift;; *) echo "unknown option $1"; exit 1;; esac; done

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env '${ENV_NAME}' 이미 있음 -> environment.yml 로 갱신"
  conda env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml"
else
  conda env create -n "${ENV_NAME}" -f "${ROOT}/environment.yml"
fi
conda run -n "${ENV_NAME}" pip install --no-build-isolation chumpy==0.70

PY="conda run -n ${ENV_NAME} --no-capture-output python"
if [[ ${DOWNLOAD} == 1 ]]; then
  cd "${ROOT}"
  mkdir -p weights/yolo weights/insightface/models weights/torch weights/llm third_party/HybrIK/pretrained_models
  [[ -f third_party/HybrIK/pretrained_models/hybrik_hrnet.pth ]] || \
    conda run -n "${ENV_NAME}" gdown 1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW -O third_party/HybrIK/pretrained_models/hybrik_hrnet.pth
  for w in yolo11m-pose.pt yolo11m-seg.pt; do
    [[ -f weights/yolo/$w ]] || wget -q -O weights/yolo/$w https://github.com/ultralytics/assets/releases/download/v8.3.0/$w; done
  if [[ ! -f weights/insightface/models/buffalo_l/w600k_r50.onnx ]]; then
    wget -q -O /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    unzip -q -o /tmp/buffalo_l.zip -d weights/insightface/models/buffalo_l && rm /tmp/buffalo_l.zip; fi
  TORCH_HOME=weights/torch ${PY} -c "import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)"
  [[ -f weights/llm/Qwen2.5-7B-Instruct/config.json ]] || ${PY} -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='weights/llm/Qwen2.5-7B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])"
fi

echo "[setup] 확인"
${PY} "${ROOT}/integrated/check_setup.py"
