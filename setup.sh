#!/usr/bin/env bash
# 설치: conda 환경 생성 -> 가중치 다운로드 -> 설치 점검.  기록은 setup.log 에 남는다.
#   ./setup.sh --download              # GitHub 에서 받은 경우 (가중치가 저장소에 없음)
#   ./setup.sh --download --no_llm     # 설명용 LLM(약 15GB) 은 받지 않음 (규칙 기반 설명만)
#   ./setup.sh --env 이름 --download   # conda 환경 이름 지정 (기본 copyright). run.sh 가 같은 환경을 쓰도록 .conda_env 에 기록
# SMPL 모델 파일은 라이선스상 자동으로 받지 않습니다 -> README_RUN.md 1절.
# 중간에 실패하면 원인을 해결한 뒤 같은 명령으로 다시 실행하세요 (이미 받은 파일은 건너뜀).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=copyright; DOWNLOAD=0; LLM=1
while [[ $# -gt 0 ]]; do case "$1" in
  --env) ENV_NAME="$2"; shift 2;; --download) DOWNLOAD=1; shift;; --no_llm) LLM=0; shift;;
  *) echo "알 수 없는 옵션: $1  (사용법: ./setup.sh [--env 이름] [--download] [--no_llm])"; exit 1;; esac; done

cd "${ROOT}"
LOG="${ROOT}/setup.log"
exec > >(tee "${LOG}") 2>&1
STEP="시작"
trap 'echo; echo "[setup] 실패: ${STEP} 단계 (setup.sh ${LINENO}행). 위 오류와 ${LOG} 를 확인하고, 해결 후 같은 명령으로 다시 실행하세요."' ERR
step() { STEP="$1"; echo; echo "==== [setup] $1"; }

H=third_party/HybrIK
smpl_check() {
  local ok=1
  local f="${H}/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
  [[ -f "$f" ]] || { echo "  !! 없음: $f"; ok=0; }
  if [[ $ok == 0 ]]; then
    echo "  SMPL 중립 모델은 라이선스상 직접 받아야 합니다 (smplify.is.tue.mpg.de 의 SMPLify zip 안 models/, README_RUN.md 1절). 없으면 motion 만 동작하지 않습니다."
    echo "    받은 파일 -> ${f}   (파일 이름 그대로)"
  fi
  return 0
}

step "0/4 사전 확인"
command -v conda >/dev/null || { echo "conda 가 없습니다. Miniconda 설치 후 다시 실행하세요."; exit 1; }
command -v wget >/dev/null && command -v unzip >/dev/null || [[ ${DOWNLOAD} == 0 ]] || { echo "wget / unzip 이 필요합니다 (apt install wget unzip)"; exit 1; }
echo "  환경 이름: ${ENV_NAME}   다운로드: $([[ ${DOWNLOAD} == 1 ]] && echo 예 || echo 아니오)   LLM: $([[ ${LLM} == 1 ]] && echo 예 || echo 아니오)"
smpl_check

step "1/4 conda 환경 '${ENV_NAME}'"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  이미 있음 -> environment.yml 로 갱신"
  conda env update -n "${ENV_NAME}" -f environment.yml
else
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

step "2/4 chumpy (HybrIK 의존성)"
conda run -n "${ENV_NAME}" pip install --no-build-isolation chumpy==0.70
echo "${ENV_NAME}" > .conda_env   # run.sh 가 이 환경으로 실행
PY="conda run -n ${ENV_NAME} --no-capture-output python"

FAILED=()
# fetch <이름> <받을 경로> <명령...>: 임시 파일로 받고 성공했을 때만 옮긴다 (실패해도 빈 파일이 남지 않게)
fetch() {
  local name="$1" out="$2"; shift 2
  if [[ -s "${out}" ]]; then echo "  [있음] ${name}"; return 0; fi
  echo "  [받는 중] ${name}"
  mkdir -p "$(dirname "${out}")"
  local tmp="${out}.part"; rm -f "${tmp}"
  if "$@" "${tmp}" && [[ -s "${tmp}" ]]; then mv "${tmp}" "${out}"; else rm -f "${tmp}"; FAILED+=("${name}"); echo "  !! ${name} 실패"; fi
}
get_hybrik() { conda run -n "${ENV_NAME}" gdown 1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW -O "$1"; }
get_url()    { wget -q --tries=3 -O "$2" "$1"; }

if [[ ${DOWNLOAD} == 1 ]]; then
  step "3/4 가중치 다운로드"
  fetch "HybrIK checkpoint" ${H}/pretrained_models/hybrik_hrnet.pth get_hybrik
  for w in yolo11m-pose.pt yolo11m-seg.pt; do
    fetch "YOLO11 ${w}" weights/yolo/${w} get_url https://github.com/ultralytics/assets/releases/download/v8.3.0/${w}; done
  if [[ ! -s weights/insightface/models/buffalo_l/w600k_r50.onnx ]]; then
    fetch "InsightFace buffalo_l" weights/insightface/buffalo_l.zip get_url https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    if [[ -s weights/insightface/buffalo_l.zip ]]; then
      mkdir -p weights/insightface/models/buffalo_l
      if unzip -q -o weights/insightface/buffalo_l.zip -d weights/insightface/models/buffalo_l; then rm weights/insightface/buffalo_l.zip
      else FAILED+=("InsightFace buffalo_l (압축 해제)"); fi
    fi
  else echo "  [있음] InsightFace buffalo_l"; fi
  echo "  [확인] torchvision Faster R-CNN"
  TORCH_HOME=weights/torch ${PY} -W ignore -c "import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)" \
    || FAILED+=("torchvision Faster R-CNN")
  if [[ ${LLM} == 1 ]]; then
    echo "  [확인] Qwen2.5-7B-Instruct (약 15GB, 끊겨도 다시 실행하면 이어 받음)"
    ${PY} -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='weights/llm/Qwen2.5-7B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])" \
      || FAILED+=("Qwen2.5-7B-Instruct")
  fi
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo; echo "  !! 다운로드 실패: ${FAILED[*]}"
    echo "     네트워크 확인 후 같은 명령으로 다시 실행하세요. HybrIK checkpoint 는 Google Drive 일일 한도에 걸릴 수 있습니다"
    echo "     (브라우저로 https://drive.google.com/file/d/1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW 받아 ${H}/pretrained_models/hybrik_hrnet.pth 로 저장)."
  fi
else
  step "3/4 가중치 다운로드 (건너뜀: --download 없음)"
fi

step "4/4 설치 점검"
smpl_check
trap - ERR
if ${PY} -W ignore integrated/check_setup.py; then
  echo; echo "[setup] 완료. 실행: ./run.sh   (환경 '${ENV_NAME}' 사용)"
else
  echo; echo "[setup] MISSING 항목이 남아 있습니다. 해결한 뒤 'python integrated/check_setup.py' 로 다시 확인하세요."
  [[ ${DOWNLOAD} == 0 ]] && echo "        가중치가 없다면: ./setup.sh --env ${ENV_NAME} --download"
  exit 1
fi
