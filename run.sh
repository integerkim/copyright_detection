#!/usr/bin/env bash
# 사용법: ./run.sh                          # suspect_data/ 의 영상을 모두 검사 (보호 DB: copyright_database/)
#         ./run.sh 영상.mp4 [폴더/ ...]      # 특정 영상/폴더만 검사
#         모드:  --all (기본, 4개 모달리티 모두) | --monitor (모션만 = --skip face avatar audio)
#         옵션: [--no_llm] [--skip motion face avatar audio] [--gpu 0] [--reuse]
# 결과: results/summary.txt (전체 요약), results/<영상 이름>/report.txt, report.json
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# conda 환경: COPYRIGHT_ENV > setup.sh 가 기록한 .conda_env > copyright
ENV_NAME="${COPYRIGHT_ENV:-$(cat "${ROOT}/.conda_env" 2>/dev/null || echo copyright)}"
# 모드 인자(--all / --monitor)는 위치와 상관없이 먼저 빼내고, 나머지는 기존 방식대로 처리한다.
MODE=(); ARGS=()
for a in "$@"; do
  case "$a" in
    --all) MODE=();;
    --monitor) MODE=(--skip face avatar audio);;
    *) ARGS+=("$a");;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
VIDEOS=(); OPTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in --*) OPTS+=("$@"); break;; *) VIDEOS+=("$(realpath "$1")"); shift;; esac
done
cd "${ROOT}"
exec conda run -n "${ENV_NAME}" --no-capture-output python -W ignore integrated/run_detection.py ${VIDEOS[@]+--video "${VIDEOS[@]}"} ${MODE[@]+"${MODE[@]}"} ${OPTS[@]+"${OPTS[@]}"}
