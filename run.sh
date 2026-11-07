#!/usr/bin/env bash
# 사용법: ./run.sh                          # suspect_data/ 의 영상을 모두 검사 (보호 DB: copyright_database/)
#         ./run.sh 영상.mp4 [폴더/ ...]      # 특정 영상/폴더만 검사
#         옵션: [--no_llm] [--skip motion face avatar audio] [--gpu 0] [--reuse]
# 결과: results/summary.txt (전체 요약), results/<영상 이름>/report.txt, report.json
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# conda 환경: COPYRIGHT_ENV > setup.sh 가 기록한 .conda_env > copyright
ENV_NAME="${COPYRIGHT_ENV:-$(cat "${ROOT}/.conda_env" 2>/dev/null || echo copyright)}"
VIDEOS=(); OPTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in --*) OPTS+=("$@"); break;; *) VIDEOS+=("$(realpath "$1")"); shift;; esac
done
cd "${ROOT}"
exec conda run -n "${ENV_NAME}" --no-capture-output python -W ignore integrated/run_detection.py ${VIDEOS[@]+--video "${VIDEOS[@]}"} ${OPTS[@]+"${OPTS[@]}"}
