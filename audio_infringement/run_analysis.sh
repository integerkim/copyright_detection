#!/usr/bin/env bash
# Usage: ./run_analysis.sh <query> [reference] [output_dir]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
QUERY="${1:?usage: ./run_analysis.sh <query audio/video> [reference] [output_dir]}"
REFERENCE="${2:-${ROOT}/../copyright_database/audio/media1_stage.mp4}"
OUTPUT="${3:-${ROOT}/../results/audio_standalone}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python environment not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${QUERY}" && -f "${REFERENCE}" ]] || { echo "Input file missing" >&2; exit 1; }
echo "[1/4] Input   query=${QUERY} reference=${REFERENCE}"
echo "[2/4] Low→High features and metric alignment"
cd "${ROOT}"
"${PYTHON_BIN}" -m audio.code.audio_main --query "${QUERY}" --reference "${REFERENCE}" \
  --config "${ROOT}/audio/audio_config.yaml" --output-dir "${OUTPUT}" --verbose --no-json-stdout
echo "[3/4] Score aggregation"
"${PYTHON_BIN}" - "${OUTPUT}/result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); e=d["evidence"]
for name,score in e["metric_scores"].items():
    print(f"  {name}: score={score}, contribution={e['contributions'][name]}")
print(f"  directionality={e['directionality']}, penalty={e['direction_penalty']}")
print(f"  cross_feature_consensus={e['cross_feature_consensus']}, penalty={e['consensus_penalty']}")
print(f"  track_score={d['track_score']} ({d['decision']})")
PY
echo "[4/4] Reports ${OUTPUT}/result.txt ${OUTPUT}/result.json ${OUTPUT}/feature_scores.csv"
