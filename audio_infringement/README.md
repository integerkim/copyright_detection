# Audio 모달리티: 음악 유사성 증거 분석

검사 영상의 음악을 보호 음원(`copyright_database/audio/`)과 비교해 0~100점의 `track_score`를 냅니다.

통합 실행(`./run.sh`)에서는 입력 영상에 오디오 트랙이 있으면 ffmpeg로 분리한 뒤 보호 음원마다 이 모듈을 실행하고,
가장 높은 점수를 씁니다. 환경은 루트의 `environment.yml`(conda `copyright`) 하나를 씁니다.

## 파일

```text
audio_infringement/
├── audio/
│   ├── preprocess/audio_preprocess.py   # 영상·음원에서 분석용 WAV 추출
│   ├── code/
│   │   ├── audio_main.py                # 전체 분석 실행
│   │   ├── evidence_feature.py          # Low/High feature 생성
│   │   ├── cover_similarity.py          # 화성 정렬과 점수 통합
│   │   └── report.py                    # 결과 파일 생성
│   ├── tests/                           # 단위 테스트 (합성 신호, pytest)
│   └── audio_config.yaml                # 점수 가중치와 판정 기준 (검토 40, 높음 70)
├── pyproject.toml                       # pytest 설정
└── run_analysis.sh                      # 단독 실행
```

## 단독 실행 (선택)

검사 파일 → 보호 원본 → 결과 폴더 순서입니다. WAV, MP3, MP4 등을 지원합니다.

```bash
conda activate copyright
cd audio_infringement
./run_analysis.sh query.mp3                                   # 원본 기본값: copyright_database/audio/media1_stage.mp4
./run_analysis.sh query.mp3 reference.mp4 ../results/audio_x  # 원본·결과 폴더 지정
python -m pytest -q                                           # 단위 테스트
```

## 점수 계산 구조

```text
원본 영상 + 검사 음원
  ↓
FFmpeg 분석용 WAV 추출
  ↓
Low-level 관측값 추출
  ↓
High-level 시간 패턴 생성
  ↓
영역별 패턴 정렬 및 유사도 계산
  ↓
시간 방향 검사
  ↓
최종 track_score
```

Low-level은 프레임 또는 이벤트 단위의 관측값이며 점수가 아닙니다. High-level은 해당 관측값을 시간 순서가 있는 음악 패턴으로 변환한 결과입니다.

| 분석 영역 | Low-level | High-level | 점수 계산 |
|---|---|---|---|
| Tonal/Harmony | Essentia HPCP | 조옮김을 보정한 화성 진행의 binary cross-similarity | Smith–Waterman 국소 정렬 |
| Rhythm/Groove | onset, BPM, beat | 박자별 16-bin groove sequence | subsequence-DTW |
| Timbre | MFCC frame | MFCC timbre trajectory | subsequence-DTW |
| Direct reuse | spectral peak hash | 같은 시간 이동량에 모이는 hash 경로 | dominant offset 일치율 |

화성 분석은 조옮김과 일부 구간 재사용을, 리듬 분석은 타격 위치와 박자 패턴을, 음색 분석은 시간에 따른 스펙트럼 질감을, direct reuse 분석은 거의 동일한 오디오 구간을 탐지합니다. 리듬과 음색의 High-level 증거가 함께 나타나는지도 확인하므로 한 영역만 우연히 비슷한 무관 음원은 감점됩니다. 마지막으로 시간 방향 검사를 적용해 최종 점수를 생성합니다.

## 출력 (`results/<영상 이름>/audio/<보호 음원 이름>/`)

- `result.txt`: 사람이 읽는 분석 요약
- `result.json`: 전체 영역별 점수와 분석 근거
- `feature_scores.csv`: 영역별 점수
- `intermediate/`: 실제로 사용한 추출 WAV와 Low/High feature 배열

## 개발 당시 검증 기록

테스트 음원은 배포본에서 뺐고, 수치만 남깁니다. 기준 원본은 `media1_stage.mp4`입니다.

| 테스트 음원 | `track_score` |
|---|---:|
| tempo 1.15배 | 87.58 |
| pitch +2 semitone | 67.89 |
| 사용자 변형본 | 60.76 |
| 무관 음원 | 23.00 |
| 전체 역재생 | 18.37 |

`track_score`는 비교 검토를 위한 기술적 지표이며 법적 저작권 침해를 확정하지 않습니다.
