# Motion 모달리티: DB 기반 모션 유사도 탐색

입력 비디오에서 춤 동작(SMPL 모션)을 추출해 보호 DB(`copyright_database/motion/`)의 모든 모션과 비교합니다.
결과로 **가장 유사한 DB 모션**(곡, 파트, 퍼포머, 일치 구간)과 **유사도 점수(0~100)**를 냅니다.

통합 실행(`./run.sh`)에서 자동으로 호출되므로 보통은 이 폴더를 직접 실행할 일이 없습니다.
환경은 루트의 `environment.yml`(conda `copyright`) 하나를 씁니다.

## 파일

```
motion_infringement/motion/
├── preprocess/motion_preprocess.py   # 비디오 → SMPL 모션(.npz) 추출 (HybrIK, third_party/HybrIK)
└── code/
    ├── motion_main.py                # 진입점: 추출 → DB 탐색 → result.txt / result.json / all_db_scores.csv
    ├── motion_metric.py              # 모션 유사도 metric (PSC / AC / DC → Overall)
    └── motion_canonical.py           # 비교 전 포즈 정규화 (twist 제거)
```

## 단독 실행 (선택)

```bash
conda activate copyright
cd motion_infringement/motion/code
python motion_main.py --video ../../../suspect_data/a.mp4 --result_dir ../../../results/a/motion
python motion_main.py --query_npz ../../../results/a/motion/a_smpl.npz      # 모션을 이미 추출했다면
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--db_dir` | `copyright_database/motion` | 비교할 DB 폴더 (여러 개 가능, 하위 폴더 포함). SMPL npz: `poses_axis_angle [T,24,3]`, `trans [T,3]`, 30fps |
| `--topk` | 5 | result.txt에 표시할 상위 DB 모션 개수 |
| `--stride_sec` | 0.5 | 부분 구간 탐색의 창 이동 간격 (초) |
| `--workers` | 32 | DB 비교 병렬 프로세스 수 |
| `--no_subsequence` | - | 부분 구간 탐색을 끄고 시퀀스 전체끼리만 비교 |
| `--no_canonical` | - | 포즈 정규화 끄기 |

소요 시간: HybrIK 추출은 5초 영상 기준 약 40초(GPU), DB 250개 탐색은 32 workers 기준 약 2~3분(CPU)입니다.

## 동작 방식

1. **모션 추출** (`motion_preprocess.py`): HybrIK로 프레임별 SMPL 24관절 회전을 추정하고, DB와 같은 30fps로 리샘플링합니다.
2. **포즈 정규화** (`motion_canonical.py`): query와 DB 양쪽에 적용합니다.
   - 관절 위치는 그대로 두고, 각 관절 회전을 뼈 방향만 맞추는 최소 회전(swing)으로 다시 계산합니다.
   - mocap 리타깃 SMPL과 영상에서 추정한 SMPL은 같은 자세라도 뼈 축 비틀림(twist) 값이 크게 다릅니다. 이 차이가 동작 차이로 잡히지 않게 하려는 단계입니다.
3. **부분 구간 탐색**: 짧은 쪽 시퀀스를 긴 쪽 위에서 0.5초 간격으로 밀며 비교하고, 가장 잘 맞는 구간을 고릅니다. 일부 구간만 가져간 영상에도 대응하기 위해서입니다. 좌우반전된 query도 같이 비교합니다.
4. **유사도 metric** (`motion_metric.py`):
   - PSC (Pose Signature Consistency): 상체 중심의 포즈 형태 (6D rotation)
   - AC (Articulation Consistency): 좌우·상하체·팔다리 간 협응 패턴
   - DC (Dynamic Flow Consistency): 속도·가속도·운동 에너지의 시간 흐름
   - **Overall = 0.5·PSC + 0.3·AC + 0.2·DC.** 세 항목은 공통 DTW 정렬 위에서 계산하고, 각 점수는 0.2 × DTW 유사도 + 0.8 × local correlation입니다.
5. **결과**: DB 모션을 Overall 점수 순으로 정렬합니다. 1위가 가장 유사한 모션이고, 점수는 유사도 × 100입니다.

## 출력 (`results/<영상 이름>/motion/`)

`result.txt`

```
---- Most similar DB motion ----
DB file            : cupid_chorus_1_KHJ_Skeleton 002-smpl_pose_traj.npz
Song / Part        : cupid / chorus_1
Performer          : KHJ
Mirrored match     : no
Matched DB segment : 0.0s - 8.0s

---- Similarity score (0-100) ----
Overall            :  64.33
  PSC (Pose Signature Consistency) :  62.82
  AC  (Articulation Consistency)   :  68.02
  DC  (Dynamic Flow Consistency)   :  62.58
```

- `result.json`: 통합 판정용 수치 (1위와 상위 목록)
- `all_db_scores.csv`: DB 전체 순위와 점수
- `<영상 이름>_smpl.npz`: 추출된 모션 (다시 실행할 때 재사용)

## 개발 당시 검증 기록

데모 영상은 배포본에서 뺐고, 수치만 남깁니다.

| 입력 영상 | 가장 유사한 DB 모션 | 일치 구간 | Overall | PSC | AC | DC | 2위 점수 |
|---|---|---|---|---|---|---|---|
| DB 모션(cupid_chorus_1_KHJ)을 따라 춘 생성 영상 | cupid / chorus_1 / KHJ | 0.0–8.0s | **64.33** | 62.82 | 68.02 | 62.58 | 59.16 |
| 위 영상 좌우반전 + 상체 기울이기 | cupid / chorus_1 / KHJ (좌우반전 일치) | 0.0–8.0s | **61.68** | 60.77 | 64.83 | 59.21 | 58.87 |
| DB 모션(cam00)을 따라 춘 생성 영상 | cam00 | 0.0–5.0s | **76.45** | 78.08 | 74.37 | 75.51 | 59.45 |
| 위 영상 좌우반전 | cam00 (좌우반전 일치) | 0.0–5.0s | **74.32** | 75.67 | 72.03 | 74.41 | 59.75 |

- 4개 모두 실제 원본 DB 모션을 1위로 찾았고, 원본에서 가져온 구간도 정확히 짚었습니다.
- 같은 안무(cupid chorus_1)를 다른 퍼포머가 춘 DB 모션은 30위 아래였습니다. 이 탐색은 "같은 안무"보다 "같은 녹화본"을 찾아냅니다.
- 입력 영상은 DB 모션을 다른 인물에게 입힌 생성 영상(MusePose)입니다.
