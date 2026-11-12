# 통합 저작권 도용 탐지 — 실행 가이드

`suspect_data/`에 넣은 비디오를 하나씩 **모션 / 얼굴 / 아바타 / 오디오** 4개 모달리티로 나눠
`copyright_database/`의 보호 데이터와 비교하고, 영상마다 다음을 출력합니다.

- 도용 여부
- 도용 모달리티
- 모달리티별 세부 점수
- 한국어 설명

이 폴더만 있으면 처음부터 끝까지 실행되도록 코드, DB, 모델 가중치(HybrIK 포함)가 모두 들어 있습니다.

```
copyright_database/     보호 DB (모달리티별 폴더: motion / face / avatar / audio)  ← 보호 데이터를 넣는 곳
suspect_data/           검사할 영상                                                ← 의심 영상을 넣는 곳
results/                검사 결과 (summary.txt + 영상별 폴더)
```

```
입력 비디오 (suspect_data/*.mp4 하나씩)
 ├─ 모션   : HybrIK → SMPL 모션 → DB 250개 탐색 (PSC / AC / DC)
 ├─ 얼굴   : InsightFace 검출 + ArcFace 신원 임베딩 → DB 인물별 비교
 ├─ 아바타 : YOLO11 마스크·관절 → 체형·외형 특징 → DB 인물별 비교
 ├─ 오디오 : 영상에서 음원 분리(ffmpeg) → 화성/리듬/음색/fingerprint 비교   [별도 판정]
 ├─ 판정   : 모달리티별 점수 정규화 → 최댓값 → 도용 의심 / 검토 필요 / 도용 근거 부족
 └─ 설명   : 규칙 기반 설명 생성 → LLM(Qwen2.5-7B-Instruct)이 문장만 다듬음
```

## 1. 설치

요구 사항: Linux, NVIDIA GPU 1장(VRAM 24GB 이상 권장, LLM이 약 15GB 사용), conda, wget, unzip, CPU 멀티코어(모션 DB 탐색용)

GitHub에서 받은 저장소에는 가중치가 없습니다. 아래 두 단계를 거칩니다.

**1) SMPL 모델 넣기 (직접, 1개 파일)**: 라이선스상 재배포가 안 되어 자동으로 받지 않습니다.
중립(neutral) 모델은 따로 전달드리겠습니다. basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 이름입니다. 

```
third_party/HybrIK/model_files/
├─ **basicModel_neutral_lbs_10_207_0_v1.0.0.pkl   <- 직접 넣는 파일**
├─ J_regressor_h36m.npy                          (저장소에 포함)
└─ h36m_mean_beta.npy                            (저장소에 포함)
```

**2) 환경 + 가중치**

```bash
cd copyright_detection
./setup.sh --download              # conda 환경 'copyright' 생성 + 가중치 다운로드(약 16GB) + 설치 점검
./setup.sh --env 이름 --download   # 환경 이름을 바꿀 때. run.sh 가 같은 환경을 쓰도록 .conda_env 에 기록됨
```

- 마지막에 `모든 항목 정상`이 나오면 설치가 끝난 것입니다. `MISSING`이 남으면 표시된 안내대로 해결합니다.
- --> 아마 smpl 안넣고 setup 돌리면 missing 나올겁니다..!
- 중간에 실패하면 원인을 해결하고 같은 명령으로 다시 실행합니다. 이미 받은 파일은 건너뜁니다. 전체 기록은 `setup.log`에 남습니다.

`./run.sh`는 시작할 때 필요한 파일을 먼저 확인합니다. 빠진 파일이 있으면 몇 분씩 돌다 '판정 불가'를 내지 않고, 바로 멈추고 무엇이 없는지 알려 줍니다.

## 2. 실행
요약
0. git clone https://github.com/integerkim/copyright_detection.git
1. cd ~ ./copyright_detection/
2. ./setup.sh --download
3. conda activate copyright
4. ./run.sh

```bash
./run.sh                                    # suspect_data/ 의 영상을 모두 검사
./run.sh --all                              # 위와 같음 (4개 모달리티 모두)
./run.sh --monitor                          # 모션만 검사 (= --skip face avatar audio)
./run.sh a.mp4 b.mp4                        # 특정 영상만 (폴더를 줘도 됨)
./run.sh --no_llm                           # LLM 없이 (규칙 기반 설명만)
./run.sh --skip motion                      # 특정 모달리티 제외
./run.sh --reuse                            # 이미 돌린 영상: config.yaml 의 판정 기준만 바꿨을 때 판정·설명만 다시 계산
```

Python에서 직접 실행하려면 `conda activate copyright`(설치 때 지정한 환경) 후 `python integrated/run_detection.py [--video ...] [옵션]` (옵션 동일).

결과는 `results/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `summary.txt` / `summary.json` | 이번에 검사한 영상 전체를 한 줄씩 정리 (영상, 최종 판정, 점수, 도용 모달리티, 오디오) |
| `<영상 이름>/report.txt` | 사람이 읽는 요약 (판정, 모달리티별 점수, 설명) |
| `<영상 이름>/report.json` | 전체 수치 (모달리티 원점수·세부 점수·증거 점수·판정·설명) |
| `<영상 이름>/motion/` `face/` `avatar/` `audio/` | 모달리티별 원본 결과 (각 모듈이 원래 내던 result.txt / json / csv) |

처리 시간은 5초 영상 1개 기준 약 4~5분입니다(A6000). 대부분은 모션이 차지합니다(HybrIK 약 1분 + DB 탐색 약 2~3분).

#FYI: 출력 프롬프트를 조절하는 프롬프트는 integrated/explain.py에 있습니다.

## 3. 출력 예시 (`results/kdg_face_avatar_copy/report.txt`)

```
최종 판정 : 도용 의심   (도용 모달리티: 얼굴, 아바타)
최종 점수 : 100.0 / 100
패턴     : 얼굴, 아바타 모달리티가 함께 높음
교차 확인 : 얼굴과 아바타가 모두 같은 보호 인물(KDG)을 가리킵니다
오디오    : 검토 필요 (증거 점수 44.0)  [별도 판정]

모달리티     증거   원점수  판정          가장 닮은 DB
모션          0.0    59.1  도용 근거 부족  fakelove / bridge_2 / KHJ
얼굴        100.0   100.0  도용 의심      KDG
아바타       64.2    77.4  도용 의심      KDG
오디오       44.0    58.0  검토 필요      media1_stage.mp4

설명 (LLM: Qwen2.5-7B-Instruct)
최종 판정은 '도용 의심'입니다. 얼굴과 아바타 모달리티의 점수가 높아, 이들은 도용 모달리티로
분류되었습니다(최종 점수 100.0). 얼굴은 DB 인물 'KDG'와 신원 유사도 100.0점으로, ...
```

## 4. 판정 방식

원점수는 모달리티마다 척도가 달라서 그대로 비교하면 안 됩니다. 예를 들어 모션은 관련 없는 영상도 DB 1위가 58~62점이 나오고, 오디오는 58점이면 이미 "검토" 구간입니다. 그래서 다음 순서로 판정합니다.

1. **증거 점수(0~100)로 정규화**: 모달리티마다 기준점을 두고 구간 선형으로 변환합니다.
   - floor → 0점
   - review → 35점 (검토 필요 경계)
   - tau → 50점 (도용 의심 경계)
   - high → 100점
2. **모달리티별 판정**: 증거 50 이상이면 도용 의심, 35 이상이면 검토 필요, 그 아래는 도용 근거 부족
3. **최종 점수**: 영상 모달리티(모션/얼굴/아바타) 증거 점수의 **최댓값**. 한 모달리티만 확실히 베껴도 도용입니다.
   - `config.yaml`의 `fusion.weights`로 모달리티별 가중치를 줄 수 있습니다.
4. **오디오**: 별도로 판정해서 함께 보고합니다. `include_audio_in_final: true`로 바꾸면 최종 판정에도 포함됩니다.
5. **판정 불가**: 사람이 안 보이거나, 얼굴이 없거나, 오디오 트랙이 없으면 해당 모달리티는 "판정 불가"로 표시하고 나머지로 판정합니다.

| 모달리티 | 원점수 | 기준점 (floor / review / tau / high) | 근거 |
|---|---|---|---|
| 모션 | Overall | 60 / 64 / 68 / 80 | 관련 없는 영상의 DB 1위 58~62, 복제 영상 74~77 |
| 모션 | 1위 − 2위 차이* | 1.5 / 2.5 / 4.5 / 10 | 관련 없는 영상 0.4~1.1, 복제 영상 15~17, README 사례 2.8~5.2 |
| 얼굴 | 신원 점수 | 0 / – / 50 / 100 | 점수 자체가 ArcFace 코사인 0.40=50점으로 보정됨 (아래 참고) |
| 아바타 | S_id, S_appear 중 강한 쪽 | S_id 40/–/70/90, S_appear 50/–/75/92 | 아바타 모듈 자체 기준 (신원 70, 외형 75) |
| 오디오 | track_score | 25 / 40 / 70 / 90 | 오디오 모듈 자체 기준 (검토 40, 높음 70) |

\* 모션 증거 점수 = max(Overall 기준, 1위−2위 차이 기준). 2위는 1위와 다른 DB 모션 중 최고점이며, 같은 곡·파트·퍼포머의 중복 파일은 제외합니다. 같은 녹화본을 베낀 경우 1위만 두드러지게 높게 나오는 점을 이용합니다.

모든 기준점은 `config.yaml`에 있고, **데모 데이터로 정한 잠정값**입니다.

## 5. 모달리티별 변경 사항

각 모듈 코드는 최대한 그대로 두고 통합 어댑터(`integrated/modalities.py`)에서 호출합니다.

- **모션** (`motion_infringement/`): 하드코딩된 HybrIK 경로를 `third_party/HybrIK`로 바꿨고, `result.json` 출력을 추가했습니다. 지표와 탐색 로직은 그대로입니다.
  - torch 2.4 환경에서 재현을 확인했습니다: 예시 영상 Overall 76.62 / 2위 59.31 (README: 76.45 / 59.45).
- **얼굴** (`face_infringement/face_identity.py`, 신규): 판정 근거를 **ArcFace 신원 임베딩**으로 바꿨습니다.
  - 기존 기하 특징 방식(`face_geometry.py`, dlib 68점)도 KDG 원본 vs 생성 영상에서는 0.862로 "도용 의심"을 냅니다.
  - 하지만 sigma/tau가 예시값이라(코드 자체 경고) 판정 근거가 없고, 5개 비율 특징의 사람 간 구분력을 보정할 데이터도 없습니다.
  - ArcFace는 신원 구분용으로 대규모 학습된 모델이라 적은 데이터로도 경계가 분명합니다(아래 보정 결과).
  - 세부 지표: 신원 유사도, 정면/측면 얼굴 유사도, 프레임 일관성
  - 보정 결과 (`face_infringement/calibration.txt`): 데모 영상 13개(7명) 기준으로 다른 사람 쌍 61개는 cos 최대 0.276, 같은 사람 쌍 12개(다른 카메라 시점 포함)는 최소 0.777, 생성 영상 vs 원본은 0.62~0.77
  - 임계값: cos 0.40 = 50점
  - 기존 `face_geometry.py`는 그대로 남아 있고 이 환경에서도 실행됩니다.
- **아바타** (`avatar_infringement/demo/`): YOLO 가중치 경로를 `weights/yolo`로 바꿨습니다. `copyright_database/avatar/` 아래 폴더 하나를 인물 한 명으로 보고 모두 비교합니다.
- **DB 위치**: 각 모듈 안에 있던 보호 데이터(모션 npz 250개, 얼굴 KDG/SSH/YHS, 아바타 KDG, 보호 음원)를 모두 `copyright_database/`로 옮겼습니다. 각 모듈을 단독으로 실행할 때의 기본 경로도 여기를 가리킵니다.
- **오디오** (`audio_infringement/`): 입력 영상에 오디오 트랙이 있으면 ffmpeg로 분리(44.1kHz 스테레오 WAV)해서 비교합니다. 트랙이 없거나 거의 무음(RMS −45dB 미만)이면 "판정 불가"로 처리합니다. 점수 계산은 그대로입니다.
- **LLM**: 규칙 기반 설명(정확한 사실)을 먼저 만들고, LLM은 그 문장을 다듬기만 합니다.
  - 출력에 원문에 없는 숫자가 있거나, 판정 문구가 빠졌거나, 한자가 섞이면 버리고 규칙 기반 설명을 씁니다(최대 3회 재시도).
  - Qwen2.5-1.5B는 숫자는 맞추지만 의미를 뒤집는 문장을 만든 사례가 있어 7B를 씁니다.

## 6. 보호 DB 추가 / 교체 (`copyright_database/`)

파일을 해당 폴더에 넣기만 하면 다음 실행부터 반영됩니다. 자세한 규칙은 [copyright_database/README.md](copyright_database/README.md)에 있습니다.

| 모달리티 | 위치 | 넣는 것 |
|---|---|---|
| 모션 | `copyright_database/motion/` | SMPL npz (`poses_axis_angle [T,24,3]`, `trans [T,3]`, 30fps) **또는 안무 영상**. 영상은 첫 실행 때 HybrIK로 npz 변환 후 캐시. 파일명 `곡_파트_번호_퍼포머` 형식이면 결과에 곡/파트/퍼포머로 표시 |
| 얼굴 | `copyright_database/face/<인물>/` | 그 인물의 영상 또는 사진 (여러 개 가능, 다양한 각도 권장) |
| 아바타 | `copyright_database/avatar/<인물>/` | 전신 영상 (영상 하나 = 시점 하나). `cam00...` 이름이 있으면 그 시점끼리 비교 |
| 오디오 | `copyright_database/audio/` | 보호 음원 또는 음악이 들어간 영상 |

DB 특징은 `cache/db/<모달리티>/`에 저장되고, 파일이 바뀌면 자동으로 다시 계산합니다. `cache/`는 지워도 됩니다.

DB를 늘리면 다른 사람·다른 동작 간 점수 분포가 바뀌므로 기준점을 다시 확인해야 합니다. 얼굴은 `python face_infringement/calibrate_identity.py`로 DB 인물끼리의 유사도 분포를 확인할 수 있습니다(같은 인물 최솟값과 다른 인물 최댓값 사이에 기준 0.40이 있어야 함).

## 7. 테스트 결과 (`suspect_data/`, 기존 데이터로 구성)

| 영상 | 내용 | 모션 | 얼굴 | 아바타 | 최종 | 오디오 (별도) |
|---|---|---|---|---|---|---|
| `kdg_face_avatar_copy` | KDG 원본으로 만든 생성 영상 (`kdg_vid_df`) | 59.1 → 근거 부족 | **KDG 100** | **KDG 77.4 → 의심** | **도용 의심** (얼굴·아바타) | 58.0 → 검토 필요 |
| `motion_copy` | DB 모션(cam00)을 다른 인물에 입힌 영상 | **76.6 → 의심** | 0 | 37.3 | **도용 의심** (모션) | 오디오 없음 |
| `motion_copy_mirror` | 위 영상 좌우 반전 | **74.3 → 의심** (반전 일치) | 0 | 36.4 | **도용 의심** (모션) | 오디오 없음 |
| `motion_copy_with_music` | `motion_copy` + 보호 음원 tempo 1.15배 | **76.5 → 의심** | 0 | 37.3 | **도용 의심** (모션) | **84.8 → 도용 의심** |
| `ssh_real_cam00` | DB 인물 SSH의 실사 영상 | 62.0 → 근거 부족 | **SSH 100** | 46.5 | **도용 의심** (얼굴) | 오디오 없음 |
| `unrelated_dance` | 무관한 댄스 영상 (HybrIK 예제) | 58.3 | 0 | 12.3 | 도용 근거 부족 | 오디오 없음 |

`ssh_real_cam00`의 얼굴 판정은 SSH가 얼굴 DB에 등록된 보호 인물이라서 나오는 정상 동작입니다(실제 인물 영상 = 해당 인물 얼굴 사용).

## 8. 한계

- **기준점이 잠정값입니다.** 인물 7명, 모션 DB 250개, 보호 음원 1개, 테스트 영상 6개로 정했습니다. 오탐률(FAR)·미탐률을 말할 수 있는 평가셋이 아닙니다.
- 모션은 "같은 녹화본"을 잘 찾고 "같은 안무를 다른 사람이 춘 것"은 잘 못 찾습니다(원 모듈 README와 같음).
- 아바타는 입력과 같은 카메라 시점이 DB에 있어야 신원 점수가 안정적입니다.
- 오디오 데모 사례(`kdg_face_avatar_copy`, 58점)는 리듬·음색만 높고 화성·fingerprint가 거의 0이라서, 원 모듈 기준대로 "검토 필요"로 둡니다.
- 점수는 기술적 유사도 지표이며 법적 저작권 침해를 확정하지 않습니다.

## 9. 포함된 외부 구성요소와 라이선스 (배포 전 확인 필요)

| 구성요소 | 위치 | 라이선스 |
|---|---|---|
| HybrIK 코드 | `third_party/HybrIK` (commit `HybrIK_COMMIT.txt`) | MIT |
| SMPL 모델 파일 (SMPL-X는 미사용) | `third_party/HybrIK/model_files` | **MPI 비상업 연구용** (상업 이용 시 별도 라이선스) |
| HybrIK 체크포인트 | `third_party/HybrIK/pretrained_models` | HybrIK 배포본 (SMPL 조건 따름) |
| YOLO11 (ultralytics) | `weights/yolo` | **AGPL-3.0** (상업 비공개 이용 시 Enterprise 라이선스) |
| InsightFace buffalo_l | `weights/insightface` | 코드 MIT, **모델 가중치는 비상업 연구용** |
| Qwen2.5-7B-Instruct | `weights/llm` | Apache-2.0 |
| torchvision Faster R-CNN | `weights/torch` | BSD-3 |

상업적으로 이용한다면 SMPL, YOLO11, InsightFace 가중치의 라이선스를 먼저 해결해야 합니다.

## 10. 폴더 구조

```
copyright_detection/
├─ run.sh / setup.sh / config.yaml / environment.yml / environment.lock.yml / README_RUN.md
├─ copyright_database/    보호 DB: motion/ (250개) · face/<인물>/ · avatar/<인물>/ · audio/
├─ suspect_data/          검사할 영상 (지금은 테스트 영상 6개)
├─ results/               검사 결과: summary.txt + <영상 이름>/
├─ cache/                 DB 특징 캐시 (자동 생성, 지워도 됨)
├─ integrated/            통합 파이프라인 (run_detection.py, modalities.py, fusion.py, explain.py, check_setup.py)
├─ motion_infringement/   모션 모듈 코드
├─ face_infringement/     얼굴 모듈 코드 (face_identity.py 신규, face_geometry.py 기존)
├─ avatar_infringement/   아바타 모듈 코드
├─ audio_infringement/    오디오 모듈 코드
├─ third_party/HybrIK/    HybrIK 코드 (SMPL·체크포인트는 설치 때 추가)
└─ weights/               yolo / insightface / torch / llm
```
