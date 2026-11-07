# Avatar 모달리티: 체형·외형 기반 도용 탐지

입력 영상의 사람을 보호 DB(`copyright_database/avatar/<인물>/`)의 촬영 영상과 비교해 점수(0~100)를 냅니다.

- S_id: 신원 단서 (머리 모양·덩치·맨살 폭)
- S_appear: 외형 (옷 색·질감·형태)
- S_avatar: 전체 도용 의심도
- 일치한 DB 시점, 판정, 특징별 근거 (일치 / 유사 / 차이)

통합 실행(`./run.sh`)에서 자동으로 호출됩니다. 환경은 루트의 `environment.yml`(conda `copyright`) 하나를 쓰고,
YOLO11 가중치는 `weights/yolo/`에 있습니다.

## 파일

```
avatar_infringement/demo/avatar/
├─ preprocess/avatar_preprocess.py   # 영상 → 사람 마스크·2D 관절 → 외형/신원 특징 추출 (JSON)
└─ code/
    ├─ avatar_main.py                # 진입점 (특징 불러오기 → DB 시점별 비교 → result.txt)
    ├─ avatar_metric.py              # 외형 특징 정의, 유사도, 점수, 판정
    ├─ avatar_id.py                  # 신원 단서 특징 (머리 모양 / 맨살 팔다리 폭 / 덩치), Fisher 가중치
    └─ avatar_body.py                # [참고] 골격 비율 불변량 (판정 미반영)
```

DB 특징은 `cache/db/avatar/<인물>/`에 저장되고, DB 영상이 바뀌면 자동으로 다시 추출합니다.

## 단독 실행 (선택)

```bash
conda activate copyright
cd avatar_infringement/demo/avatar/code
python avatar_main.py --device 0                     # suspect_data/ 의 mp4 vs copyright_database/avatar/KDG (cam00 시점끼리)
python avatar_main.py --input a.mp4 --db_dir ../../../../copyright_database/avatar/SSH --db_name SSH --device 0
python avatar_main.py --db_view auto                 # 점수가 가장 높은 DB 시점과 비교
python avatar_main.py --self_check                   # DB 카메라 1대씩 빼고 '같은 사람' 기준 점수 확인
```

단독 실행 결과는 `results/avatar_standalone/`, 특징은 `cache/avatar_standalone/`에 저장됩니다.

## 판정 기준 (잠정)

신원 70점, 외형 75점입니다.

| 신원 | 외형 | 판정 |
|---|---|---|
| 넘음 | 넘음 | DB 도용 의심 |
| 넘음 | 못 넘음 | ID 도용 의심 (Face로 교차 확인) |
| 못 넘음 | 넘음 | 외형 모방 의심 |
| 못 넘음 | 못 넘음 | 도용 아님 |

## 방법 요약

1. 프레임마다 YOLO11-seg로 사람 마스크, YOLO11-pose로 COCO 17관절을 얻습니다. 배경과 얼굴 영역은 비교에서 뺍니다.
2. 전신이 보이고 직립인 프레임만 쓰고, 몸 방향(정면/비스듬/측면)을 판정합니다.
3. 특징을 추출합니다.
   - 외형: 상의·하의·피부·머리카락 색(Lab), 상의·하의 질감(LBP), 소매·하의 길이
   - 신원 단서: 머리 폭/어깨 폭, 앞머리 길이, 옆머리 끝, 머리카락 면적, 목/머리 폭, 엉덩이/어깨 폭, 손목/아래팔 폭
4. 특징별로 z = |입력 − DB| / σ, s = exp(−z²/2)를 계산합니다. z ≤ 1이면 일치, z ≤ 2면 유사, 그보다 크면 차이로 표시합니다.
5. 신원 단서는 Fisher 비율(σ²between / σ²within)로 가중 평균합니다. 5명·21개 영상으로 추정했고, 1 미만인 특징은 가중치 0입니다.
6. DB 카메라 영상마다 프로필을 따로 만들고, **입력과 같은 카메라 시점(기본 cam00)끼리** 비교합니다. 시점이 다르면 원근 때문에 머리 모양과 폭 비율이 달라지기 때문입니다.

## 개발 당시 검증 기록

데모 영상은 배포본에서 뺐고, 수치만 남깁니다.

| 입력 | 일치 DB 시점 | S_id | S_appear | S_avatar | 판정 |
|---|---|---|---|---|---|
| Seedance 변환 영상 (KDG cam00을 얼굴만 바꿔 생성) | cam00 | 79.6 | 78.8 | **79.2** | **DB 도용 의심** |
| SSH cam00 실사 (다른 사람) | cam00 | 27.5 | 64.9 | **46.2** | **도용 아님** |

같은 방식으로 다른 사람의 실사 10개(SSH·YHS 각 5대, 같은 시점끼리)를 비교하면 신원 점수 최고값이 64.3이고,
DB 도용이나 ID 도용으로 나온 건 없었습니다.

## 한계

- 입력과 같은 시점이 DB에 없으면 같은 사람이어도 신원 점수가 떨어집니다(개발 당시 DB 카메라 1대씩 빼고 비교했을 때 27~57). 실제 생성 영상은 시점을 모르니, DB에 스튜디오 시점을 모두 넣고 입력 시점을 추정하는 단계가 필요합니다.
- 가중치와 임계값은 5명 데이터 기준 잠정값입니다.
- 머리 모양은 모자나 헤어스타일 변경에 약하고, 맨살 폭 특징은 반팔·반바지 촬영 데이터가 있어야 측정됩니다.
