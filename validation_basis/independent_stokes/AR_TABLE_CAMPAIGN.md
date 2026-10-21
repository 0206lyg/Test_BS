# AR2·5·10·20 표 생성과 독립 정확도 평가

사용자가 실행할 작업은 **빌드 배치 한 번과 계산 제출 명령 한 번**이다.
입력 방향·간격 목록, 고립 검증, 독립 Stokes 계산, 표 생성, 별도 정확도
비교를 자동으로 연결한다. 개별 행렬이나 JSON을 손으로 옮길 필요가 없다.

## Negishi에서 실행

새 ZIP은 최상위 폴더를 추가하지 않는 전체 소스 묶음이다. 파일을 기존
`/home/lee6456/slurry`에 덮어 푼 뒤 그 디렉터리에서 다음을 실행한다.
기존 결과 디렉터리와 외부 dependency를 ZIP 안에 넣지 않았다.

```bash
cd /home/lee6456/slurry
sbatch build_reference_tables_cpu.sbatch
```

빌드 로그 마지막에 `READY /scratch/negishi/lee6456/slurry_reference_build/build_reference_manifest.json`
이 나온 뒤 다음 명령을 실행한다.

```bash
bash submit_reference_tables.sh --output-dir /scratch/negishi/lee6456/reference_tables_AR2_5_10_20
```

이 명령이 네 AR의 작업 배열과 마지막 집계 작업을 함께 제출한다. 기본
동시 작업은4개, 작업당8 CPU·32 GiB·24시간이다. 실제 풀이의 메모리
예산은28 GB이며 자원 부족을 정확도 실패로 처리하지 않는다. 계산 시간은
실측하지 않았으며 아래의 작은 구현 검사에서 전체 시간을 추정하지 않는다.
Slurm 로그는 제출 디렉터리에, 행렬·checkpoint·표·보고서는 지정한 scratch
결과 디렉터리에 저장한다. 전체 계산에는 큰 작업 디스크가 필요하다.

빌드에는 기존 프로젝트와 같은 `anaconda/2024.02-py311`, `gcc/12.2.0`,
`cmake/3.23.1` 모듈을 사용한다. NumPy·SciPy·Matplotlib과 GNU Fortran,
BLAS/LAPACK이 필요하며 pinned fmm3dbie/FMM3D를 내려받아 별도 위치에서
빌드한다. 확인된 quadrature 수정도 별도로 적용한다. LAMMPS 빌드나 입자
packing은 이 표 생성 작업에 필요하지 않다.

제출 없이 명령·작업 수만 확인하려면 같은 제출 명령에 `--dry-run`을
추가한다. 이 옵션도 실행 계획 파일은 생성하지만 Slurm에 제출하지 않는다.

## 기본 계산 내용

| 항목 | 기본 설정 |
| --- | --- |
| AR | 2,5,10,20 |
| polar semiaxis c / 두께 | 150 nm /300 nm |
| equatorial semiaxis a | AR×150 nm |
| hydrodynamic 간격 | 실제 매끈한 표면 간격;1 nm 아래만 제한 |
| 표의 최대 간격·anchor | h/c=1, 즉150 nm인 명시적 시험 설정 |
| AR별 표 생성 방향 | theta4×theta4×phi8=128 |
| 표의 간격점 | 초기129개, 필요한 구간만 적응 세분; 기본 상한1024 |
| AR별 별도 평가 지점 | 24개 방향·간격 조합 |
| 평가용 기준해 | 각 지점에서 p6/p8 두 해상도 |
| pair 작업 수 | AR별176개, 네 AR 합계704개 |
| 고립 입자 검사 | 각 방향에서 두 입자; 간격이 달라도 같은 검사를 재사용 |

150 nm는 검증된 보편적 전환거리가 아니다. AR별 정확도를 비교하기 위한
초기 계산 범위다. 필요하면 `--anchor-over-c`를 바꿔 새 campaign을 만들거나
`--ars 20`처럼 특정 AR만 다른 범위에서 계산한다. 실제 물리적 유효 범위는
결과로 판단한다. 3 nm의 기계적 roughness는 hydrodynamic 간격에서 빼지 않는다.

네 개의 별도 방향에서 로그 분포의5개 간격을 검사하고, 표 격자에 속한
방향 하나에서 anchor를 제외한4개 간격도 검사한다. 따라서 방향 보간과
간격 연결을 함께 검사하는 지점, 방향 보간만 검사하는 anchor 지점,
간격 연결을 검사하는 지점이 포함된다. 학습에 사용한 동일 지점을 별도
검증이라고 세지 않으며, pole·director 부호로 동등한 방향도 구분한다.

기본 pair 격자는 level1·rim1·contact1이며 표 입력 p8, 별도 비교 p6/p8이다.
완료한 pair의 표본 경계 오차가 개발 ceiling1%보다 크면 같은 조건에서
최대2회 residual 기반 패널 세분을 시도한다. 자원·선형 풀이 실패 때는
더 큰 계산을 자동 실행하지 않는다. 모든 시도와 실패 원인을 보존한다.

## 결과에서 볼 파일

`/scratch/negishi/lee6456/reference_tables_AR2_5_10_20/` 아래에 생성된다.

| 파일 | 의미 |
| --- | --- |
| `campaign_summary.json` | AR별 표 생성 상태, PASS/FAIL/UNRESOLVED 개수, 실패 작업 |
| `accuracy/accuracy.csv` | AR·방향·간격별 힘·토크·stresslet·에너지 오차와 기준해 변화 |
| `accuracy/accuracy.png` | AR별 gap에 따른 오차와 두 해상도의 변화 |
| `accuracy/accuracy.json` | 출처, 비교 조건, 판정 근거를 포함한 상세 결과 |
| `AR20/table_attempts/attempt_0001/reference.sft` 등 | 성공한 AR의153성분 전체 방향 표 |
| 각 표 옆 `pipeline.json` | 실제 입력·표·실행 파일·수치 소스의 식별 기록 |
| `tasks/task_000000/task.json` 등 | 개별 기준해의 선택 결과, 해시, 재시도 기록 |

완료된 표만으로 정확도 통과를 선언하지 않는다. 별도 Stokes 결과와
직접 비교한 full17 상대 에너지 오차 및 힘·토크·stresslet 행 블록의
상대 오차를 함께 사용한다. 작은 고유값 방향의 오차도 에너지 검사로 본다.

| 판정 | 조건과 의미 |
| --- | --- |
| PASS | 독립성과 두 해상도 검사가 충족된 표본에서 모델 오차가 선택한 허용치 이하 |
| FAIL | 기준해 비교가 가능한 표본에서 표 응답의 오차가 선택한 허용치를 초과 |
| UNRESOLVED | 기준해 미완료·경계 검사 실패·큰 해상도 변화·출처/조건 불일치로 판단 보류 |

기본 모델 허용치는5%, 기준해의 두 해상도 변화 허용치는1%, 표본 경계
잔차 ceiling은1%다. `--model-tolerance`, `--refinement-tolerance`,
`--boundary-limit`으로 명시적으로 정한다. 이 값들은 개발용 판정 설정이며
수학적 오차 상한이나 점도 오차 허용치가 아니다. 기존0.1% 기준과 세 격자
정밀 검증은 별도 진단으로 남고, 이번 평가를 막는 조건으로 사용하지 않는다.
PASS는 검사한 표본에 대한 잠정 판정이다. 모든 방향·간격이나 bulk 점도를
검증했다는 뜻은 아니므로 `physical_accuracy_qualified=false`를 유지한다.

모든 표 입력 방향이 확보되어야 그 AR의 표를 생성한다. 입력 한 곳이
실패하면 그 AR의 표는 미완료로 표시하고 다른 AR 처리는 계속한다.
마지막 집계는 작업 실패가 있어도 실행하여 빠진 기준해도 CSV에
UNRESOLVED로 남긴다. 가짜 방향이나 대체 행렬을 채우지 않는다.

## 재실행과 격자 변경

중단 후 같은 계획을 다시 제출한다.

```bash
bash submit_reference_tables.sh --campaign /scratch/negishi/lee6456/reference_tables_AR2_5_10_20/campaign.json
```

완료된 검사·pair 결과는 task/config/source/checkpoint 해시가 같으면 재사용한다.
미완료 시도의 파일은 보존하고 새 attempt에서 해당 단계를 계산한다.
선형 풀이 중간의 분해 상태를 이어서 실행하는 기능은 아니다. 기존 해의
경계 검사 중 중단됐다면 현재 worker는 그 pair 풀이를 다시 실행할 수 있다.

방향 보간 오차가 크면 새 출력 디렉터리에 방향 격자를 늘린 계획을 만든다.
다음은 실행 가능한 예이며 정확도 향상을 보장하는 설정은 아니다.

```bash
bash submit_reference_tables.sh --ars 2,5,10,20 --theta-nodes 6 --phi-nodes 12 --output-dir /scratch/negishi/lee6456/reference_tables_AR2_5_10_20_angular_refined
```

기준해 해상도 문제와 연결식 문제는 구분한다. 기준해가 안정됐는데 FAIL이면
임의의 수치 보정으로 통과시키지 않고 연결식·방향 보간을 비교한다.1 nm
기준해가 미완료이면 그 범위의 정확도는 UNRESOLVED이며, 먼저 완료된
간격의 비교 결과는 그대로 확인할 수 있다.

## 현재 모델과 이번 구현 검증

현재 모델은 `JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1`이다. 기존 상수
잔차 연결식의 중간 간격 음의 소산 문제를 피하기 위한 후보이며, 그 유한
간격 정확도가 확정된 것은 아니다. 이번 campaign은 바로 그 정확도를
AR별·간격별로 판단할 자료를 만든다. 설명은 [PROVISIONAL_PIPELINE.md](PROVISIONAL_PIPELINE.md)에 있다.

여기서는 네 AR 각각에서 작은 실제 고립2회·pair1회, 총12회로 실행 연결을
검사했다. 이 격자는 의도적으로 작아서 정확도 검사를 통과하지 않았고,
그 사실을 completed_unaccepted로 기록했다. 네 AR의 실제 전체 표 계산은
아직 실행하지 않았다. 합성 표 조회·회전·1 nm 하한 검사와 Python104개 중
103개 통과(선택적 성능 검사1개 생략)는 구현 검사이며 물리 정확도와 별개다.

현재 표 생성기는512 MiB payload 한도를 유지한다. 많은 방향과 간격점이
필요해 이를 넘으면 원인을 기록하고 중단한다. 큰 표를 조용히 잘라서
저장하지 않는다.
