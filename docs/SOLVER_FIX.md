# 500개 입자 solver 수정 및 검증

현재 유체력·접촉 모델에 맞춰 접촉력을 독립 변수로 유지하는 proximal semismooth Newton + right FGMRES로 변경했다. 기존 velocity-only 풀이에서는 강한 접선 스프링의 좁은 sticking 구간 때문에 작은 보폭이 반복되었다. 결합 Schur 전처리기만 바꾸는 것으로는 이 외부 비선형 실패가 해결되지 않았다.

새 방식은 원래 normal penalty·탄성/Coulomb 접촉법칙과 같은 해를 갖는다. 최종 수렴은 원래 접촉법칙으로 다시 계산한 힘·mobility 잔차로 검사한다. 1 nm 유체 간격 floor, 근·중거리 모델, 접촉·마찰 물성과 시간간격은 바꾸지 않았다. 중거리 mode 합의 중복 연산 제거와 병렬화도 포함했다.

## 확인한 실행

로그의 N=500, phi=0.24, seed=1729, 무작위 초기화, dt=1e-6 s, 전단률 100 s^-1를 사용했다. 기존 로그와 packing 진행 기록은 일치한다. 원본 입자 좌표 파일은 업로드 로그에 없으므로 사용자 실행과 좌표 파일이 byte 단위로 동일하다는 주장은 하지 않는다. 아래 네 검증은 보관된 고정 초기 좌표로 재현 가능하다.

8 OpenMP threads, 단일 LAMMPS rank의 로컬 측정이다. 시간 열은 geometry·mobility 준비·solve를 포함한 `profile_total_s`이다. 초기화(step 0)와 이동 후 두 단계(step 1,2)를 나란히 기록했다.

| AR | Newton 0/1/2 | Krylov 0/1/2 | 전체 계산시간 0/1/2 (s) | 최대 상대 물리 잔차 |
| --- | --- | --- | --- | --- |
| 2 | 9 / 8 / 8 | 61 / 53 / 57 | 18.67 / 15.61 / 16.26 | 1.92e-07 |
| 5 | 10 / 8 / 8 | 64 / 65 / 60 | 8.85 / 8.73 / 8.00 | 1.83e-07 |
| 10 | 11 / 9 / 7 | 74 / 69 / 62 | 8.72 / 7.36 / 6.22 | 2.81e-07 |
| 20 | 10 / 7 / 7 | 83 / 76 / 65 | 5.47 / 4.67 / 3.96 | 1.76e-07 |

네 경우 모두 step 2의 final.data를 출력했고, 위치와 배향이 실제로 변했다. 최대 상대 물리 잔차는 2.81e-7로 설정 허용오차 1e-6 이내이다. 힘·토크·stresslet을 포함한 trajectory 값은 유한하다. 소산 비교의 최대 상대 차이는 8.90e-9이며, 이는 동일 모델 내부의 일관성 검사다.

전체 빌드에서 CTest 31/31과 플러그인 로드 검사가 통과했다. 독립 해석해, contact Jacobian·Schur 축약, 원래 방정식의 최종 잔차, history 보존 및 warm start를 검증한다.

## 실제 실행기 경로

`run_bridge.py`로 별도 N=500, AR=10 입력을 준비하고 10 step을 완료했다. private runtime의 라이브러리 격리 검사를 거쳤으며, 종료 뒤 10개 runtime 파일 기록도 재검증했다. 이 경로의 최대 상대 잔차는 8.850e-07, 물리적 power로 정규화한 최대 power balance 차이는 3.879e-09다.

초기 계산은 9.14 s, 마지막 세 단계는 4.67 / 3.72 / 4.04 s였다. 이 시간은 Negishi에서의 성능 보장이 아니며 긴 정상상태 실행의 평균 비용도 아니다. 정상상태 점도와 다중 MPI rank는 이번 검증 범위에 포함하지 않는다.

step 0은 평균을 누적한 시간이 0이므로 평균 열만 NaN이고, 순간 응력·힘·소산·잔차는 유한하다. `prox_*` trace의 `residual`은 원래 물리 방정식의 잔차, `merit`는 line search가 감소시키는 확장 잔차이다. `linear_residual`이 해당 선형계의 실제 잔차이다. 구형 경로용 `linear_defect`, `yield_distance` 필드는 proximal 경로에서 계산하지 않으므로 그 기본값을 정확도 지표로 해석하지 않는다.

원시 결과는 `validation_solver500/final/`, 실제 실행기 10단계 결과는 `validation_solver500/approved_runner_ar10/`, 빌드 기록은 `validation_solver500/provenance/`에 있다. AR 2·10은 이전에 수렴한 동일 물리 모델 결과와도 비교했다. 비교 수치는 `comparisons/`에 있으며 허용오차를 낮춰 결과를 맞추지 않았다.

실행 명령과 출력 설명은 [BRIDGE_RUN.md](BRIDGE_RUN.md), 삭제 가능한 폴더는 [LEGACY_CLEANUP.md](LEGACY_CLEANUP.md)를 따른다. 이번 수정에는 기존 build/runs 폴더 삭제가 필요 없다.
