# Test_BS — AR별 Stokes 기저·저항표 생성 및 실행

Release: `20261018-ar-basis-pipeline-trial`

요청한 세 작업을 연결한 개발본입니다. 입력한 AR의 spheroid 형상에 대해 Stokes 기저를
실제로 계산하고, 두 입자 저항표를 생성·저장한 뒤 implicit dynamics와 전단응력·점도에
사용합니다. AR10 계수의 이름이나 길이만 바꾸는 방식이 아닙니다. 독립 두 입자 검증은
현재 AR10에서 수행했습니다. 다른 AR에는 독립 검증 미실시 상태가 실행 기록에 남습니다.

## 설치와 실행

ZIP은 기존 slurry 루트에 풉니다. 새 실행 파일이 필요하므로 다시 빌드합니다.
LAMMPS 전체 소스·빌드 산출물·대형 결과 dump는 ZIP에 포함하지 않았습니다.

```bash
cd /home/lee6456/slurry
unzip -o Test_BS_AR_basis_pipeline.zip
rm -f lammps_slurry/src/slurry_fast_residual.h
SLURRY_ROOT="$PWD" sbatch build_slurry_cpu.sbatch
```

위에서 지우는 파일은 더 이상 참조하지 않는 구형 residual-table 헤더입니다.
빌드 완료 및 plugin load 확인 후 실행합니다. **새 모델은 반드시
`--lubrication-model finite_gap_fast`로 선택합니다.** 기존 `auto`는 기존 경로를 유지합니다.

아래 명령은 포함된 새 AR10 기저로 정밀 저항표를 생성하고, 검사를 거쳐 짧게 실행합니다.
최초 정밀 표 생성은 계산량을 외삽하면 8 thread에서도 수일 규모가 될 수 있습니다.
이는 전체 표의 실측 시간이 아닙니다. 시간 제한에 걸려도 같은 명령을 다시 제출하면
완료한 계산 조각을 재사용합니다. 아래 24시간은 한 작업의 할당 예시이며 완료 시간
보장이 아닙니다. 완성된 표는 이후 실행에서 재사용합니다.

```bash
SLURRY_ROOT="$PWD" sbatch --cpus-per-task=8 --mem=16G --time=24:00:00 \
  lammps_slurry/run_slurry_cpu.sbatch \
  --ar 10 --phi 0.24 --count 100 \
  --lubrication-model finite_gap_fast \
  --finite-gap-fast-source "$PWD/lammps_slurry/models/ar10_regenerated.ssb" \
  --finite-gap-fast-threads 8 \
  --steps 20 --prep-steps 0 --output-interval 1
```

이 짧은 실행은 정상상태 점도 측정용이 아닙니다. 연결만 먼저 확인하려면 위 명령에
`--finite-gap-fast-smoke`를 추가합니다. 이 옵션은 거친 별도 표를 만들며 **정량 점도
비교에 사용하면 안 됩니다.** 거친 표를 정밀 표로 자동 승격하지 않습니다.

다른 AR에서는 `--ar`와 `--finite-gap-fast-source`의 파일명을 함께 바꿉니다.
예를 들어 AR5에 존재하지 않는 `models/ar5_regenerated.ssb` 경로를 지정하면 AR5 기저부터
새로 계산합니다. 기존 AR10 파일을 AR5에 지정하면 오류로 중단합니다.
AR1은 별도의 구형 모델 `--ar 1 --lubrication-model mari`를 사용합니다.
대상은 같은 크기·같은 AR의 oblate spheroid이며, 혼합 AR·다분산·prolate 모델은 아닙니다.

전체 옵션: `bash lammps_slurry/run_slurry_cpu.sbatch --help`.
표는 기저 옆 `.fast_cache/`에 저장합니다. 기저 내용, 수치 소스, 생성 실행 파일,
AR·격자·간격 설정이 일치할 때 재사용합니다. 시간 제한 후 재개할 때도 이 조건을
유지해야 합니다. 검사 실패는 원인을 출력하고 중단하며, 다른 모델로 대체하지 않습니다.

## 이번 변경

- Eigen만 사용하는 `slurry_generate_basis`를 추가했습니다. 단일입자 표면의 Stokes
  경계조건을 Fourier 모드별로 풀어 병진·회전·strain 응답과 힘·토크·stresslet을 생성합니다.
  벌크 점도나 두 입자 기준값을 fitting하지 않습니다.
- 새 AR10 기저는 차수 12, source ring 96개입니다. V3 모델은 유효 유체 간격
  `h_eff/c >= 0.5`에서 해당 간격의 Stokes 저항을 직접 계산해 표에 넣습니다.
  그보다 좁은 간격에서는 기존 film 저항과 `h_eff/c=0.5, 1`의 잔여 저항을 연결합니다.
  유한 간격을 두 기준점으로 연결하던 이전 식은 225 nm 검사에서 6.49% 오차로
  5% 기준을 넘었으므로 교체했습니다. 기저 파일의 구형 외측 차수 8 슬롯은 형식
  호환용으로만 보존되며 V3 저항 계산에는 사용하지 않습니다.
- 저항표 형식은 V2를 유지하며 `h_eff/c=0.5, 1`을 명시합니다. 행렬 로그를 보간하고 지수로 복원하여 전체
  두 입자 저항의 양정성을 유지합니다. 보정항을 임의로 자르거나 음의 고유값을 강제로
  양수로 바꾸지 않습니다. 다체 전체 소산은 실행 중 별도로 검사합니다.
- 표 생성은 OpenMP 병렬화, 완료 조각 저장, 재개, 손상·조건 불일치 검사를 지원합니다.
  완성되지 않은 표를 runtime에서 읽지 않습니다.
- 실제 적분에서는 표를 조회하고 같은 저항 행렬을 힘·토크·응력·전력에 사용합니다.
  매 timestep마다 Stokes 경계값 문제나 film 격자를 다시 풀지 않습니다.
- 거의 평행한 입자에서 상쇄 오차로 회전 좌표계 검사가 실패하던 문제를 수정했습니다.
  입력 형상과 기존 검사 허용오차를 유지하고 직교 좌표계 구성만 바로잡았습니다.

## 확인한 범위

오차는 6개 상대운동과 5개 strain 성분의 조합에 대한 최대 에너지 오차입니다.
아래는 실제 연결 모델을 독립 계산과 비교한 배치별 결과이며, 전체 배치의 오차 상한은 아닙니다.

| AR10 배치 | 간격 | 전체 11성분 오차 |
|---|---:|---:|
| 옆으로 어긋난 평행 입자 | 75 nm | 0.190% |
| 10도 기울어진 입자 | 150 nm | 0.444% |
| 별도 평행·엇갈림 배치 | 112.5 nm | 0.0802% |

112.5 nm 독립 기준은 runtime 기저와 film 식을 사용하지 않는 두 입자 Stokes 계산입니다.
기준 계산의 모든 17성분 조합에서 경계조건 오차 0.0532%, 해상도 증가에 따른 11성분
저항 변화 0.000468%를 확인했습니다. 실제 모델의 힘·토크 자유 조건에서 strain 응답
오차는 0.00363%입니다. 같은 계열의 고차 기저를 사용한 별도 수치 검사에서는
37.5 nm의 오차가 3.249%, 225 nm의 오차가 0.0162%였습니다. 두 기준 계산 모두
차수와 적분 해상도의 수렴을 확인했습니다. 이전 연결식이 225 nm에서 보인
6.49% 오차와 실패 기록도 검증 자료에 보존했습니다.

3 nm의 정밀 보간 오차는 같은 모델의 직접 계산 대비 0.0562%였습니다. V3에서는
이 구간의 식·기저·보간 절점과 가중치가 같고, 같은 배치의 저항 행렬도 정확히 같은 것을
확인해 기존 보간 결과를 이어 사용했습니다. 3,072개 절점 계산을 새로 반복한 결과는
아닙니다. 이 검사는 해당 배치의 보간 검사이며 독립적인 Stokes 정확도 검사는 아닙니다.
실제 LAMMPS에서 2입자·연결된
8입자 모두 dt=10⁻⁶ s, 전단률 100/s로 20 step을 통과했습니다. 수정되지 않은 핵심 계산의
CTest 4개 통과 기록도 포함했습니다.
최종 검사 결과와 미완료 항목은 `RESULT.json`에 정리합니다.

AR10 전체 정밀 표 생성과 모든 방향의 검증, 다른 AR, 정상상태 벌크 점도,
전단률·시간 간격·cutoff 수렴, MPI 가속률까지 확인한 배포본은 아닙니다.
정밀 기본 표는 AR10에서 약 810 MiB입니다. 최초 표 생성 비용과 적분 비용은 다릅니다.
8개 thread 옵션은 **표 생성**에 사용합니다. 기존 전역 implicit solver는 기본적으로
rank 0에서 풀며, 이번 작업에서 8-rank 적분 가속이나 전체 3배 가속을 측정하지 않았습니다.

## 물리 조건과 출력

전체 두께 300 nm, 폭 `300 nm × AR`입니다. 접촉은 3 nm부터 활성화하며 기본 sliding
마찰계수는 0.1입니다. 활성화 하중 장벽과 contact attraction은 없습니다. 접촉 중에도
lubrication을 유지하며 유체 간격은 `max(h,0)+0.15 nm`입니다. 접촉 거칠기와 유체 정규화
길이는 독립입니다. 외측 간격은 기본 `0.5*a`이고 마지막 20%에서 보정 행렬 전체를
매끄럽게 감쇠합니다. AR10에서는 외측 간격 750 nm입니다.

`history.csv`에 총응력·점도·상대점도와 기존 성분 분해, `sigma_finite_gap_Pa`,
`eta_finite_gap_Pa_s`, 보정 전력, 힘 평형 잔차를 기록합니다. 부호가 있는 보정 전력은
전체 소산과 구분합니다. 정상상태 판단 후 interval mean 열로 평균을 계산합니다.
`parameters.txt`에는 선택 모델·AR·표·검사 상태를 남깁니다.

빌드·표 생성·실행에는 기존 GCC/CMake/OpenMPI, 포함된 Eigen, Python 3.6 이상을 사용하며
새 Python 패키지는 필요하지 않습니다. `validation_basis/full17_reference.py`를 직접
재실행하는 선택적 독립 검증에만 NumPy/SciPy가 필요합니다. 일반 실행의 dependency는 아닙니다.
Negishi 모듈은 build 스크립트의 기존 설정을 유지합니다.

`models/ar10_matched.ssb`는 기존 회귀 검사 입력입니다. 이번 새 기저는
`models/ar10_regenerated.ssb`입니다. `validation_basis/run_physical_gate.py`와
`run_interpolation_gate.py`는 실행 전 검사 코드이므로 삭제하면 안 됩니다.
