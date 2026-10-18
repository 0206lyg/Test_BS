# Test_BS — AR별 Stokes 기저·저항표

Release: `20261018-ar-basis-resolution-fix`

기준 Git 커밋: `e2475dabec61dcb466fa8c1c9abb4c7620ab84d9`

## AR20 표 생성 실패 수정

기존 기본값은 모든 AR에 source ring 96개와 고정 각도 적분 해상도를 사용했습니다.
AR20에서 기저 경계조건 오차가 21.36%가 되어, 두 입자 표 생성 전에 정확도 검사에서
중단됐습니다. 이번 수정은 기저 생성 해상도를 형상에 맞게 조절합니다.
접촉·마찰·lubrication 식과 기존 합격 기준은 유지합니다.

| AR | Source rings | 두 입자 계산용 각도 분할 | 기저 fitting / 독립 확인용 각도 분할 |
|---:|---:|---:|---:|
| 2·5·10 | 96 | 128 | 4096 / 8192 |
| 20 | 192 | 256 | 16384 / 32768 |
| 25 | 240 | 320 | 25600 / 51200 |

이는 수치 해상도 선택식이며 모든 AR의 정확도를 보증하는 표가 아닙니다.
두께 방향 반축 c, 폭 방향 반축 a인 confocal source의 표면까지 최소 거리는
`a-sqrt(a*a-3*c*c/4)`입니다. 큰 AR에서는 `3*c*c/(8*a)`로 작아지므로 source ring
분할은 AR에, 기저 각도 적분 분할은 AR²에 비례시킵니다. 두 입자 계산의 각도 분할은
고정된 h/c 기준 간격을 분해하도록 AR에 비례시킵니다. 이후 실제 수치 검사를 거칩니다.

생성 직후 경계조건·단일입자 저항·저장/읽기 검사를 통과한 기저만 사용 파일로 설치합니다.
실패하면 측정값·기준값·보고서 경로를 출력하고 중단합니다. 실패한 후보를 사용 파일로
남기지 않습니다. 기존 버전이 남긴 실패 파일은 자동으로 덮어쓰지 않습니다.
아래 새 파일명을 사용하면 이전 실패 기저가 재사용되지 않습니다.

## 적용과 실행

수정 ZIP을 `/home/lee6456/slurry`에 둔 후 다음 명령을 순서대로 실행합니다.
이 ZIP은 위 Git 커밋에 적용하는 변경 파일 모음입니다. 전체 LAMMPS나 생성된 대형
기저·표는 포함하지 않습니다. README는 이 파일 하나만 제공합니다.

```bash
cd /home/lee6456/slurry
unzip -o Test_BS_AR20_basis_fix.zip
export SLURRY_ROOT=/home/lee6456/slurry
sbatch build_slurry_cpu.sbatch
```

빌드 작업이 성공한 다음 아래 한 작업을 제출합니다. 변경된 입력 생성 코드와 빌드
기록을 일치시키기 위해 빌드 절차가 필요합니다.

```bash
sbatch --job-name=table_AR20_fix --output=table_AR20_fix_%j.out --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G --time=24:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 20 --phi 0.24 --count 300 --lubrication-model finite_gap_fast --finite-gap-fast-source /home/lee6456/slurry/lammps_slurry/models/ar20_resolved.ssb --finite-gap-fast-threads 16 --steps 20 --prep-steps 0 --output-interval 1 --average-start-strain 0
```

순서는 기저 생성 → 기저 검사 → 정밀 두 입자 표 생성 → 보간 검사 → 20 step 실행입니다.
20 step은 연결 확인용입니다. 정상상태 점도 측정에는 충분한 strain이 필요합니다.
전체 표 생성은 수일 규모가 될 수 있으며, 여기서 전체 소요 시간을 실측하지 않았습니다.
시간 제한 후 동일 명령을 제출하면 완료된 표 계산 조각을 재사용합니다.

`GENERATE`는 기저 생성 단계, `Building matrix-log/Fourier table profile=full`은
정밀 표 생성 단계, `CHUNK_COMPLETED`는 계산 조각 완료, `CHECKPOINT_RESUMED`는
재개, `TABLE_WRITTEN`은 표 저장입니다. 기저의 kernel/경계 확인 중에는 로그가 잠시
늘지 않을 수 있습니다. 작업 번호를 넣은 실제 로그 파일을 `tail -f`로 확인합니다.

16 threads는 표 생성용 시작 설정이며 최적값을 실측한 결과는 아닙니다. MPI ntasks는 1로
둡니다. 기저 생성은 직렬이고 두 입자 표 생성은 OpenMP를 사용합니다. 실제 다체 implicit
solver는 기본적으로 rank 0에서 풀므로, 완성된 표를 쓰는 실행은 별도로 1 CPU부터 시작합니다.

AR2·5·10의 자동 생성 설정은 종전과 같습니다. 다른 AR에는 AR와 기저 파일명을 함께
지정합니다. AR1은 `--lubrication-model mari`를 사용합니다. `auto`는 기존 모델을 선택하므로
새 표를 쓰려면 `finite_gap_fast`를 명시해야 합니다. 혼합 AR·다분산·prolate는 대상이 아닙니다.

## 검증 범위

이번 AR20 수정 검사는 `validation_basis/ar20_resolution_fix.json`에 기록합니다.
최종 기저 경계조건 오차는 0.4985%(기준 1%), 단일입자 저항 오차는 0.0000371%(기준 0.1%)입니다.
면–가장자리 배치에서 계산용 각도 분할 256→512의 저항 변화는 0.00348%였습니다.
생성 해상도 비교, 기존 단일입자 검사, 두 입자 각도 적분 수렴 검사를 구분합니다.
이 검사를 AR20의 독립 두 입자 정확도, 전체 정밀 표 완성 또는 벌크 점도 검증으로
해석하면 안 됩니다. 전체 정밀 표의 1% 보간 검사는 표 생성 후 자동으로 실행됩니다.

기존 AR10 독립 두 입자 검사에서는 75 nm의 엇갈림 배치 0.190%, 150 nm 기울어진 배치
0.444%, 별도 112.5 nm 배치 0.0802%의 전체 11성분 에너지 오차를 확인했습니다.
3 nm의 0.0562% 기록은 같은 모델에 대한 국소 보간 검사이며 독립 Stokes 정확도가 아닙니다.
2·8입자에서 dt=1e-6 s, 전단률 100/s, 20 step의 기존 적분 확인 기록도 남아 있습니다.
기존 기록과 미완료 항목은 `RESULT.json`과 `validation_basis/`에 있습니다.

## 물리 조건과 출력

두께는 300 nm, 폭은 `300 nm × AR`입니다. 3 nm부터 기계적 접촉과 sliding friction을
활성화하며 기본 마찰계수는 0.1입니다. 활성화 하중 장벽과 contact attraction은 없습니다.
접촉 중에도 lubrication을 유지하고 유체 간격은 `max(h,0)+0.15 nm`입니다.
외측 간격은 기본 `0.5*a`이며 마지막 20%에서 전체 보정 행렬을 매끄럽게 감쇠합니다.

V3는 `h_eff/c >= 0.5`에서 해당 간격의 near Stokes 기저를 쓰고, 그 아래에서 기존 film
저항과 두 기준 간격 0.5·1의 잔여 저항을 연결합니다. 파일에 남은 far 슬롯은 V3의
저항 계산에 사용하지 않습니다. 실제 적분은 저장된 저항표를 조회합니다.

`history.csv`에는 총응력·점도·상대점도·성분별 contribution·힘 평형 잔차를,
`parameters.txt`에는 선택 모델·AR·표·검사 상태를 남깁니다. 표와 생성 조건은 기저 옆
`.fast_cache/`에 저장됩니다. 거친 `--finite-gap-fast-smoke` 표는 정량 점도에 사용하지 않습니다.

일반 빌드·표 생성·실행에는 기존 GCC/CMake/OpenMPI, 포함된 Eigen, Python 3.6 이상을
사용하며 새 Python 패키지를 추가하지 않습니다. 선택적 독립 검증 스크립트 일부에만
NumPy/SciPy가 필요합니다. `validation_basis/run_physical_gate.py`,
`run_interpolation_gate.py`, 회귀 입력 `models/ar10_matched.ssb`는 유지해야 합니다.
