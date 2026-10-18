# Test_BS — 표 생성 실패 수정

Release: `20261018-pair-quadrature-runtime-fix`
기준 Git: `e2475dabec61dcb466fa8c1c9abb4c7620ab84d9`

## 수정 내용

AR20은 source 기저 해상도를 늘린 뒤에도, 다른 입자의 유동을 받는 표면 격자가
15×29로 남아 있어 실패했습니다. 기본 표면 격자를 45×87로 늘렸습니다.
실패했던 실제 방향에서 raw reciprocity 오차는 1.001139% → 0.005519%로
감소했습니다. 허용 기준 1%는 유지하며, 실패한 계산을 강제로 대칭화해 통과시키지 않습니다.
큰 중간 행렬은 32개 표면점씩 계산해 작업자당 메모리를 줄였습니다. 실패했던
원본 격자 입력에서 기존 방식과 전체 17×17 행렬이 바이트 단위로 일치했습니다.

AR2·5의 SIGBUS 원인은 제공된 로그만으로 확정하지 못했습니다. 실행 중 빌드 파일이
교체되는 위험을 막기 위해 코드·실행 파일·공유 라이브러리를 작업별 복사본으로 고정합니다.
SIGBUS에 한해서 검증된 체크포인트로 새 프로세스에서 한 번 재시도합니다. 반복되면 중단하고
노드·작업·종료 코드 기록을 남깁니다. 수치 검사 실패는 재시도하거나 건너뛰지 않습니다.
기존 기본 옵션으로 Mari/auto를 실행할 때 fast 전용 옵션으로 잘못 판단하던 문제도 수정합니다.

단일입자 검사만으로 표 생성을 시작하지 않습니다. AR2·5·20 등에서는 5개 방향의 실제
두 입자 저항을 먼저 검사합니다. h/c=0.5·1의 raw 전체 행렬과 h/c=0.001·0.02의
연결된 전체 저항을 합쳐 20개 기록이 모두 있어야 통과합니다. 이는 상반성·잔차·양의
소산 검사입니다. 독립 Stokes 정확도 검사와는 다릅니다. AR10 독립 기준 검사는 유지합니다.

## 적용

ZIP은 위 Git 백업에 덮어쓰는 누적 수정 파일입니다. 전체 LAMMPS, 빌드 결과,
대형 기저·표·실행 dump는 포함하지 않습니다. README는 이 파일 하나입니다.

```bash
cd /home/lee6456/slurry
unzip -o Test_BS_pair_table_fix.zip
export SLURRY_ROOT=/home/lee6456/slurry
sbatch build_slurry_cpu.sbatch
```

빌드 성공 후 아래 명령을 각각 제출합니다. 새 기저 이름을 사용해야 이전의 성긴
표면 격자가 재사용되지 않습니다. 16 threads는 기존 실행과 같은 시작 설정이며 최적값을
실측한 것은 아닙니다. MPI task는 1로 둡니다.

```bash
sbatch --job-name=table_AR2_fixed --output=table_AR2_fixed_%j.out --ntasks=1 --cpus-per-task=16 --mem=32G --time=24:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 2 --phi 0.24 --count 300 --lubrication-model finite_gap_fast --finite-gap-fast-source /home/lee6456/slurry/lammps_slurry/models/ar2_targets3.ssb --finite-gap-fast-threads 16 --steps 20 --prep-steps 0 --output-interval 1 --average-start-strain 0
```

```bash
sbatch --job-name=table_AR5_fixed --output=table_AR5_fixed_%j.out --ntasks=1 --cpus-per-task=16 --mem=32G --time=24:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 5 --phi 0.24 --count 300 --lubrication-model finite_gap_fast --finite-gap-fast-source /home/lee6456/slurry/lammps_slurry/models/ar5_targets3.ssb --finite-gap-fast-threads 16 --steps 20 --prep-steps 0 --output-interval 1 --average-start-strain 0
```

```bash
sbatch --job-name=table_AR20_fixed --output=table_AR20_fixed_%j.out --ntasks=1 --cpus-per-task=16 --mem=32G --time=24:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 20 --phi 0.24 --count 300 --lubrication-model finite_gap_fast --finite-gap-fast-source /home/lee6456/slurry/lammps_slurry/models/ar20_targets3.ssb --finite-gap-fast-threads 16 --steps 20 --prep-steps 0 --output-interval 1 --average-start-strain 0
```

순서는 기저 생성 → 두 입자 사전 검사 → 정밀 표 생성 → 보간 검사 → 20 step입니다.
20 step은 연결 확인용이며 정상상태 점도 측정이 아닙니다. 시간 제한으로 종료되면 같은
명령을 다시 제출합니다. 완료된 표 조각은 기저 옆 `.fast_cache/`에서 재사용합니다.
이번에는 기저 격자가 바뀌므로 이전 버전의 표 조각은 재사용하지 않습니다.

로그에서 `GENERATE`는 기저 생성, `PAIR_CHECK`는 사전 검사,
`CHUNK_COMPLETED`는 표 조각 완료, `CHECKPOINT_RESUMED`는 재개,
`TABLE_WRITTEN`은 표 저장입니다. 실제 출력 파일명을 넣어 `tail -f`로 확인합니다.
표면 격자가 늘어 이전보다 표 생성 비용이 큽니다. 전체 정밀 표 시간은 실측하지 않았습니다.
완성된 표를 조회하는 dynamics에는 이 표 생성 비용이 반복되지 않습니다.

## 검증 범위와 남은 항목

AR2·5·20 각각 두 입자 사전 검사 20개를 통과했습니다. 실제 Stokes 저항으로
각각 640개 격자점(4×4×8×5)의 coarse 표를 끝까지 생성했습니다. 모든 격자점의
재조회·양의 소산 검사와 추가 32개 위치의 양의 소산·회전 공변성 검사를 통과했습니다.
생성 시 4·4·8 threads에서 1 thread로 바꿔 체크포인트를 재개해도 파일이 바이트 단위로 같았습니다.
새 CMake Release 빌드와 실제 serial LAMMPS의 plugin 로딩도 확인했습니다.
추가 Python 패키지 없이 실행 경로 27개, runtime 보호 4개, SIGBUS 복구 5개 시험을 통과했습니다.

결과 요약은 `RESULT.json`, 상세 수치는 `validation_basis/table_failure_fix.json`에 있습니다.
위 coarse 표는 생성·저장·재개 기능 검사용이며 연구용 정밀 표가 아닙니다.
AR20 실패 배치에서 표면 격자 3배→4배의 전체 저항 변화는 0.01885%지만,
기저 차수 12→14의 전체 저항 에너지 기준 최대 변화는 6.6248%, 14→16은 2.3722%입니다.
차수를 높일 때 변화는 줄었지만 1% 이내로 수렴하지는 않았습니다. 기본 12차 기저를
물리 정확도가 수렴한 결과로 사용하면 안 됩니다. 이 차이는 벌크 점도 오차의 직접 추정값은 아닙니다.
16차 비교는 진단용 실행 파일의 저장 용량 제한만 늘려 수행했으며, 배포 코드의 기본 차수와
저장 용량 제한은 유지합니다. 이 패치는 표 생성 실패 수정이며 고차 기저 정확도 검증 완료판은 아닙니다.
전체 정밀 표 및 그 보간 검증, 독립 좁은 간격 Stokes 정확도, 정상상태 점도는 별도입니다.

V3의 h_eff/c<0.5 구간은 기존 film 저항과 두 기준 간격의 잔여 저항 연결식을 유지합니다.
이번 패치는 이 구간을 직접 Stokes 해로 교체하지 않습니다. 두께 300 nm,
3 nm부터 접촉·마찰(μ=0.1), 활성화 하중 장벽·attraction 없음이라는 물리 조건도 유지합니다.
추가 Python 패키지는 필요하지 않습니다. 실행 도구는 Python 3.6 이상과 기존
GCC/CMake/OpenMPI 및 포함된 Eigen을 사용합니다.
