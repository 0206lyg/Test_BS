# Slurry 실행

실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다. 파일 앞부분의 `PARTICLE_SHAPE=spheroid|flake`, `AR`, `PHI`, `COUNT`, `STEPS` 등을 직접 수정한 뒤 인수 없이 제출합니다. 기본값은 AR2, φ=0.24, 입자 500개, 100000 step, 기록 간격 100 step입니다. `PREPARE_ONLY=1`이면 입자와 입력만 준비하며 LAMMPS를 실행하지 않습니다.

기존 빌드 스크립트로 새 소스를 빌드한 뒤 실행합니다. 현재 소스와 실행 파일이 빌드 기록과 일치해야 합니다. 실행기는 해당 실행의 프로그램과 공유 라이브러리를 `runtime/`에 복사한 뒤 입자 준비와 LAMMPS를 시작합니다.

빌드 제출: `sbatch build_slurry_cpu.sbatch`

빌드 완료 후 실행 제출: `sbatch run_slurry_cpu.sbatch`

`OUTPUT_DIR`를 비워두면 `runs/ar2_phi0.24_n500_<job ID>`처럼 실행마다 별도 폴더가 생깁니다. `OUTPUT_DIR`에 직접 새 경로를 지정할 수도 있습니다. 상대 경로는 저장소 최상위 기준입니다. 기존 실행 폴더에는 덮어쓰지 않습니다. `BUILD_ENV`를 비워두면 `build/lammps/env.sh`의 빌드를 사용합니다.

자원 기본 요청은 **1 node, MPI 1 rank, 물리 코어 16개, RAM 96 GB**입니다. 코어 수는 파일 상단의 `#SBATCH --cpus-per-task`에서 바꿉니다. Python 실행기는 한 번만 실행하며, MPI 빌드일 때 LAMMPS 하나를 `mpirun -np 1 --map-by slot:PE=<할당 코어 수> --bind-to core --report-bindings`로 시작합니다. 실행 실패 후 다른 명령으로 재시도하지 않습니다. `console.log`에서 MPI 바인딩을 확인하고 `launch.json`에서 실제 명령과 스레드 설정을 확인할 수 있습니다. 실행 배치 파일을 `mpirun`이나 다중 task `srun`으로 감싸지 않습니다.

현재 경로는 `fix slurry/bridge`를 사용합니다. 근거리에는 spheroid의 compact bridge v4 또는 flake의 면–면 Reynolds 박막·가장자리 근사, 중거리에는 ellipsoid 형상함수를 유지하는 FFTW particle–mesh 연산을 사용합니다. `MOBILITY_BACKEND=direct`는 같은 Fourier 합의 직접 합산 기준 경로이며 밀집 행렬 LU solver가 아닙니다. 유한 간격의 모델 선택과 중거리 해상도는 별도 수렴 항목입니다.

solver는 접촉력도 비선형 미지수로 유지하는 semismooth Newton과 right FGMRES를 사용합니다. 선형 Newton 단계에서는 국소 접촉 증분을 정확히 제거하고, 원래 접촉 법칙의 물리 잔차로 최종 수렴을 검사합니다. `SOLVER_TOLERANCE`가 그 상대 허용오차이며, `KRYLOV_ITERATIONS`는 한 선형계의 반복 상한입니다.

`TRACE_SOLVER=1`로 바꾸면 Newton/선형 반복과 시간이 출력됩니다. line search에 사용하는 `merit`가 감소하는 도중 원래 방정식의 `residual`은 일시적으로 커질 수 있으며, 최종 허용오차는 원래 물리 잔차에 적용합니다. CSV의 전체 Newton/Krylov 반복 수와 `mobility_applications`에는 실제 방정식 검증 비용도 포함됩니다.

`INITIALIZATION=random`은 기하 전용 성장 packer를 사용합니다. `aligned`는 대칭축이 z인 정렬 격자이며 평형화된 무작위 미세구조를 뜻하지 않습니다. 요청한 초기화 방식이 실패했다고 다른 방식으로 자동 전환하지 않습니다.

주요 기본값은 두께 300 nm, 매질 점도 0.01 Pa·s, xz 전단률 100 s⁻¹, `max(actual_gap,1 nm)` 유체 간격, 접촉 시작 간격 3 nm, 법선 강성 100 N/m, 접선 강성 80 N/m, 마찰계수 0.1입니다. `--gap-floor-nm`과 `--contact-gap-nm`는 서로 독립적입니다. `--spectral 2`가 기본 중거리 해상도입니다. `--spectral-tolerance 1e-8`은 Fourier 절단 허용값이며 격자 보간·표면 구적·고파수 상호작용 생략을 포함한 전체 오차가 아닙니다.

근거리 이미지 검사는 spheroid의 경우 `3a`, flake의 경우 원기둥 경계반경 `sqrt(a²+c²)`에 따른 충분한 범위의 모든 이미지와 자기 이미지를 사용합니다. `L>6a` 제한은 없습니다. 무작위 성장·간격 정리 packer는 spheroid는 `L>4a+2h_c`, flake는 `L>4sqrt(a²+c²)+2h_c`를 필요로 하며, 이 조건은 유체력의 제한이 아닙니다. 정렬·무작위 초기화 모두 입력을 쓰기 전 실제 표면 gap을 검사합니다. `packing_clearance.json`의 `minimum_surface_gap_m`이 요청한 접촉 시작 간격 이상이어야 합니다. 입자 크기와 상자 크기는 유지합니다.

한 step의 접촉력은 실제 Euler 이동·quaternion 회전·전단 격자 갱신 후 gap으로 계산한 압축량과 일치할 때만 수락합니다. 내부 보정 반복은 물리 시간을 추가로 진행하거나 접선 이력을 중복 누적하지 않습니다. CSV의 `actual_endpoint_min_gap_m`, `contact_gap_consistency_m`, `geometry_iterations`, `normal_elastic_energy_J`, `tangential_elastic_energy_J`는 해당 출력 시점에서 다음 이동을 위해 구한 상태의 진단입니다. `min_gap_m`은 현재 위치의 gap입니다. 유한 스프링이므로 모든 하중에서의 비침투를 강제하는 hard wall은 아닙니다.

```bash
python3 lammps_slurry/tools/summarize_bridge.py \
  runs/ar20_bridge_random/history.csv --start-strain 5 \
  --blocks 5 --output runs/ar20_bridge_random/viscosity_summary.json
```

평균은 출력 순간의 값들을 단순 평균하지 않고 매 시간단계에 누적한 interval mean을 시간 가중합니다. 지정한 strain 창 안에 완전히 들어오는 구간만 사용하며, 빈 창은 오류로 알립니다. 블록 평균의 상대 표준편차와 전·후반 차이를 출력하지만 정상상태를 자동으로 인증하지는 않습니다. `sigma_hydro_Pa`에는 계산기가 정의한 유체 응력을 이미 반영하므로 LAMMPS pair virial을 다시 더하지 않습니다.

`run_bridge.py --prepare-only`로 검증된 runtime과 입력만 만들 수 있습니다. `--help`는 준비·물성·해상도 옵션 전체를 출력합니다. `final.data`는 마지막 기하를 보존하지만 마찰 이력이 포함된 재시작 파일은 아닙니다.

`trajectory.lammpstrj`의 `f_hydro[1:17]`는 다음 SI 물리량입니다. `pair_style zero`의 0인 force/torque 대신 이 열을 사용합니다.

| 열 | 물리량 |
| --- | --- |
| 1–3 | 입자에 가해지는 유체력 F (N) |
| 4–6 | 입자에 가해지는 유체 토크 T (N·m) |
| 7–11 | 입자가 유체에 가하는 stresslet의 STF5 성분 (N·m) |
| 12–14 | 입자에 가해지는 접촉력 (N) |
| 15–17 | 입자에 가해지는 접촉 토크 (N·m) |

STF5 기저의 순서는 `diag(1,-1,0)/sqrt(2)`, `diag(1,1,-2)/sqrt(6)`, 대칭 xy/xz/yz이며 비대각 기저의 두 성분은 각각 `1/sqrt(2)`입니다. Stresslet은 on-fluid 부호이므로 `sigma_hydro_Pa`와 직접 같은 부호라고 해석하거나 다시 합산하지 않습니다.

직접 LAMMPS 입력을 작성할 때는 `fix deform`를 `slurry/bridge`보다 먼저 정의하고 xz 전단에 `remap v`를 사용해야 합니다. 현재 구현은 `deform 0`과 일정 전단률을 요구합니다. 생성기는 이 조건을 자동으로 적용합니다.

```lammps
plugin load ${slurry_plugin}
# units/box/particles 설정 후:
fix shear all deform 0 xz erate 100 remap v units box
fix hydro all slurry/bridge 0.01 100 1e-9 100 contact 3e-9 0.1 80 mobility_backend mesh spectral 2 spectral_tolerance 1e-8 history history.csv 100 5
```

입력의 `${slurry_plugin}` 참조 자체에는 따옴표를 붙이지 않습니다. 실행기는 `-var` 값에 경로를 감싼 실제 큰따옴표를 넣어 공백이 있는 경로도 처리합니다. 일반 실행에는 위 launch script를 사용하면 됩니다.

Flake의 실제 원기둥 체적·질량·관성, 간격 인증, 중거리 근사 범위는 [형상별 공통 실행 안내](../validation_flake_integration/README_KO.md)에 정리했습니다. Flake의 실제 표면 겹침은 위치·마찰 이력을 갱신하기 전에 거부합니다.
