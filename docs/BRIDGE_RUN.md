# Compact bridge 실행

새 경로는 `fix slurry/bridge`를 사용합니다. 이전 `finite_gap_fast` 기저·저항표를 생성하거나 읽지 않습니다. 근거리에는 검증한 compact bridge v4, 중거리에는 ellipsoid 형상함수를 유지하는 particle–mesh FFT를 사용합니다. `--mobility-backend direct`는 같은 Fourier 합의 직접 계산 기준 경로입니다. 유한 간격의 모델 선택과 중거리 해상도는 별도 수렴 항목입니다.

기존 빌드 스크립트로 새 소스를 빌드한 뒤 실행합니다. 현재 소스와 실행 파일이 빌드 기록과 일치해야 합니다. 실행기는 해당 실행의 프로그램과 공유 라이브러리를 `runtime/`에 복사한 뒤 입자 준비와 LAMMPS를 시작합니다.

```bash
sbatch build_slurry_cpu.sbatch
# 위 빌드 완료 후:
sbatch lammps_slurry/run_bridge_cpu.sbatch \
  --output runs/ar10_bridge_solver_check --ar 10 --count 500 --phi 0.24 \
  --initialization random --steps 2 --output-interval 1 \
  --average-start-strain 0
```

위 짧은 실행에서 `history.csv`의 step 0, 1, 2와 계산시간을 확인합니다. 출력 폴더는 기존 실행과 다른 새 이름이어야 합니다. 기본값 `random`은 기존 기하 전용 성장 packer를 사용합니다. 선택 가능한 `aligned`는 대칭축이 z인 정렬 격자이며 평형화된 무작위 미세구조를 뜻하지 않습니다.

현재 solver는 접촉력도 미지수로 유지하는 semismooth Newton과 right FGMRES를 사용합니다. 국소 접촉 변수를 제거한 sparse Schur 전처리기를 사용하며, 마지막에는 원래 접촉 법칙으로 힘을 다시 계산해 `--tolerance`를 검사합니다. 별도 solver 선택이나 continuation 환경변수가 필요하지 않습니다. `--iterations`는 한 선형계의 Krylov 상한입니다.

`SLURRY_BRIDGE_TRACE=1`을 export한 뒤 제출하면 `prox_*`/`polish_*` 단계, 원래 방정식의 `residual`, line search에 사용하는 `merit`, 반복 횟수와 시간이 표시됩니다. `merit`가 감소하는 도중 `residual`은 일시적으로 커질 수 있으며, 최종 허용오차는 원래 `residual`에 적용합니다. CSV의 전체 Newton/Krylov 반복 수와 `mobility_applications`에는 마지막 실제 방정식 검증·보정 비용도 포함됩니다. `continuation_*` 열은 현재 기본 경로에서 0입니다.

```bash
sbatch lammps_slurry/run_bridge_cpu.sbatch \
  --output runs/ar20_bridge_random --ar 20 --count 500 --phi 0.24 \
  --initialization random --steps 100000 --timestep 1e-6 \
  --output-interval 100 --average-start-strain 5
```

주요 기본값은 두께 300 nm, 매질 점도 0.01 Pa·s, xz 전단률 100 s⁻¹, `max(actual_gap,1 nm)` 유체 간격, 접촉 시작 간격 3 nm, 법선 강성 100 N/m, 접선 강성 80 N/m, 마찰계수 0.1입니다. `--gap-floor-nm`과 `--contact-gap-nm`는 서로 독립적입니다. `--spectral 2`가 기본 중거리 해상도입니다. `--spectral-tolerance 1e-8`은 Fourier 절단 허용값이며 격자 보간·표면 구적·고파수 상호작용 생략을 포함한 전체 오차가 아닙니다.

근거리 이미지 검사는 `3a` 범위의 모든 이미지와 자기 이미지를 사용합니다. `L>6a` 제한은 없습니다. 무작위 성장·간격 정리 packer는 `L>4a+2h_c`를 필요로 하며, 이 조건은 유체력의 제한이 아닙니다. 정렬·무작위 초기화 모두 입력을 쓰기 전 실제 표면 gap을 검사합니다. `packing_clearance.json`의 `minimum_surface_gap_m`이 요청한 접촉 시작 간격 이상이어야 합니다. 입자 크기와 상자 크기는 유지합니다.

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
