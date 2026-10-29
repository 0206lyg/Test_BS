# Slurry 실행

실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다. 파일 앞부분의 `AR`, `PHI`, `COUNT`, `STEPS` 등을 직접 수정한 뒤 인수 없이 제출합니다. 현재 실행 경로는 `PARTICLE_SHAPE=spheroid`, **1 ≤ AR ≤ 5**를 지원하며 유체 backend 기본값은 `nearfield`입니다. 저장소 예제는 AR2, φ=0.24, 입자 500개, 40000 step, 기록 간격 40 step입니다. 사용자가 편집한 기존 값은 overlay 적용 시 보존됩니다. `PREPARE_ONLY=1`이면 입자와 검증된 runtime·입력만 준비하며 LAMMPS를 실행하지 않습니다.

## 적용·빌드·제출

저장소 디렉토리에서 overlay와 선택한 수치 설정을 적용한 뒤 기존 빌드 스크립트로 **다시 빌드**합니다. ZIP은 `run_slurry_cpu.sbatch`를 제외합니다. 새 유체 solver를 사용하려면 plugin을 다시 만들어야 합니다.

```bash
unzip -DD -o slurry_nearfield_overlay_20261028.zip
python3 apply_nearfield_settings.py
sbatch build_slurry_cpu.sbatch
```

`apply_nearfield_settings.py`는 `MOBILITY_BACKEND=nearfield`, `NORMAL_STIFFNESS=30`, `TANGENTIAL_STIFFNESS=24`, `ROLLING_STIFFNESS=24` 네 assignment만 변경합니다. Timestep, step 수, 출력 간격, AR·농도·입자 수, 경로와 자원 요청을 유지합니다. 최초 변경 전 `.before_nearfield_update` 백업을 남기며, 다른 스크립트 경로에는 `--file`을 사용합니다. 빌드 로그의 `Build complete`를 확인한 뒤 제출합니다.

```bash
sbatch run_slurry_cpu.sbatch
```

현재 소스와 실행 파일이 빌드 기록과 일치해야 합니다. 실행기는 해당 실행의 프로그램, 공유 라이브러리와 준비 스크립트를 `runtime/`에 복사한 뒤 입자 준비와 LAMMPS를 시작합니다. 빌드·소스를 나중에 변경해도 이미 시작한 실행이 다른 라이브러리로 바뀌지 않도록 합니다.

`OUTPUT_DIR`를 비워두면 `runs/spheroid_ar2_phi0.24_n500_<job ID>/`처럼 실행마다 별도 폴더가 생깁니다. `OUTPUT_DIR`에 직접 새 경로를 지정할 수도 있습니다. 상대 경로는 저장소 최상위 기준입니다. 기존 실행 폴더에는 덮어쓰지 않습니다. `BUILD_ENV`를 비워두면 `build/lammps/env.sh`의 빌드를 사용합니다.

자원 기본 요청은 **1 node, MPI 1 rank, 물리 코어 16개, RAM 96 GB**입니다. 코어 수는 파일 상단의 `#SBATCH --cpus-per-task`에서 바꿉니다. Python 실행기는 한 번만 실행하며, MPI 빌드일 때 LAMMPS 하나를 `mpirun -np 1 --map-by slot:PE=<할당 코어 수> --bind-to core --report-bindings`로 시작합니다. 실행 실패 후 다른 명령으로 재시도하지 않습니다. `console.log`에서 MPI 바인딩을 확인하고 `launch.json`에서 실제 명령과 스레드 설정을 확인할 수 있습니다. 실행 배치 파일을 `mpirun`이나 다중 task `srun`으로 감싸지 않습니다.

이 자원 예제는 이전 FFT 경로와 호환하도록 남아 있습니다. 기본 nearfield의 geometry와 희소 분해는 직렬로 계산되며 16개 코어의 비례 가속을 약속하지 않습니다. Helper는 사용자 자원을 변경하지 않습니다.

Negishi의 기본 빌드 모듈은 Python, GCC/OpenMP, OpenMPI, CMake, FFTW 및 static oneMKL입니다. 기존 mesh/direct 경로를 함께 빌드하기 위한 의존성이며 nearfield 실행에서 FFT나 PARDISO를 호출하지 않습니다. 사이트 모듈 이름이 다르면 `SLURRY_PYTHON_MODULE`, `SLURRY_GCC_MODULE`, `SLURRY_MPI_MODULE`, `SLURRY_CMAKE_MODULE`, `SLURRY_FFTW_MODULE`, `SLURRY_MKL_MODULE`을 지정합니다.

## 크기·regularization·마찰

`NOMINAL_DIAMETER_UM=8`은 **체적 등가 직경 D=8 μm**입니다. 두께 또는 장축 직경을 8 μm로 고정하는 설정이 아닙니다. AR=a/c에서 equatorial semiaxis a와 polar semiaxis c는

\[
a=(D/2)\mathrm{AR}^{1/3},\qquad c=(D/2)\mathrm{AR}^{-2/3}
\]

이며, 입자 체적 `4*pi*a*a*c/3 = pi*D^3/6`은 AR에 무관합니다. 기준 반경과 계산 정규화 길이는 `a_ref=D/2=4 μm`입니다. 생성한 입자의 실제 축 길이와 이 정의를 `parameters.json`에 기록합니다. Fix는 입력 spheroid의 체적 등가 반경이 지정한 a_ref와 일치하는지도 검사합니다.

`LUBRICATION_EPSILON=0.001`은 Mari의 **반경 정규화**를 따릅니다. 실제 표면 gap을 h라 할 때 스칼라 lubrication 함수에 넣는 길이는

\[
h_{\rm reg}=\max(h,0)+\epsilon a_{\rm ref}.
\]

따라서 D=8 μm에서 추가 regularization 길이는 **4 nm**이며 AR을 바꿔도 유지됩니다. `epsilon*D=8 nm`를 쓰는 정의가 아닙니다. 표면 기하와 접촉 판정은 실제 h를 사용합니다. 이 길이는 명시적인 표면 roughness층이나 양의 gap에서 마찰을 켜는 shell이 아닙니다.

| 배치 설정 | CLI | 기본값·단위 |
| --- | --- | --- |
| `NOMINAL_DIAMETER_UM` | `--nominal-diameter-um` | 8 μm |
| `LUBRICATION_EPSILON` | `--lubrication-epsilon` | 0.001, 무차원 |
| `FRICTION` | `--friction` | μs=10, 무차원 |
| `ROLLING_FRICTION` | `--rolling-friction` | μr=10, 무차원 |
| `NORMAL_STIFFNESS` | `--normal-stiffness` | 30 N/m |
| `TANGENTIAL_STIFFNESS` | `--tangential-stiffness` | 24 N/m |
| `ROLLING_STIFFNESS` | `--rolling-stiffness` | 24 **N/m**, rolling quasi-force 강성 |

법선력은 실제 겹침 `max(-h,0)`에 대한 선형 스프링입니다. 기본 IMEX의 sliding·rolling 마찰 상한은 step 시작 시 실제 접촉의 법선 하중에서 정합니다. 새 접촉의 마찰 활성화는 다음 step부터 반영되며, 열리는 step의 남은 마찰력도 힘 균형에 포함합니다. 끝점에서 열린 접촉의 이력은 다음 step에 넘기지 않습니다. Rolling의 기준 길이는 Singh의 `a_ij=2*a_i*a_j/(a_i+a_j)`에 nominal radius를 넣은 값이며, 현재 단분산에서는 `a_ij=a_ref`입니다. 내부 각도 스프링 강성으로 환산하면 `kr*a_ref^2` N·m/rad입니다. **이전 `--rolling-stiffness`의 N·m/rad 수치를 현재 N/m 옵션에 그대로 넣으면 안 됩니다.** μr=0이면 rolling 저항이 꺼집니다.

현재 기본 μs=μr=10은 강한 마찰의 시험값이며 무한대 또는 포화한계가 검증된 값이 아닙니다. 그 자체로 graphite 실험 계수나 φJ≈0.30을 보장하지 않습니다.

다음 설정은 실행 옵션에서 제거했으며 예전 입력을 주면 오류로 알립니다: `--thickness-nm`, `--gap-floor-nm`, `--contact-gap-nm`, 모든 `--rough-*`, `--rolling-length-mode`, `--rolling-length-nm`, 모든 `--twisting-*`. 새 실행 경로에는 rough-hydrodynamics 추가항이나 twisting 접촉 스프링이 없습니다. Flake는 이 겹침 기반 접촉 모델을 지원하지 않으므로 선택할 수 없습니다.

## 초기배치와 solver

`INITIALIZATION=random`은 기하 전용 성장 packer를 사용합니다. `aligned`는 대칭축이 z인 정렬 격자이며 평형화된 무작위 미세구조를 뜻하지 않습니다. 요청한 초기화 방식이 실패했다고 다른 방식으로 자동 전환하지 않습니다.

두 초기화 방식 모두 입력을 쓰기 전 실제 주기 표면 gap이 **0 이상**임을 인증합니다. `packing_clearance.json`의 `target_gap_m`은 0이고 `minimum_surface_gap_m`도 0 이상이어야 합니다. 3 nm 또는 다른 양의 접촉 여유를 강제로 넣지 않습니다. 입자 크기와 상자 크기는 유지합니다. 무작위 성장 packer는 `L>4a`를 필요로 하며, 이 조건은 유체력의 제한이 아닙니다. Nearfield는 두 입자의 bounding radius 합에 `0.5*a_ref`를 더한 범위에서 주기 이미지를 탐색한 뒤 실제 표면 gap·곡률 cutoff를 적용합니다. 이전 mesh/direct의 `3a` 범위와 다르며, `L>6a` 제한은 없습니다.

현재 기본 `nearfield`는 단일 spheroid의 비등방 Stokes drag, Jeffery 회전·isolated stresslet 및 가까운 pair의 lubrication을 합한 저항 모델입니다. 구형 pair는 Mari의 leading squeeze·shear·pump 계수를 사용합니다. 비구형 pair는 실제 표면 곡률·지렛팔·접선과 회전 결합을 사용하는 확장입니다. Spheroid kernel 자체의 최소 normal-square completion은 유지하지만 이전 compact bridge의 finite-gap Schur completion과 주기 전역 mobility는 사용하지 않습니다. 이 범위와 구형 논문과의 차이는 [nearfield 변경 명세](NEARFIELD_UPDATE.md)에 정리합니다.

구형의 outer surface gap은 `0.5*a_ref`이고, 비구형은 local curvature로 범위를 더 제한합니다. 이는 접촉 regularization `epsilon*a_ref`와 별개입니다. `MOBILITY_BACKEND=mesh|direct`를 명시하면 이전 compact bridge와 주기 hydrodynamics 비교 경로를 선택합니다. `direct`는 Fourier 직접 합산이며 nearfield의 희소 직접 풀이와 다른 의미입니다. `SPECTRAL`과 `SPECTRAL_TOLERANCE`는 nearfield에서는 사용하지 않습니다.

기본 nearfield 적분은 **`contact_scheme imex`**입니다. 법선·sliding·rolling spring을 implicit하게 풀고, 마찰 상한에는 step 시작 상태의 `N_n=kn*max(-h_n,0)`을 고정해 씁니다. 임시 geometry 보정으로 이 상한을 바꾸지 않습니다. Sparse 6N Jacobian에 대한 FGMRES는 접촉 Jacobian의 SPD LDLT 분해를 preconditioner로 사용합니다. 첫 Newton 반복에서 분해하고 재사용하며, 선형 반복이 40회를 넘으면 다음 Newton 반복에서 갱신합니다. 현재 Jacobian에 대한 실제 선형계 잔차를 만족하지 못하면 SPD LDLT 직접 풀이를 사용합니다. Hydrodynamic LDLT 기반 경로도 비교용으로 남깁니다. 볼록 목적함수 감소와 최종 힘·토크 균형을 검사합니다. 전체 11성분 관계의 힘·토크·stresslet 복원은 유지합니다.

끝점의 법선 하중까지 Coulomb 상한에 넣는 `contact_scheme coupled`는 비교용입니다. 이 경로는 비대칭 Jacobian과 LU fallback을 허용하지만, 실제 AR5 실행에서 strain 0.11275에 반복이 멈춘 기록이 있습니다. Generator를 직접 사용할 때 `--contact-scheme coupled`로 선택할 수 있습니다. 일반 배치 실행은 generator 기본값인 `imex`를 사용하므로 사용자 스크립트에 새 assignment를 추가할 필요가 없습니다. 기존 mesh/direct는 Heun을 유지합니다.

`TRACE_SOLVER=1`은 잔차와 풀이 시간을 출력합니다. Nearfield는 Fourier mode·FFT plan·전역 mobility 적용을 사용하지 않지만, 접촉과 끝점 보정에는 반복이 필요합니다. Symbolic analysis·numeric factorization·reuse counter는 실제 희소 분해를 셉니다. 수치 강성은 실제 겹침이 구형의 반지름 또는 oblate 입자의 최소 반축의 5% 미만인지 확인하며 선택합니다. [이전 안정성 기록](CONTACT_STABILITY_UPDATE.md)은 이전 유체 모델의 결과입니다.

기본 매질 점도는 0.01 Pa·s, xz 전단률은 100 s⁻¹, timestep은 2.5e-6 s입니다. `kn=30`, `kt=kr=24 N/m`에서 AR5·N500은 1000 step, strain 0.25까지 완료했고 최대 겹침은 최소 반축의 2.374%였습니다. AR1·AR2는 각각 200 step을 통과했습니다. Timestep 비교와 실행 비용 등 검증 범위는 [변경 기록](NEARFIELD_UPDATE.md)에 있습니다. 강성을 높였다고 timestep을 자동으로 1/n로 줄이지 않습니다. Helper는 timestep·총 step 수·출력 간격을 바꾸지 않습니다. AR에 따른 발산 농도를 비교하려면 같은 크기·epsilon·전단 조건에서 다음을 확인해야 합니다.

- μs와 μr를 더 높였을 때 점도·접촉 통계 변화가 작아지는지 확인하여 강한 마찰의 포화를 판단합니다.
- 기존 timestep에서 실제 겹침과 점도를 먼저 측정하고, 별도의 timestep·강성 비교로 결과의 수치 의존성을 확인합니다. 강성을 바꾸는 모든 실행에 timestep 축소를 강제하지 않습니다.
- 입자 수, seed, 평균 strain 창 및 pair regularization·범위에 따른 결과 변화를 확인합니다. 이전 유체 모델과의 비교는 모델 변경의 영향을 따로 측정합니다.

이는 필요한 검증 항목이며, 기본 설정에 대해 이미 수행한 φJ 검증을 뜻하지 않습니다. 작은 solver 잔차와 작은 물리적 spring 오차는 별개의 요구입니다.

## 출력과 점도 평균

`parameters.json`은 nominal diameter, reference radius, 실제 축 길이, regularization 길이와 rolling 단위를 보존합니다. 기본 IMEX의 `history.csv`에는 `time_integrator=imex_frozen_cap`, `force_geometry_time_s=t_n`, `force_history_time_s=t_n+dt`, `friction_cap_time_s=t_n`이 기록됩니다. 응력은 시작 기하에서 평가한 유체 저항·접촉 방향과 갱신한 spring 이력으로 푼 해당 step의 힘 균형에 대응합니다. `actual_endpoint_min_gap_m`, `actual_endpoint_overlap_m`과 겹침 비율은 `next_step_end_time_s`의 실제 끝점 기하입니다. `contact_gap_consistency_m`은 끝점 geometry와 법선 접촉력의 일치 검사입니다.

`max_normal_cap_lag_N`과 `normal_cap_lag_l1_fraction`은 마찰 상한에 쓴 시작 하중과 끝점 하중의 차이를 기록합니다. `max_sliding_cap_excess_N`·`max_rolling_cap_excess_N_m`은 끝점 하중 기준 Coulomb 상한의 초과량입니다. 이것은 하중 지연에 따른 이산화 오차이며, 풀이 후 force clipping으로 숨기지 않습니다. `detached_history_release_energy_J`는 열린 접촉의 이력을 commit 시 제거한 에너지입니다. 필드별 정의는 [변경 기록](NEARFIELD_UPDATE.md)에 있습니다.

`next_step_rolling_projection_work_J`는 한 이동의 projection work이고, `rolling_plastic_dissipation_W`는 이를 timestep으로 나눈 값입니다. 비교용 mesh/direct Heun의 응력·이력은 현재 시점이며, 끝점 힘 일치 조건을 강제하지 않아 `contact_gap_consistency_m`은 NaN입니다.

```bash
python3 lammps_slurry/tools/summarize_bridge.py \
  runs/spheroid_ar2_phi0.24_n500_JOB_ID/history.csv --start-strain 5 \
  --blocks 5 --output runs/spheroid_ar2_phi0.24_n500_JOB_ID/viscosity_summary.json
```

위 경로의 `JOB_ID`는 실제 실행 폴더 이름으로 바꿉니다. 평균은 출력 순간의 값들을 단순 평균하지 않고 매 시간단계에 누적한 interval mean을 시간 가중합니다. 지정한 strain 창 안에 완전히 들어오는 구간만 사용합니다. 독립 요약 명령의 빈 창은 오류이며, 실행기의 자동 요약은 사용 가능한 평균이 없다고 기록합니다. 블록 평균의 상대 표준편차와 전·후반 차이를 출력하지만 정상상태를 자동으로 인증하지는 않습니다. `sigma_hydro_Pa`에는 계산기가 정의한 유체 응력을 이미 반영하므로 LAMMPS pair virial을 다시 더하지 않습니다.

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

## 직접 LAMMPS 입력을 작성할 때

`fix deform`를 `slurry/bridge`보다 먼저 정의하고 xz 전단에 `remap v`를 사용해야 합니다. 현재 구현은 `deform 0`과 일정 전단률을 요구합니다. 생성기는 이 조건을 자동으로 적용합니다.

```lammps
plugin load ${slurry_plugin}
# units/box/particles 설정 후:
fix shear all deform 0 xz erate 100 remap v units box
fix hydro all slurry/bridge 0.01 100 8e-6 0.001 30 contact 10 24 rolling 10 24 shape spheroid mobility_backend nearfield contact_scheme imex history history.csv 40 5
timestep 2.5e-6
```

필수 숫자 인자는 순서대로 `eta`, `shear_rate`, `nominal_diameter_m`, `epsilon`, `kn_N_per_m`입니다. `contact` 뒤에는 μs, kt를, `rolling` 뒤에는 μr, kr를 넣습니다. 현재 모델은 단분산 volume-equivalent diameter를 사용하며 각 입자가 해당 D와 AR 범위에 맞아야 합니다.

입력의 `${slurry_plugin}` 참조 자체에는 따옴표를 붙이지 않습니다. 실행기는 `-var` 값에 경로를 감싼 실제 큰따옴표를 넣어 공백이 있는 경로도 처리합니다. 일반 실행에는 위 launch script를 사용하면 됩니다.
