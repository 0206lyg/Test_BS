# Test_BS — spheroid suspension hydrodynamics

2026-10-28 AR5 solver 수렴 보강: [실패 원인·수정·검증 범위](docs/AR5_SOLVER_RECOVERY.md).

2026-10-28 bidisperse 확장: [두 크기 입자·초기 혼합·검증 범위](docs/BIDISPERSE_UPDATE.md).

이전 빌드 테스트 수정 기록: [기대값 누락 수정 및 packer 포함 검증](docs/BUILD_TEST_FIX_20261028.md).


현재 실행 모델은 `fix slurry/bridge`입니다. **입자 평균 체적을 고정한 단분산 또는 bidisperse oblate spheroid**의 AR, 체적분율, sliding·rolling 마찰을 바꾸며 입자의 위치·배향과 xz 전단 응력을 계산합니다. 현재 실행 경로는 `PARTICLE_SHAPE=spheroid`, **1 ≤ AR ≤ 5**를 지원합니다. 기저나 저항표를 먼저 생성하지 않습니다.

기본 유체 모델은 **`nearfield`: 단일 spheroid의 Stokes drag + 가까운 입자 사이 lubrication**입니다. 구형에서는 Mari의 squeeze·shear·pump 계수를 사용하며, 비구형에서는 실제 표면 간격과 곡률을 사용하는 확장 모델을 사용합니다. 단일 입자의 비등방 병진·회전 drag, Jeffery 회전과 전체 isolated stresslet은 유지합니다. 주기 전역 FFT 유체 연산은 기본 실행에서 제외했습니다. 이 변경의 목적은 AR별 유효 jamming 경향을 조사하는 비용을 줄이는 것이며, graphite의 φJ≈0.30 또는 특정 점도를 검증한 결과가 아닙니다.

실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다. 파일 상단의 `AR`, `PHI`, `COUNT`, `STEPS`, `OUTPUT_INTERVAL` 및 물성·solver 설정을 직접 고친 뒤 제출합니다. 실행 인자를 매번 입력할 필요가 없습니다.

## Overlay 적용과 실행

저장소 디렉토리에서 다음 명령으로 적용하고 **반드시 다시 빌드**합니다. ZIP에는 `run_slurry_cpu.sbatch`가 포함되지 않으므로 사용자가 편집한 실행 스크립트는 압축 해제 시 보존됩니다. 아래 helper는 크기 분포와 크기비 설정 및 이를 전달하는 실행 인자만 추가·갱신합니다.

```bash
unzip -DD -o slurry_bidisperse_overlay_20261028.zip
python3 apply_bidisperse_settings.py
sbatch build_slurry_cpu.sbatch
```

Helper는 첫 변경 전에 `run_slurry_cpu.sbatch.before_bidisperse_update`를 저장합니다. AR, 농도, 입자 수, 강성, timestep, step 수, 출력 간격, 경로, Slurm 자원은 바꾸지 않습니다. 다른 위치의 스크립트는 `--file /path/to/run_slurry_cpu.sbatch`로 지정합니다. 빌드 로그의 `Build complete`를 확인한 뒤 기존 실행 스크립트를 제출합니다.

```bash
sbatch run_slurry_cpu.sbatch
```

저장소 예제 설정은 AR=2, φ=0.24, 입자 1000개, 200000 step, 기록 간격 40 step입니다. 기본 전단률 100 s⁻¹와 timestep 2.5e-6 s에서 목표 strain은 50입니다. 사용자의 기존 설정은 helper 적용 후에도 유지됩니다. 실행 결과는 기본적으로 `runs/spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`에 저장됩니다. 기존 실행 폴더는 덮어쓰지 않습니다.

ZIP은 기존 파일을 삭제할 수 없으므로, **성공한 빌드의 마지막 단계**에서 이전 `lammps_slurry/run_bridge_cpu.sbatch`와 `lammps_slurry/run_slurry_cpu.sbatch` 두 파일을 삭제합니다. 최상위의 새 `run_slurry_cpu.sbatch`를 사용합니다. 과거 flake 관련 소스와 검증 기록이 남아 있어도 현재 실행기의 flake 선택은 지원되지 않습니다.

## 입자 크기와 접촉 설정

`NOMINAL_DIAMETER_UM=8`은 두께나 장축 직경이 아니라 **평균 입자 체적에 대응하는 직경**입니다. 즉, `mean(D_i^3)=D_nom^3`를 유지합니다. 단분산에서는 모든 입자의 체적 등가 직경이 D_nom이고, bidisperse에서는 두 집단에 다른 직경을 부여합니다. 각 입자의 축은 다음과 같습니다.

\[
a_i=\frac{D_i}{2}\mathrm{AR}^{1/3},\qquad
c_i=\frac{D_i}{2}\mathrm{AR}^{-2/3},\qquad
V_i=\frac{\pi D_i^3}{6}.
\]

기본 크기비는 `D_large/D_small=1.4`이고 두 집단의 총 고체 부피를 가깝게 맞춥니다. N=1000이면 작은 입자 733개, 큰 입자 267개입니다. D_nom=8 μm일 때 실제 직경은 각각 약 7.04282 μm와 9.85995 μm입니다. 두 집단에 같은 AR을 적용하므로 이 개수비는 AR에 따라 바뀌지 않습니다. 정수 개수 때문에 두 집단의 부피가 정확히 같지는 않지만 전체 입자 체적과 요청한 φ는 유지합니다.

초기배치는 희박한 상태에서 공간 구역별 종 조성을 맞추고, 구역·삽입 순서·구역 내 위치·배향을 seed로 무작위화합니다. 이후 입자 크기를 함께 키우며 실제 크기를 사용하는 겹침 완화와 주기 경계 전체의 표면 간격 인증을 수행합니다. 구역은 최종 위치를 구속하지 않으며, 이 과정은 물리적 평형화가 아닙니다.

`LUBRICATION_EPSILON=0.001`의 gap 정규화 길이는 각 쌍의 체적 등가 반경 `r_i=D_i/2`, `r_j=D_j/2`를 사용한 `epsilon*(r_i+r_j)/2`입니다. 따라서 D_nom=8 μm에서 4 nm는 기준값이며, bidisperse의 실제 정규화 길이는 쌍 종류에 따라 달라집니다. 접촉은 실제 gap이 음수일 때 활성화하고, 초기배치는 gap ≥ 0을 인증합니다.

| 배치 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `SIZE_DISTRIBUTION` | bidisperse | `bidisperse` 또는 `monodisperse` |
| `SIZE_RATIO` | 1.4 | 큰 입자/작은 입자의 선형 크기비 |
| `NOMINAL_DIAMETER_UM` | 8 | 평균 입자 체적에 대응하는 직경, μm |
| `LUBRICATION_EPSILON` | 0.001 | 쌍 평균 체적 등가 반경으로 정규화한 gap 추가량 |
| `FRICTION` | 10 | sliding Coulomb 계수 μs |
| `ROLLING_FRICTION` | 10 | rolling Coulomb 계수 μr |
| `NORMAL_STIFFNESS` | 30 | 법선 스프링 강성, N/m |
| `TANGENTIAL_STIFFNESS` | 24 | sliding 스프링 강성, N/m |
| `ROLLING_STIFFNESS` | 24 | rolling quasi-force 스프링 강성, **N/m** |

Rolling 길이는 쌍별 `2*r_i*r_j/(r_i+r_j)`이고 대응하는 각도 스프링 강성은 `ROLLING_STIFFNESS*rolling_length^2`입니다. 단분산에서는 기존 `r_i=r_j=D/2` 식으로 돌아갑니다. 서로 다른 크기는 nearfield 경로에서 지원하며 mesh/direct는 단분산 비교용입니다.

μs=μr=10은 강한 마찰의 **시험 시작값**입니다. 포화한계나 측정된 graphite 계수로 확정한 값이 아닙니다. AR별 φJ를 비교하기 전에 마찰계수 증가에 대한 포화, timestep 및 법선·접선·rolling 강성에 대한 수렴, 실제 겹침 크기를 확인해야 합니다. Solver 잔차가 작다는 것만으로 유한 스프링의 물리적 오차나 정상상태 점도의 수렴이 보장되지는 않습니다.

## 계산과 병렬 실행

기본 `nearfield`는 **희소 6N 저항·접촉 방정식**과 `contact_scheme imex`를 사용합니다. 법선·sliding·rolling spring은 implicit하게 갱신하고 마찰 상한은 step 시작 시점의 실제 법선 하중으로 고정합니다. 실제 끝점 gap은 별도 geometry 보정으로 맞춥니다. 이 고정 상한 문제의 볼록 목적함수를 이용해 Newton 반복을 안정화하며, 접촉 Jacobian의 SPD LDLT 분해를 재사용하는 FGMRES와 SPD LDLT 직접 fallback을 사용합니다. 매 선형계의 현재 Jacobian에 대한 실제 잔차를 검사하므로 preconditioner 재사용이 힘 방정식을 바꾸지는 않습니다. 전체 11성분 관계의 힘·토크·stresslet을 복원하며 runtime FFT와 전역 mobility 적용은 없습니다.

마찰 상한의 하중 지연은 시간 이산화 근사입니다. 끝점 Coulomb 상한에 대한 초과량과 하중 지연을 CSV에 기록하며, 힘 균형을 깨는 사후 clipping은 하지 않습니다. 완전 결합 끝점 Coulomb을 사용하는 `contact_scheme coupled`는 비교용입니다. 이 후보가 AR5에서 strain 0.11275에 멈춘 결과와, explicit Heun의 실패는 [변경 기록](docs/NEARFIELD_UPDATE.md)에 구분해 남깁니다.

기본 설정은 `kn=30`, `kt=kr=24 N/m`, timestep `2.5e-6 s`, `mu_s=mu_r=10`입니다. 다음 수치는 이전 단분산 릴리스의 검증이며 새 bidisperse 검증과 구분합니다. **단분산 AR5·N500·φ=0.24는 strain 0.25까지 1000 step을 완료했고 최대 겹침은 최소 반축의 2.374%였습니다.** AR1·AR2는 각각 200 step을 통과했습니다. 실제 검증 범위와 timestep 비교는 [변경 기록](docs/NEARFIELD_UPDATE.md)에 있습니다. AR5 겹침 비교에서 `kn=3`과 `10 N/m`은 5% 기준을 넘었기 때문에 강성을 높였습니다. Timestep은 자동으로 반비례 축소하지 않으며, helper도 사용자의 timestep을 변경하지 않습니다. 예제의 200000 step과 `2.5e-6 s`는 총 0.5 s, 전단률 100 s⁻¹에서 strain 50입니다. 새 bidisperse 및 고강성 시험 결과는 [이번 변경 기록](docs/BIDISPERSE_UPDATE.md)을 확인합니다.

이전 단분산 릴리스의 동일 장비·1 thread 시험에서 첫 20 step 평균 비용은 AR1에서 13.6847→0.072855 s, AR5에서 8.58493→0.242320 s였습니다. 이전 mesh+Heun·kn=0.3과 새 nearfield+IMEX·kn=30의 전체 설정 비교입니다. AR5 1000 step 전체는 597.52 s였으며 이 수치를 strain 10의 예상 시간으로 외삽하지 않습니다.

Pair adapter와 희소 solver의 독립 검사는 힘·토크 균형, stress 부호, 소산, 구형 한계 및 isolated spheroid의 Jeffery 회전을 확인합니다. 이 결과를 N500 장기 안정성이나 정상상태 점도 검증으로 확대하지 않습니다. 새 모델의 범위와 검증 상태는 [nearfield 변경 기록](docs/NEARFIELD_UPDATE.md)에 정리합니다. [이전 접촉 안정성 기록](docs/CONTACT_STABILITY_UPDATE.md)의 시간·점도 수치는 이전 유체 모델에 해당합니다.

`MOBILITY_BACKEND=nearfield`가 기본값입니다. 이전 `mesh`와 `direct`는 비교용으로 남아 있으며 compact bridge·주기 hydrodynamics와 explicit Heun을 사용합니다. 여기서 `direct`는 Fourier 직접 합산을 뜻하며 nearfield의 희소 직접 풀이와 별개의 옵션입니다. 어느 경로도 전체 밀집 행렬을 구성하는 방식으로 전환하지 않습니다.

한 작업은 **MPI 프로세스 1개**를 실행합니다. 기본 nearfield의 geometry와 희소 분해는 직렬 경로이며, 코어를 많이 요청한 만큼 빨라지는 구조가 아닙니다. 사용자의 기존 자원 요청은 그대로 보존합니다. 저장소 예제의 물리 코어 16개·96 GB는 이전 경로와의 호환 설정이며 nearfield의 최소 자원 요구를 뜻하지 않습니다. Launcher는 바인딩을 출력하고, 시작 실패 시 다른 실행 방식으로 재시도하거나 시뮬레이션을 중복 실행하지 않습니다.

기존 mesh/direct 경로를 함께 빌드하므로 현재 빌드 절차에는 GCC/OpenMP, FFTW 개발 파일과 **static oneMKL LP64/GNU-thread 라이브러리**가 필요합니다. 이것은 기본 nearfield가 실행 중 FFT나 PARDISO를 사용한다는 뜻이 아닙니다. `build_slurry_cpu.sbatch`는 Negishi의 `fftw`, `intel-mkl` 모듈을 불러옵니다. 사이트의 모듈 이름이 다를 때는 `SLURRY_FFTW_MODULE`, `SLURRY_MKL_MODULE`을 지정합니다. MKL은 정적으로 링크하고 FFTW 등 연결 라이브러리는 실행별 private runtime에 보관합니다.

## 출력과 해석

기본 매질 점도는 0.01 Pa·s, 전단률은 100 s⁻¹, timestep은 2.5e-6 s입니다. `SPECTRAL`과 `SPECTRAL_TOLERANCE`는 nearfield 계산에는 쓰지 않습니다. `SOLVER_TOLERANCE`는 실제 힘·mobility 관계의 잔차 검증에 적용합니다. Newton·선형 반복과 실제 잔차를 기록하며 마찰 하중 지연은 별도 진단으로 남깁니다.

`parameters.json`에는 기준 D, 크기 집단별 개수·실제 직경·축 길이·부피비(`size_populations`), 초기·최종 혼합 진단, lubrication 정규화와 rolling 강성 단위를 기록합니다. `history.csv`는 매질·유체·접촉 응력과 순간 점도, 매 시간단계에 누적한 평균, 소산, solver 잔차 및 비용을 출력합니다. `trajectory.lammpstrj`는 입자별 유체력·토크·stresslet과 접촉력·토크를 별도로 출력합니다. 평균창은 `AVERAGE_START_STRAIN`으로 지정합니다. `final.data`는 기하 저장이며 마찰 이력을 보존하는 restart가 아닙니다.

- [실행·출력·점도 평균](docs/BRIDGE_RUN.md)
- [현재 bidisperse 설정·초기 혼합·검증 상태](docs/BIDISPERSE_UPDATE.md)
- [nearfield 모델과 이전 단분산 검증 기록](docs/NEARFIELD_UPDATE.md)
- [이전 Mari/Singh 접촉 변경 기록](docs/MARI_SINGH_OVERLAY.md)
- [비교용 compact bridge 구현 명세](docs/COMPACT_BRIDGE_MODEL.md)
- [비교용 주기 중거리의 근사와 해상도](docs/PERIODIC_MIDDLE.md)
- [이전 FFT·접촉 변경과 검증 기록](docs/FFT_CONTACT_UPDATE.md)

최대 겹침 5% 미만은 수치 강성 선택 기준입니다. 각 접촉 쌍에서 더 작은 최소 반축을 분모로 사용하며, AR1에서는 두 입자 중 작은 반경입니다. `max_overlap_fraction_min_semiaxis`는 이 쌍별 비율의 최댓값입니다. Bidisperse에서는 `max_overlap_m/minimum_semiaxis_m`과 반드시 같지는 않습니다. 농도·형상·응력 조건을 바꿀 때는 겹침뿐 아니라 강성과 시간간격에 대한 점도·접촉응력 의존성을 함께 확인해야 합니다. 이력 없는 geometry 파일은 물리적 restart가 아닙니다.

기존 검증 폴더의 결과는 해당 소스 버전과 조건에서 얻은 기록입니다. 짧은 실행 검사나 kernel 시간만으로 정상상태 점도 또는 전체 실행 속도를 판정하지 않습니다.
