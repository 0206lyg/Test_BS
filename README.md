# Test_BS — spheroid suspension hydrodynamics


현재 실행 모델은 `fix slurry/bridge`입니다. **체적 등가 직경 D를 고정한 oblate spheroid**의 AR, 체적분율, sliding·rolling 마찰을 바꾸며 입자의 위치·배향과 xz 전단 응력을 계산합니다. 현재 실행 경로는 `PARTICLE_SHAPE=spheroid`, **1 ≤ AR ≤ 5**를 지원합니다. 기저나 저항표를 먼저 생성하지 않습니다.

현재 모델은 Mari의 무차원 lubrication regularization과 Singh의 sliding·rolling 접촉 법칙을 사용합니다. 기존 spheroid 근거리 bridge 및 주기 중거리 mobility는 유지하므로, 유체 연산자 전체가 Mari의 구형 입자 모델과 같다는 뜻은 아닙니다. **Graphite의 발산 농도 φJ≈0.30 또는 상대점도 100을 재현했다고 주장하는 변경이 아닙니다.**

실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다. 파일 상단의 `AR`, `PHI`, `COUNT`, `STEPS`, `OUTPUT_INTERVAL` 및 물성·solver 설정을 직접 고친 뒤 제출합니다. 실행 인자를 매번 입력할 필요가 없습니다.

## Overlay 적용과 실행

저장소 디렉토리에서 다음 명령으로 적용하고 **반드시 다시 빌드**합니다. 수정한 solver를 사용하려면 새 소스에서 plugin을 다시 만들어야 합니다. 이번 solver 변경은 기존 실행 입력 형식을 유지합니다.

```bash
unzip -o slurry_contact_stability_overlay_20261027.zip
python3 apply_contact_settings.py
sbatch build_slurry_cpu.sbatch
```

빌드 로그의 `Build complete`를 확인한 뒤 `run_slurry_cpu.sbatch` 상단 설정을 편집하고 제출합니다.

```bash
sbatch run_slurry_cpu.sbatch
```

기본값은 AR=2, φ=0.24, 입자 500개, 40000 step, 기록 간격 40 step입니다. 실행 결과는 기본적으로 `runs/spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`에 저장됩니다. 기존 실행 폴더는 덮어쓰지 않습니다.

ZIP은 기존 파일을 삭제할 수 없으므로, **성공한 빌드의 마지막 단계**에서 이전 `lammps_slurry/run_bridge_cpu.sbatch`와 `lammps_slurry/run_slurry_cpu.sbatch` 두 파일을 삭제합니다. 최상위의 새 `run_slurry_cpu.sbatch`를 사용합니다. 과거 flake 관련 소스와 검증 기록이 남아 있어도 현재 실행기의 flake 선택은 지원되지 않습니다.

## 입자 크기와 접촉 설정

`NOMINAL_DIAMETER_UM=8`은 두께나 장축 직경이 아니라 **체적 등가 직경**입니다. AR=a/c일 때 semiaxis는 다음과 같으며, AR을 바꿔도 입자 체적과 기준 반경은 고정됩니다.

\[
a=\frac{D}{2}\mathrm{AR}^{1/3},\qquad
c=\frac{D}{2}\mathrm{AR}^{-2/3},\qquad
V=\frac{\pi D^3}{6},\qquad a_{\rm ref}=D/2.
\]

`LUBRICATION_EPSILON=0.001`은 Mari의 **반경 정규화**를 사용합니다. 스칼라 lubrication gap은 `max(actual_gap,0)+epsilon*a_ref`이며 D=8 μm에서는 추가 정규화 길이가 **4 nm**입니다. 이것은 표면의 4 nm 두께 roughness층이나 접촉 시작 간격이 아닙니다. 실제 gap이 음수가 되어 입자가 약하게 겹칠 때 접촉 스프링과 마찰이 작용하며, 초기배치는 실제 gap ≥ 0을 인증합니다.

| 배치 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `NOMINAL_DIAMETER_UM` | 8 | 체적 등가 직경, μm |
| `LUBRICATION_EPSILON` | 0.001 | D/2로 정규화한 무차원 gap 추가량 |
| `FRICTION` | 10 | sliding Coulomb 계수 μs |
| `ROLLING_FRICTION` | 10 | rolling Coulomb 계수 μr |
| `NORMAL_STIFFNESS` | 0.3 | 법선 스프링 강성, N/m |
| `TANGENTIAL_STIFFNESS` | 0.24 | sliding 스프링 강성, N/m |
| `ROLLING_STIFFNESS` | 0.24 | rolling quasi-force 스프링 강성, **N/m** |

Rolling의 기준 길이도 `a_ref=D/2`를 사용합니다. `ROLLING_STIFFNESS`는 과거의 N·m/rad 값과 단위가 다르며, 현재 단분산 설정에서 대응하는 각도 스프링 강성은 `ROLLING_STIFFNESS*a_ref^2`입니다. 기존 두께, 절대 gap floor, 양의 접촉 shell, rough-hydrodynamics, rolling length, twisting 설정은 실행 옵션에서 제거했습니다.

μs=μr=10은 강한 마찰의 **시험 시작값**입니다. 포화한계나 측정된 graphite 계수로 확정한 값이 아닙니다. AR별 φJ를 비교하기 전에 마찰계수 증가에 대한 포화, timestep 및 법선·접선·rolling 강성에 대한 수렴, 실제 겹침 크기를 확인해야 합니다. Solver 잔차가 작다는 것만으로 유한 스프링의 물리적 오차나 정상상태 점도의 수렴이 보장되지는 않습니다.

## 계산과 병렬 실행

중거리는 ellipsoid 형상함수를 유지하는 particle–mesh FFT이며 FFTW를 사용합니다. 현재 배치와 예측 배치에서 고정된 접촉력으로 **선형 17N 힘·mobility 방정식을 각각 한 번씩** 풉니다. Heun predictor–corrector로 위치·배향·sliding·rolling 이력을 갱신하며, Newton 접촉 반복과 강성 continuation은 production에서 실행하지 않습니다. PARDISO는 희소 전처리기에 사용합니다.

선택한 수치 강성은 `kn=0.3`, `kt=kr=0.24 N/m`, 시간간격은 `2.5e-6 s`입니다. 마찰계수 `mu_s=mu_r=10`은 유지합니다. `apply_contact_settings.py`는 실행 스크립트의 다른 설정을 보존하며 기존 총 물리시간과 출력 시간 간격에 맞춰 `STEPS`와 `OUTPUT_INTERVAL`을 조정합니다. 기본 0.1 s 계산은 40000 step입니다.

AR1 N500의 기존 실패 시간 62 microseconds를 지나 120 microseconds까지 통과했고, 최대 겹침은 반지름의 약 0.0202%였습니다. 동일 1 thread 비교에서는 dt 1 microsecond 단일 선형 풀이보다 선택한 Heun 설정의 CPU 시간이 22.05% 적었습니다. 이는 짧은 검증 결과이며 장기 안정성 또는 정상상태 점도의 검증이 아닙니다. [변경·검증 기록](docs/CONTACT_STABILITY_UPDATE.md), [저항계수 차이](docs/HYDRO_COEFFICIENT_AUDIT.md)를 참고하십시오.

**전체 밀집 행렬을 만들어 직접 푸는 방식은 사용하지 않습니다.** 배치의 `MOBILITY_BACKEND=direct`는 검증용 Fourier 직접 합산을 뜻하며 밀집 직접 solver가 아닙니다. 기본값은 `mesh`입니다.

한 작업은 **MPI 프로세스 1개**와 OpenMP 스레드를 사용합니다. 기본 자원 요청은 물리 코어 16개, 메모리 96 GB이며 배치 상단의 `#SBATCH` 항목에서 변경합니다. launcher는 한 MPI 프로세스에 할당된 코어를 명시적으로 배정하고 바인딩을 출력합니다. 시작 실패 시 다른 실행 방식으로 재시도하거나 시뮬레이션을 중복 실행하지 않습니다.

프로덕션 빌드에는 GCC/OpenMP, FFTW 개발 파일과 **static oneMKL LP64/GNU-thread 라이브러리**가 필요합니다. `build_slurry_cpu.sbatch`는 Negishi의 `fftw`, `intel-mkl` 모듈을 추가로 불러옵니다. 사이트의 모듈 이름이 다를 때는 `SLURRY_FFTW_MODULE`, `SLURRY_MKL_MODULE`을 지정합니다. 필요한 최적화 라이브러리가 없으면 느린 backend로 자동 전환하지 않고 빌드에서 원인을 알립니다. MKL은 정적으로 링크하고 FFTW 등 연결 라이브러리는 실행별 private runtime에 보관합니다.

## 출력과 해석

기본 매질 점도는 0.01 Pa·s, 전단률은 100 s⁻¹, timestep은 2.5e-6 s입니다. `SPECTRAL=2`는 저비용 중거리 근사이며 4 또는 8과 비교하여 해상도 영향을 검사할 수 있습니다. `SPECTRAL_TOLERANCE=1e-8`은 Fourier 절단 허용값이며 전체 유체력 오차가 아닙니다.

`parameters.json`에는 D, a_ref, 실제 축 길이, lubrication 정규화와 rolling 강성 단위를 기록합니다. `history.csv`는 매질·유체·접촉 응력과 순간 점도, 매 시간단계에 누적한 평균, 소산, solver 잔차 및 비용을 출력합니다. `trajectory.lammpstrj`는 입자별 유체력·토크·stresslet과 접촉력·토크를 별도로 출력합니다. 평균창은 `AVERAGE_START_STRAIN`으로 지정합니다. `final.data`는 기하 저장이며 마찰 이력을 보존하는 restart가 아닙니다.

- [실행·출력·점도 평균](docs/BRIDGE_RUN.md)
- [Mari/Singh 변경의 과학적 범위와 확인 사항](docs/MARI_SINGH_OVERLAY.md)
- [연결식·단위·접촉·응력 구현 명세](docs/COMPACT_BRIDGE_MODEL.md)
- [주기 중거리의 근사와 해상도](docs/PERIODIC_MIDDLE.md)
- [이전 FFT·접촉 변경과 검증 기록](docs/FFT_CONTACT_UPDATE.md)

최대 겹침 5% 미만은 수치 강성 선택 기준입니다. AR1에서는 반지름, oblate spheroid에서는 가장 작은 반축을 기준으로 평가합니다. 농도·형상·응력 조건을 바꿀 때는 겹침뿐 아니라 강성과 시간간격에 대한 점도·접촉응력 의존성을 함께 확인해야 합니다. 이력 없는 geometry 파일은 물리적 restart가 아닙니다.

기존 검증 폴더의 결과는 해당 소스 버전과 조건에서 얻은 기록입니다. 짧은 실행 검사나 kernel 시간만으로 정상상태 점도 또는 전체 실행 속도를 판정하지 않습니다.
