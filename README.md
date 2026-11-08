# Test_BS — baseline 및 인력 suspension 실행 안내

## CMC별 0.1 s⁻¹ 시험: `run_cmc.sbatch`

현재 CMC 농도별 수동 인력 시험은 **`run_cmc.sbatch`**로 실행한다. `run_slurry_practical_cpu.sbatch`의 입자·마찰·인력 kernel·practical solver·강성 조절·허용오차를 그대로 사용하며, total CMC에서 계산한 **free CMC로 매질 점도만** 자동 설정한다. `COHESION_WORK`와 `BACKGROUND_HAMAKER`는 서로 독립적인 수동 입력이다. CMC 농도를 바꿔도 두 값을 자동 감쇠하지 않는다.

파일 상단에서 다음 세 값을 편집한 뒤 저장소 루트에서 제출한다. 아래 인력 수치는 practical launcher에서 유지한 기본값이며 4 g/L에 보정한 값이 아니다. `COHESION_WORK` 단위는 **J/m²**이므로, 예를 들어 2.199 mJ/m²를 시험하려면 `0.002199`를 입력한다.

```bash
CMC_TOTAL=4                         # total CMC, g/L of liquid; 0--16
COHESION_WORK=0.0212434877286055      # J/m², manual
BACKGROUND_HAMAKER=9.9e-20           # J, manual
```

```bash
sbatch run_cmc.sbatch
```

접촉+lubrication 기준 시험은 두 인력을 모두 `0`으로 지정한다. 별도 command-line 인자는 받지 않는다. `PREPARE_ONLY=1`은 기존 verified-build 입력 생성만 수행한다. 이번 추가 파일은 기존 solver, 입력 생성기, build fingerprint를 수정하지 않으므로 **이미 현재 practical solver를 빌드했다면 재빌드가 필요 없다.**

기본 전단률은 `SHEAR_RATE=0.1` s⁻¹이다. 기존 `TIMESTEP=1e-6` s와 평균 시작 strain 5를 유지하면서 총 strain 50까지 가도록 **`STEPS=500000000`**으로 지정했다. 물리 시간은 500 s이며, 기존 100 s⁻¹ 시험보다 step 수가 1000배다. `STEPS=500000`을 그대로 쓰면 총 strain 0.05여서 평균 구간에 도달하지 못한다. timestep은 자동 확대하지 않는다. 실행 길이는 `STEPS`로 직접 변경할 수 있으며 평균을 얻으려면 `abs(SHEAR_RATE)*TIMESTEP*STEPS > AVERAGE_START_STRAIN`이어야 한다. 기본 출력 간격 400 step도 유지하므로 전체 실행은 약 125만 개 출력 구간을 갖는다. 이 step 수는 목표 실행 길이이며 7일 walltime 내 완료를 보장하지 않는다.

### 농도 분배와 점도

출처: [ACS Nano, DOI 10.1021/acsnano.6c10201](https://doi.org/10.1021/acsnano.6c10201), 90k CMC. 모든 농도 단위는 **액체 부피 기준 g/L**이다. `CMC_TOTAL`은 투입량이며 매질 점도표에는 **free CMC**를 대입한다. 흡착량은 프로젝트에서 채택한 포화량 2.8 g/L와 total 4 g/L에서 adsorbed 2.7 / free 1.3 g/L를 연결하는 다음 명목 질량수지를 사용한다.

$$
c_t=c_a+c_f,\qquad c_a=\frac{q_{\max}c_f}{K_d+c_f},\qquad
q_{\max}=2.8,\quad K_d=1.3\left(\frac{2.8}{2.7}-1\right)=0.048148148\;\mathrm{g/L}.
$$

$$
c_f=\frac{c_t-q_{\max}-K_d+\sqrt{(q_{\max}+K_d-c_t)^2+4K_dc_t}}{2}.
$$

이는 각 농도의 흡착량을 직접 측정했다는 의미가 아니라, 채택한 ACS 흡착 기준값을 만족시키는 연속 보간 모델이다. 원문의 흡착 측정은 graphite+carbon black 시스템이며, 이를 현재 graphite 모델에 적용하는 명목 분배다. 작은 농도에서는 위 근의 수치 소거를 피하는 동치 식을 계산한다. 지원 범위는 total 0–16 g/L이며 범위 밖이나 NaN/Inf 입력은 실행 전 거부한다.

매질 점도는 **Fig. 2(c)의 Cross fitting에서 얻은 90k specific viscosity**를 `η_m = 0.89 × (1 + η_sp)` mPa·s로 변환한 표를 사용한다. 90k 매질을 Newtonian으로 근사하므로 전단률에 따라 추가 보정하지 않는다. 이전 Fig. 2(b) 점도표와 혼용하지 않는다. 그림 판독값이므로 아래 소수 자릿수는 계산 재현용이며 측정 정밀도를 뜻하지 않는다.

| Total CMC (g/L) | Adsorbed CMC (g/L) | Free CMC (g/L) | 매질 점도 (mPa·s) | solver 입력 (Pa·s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.890000 | 0.000890000 |
| 1 | 0.974305 | 0.025695 | 0.912784 | 0.000912784 |
| 2 | 1.898588 | 0.101412 | 0.979922 | 0.000979922 |
| 4 | 2.700000 | 1.300000 | 2.042713 | 0.002042713 |
| 5 | 2.741552 | 2.258448 | 2.428364 | 0.002428364 |
| 6 | 2.759012 | 3.240988 | 2.762316 | 0.002762316 |
| 7 | 2.768499 | 4.231501 | 3.172650 | 0.003172650 |
| 8 | 2.774436 | 5.225564 | 3.311921 | 0.003311921 |
| 10 | 2.781448 | 7.218552 | 4.567323 | 0.004567323 |
| 12 | 2.785445 | 9.214555 | 5.126602 | 0.005126602 |
| 16 | 2.789832 | 13.210168 | 7.233567 | 0.007233567 |

계산에 사용한 free CMC 점도 원표는 다음과 같다. 농도와 **절대 점도에 대해 구간별 선형 보간**하며 로그 보간하지 않는다. 0–1.5 g/L 구간은 물과 첫 판독점 사이의 보간이다. 4/5 및 8/9 g/L에서 판독값의 작은 역전을 제거하기 위해 각 쌍을 같은 가중치로 평균했다. 이 표는 total CMC가 아니라 순수 용액의 CMC 농도, 즉 현재 계산의 free CMC가 독립변수다.

| Free CMC (g/L) | 사용 점도 (mPa·s) |
| ---: | ---: |
| 0 | 0.890000000 |
| 1.5 | 2.220053693 |
| 3 | 2.632034186 |
| 4 | 3.172650056 |
| 5 | 3.172650056 |
| 6 | 3.790085754 |
| 7 | 4.467310797 |
| 8 | 4.924922244 |
| 9 | 4.924922244 |
| 10 | 5.864917005 |
| 12 | 6.498789916 |
| 16 | 8.927466454 |

변환은 `lammps_slurry/tools/cmc_medium.py`에서 수행한다. 출력 로그 `slurry_cmc_manual_<job ID>.out`에 total/adsorbed/free CMC, 매질 점도, 출처, 실제 수동 인력 입력을 남긴다. 기본 결과 폴더는 `runs/cmc<TOTAL>_spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>`이며, 기존 `parameters.json`과 `launch.json`에는 실제 전달한 점도와 인력이 기록된다. `parameters.json`의 `viscosity`는 **Pa·s**다. 조성 원문 기록은 sbatch 출력 로그에 있으므로 결과 폴더와 함께 보관한다. 실행 없이 계산만 확인하려면 다음을 사용한다.

```bash
python3 lammps_slurry/tools/cmc_medium.py --total 4
```

검증은 `python3 -m unittest discover -s lammps_slurry/tests -p 'test_cmc*.py' -v`로 실행한다. 새 매질 경로 7개와 기존 CMC 경로 15개가 통과했다. 농도별 점도, 두 인력의 독립 수동 입력, practical 인자 보존과 실제 `in.bridge` 입력 생성을 확인했다. 이 검증은 입력 생성 검증이며 0.1 s⁻¹의 장시간 유변 계산은 별도로 제출한다.

기존 `run_slurry_cmc_cpu.sbatch`는 이전 Fig. 2(b) 표와 자동 인력 감쇠식을 사용하는 별도 legacy 시험 파일이다. 이번 수동 인력 sweep에는 사용하지 않는다. 해당 파일과 [이전 모델 문서](docs/CMC_MINIMAL_MODEL.md)는 기존 시험 재현을 위해 유지한다.

## 기존 두 실행 파일

이 문서가 현재 빌드·실행·설정·정리 안내다. `run_slurry_cpu.sbatch`는 contact friction + lubrication baseline용, `run_slurry_practical_cpu.sbatch`는 인력이 있는 별도 시험용이다. 두 실행 파일은 합치지 않는다. 두 시험 모두 공통 solver의 practical 계산 경로와 자동 법선 강성 조절을 사용한다. 별도 `apply_*.py` 실행이나 SHA256 수동 검사는 필요 없다.

## 적용 및 실행

`build_slurry_cpu.sbatch`와 `lammps_slurry/`가 있는 저장소 루트에서 실행한다. 이번 baseline overlay는 Git `6bf0ac4bb8423003922dfc7a6a3861e4aef3e1e7`에 이미 포함된 개선 solver를 사용한다. 교체 파일은 baseline 실행 파일, 기존 테스트와 이 README 3개다. 인력 시험 실행 파일과 C++ solver는 변경하지 않는다. 기존 baseline 실행 파일을 교체하므로 사용자 설정은 적용 뒤 상단에서 편집한다.

```bash
unzip -DD -o slurry_baseline_practical_overlay.zip
sbatch run_slurry_cpu.sbatch
```

현재 Git의 개선 solver를 이미 빌드했다면 다시 빌드할 필요가 없다. 이전 solver 바이너리를 사용 중이면 먼저 `sbatch build_slurry_cpu.sbatch`를 실행하고 `Build complete` 뒤에 baseline을 제출한다. 이번 ZIP만으로 과거 Git의 solver 소스까지 갱신하지는 않는다.

인력 시험은 계속 별도 파일로 제출한다.

```bash
sbatch run_slurry_practical_cpu.sbatch
```

기존 빌드 절차와 의존성은 유지한다. 빌드 시 GCC, OpenMPI, CMake, FFTW, static oneMKL 모듈을 사용한다. 기본 nearfield 시뮬레이션에서 FFT를 사용한다는 의미는 아니다. 각 시뮬레이션은 MPI 프로세스 하나이며, 기본 요청 자원은 CPU 2개·32 GB다.

## 실행 설정

각 시험의 설정은 해당 실행 파일 상단에서 독립적으로 편집한다.

| 설정 | baseline: `run_slurry_cpu.sbatch` | 인력: `run_slurry_practical_cpu.sbatch` |
| --- | --- | --- |
| 입자 | AR=5, phi=0.34, N=1000 | AR=4, phi=0.24, N=1000 |
| 크기 | 평균 체적 등가 직경 8 μm, bidisperse 크기비 1.4 | 동일 |
| 전단률·매질 점도 | 100 /s, 0.01 Pa·s | 100 /s, 0.00089 Pa·s |
| 시간 간격·step 수 | 2.5 μs, 200000 step, 목표 strain 50 | 1 μs, 500000 step, 목표 strain 50 |
| 출력 간격·평균 시작 | 400 step, strain 5 | 동일 |
| 초기 법선·접선·rolling 강성 | kn=300, kt=kr=100 N/m | kn=900, kt=kr=300 N/m |
| sliding·rolling 마찰계수 | 각각 10 | 각각 10 |
| 접촉 인력 | 없음; 관련 입력 변수를 두지 않음 | 0.0212434877286055 J/m², 범위 0.67 nm |
| background potential | 없음; 관련 입력 변수를 두지 않음 | Hamaker 상수 9.9e-20 J |
| 최종 solver 상대 허용오차 | 1e-4 | 1e-4 |
| 중간 선형계 기준·Newton 반복 한도 | 1e-4, 100회; 실패 시 같은-dt ADMM 복구 추가 | 동일 |

Baseline은 인력 옵션을 전달하지 않으며 입력 생성기의 기본값이 두 인력을 비활성화한다. `surface_integrator practical`이라는 내부 옵션 이름은 계산 경로를 선택할 뿐 인력을 켜지 않는다. Baseline의 solver 허용오차는 이전 1e-6에서 1e-4로 바꿨으며, timestep과 step 수는 줄이거나 늘리지 않았다. 초기 kn만 100에서 300 N/m로 변경했고 kt와 kr는 각각 100 N/m로 보존했다. 실행 중에는 overlap에 따라 kn만 조절한다.

AR 범위는 1–5다. 평균 체적과 요청한 phi를 유지하며, N1000·크기비 1.4에서는 작은 입자 733개와 큰 입자 267개를 사용한다. 초기배치 seed는 1729다. 표의 baseline phi=0.34는 기존 Git의 실행 설정을 보존한 값이며, 아래 단기 검증의 phi=0.22와 구분한다.

## 모델과 solver

유체 모델은 isolated spheroid drag와 pair lubrication이다. 기존 face/edge-face 보정과 공통 taper를 유지한다. Lubrication 정규화는 `epsilon=0.001`에 쌍 평균 체적 등가 반경을 곱한다. 기계적 접촉은 실제 signed gap이 음수일 때 활성화한다. Rolling 길이는 `2*r_i*r_j/(r_i+r_j)`이며 rolling 각도 강성은 `kr*rolling_length²`다.

법선·sliding·rolling spring은 IMEX로 풀고, 마찰 한계는 시작 상태의 법선 하중으로 고정한다. 인력 시험의 potential은 유지하며, baseline에는 이를 추가하지 않는다. 두 경로 모두 끝점 접촉 응답과 지정된 허용오차에서의 힘·토크 평형을 검사한다.

두 실행 파일이 사용하는 공통 solver의 주요 기능은 다음과 같다.

1. **Overlap 제어:** 끝점 overlap이 작은 최소 반축의 5%를 넘으면 먼저 같은 dt에서 kn을 높여 다시 푼다. 3% 초과 시 다음 step의 kn을 미리 올리고 목표값은 2%다. 1% 미만 상태가 strain 0.02 동안 유지되면 초기 kn 쪽으로 낮춘다. 상한은 초기 kn의 1000배다. 조절 기준은 실행 파일의 `KN_TARGET`, `KN_RAISE`, `KN_LOWER`, `KN_RELAX_STRAIN`, `KN_MAX_FACTOR`다. 활성화 여부를 따로 지정할 필요는 없다.
2. **중간 선형 풀이 비용 절감:** Newton 방향을 구하는 중간 선형 풀이의 정확도를 잔차 감소에 맞춰 조절한다. 중간 선형계의 허용오차와 최종 힘·토크 평형 허용오차는 별개이며, 최종 판정에는 실행 파일의 `SOLVER_TOLERANCE`를 사용한다. Practical 경로는 전처리 행렬 재사용과 접촉 후보 처리 최적화도 활성화한다.
3. **같은 dt 실패 회복:** Newton 반복 초과 또는 line search 실패 시 마지막 유효 속도에서 최대 100회 ADMM 회복을 시도하고, 필요하면 Newton으로 마무리한다. 이 과정은 물리적 substep이 아니며 kn·dt·마찰 한계·입력 이력을 고정한다. 실패한 trial의 이력은 누적하지 않는다.

자동 조절은 kn만 바꾼다. 따라서 고정 kn의 유한 spring과 궤적·탄성 에너지가 완전히 같은 모델이라고 주장하지 않는다. kt, kr, 마찰계수와 인력 범위는 바꾸지 않는다. 강성 변경만으로 마찰 한계가 갑자기 증가하지 않도록 직전 수용 상태의 하중을 사용하고, 전처리기는 새 강성에 맞춰 갱신한다. 모든 solver 회복이 실패하거나 강성 상한에서도 overlap 제한을 만족하지 못하면 기존 제한적 dt 축소 경로가 남아 있다.

## 출력과 확인 범위

Baseline 기본 출력은 `runs/spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`, 인력 시험은 `runs/practical_spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`다. `history.csv`에는 응력·점도·overlap·반복 횟수·잔차·시간과 함께 `kn_N_per_m`, `kn_stiffening_retries_total`, `kn_adjustments_total`이 기록된다. `parameters.json`에는 입력 설정과 강성 제어 규칙이 저장된다. `final.data`는 기하 저장이며 접촉 이력까지 보존한 restart 파일은 아니다.

이번 baseline 변경에서는 실행 인자 전달, 인력 옵션 생략 시 실제 생성 입력에서 인력이 없는지, 기존 practical 실행 파일의 보존, 소스 검사와 기존 Python 테스트를 확인했다. 기존 bridge 테스트 31개(3개 환경 의존 검사 제외), bundle guard 테스트 8개, shell 문법·소스 검사가 통과했다. 실제 단기 계산 결과는 아래와 같다. 정상상태 점도나 N1000 목표 strain 50의 완료를 뜻하지 않는다. 서로 다른 과도 궤적의 응력이 일치해야 한다는 조건은 사용하지 않는다.

같은 매질 점도 0.01 Pa·s, dt=2.5 μs에서 비교했다. 기존/개선 경로 모두 최신 공통 C++ core를 사용하며 기존은 최종 허용오차 1e-6·Newton 한도 1000, 개선은 1e-4·100이다. 아래 시간은 초기화 포함 전체 실행 시간이며, 알고리즘과 수치 설정 변경을 함께 비교한 단기 결과다. 강성은 kn/kt/kr 순서다.

| 시험 | step 수 | 기존 경로 | 개선 경로 | 개선 경로 최대 끝점 overlap |
| --- | ---: | ---: | ---: | ---: |
| N100, AR5, phi=0.22, 300/100/100 | 200 | 5.684 s | 5.232 s | 0.00215% |
| N1000, AR4, phi=0.24, 300/100/100 | 5 | 10.503 s | 8.343 s | 0.11445% |
| N1000, AR4, phi=0.24, 900/900/900 | 3 | 10.401 s | 10.305 s | 0.09481% |

별도 N100, AR5, phi=0.22의 기존 조밀한 배치에서는 수용 끝점 overlap 3.1565%에 반응해 kn이 300→473.475 N/m로 증가했다. kt=kr=100을 유지하면서 다음 step overlap은 2.5510%, 10 step 뒤에는 0.7648%였다. 모든 단기 실행에서 오류·거부 trial·dt 분할이 없었고, 인력의 응력·에너지·power는 모두 0이었다. 새 경로의 최대 상대 잔차는 9.86e-5 이하로 지정한 1e-4 이내였다. 5% 초과 시 같은-dt 강성 재시도 경로는 이번 시험에서 발동하지 않았다.

N1000 시험과 별도 조밀한 N100 시험은 기존 geometry만 사용하고 양쪽의 접촉 이력을 초기화했다. 실제 이력을 이어받은 restart가 아니다. 초기 kn300 조건의 약21% 단축을 모든 조건의 보편적 개선율로 해석하지 않는다. all900에서는 시간 차이가 약1%로 작았다. 긴 strain에서의 안정성·정상상태 점도는 이번 짧은 검증 범위를 벗어난다.

## 기존 Git에서 정리할 파일

ZIP 덮어쓰기는 기존 파일을 삭제하지 않는다. 아래 명령은 이번 README로 대체된 안내, 사용하지 않는 과거 설정 도우미, 배포용 checksum 목록을 정리한다. 시뮬레이션 결과는 삭제하지 않는다.

```bash
rm -f -- README_KO.md README_PRACTICAL_KO.md README_UPDATE_KO.md \
  PRACTICAL_UPDATE.sha256 KN_ADAPT_UPDATE.sha256 \
  AR5_SOLVER_OVERLAY_MANIFEST.json BIDISPERSE_OVERLAY_MANIFEST.json \
  SOLVER_OVERLAY_MANIFEST.json SOURCE_MANIFEST.json STRONG_ROLLING_OVERLAY_MANIFEST.json \
  apply_bidisperse_settings.py apply_contact_settings.py apply_nearfield_settings.py apply_solver_limits.py \
  apply_kn_adapt_settings.py docs/KN_ADAPT_KO.md
```

`apply_kn_adapt_settings.py`, `KN_ADAPT_UPDATE.sha256`, `docs/KN_ADAPT_KO.md`는 이전 kn overlay를 풀었던 경우에만 있다. 기존 `test_bridge_tools.py`의 과거 설정 도우미 검사를 제거하고 새 실행 인자를 반영했다. 사용자가 실행 설정을 편집해도 테스트가 기본값을 강요하지 않도록 테스트 안에서만 비교용 값을 지정한다. 도우미 삭제 후에도 빌드가 해당 파일을 요구하지 않는다. CMake와 다른 테스트 파일은 유지한다.

| 분류 | 경로와 이유 |
| --- | --- |
| overlay 적용 후 삭제 가능한 과거 설정 스크립트 | `apply_bidisperse_settings.py`, `apply_contact_settings.py`, `apply_nearfield_settings.py`, `apply_solver_limits.py`: 실행 파일을 한 번 패치하던 도우미다. 이번 교체본은 이 파일들을 빌드·실행에서 요구하지 않는다. |
| 빌드에 불필요한 과거 결과 묶음 | `validation_ar5_recovery_20261028/`, `validation_bidisperse_20261028/`, `validation_contact_stability_20261027/`, `validation_nearfield_20261028/`, `validation_practical_20261102/`, 루트 `RESULT.json`. 과거 결과 보관 필요가 없을 때 삭제할 수 있다. |
| 유지 | `build_slurry_cpu.sbatch`, `lammps_slurry/tools/`의 빌드·실행 도구, `release.json`, `lammps_slurry/src/`, `vendor/`, CMake와 기존 테스트·fixture. |
| 현재 빌드 테스트 때문에 유지 | `diagnose_flake.py`는 `test_diagnose_flake.py`가 참조한다. 현재 상태에서 단독 삭제하면 전체 빌드 테스트가 실패한다. |
| 폴더 전체 삭제 금지 | `validation_basis/`의 `run_physical_gate.py`, `run_interpolation_gate.py`는 소스 검사·offline 도구가 참조한다. `validation_solver500/fixtures/ar2/packing.final`, `fixtures/ar20/packing.final`은 packing 테스트 입력이다. |
| 기존 baseline·sweep | `run_slurry_cpu.sbatch`는 baseline용 독립 실행 파일이며 `sweep_slurry.sh`의 template이다. 인력 시험은 `run_slurry_practical_cpu.sbatch`와 `sweep_practical_slurry.sh`를 사용한다. |

SHA256는 파일 내용의 지문이다. 별도 `.sha256` 목록과 과거 overlay manifest는 실행 입력이 아니므로 위처럼 삭제해도 된다. `release.json`과 `bundle_guard.py` 내부의 소스/빌드 일치 검사는 기존 빌드가 사용하는 기능이므로 유지한다. 사용자가 checksum 명령을 따로 실행할 필요는 없다.

