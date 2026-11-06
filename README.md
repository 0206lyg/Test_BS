# Test_BS — practical 실행 안내

이 문서가 현재 빌드·실행·설정·정리 안내다. `run_slurry_practical_cpu.sbatch`를 직접 교체하며, 자동 법선 강성 조절과 개선된 solver가 기본으로 작동한다. 별도 `apply_*.py` 실행이나 SHA256 수동 검사는 필요 없다.

## 적용 및 실행

`build_slurry_cpu.sbatch`와 `lammps_slurry/`가 있는 저장소 루트에서 실행한다. 이 overlay는 Git `e2a05b08ae5e7e0422feaeb63e5bafe308c2f8db`의 소스에 적용한다. 실행 파일도 교체하므로 AR·농도 등 원하는 값은 교체된 파일 상단에서 편집한다.

```bash
unzip -DD -o slurry_kn_adapt_same_dt_overlay_20261106.zip
sbatch build_slurry_cpu.sbatch
```

빌드 로그에 `Build complete`가 출력된 뒤 실행한다.

```bash
sbatch run_slurry_practical_cpu.sbatch
```

기존 빌드 절차와 의존성은 유지한다. 빌드 시 GCC, OpenMPI, CMake, FFTW, static oneMKL 모듈을 사용한다. 기본 nearfield 시뮬레이션에서 FFT를 사용한다는 의미는 아니다. 각 시뮬레이션은 MPI 프로세스 하나이며, 기본 요청 자원은 CPU 2개·32 GB다.

## 실행 설정

설정은 모두 `run_slurry_practical_cpu.sbatch` 상단에 있다.

| 설정 | 기본값 |
| --- | --- |
| 입자 | oblate spheroid, AR=4, phi=0.24, N=1000 |
| 크기 | 평균 체적 등가 직경 8 μm, bidisperse 크기비 1.4 |
| 전단률·매질 점도 | 100 /s, 0.00089 Pa·s |
| 시간 간격·step 수 | 1 μs, 500000 step, 목표 strain 50 |
| 출력 간격·평균 시작 | 400 step, strain 5 |
| 초기 법선·접선·rolling 강성 | kn=900, kt=kr=300 N/m |
| sliding·rolling 마찰계수 | 각각 10 |
| 접착 에너지·범위 | 0.0212434877286055 J/m², 0.67 nm |
| background Hamaker 상수 | 9.9e-20 J |

AR 범위는 1–5다. 평균 체적과 요청한 phi를 유지하며, N1000·크기비 1.4에서는 작은 입자 733개와 큰 입자 267개를 사용한다. 초기배치 seed는 1729다.

인력이 없는 contact + lubrication baseline도 같은 실행 파일로 돌릴 수 있다. 이 경우 `COHESION_WORK=0`, `BACKGROUND_HAMAKER=0`으로 설정한다. 나머지 설정은 실행 파일에 지정한 값을 사용한다.

## 모델과 solver

유체 모델은 isolated spheroid drag와 pair lubrication이다. 기존 face/edge-face 보정과 공통 taper를 유지한다. Lubrication 정규화는 `epsilon=0.001`에 쌍 평균 체적 등가 반경을 곱한다. 기계적 접촉은 실제 signed gap이 음수일 때 활성화한다. Rolling 길이는 `2*r_i*r_j/(r_i+r_j)`이며 rolling 각도 강성은 `kr*rolling_length²`다.

법선·sliding·rolling spring은 IMEX로 풀고, 마찰 한계는 시작 상태의 법선 하중으로 고정한다. 인력·background potential과 기존 최종 힘 평형·끝점 기하 검사는 유지한다.

이번 변경은 다음 세 가지다.

1. **Overlap 제어:** 끝점 overlap이 작은 최소 반축의 5%를 넘으면 먼저 같은 dt에서 kn을 높여 다시 푼다. 3% 초과 시 다음 step의 kn을 미리 올리고 목표값은 2%다. 1% 미만 상태가 strain 0.02 동안 유지되면 초기 kn 쪽으로 낮춘다. 상한은 초기 kn의 1000배다. 조절 기준은 실행 파일의 `KN_TARGET`, `KN_RAISE`, `KN_LOWER`, `KN_RELAX_STRAIN`, `KN_MAX_FACTOR`다. 활성화 여부를 따로 지정할 필요는 없다.
2. **중간 선형 풀이 비용 절감:** Newton 방향을 구하는 중간 선형 풀이의 정확도를 잔차 감소에 맞춰 조절한다. 최종 물리 잔차 기준은 그대로다.
3. **같은 dt 실패 회복:** Newton 반복 초과 또는 line search 실패 시 마지막 유효 속도에서 최대 100회 ADMM 회복을 시도하고, 필요하면 Newton으로 마무리한다. 이 과정은 물리적 substep이 아니며 kn·dt·마찰 한계·입력 이력을 고정한다. 실패한 trial의 이력은 누적하지 않는다.

자동 조절은 kn만 바꾼다. kt, kr, 마찰계수와 인력 범위는 바꾸지 않는다. 강성 변경만으로 마찰 한계가 갑자기 증가하지 않도록 직전 수용 상태의 하중을 사용하고, 전처리기는 새 강성에 맞춰 갱신한다. 모든 solver 회복이 실패하거나 강성 상한에서도 overlap 제한을 만족하지 못하면 기존 제한적 dt 축소 경로가 남아 있다.

## 출력과 확인 범위

기본 출력은 `runs/practical_spheroid_ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`다. `history.csv`에는 응력·점도·overlap·반복 횟수·잔차·시간과 함께 `kn_N_per_m`, `kn_stiffening_retries_total`, `kn_adjustments_total`이 기록된다. `parameters.json`에는 입력 설정과 강성 제어 규칙이 저장된다. `final.data`는 기하 저장이며 접촉 이력까지 보존한 restart 파일은 아니다.

이 결합 solver의 강성 제어·접촉법칙·이력 검사와 kn=900/4580에서의 같은-dt 회복 검사를 통과했다. 이번 포장에서는 solver 소스를 다시 바꾸지 않았다. 교체 실행 파일의 인자 전달, 인력-off 설정, shell 문법과 소스 빌드 전처리 검사를 확인했다. 정상상태 점도나 N1000 목표 strain 50의 완료를 뜻하지 않는다. 서로 다른 과도 궤적의 응력이 일치해야 한다는 조건은 사용하지 않는다.

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
| 기존 baseline·sweep | `run_slurry_cpu.sbatch`는 기존 테스트와 `sweep_slurry.sh`가 참조하므로 유지한다. `sweep_slurry.sh`의 기본 template은 기존 baseline 실행 파일이며, practical 단일 실행에는 사용하지 않는다. |

SHA256는 파일 내용의 지문이다. 별도 `.sha256` 목록과 과거 overlay manifest는 실행 입력이 아니므로 위처럼 삭제해도 된다. `release.json`과 `bundle_guard.py` 내부의 소스/빌드 일치 검사는 기존 빌드가 사용하는 기능이므로 유지한다. 사용자가 checksum 명령을 따로 실행할 필요는 없다.
