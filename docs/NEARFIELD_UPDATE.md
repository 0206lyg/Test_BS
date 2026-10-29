# Nearfield 유체 모델 변경 — 2026-10-28

기본 backend를 `nearfield`로 바꿉니다. 단일 spheroid의 isolated Stokes 응답과 가까운 입자 사이 lubrication을 사용하여 AR별 상대적 jamming 경향을 조사하는 계산 비용을 줄이는 변경입니다. 기본 접촉 적분은 `contact_scheme imex`입니다. 법선·sliding·rolling spring은 implicit하게 갱신하고, 마찰 상한에는 step 시작 시점의 실제 법선 하중을 사용합니다. 비교용 `coupled`는 끝점 Coulomb 조건을 연립하며, mesh/direct는 이전 Heun을 유지합니다.

## 유지·생략하는 물리

| 구성 | 현재 nearfield |
| --- | --- |
| 단일 입자 | 실제 spheroid의 비등방 병진·회전 drag, Jeffery strain torque, 전체 isolated stresslet |
| 구형 pair | Mari leading squeeze·shear·pump; normal `1/(h+epsilon)`, tangential logarithm |
| 비구형 pair | 실제 closest gap·곡률·지렛팔을 사용하는 `SlurrySpheroidHydro` 확장 |
| 접촉 | 실제 gap<0의 법선 spring, sliding·rolling spring과 Coulomb 제한 |
| 생략 | 주기 전역 many-body mobility, FFT 유체장, 이전 compact bridge finite-gap 모델 |

Mari의 저항 모델은 one-body drag와 sparse pair lubrication을 합하고 full long-range hydrodynamics를 생략합니다. Squeeze·shear·pump를 포함하며 비발산 twist mode는 생략합니다. Singh은 이 계열의 구형 입자 모델에 sliding·rolling 접촉 제약을 사용합니다. 두 논문이 비구형 입자에 대한 현재 계수를 검증한 것은 아닙니다.

- Mari et al., *J. Rheol.* 58, 1693 (2014), §II.1.2, Appendix A: <https://arxiv.org/html/1403.6793v4>
- Singh et al., *Phys. Rev. Lett.* 124, 248005 (2020), method 및 supplement: <https://arxiv.org/pdf/2002.10996v2>
- 비교한 공개 LF_DEM 코드: `ryseto/LF_DEM` commit `00fa7cc094d6fe4c459faa478116defd3468a33d`. [기본 설정](https://github.com/ryseto/LF_DEM/blob/00fa7cc094d6fe4c459faa478116defd3468a33d/LF_DEM/SimulationInit.cpp#L954-L962), [outer cutoff](https://github.com/ryseto/LF_DEM/blob/00fa7cc094d6fe4c459faa478116defd3468a33d/LF_DEM/System.cpp#L2665-L2673). 공개 코드의 default와 논문의 개별 실행 설정은 구분합니다.

`a_ref=D/2`를 고정합니다. 스칼라 lubrication gap은 `max(h,0)+0.001*a_ref`이고, 구형 pair의 outer surface gap은 `0.5*a_ref`입니다. 두 길이는 서로 다른 역할을 합니다. 비구형은 `min(0.5*a_ref, 0.5*R_ref)` 범위를 사용하며 `R_ref`는 pair의 local curvature로 계산합니다. 양의 접촉 shell을 추가하지 않습니다.

비구형 kernel은 leading cubic-surface coupling을 유지하고, 양의 소산을 위해 기존 normal-square completion을 사용합니다. 이것은 명시적인 근사이며, 이전 compact bridge의 11D Schur completion과 다릅니다. 구형과 달리 이 확장 모델의 AR별 정량 점도 또는 발산 농도는 별도 검증 대상입니다.

## 저항 풀이와 응력

Nearfield는 입자당 병진·회전 6성분 속도를 다음 힘 균형으로 풉니다.

\[
P^T(R_{\rm self}+R_{\rm near})(P\,u+q_E)
-F_{\rm ext}-F_{\rm contact}(u,\Delta t;N_n)=0.
\]

`q_E`에는 prescribed `-E`의 STF5 성분이 들어갑니다. 기본 IMEX의 법선 하중은 실제 이동 끝점 gap과 일치하도록 보정합니다. Sliding·rolling 탄성 이력도 implicit하게 갱신하되, 각 마찰 상한은 **원래 step 시작 상태의** `N_n=kn*max(-h_n,0)`에서 한 번 계산해 고정합니다. Geometry 보정 중에 임시 gap으로 이 상한을 다시 계산하지 않습니다. 고정된 기하에서 이 문제는 강볼록 에너지 최소화이며, 접촉 Jacobian은 대칭 양의 준정부호입니다.

Newton 방향은 명시적인 sparse 6N 행렬에 대한 FGMRES로 계산합니다. 기본 preconditioner는 첫 Newton 반복의 접촉 Jacobian을 SPD LDLT로 분해한 것입니다. 이후 반복에서 이를 재사용하고, 이전 선형 반복이 40회를 넘으면 갱신합니다. 매번 현재 Jacobian에 대한 실제 잔차를 확인하므로 오래된 분해를 쓰더라도 풀어야 할 힘 방정식은 바뀌지 않습니다. Hydrodynamic LDLT preconditioner도 비교용으로 남깁니다. 선형계의 실제 잔차가 기준을 통과하지 못하면 SPD LDLT 직접 풀이로 전환합니다. 볼록 목적함수의 감소와 최종 물리적 힘·토크 잔차를 검사합니다. 이 방향 계산에는 FFT나 전역 mobility 연산이 없습니다. Isolated body-frame response는 축 길이·점도·구적 차수별로 캐시하고 현재 orientation으로 회전합니다.

이 시간 이산화는 **끝점 법선 하중을 쓰는 완전 결합 Coulomb 조건과 동일하지 않습니다.** 접촉이 새로 생기는 step에서는 마찰이 다음 step부터 켜지고, 열리는 step에는 기존 하중에 해당하는 마찰력이 남을 수 있습니다. 그 힘을 풀이 후 잘라내면 힘 균형이 깨지므로 자르지 않습니다. 실제 끝점에서 접촉이 열렸으면 다음 step에 넘길 탄성 이력만 제거하고 방출 에너지를 기록합니다. 하중 지연 오차는 CSV에 측정하며, timestep 의존성 비교로 영향을 확인합니다.

끝점의 법선 하중까지 Coulomb 상한에 넣는 `contact_scheme coupled`는 비교용으로 남아 있습니다. 이 경로의 Jacobian은 비대칭이며 FGMRES와 직접 LU fallback을 사용합니다. 실제 AR5 실행에서 마찰 경계에 걸려 실패한 기록이 있으므로 기본값으로 사용하지 않습니다. 두 적분법 모두 끝점 geometry 일치 보정을 사용합니다.

속도를 얻은 뒤 전체 11성분 moment를 복원합니다. 물리 stress는 strain-conjugate moment 합의 음수이며, pair lubrication stress를 별도 virial로 다시 더하지 않습니다. 접촉 virial과 solvent stress는 각각 한 번씩 합합니다. Pair adapter는 uniform strain을 두 입자 strain의 평균으로 표현하므로, production의 동일 imposed strain에서 기존 kernel의 힘·토크·총응력·소산을 보존합니다.

Nearfield에는 runtime FFT와 전역 mobility 적용이 없습니다. 접촉과 endpoint geometry에는 반복이 필요하며, geometry와 희소 분해는 직렬 경로입니다. 기존 Slurm 코어 요청을 그대로 두어도 그 코어 수만큼의 가속을 보장하지 않습니다. 비교용 `mesh`·`direct` backend는 남아 있어 빌드 시 FFTW·oneMKL 의존성은 유지됩니다.

## 기존 실행 스크립트에 적용

Overlay ZIP은 `run_slurry_cpu.sbatch`를 포함하지 않습니다. 압축 해제로 사용자 스크립트를 덮어쓰지 않으며, 다음 helper가 정확히 네 assignment만 변경합니다.

```bash
unzip -DD -o slurry_nearfield_overlay_20261028.zip
python3 apply_nearfield_settings.py
sbatch build_slurry_cpu.sbatch
# 빌드 완료 확인 후:
sbatch run_slurry_cpu.sbatch
```

| Assignment | 적용값 |
| --- | --- |
| `MOBILITY_BACKEND` | `nearfield` |
| `NORMAL_STIFFNESS` | `30` N/m |
| `TANGENTIAL_STIFFNESS` | `24` N/m |
| `ROLLING_STIFFNESS` | `24` N/m |

Timestep, `STEPS`, `OUTPUT_INTERVAL`, AR, 농도, 입자 수, 마찰계수, 경로 및 자원은 유지합니다. 각 assignment가 정확히 한 번 없으면 helper가 오류를 내며 파일을 쓰지 않습니다. 첫 수정 전 `run_slurry_cpu.sbatch.before_nearfield_update`를 저장합니다. `--file`로 다른 스크립트를 지정할 수 있습니다. Step 수를 재조정하던 과거 `apply_contact_settings.py`와 혼용하지 않습니다.

현재 예제와 검증에 사용한 timestep은 이전과 같은 `2.5e-6 s`입니다. 강성은 실제 AR5 겹침 비교에 따라 선택하며, timestep을 강성에 반비례해 자동 축소하지 않습니다. Helper가 이 값을 강제하지 않으므로 기존 파일의 `TIMESTEP`을 확인합니다. Rolling stiffness 단위는 N/m이고 실제 각도 spring 강성은 `kr*a_ref^2` N·m/rad입니다.

## 출력 시간과 마찰 상한 진단

`time_integrator=imex_frozen_cap`가 기본 실행을 표시합니다. `force_geometry_time_s=t_n`은 유체 저항·접촉 방향을 평가한 기하 시점이고, `force_history_time_s=t_n+dt`는 갱신한 spring 이력의 시점입니다. 응력은 이 step의 힘 균형에 사용한 힘으로 계산합니다. `actual_endpoint_overlap_m`과 `actual_endpoint_max_overlap_fraction_min_semiaxis`은 실제 이동 끝점의 겹침입니다.

| CSV 필드 | 의미 |
| --- | --- |
| `friction_cap_time_s` | IMEX 마찰 상한을 정한 시작 시점 `t_n` |
| `max_normal_cap_lag_N` | 접촉별 `abs(N_n-N_end)`의 최댓값 |
| `normal_cap_lag_l1_fraction` | 하중 차이 합 / 시작·끝점 하중 합 중 큰 값 |
| `max_sliding_cap_excess_N` | step 힘이 끝점 Coulomb 상한을 초과한 양의 최댓값 |
| `max_rolling_cap_excess_N_m` | rolling moment의 끝점 상한 초과량 |
| `detached_history_release_energy_J` | 열린 접촉의 이력을 commit 시 제거한 탄성 에너지 |

끝점 상한 초과량은 숨기거나 사후 clipping하지 않습니다. IMEX가 의도한 하중 지연 오차를 확인하는 값이며, 큰 오차가 지속되면 그 조건의 점도·접촉 통계에 대한 timestep 의존성을 확인해야 합니다.

## 검증 상태

Kernel·선형계 검사는 다음 범위를 다룹니다. 장기 정상상태 점도의 검증을 대신하지 않습니다.

- `test_nearfield_hydro.cpp`: AR1/2/5, 서로 다른 크기의 구, SI·무차원 길이, 얕은 overlap에서 기존 kernel의 힘·토크·총응력·소산과 grand22 adapter 비교; reciprocity, PSD, strain 미분, cutoff 확인.
- `test_nearfield_solver.cpp`: full11 dense reference와 sparse6N force balance, 접촉 힘·토크 부호, factor cache 무효화, isolated spheroid drag·Jeffery 회전·stress 비교.
- `test_nearfield_imex.cpp`와 `test_nearfield_step.cpp`: 고정 마찰 상한, 접촉 미분·소산 및 끝점 geometry/contact 일치 검사. 최종 실행 결과는 아래 표에 함께 기록합니다.

**폐기한 기본 적분 후보의 실패:** Explicit nearfield Heun은 `kn=3`, `kt=kr=2.4 N/m`, `dt=2.5e-6 s`의 유한 preload 시험에서 탄성 에너지가 한 단계에 151–7088배 증폭했습니다. 실제 AR5·N500 실행도 step 138, strain 0.0345에서 실패했습니다. 완전 결합 backward Euler 후보는 AR5에서 450 step을 통과했지만 다음 step, strain 0.11275에서 마찰 경계 반복이 멈췄습니다. 당시 최대 겹침은 최소 반축의 약 1.24%였으므로 겹침 5% 초과로 종료된 경우가 아닙니다. 이 두 결과를 현재 IMEX의 성공 기록에 합산하지 않습니다.

**IMEX 강성 선택 비교:** 같은 `dt=2.5e-6 s`, AR5·N500에서 `kn=3 N/m`은 strain 약 0.20에서 최소 반축 대비 겹침 5%를 넘었고, strain 0.3395에서 검사를 의도적으로 중단했습니다. `kn=10 N/m`은 1000 step, strain 0.25까지 solver 오류 없이 완료했지만 최종 preconditioner를 사용한 비교에서 최대 겹침이 6.40844%였습니다. 두 경우는 접촉 풀이 실패가 아니라 요청한 겹침 기준을 만족하지 못한 경우입니다. 따라서 timestep은 유지하고 `kn=30`, `kt=kr=24 N/m`을 기본 설정으로 선택했습니다. 이 값에서 AR5의 strain 0.25까지 최대 겹침은 2.37379%였습니다.

최종 소스의 C++ 핵심 검사 10개, Python 검사 24개와 bundle 검사 7개가 통과했습니다. 빌드 테스트 수정 후 compiled packer를 연결하여 Python 검사를 모두 실행했으며, 건너뛴 검사는 없습니다. 실제 입자 실행은 모두 N500, φ=0.24, D=8 μm, 매질 점도 0.01 Pa·s, 전단률 100 s⁻¹, μs=μr=10, `kn=30`, `kt=kr=24 N/m`, **기존과 같은 `dt=2.5e-6 s`**입니다.

| AR | 완료 step / strain | 최대 겹침 / 최소 반축 | 최대 상대 solver 잔차 |
| --- | --- | --- | --- |
| 1 | 200 / 0.05 | 0.00122469% | 1.724e-10 |
| 2 | 200 / 0.05 | 0.00270233% | 2.741e-10 |
| 5 | 1000 / 0.25 | 2.37379% | 9.959e-8 |

세 실행 모두 FFT·전역 mobility 호출 수는 0입니다. AR5 1000 step의 전체 wall time은 597.52 s, 완료 step 내부 시간 평균은 0.59626 s였습니다. 마지막 상대 겉보기 점도는 256.475였으며, 탄성 하중이 증가하는 초기 과도 상태의 값입니다. 이를 정상상태 점도나 jamming 판정으로 사용하지 않습니다.

같은 AR5 초기배치와 강성에서 timestep만 바꿔 strain 0.05를 비교했습니다. 이 검사를 위해 생산 timestep을 변경하지 않았습니다.

| timestep (μs) | strain 0.05의 상대 겉보기 점도 | 평균 마찰 하중 지연 비율, strain 0.01–0.05 |
| --- | --- | --- |
| 2.5 | 9.67404 | 3.72164% |
| 1.25 | 9.73885 | 1.69674% |
| 0.625 | 9.79410 | 0.725132% |

기본 timestep과 1/4 timestep의 이 시점 점도 차이는 1.226%였습니다. 이 초기 구간에서는 하중 지연 오차가 timestep 감소와 함께 줄었지만, strain 0.25 이후 또는 다른 농도의 정확도를 인증하는 결과는 아닙니다.

동일 장비·1 thread·동일 500개 입자 초기배치에서 **첫 20개 step**의 내부 시간을 비교했습니다. 이전 설정은 mesh+Heun, `kn=0.3 N/m`이며, 새 설정은 nearfield+IMEX, `kn=30 N/m`입니다. 유체 모델과 접촉 적분·강성이 함께 바뀐 전체 설정 비교입니다.

| AR | 이전 평균 step 시간 | 새 평균 step 시간 | 시간 비율 |
| --- | --- | --- | --- |
| 1 | 13.6847 s | 0.072855 s | 187.8배 |
| 5 | 8.58493 s | 0.242320 s | 35.43배 |

초기 20 step 가속률을 후반 고접촉 상태나 strain 10의 실행 시간으로 외삽하지 않습니다. AR5 1000 step 평균이 초기 20 step보다 큰 것도 접촉 상태에 따라 비용이 달라짐을 보여 줍니다. 검증 원본은 `validation_nearfield_20261028/FINAL_VALIDATION.json`과 `FINAL_TIMING.json`에 보관합니다.

최대 overlap 기준은 AR1에서는 반지름, oblate spheroid에서는 최소 반축의 5%입니다. 이 기준을 만족해도 발산 농도 비교에는 충분한 strain 평균, seed·입자 수 영향과 timestep·강성 의존성 확인이 필요합니다. 이 변경은 attraction을 포함하지 않으며 그 안정성을 검증한 것도 아닙니다. 짧은 실행 성공을 strain 10, 정상상태 점도 또는 다른 농도까지의 안정성으로 확대하지 않습니다.
