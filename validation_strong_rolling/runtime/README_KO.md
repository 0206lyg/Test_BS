# 강한 비접착 마찰·rolling 실제 LAMMPS 비교

이 폴더는 수정된 접촉 계수를 실제 pinned LAMMPS에 입력하여 측정한 짧은 전단 과도응답이다. 정상상태 점도 또는 jamming fraction을 측정한 자료로 해석하면 안 된다.

공통 조건: spheroid AR20, 두께 300 nm, 입자 108개, 점도 0.01 Pa s, 전단률 100 /s, dt=1e-6 s, 접촉 시작 간격 10 nm, 유체 간격 floor 1 nm, kn=100 N/m, kt=80 N/m. 인력 없음. 거칠기 유체항 A=9 nm, w=45 nm, coverage=.78은 모든 경우에 동일하다. Twisting 계수 .1 및 길이 45 nm도 동일하다.

| 경우 | sliding 계수 | rolling 계수 | rolling 길이 | rolling 강성 |
|---|---:|---:|---:|---:|
| base | .1 | .1 | 45 nm | 2.025e-13 N m/rad |
| sliding | .3 | .1 | 45 nm | 2.025e-13 N m/rad |
| strong | .3 | .3 | 3000 nm | 9e-10 N m/rad |

각 농도에서 production random-growth packer와 seed1729로 한 번 준비한 입자 배치를 복사하여 비교했다. `particles.data`의 SHA256으로 동일성을 확인한다. packing의 모든 주기 이미지에 대한 10 nm clearance 인증을 통과했다. Random initialization은 seed 단계의 무작위성을 뜻하며, 성장·기하 완화 후 등방성을 보장하지 않는다. 초기 nematic order S는 phi=.24에서 .867859, phi=.30에서 .919221이었다.

- phi=.24: `ar20_base200`, `ar20_sliding200`, `ar20_strong200`; 200 steps, strain .02.
- phi=.30: `ar20_phi30_base`, `ar20_phi30_strong`; 20 steps, strain .002.
- `ar20_base`, `ar20_strong`은 앞서 실행한 20-step 성능 확인으로 200-step 결과에 포함된다.

최종 숫자, 잔차, 소산 일관성, 입력·출력·binary hash는 `runtime_validation.json`에 있다. `all_complete`가 true인 최종 파일만 완성 자료로 사용한다. 평균은 기록된 마지막 절반 구간의 산술 평균이며 정상상태 평균을 의미하지 않는다.

| 농도 / 비교 | 최종 eta_r | 마지막 절반 평균 eta_r | 최종 접촉 수 |
|---|---:|---:|---:|
| .24 base | 7.329678 | 7.230150 | 59 |
| .24 sliding | 7.763403 | 7.636656 | 60 |
| .24 strong | 8.006402 | 7.852452 | 55 |
| .30 base | 7.906993 | 7.771613 | 10 |
| .30 strong | 8.057126 | 7.876678 | 10 |

phi=.24에서 최종값 기준 sliding 강화만으로 +5.92%, rolling 강화까지 포함하면 +9.23%였다. 동일 구간 평균 기준은 각각 +5.62%, +8.61%이다. rolling 강화가 sliding 강화에 더해 제공한 증가율은 최종값 +3.13%, 구간 평균 +2.83%이다. phi=.30의 훨씬 짧은 시험에서는 최종값 +1.90%였다. 현재 측정은 목표 eta_r~100 또는 phi_J~.31 재현 결과가 아니다.

phi=.24의 마지막 50 step (151–200) 평균은 base 7.343375, sliding 7.774078, strong 7.995660이다. Base 대비 증가는 +5.87%, +8.88%; strong의 sliding 대비 증가는 +2.85%이다. 최종 평균 접촉 차수 2C/N은 각각 1.0926, 1.1111, 1.0185이고, 같은 마지막 50-step 평균은 1.0652, 1.0948, .9996이다. S 최종값은 각각 .865578, .865541, .865756으로 초기의 강한 nematic order가 이 짧은 시간 동안 거의 유지되었다.

다섯 실행 모두 지정 step을 완료했다. 전체 최대 solver relative residual은 9.75e-8 미만, 상대 power balance error는 4.29e-7 미만이었다. 플러그인/solver 코드는 이 비교를 위해 변경하지 않았다.

실행환경은 LAMMPS stable_30Sep2026 commit 8de817dd79bfe4525d5d39246a212d833e6dee07, 전체 repository plugin, serial MPI stubs, OpenMP, -O1, Eigen 전처리기 및 builtin FFT reference backend이다. phi=.24 실행은 경우당 2 threads, phi=.30은 경우당 1 thread를 사용했다. Negishi의 MPI/FFTW/MKL 성능 시험은 아니다.

각 case 폴더의 `in.bridge`, `particles.data`를 사용하고 `slurry_plugin` 변수를 동일 모델의 `libslurry.so` 경로로 지정하면 직접 재실행할 수 있다. 입력에는 run 길이 및 모든 물리 계수가 들어 있다. `summarize_runtime.py`는 CSV와 입력을 읽어 보고서를 다시 작성한다. 다만 provenance binary 경로는 이 검증을 수행한 로컬 경로이므로 다른 환경에서는 수정해야 한다.
