# 2026-10-22 CPU 실행 경로 개선

기준 소스는 Test_BS `2476215eb6da89c978cdd602e44f47cd0f10881a`이다.
근거리 연결식, 중거리 물리 모델, 접촉/마찰 법칙, timestep 및 허용오차는 유지했다.
전체 밀집 직접해법은 사용하지 않는다.

## 구현

- FFTW OpenMP 변환: 격자/스레드 수/버퍼 주소가 같으면 FFT 계획을 재사용한다.
- 입자별 격자 분배를 thread-private 격자에 병렬 수행하고, 정해진 순서로 합산한다.
  격자 버퍼와 mobility 객체는 시간단계 사이에도 유지된다.
- Newton 선형계에서만 접촉 증분을 정확히 제거한다. 비선형 접촉력, Coulomb 마찰,
  line search, endpoint 간격 보정 및 최종 실제 물리 잔차 검사는 유지한다.
- 전체 Newton 블록이 `[A B; C D]`이고 우변이 `-[r_y;r_lambda]`일 때
  `(A-B D^-1 C) delta_y = -r_y + B D^-1 r_lambda`를 FGMRES로 풀고,
  `delta_lambda = -D^-1(r_lambda + C delta_y)`로 복원한다.
  각 접촉의 D는 3×3이다. Krylov 차원은 `17N+3Nc`에서 `17N`으로 줄어든다.
- 희소 전처리기는 MKL PARDISO를 사용한다. 구조가 같으면 symbolic analysis를,
  값까지 정확히 같으면 numeric factorization도 재사용한다. 값이 바뀌면 분해를
  갱신하며 근사적으로 오래된 factor를 쓰지 않는다. 고립 입자 블록, norm monitor,
  고정 근거리 조립 및 Krylov 버퍼도 재사용한다.
- 정확한 `M*0=0`, 이미 계산된 선형 응답 및 수락된 방향의 응답을 재사용하여
  중복 mobility 호출을 줄인다. 반환 직전에는 원래 물리 잔차를 새로 계산한다.

## 빌드와 실행

`slurry` 디렉터리에서 overlay를 해제한 뒤 빌드한다.

`unzip -o slurry_speed_overlay_20261022.zip`

`sbatch build_slurry_cpu.sbatch`

빌드 성공 후 최상위 `run_slurry_cpu.sbatch`의 상단 설정을 편집한다.
AR, PHI, COUNT, STEPS, timestep, 물성, 접촉, 초기 배치, solver 설정과 출력 경로가
한곳에 있다. 실행 인자를 따로 입력하지 않는다.

`sbatch run_slurry_cpu.sbatch`

기본 요청은 한 노드, MPI rank 1개, OpenMP 16개, 메모리 96 GB이다.
`mpirun -np 1 --map-by slot:PE=16 --bind-to core`에 해당하는 명령을 한 번만
실행한다. 실패한 실행을 다른 경로로 자동 재시도하지 않는다. Python driver를
MPI 안에서 다시 실행하는 것도 거부한다. 실제 명령과 thread 설정은 run의
`launch.json`, OpenMPI binding은 `console.log`에 남는다.

생산 빌드는 FFTW와 static MKL LP64/GNU threading을 요구한다. MKL을 정적으로
연결하여 private runtime 밖에서 CPU dispatch 라이브러리를 불러오는 문제를
피하고, OpenMP runtime은 GCC의 libgomp 하나를 사용한다. FFTW와 LAMMPS 등의
동적 의존성은 기존 private runtime 복사/경로 검증을 거친다.

성공적으로 빌드한 뒤에만 다음 구형 실행 파일 두 개를 제거한다.

- `lammps_slurry/run_bridge_cpu.sbatch`
- `lammps_slurry/run_slurry_cpu.sbatch`

최상위 `build_slurry_cpu.sbatch`는 빌드용으로 계속 사용한다.

## 진단 및 재현

로컬 8스레드, N=500, phi=0.24, 5 step의 단독 실행에서 측정했다.
초기 step 0을 제외한 `profile_total_s` 중앙값은 다음과 같다.
컴파일 또는 다른 시뮬레이션을 동시에 실행하지 않았다.

| AR | 이전 전체 step | 개선 후 전체 step | 가속률 |
|---|---:|---:|---:|
| 2 | 7.7497 s | 1.8104 s | 4.28배 |
| 20 | 5.4329 s | 2.3380 s | 2.32배 |

36개 CTest와 실제 private runtime을 통한 N=32, 3 step 실행을 통과했다.
N=500 비교에서는 응력뿐 아니라 입자별 속도·배향·힘·토크·stresslet을 비교했다.
소스 fingerprint, 실행 바이너리 hash, 측정 환경, 최종 입자 프레임과 원시
history는 `validation_speed/results`에 포함했다. Negishi의 실제 MPI 바인딩과
가속률은 이 로컬 측정에 포함되지 않는다.

`history.csv`에 `contact_candidates`, `krylov_dimension`, `fft_plan_builds`,
`mobility_threads`, symbolic/numeric factorization 누적 횟수 및 재사용 횟수를
추가했다. 동일 격자에서 FFT 계획 횟수가 계속 증가하면 재사용 조건을 확인한다.
입자가 움직여 행렬 값이 달라지면 numeric 재분해가 발생하는 것은 정상이다.

`validation_speed/compare_speed.py`는 동일한 인증 초기 배치를 두 플러그인에
순서대로 적용하고 원시 응력·잔차·소산·시간 기록을 보존한다. 선택적으로 최종
입자별 위치·배향·속도·힘·토크·stresslet도 비교한다. 원시 자료와 측정 조건은
`validation_speed/results`에 둔다.

이 검증은 짧은 시간단계의 수치적 동등성과 실행 비용에 대한 것이다.
정상상태 점도, 근거리 모델의 물리 정확도 또는 Negishi에서의 동일한 가속률을
새로 검증했다는 의미는 아니다.
