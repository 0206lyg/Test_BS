# Spheroid / flake 공통 실행 overlay

기준 저장소: `0206lyg/Test_BS`, `5942c63e13f5d86b3684b93eb6f6322f7bfb8b16`.
`slurry` 폴더에 압축을 풀고 기존 `build_slurry_cpu.sbatch`로 다시 빌드한다.
기존 spheroid 코드는 유지하며, 독립 cylinder pilot 실행기는 이 실행 경로에서 사용하지 않는다.

## 실행

`run_slurry_cpu.sbatch`의 `PARTICLE_SHAPE=spheroid` 또는 `PARTICLE_SHAPE=flake`만 변경해 형상을 선택한다.
AR, PHI, COUNT, STEPS, TIMESTEP, OUTPUT_INTERVAL 등도 같은 파일의 상단에서 입력한다.
제출 명령은 `sbatch run_slurry_cpu.sbatch`이며 실행 인수를 추가할 필요가 없다.
C++ 플러그인과 packer가 변경됐으므로 overlay 적용 뒤 한 번은 `sbatch build_slurry_cpu.sbatch`가 필요하다.
빌드 완료 후 run을 제출한다. 기존 실행 중인 job의 private runtime은 바뀌지 않는다.

## 공통 경로와 형상별 차이

| 항목 | spheroid | flake |
|---|---|---|
| 기하 | oblate spheroid | 닫힌 연속 원기둥, 평평한 원형 양면 |
| 체적 | 4πa²c/3 | 2πa²c |
| 초기배치 | 기존 무작위 seed, 성장, FIRE 완화 | 같은 절차와 seed 생성기; 원기둥 support 기하로 성장/완화 |
| 최종 초기 간격 | 실제 spheroid 표면 간격 | 실제 원기둥의 지지평면 하한으로 모든 관련 주기 이미지 간격 인증 |
| 근거리 유체력 | 기존 compact bridge | 개발된 면–면 Reynolds 박막 + 양의 가장자리 근사 |
| 중거리·고립 입자 | 기존 periodic mesh 모델 | 같은 a,c의 spheroid를 사용하는 명시적 근사 |
| 접촉 | 단일 접촉점 선형 스프링/Coulomb | 같은 법칙; 원기둥 접촉점·법선·다음 단계 간격 |
| 적분/solver | 공통 Newton–FGMRES, FFTW, MKL-PARDISO | 같은 경로 |

`a=AR×t/2`, `c=t/2`이다. 각 형상의 실제 체적으로 `L=(N V_particle/PHI)^(1/3)`를 계산한다.
같은 N, AR, 두께, PHI에서 flake 상자는 spheroid 상자보다 1.5^(1/3)배 크다.
같은 seed와 준비 알고리즘을 써도 서로 다른 형상은 다른 최종 미세구조를 만들 수 있다.
입자 성장/완화는 기하 준비 과정이며 물리적 정상상태 전단은 아니다.
목표 PHI에 도달하지 못하면 낮은 PHI를 목표값인 것처럼 출력하거나 정렬 배치로 자동 변경하지 않는다.

Flake 저장에는 LAMMPS `atom_style ellipsoid`의 위치·반축·quaternion 컨테이너를 사용한다.
이것이 실제 접촉 기하를 spheroid로 바꾸지는 않는다.
입력 저장 density는 1.5배로 조정해 실제 원기둥 질량을 만들며, fix는 원기둥 관성을 사용한다.
따라서 dump의 반축을 일반 ellipsoid glyph로 그리면 flake가 잘못 시각화될 수 있다.

## 물성 및 근사 범위

기본값: 매질 점도 0.01 Pa·s, 접촉 시작 표면 간격 3 nm, 유체력 gap floor 1 nm,
sliding friction 0.1, 정상/접선 강성 100/80 N/m. 형상 선택으로 이 값들은 바뀌지 않는다.
접촉 영역 전체에 분포된 마찰·토크는 이번 통합에 새로 도입하지 않았다.
Flake 중거리/고립 입자 이동도는 정확한 원기둥 Stokes 해가 아니며, 가장자리 유체력도 근사이다.
면–면 박막 수치 검증과 다입자 점도의 물리적 정확도는 별개의 문제이다.

## 출력

- `parameters.json`: 실제 형상, 체적, 질량, PHI, 상자 크기, 근사와 준비 조건.
- `packing_history.csv`, `packing_clearance.json`: 목표 농도 도달 및 주기 표면 간격 인증.
- `history.csv`: 매질/유체력/접촉 응력, 상대점도, 매 step을 포함하는 출력구간 평균, solver 및 시간 진단.
- `trajectory.lammpstrj`, `final.data`: 위치·배향과 최종 상태.
- `viscosity_summary.json`, `viscosity_summary.txt`: `AVERAGE_START_STRAIN` 이후의 시간 가중 평균.
  짧은 실행이 평균 시작 strain에 도달하지 않았다면 평균 없음이 명시되며, 0 또는 정상상태 점도를 만들어 내지 않는다.

평균 출력만으로 정상상태 도달을 보증하지 않는다. 이번 통합 검증의 조건과 결과는 같은 폴더의 `RESULT.json`에 기록한다.

## 이번 통합에서 실행한 검증

- 최종 빌드의 CTest 40/40 통과. FFTW와 static MKL-PARDISO를 실제로 사용했다.
- 공통 batch → private runtime → 무작위 준비 → LAMMPS 운동 → 점도 요약 전체 경로에서 AR20, N108, PHI=0.24를 flake 20 step / spheroid 3 step 실행했다.
- 별도로 N500 flake의 목표 PHI=0.24 준비와 실제 운동 3 step을 확인했다. 초기 clearance는 3.0000884435 nm 이상, 운동 중 실제 표면 겹침은 없었다.
- 최신 main과 spheroid 3 step 비교에서 시간 측정 열을 제외한 응력·궤적·최종 데이터가 동일했다. 실제 LAMMPS의 원기둥 질량과 관성 적분도 확인했다.
- Flake 동일 N108 배치에서 1/4 threads의 마지막 상대점도 차이는 약 6.5e-8이다.

N108 flake는 strain 0.002까지이며 상대점도는 4.16→5.10, 구간 평균 4.777이다. 같은 준비 절차의 spheroid 3 step 평균은 5.518이다. 초기 미세구조와 관측 시간이 다르므로 이 수치로 정상상태 우열을 정하지 않는다. **이번 결과는 flake의 정상상태 점도 상승을 입증하지 않는다.**

이 환경의 4 threads에서 N108은 약 0.9초/step, N500은 약 12초/step이었다. N500 비용의 대부분은 solver이며, 장시간 실행이 빠르다는 검증은 아니다. 초기 비교는 COUNT=108 등 작은 입자 수부터 진행할 수 있다. Negishi의 MPI/Slurm 자체는 이 환경에서 실행하지 않았으며 공통 launcher와 private library 복사 경로는 로컬 실제 실행으로 검증했다.

면–면 모델은 유한 겹침 영역의 가장자리 압력 0 조건을 사용한다. 기본 압력 다항식 차수 4, 구적 24, 역간격 보강 2이며, 면 기울기 0.10–0.35 rad, 마지막 1% 겹침 면적 및 gap/radius 0.1–1 구간의 감쇠는 기존 flake 모델 선택이다. 일반 배향 원기둥의 엄밀한 Stokes 해를 구현했다는 뜻은 아니다.

같은 무작위 N108 배치가 1 step에서 중단되던 문제는 GJK 삼각형 투영의 수치적 상쇄와 접촉 endpoint/내부 solver 오차 혼동을 수정해 해결했다. 최종 힘·work 허용오차를 완화하지 않았다. 실제 표면 겹침이나 새로 누락된 접촉이 발생하면 명시적으로 중단하며, 자동 timestep 재시도는 추가하지 않았다.
