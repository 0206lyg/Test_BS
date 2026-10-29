# 비접착 sliding 0.3 + 입자 규모 rolling 0.3

개발 기준: 잘 분산된 비접착 suspension에서 유효 jamming 농도 약 0.31을 재현하는 것이 현재 목표다. 이 값은 아직 계산으로 얻은 결과가 아니다. 인력은 이후 yielding 모델을 위한 별도 기능으로 둔다. 작은 거칠기 유체 보정을 목표 달성의 대체물로 사용하지 않는다. 다분산성은 별도 비교를 실행할 후보이며, 비교 없이 낮은 점도의 원인으로 단정하지 않는다.

## 이번 설정

`run_slurry_cpu.sbatch` 안의 설정을 직접 편집하고 기존처럼 제출한다.

| 변수 | 값 | 의미 |
|---|---:|---|
| FRICTION | 0.3 | sliding Coulomb 계수 |
| ROLLING_FRICTION | 0.3 | rolling Coulomb 계수 |
| ROLLING_LENGTH_MODE | particle | 현재 동일 크기 입자의 장반경 a=AR×두께/2 |
| ROLLING_LENGTH_NM | 0 | 양수이면 위 모드 대신 해당 nm 값을 사용 |
| CONTACT_GAP_NM | 10 | 두 매끈한 입자 표면 사이의 접촉 시작 간격 |
| GAP_FLOOR_NM | 1 | 실제 유체 간격의 regularization floor |
| NORMAL_STIFFNESS | 100 N/m | 법선 탄성 강성 |
| TANGENTIAL_STIFFNESS | 80 N/m | 접선 탄성 강성 |
| TWISTING_FRICTION | 0.1 | 기존 값 유지 |
| TWISTING_LENGTH_NM | 0 | 기존 rough footprint 반경 45 nm 사용 |

거칠기 유체항의 A=9 nm, footprint 반경=45 nm, coverage=0.78은 비교 간에 유지했다. 새로운 인력·접착·법선 저항은 추가하지 않았다.

AR20, 두께300 nm에서 a=3000 nm이고 rolling 토크 한계는 `0.3*N*3000 nm`다. 기존 `0.1*N*45 nm`의 200배이며 sliding 한계력은 3배다. 자동 rolling 강성은 `kn*a*a=9e-10 N m/rad`다. 이 토크 길이는 입자 규모의 유효 회전 구속을 뜻하며, 45 nm 돌기 자체의 접촉 반경이나 측정된 흑연 토크 법칙이라는 뜻은 아니다.

## kt=80의 뜻

접촉 중 탄성 접선 변위 이력 xi_t에 대해 trial force는 `kt*xi_t`다. 힘이 `mu*N`에 도달하면 미끄러지고 이력은 `F_t/kt`로 되돌려진다. 따라서 kt는 마찰계수나 점도가 아니다. kt를 높이면 탄성 변형이 작아지며, sliding 한계 `mu*N` 자체는 변하지 않는다.

예를 들어 N=1 nN, mu=0.3에서 한계력은0.3 nN이고 kt80일 때 도달 변위는0.00375 nm다. kn100에서 법선 압축량은0.01 nm다. 현재 kt는 kn의0.8배인 penalty 접촉 설정으로, 흑연의 직접 측정된 접선 강성이 아니다.

## 적용

slurry 디렉터리에서 기존 batch를 먼저 다른 이름으로 보관하고 overlay를 푼다. 수정본 batch의 AR, PHI, COUNT, kn 및 실행 시간 설정을 확인한다. 새 기본 AR은20, COUNT는500이며 검증 계산은N108이다.

`sbatch build_slurry_cpu.sbatch`로 기존 빌드와 새 입력 생성기의 identity를 갱신한다. 이번 변경은 C++ 저항식·접촉 solver를 바꾸지 않으며, 빌드 스크립트는 기존 CMake 빌드를 재사용한다. 빌드 완료 뒤 `sbatch run_slurry_cpu.sbatch`로 하나의 계산만 실행한다. 기존 실행 중인 private snapshot에는 영향을 주지 않는다.

## 검증 범위

새 강성으로 단일 접촉의 정확한 탄성·미끄럼 해와 에너지 수지를 검사했다. 원시 자료는 `validation_strong_rolling/contact_law/`에 있다. 입력 모드와 AR에 따른 길이 변환 및 legacy 기본값은 기존 Python suite에서 검사했다.

실제 LAMMPS 비교는 `validation_strong_rolling/runtime/`의 입력·초기 배치·history·요약 JSON에 기록한다. 동일한 초기 배치에서 기존 설정, sliding만0.3, sliding과rolling을 함께 강화한 경우를 비교한다. 짧은 전단의 점도와 정상상태 점도는 구분하며, 한 농도에서 jamming 농도를 역산하여 목표0.31을 재현했다고 보고하지 않는다.

AR20, N108, phi0.24, kn100, kt80, 전단률100/s, dt1e-6s의 200 step(누적 strain0.02) 결과:

| 설정 | step200 상대점도 | 활성 접촉 수 |
|---|---:|---:|
| 기존 sliding0.1, rolling0.1×45nm | 7.32968 | 59 |
| sliding만0.3 | 7.76340 | 60 |
| sliding0.3, rolling0.3×3000nm | 8.00640 | 55 |

새 설정은 같은 시점의 기존 설정보다9.23%, sliding만 높인 설정보다3.13% 높았다. 이 단기 비교에서 수십 배 상승은 나타나지 않았다. 동일 초기 좌표·배향·packing을 사용했고 초기 nematic order S는0.867859였다. random 삽입 뒤 성장·완화한 배치는 이미 강하게 배향되어 있었다. 현재 결과를 등방성 초기 배치나 정상상태 농도 발산 측정으로 해석하지 않는다.

phi0.30에서도 별도 동일 packing의20step(strain0.002) 비교를 수행했다. 기존7.90699, 새 설정8.05713이며 초기 S=0.919221이었다. 이 결과는 고농도 단기 실행 확인이며 phi_J=0.31 재현 결과가 아니다. 모든 계산은 pinned LAMMPS의 로컬 Eigen/내장 FFT reference backend에서 수행했으며, Negishi FFTW/MKL 실행속도 측정은 아니다.

물리 참고: Singh et al., Phys. Rev. Lett. 124,248005(2020), https://arxiv.org/abs/2002.10996 . 논문의 rolling torque는 입자 규모 reduced radius를 사용한다. 이를 spheroid 장반경으로 적용한 것은 명시적인 유효 모델 선택이다.
