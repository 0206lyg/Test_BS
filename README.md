# Test_BS — compact spheroid hydrodynamics

현재 실행 모델은 `fix slurry/bridge`입니다. AR을 입력받아 compact lubrication v4, 주기 중거리 mobility, 접촉·마찰을 함께 풀고 입자의 위치·배향과 xz 전단 응력을 계산합니다. 기저나 저항표를 먼저 생성하지 않습니다.

실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다. 파일 상단의 `AR`, `PHI`, `COUNT`, `STEPS`, `OUTPUT_INTERVAL` 및 물성·solver 설정을 직접 고친 뒤 제출합니다. 실행 인자를 매번 입력할 필요가 없습니다.

## Overlay 적용과 실행

`slurry` 저장소 디렉토리에서 `unzip -o slurry_speed_overlay_20261022.zip`을 실행합니다.

`sbatch build_slurry_cpu.sbatch`로 다시 빌드합니다. 빌드 로그의 `Build complete`를 확인한 뒤 `run_slurry_cpu.sbatch` 상단 설정을 편집합니다.

`sbatch run_slurry_cpu.sbatch`로 한 실행을 제출합니다. 기본값은 AR=2, φ=0.24, 입자 500개, 100000 step, 기록 간격 100 step입니다. 실행 결과는 기본적으로 `runs/ar<AR>_phi<PHI>_n<COUNT>_<job ID>/`에 저장됩니다. 기존 실행 폴더는 덮어쓰지 않습니다.

ZIP은 기존 파일을 삭제할 수 없으므로, **성공한 빌드의 마지막 단계**에서 이전 `lammps_slurry/run_bridge_cpu.sbatch`와 `lammps_slurry/run_slurry_cpu.sbatch` 두 파일을 삭제합니다. 최상위의 새 `run_slurry_cpu.sbatch`를 사용합니다.

## 계산과 병렬 실행

중거리는 ellipsoid 형상함수를 유지하는 particle–mesh FFT이며 FFTW를 사용합니다. 격자 분배를 병렬화하고 FFT 계획·작업 버퍼를 재사용합니다. 접촉력을 독립적인 비선형 변수로 유지하는 Newton–FGMRES에서 접촉 증분을 블록 소거해 선형계 크기를 줄이고, PARDISO 전처리기의 행렬 구조 분석과 재사용 가능한 분해를 보존합니다. 최종 수렴은 원래 힘·mobility 방정식으로 검사합니다.

**전체 밀집 행렬을 만들어 직접 푸는 방식은 사용하지 않습니다.** 배치의 `MOBILITY_BACKEND=direct`는 검증용 Fourier 직접 합산을 뜻하며 밀집 직접 solver가 아닙니다. 기본값은 `mesh`입니다.

한 작업은 **MPI 프로세스 1개**와 OpenMP 스레드를 사용합니다. 기본 자원 요청은 물리 코어 16개, 메모리 96 GB이며 배치 상단의 `#SBATCH` 항목에서 변경합니다. launcher는 한 MPI 프로세스에 할당된 코어를 명시적으로 배정하고 바인딩을 출력합니다. 시작 실패 시 다른 실행 방식으로 재시도하거나 시뮬레이션을 중복 실행하지 않습니다.

프로덕션 빌드에는 GCC/OpenMP, FFTW 개발 파일과 **static oneMKL LP64/GNU-thread 라이브러리**가 필요합니다. `build_slurry_cpu.sbatch`는 Negishi의 `fftw`, `intel-mkl` 모듈을 추가로 불러옵니다. 사이트의 모듈 이름이 다를 때는 `SLURRY_FFTW_MODULE`, `SLURRY_MKL_MODULE`을 지정합니다. 필요한 최적화 라이브러리가 없으면 느린 backend로 자동 전환하지 않고 빌드에서 원인을 알립니다. MKL은 정적으로 링크하고 FFTW 등 연결 라이브러리는 실행별 private runtime에 보관합니다.

## 물성·출력

기본 물성은 두께 300 nm, 매질 점도 0.01 Pa·s, 유체 gap floor 1 nm, 접촉 시작 간격 3 nm, 법선 강성 **100 N/m**, 접선 강성 80 N/m, 마찰계수 0.1입니다. 표면 기하는 실제 간격을 사용하며 스칼라 gap 함수에만 floor를 적용합니다. 기본 전단률은 100 s⁻¹, timestep은 1e-6 s입니다.

초기배치는 실제 주기 표면 gap이 3 nm 이상임을 인증하며, 접촉력은 실제 위치·배향 갱신 후의 gap과 일치하도록 보정합니다. `SPECTRAL=2`는 저비용 중거리 근사입니다. 4 또는 8은 생략한 고파수 입자 간 상호작용의 영향을 줄이는 설정입니다. `SPECTRAL_TOLERANCE=1e-8`은 Fourier 절단 허용값이며 전체 유체력 오차가 아닙니다.

`history.csv`는 매질·유체·접촉 응력과 순간 점도, 매 시간단계에 누적한 평균, 소산, solver 잔차 및 비용을 출력합니다. `trajectory.lammpstrj`는 입자별 유체력·토크·stresslet과 접촉력·토크를 별도로 출력합니다. 평균창은 배치의 `AVERAGE_START_STRAIN`으로 지정합니다. `final.data`는 기하 저장이며 마찰 이력을 보존하는 restart가 아닙니다.

- [실행·출력·점도 평균](docs/BRIDGE_RUN.md)
- [연결식·단위·접촉·응력 구현 명세](docs/COMPACT_BRIDGE_MODEL.md)
- [주기 중거리의 근사와 해상도](docs/PERIODIC_MIDDLE.md)
- [이전 FFT·접촉 변경과 검증 기록](docs/FFT_CONTACT_UPDATE.md)

기존 검증 폴더의 결과는 해당 소스 버전과 조건에서 얻은 기록입니다. 짧은 실행 검사나 kernel 시간만으로 정상상태 점도 또는 전체 실행 속도를 판정하지 않습니다.
