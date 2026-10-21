# LAMMPS 연결 검증과 재현

이 폴더는 채택한 연결모델을 실제 입자 운동·주기 전단·접촉·응력 출력에 연결했는지 확인한다. 정확한 Stokes 해나 정상상태 점도를 검증하는 자료는 아니다. 기하 배치를 무작정 늘리지 않고, 한 고정 다입자 초기 배치와 동등한 주기 표현을 비교한다.

| 자료 | 검사 범위 |
| --- | --- |
| `periodic_check/result.json` 및 하위 원시 파일 | 두 입자의 실제 이동, 주기 wrapping, triclinic box flip, 등가 격자·평행이동, 힘·토크 평형, 응력 일률 |
| `fixtures/random108_ar20.data` | 다입자 재현용 고정 초기 배치. AR20·N108·φ0.24이며 기하 성장으로 만들었다. 정상상태 미세구조가 아니다. |
| `run_bulk_check.py` | 동일 초기 배치의 20-step, half-dt 40-step, 중거리 spectral4 run-0 비교 및 원시 출력 분석 |
| `../lammps_slurry/tests/test_bridge_tools.py` | 입력 생성, 전체 이미지 겹침 검사, 시간 가중 점도 평균, plugin 경로 전달 |

주기 경계 실행 결과의 세부 설명은 [periodic_check/README.md](periodic_check/README.md)에 있다. 다입자 결과는 완료된 실행의 `result.json`을 확인한다. 비교 실행 중 하나라도 미완료이거나 실패하면 `run_bulk_check.py`는 최종 비교 보고서를 생성하지 않는다.

## 이번 승인 빌드의 실행 결과

`RESULT.json`은 승인 빌드의 source fingerprint와 모든 실행의 binary hash가 일치하는지 확인한 통합 요약이다. 전체 CTest 30개, 새 fix 로딩, private runtime 실행, 실제 주기 wrapping·box flip, 아래 다입자 실행이 모두 통과했다. MPI 다중 rank는 실행하지 않았다.

| 검사 | 실제 결과 |
| --- | --- |
| AR20·N108·φ0.24, spectral2 | dt=1 μs로 20 steps, 최종 strain 0.002 |
| 같은 초기 배치에서 dt/2 | 40 steps; 최종 순간 점도 차이 **0.1204%**, 같은 시간창 평균 차이 **0.06390%** |
| 힘·토크 평형 | 두 시간간격 실행의 최대 상대 잔차 각각 7.82e-7, 6.28e-7 미만 |
| 응력·일률 | stresslet에서 재구성한 유체 응력 차이 5.4e-15 Pa 미만; 최대 상대 일률 오차 1.89e-7 미만 |
| 같은 초기 상태, spectral2→4 | 순간 점도 0.0756768→0.0819347 Pa·s, 상대 차이 **7.64%** |
| 실제 비용, 8 threads | spectral2 첫 풀이 **10.02 s**, 이후 20회 평균 **1.89 s**, 마지막 10회 평균 **1.28 s** |
| 높은 해상도의 비용 | spectral4 첫 풀이 **116.66 s**; 이를 시간단계 비용으로 일반화하지 않는다. |

기본 실행의 시간 평균 0.06743994 Pa·s는 **20 μs의 과도 평균**이다. 정상상태 점도나 정확한 Stokes 해에 대한 오차로 해석하지 않는다. 시간간격 변화와 중거리 해상도 변화는 서로 다른 검사다. 비용에는 기하·중거리 구성·접촉을 포함한 동시 풀이가 들어가며, 입자 packing과 출력 I/O는 제외한다. 초기 일부 구간에는 짧은 보조 검사가 동시에 실행됐다. 원시 입력·history·trajectory·실행 hash는 `bulk_check/`에 보존했다.

## 수학·도구 검사

아래 명령은 저장소 루트에서 실행한다. 기본 빌드를 사용했다면 CMake build directory는 `build/lammps-slurry`다.

```bash
ctest --test-dir build/lammps-slurry --output-on-failure
python3 lammps_slurry/tests/test_bridge_tools.py
```

## 실제 주기 경계 실행

먼저 빌드를 완료하고 생성된 환경을 읽는다. 아래 주기 검증 도구에는 NumPy가 필요하며, 일반 준비·실행·점도 요약 도구에는 별도 Python 패키지가 필요하지 않다.

```bash
source build/lammps/env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 validation_bridge/run_periodic_check.py \
  --mpi --executable "$SLURRY_LAMMPS_EXECUTABLE" \
  --plugin "$SLURRY_LAMMPS_PLUGIN" \
  --output runs/periodic_bridge_recheck
```

`--mpi`는 기본 MPI 빌드용으로 `mpirun` 또는 `mpiexec`의 1 rank를 사용한다. `--serial`로 만든 LAMMPS를 검사할 때는 `--mpi`를 생략한다.

같은 두 입자 궤적을 기준 격자, 동등 격자, 전체 평행이동으로 실행한다. 실제 box flip과 입자 경계 통과가 일어나야 통과한다. `remap x` 입력을 거부하는지도 확인한다. 원시 파일을 보존하려면 새 output 경로를 사용한다.

## 고정 다입자 실행

이 검사는 중거리·근거리·접촉·마찰을 동시에 푼다. OpenMP 8 threads를 설정하며, 각 case의 실행 전후 binary hash와 결과 파일 hash를 기록한다. 실행 중 해당 build directory를 갱신하지 않는다. 일반 장시간 실행의 격리된 runtime은 `run_bridge_cpu.sbatch`가 제공한다.

```bash
source build/lammps/env.sh
python3 validation_bridge/run_bulk_check.py \
  --run runs/bulk_bridge_recheck --mpi \
  --lammps "$SLURRY_LAMMPS_EXECUTABLE" \
  --plugin "$SLURRY_LAMMPS_PLUGIN" \
  --threads 8 \
  --output runs/bulk_bridge_recheck/result.json \
  --states-output runs/bulk_bridge_recheck/endpoint_states.json
```

실행은 세 하위 폴더를 만든다.

- `random108`: dt=10⁻⁶ s, 20 steps, spectral2.
- `random108_halfdt`: dt=0.5×10⁻⁶ s, 40 steps, spectral2.
- `random108_spectral4`: 같은 초기 상태와 dt에서 spectral4, run 0.

처음 두 실행의 물리 시간과 전단 변형률은 같다. spectral 비교는 초기 상태만 사용하므로 시간에 따른 미세구조 변화가 섞이지 않는다. `spectral4`라는 폴더 이름은 통과·수렴을 뜻하지 않으며 실제 보고서에 결과 또는 실행 오류가 기록된다.

## 기존 원시 출력만 다시 분석

```bash
python3 validation_bridge/run_bulk_check.py \
  --analyze-existing runs/bulk_bridge_recheck \
  --threads 8 \
  --output runs/bulk_bridge_recheck/reanalysis.json
```

다음 항목을 원시 history와 trajectory에서 다시 계산한다.

- 매 step interval mean을 시간 가중한 점도·응력 평균.
- 같은 최종 시간의 주기 위치·배향 차이와 입자별 F/T/stresslet 차이.
- 입자별 유체력+접촉력 및 유체 토크+접촉 토크의 잔차.
- on-fluid stresslet의 `−sum(Sxz)/V`와 출력 유체 응력의 일치.
- 전체 응력×전단률×부피와 응력 일률의 일치, 응력 일률과 유체 소산+접촉 일률의 차이.
- 초기 solve 및 이후 solve의 실제 구성·풀이 비용. 입자 packing, 실행기 시작, 출력 I/O 비용은 별도다.

`--analyze-existing`에 현재 plugin 경로를 주더라도 그것만으로 과거 실행 binary가 증명되지는 않는다. `launch.json`이 없으면 실행 binary provenance는 미확인으로 표시한다. 새로운 `--run` 기록은 실행 전후 binary와 결과 파일 hash를 검증한다. source hash는 관찰한 소스의 식별자이며, 실제 컴파일에 쓰였다는 증명은 별도의 build manifest가 담당한다.

단기 평균과 시간간격·해상도 차이는 이 근사모델 안에서의 비교다. 전단률 100 s⁻¹에서 20 μs는 strain 0.002이므로 정상상태 점도로 보고하지 않는다. 접촉 탄성 에너지가 풀리는 동안 접촉 일률은 음수가 될 수 있다.
