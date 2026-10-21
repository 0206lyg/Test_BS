# 실제 LAMMPS 주기 경계·box flip 검증

`slurry/bridge`를 실제 LAMMPS에서 사용해 두 AR2 입자를 움직였다.
두께 300 nm, 점도 0.01 Pa·s, 초기 표면 간격 2 nm, 접촉 시작 간격 3 nm,
마찰계수 0.1, 전단률 100 s⁻¹, 시간 간격 10⁻⁶ s로 4 step을 실행했다.
초기 tilt는 `xz/L = 0.4999`다. 정상상태 점도 시험은 아니다.

동일한 물리 배치를 세 표현으로 실행했다.

- `original/`: 기준 배치. step 2에서 실제 triclinic box flip 발생.
- `rebased/`: 처음부터 `xz`를 상자 길이만큼 줄인 동등 격자.
- `translated/`: 전체 위치를 옮긴 동등 배치. step 2에서 입자 하나가 z 경계를
  실제 통과하고 image flag가 바뀌며, 같은 step에 box flip도 발생.

모든 출력 step에서 접촉이 유지됐다. 등가 표현 사이 전체 응력·소산·점도
차이는 최대 상대 6.79×10⁻¹², 개별 입자의 유체력·토크·stresslet 및 접촉력·토크
차이는 최대 상대 2.87×10⁻¹⁰이었다. 유체력과 접촉력의 합은 상대 1.52×10⁻¹²,
토크 합은 상대 2.85×10⁻¹⁰, 응력 일률과 소산의 차이는 상대 1.15×10⁻¹⁵ 이하였다.
`remap x`를 지정한 입력은 명시적인 `remap v` 요구 오류로 거부됐다.

최종 승인 빌드로 다시 실행해 같은 결과를 확인했다. 실행 시간 원시는
`result.json`에 기록했다. 동시에 별도의 다입자 검증이 실행 중이었으므로
이 시간은 독립 성능 벤치마크로 사용하지 않는다.

`result.json`은 strict JSON이며 초기 평균처럼 정의되지 않는 값은 `null`이다.
해당 파일에 사용한 plugin과 runner의 SHA-256이 들어 있다. 각 폴더에는
입력·원시 trajectory·history·LAMMPS 로그가 함께 있다.

여기 기록한 결과는 로컬 **serial LAMMPS**에서 확인했다. MPI 다중 rank 검증 결과가 아니다.

로컬 serial 재현 명령(실행 파일과 plugin의 경로를 자신의 빌드 위치로 지정):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation_bridge/run_periodic_check.py \
  --executable /path/to/lmp --plugin /path/to/libslurry.so
```

클러스터의 MPI LAMMPS를 사용할 때는 같은 MPI 모듈을 불러온 뒤 `--mpi`를
추가한다. 이 옵션은 `mpirun` 또는 `mpiexec`로 **1 rank**를 실행한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation_bridge/run_periodic_check.py \
  --mpi --executable /path/to/mpi/lmp --plugin /path/to/matching/libslurry.so
```

MPI launcher를 찾지 못하면 계산을 시작하지 않고 오류를 표시한다. MPI 옵션의
제공 자체가 클러스터 실행이나 다중 rank 성능을 검증했다는 뜻은 아니다.
