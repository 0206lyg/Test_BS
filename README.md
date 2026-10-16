# Test_BS — Graphite slurry

LAMMPS implicit-solvent MD/DEM: lubrication, 3 nm 거칠기 접촉, Coulomb sliding 마찰,
전단응력과 점도를 계산합니다. 활성화 장벽·contact attraction·rolling friction은 없습니다.

## 빌드와 실행

이 ZIP에는 `Test_BS/` 상위 폴더가 없습니다. ZIP을 기존 `/home/lee6456/slurry`에
놓고 같은 디렉터리에서 덮어씁니다. 기존 LAMMPS 설치는 재사용합니다.

```bash
cd /home/lee6456/slurry
unzip -o Test_BS_solver_update_fixed.zip
SLURRY_ROOT="$PWD" sbatch build_slurry_cpu.sbatch
# Build complete와 Plugin load verified를 확인한 뒤:
SLURRY_ROOT="$PWD" sbatch --time=12:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 10 --phi 0.24
```

소스·실행 루트·빌드된 바이너리의 일치를 확인한 뒤 packing을 시작합니다.
구버전 혼용이나 미완료 빌드는 오류와 함께 중단합니다. 예전에 생긴 중첩 `Test_BS/`
폴더로 이동할 필요가 없습니다. 실행 스크립트의 AR·φ 변경만으로는 재빌드가 필요 없습니다.

| 작업 | MPI rank | rank당 CPU | 용도 |
| --- | ---: | ---: | --- |
| 빌드 | 1 | 4 | 병렬 컴파일 |
| 시뮬레이션 권장값 | 1 | 1 | 현재 solver는 rank 0에서 직렬 실행 |

4 rank의 추가 가속은 확인되지 않았습니다. 독립적인 조건을 동시에 계산한다면
4코어로 1코어 작업 4개를 실행할 수 있습니다.
Slurm 시간 제한은 기본 30분이므로 긴 계산은 `sbatch --time=24:00:00 ...`처럼 조절합니다.

기본 모듈은 `anaconda/2024.02-py311`, `gcc/12.2.0`, `openmpi/4.1.4`,
`cmake/3.23.1`입니다. 모듈명은 `SLURRY_PYTHON_MODULE`, `SLURRY_GCC_MODULE`,
`SLURRY_MPI_MODULE`, `SLURRY_CMAKE_MODULE`로 변경할 수 있습니다.
LAMMPS 버전은 `lammps_slurry/lammps-version.json`으로 고정하며 첫 빌드에서 내려받습니다.
Eigen 헤더는 포함되어 있으며 PETSc·OpenLB 설치는 필요하지 않습니다.

## 실행 조건

`lammps_slurry/run_slurry_cpu.sbatch` 상단의 값을 수정하거나 명령 뒤에서 덮어씁니다.

```bash
SLURRY_ROOT="$PWD" sbatch --time=12:00:00 lammps_slurry/run_slurry_cpu.sbatch --ar 2 --phi 0.24
```

| 항목 | 기본값 |
| --- | --- |
| 입자 수 / AR / 체적분율 | 300 / 2 / 0.24 |
| 입자 전체 두께 / 밀도 | 300 nm / 2400 kg/m³ |
| 매질 점도 / 전단률 | 0.01 Pa·s / 100 s⁻¹ |
| 시간 간격 / 생산 step | 10⁻⁶ s / 100000 |
| 전단 전 완화 / 출력 간격 | 1000 step / 100 step |
| 점도 평균 시작 strain | 5 |
| 접촉 시작 간격 / 마찰계수 | 3 nm / 0.1 |
| 수직 / 접선 강성 | 1 / 80 N/m |

Oblate 반축은 `(AR*t/2, AR*t/2, t/2)`이며 `t`는 전체 두께입니다.
`lubrication-model=auto`는 AR=1에서 Mari 구형 모델, 그 외에는 국소 곡률 기반
spheroid 모델을 선택합니다. 접촉 구간에서도 유한한 lubrication을 함께 유지합니다.
구형은 `h_reg=max(h,0)+0.001*a`, cutoff는 `0.5*a`입니다.
비구형은 정규화 길이 0.15 nm, cutoff `min(75 nm, 0.5*Rlocal)`을 사용합니다.
`--contact-gap-nm`, `--friction`, `--tangential-stiffness` 등 전체 옵션은
`bash lammps_slurry/run_slurry_cpu.sbatch --help`에서 확인합니다.

## Solver와 결과

기본값은 검증한 조합인 `ilu / root / fixed / cache on`이며
ILUT drop/fill/refresh는 `1e-3 / 4 / 40`입니다. 준비·생산 계산 모두 적용됩니다.
재사용하는 것은 전처리이며, 현재 Jacobian·힘·접촉 이력과 최종 힘 수렴 기준은 유지합니다.

AR=1, N=300, φ=0.55의 짧은 비교에서 solver 개선판은 약 1.69배 가속했습니다.
19개 C++ 검사와 AR=2 두 입자 검증을 통과했지만, 비구형 벌크 점도의 정량 검증은
별도로 필요합니다.

결과는 기본 `/scratch/negishi/$USER/slurry_md_<jobid>_<시각>/`에 저장하고,
Slurm 출력은 제출 디렉터리에 둡니다. 결과 루트는 `SLURRY_RESULTS_ROOT`,
개별 실행 폴더는 `--output 새폴더`로 지정합니다.

- `history.csv`: 순간·출력 구간 평균 응력/점도, 성분별 기여, 접촉·solver 진단.
- `rheology_summary.txt`: 선택한 strain 구간의 평균 점도와 성분별 기여.
- `trajectory.lammpstrj`: 입자 궤적. 출력 간격은 `--output-interval`로 조절합니다.
- `parameters.txt`, `particles.data`, `in.slurry`: 실제 조건과 초기 상태, 실행 입력.

평균 구간은 정상상태를 자동 판정하지 않습니다.

ZIP에는 소스·빌드/실행 도구·빌드에 필요한 검사 입력과 vendor 라이선스만 포함합니다.
과거 실행 결과와 별도 검증 문서는 포함하지 않습니다.
