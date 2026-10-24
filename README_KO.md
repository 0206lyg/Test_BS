# Test_BS 실행 안내

현재 안내는 [README.md](README.md)를 기준으로 합니다. 실행 배치 파일은 저장소 최상위의 **`run_slurry_cpu.sbatch` 하나**입니다.

1. `slurry` 디렉토리에서 `unzip -o slurry_flake_overlay_20261023.zip`을 실행합니다.
2. `sbatch build_slurry_cpu.sbatch`로 빌드하고 로그의 `Build complete`를 확인합니다.
3. `run_slurry_cpu.sbatch` 상단에서 `PARTICLE_SHAPE=spheroid` 또는 `flake`, `AR`, `PHI`, `COUNT`, `STEPS`, `OUTPUT_INTERVAL` 및 필요한 설정을 편집합니다.
4. `sbatch run_slurry_cpu.sbatch`로 제출합니다. 명령행 실행 인자는 필요하지 않습니다.

기본값은 입자 500개, 100000 step, 기록 간격 100 step, 한 MPI 프로세스와 OpenMP 16스레드, 메모리 96 GB입니다. 프로덕션 빌드는 FFTW와 static oneMKL LP64/GNU-thread PARDISO를 요구합니다. 전체 밀집 직접 solver는 사용하지 않습니다.

성공한 빌드가 이전 `lammps_slurry/run_bridge_cpu.sbatch`와 `lammps_slurry/run_slurry_cpu.sbatch`를 삭제합니다. 새 최상위 배치를 사용하며 실행 폴더는 덮어쓰지 않습니다.

형상별 체적·박막·접촉 모델과 검증 결과는 [공통 실행 안내](validation_flake_integration/README_KO.md)에 있습니다.
