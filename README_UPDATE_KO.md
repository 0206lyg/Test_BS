# N=1000 경량 업데이트 적용

이 ZIP은 기존 `Test_BS` 소스 폴더에 적용하는 업데이트입니다. `build_slurry_cpu.sbatch`와 `lammps_slurry/`가 있는 폴더에서 사용합니다. 원본 20261102 전체판과 같은 N=1000 solver와 실행 설정을 사용하며, 공용 Eigen 헤더, 기존 모델 파일, 초기 입자 덤프 및 실행 이력은 배포 파일에서 제외했습니다.

1. ZIP을 소스 폴더에 풀고 같은 이름의 파일을 교체합니다. 수정해 둔 `run_slurry_cpu.sbatch`를 보관하려면 압축을 풀기 전에 다른 이름으로 복사합니다.
2. 그 폴더에서 아래 명령으로 빌드 입력을 확인합니다. 성공하면 아무것도 출력하지 않고 종료합니다. 파일이 빠졌거나 다르면 해당 경로가 출력됩니다.

   ```bash
   sha256sum --quiet -c PRACTICAL_UPDATE.sha256
   ```

3. `sbatch build_slurry_cpu.sbatch`로 빌드를 제출합니다. 로그에 `Build complete`가 뜨면 `sbatch run_slurry_practical_cpu.sbatch`로 실행합니다.

새 작업 폴더에서 시작할 때는 `Test_BS`의 기준 커밋 `fffe06ce1e44bebb8ae9af92feeee3d069a5333e`를 먼저 준비하고 같은 방법으로 ZIP을 적용합니다. 기존 전체판을 풀어 둔 소스 폴더에도 적용할 수 있습니다. N=1000 설정과 측정 결과는 `README_PRACTICAL_KO.md`와 `validation_practical_20261102/FINAL_REPORT.md`에 있습니다.
