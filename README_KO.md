# Test_BS 실행 안내

2026-10-28 AR5 solver 수렴 보강: [실패 원인·수정·검증 범위](docs/AR5_SOLVER_RECOVERY.md).

현재 한국어 실행 안내는 [README.md](README.md)를 기준으로 합니다. 두 크기 입자, 초기 혼합 알고리즘과 고강성 검증 범위는 [bidisperse 변경 기록](docs/BIDISPERSE_UPDATE.md)에 있습니다.

현재 모델은 `PARTICLE_SHAPE=spheroid`, `MOBILITY_BACKEND=nearfield`입니다. 저장소에서 적용하고 다시 빌드합니다.

```bash
unzip -DD -o slurry_bidisperse_overlay_20261028.zip
python3 apply_bidisperse_settings.py
sbatch build_slurry_cpu.sbatch
```

Helper는 기존 `run_slurry_cpu.sbatch`의 입자 수, AR, 농도, 강성, timestep, step 수와 자원 설정을 보존합니다. 빌드 로그의 `Build complete`를 확인한 뒤 `sbatch run_slurry_cpu.sbatch`로 실행합니다.

과거 flake 및 유체 모델 문서와 검증 폴더는 해당 버전의 기록입니다. 현재 설정이나 검증 범위는 위 두 문서를 사용합니다.
