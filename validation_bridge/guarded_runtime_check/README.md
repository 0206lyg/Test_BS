# 승인 빌드와 private runtime 실행 연결 확인

실제 `tools/build.py`가 CTest 30개와 네 plugin style 로딩을 통과해 만든
승인 manifest를 사용했다. `tools/run_bridge.py`로 AR2·2입자·phi 0.01의
aligned 입력을 생성하고 serial LAMMPS에서 2 step을 실행했다.

- `check_build`가 현재 소스 fingerprint와 실행 파일·plugin·packer를 승인했다.
- 입력 생성기와 실행 파일·공유 라이브러리가 실행 폴더의 private 사본으로 복사됐다.
- 복사 파일의 SHA-256을 다시 대조했다.
- 실제 `ldd` 결과에서 시스템 기본 libc 계열을 제외한 연결 라이브러리가
  모두 해당 private `runtime/lib/` 경로로 해석됨을 확인했다.
- `history.csv`에 step 0·1·2가 기록됐으며 계산이 정상 종료됐다.
- `launch.json`의 입력·입자 파일 hash와 실제 파일이 일치했다.

검증 후 Git에 바이너리를 넣지 않기 위해 `runtime/bin/`과 `runtime/lib/`의
private 사본을 제거했다. `runtime/runtime_identity.json`과 `result.json`에
검증 당시 경로·hash·승인 build identity를 남겼다. 따라서 이 검증 결과 폴더를
기존 runtime으로 직접 재시작하는 용도로 사용하지 않는다.

사용자 환경에서는 기존 빌드를 승인한 뒤 새 출력 폴더로 재실행한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python lammps_slurry/tools/run_bridge.py \
  --output runs/guarded_check --ar 2 --count 2 --phi 0.01 \
  --initialization aligned --steps 2 --output-interval 1 --average-start-strain 0
```

기본 위치 외부에 빌드했다면 `--build-env /path/to/build/env.sh`를 함께 지정한다.
이 실행은 launcher·격리·입력·출력의 연결 시험이며 정상상태 점도 측정이 아니다.
