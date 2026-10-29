# 이전 테이블 경로 정리 목록

새 실행은 `lammps_slurry/run_general_cpu.sbatch` → `prepare_case.py` →
`fix slurry/implicit ... lubrication general`로 이어진다. AR·배향·거리에서
계산하므로 `.ssb`, `.sfg`, `.sft`와 배향별 Q 앵커 생성 작업은 필요하지 않다.
아래 파일은 이번 변경에서 삭제하지 않았다. 기존 결과를 재현할 때만 사용한다.

| 파일 | 새 general 실행에서의 역할 | 삭제 또는 보관 판단 |
|---|---|---|
| `lammps_slurry/tools/slurry_generate_basis.cpp` | 없음 | 이전 scattering 기저 생성기. 기존 finite_gap 모드를 폐기할 때 삭제 후보 |
| `lammps_slurry/tools/precompute_fast.cpp` | 없음 | 배향·거리 테이블 생성기. 기존 finite_gap_fast 모드를 폐기할 때 삭제 후보 |
| `lammps_slurry/tools/basis_cylindrical_kernel.h` | 없음 | 위 기저 생성기의 전용 보조 코드. 생성기와 함께 정리 |
| `validation_basis/run_interpolation_gate.py` | 없음 | 이전 테이블 보간 검사. 관련 CMake 대상 및 guard 참조와 함께 정리 |
| `validation_basis/run_physical_gate.py` | 없음 | 이전 기저/provider 검사 실행기. 새 일반식 실행의 전제 조건으로 쓰지 않음; 재현 기록과 함께 보관 권장 |
| `validation_basis/run_matched_query.py` | 없음 | 이전 matched provider 조회기. 비교 연구가 끝나면 삭제 후보 |
| `validation_basis/compare_provider_queries.py` | 없음 | 이전 provider 비교 도구. 비교 연구가 끝나면 삭제 후보 |
| `validation_basis/prepare_reference_overlay.py` | 없음 | 이전 reference overlay 작업. 기존 결과 재현용으로 보관 가능 |
| `validation_basis/run_runtime_smoke.py` | 없음 | 이전 hybrid/table runtime fixture 실행기. 새 일반식 검증으로 대체된 뒤 삭제 후보 |
| `validation_basis/check_generator_checkpoint.py` | 없음 | 이전 테이블 체크포인트 검사. 해당 생성기 폐기 시 함께 삭제 후보 |
| `validation_finite_gap/test_fast_plumbing.py` | 없음 | 이전 table 옵션·캐시 경로 회귀 검사. legacy 모드 보존 중에는 유지 |
| `validation_finite_gap/test_checkpoint_runtime.py`, `test_process_recovery.py` | 없음 | 이전 테이블 생성의 재개·SIGBUS 복구 검사. legacy 생성기 보존 중에는 유지 |

`validation_basis/*.json`, `docs/legacy/RESULT_20261018.json`은 과거 버전의 검증 기록이다.
새 일반식 계산기의 검증 성적으로 재사용하면 안 된다. 원본 기록을 남겨 두고
새 결과를 별도 파일에 기록한다.

`build_slurry_cpu.sbatch`, `lammps_slurry/tools/build.py`, `bundle_guard.py`,
`runtime_snapshot.py`, `prepare_case.py`, `summarize_rheology.py`와 기하 packer는
새 실행에도 사용하므로 삭제 대상이 아니다. `validation_finite_gap/test_runtime_snapshot.py`도
일반 실행에서 쓰는 실행 파일 고정 기능을 검사하므로 유지한다.

기존 `finite_gap`/`finite_gap_fast` 옵션을 남긴 상태에서 위 생성기·검사 파일만
먼저 지우면 CMake나 source identity 검사가 실패한다. **당장 필요한 조치는 파일
삭제가 아니라 예전 테이블 제출 명령의 사용 중단**이다. 일반식 전용 빌드는 다음처럼
기존 offline 실행 파일을 빌드하지 않도록 선택할 수 있다.

```bash
sbatch build_slurry_cpu.sbatch --without-legacy-table-tools
```

이 옵션은 기존 계산 결과·모델 파일을 지우지 않는다. legacy 경로를 완전히 제거하려면
위 목록과 함께 해당 CMake 대상, provider 분기, 옵션 파서 및 guard의 참조를 정리해야 한다.
