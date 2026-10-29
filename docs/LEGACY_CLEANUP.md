# 기존 폴더 정리 판단

기준: Test_BS `b8f7890`의 실제 CMake·빌드·실행 도구 참조를 확인했다.
이번 통합에서는 아래 폴더를 자동으로 삭제하지 않는다. 새 compact 모델은
기존 `.ssb` 기저나 `.sft` 표 생성 결과를 수치 입력으로 사용하지 않는다.

## 폴더 전체를 제거해도 새 모델의 계산에 영향이 없는 항목

| 폴더 | 내용 | 삭제 시 잃는 것 |
| --- | --- | --- |
| `lammps_slurry/cases/` | 예전 Stage 1 JSON 예제. 현재 입력 생성기의 실행 입력이 아니다. 두께·밀도·시간 간격도 현재 목표와 다르다. | 이전 예제 한 개 |
| `lammps_slurry/tests/reports/` | 이전 Mari 구현 검토 JSON | 이전 검토 기록 |
| `validation_finite_gap/` | 이전 finite-gap 표·체크포인트·런타임 보호의 독립 검증 스크립트와 결과 | 이전 모델 및 구형 실행 도구의 재현 시험 |

이 항목들은 현재 저장소의 빌드 및 실행에서 필수 입력으로 참조되지 않는다.
Git에 이미 보관된 기록이므로 필요하면 해당 커밋에서 복구할 수 있다.

## 현재 빌드·검증 경로를 확인하고 보존할 항목

| 폴더 | 새 compact 모델의 필요 여부 | 구형 도구를 남기는 경우의 조건 |
| --- | --- | --- |
| `validation_basis/` | 수치 입력으로 불필요 | 현재 `bundle_guard.py`가 두 gate 스크립트를 필수 파일과 fingerprint 입력으로 지정한다. 이번 overlay에서도 폴더 전체 삭제 금지. |
| `lammps_slurry/models/` | `.ssb`·`.sft` 등의 기저/표가 불필요 | 현재 전체 빌드의 `ar10_matched.ssb` 기반 회귀 시험 등에 필요하므로 이번 overlay에서는 보존. 실행 경로 변경만으로 삭제하지 않는다. |
| 모델 파일 옆 `*.fast_cache/` 및 표 생성 체크포인트 폴더 | 불필요 | 구형 표 생성을 재개할 필요가 없고 해당 파일을 사용하는 실행 중 작업이 없을 때 삭제 가능. 현재 Git에는 이 생성물 폴더가 포함되어 있지 않다. |
| 기존 `build/` 하위의 구형 빌드 폴더 | 새 빌드로 재생성 가능 | 작업이 실행 중이지 않고 새 실행 파일·plugin·packer가 설치되고 경로를 바꾼 뒤 삭제. 실행 중 공유 라이브러리를 덮어쓰거나 삭제하지 않는다. |

`validation_basis/`와 `lammps_slurry/models/`는 compact bridge를 실행하더라도 현재 전체 빌드·검증에 필요하다. 별도로 빌드 체계를 정리하기 전에는 삭제하지 않는다. 이번 solver 수정에 기존 `build/` 또는 `runs/` 폴더 삭제는 필요 없다.

## 폴더 전체를 삭제하면 안 되는 항목

- `lammps_slurry/src/`: 새 plugin과 접촉·입자 운동의 소스가 함께 들어 있다.
  구형 헤더가 섞여 있더라도 폴더 전체 삭제 대상은 아니다.
- `lammps_slurry/tools/`: 빌드·입자 초기화·입력 생성·점도 분석 도구가 함께 있다.
- `lammps_slurry/tests/`와 `lammps_slurry/tests/data/`: 새 검증과 기존 접촉·마찰
  회귀 시험을 포함한다. 위에서 별도로 명시한 `tests/reports/`만 분리 가능하다.
- `lammps_slurry/vendor/eigen3/`: C++ 행렬 계산 의존성.
- `lammps_slurry/vendor/gr_re2/`: 최근접점·법선·곡률 및 접촉 기하 의존성.
- 사용 중인 LAMMPS 소스·설치 디렉터리: plugin 빌드 및 실행에 필요하다.

`SOURCE_MANIFEST.json`, `RESULT.json`, `release.json` 등은 폴더가 아닌 배포·검증
메타데이터다. 구형 결과와 새 결과를 혼동하지 않도록 이름·설명을 갱신하되,
빌드가 실제 참조하는 `release.json` 등을 단순히 삭제해서는 안 된다.
