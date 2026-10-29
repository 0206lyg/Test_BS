# Negishi runtime snapshot 수정

적용 대상: `20261020-general-hydrodynamics-lammps`.
`build_slurry_44521656.out`에서는 컴파일은 완료됐지만, 자동 검사 35개 중
`slurry_runtime_snapshot` 하나가 실패해 빌드가 종료됐다.

## 원인과 수정

일부 링커가 생성하는 `DT_RPATH`는 `LD_LIBRARY_PATH`보다 우선한다.
따라서 실행 파일을 복사하고 라이브러리 폴더를 지정해도 원본 라이브러리를
읽을 수 있었다. 이 조건을 강제로 만들어 로그의 두 오류를 재현했다.

실행용 복사본의 `DT_RPATH` 태그만 `DT_RUNPATH`로 바꾼다.
원본 실행 파일과 라이브러리는 수정하지 않는다. 원본 및 변환된 복사본의
SHA256을 각각 기록·검증하며, 실제 라이브러리 로딩 경로 검사도 유지한다.
기존 RUNPATH 파일은 바이트 단위로 그대로 복사한다.
유체력, 접촉, solver, 시간 적분의 수식과 파라미터는 변경하지 않았다.

## 적용

`slurry_runtime_rpath_fix.zip`을 Negishi의 `$HOME`에 업로드한 뒤 실행한다.

```bash
cd "$HOME/slurry"
unzip -o "$HOME/slurry_runtime_rpath_fix.zip"
export SLURRY_ROOT="$PWD"
sbatch build_slurry_cpu.sbatch --without-legacy-table-tools
```

기존 `build` 디렉토리를 유지한다. 완료된 C++ 컴파일 결과를 재사용하며,
검사와 플러그인 로딩이 통과한 뒤 새 빌드 manifest를 만든다.
검사 생략이나 실패 무시는 사용하지 않는다.

## 확인한 결과

- 기존 오류 두 가지 재현: 원본 경로로의 library escape, private 실행의 SIGBUS.
- 수정 후 runtime 검사 5개 통과: executable과 종속 DSO의 강제 old RPATH,
  원본 library 절단 후 private copy 생존, 기존 RUNPATH 보존 포함.
- 전체 CTest 35/35 및 incremental build/plugin load 통과.
- 이전에 검증한 실행 바이너리 SHA256은 동일함.

자세한 기록은 `validation_general/runtime_rpath_fix/`에 있다.
수정본의 Negishi 빌드는 위 명령으로 재제출해야 한다.
