# 일반 유체력 연결 실행의 원본 기록

`energy` 근접 유한항 완성과 `relative` 다입자 결합을 사용한 실제 serial LAMMPS
실행의 입력·출력이다. 각 실행은 입자 64개, 전단률 100 s⁻¹, 시간 간격 10⁻⁶ s,
**3 step**이다. 힘 잔차, 응력–동력 일관성, 최종 파일 저장을 확인한 연결 검사이며
정상상태 점도나 최대 충전율 근처의 점도 검증은 아니다.

| 실행 | AR | 체적분율 | step 수 |
|---|---:|---:|---:|
| AR2 | 2 | 0.24 | 3 |
| AR5 | 5 | 0.10 | 3 |
| AR10 | 10 | 0.06 | 3 |
| AR20 | 20 | 0.04 | 3 |

각 폴더의 `parameters.txt`, `in.slurry`, `particles.data`, `history.csv`,
`log.lammps`, `final.data`는 원본 바이트를 그대로 복사했다.
`evidence_manifest.json`에 SHA256과 크기를 기록했다. 실행 시간과 force/power 검사
결과, 실행 당시 코드·실행 파일 식별 정보는 상위의
`lammps_smoke_energy_serial.json`에 있다. 큰 trajectory와 실행 바이너리는 포함하지 않았다.

기록에 있는 `/workspace/scratch/...` 경로는 **당시 실행 환경의 경로**다.
다른 컴퓨터에서 그 경로를 사용하면 안 된다. 기록을 다시 분석하는 데는 LAMMPS가
필요하지 않으며, 저장소 루트에서 다음 명령을 실행하면 된다.

```bash
python3 validation_general/run_lammps_smoke.py \
  --analyze-existing \
    validation_general/smoke_energy_serial/AR2 \
    validation_general/smoke_energy_serial/AR5 \
    validation_general/smoke_energy_serial/AR10 \
    validation_general/smoke_energy_serial/AR20 \
  --output /tmp/general_saved_evidence_analysis.json
```

새 실행에는 `validation_general/run_lammps_smoke.py`의 `--lammps`, `--plugin`,
`--packer`, `--workdir`에 현재 설치 경로를 전달한다. 이 기록과 같은 길이의 4개
실행을 요청하려면 `--all-ar --steps 3`을 추가한다. 클러스터 제출은 저장소의
`lammps_slurry/run_general_cpu.sbatch`를 사용한다. 저장된 입력을 직접 실행하려면
새 작업 폴더에 `in.slurry`와 `particles.data`를 복사하고, 현재 plugin 경로를
`-var slurry_plugin`으로 전달한다. 기록 폴더에 출력을 덮어쓰지 않는다.
