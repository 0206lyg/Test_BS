# Test_BS — compact spheroid hydrodynamics

새 실행 경로는 `fix slurry/bridge`입니다. AR을 입력받아 검증한 compact lubrication v4, 주기 중거리 mobility, 접촉·마찰을 함께 풀고 입자의 위치·배향과 xz 전단 응력을 계산합니다. 기저나 저항표를 먼저 생성하지 않습니다. 현재 solver는 접촉력을 독립 변수로 유지하는 proximal Newton + FGMRES이며, 최종 수렴은 원래 힘·mobility 방정식으로 검사합니다.

```bash
sbatch build_slurry_cpu.sbatch
# 빌드 완료 후, 짧은 연결 실행:
sbatch lammps_slurry/run_bridge_cpu.sbatch \
  --output runs/ar10_solver_check --ar 10 --count 500 --phi 0.24 \
  --steps 10 --output-interval 1 --average-start-strain 0
```

기본 물성은 두께 300 nm, 매질 점도 0.01 Pa·s, 유체 gap floor 1 nm, 접촉 시작 간격 3 nm, 마찰계수 0.1입니다. 표면 기하는 실제 간격을 사용하며 스칼라 gap 함수에만 floor를 적용합니다. 기본 전단률은 100 s⁻¹, timestep은 1e-6 s입니다.

- [실행·출력·점도 평균](docs/BRIDGE_RUN.md)
- [연결식·단위·접촉·응력 구현 명세](docs/COMPACT_BRIDGE_MODEL.md)
- [주기 중거리의 추가 근사와 해상도·비용](docs/PERIODIC_MIDDLE.md)
- [삭제 가능한 기존 폴더](docs/LEGACY_CLEANUP.md)
- [500개 입자 solver 수정·실행 결과](docs/SOLVER_FIX.md)
- [500개 입자 원시 결과·재현 입력](validation_solver500/README.md)
- [이전 모델 통합 검증 기록](validation_bridge/README.md)

중거리 기본값 `--spectral 2`는 저비용 근사입니다. 고파수의 입자 간·주기 이미지 항은 생략하고 단독입자 tail을 적분합니다. `--spectral 4` 또는 `8`은 그 생략 영향을 줄이는 설정입니다. `--spectral-tolerance 1e-8`은 Fourier 절단 허용값이며 전체 유체력 오차가 아닙니다. 기존 두 문제 배치의 변화는 위 중거리 문서에 수치로 기록했습니다.

`history.csv`는 매질·유체·접촉 응력과 순간 점도, 매 시간단계에 누적한 평균, 소산, solver 잔차 및 비용을 출력합니다. `trajectory.lammpstrj`는 입자별 유체력·토크·stresslet과 접촉력·토크를 별도로 출력합니다. 평균창은 `--average-start-strain`으로 지정하며 짧은 실행의 점도를 정상상태 값으로 해석하지 않습니다.

이전 `run_slurry_cpu.sbatch`, `finite_gap*` 모델과 표 생성 도구는 기존 결과 재현을 위해 남겼습니다. 새 모델 실행에는 `run_bridge_cpu.sbatch`를 사용합니다. 이번 변경에서 기존 폴더를 삭제하지 않았습니다. `final.data`는 기하 저장이며 마찰 이력을 보존하는 restart가 아닙니다.
