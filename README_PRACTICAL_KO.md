# N=1000 실용 실행 모드

이번 실행 파일은 `run_slurry_practical_cpu.sbatch`입니다. 압축을 푼 소스 디렉터리에서 빌드를 제출합니다.

```bash
sbatch build_slurry_cpu.sbatch
```

빌드 로그의 `Build complete`를 확인한 다음 실행합니다.

```bash
sbatch run_slurry_practical_cpu.sbatch
```

기본값은 N=1000, AR=4, 체적분율 0.24, 크기비 1.4의 이분산 입자입니다. 전단률 100/s, dt=1 µs, 500,000 step으로 목표 strain은 50입니다. 출력 간격은 400 step이며, 변경할 설정은 실행 파일 상단에 있습니다. 이 목표 strain 50까지의 완료를 이번 국소 시험으로 보장하지는 않습니다.

0.67 nm 접착 범위와 기존 물성은 유지했습니다. 물의 점도는 0.00089 Pa·s, kn=900 N/m, kt=kr=300 N/m, sliding/rolling 마찰계수는 각각 10입니다. 기존 cohesion/background 포텐셜을 그대로 사용합니다.

`surface_integrator practical`은 pair별 힘·토크 변화, 미세한 일 오차, frozen cap 변화 때문에 전체 step을 쪼개지 않습니다. 내부 풀이를 최종 허용오차보다 무조건 10배 더 엄격하게 계산하는 조건과 pair별 내부/끝점 힘 일치 조건도 사용하지 않습니다. 실제 끝점 접촉력으로 계산한 전역 힘 평형은 검사합니다. 비선형 및 선형 상대 허용오차는 각각 1e-4입니다.

NaN, 접촉 후보 누락, 실제 solver 실패 또는 작은 반축의 5%를 넘는 겹침은 계속 검사합니다. 이런 실제 실패에만 제한적으로 복구하며, 매 새 macro step은 원래 dt로 다시 시작합니다. 이번 N=1000 측정에서는 분할과 재시도가 한 번도 없었습니다.

검증 기록은 `validation_practical_20261102/FINAL_REPORT.md`에 있습니다. 인력이 있는 계는 300초 계측 동안 607 step, strain 0.0607까지 진행했습니다. 이는 정상상태 유변학 검증이 아니라, 불필요한 제한을 제거한 N=1000 진행 및 계산 비용 시험입니다. 기존 응력 곡선을 정답으로 삼아 맞추지 않았습니다.

`max_sliding_cap_excess_N` 등의 기존 출력은 끝점 normal load와 frozen cap의 차이를 포함합니다. 이 값만으로 실제로 푼 frozen-cap 마찰 법칙이 위반됐다고 해석하지 마십시오.
