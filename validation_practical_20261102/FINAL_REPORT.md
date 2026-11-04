# N1000 실제 진행 결과

**0.67nm와 물리 계수를 유지한 N1000 계산이300초 동안607step,607μs, strain0.0607까지 진행했다.** 내부 수용607회·거절0회였다. 요청한1ms/1000step은 완료하지 못했고, 원래300초 예산에서 종료했다. 예산 연장은 적용되지 않았다.

| 측정 범위 | 결과 |
|---|---:|
| 실제 진행구간 wall profile |298.281초|
| 전체 wall / CPU시간 |300.047초 /299.947초|
| 실제 진행 처리율 |2.035μs/초|
| 최대 overlap/minor semiaxis |0.15847%|
| 최대 상대 force/mobility 잔차 |9.9485×10⁻⁵|
| 최대 상대 대수적 power-balance 오차 |0.13505%|
| 비유한 값 / 기록된 solver 오류 |0 /0|

수용된 구간만 처리율에 포함했다. 마지막에 미리 계산한 다음 구간과 미완료 작업은 전체 실행시간에 남겼다. 전607step에서 macro dt=1μs를 유지했고, 실패 복구용 세분화는 발생하지 않았다. Trajectory는0,100,…,600step의7개 snapshot을 저장했다.

## 인력 on/off — 공통 첫100μs만 비교

같은1000입자, plugin, dt, 수치 설정에서 cohesion/background만 껐다.

| 첫100μs / strain0.01 | 인력 on | 인력 off |
|---|---:|---:|
| 진행구간 비용 |38.616초|44.571초|
| 수용 / 거절 |100 /0|100 /0|
| 평균 응력 |0.603997Pa|0.258550Pa|

이 관측 구간에서는 인력 도입에 따른 수백 배 비용 증가가 나타나지 않았다. On/off는 서로 다른 물리 궤적이므로 인력 자체가 항상 계산을 빠르게 한다는 뜻은 아니다.

기존과 동일한 fixture·물리 입력의 **첫1μs만** 비교하면89.9865→0.389585초, 내부 수용1360→1회다. 약231배는 첫 구간 비용비이며607μs나 장시간 전체의 속도 향상으로 환산하지 않았다. 첫 구간 응력은 기존보다9.52%낮았으며, 기존 응력값을 정답이나 목표값으로 삼지 않았다.

## 무엇을 완화했고 무엇을 유지했나

`practical`은 force/torque 변화율, surface-work, cap-lag 기반 시간 정확도 제한과 별도의 pair별 constitutive 일치 종료조건을 제거했다. 전체 잔차·선형 목표는 각각1e-4, 내부 tolerance 배율은1이다. 실제 endpoint의 IMEX force/mobility 균형, 유한성, 새 접촉 검증,5%overlap 기준은 유지했다. 실제 실패 시에만 dt/64까지·최대128회로 복구를 제한했다. 이번 nonlinear 한도는 console에서100회로 확인됐으며, `iterations=1000`은 별도의 선형 한도다.

AR4, phi0.24, kn900, kt=kr300, 마찰10, 인력 범위0.67nm와 물리 kernel은 유지했다. 그러나 시간 이산화 정확도까지 자동으로 증명되는 것은 아니다. 대수적 power balance와 signed contact work를 물리적 소산 또는 경로 일 보존으로 혼동하지 않았다.

## 사용·검증 범위

새 공개 실행기는 `run_slurry_practical_cpu.sbatch`다. 설정부를 확인한 뒤 `sbatch run_slurry_practical_cpu.sbatch`로 실행한다. 설정된500000step/strain50은 **실행 목표이며 검증 완료 범위가 아니다.** 이번 확인은 N1000·strain0.0607까지이며, 정상 유동·장시간 rheology·strain50은 검증하지 않았다.

73개 source/header와 실제 binary가 build manifest 및 두 실행 기록에 일치했다. 감사한 물리 kernel은 변경 전과 byte-identical이며,입력 생성 테스트는 31개 중 28개가 통과했고, 기존 packer 의존 테스트 3개는 건너뛰었다. Bundle 검사 8개와 C++ 끝점 제어·기존 nearfield_step 회귀 시험도 통과했다. 전체 원시 수치·구간·일·provenance는 `FINAL_REPORT.json`, 초기 구간 감사는 `practical_legacy_first_interval_comparison.json`에 있다.
