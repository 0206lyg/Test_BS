# AR20 한 배치 — full17 개발·검증 현황

> **이전 단계의 기록입니다.** 아래 75 nm 고정 시험과 당시의 다음 개발 순서는 기존 유한 기저 방식의 진단 이력이며 현재의 필수 anchor 조건이 아닙니다. 현재 전략과 판정은 [AR20_INDEPENDENT_STATUS.md](AR20_INDEPENDENT_STATUS.md)를 따릅니다. 과거 수치·검사 결과는 변경하지 않고 보존합니다.

후보 release: `20261018-janoschek-hardfloor-full17-candidate` · 기준 commit: `b8f78900f8c180d6c37e97f1a2682ff40d6bbd0b`

현재 **코드 연결·대수 검사는 통과했으나 직접 Stokes 기준 계산의 정확도는 미통과**다. 아래 수치는 미수렴 L22 행렬을 사용한 진단이며, 1 nm 유체역학 정확도나 실제 점도 오차를 나타내지 않는다. 다른 AR이나 전체 저항표 생성은 실행하지 않았다. 구현 후보판과 재현 가능한 실패 진단을 보존한 기록이며, 수렴한 calibration anchor는 아직 확보하지 못했다.

고정 배치: AR=20, c=150 nm, θᵢ=0.0049245303592113559, θⱼ=0.041010932014513654, φ=1.7016960206944713 rad. 직접 계산 기준 간격은 h/c=0.5, 즉 75 nm다. 새 regularization은 h_eff=max(h,1 nm)이며, 3 nm를 4 nm로 늘리지 않는다.

## 1. 근접 항과 경계 조건

Janoschek Appendix의 raw 근접 연산자를 full17로 연결했다. 고유값 clipping이나 물리 행렬의 제곱 완성은 사용하지 않는다. 이후 원문 대조에서 Appendix A에 실제로 제시된 16개 독립 leading 저항 성분의 구현을 확인했다. Normal–normal K33의 subleading log(h)는 논문 자체의 의도적 생략이며 Appendix 식의 전사 누락이 아니다. 유한 간격항 전체를 해석적으로 복구한 모델도 아니다. 첫 anchor의 상수 잔차는 생략된 log 항이나 간격에 따라 변하는 유한항의 완전한 복원을 뜻하지 않는다. Full17 strain·stresslet 확장의 대수 일관성과 독립 물리 정확도도 별개다. [원문·구현 대조 기록](JANOSCHEK_APPENDIX_A_AUDIT.md)

[근접 연산자](../lammps_slurry/src/slurry_singular_pair.h) · [근접 항 시험](../lammps_slurry/tests/test_singular_pair.cpp)

## 2. 직접 Stokes anchor: 현재 막힌 단계

L22에서는 실제 고립 self 블록을 사용했다. 작은 선형계 잔차나 reciprocity만으로 경계 조건 충족을 판정하지 않고, 해를 맞춘 점 이외의 표면에서 실제 속도를 검사했다.

| 검사 | 현재 값 | 판정 |
|---|---:|---|
| L18 → L20 full17 최대 상대 에너지 변화 | 0.77528% | 행렬 변화 1% 이내 |
| L20 → L22 full17 최대 상대 에너지 변화 | 0.21459% | 행렬 변화 1% 이내; 경계조건 실패로 전체 미통과 |
| L22 raw reciprocity | 4.33×10⁻⁷ | 작지만 경계 정확도 인증이 아님 |
| 표면 전체 검사점의 최대 scaled no-slip 오차 | 66.8477% | 실패 |
| 좁은 틈 검사점의 최대 scaled no-slip 오차 | 58.1625% | 실패 |

L20 → L22에서 raw 힘·토크·stresslet 행 블록의 Frobenius 변화는 각각 0.05436%, 0.08730%, 0.05457%다. 연속 두 차수 비교가 1% 아래여도 실제 경계 속도 오차가 크므로 이를 물리 정확도의 증거로 채택하지 않는다.

오차에는 표면 기저의 교차 유동 표현·투영 문제가 남아 있다. 같은 배치에서 실제 self 응답, source 각도 적분, source 깊이, 기저 차수, 직접 경계 least-squares를 순서대로 수정·재검증했다. 이 단계가 통과하기 전에는 해당 anchor를 물리적으로 검증된 표 생성 입력으로 채택하지 않는다.

마지막 L22 직접 경계 least-squares 풀이에서도 global 최대 no-slip 오차 **34.793%**, 좁은 틈 최대 **31.514%**로 실패했다. 평균적인 weighted boundary residual도 4.829%이며, L16 → L22 least-squares 저항의 에너지 변화는 20.606%다. 기존 Galerkin L22와의 전체 저항 차이도 11.659%다. 두 풀이 모두 검증된 anchor로 채택하지 않았다. 새 풀이의 작은 오차값만 보고 기존 계산을 교체하지 않았다.

source 각도 적분 256 → 512는 전체 저항을 약 0.000307%만 바꾸고, source 깊이를 0.5 → 0.25로 바꾸어 고립 기저 fitting을 개선해도 pair 경계 오차는 거의 유지됐다. 상대 spheroid의 rim이 투영되는 부근에서 교차 유동을 표현하는 유한 기저의 해상도가 지배적인 병목이다. 다음 수정에는 그 부근의 국소·방향별 기저 보강 또는 독립적인 적응형 Stokes/BEM 기준해가 필요하다. 허용 오차를 완화하거나 anchor 간격을 옮겨 통과 처리하지 않았다.


[L22 행렬](ar20_full17_diagnostics/L22_h050_t3_p256.json) · [독립 평가점의 no-slip 검사](ar20_full17_diagnostics/L22_h050_t3_p256_boundary.json) · [차수 변화](ar20_full17_diagnostics/L16_L22_refinement.json)

## 3. full17 유지와 153성분 저장

기존 full17→11은 상대운동을 고정하고 공통운동 여섯 개를 최소화하는 축약이었다. 따라서 지정한 두 입자 속도에 대한 full17 응답의 가역 압축이 아니다. 새 경로는 두 입자의 병진·회전 12개와 strain 5개를 유지하고, 대칭 행렬의 독립 성분 153개를 저장한다. global 입자 속도 미지수는 증가하지 않는다.

같은 L22 행렬을 대상으로 기존 11성분을 고립 저항과 다시 조립하면 최대 상대 에너지 차이는 72.2549%, 공통운동 응답 차이는 58.8439%, 물리적으로 고정한 입자의 intrinsic strain stress 차이는 26.1581%였다. 이는 **미검증 행렬 내부의 축약 손실 진단**이다. 힘·토크가 자유인 strain stress는 대수적으로 보존되므로 그 검사만으로 축약 손실을 발견할 수 없다.

새 full17 correction은 `직접 Stokes − 고립 저항 − 근접 항`의 첫 anchor 잔차를 근접 항에 다시 더한다. 직접 계산 구간에 별도 film 항을 더하지 않는다. 상수 잔차 연결의 물리 정확도는 아직 다른 간격의 수렴한 직접 계산과 비교하지 못했다.

| full17 연결 검사 | 결과 |
|---|---:|
| 75 nm anchor 복구 최대 상대 에너지 오차 | 2.18×10⁻¹² |
| 75, 37.5, 3, 1 nm에서 total full17 최소 고유값 | 10.1197–10.7333, 모두 양수 |
| 같은 네 간격에서 153성분 log 저장·복구 에너지 오차 | 최대 2.38×10⁻¹⁰ |
| force/torque/stresslet 좌표 연결 오차 | 최대 8.91×10⁻¹⁸ |
| 독립 long-double 에너지 미분 검사 오차 | 최대 4.80×10⁻¹⁰ |

153성분 검사는 한 절점의 저장·복구 검사다. 전체 표 생성이나 방향·간격 보간 검사가 아니다. total SPD도 물리 정확도 또는 다체 조립 SPD의 증명은 아니다.

[축약 진단](ar20_full17_diagnostics/L22_h050_reduction_audit.json) · [full17 연결·저장 진단](ar20_full17_diagnostics/L22_v4_full17_matching_audit.json) · [153성분 저장 코드](../lammps_slurry/src/slurry_spd_table17.h)

## 4. runtime 연결과 다음 통과 조건

full17 correction은 실제 입자 중심의 비아핀 속도 좌표에 직접 적용한다. 힘·토크·intrinsic stresslet은 같은 quadratic에서 얻는다. 물리적으로 고정한 입자의 strain 시험은 Uᵢ=0을 사용하므로 비아핀 입력이 −E·xᵢ이며, 앞 12개 입력을 0으로 놓는 시험과 구분한다. intrinsic stresslet과 원점 기준 force dipole도 구분한다.

유효 간격 1 nm를 사용하면서 실제 간격을 0, 0.5, −0.15 nm로 바꾼 작은 probe에서 full17 SPD와 에너지 미분 일관성을 확인했다. −0.15 nm는 overlap regularization의 수치적 안정성 검사이며, 겹친 고체 사이의 실제 유체 문제를 뜻하지 않는다. 별도 LAMMPS/MPI 및 다체 조립 검증은 이 probe로 대체하지 않는다.

다음 순서는 **같은 75 nm 배치의 no-slip·해상도 수렴 복구 → 수렴한 직접 계산과 근접 모델의 다른 간격 비교 → 같은 AR20의 다른 방향 및 보간 검사 → 소규모 runtime 검사**다. 현재 보고서의 `physical_accuracy_qualified`는 false이며, 다른 간격의 직접 기준이 없으므로 `physical_overlap_error_gate_passed`는 null이다.

재현 예시 — 저장소 루트에서:

```bash
g++ -O1 -std=c++17 -I lammps_slurry/src -I lammps_slurry/vendor/eigen3 validation_basis/check_ar20_full17_matching.cpp -o /tmp/check_ar20_full17_matching
/tmp/check_ar20_full17_matching validation_basis/ar20_full17_diagnostics/L22_h050_t3_p256.json
```

수렴한 다른 간격의 JSON을 뒤에 추가하면 모든 제공된 간격을 비교한다. anchor와 같은 간격의 재구성은 held-out 통과 횟수에 포함하지 않으며, 모델 차이가 1%를 넘으면 별도 gate를 실패시키고 종료 코드 2를 반환한다.

## 구현 계약 검사와 배포 범위

- standalone C++ 대상 11개 빌드 성공, CTest 12/12 통과. hard-floor, full17 저장, 생성기 직렬화, Janoschek 계수, 힘·토크·stress 대수 검사를 포함한다.
- 합성 입력 준비 검사 27개 중 25개 통과, 실제 packer가 필요한 2개는 생략했다.
- 실제 표 생성 driver에 합성 SPD 입력을 연결해 45개 청크의 직렬/4스레드 결과가 바이트 단위로 같고, 24개 완료 청크에서 중단·재개가 일치함을 확인했다. 물리적인 전체 표를 만든 것은 아니다.
- 프로세스 복구 5/5, runtime snapshot 1/1 통과. 소스 release hash 검사도 통과했다.
- LAMMPS 헤더·설치가 없어 실제 plugin 빌드 및 MD 실행은 수행하지 않았다.
- 과거 AR10 film/reduced11 회귀 검사는 보존하고 CMake의 명시적 `SLURRY_ENABLE_HISTORICAL_MODEL_TESTS=ON`에서만 등록한다. 이번에는 실행하지 않았다.

[구현 검사 결과](ar20_full17_diagnostics/implementation_contracts_result.json) · [현재 체크포인트 결과](../validation_finite_gap/checkpoint_runtime_full17_result.json) · [직접 해 진단·재현 명령](AR20_RECORDED_POSE_VALIDATION.md)

수정본은 Git commit `b8f78900f8c180d6c37e97f1a2682ff40d6bbd0b`에 적용하는 개발 후보판이다. 기존 표·체크포인트는 새 모델에서 거부한다. 새 표 생성은 직접 해와 연결 오차의 물리 검증이 통과한 다음 단계다. GitHub에는 push하지 않았다.
