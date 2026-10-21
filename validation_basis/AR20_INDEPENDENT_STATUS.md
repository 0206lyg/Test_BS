# AR20 독립 Stokes 기준해 개발 현황

## 2026-10-19 잠정 입력을 이용한 개발 완료

사용자 결정에 따라 9963절점 해의 최대 경계 잔차 **0.753223%를 그대로
기록한 채 개발용 기준으로 채택**했다. 0.1%는 이번 구현 진행의 필수 문턱이
아니다. 보고서의 엄격 수렴 판정을 바꾸지 않고, 실제 입력 → 간격 모델 →
153성분 표 → 기존 runtime의 힘·토크·stresslet까지 실행했다.

1–150 nm에 222개 적응 간격점을 쓴 한 방향 표가 생성됐다. 선택한 후보식
대비 표 보간 최대 상대 에너지 오차는 검사점에서 8.52×10⁻⁷이며, 실제
runtime의 힘·토크·stresslet 비교는 모두 통과했다. 3.15 nm는 그대로 쓰고
1 nm 아래에서만 제한한다. 이 전달 오차는 물리 오차가 아니다.

이 과정에서 기존 상수 잔차 연결식의 음의 소산을 확인했다. 새
`JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1` 후보를 명시적으로 선택해
양의 소산을 유지하는 연결을 구현했다. 기존 식과의 차이는 상당할 수 있으며
현재 정확도 추정치가 없다. **정밀 anchor 하나를 얻는 것만으로 검증이
끝나는 것은 아니다. 별도 간격의 독립 해로 새 연결식의 힘·토크·stresslet을
비교하는 작업이 남아 있다.**

[실행·모델·교체 절차](independent_stokes/PROVISIONAL_PIPELINE.md) ·
[실제 표 생성과 runtime 결과](independent_stokes/results/provisional_pipeline/actual_candidate_003/pipeline.json)

다른 AR의 계산이나 실제 전체 방향 표 생성은 수행하지 않았다. 일반화된
입력과 전체 격자 경로는 구현 검사 대상이며, 현재 자료가 한 방향이라는
사실을 유지한다. 아래는 이전 독립 풀이와 엄격 수렴 검사의 보존 기록이다.

## 독립 풀이와 이전 수렴 검사 기록

**이전 엄격 기준에 따른 판정: 고립 입자의 힘·토크·stresslet 추출은 통과했지만, 두 입자 기준해와 lubrication 연결의 물리 정확도는 아직 미통과다.** 가장 잘 분해된 완료 결과는 여전히 9963절점 계산이며 최대 독립 표면 no-slip 오차 0.753223%가 0.1% 기준을 넘는다. 같은 150 nm에서 p8·159패널·12879절점 out-of-core v2는 전체 행렬을 932.935845초에 조립했으나 LU 전 자원 검사에서 중단됐다. 메모리 처리와 사전 검사를 수정한 v3 재시도도 시작 단계의 자원 검사에서 중단됐다. 이 두 재시도에서 새 밀도·힘·토크·stresslet·경계 정확도 결과는 얻지 못했다.

최신 재시도와 근거 파일·소스 해시를 모은 [2026-10-19 실행 요약](independent_stokes/results/resumed_ar20_attempt_20261019.json)을 함께 보존했다.

## 현재 전략과 범위

생산 모델은 실제 매끈한 표면 간격에 `h_eff = max(h, 1 nm)`를 적용한다. 기계적 roughness/contact 시작 간격 3 nm는 별도다. 따라서 실제 간격 3.15 nm의 lubrication 계산 간격도 3.15 nm다.

현재 Janoschek 구현은 **논문 Appendix A에 제시된 16개 독립 leading 저항 성분을 구현한 후보 모델**이다. 원문의 식과 코드를 대조하고 같은 AR20 배치에서 성분별 검사를 통과했다. Normal–normal K33의 subleading log 항은 논문 자체가 의도적으로 제외한 항이며, 이를 Appendix 식의 전사 누락으로 해석하지 않는다. 다만 모든 발산항이나 유한 간격항을 포함한 해는 아니고, 상수 anchor 잔차도 이를 완전히 복구하지 않는다. 논문의 국소 6좌표 저항에서 확장한 full17 strain·stresslet 경로는 에너지 미분의 대수 일관성을 확인했으나 독립 물리 검증은 남아 있다. 이 검사나 153성분 저장 검사는 1 nm 정확도를 뜻하지 않는다. [원문·구현 대조 기록](JANOSCHEK_APPENDIX_A_AUDIT.md)

기록된 AR20 방향 하나를 유지하며, 기존 유한 기저 방식과 독립적인 표면 경계적분 기준해를 개발했다. 정확한 spheroid 기하, 특이·준특이 적분, combined-layer 표현을 사용하고, 밀도를 traction으로 오인하지 않도록 힘·토크·intrinsic stresslet을 해당 표현의 모멘트 식으로 추출한다.

**75 nm는 과거 실패 시험의 간격이며, 필수 matching anchor가 아니다.** 현재 독립 풀이의 수렴 문제를 같은 방향의 150 nm에서 조사하고 있다. 이 간격도 검증된 anchor로 채택한 것이 아니다. 수렴한 기준해와 별도 간격의 비교를 확보한 뒤 연결 간격을 정한다. 다른 AR이나 전체 표 생성은 실행하지 않았으며, **1 nm 직접 기준해와 모델 정확도는 검증하지 않았다.**

## 확인한 결과

| 항목 | 결과 | 판정 |
| --- | --- | --- |
| 수정된 quadrature backend의 고립 입자 힘 오차 | 약 0.001265% | 고립 모멘트 검사 통과 |
| 같은 고립 입자의 토크 오차 | 약 0.002680% | 통과 |
| 같은 고립 입자의 intrinsic stresslet 오차 | 약 0.002374% | 통과 |
| 고립 입자의 독립 표면 검사점 최대 no-slip 오차 | 약 0.04119% | 0.1% 기준 통과; 표면 전체의 엄밀한 상한은 아님 |
| 대형 계산 전 가장 작은 pair 최대 no-slip 오차 | 2.58986% | 0.1% 기준 미통과 |
| 72패널에서 p6 → p8 적분 저항의 최대 상대 에너지 변화 | 약 0.043% | 작은 변화만으로 경계조건·물리 수렴을 인정하지 않음 |
| 최초 수정 backend, 9963절점, 비등방 격자 | 8 GiB cgroup 한도에서 OOM 종료 | 실패 기록 보존; 물리 결과 없음 |
| 같은 9963절점, 정확한 블록별 조립, `mmap-dense` | 1224.97초에 계산 완료 | 물리 정확도는 아래 경계오차로 미통과 |
| 완료된 9963절점 pair의 최대 no-slip 오차 | 전체 0.753223%, gap patch 0.229996% | 둘 다 0.1% 기준 미통과 |
| 같은 풀이의 선형 상대 잔차 | 1.44376×10⁻¹³ | 작은 잔차가 표면 수렴을 보증하지 않음 |
| raw reciprocity 오차 | Frobenius 2.44165×10⁻⁶, 에너지 기준 7.18283×10⁻⁶ | 통과하나 pair 정확도는 미통과 |
| 실제 완료 행렬의 153성분 저장·복구 에너지 오차 | 2.65746×10⁻¹² | 대수 검사만 통과 |
| 후속 p7·159패널·10176절점, private COW | 전체 조립 674.045초 후 LU 직전 중단 | 필요 7.81984 GB > 여유 7.37424 GB; 물리 결과 없음 |
| p8·159패널·12879절점, out-of-core v2 | 전체 조립 932.935845초 후 LU 전 중단 | 필요 2.279 GB > 여유 1.211 GB; 새 밀도 없음 |
| 같은 p8 격자, out-of-core v3, LU 블록 512/256 | 시작 단계 메모리 검사에서 중단 | 필요 0.830 GB > 여유 0.668 GB; 조립·풀이 미실행 |
| v2 → v3, 같은 배치의 작은 300절점·17개 RHS 비교 | 원래 행렬 A가 비트 단위로 일치; 밀도 차이 1.09449×10⁻¹⁴, raw17 차이 2.72680×10⁻¹⁴ | 저장·메모리 처리 구현 검사만 통과 |

고립 결과는 기록된 방향의 두 입자를 각각 분리한 검사다. 두 입자 상호작용의 정확도를 대신하지 않는다. 이전 p8 adapt1과 새 완료 행렬 사이의 raw 힘·토크·stresslet 행 블록 Frobenius 변화는 각각 0.00104244%, 0.00215753%, 0.00407120%이며 최대 상대 에너지 변화도 약 0.00623%다. **이 작은 적분량 변화로 남아 있는 표면 오차를 통과 처리하지 않았다.** 두 기록은 backend도 다르므로 동일 backend의 순수 격자 수렴으로 해석하지 않는다.

근거: [고립 입자 0](independent_stokes/results/isolated0_p6_l1_rim1_patched.json) · [고립 입자 1](independent_stokes/results/isolated1_p6_l1_rim1_patched.json) · [이전 pair p8](independent_stokes/results/pair_h150_p8_adapt1.json) · [OOM 실패 기록](independent_stokes/results/pair_h150_p8_anisotropic_patched.json) · [수정 후 완료 계산](independent_stokes/results/pair_h150_p8_anisotropic_chunked.json) · [검증 요약](independent_stokes/results/pair_h150_p8_anisotropic_chunked_validation_summary.json) · [실제 153성분 검사](independent_stokes/results/pair_h150_p8_anisotropic_chunked_encoding_algebra.json)

## 수정한 문제와 남은 원인

상위 라이브러리의 quadrature 선택 함수에서 초기화되지 않은 `norder`를 읽는 결함을 확인했다. p≥4용 기존 분기를 명시적으로 선택하도록 한 객체만 수정해 별도 라이브러리를 만들었고, 원본은 보존했다. 그러나 수정 전후 동일 밀도의 경계 평가가 거의 같았으므로 **이 결함을 남은 pair 오차의 주원인이라고 결론내리지 않았다.** 적분 허용치가 내부 구간으로 묶이므로 같은 구간의 값만 바꾸는 것은 독립적인 적분 수렴 시험이 아니다.

FMM 경로에서는 얇고 긴 패널의 far 적분 해상도가 충분하지 않은 문제가 관찰됐다. 이번 대형 기준 비교는 그 경로를 피하고 file-backed dense 조립과 LU를 사용한다. 이는 메모리 중복을 줄이는 계산 방법이며 정확도 허용치를 완화하는 조치가 아니다. 국소 경계 밀도와 표면·적분 해상도의 수렴은 계속 별도로 확인해야 한다.

최초 file-backed 계산도 native quadrature가 전체 작업 배열을 만들면서 8 GiB 제한을 넘었다. 이를 표적 절점 128개씩 같은 정확한 quadrature를 조립하도록 고쳤다. 같은 AR20의 2352절점 비교에서 17개 해 전체의 상대 차이는 기존 dense 풀이 대비 약 2.29×10⁻¹⁵였다. 이는 조립 방식의 일치 검사이며 큰 격자의 물리 수렴을 뜻하지 않는다. [블록 조립 비교](independent_stokes/results/mmap_chunked_p6_check.json)

Out-of-core v2는 전체 행렬을 파일에 조립했지만 해시 계산 이후 LU의 메모리 검사에서 허용되지 않았다. 검사는 clean inactive file cache만 회수 가능한 것으로 인정하는 보수적인 방식이다. 당시 anonymous 메모리는 약 0.166 GB, file 메모리는 8.228 GB이며 그중 active file은 6.884 GB였다. 따라서 이 실패는 제한된 작업 배열 자체가 8 GiB에 들어갈 수 없다는 증명도, 모든 file cache가 자기 해시 계산에서 생겼다는 증명도 아니다. v3는 자기 작업 파일의 읽은 구간에만 cache 해제 조언을 적용하는 해시 방식과, 조립·LU·잔차 평가 중 최대 요구량을 시작 전에 검사하는 절차를 추가했다. Cache 조언은 실제 해제를 보장하지 않는다. 수정된 `out_of_core_dense.py`의 SHA256은 `112e816727006d8dee92e331a59e8bf035c4481dab64739fd2f2876cf6d99f43`이다. 작은 실제 Stokes 비교에서 v2와 원래 A가 같고 원래 연산자로 재계산한 최대 잔차는 4.08190×10⁻¹⁴였다. 이 구현 검사는 12879절점 해의 완성을 뜻하지 않는다. [v2 실패](independent_stokes/results/pair_h150_p8_159_ooc_buffered_retry.json) · [v3 구현 비교](independent_stokes/results/ooc_v3_assembly_equivalence.json) · [v3 자원 검사 실패](independent_stokes/results/pair_h150_p8_159_ooc_v3_resource_retry.json)

자원을 맞추기 위한 격자 축소도 저장된 9963절점 밀도로만 조사했다. 높은 경계오차가 있던 24개 영역을 유지하고 밀도 jump 변화 지표를 2×10⁻⁴ 이하로 제한하면 159→157패널까지만 줄어 행렬 하나가 여전히 11.644 GB다. 138패널 이하에는 이 후보 집합에서 약 0.008506의 지표 허용이 필요해 채택하지 않았다. 이 지표는 전체 속도 오차의 상한도, 새 Stokes 풀이도 아니다. [축소 진단](independent_stokes/results/pair_h150_p8_coarsened_hotspots_preserved_projection_audit.json)

FMM far order를 48 높인 별도 300절점 저장 밀도 검사는 첫 RHS에서 연산자 적용 오차 2.82263×10⁻⁷을 보였지만, 예상 밖의 native RSS 약 1.77 GiB 때문에 중단했다. 이는 512 MB 작업량 추정이 작았다는 근거이며 12879절점 FMM 계산의 불가능성을 뜻하지 않는다. 17개 RHS 전체 검사나 새 선형계 풀이를 완료한 결과도 아니다. [FMM 한 RHS 진단](independent_stokes/results/saved_n300_fmm48_action_comparison.json)

완료된 큰 격자의 collocation 절점 36개를 더 엄격한 quadrature로 재평가한 최대 scaled 오차는 약 2.16×10⁻⁹였다. 검사한 절점의 조립·적분은 일치하지만 그 사이 독립 검사점의 오차는 여전히 크다. 따라서 절점 사이 경계 밀도의 표현 해상도가 다음 수정 대상으로 타당하다는 근거가 되며, 전체 표면 오차의 원인이 확정됐다는 뜻은 아니다. [36절점 검사](independent_stokes/results/pair_h150_p8_anisotropic_chunked_collocation_probe.json)

별도의 **기하만 계산한** 검사에서는 75 nm에서 국소 이차 간격의 특성 타원 `½ xᵀHx=h`가 일부 방향에서 입자 표면의 평면 투영 밖으로 나갔다. 720개 각도 표본 중 304개는 두 표면 중 적어도 하나의 투영 밖이었다. 이는 그 위치에서 두 표면 사이의 수직 간격을 정의할 수 없다는 기하 정보다. **lubrication 힘·토크·stresslet 오차의 크기나 특정 anchor의 적합성을 입증하는 결과가 아니다.** [정확한 간격 기하 진단](independent_stokes/results/ar20_exact_gap_geometry.json)

두 입자의 병진·회전 12개와 strain 5개를 모두 유지한다. 대칭 17×17 행렬의 153성분 저장은 가역적인 대수 표현이다. 기존 11좌표 축약에서 빠지는 공통 운동을 보존하지만, 잘못된 Stokes 행렬을 정확하게 만드는 방법은 아니다.

근거: [quadrature 수정과 제한](independent_stokes/patches/README.md) · [같은 밀도의 backend 비교](independent_stokes/results/pair_h150_p8_adapt1_cross_library_quadrature_comparison.json)

## 구현 검사와 다음 판정

수렴 판정은 최신 세 격자의 라이브러리·backend·geometry 해시와 연산자 설정이 같아야 통과하도록 강화했다. 기존 세 기록은 경계오차뿐 아니라 이 동일성 조건도 충족하지 않는다. 서로 다른 구현의 작은 행렬 차이를 동일 구현의 격자 수렴으로 인정하지 않는다.

최신 Python 검사는 **49개 중 48개 통과, 선택적 benchmark 1개 생략**이다. 생산 수식이 변경되지 않은 이전 관련 CTest 13개 통과 기록도 보존했다. 이전의 범위 제한 없는 CTest 시도에서는 미빌드 실행 파일 때문에 15개가 실행되지 않았으므로 전체 스위트 통과로 보고하지 않는다. 이 검사들은 구현·입출력 계약 검증이며 pair 물리 정확도와 구분한다. [실행 결과 기록](independent_stokes/results/implementation_validation.json) · [최신 Python 로그](independent_stokes/results/independent_python_tests_ooc_v3.log) · [관련 CTest 기록](independent_stokes/results/related_ctest.xml)

진단 당시 작업 디스크 여유도 약 9 GB였고 이후 더 줄어, 12879절점 행렬과 0.5 GB reserve에 필요한 약 12.44 GB보다 작다. 다음 실행은 안정적으로 **32 GiB 이상 메모리와 32 GB 이상 여유 작업 디스크**를 확보한 CPU 환경에서 같은 p8·159패널 배치를 유지하는 것이 타당하다. 이는 자원 권고이며 정확도 보장이 아니다. 기존 `mmap-dense`/COW 경로나 v3의 512/256 블록 경로를 사용할 수 있고, 실제 메모리·디스크 검사는 그대로 적용한다. [이전 p7 실패](independent_stokes/results/pair_h150_p7_159_cow.json) · [out-of-core 실행 절차](independent_stokes/OUT_OF_CORE_SOLVER.md)

다음 재현 명령은 아직 실행하지 않았다. 같은 배치의 독립 표면 no-slip, raw reciprocity, 힘·토크·stresslet, full17 에너지와 동일 연산자의 세 격자·적분 수렴을 다시 판정해야 한다. 그 기준을 통과한 anchor와 별도 간격의 기준해가 있어야 근접 모델의 연결 오차를 평가할 수 있다. 이 엄격 물리 판정은 유지하되 잠정 표 개발은 위의 별도 경로로 진행했다. 다른 AR·전체 방향 표의 실제 계산과 suspension 점도 검증은 이번에 수행하지 않았다.

[실행·비교 절차](independent_stokes/README.md) · [이전 유한 기저 진단 기록](AR20_RECORDED_POSE_VALIDATION.md) · [이전 full17 구현 기록](AR20_FULL17_STATUS.md)
