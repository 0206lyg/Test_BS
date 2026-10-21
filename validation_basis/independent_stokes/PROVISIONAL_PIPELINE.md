# 독립 Stokes 결과를 사용하는 잠정 표 개발 경로

현재 9963절점 AR20 결과를 개발용 입력으로 사용한다. 표본 경계 속도
최대 잔차 0.753223%와 RMS 0.026167%를 그대로 기록한다. 기존 0.1%
정밀 검증은 후속 정확도 평가이며, 이 경로의 개발을 막는 조건이 아니다.
이 값들은 힘·토크·stresslet·점도의 상대 오차 상한이 아니다.

## 이번 경로가 수행하는 일

`완료된 독립 JSON → 출처를 포함한 raw17 입력 → 명시적으로 선택한 연결식
→ 153성분 표 → 기존 Provider.make_pair → 힘·토크·intrinsic stresslet`

`prepare_reference_source.py`는 이전 기저 풀이를 호출하지 않는다. 원래
289개 raw 성분과 원본 보고서·checkpoint 해시를 보존한다. 실제 gap은
매끈한 표면 사이의 거리이며 `h_eff=max(h,1 nm)`를 적용한다. 기계적
roughness 3 nm는 여기서 빼지 않는다.

기본 임시 허용치는 표본 경계 잔차 1%다. 이는 명시적인 개발용 ceiling이며
수렴 인증이 아니다. 고립 모멘트 검증, 유효한 경계 평가, 선형 잔차,
geometry/units, 상반성, SPD와 출처 검사는 별도로 유지한다. 모든 산출물에
`physical_accuracy_qualified=false`가 남는다.

## 중간 간격에서 발견한 연결식 문제

기존 상수 보정식 `G(h)=G_anchor+L(h)-L(h_anchor)`는 이 실제 입력을
사용했을 때 37.5 nm에서 최소 고유값 약 −58849를 만들었다. 따라서
현재 입력을 그대로 사용한다는 결정만으로 전체 연결이 완성되지는 않는다.
실패한 식은 `--matched-model constant-residual`로 재현할 수 있다.

별도 후보 `--matched-model dissipative-increment`는 다음의 **새로운 유한
간격 연결 모델**이다. 원래 Janoschek leading 연산자는 수정하지 않는다.

고정 배치의 contact-limit map을 T, anchor를 h_a라고 할 때,

\[
t=\log(h_a/h),\quad d_n=k_n(h_a)(e^t-1),\quad d=b\,t,
\]
\[
\Delta K_+=tD+
\left(\sqrt{d_n}e_n+\frac{d}{\sqrt{d_n}}\right)
\left(\sqrt{d_n}e_n+\frac{d}{\sqrt{d_n}}\right)^T,
\qquad G(h)=G_a+T^T\Delta K_+T.
\]

D는 원래 접선 저항의 양의 준정부호 Gram 계수이며 b는 normal-cross
로그 계수다. anchor에서는 G_a를 그대로 반환한다. 원래 증분에 더해지는
항은 `d d^T/d_n`이다. 이 항은 anchor와 h→0에서 사라지며, 작은 h에서는
`O(h log²(h_a/h))`이므로 현재 구현한 특이항의 계수를 바꾸지 않는다.
소산의 양수성을 얻기 위한 고유값 clipping은 하지 않는다.

이 구성은 정확한 유한 간격 Stokes 해를 유도한 결과가 아니다. 모델이
양의 소산을 갖는다는 것과 실제 저항을 정확하게 예측한다는 것은 다른
판정이다. 생성 보고서에 기존 상수식과의 차이를 별도로 기록한다. 그
차이는 물리 오차 추정치가 아니다. K33의 subleading log 누락도 복구하지
않는다. 새 모델 이름은 `JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1`이다.

## 완료된 실제 배치 결과

[actual_candidate_003/pipeline.json](results/provisional_pipeline/actual_candidate_003/pipeline.json)은
원본 JSON·밀도 checkpoint·고립 검사 출처를 유지한 실제 실행이다. 새 Stokes
풀이는 수행하지 않았다. 17개 병진·회전·strain 모드를 모두 검사했다.

| 검사 | 결과 |
| --- | --- |
| 실제 표 범위 | 기록된 AR20 상대 방향 하나,1–150 nm |
| 간격점 수 | 로그 균일129개에서 적응 세분222개로 증가 |
| 153성분 복구 최대 상대 에너지 오차 | 2.06×10⁻¹⁰ |
| 간격 보간 최대 상대 에너지 오차 | 7개 내부 검사점/구간에서8.52×10⁻⁷ |
| runtime 힘 연산자 최대 상대 오차 | 1.74×10⁻⁸ |
| runtime 토크 연산자 최대 상대 오차 | 4.96×10⁻⁸ |
| runtime intrinsic stresslet 연산자 최대 상대 오차 | 1.74×10⁻⁸ |
| runtime 검증 조건 | 7개 간격×원래 방향·전체 회전·director 부호 변환 |
| 실제3.15 nm / 실제0.5 nm의 유효 간격 | 3.15 nm /1 nm |

위 수치는 선택한 후보식과의 전달 오차다. 알려진 참 Stokes 응답과의 오차가
아니다. 균일129개와256개의 최초 표 시도는 보간 오차가 커서 실패했고,
그 로그를 보존한 뒤 같은 배치의 간격점을 적응적으로 배치해 통과했다.

기존 상수식과 새 후보의 차이는 후보 에너지 metric에서 최대1.883이다.
작은 수정으로 간주할 수 없으며 물리 오차 상한으로 해석해서도 안 된다.
후속 물리 검사는 anchor 격자 개선과 함께 별도 간격에서 모델 비교를 해야
한다. 1 nm에서 실행됐다는 사실만으로 이 간격의 물리 정확도를 승인하지 않는다.

별도의 **합성 전체 격자 구현 검사**는64방향과243간격점에서 생성·runtime
호출을 통과했다. 이 fixture는 Stokes 해가 아니며 실제 전체 방향 표가 아니다.
[합성 검사 기록](results/provisional_pipeline/fullgrid_contract/README.md)을 참고한다.

## 같은 AR20 입력으로 실행

압축을 푼 `Test_BS` 디렉터리에서 실행한다. C++17 컴파일러, CMake,
Python3.10 이상의 NumPy·SciPy가 필요하다. 이 단계에는 LAMMPS, MPI 또는 외부
Stokes 라이브러리 빌드가 필요하지 않다.

```bash
cmake -S lammps_slurry -B build/reference-pipeline \
  -DSLURRY_BUILD_LAMMPS_PLUGIN=OFF -DSLURRY_ENABLE_LU=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/reference-pipeline \
  --target slurry_precompute_reference17 slurry_reference_pipeline_probe -j 1

python3 validation_basis/independent_stokes/run_reference_table_pipeline.py \
  validation_basis/independent_stokes/results/pair_h150_p8_anisotropic_chunked.json \
  --output-dir reference_trial_001 \
  --generator build/reference-pipeline/slurry_precompute_reference17 \
  --checker build/reference-pipeline/slurry_reference_pipeline_probe \
  --matched-model dissipative-increment --layout sparse
```

출력은 `reference_trial_001/pipeline.json`, `reference.sft`, `generation.json`,
`runtime.json`과 원본 출처를 담은 `reference.source17.manifest.json`이다.
기존 디렉터리를 덮어쓰지 않는다. 실행 성공은 이 모델의 전달·저장·호출
성공이며, 1 nm 직접 Stokes 검증의 성공이 아니다.

간격점은 먼저 로그 간격으로 배치한 뒤 보간 오차가 큰 구간을 세분한다.
생성 보고서는 각 시도의 최대 오차·위치·최종 간격점 수를 보존한다.
보간 오차는 선택한 모델과 저장 표 사이의 차이다.

## 고해상도 결과로 교체

같은 AR·방향·anchor에서 새 보고서를 만든 뒤 위 명령의 첫 입력 경로를
새 보고서로 바꾸고 `--output-dir reference_trial_002`처럼 새 출력 위치를
지정한다. 자동으로 가장 최근 파일을 추측하지 않는다. 원본 보고서와
checkpoint의 해시가 입력·표·실행 기록에 연결되므로 이전 표와 새 표를
혼동하지 않는다. 생성기와 runtime 보고서에는 컴파일된 수치 소스의 fingerprint도
들어가며, case 준비는 현재 코드와 일치하는지 검사한다. 원래 경로가 사라졌을 때만 인접 checkpoint/evidence를
찾으며, 그 경우에도 원래 SHA256과 일치해야 한다.

## 다른 AR·방향과 전체 방향 표

독립 풀이의 `run_reference.py`와 연산자 helper는 이제
`--aspect-ratio --c-nm --theta-i --theta-j --phi`를 받는다. 각도는 rad이며
표를 위한 canonical theta 범위는 [0,pi/2]다. 생략하면 기존 기록된 AR20
배치와 같은 기하를 만든다. 새 AR을 계산했다는 뜻은 아니며, 각 AR의
고립 모멘트와 표면 해상도는 사용자가 실행한 결과로 확인한다.

한 source bundle은 동일 AR·c·floor·anchor를 갖는 여러 방향의 보고서를
받는다. 현재 제공된 실제 자료는 한 방향뿐이므로 `sparse` 표는 그 방향과
동등한 전체 회전·director 부호 변환만 받는다. 다른 방향에는 명시적으로
오류를 내며, 이 표로 무작위 배향 bulk suspension을 시작하지 않는다.

모든 theta×theta×phi 방향에 실제 입력을 갖추면 `--layout full-grid`로
동일 경로를 실행한다. 공통 theta 노드는 최소 4개이며 0과 pi/2를 포함하고,
phi는 [-pi,pi)의 짝수 개 균일 주기 격자여야 한다. Pole의 각 frame도
입력해야 한다. 누락 방향을 현재 한 행렬로 복제하지 않는다.

전체 격자의 `pipeline.json`은 `prepare_case.py`의 명시적인
`--finite-gap-fast-reference-pipeline` 입력으로 사용할 수 있다. 이 경로는
기존 Basis를 다시 계산하지 않고 산출물 해시·AR·크기·floor·범위를 확인한다.
이 옵션은 `--lubrication-model finite_gap_fast`와 함께 명시해야 하며,
실제 표의 phi 노드 수에 맞는 `--finite-gap-fast-nphi`와 표 범위 이내의
`--finite-gap-range`도 지정한다. 기본 auto 모델이 새 후보를 자동 선택하지
않는다. `sparse` 결과는 bulk case 준비 단계에서 거부한다. 전체 격자의 구조가
완성됐더라도 angular 보간 정확도와 suspension 점도 검증은 별도로 남는다.

사용자 HPC의 고해상도 계산은 이 경로의 선행조건이 아니라, anchor와
모델의 오차를 평가하고 교체하는 후속 단계다. 150 nm는 이번 잠정 입력의
간격이며 보편적인 물리 전환거리로 채택한 값은 아니다.

독립 풀이의 geometry source가 일반화되었으므로 기존 풀이의 실행 cache를
새 driver의 것으로 표시하지 않는다. 새 직접 풀이에서는 현재 helper와
고립 입자 검사를 다시 준비한다. 기존 완료 보고서의 잠정 import는 원래
출처를 유지하므로 계속 가능하다.
