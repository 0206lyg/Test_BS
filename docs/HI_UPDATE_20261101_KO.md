# HI 유한 크기 보정 및 공통 taper — Git 반영 요약

기준: Test_BS `e1da67e4ded7863654af736f1162ac0a2cc702da`. 2026-11-01 검증 후보의 수식을 유지한 소형 코드 overlay다. 이번 포장 작업에서 새 동적 계산은 수행하지 않았다.

## 변경

- Face의 유한 film 크기를 반영해 곡률 기반 normal·log 저항의 과대평가를 제한한다. 기존 계수와 `min`으로 결합한다.
- 서로 다른 표면 곡률 때문에 보정이 빠지던 EF에 면별 cap을 추가한다. 각 면의 log 기준길이는 해당 입자의 최소 반축 이하로 제한한다. Normal의 대응 길이 `Lc/2`는 유한 film 근사다.
- 구와 비구형 전체 pair 저항에 동일한 비음수 quintic taper를 한 번 곱한다. 기본 시작점은 원래 cutoff의 0.75이며 마지막 25% 구간에서 0으로 매끄럽게 감소한다. 힘·토크·응력·소산에 같은 배율이 적용된다.
- Sliding·pump의 공통 log, roughness `epsilon=0.001*rbar`, 접촉 시작 위치, 원래 cutoff, 단일 입자 drag, 접촉 solver와 시간 적분을 유지한다.

일반 빌드에서 검증 모델이 선택되도록 `slurry_nearfield_hydro.h`의 `SLURRY_VALIDATION_TAPER_POLICY` 기본값을 **2**로 설정했다. 이전 검증 묶음은 기본값 0을 두고 컴파일 옵션으로 2를 지정했으므로, 그 파일을 그대로 복사하면 taper가 꺼질 수 있었다. `SLURRY_VALIDATION_TAPER_START` 기본값은 **0.75**다. Candidate 헤더의 수식은 검증본과 같으며 첫 설명 주석만 갱신했다.

기존 `test_nearfield_hydro.cpp`는 이전 미보정 pair를 기대값으로 사용했다. 기대 pair 생성만 같은 수정 kernel로 바꾸고 힘·토크·응력·소산 등의 기존 검사는 유지했다. 이 테스트는 조립 일관성을 확인하며 독립 물리 기준해 역할을 하지는 않는다.

소형 배포본의 기본 설정과 기존 검증 당시의 명시적 `policy=2, start=0.75` 설정을 각각 컴파일해 대조했다. **1,000개 배치의22×22 행렬, 총484,000개 double 값이 바이트 단위로 동일**했다. 배포본의 기존36개 조립 검사와 바깥 간극을 포함한 확장49개 검사도 통과했다. 현재 Git의 기반 spheroid/grand-pair 헤더가 검증 기준 헤더와 같은 것도 확인했다.

## 완료된 검증과 결론

**국소 HI 수정과 구현 검증 통과.** 기존 Reynolds 기준해 검증을 통과했던 FF·tilt·offset 123개 배치의 행렬을 보존했다. 두 taper 시작점 0.75와 0.5 각각 구 100쌍·비구형 500쌍에서 PSD, 입자 교환·좌표 회전, 구형 극한, 힘·토크·응력·소산 관계를 검사했다. 각 시작점의 1,000개 grand-matrix 검사에서 전체 저항이 정확히 같은 taper 배율을 받는 것을 확인했다. EF의 추가 cap도 작동한다. AR5 동일 크기 EF의 sliding 계수는 cap 이전 대비 접촉에서 0.717626배, 100 nm에서 0.537857배다.

**동적 물성 검증은 미완료.** N128, 크기비1.4, `kn=kt=kr=300 N/m`, `mu_s=mu_r=10`, 전단율100/s, 용매점도0.01 Pa·s, `dt=2.5e-6 s`, seed1729/1730으로 22개 조건을 최대 strain6까지 실행했다. AR5는 phi0.10/0.14/0.18, AR1은 phi0.24/0.28이며 cap 및 taper폭 대조군을 포함한다.

- 16개: 공통 관측 구간3.99–5.99에서 준정상 선별 기준 미통과.
- 5개: 관측 overlap/최소 반축이5%를 넘어 검증용 guard로 중단. 이 guard는 본 코드 overlay에 포함하지 않는다.
- 1개: 실행은 strain6까지 완료했으나 후반 CSV 기록 부족.
- 이번22개에서 solver 오류 종료0개. 정상 평균으로 인정한 모델 비교0개.

AR5 phi0.14의 같은 과도 구간 평균에서 EF cap 추가 효과는 두 seed에서 −40.83%, −58.30%, taper 시작점0.75→0.5 효과는 +38.47%, +69.59%였다. 정상 점도 변화율은 아니며 taper의 동적 영향이 작다는 주장은 유지하지 않는다. 별도 mu_r=0 두 seed는 strain6까지 관측 overlap 기준을 지켰지만 준정상 기준은 미통과했다. N1000의 정상 점도·phi_J·속도 향상이나 접촉 시간 적분 문제 해결을 주장하지 않는다.

## Git 반영 및 빌드

압축을 저장소 root(`lammps_slurry`와 `docs`의 상위 폴더)에 풀고 다음4개 파일만 반영한다.

1. `lammps_slurry/src/slurry_nearfield_hydro.h`
2. `lammps_slurry/src/slurry_hi_validation_candidate.h`
3. `lammps_slurry/tests/test_nearfield_hydro.cpp`
4. `docs/HI_UPDATE_20261101_KO.md`

**적용 후 기존 `build_slurry_cpu.sbatch`로 반드시 재빌드한다.** 새 헤더는 CMake와 source/runtime snapshot에 자동 포함된다. CMake, release ID, input generator, batch script를 별도로 수정할 필요는 없다. 기존 binary는 소스 변경을 반영하지 않으므로 재사용하지 않는다. 커밋·push·HPC 제출은 사용자가 수행한다.

Git에는 소스와 이 결론 문서만 있으면 된다. 원시 history, traj, 초기 배치, 실행 바이너리, 빌드 디렉터리, 중간 결과는 실행이나 새 빌드의 입력이 아니다. 독립 재검토용 원자료는 기존 전체 검증 묶음에 별도로 보존되어 있다.

- [전체 검증 보고서](https://chatgpt.com/library?file_id=libfile_cb58aa77324c8191a695b4c005ab892a&account_id=personal)
- [원자료·검증 코드 전체 묶음](https://chatgpt.com/library?file_id=libfile_3a9c18a6ba588191a3816d997cef60cb&account_id=personal)

위 링크는 원자료 소유 계정의 보관 위치이며 공개 접근을 보장하지 않는다. 외부 검토자에게 원자료가 필요하면 해당 전체 묶음을 별도로 전달하면 된다.
