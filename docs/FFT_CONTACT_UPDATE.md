# FFT·접촉·초기화 overlay (2026-10-22)

기준 저장소는 `0206lyg/Test_BS`, commit
`a764c1785d9f34d29dfecbfd635a859bb079ab9f`입니다.
overlay는 저장소 최상위에 덮어쓰는 변경 파일 묶음입니다.

## 적용 및 실행

```bash
# Test_BS 저장소 최상위에서 실행
unzip -o /path/to/Test_BS_fft_contact_overlay_20261022.zip
sbatch build_slurry_cpu.sbatch
```

빌드 로그의 `Build complete`를 확인한 뒤 실행합니다.

```bash
sbatch lammps_slurry/run_bridge_cpu.sbatch \
  --output runs/ar20_fft_kn100 --ar 20 --count 500 --phi 0.24 \
  --steps 100000 --output-interval 100 --average-start-strain 5
```

기본값은 `--mobility-backend mesh`, `--normal-stiffness 100`,
`--contact-gap-nm 3`입니다. 명시한 물성 override는 유지합니다.
이전 입력 파일에 적힌 `1 N/m`은 자동으로 바뀌지 않으므로 새 실행
폴더를 준비해야 합니다. 변경된 소스에 맞춰 반드시 다시 빌드합니다.
실행기는 소스·바이너리 fingerprint를 확인한 뒤 독립 runtime을 만듭니다.

## 변경 내용

1. **중거리:** ellipsoid의 힘·토크·stresslet 형상함수를 유지하는
   particle–mesh FFT. 표면 구적을 격자 가중치로 모아 저장하므로
   Krylov 반복마다 표면 구적이나 입자×모드 직접 합산을 반복하지 않습니다.
   소스 전달과 회수는 정확히 전치이며 Fourier 가중치는 비음수입니다.
   기존 단독입자 고파수 tail을 그대로 더합니다. 기준 계산용 직접 합산은
   `--mobility-backend direct`로 선택합니다. 벡터 세 성분을 두 복소 격자로
   묶어 FFT 횟수와 격자 메모리를 줄입니다.
2. **접촉:** 선형 법선 스프링의 기본값을 100 N/m로 변경합니다.
   접선 강성은 80 N/m, Coulomb 계수는 0.1입니다.
   접촉 활성화 하중 문턱이나 Hertz 접촉법칙은 추가하지 않습니다.
3. **초기화:** nominal overlap 제거 후 실제 gap을 이용해 간격을 정리합니다.
   상자와 입자 크기·목표 φ는 유지합니다. 정렬·무작위 초기화 모두
   주기 이미지와 자기 이미지를 포함한 실제 최소 표면 gap을 인증합니다.
   인증 실패 시 동역학 입력을 쓰지 않습니다. 결과는
   `packing_clearance.json`과 `parameters.json`에 기록합니다.
4. **접촉과 이동의 일관성:** Euler 위치 갱신, quaternion 회전 및 전단
   격자 갱신으로 얻은 endpoint gap을 이용해 법선력을 보정합니다.
   각 접촉의 힘 오차와 전체 힘·mobility 잔차가 모두 기준을 만족해야
   한 step을 수락합니다. 보정 과정은 동일한 초기 접선 이력을 사용하고,
   실제 이동에서만 그 이력을 한 번 갱신합니다.
5. **검증·출력:** 실제 초기화, FFT 기준 응답·대칭성·고유값, 회전으로
   생기는 접촉, 물리 단위의 100 N/m, 전체 LAMMPS 실행을 검사합니다.
   재현 스크립트는 `validation_fft_contact/run_validation.py`입니다.

외부 FFT 라이브러리 설치는 필요하지 않습니다. 내장 FFT는 각 방향에
`2^p` 또는 `3*2^p` 크기를 사용합니다. `spectral_tolerance=1e-8`은
기존 Gaussian Fourier 절단 기준입니다. FFT 격자 보간·표면 구적 오차와
생략된 고파수 입자 간 상호작용까지 포함한 전체 정확도를 뜻하지 않습니다.

## 추가 CSV 열

| 열 | 의미 |
|---|---|
| `geometry_iterations` | 한 물리 step의 endpoint 일관성 보정 횟수 |
| `contact_gap_consistency_m` | 활성 접촉에서 수락 직전 접촉식 gap과 실제 endpoint gap의 최대 차이 |
| `actual_endpoint_min_gap_m` | 시작 시점의 후보 쌍·이미지를 다음 위치·배향으로 이동시켜 계산한 최소 표면 gap |
| `actual_endpoint_overlap_m` | 그 endpoint의 최대 nominal overlap |
| `normal_elastic_energy_J` | 수락한 endpoint 법선 압축의 총 탄성에너지 |
| `tangential_elastic_energy_J` | 수락한 접선 스프링의 총 탄성에너지 |
| `linear_gap_defect_m` | 원래의 선형 gap 예측과 실제 endpoint gap 차이 |

`min_gap_m`은 현재 출력 시점의 실제 위치에서 계산합니다. endpoint 열은
그 위치에서 다음 이동을 위해 구한 상태이므로 두 열의 시점을 혼동하지
않아야 합니다. 마지막 행의 endpoint는 실제로 이동하지 않은 상태일 수
있습니다. `run 0`은 접선 이력을 진행하지 않습니다. 별도의 endpoint 탐색이
후보 밖에서 새로 생기는 접촉을 거부하므로 겹침이 검사 없이 통과하지
않지만, 멀리 떨어진 모든 입자쌍의 최소 gap까지 endpoint 열에 포함하지는
않습니다.

## 모델과 검증 범위

100 N/m는 유한 강성이며 모든 하중에서 nominal overlap을 강제로 막는
hard wall이 아닙니다. 위치 clipping은 없습니다. 기하 보정이 24회 안에
수렴하지 않거나 기존 후보 밖에서 새 접촉이 발견되면 해당 이동을
수락하지 않고 오류로 알립니다. 자동 timestep subdivision은 구현하지
않았습니다. 유체력, 접촉 법선과 접선 운동학은 step 시작 기하를 사용하는
1차 적분이며 법선 압축을 실제 endpoint와 일치시킨 것입니다.

무작위 packer의 성장 단계는 기존 자유 회전 FIRE 방식입니다. 초기
texture를 등방성으로 강제하지 않습니다. N=500, φ=.24에서 AR2/20의
무작위 성장은 통과했지만, AR30 무작위 성장은 새 gap 정리 이전의
기존 PW 성장 단계에서 φ≈.206 부근에 정체됐습니다. AR30 실행 검사는
명시적인 `--initialization aligned`를 사용합니다. 실행 중 숨겨진
정렬 초기화 전환은 없습니다.

짧은 실행 검사는 정상상태 점도나 목표 1 Pa·s 달성을 검증하지 않습니다.
`final.data`는 기존과 같이 기하 저장이며 접선 이력을 보존하는 restart가
아닙니다. 수치 결과와 비용은 `validation_fft_contact/RESULT.json`에 기록합니다.

## 로컬 재현

```bash
python3 validation_fft_contact/run_validation.py \
  --build-env build/lammps/env.sh --output validation_runs/fft_contact \
  --count 500 --steps 5 --ars 2 20 30 --threads 4
```

이 스크립트는 AR2/20 무작위 배치와 AR30 정렬 배치를 준비하고 실제
verified launcher를 실행합니다. 전체 출력은 지정한 새 폴더에 남습니다.
MPI 빌드에서는 기존 runner와 같이 한 MPI rank와 OpenMP를 사용합니다.

중거리 kernel 비용만 비교하려면 빌드한 test executable을 사용합니다.

```bash
build/lammps-slurry/slurry_periodic_mesh_test bench 500 2 4 mesh
build/lammps-slurry/slurry_periodic_mesh_test bench 500 2 4 direct
```

이 비용 검사는 합성 배치의 mobility 연산이며 전체 suspension step의
시간과 다릅니다. 정상상태 계산의 소요 시간을 이 값만으로 추정하지 않습니다.

## 최종 중거리 kernel 측정

N=500, OpenMP 4 threads, 동일 합성 배치에서 측정했습니다.
시간은 초, 메모리는 kernel이 보유한 배열의 decimal MB이며 프로세스 전체
메모리가 아닙니다. 직접 합산과 FFT 모두 최종 소스에서 측정했습니다.

| AR | 구성: direct / FFT | 1회 적용: direct / FFT | 배열 MB: direct / FFT |
|---|---:|---:|---:|
| 2 | 3.1365 / 0.2912 | 0.2896 / 0.2365 | 3031 / 299 |
| 20 | 0.3300 / 0.2339 | 0.0233 / 0.0409 | 304 / 213 |
| 30 | 0.2110 / 0.2475 | 0.0161 / 0.0439 | 203 / 251 |

AR2는 구성 약 10.8배, 1회 적용 약 1.22배 빨라졌고 배열 메모리는 약
10분의 1입니다. 이 N에서 AR20/30의 반복 적용은 직접 합산이 더 빠릅니다.
FFT의 격자·보간 비용 때문에 모든 AR에서 속도가 개선되지는 않았습니다.
기본 backend는 요청한 mesh를 유지하며 숨겨진 direct 전환은 없습니다.
FFT는 기존 입자×모드 배열을 없애고, 고정 형상·밀도에서 입자 수에 대해
선형인 stencil 작업과 격자 FFT 작업으로 계산 구조를 바꿉니다.

AR2/20/30의 작은 전체 행렬 검사에서 직접 합산 대비 scaled norm 상대
오차는 각각 7.34e-8 / 5.53e-8 / 7.77e-9였습니다. 대칭성 오차는
1.4e-15 이하였고 모든 고유값이 양수였습니다. 약한 개별 블록의 최대
상대오차는 6.4e-6이므로 전체 행렬 norm 오차와 구분해야 합니다.

## 최종 실제 실행 검사

최종 소스의 전체 빌드, CTest 34/34, LAMMPS plugin 로드를 통과했습니다.
로컬 CPU 1 process·OpenMP 4 threads에서 N=500, φ=.24, dt=1e-6 s로
각 배치를 5 step 진행했습니다.

| AR | 초기화 | 인증한 초기 최소 gap (nm) | 최소 endpoint gap (nm) | 최대 기하 보정 횟수 |
|---|---|---:|---:|---:|
| 2 | random | 3.00143161 | 2.99998028 | 2 |
| 20 | random | 3.00251837 | 2.99567531 | 3 |
| 30 | aligned | 54.95660899 | 54.95661035 | 1 |

모든 기록에서 nominal overlap이 없었고, 유체 소산은 비음수였습니다.
최대 상대 solver 잔차는 1.17e-8, 상대 일률 수지 오차는 2.92e-8
이하였습니다. 접촉이 활성화된 AR2/20에서 solved endpoint gap을 다음
step의 실제 gap과 비교한 최대 차이는 1.4e-21 m였습니다.

실행 CSV, 입력값, 초기 gap 인증, 빌드 로그 및 상세 결과는
`validation_fft_contact/`에 포함합니다. 이 검사는 변형률 .0005까지의
실행·접촉 일관성 검사이며 정상상태 또는 장시간 무겹침의 보장은 아닙니다.
