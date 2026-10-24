# Flake 원인 진단 overlay — 2026-10-24

이 패치는 20261023-spheroid-flake-common-run 위에 적용하는 진단 추가입니다.
기존 힘, 저항행렬, 접촉·마찰, gap floor, solver와 입자 운동을 변경하지 않습니다.
`run_slurry_cpu.sbatch`와 `build_slurry_cpu.sbatch`도 포함하지 않으므로,
직접 입력해 둔 AR·PHI·COUNT·코어·메모리 설정이 유지됩니다.

## 적용과 실행

slurry 디렉터리에서 `unzip -o slurry_flake_diagnostics_overlay.zip`을 실행합니다.

**이미 저장된 궤적부터 확인:** `python3 diagnose_flake.py`

`runs/` 안에서 완전한 frame이 있는 최신 flake 궤적을 골라 실제 경로를 표시합니다.
특정 계산을 지정하려면 `python3 diagnose_flake.py runs/계산폴더`를 사용합니다.
Python 3와 NumPy가 필요합니다. 이 명령은 시뮬레이션을 시작하거나 입력을 바꾸지 않습니다.
끝부분에서 작성 중인 불완전한 dump frame은 제외합니다.

**면·rim·중거리 성분 기록:** C++ plugin을 한 번 다시 빌드해야 합니다.
`sbatch --cpus-per-task=4 --mem=32G build_slurry_cpu.sbatch`

빌드가 성공한 뒤 기존 설정 그대로 `sbatch run_slurry_cpu.sbatch`로 실행합니다.
`PARTICLE_SHAPE=flake`인 계산에서 진단이 자동으로 켜지며,
기존 OUTPUT_INTERVAL과 같은 간격 및 마지막 step에서 기록됩니다.
현재 실행 중인 private runtime에는 새 plugin이 적용되지 않습니다.
동일 run을 추가로 제출하지 말고 새 진단 run을 시작할 때 사용하십시오.
설치·실행 과정에 별도의 test 명령이나 새 batch를 추가할 필요가 없습니다.

진단만 끄려면 실행 환경에 `SLURRY_FLAKE_DIAGNOSTICS=0`을 설정할 수 있습니다.
이는 물리 모델을 바꾸는 선택이 아닙니다. Spheroid에서는 진단 CSV를 생성하지 않습니다.

## 확인 1 — 실제 층 형성과 층 사이 미끄럼

`diagnose_flake.py`가 `flake_layering.csv`와 `flake_diagnostics_summary.json`을 만듭니다.

- `nematic_S`: 입자 축의 정렬도입니다. 이것만으로 층 형성을 판정하지 않습니다.
- `gradient_peak_order`, `director_peak_order`: 전단 gradient 및 평균 입자축 근방에서
  위치의 주기적 밀도 변조를 측정합니다. 상자에 허용되는 reciprocal lattice mode를
  쓰므로 triclinic/Lees–Edwards 이미지 선택에 불변입니다.
- `*_uniform_max_order_p95`: 같은 N과 같은 mode 탐색을 수행한 무작위 위치 기준값입니다.
  정렬된 축을 유지하더라도 위치가 무작위이면 층으로 오인하지 않도록 비교합니다.
  제외체적 입자계의 평형 통계나 물리적 p-value가 아니며, 임계값을 자동 판정하지 않습니다.
- `*_peak_wavelength_m`: 가장 큰 밀도 mode의 파장입니다. 실제 층 간격의 고조파일 수
  있으므로 단독으로 층 간격이라고 부르지 않습니다.
- 비친화 속도에서 affine shear와 평균 병진을 빼고 법선·접선 RMS를 계산합니다.
  법선 속도의 두께/strain 정규화도 기록합니다. 속도가 작다는 사실은 장기간에 걸친
  층 간 교환이 없다는 증명은 아닙니다.
- `parallel_neighbor_*`: 축 각도 ≤10°, bisector 방향 중심 간격 ≤4c,
  투영 중심 간격 ≤2a인 이웃들의 중심 상대속도입니다. 전체 필요한 주기 이미지를
  열거하고 이미지 속도 이동도 적용합니다. 정확한 표면 gap·접촉 수·막 내부 slip의
  대용으로 해석하지 않습니다.

**층간 미끄럼의 근거:** 축 정렬 + 지속적인 위치 밀도 변조 + 작고 지속적인 법선 운동 +
유한한 접선 상대운동이 함께 나타나는지 봅니다. S만 크거나 frame 두 개만 있는 상태로
정상상태 구조가 확정되었다고 판단하지 않습니다. director 고유값 간격이 작으면 그 방향은
잘 정의되지 않으므로 gradient 지표를 함께 봅니다.

기존 xz 전단 설정과 현재 dump 형식을 대상으로 합니다. mode index 상한128,
director 방향15° cone 및 최대512개의 mode를 사용하며 범위 제한 여부를 CSV에 남깁니다.
`--stride`, `--max-frames`, `--max-mode-index`는 긴 궤적의 선택적 분석에 사용할 수 있습니다.

## 확인 2 — 실제 저항 기여와 각도 감쇠

새 plugin은 `flake_channels.csv`, `flake_channel_bins.csv`를 run 폴더에 기록합니다.
추가 Reynolds solve 없이 원래 평가한 두 행렬을 출력 step에만 보관하여 계산합니다.
주기 self-image도 기존 조립 규칙대로 한 번씩 더합니다.

### 실제 계산한 기여

- `face_power_W`, `rim_power_W`, `middle_power_W`: 현재 풀린 일반화 속도에서 각 PSD
  유체 저항 채널의 소산입니다. Middle에는 사용 중인 matched-spheroid self 응답도 포함됩니다.
- `face_sigma_xz_Pa`, `rim_sigma_xz_Pa`, `middle_sigma_xz_Pa`: 해당 모멘트에서 얻은
  전단응력 성분입니다. 힘/토크의 일을 포함한 소산을 채널별 전단응력으로 바꾸지 않습니다.
- `eta_relative_face/rim/middle/contact`: 위 응력을 η_medium·γdot로 나눈 **부호 있는**
  상대점도 기여입니다. `eta_relative = 1 + face + rim + middle + contact`입니다.
  각 성분의 소산이 양수여도 전단응력 성분 자체는 음수일 수 있습니다.
- `near_moment_closure_relative`, `power_closure_relative`, `hydro_stress_closure_Pa`:
  원래 조립된 저항·소산·응력과 진단 성분 합계의 차이입니다.
- `active_contacts`, `contacted_particles`, `normal_load_sum_N`, `normal_load_max_N`:
  접촉 수뿐 아니라 실제 압착력과 접촉에 참여한 입자 수를 확인합니다.
- `sliding_dissipation_W`: 다음 암시적 증분에서 발생하는 접촉의 비가역 미끄럼 일을
  timestep으로 나눈 값입니다. `contact_power_W`는 탄성 에너지 저장·방출을 포함하며,
  순수 마찰 소산과 다릅니다. 탄성 에너지는 기존 history.csv에도 기록됩니다.

### 각도 감쇠 민감도 — 재계산한 점도가 아님

현재 모형은 `R = w_cut [ w_angle w_area F + (1-w_angle w_area) R_rim ]`입니다.
막을 실제 계산한 범위에서만 다음을 기록합니다.

`extra_face_frozen_power_W = w_cut w_area (1-w_angle) qᵀFq`

이 값은 **현재 속도와 기존 rim을 유지**하며 제거된 face 항만 다시 더한 소산입니다.
실제로 각도 감쇠를 없애려면 rim 가중치도 줄어들므로 다음 항도 별도로 기록합니다.

`replacement_frozen_power_delta_W = w_cut w_area (1-w_angle) qᵀ(F-R_rim)q`

두 번째 값은 음수일 수 있습니다. 어느 것도 새로운 입자 속도나 구조로 다시 푼
suspension 점도는 아닙니다. 이 비율만큼 최종 점도가 증가한다고 해석하면 안 됩니다.

각도 ≥0.35 rad(약20.05°)에서는 원래 막 모델을 평가하지 않습니다.
`angle_excluded_pairs`는 이 범위의 cutoff-active 쌍 수이고, 이들의 잃은 막 저항은
**미평가/알 수 없음**입니다. 이 쌍에 실제 face overlap이 있다는 뜻도 아닙니다.
낮은 감쇠 손실이 보고되어도 미평가 쌍이 많으면 전체 cutoff의 무해성을 결론 내릴 수 없습니다.

### 구간별 출력

각도: 0–5.73°, 5.73–10°, 10–15°, 15–20.05°, 20.05–90°.
Gap/floor: <1, 1–3, 3–10, 10–30, 30–100, ≥100.
Floor가1 nm인 현재 설정에서는 gap 구간 숫자가 nm와 같습니다.
각 구간의 개수·소산·전단응력·감쇠량을 합산하므로, 먼 후보 쌍 수가 많아도
실제로 중요한 작은 gap에서 어떤 채널이 작동하는지 따로 확인할 수 있습니다.

CSV의 `candidate_pairs`는 cutoff-active 진단 쌍입니다. 전체 접촉 후보 수는
기존 history.csv의 `contact_candidates`입니다.
평가된 막의 평균 겹침 비율은 `overlap_area_fraction_sum / film_overlap_pairs`입니다.
공통 작용점 speed² 합계를 cutoff-active 쌍 수로 나누고 제곱근을 취하면 RMS입니다.
이는 막 면적에 걸쳐 적분한 shear 속도가 아닙니다.

## 계산 비용과 판정 범위

추가 행렬 보관·집계는 출력 step에만 시행합니다. N500 회귀 계산에서 출력 step의
저장량은 약130 MB였습니다. `diagnostic_capture_s`와 `diagnostic_reduce_s`는 부분 계측이며
일부 복사·I/O가 빠지므로 그 합계를 전체 추가 비용이라고 부르지 않습니다.

이 패치는 Stokes 기준해와 비교해 물리 정확도를 인증하는 것이 아닙니다.
먼저 '실제 층간 미끄럼인가', '평가 범위 안의 각도 감쇠가 큰가',
'작은 gap에서 면 저항이나 접촉 하중이 거의 없는가'를 구별합니다.
근거리 행렬 자체의 공간해상도 정확도와 20° 밖의 미평가 막은 별도 문제입니다.
기존 GJK gap 수렴 중단 조건도 이번 진단판에서는 변경하지 않습니다.
중단된 run이라도 저장된 완전한 frame과 진단 행은 분석할 수 있습니다.

검증 범위와 수치는 `validation_flake_diagnostics/RESULT.json`에 기록합니다.
