# 합성 전체 격자 구현 검사

이 자료는 물리 Stokes 기준해가 아니다. 각 anchor에 임의의 `10^6 I17`을
넣어 저장·좌표 변환·runtime 연결을 검사했다. AR20의4×4×4=64개 방향과
1–150 nm 범위에서243개 간격점으로 생성했고,1344개 runtime 평가가 통과했다.
모든 물리 정확도 주장은 false이며, 실제 suspension 입력으로 사용하지 않는다.

재현용 source 생성기는 `make_synthetic_fullgrid_fixture.py`다.
`synthetic_contract_verification.json`의 commands 배열에 실제 재현 명령과
실행 파일 해시가 있다. 큰 합성 표를 배포하지 않고 생성 스크립트와 상세
요약을 보존했다. 같은 코드의 fingerprint 보고 필드 추가 전 실행이며,
새 source identity 검증 통과를 주장하는 자료가 아니다. 새 source identity를
포함한 실제 배치 검사는 ../actual_candidate_003/pipeline.json이다.

C++ Source 수준의 수학·format fixture이므로 독립 해 importer를 통과시키지
않았다. importer는 `synthetic=true` 보고서를 실제 기준해로 받아들이지 않는다.
