# 근접 유체력의 에너지 완성항

`general_near_completion energy`는 보존한 lubrication 특이계수는 그대로 두고,
유한한 간격에서 근접 연산자가 음의 소산을 만들지 않도록 **명시적인 양의 제곱항**을
완성한다. 런타임의 기본 선택이다. `raw`는 이전 평가식 재현용 진단 선택이다.

이는 논문의 새로운 특이항이나 기준해 보간이 아니다. 기존 연결식에 추가한 유한항
모델이며, 새 배향의 Stokes 계산·fitting·eigenvalue clipping을 사용하지 않는다.

## 발견한 문제

AR20 실제 64입자 실행의 첫 단계에서 입자 1–64, 표면 간격 약 1.17008 nm가 실패했다.
같은 기하의 C++와 기존 Python 근접 행렬은 상대 차이 `3.31e-14`로 일치했다.
수치적분의 outer 응답은 양의 정부호이고 충분히 수렴했으나, raw 근접식이 만든 음의
소산이 outer의 양의 소산보다 컸다. 따라서 적분 차수를 계속 늘려 해결할 문제가 아니었다.
experimental normal-log 항을 켜는 것만으로도 해결되지 않았다.

기존 식은 법선 squeeze 항, 접선·회전의 logarithm 항, 법선과 다른 운동 사이의
logarithm 결합항을 갖는다. 이를 유한한 간격에서 단순히 잘라 합치면, 점근적으로 작은
나머지 항을 생략한 결과로 에너지가 음수가 될 수 있다. 특히 한 축의 선도항이 0인
경우 결합항을 그대로 두고 유한한 대각항을 모두 버리는 것은 PSD를 보장하지 않는다.

## 모든 입력 기하에 적용되는 구성

접촉 좌표 6개를 법선 상대속도 \(u_n\)와 나머지 5개 운동 \(y\)로 나눈다.
코드의 기존 근접식은 다음 블록이다.

\[
K_{\rm raw}=\begin{pmatrix}k&c^T\\c&C\end{pmatrix},\qquad
k=f_A A_{nn}+f_B B_{nn},\quad c=f_B b,\quad C=f_B B_{yy}.
\]

기본 leading 모드에서는 \(B_{nn}=0\)이다. 볼록한 비퇴화 접촉의 \(A_{nn}>0\),
활성 구간의 \(f_A>0\)이므로 \(k>0\)이다.
\(B_{yy}\)는 코드에 이미 구현된 Couette 양의 대각항과 양의 계수를 가진 외적의 합이다.
따라서 \(C\succeq0\)이며, 이는 배향 목록을 검사해서 추정한 성질이 아니다.

다음 항을 **모든 활성 접촉에서 조건 없이** 더한다.

\[
K_{\rm add}=\begin{pmatrix}0&0\\0&cc^T/k\end{pmatrix}.
\]

그러면 임의의 \(u_n,y\)에 대해

\[
\begin{pmatrix}u_n\\y\end{pmatrix}^{T}
(K_{\rm raw}+K_{\rm add})
\begin{pmatrix}u_n\\y\end{pmatrix}
=k\left(u_n+\frac{c^Ty}{k}\right)^2+y^TCy\ge0.
\]

이 구조가 근접 연산자의 비음 소산을 보장한다. 마지막에 고유값을 잘라내는 방식이나
실패한 배향에서만 다른 모델을 켜는 방식이 아니다. 접촉 좌표에서 full17 좌표로의
기존 congruence \(T^TKT\)는 PSD, 상반성, 공통 병진 nullspace를 유지한다.
힘·토크와 합산 stresslet도 같은 완성된 행렬에서 계산하므로 에너지와 응력의 부호를
서로 다른 식으로 수정하지 않는다.

experimental `derived` normal-log 선택에서는 \(k>0\)인 경우 같은 증명이 성립한다.
그 후보 항으로 \(k\le0\)가 되면 명시적으로 중단하며, 다른 항으로 자동 대체하지 않는다.
이 선택에서 cutoff의 C1 성질까지 유지하려면 별도로
\(A_{nn}/h_c+B_{nn}/2>0\)이 필요하다. 아래의 무조건적인 cutoff 증명은
기본 leading 선택에 대한 것이다.

## 특이계수와 cutoff는 유지된다

기존 compact 함수는 \(x=h/h_c<1\)에서

\[
f_A=(1-x)^2/h,\qquad f_B=-\log x-1+x.
\]

leading 모드의 추가항 계수는 \(f_B^2/(f_AA_{nn})\)이다. 따라서

\[
h\to0:\quad K_{\rm add}=O(h\log^2h)\to0.
\]

보존한 \(A/h\)와 \(B\log h\) 계수를 하나도 바꾸지 않는다.
\(x\to1^-\)에서는 \(f_B\sim(1-x)^2/2\)이므로

\[
f_B^2/f_A\sim h(1-x)^2/4.
\]

추가항과 그 일차 미분은 cutoff에서 0이다. 기존 C1 연결도 유지된다.
누락된 법선 logarithm이나 전체 유한항을 정확히 복원했다고 주장하지는 않는다.

## 확인한 결과와 범위

실패한 기하에서 raw full17의 고립입자 저항으로 정규화한 최소 고유값은 약
`-1.27979`, experimental normal-log만 추가한 값도 약 `-0.68017`이었다.
에너지 완성 후에는 약 `+0.66646`이다. 추가행렬의 Frobenius norm은 raw full17의
약 `0.04872%`이지만, 원래 음의 에너지를 내던 운동 방향에서는 영향이 크다.
작은 전체 행렬 norm만으로 이 문제를 무시할 수 없는 이유다.

`test_general_near_energy.cpp`는 양의 제곱 항등식, 원래 특이계수 보존, 구의
추가항 0, 실제 실패 기하, 접촉 극한과 cutoff 극한을 검사한다. 이전 Python
행렬 fixture는 `energy_completion=false`의 raw 경로에서 계속 재현한다.

이 증명은 **근접 연산자의 소산 구조**에 대한 것이다. outer 모멘트 절단의 물리 정확도,
전체 다입자 closure의 정확도, 정상상태 점도 수렴까지 증명하지 않는다. outer 수치적분
검사와 다입자 조립 후의 기존 소산·잔차 검사는 계속 수행한다.
