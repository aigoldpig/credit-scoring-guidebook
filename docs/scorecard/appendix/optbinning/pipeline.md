# 다변량 처리와 점수 변환

## 4.1 여러 변수의 일괄 구간화

실무에서는 수십~수백 개 변수를 같은 규칙으로 구간화해야 한다. optbinning은 이를 위한 일괄 처리 기능(BinningProcess)을 제공하며, 개념적으로는 다음 세 층으로 이뤄진다.

| 층 | 내용 | 설계 포인트 |
|---|---|---|
| 전역 기본 규칙 | 모든 변수에 공통 적용할 최대 Bin 수, 최소 Bin 비중, 최대 p-value 등 | 포트폴리오 불량률·표본 크기에 맞춰 한 번 정한다 |
| 변수별 예외 | 특정 변수만 단조성 방향·Bin 수를 따로 지정 | **업무 논리상 방향이 확실한 변수는 방향을 명시**한다 |
| 자동 선별 | IV 등 기준으로 변수를 걸러냄(예: IV < 0.02 제외), 기준과 무관하게 반드시 남길 변수 지정 가능 | 선별 결과는 후보 목록일 뿐 최종 결정이 아니다 |

변수별 방향을 지정할 때는 **Event rate 기준**이라는 점을 다시 확인한다([핵심 옵션 2.1절](params.md)).

| 변수 | 업무 논리 | 지정할 방향 |
|---|---|---|
| 카드 사용률 | 높을수록 위험 | Event rate 증가(ascending) |
| 업력 | 길수록 안전 | Event rate 감소(descending) |
| 부채비율 | 높을수록 위험 | Event rate 증가(ascending) |
| 영업이익률 | 방향은 분명하나 극단값 구간 확인 필요 | Event rate 감소(descending) + Bin 수 상한 축소 |

!!! warning "자동 선별의 한계"
    IV 기준 자동 선별은 **변수 하나씩** 본다. 서로 상관이 높은 변수 두 개가 모두 IV가 높으면 둘 다 통과하고, 다변량 단계에서 부호 역전이나 VIF 문제를 일으킨다. 또 범주형 변수는 일괄 처리 대상 목록에 포함된 변수 중에서 지정해야 한다. 목록에 없는 변수를 범주형으로 지정하면 처리되지 않거나 오류가 난다. 자동 선별 결과는 본문 [정보영역별 변수 선정](../../part3_variable_selection/domain_selection/index.md)과 [다변량 회귀](../../part4_modeling/multivariate-regression.md)의 검토를 거쳐 확정한다.

---

## 4.2 점수 변환 (Scorecard 기능)

optbinning은 "일괄 구간화 → 로지스틱 회귀 → 점수 변환"을 하나로 묶는 기능도 제공한다. 편리하지만 다음 세 가지를 이해하고 써야 한다.

### (1) 부호 — 라이브러리 내부 계산은 가이드북 규약과 같다

라이브러리 소스 코드를 보면 점수 변환은 다음과 같이 계산된다(오름차순 점수 기본 설정 기준).

$$
\text{Factor} = \frac{\text{PDO}}{\ln 2}, \qquad \text{Offset} = S_0 - \text{Factor}\cdot\ln\theta_0 \tag{1}
$$

$$
\text{배점}_{jb} = -\text{Factor}\cdot\left(\beta_j\,\text{WoE}_{jb} + \frac{\beta_0}{k}\right) + \frac{\text{Offset}}{k} \tag{2}
$$

여기서 \(\beta\)는 Bad logit 회귀계수(WoE 변수의 정상 부호는 음수), \(k\)는 모형 변수 수, \(S_0\)는 기준 Odds \(\theta_0\)에서의 점수다. 모든 배점을 더하면 \(\text{Score} = \text{Offset} + \text{Factor}\cdot\ln(\text{Good Odds})\)가 되어 본문 Part 5의 규약(배점 \(= -\text{Factor}\cdot\beta_j\cdot\text{WoE}_{ij}\) + 상수 배분)과 일치한다. 식 (2)에서 상수항은 변수 수 \(k\)로 균등 배분된다.

!!! danger "기준 Odds 인자는 Good:Bad로 넣는다"
    식 (1)의 Offset 정의에 따라 총점이 \(S_0\)가 되는 지점은 \(\text{Good Odds} = \theta_0\)일 때다. 따라서 기준 Odds 인자에는 **Good:Bad** 값(예: 정상 19 : 불량 1이면 19)을 넣어야 한다. 공식 문서는 이 인자의 방향을 명시하지 않으므로, "Bad:Good = 1:19"라는 생각으로 1/19를 넣으면 기준점이 정반대 위험 수준에 놓인다. 적용 전 소수의 테스트 차주로 "불량률이 낮은 차주가 높은 점수를 받는지"를 반드시 확인한다.

    <div class="source-ref">출처: <a href="https://github.com/guillermo-navas-palencia/optbinning" target="_blank">optbinning 소스 코드(scorecard 모듈)</a>, <a href="https://gnpalencia.org/optbinning/scorecard.html" target="_blank">공식 문서 — Scorecard</a></div>

### (2) 회귀 추정기 — 기본값이 '규제 회귀'인 경우가 많다

점수 변환 기능은 외부 회귀 추정기를 받아 쓴다. 널리 쓰이는 범용 머신러닝 라이브러리의 로지스틱 회귀는 **기본값이 L2 규제(정규화)** 다. 규제가 걸리면 계수가 0 쪽으로 줄어들어, 부록 B에서 다루는 MLE·Wald 검정·p-value 해석과 맞지 않고 배점도 작아진다. 스코어카드 목적이라면 **규제를 끄거나(무규제 MLE)** 규제 강도를 사실상 0으로 두고, 계수·표준오차를 통계 패키지 결과와 대조한다.

### (3) 스케일링 방식

| 방식 | 설명 | 비고 |
|-----------------|------|------|
| PDO-Odds | 기준 Odds \(\theta_0\)에서 \(S_0\)점, Odds가 2배가 될 때마다 PDO점 증가 | 업계 표준. 본문 Part 5와 같은 체계 |
| Min-Max | 점수 범위를 [min, max]로 선형 고정 | 점수의 Odds 해석(PDO)이 사라진다. 대외 표시용 |

이 밖에 각 변수의 최소 배점을 0으로 맞추고 차이를 상수에 몰아주는 옵션도 있다. 이는 [부록 B 기준점 설정](../logistic-fitting/woe-entry.md)의 평행이동과 같은 조작이며 예측값을 바꾸지 않는다.

---

## 4.3 실무 옵션 체크리스트

| 상황 | 권장 설정 | 이유 |
|------|----------|------|
| 첫 탐색, 방향을 모름 | 단조성 auto, Solver CP, 최대 Bin 7 | 데이터가 말하는 방향을 먼저 확인 |
| 업무 논리상 방향 확실 | ascending/descending **명시** (Event rate 기준) | auto가 우연한 패턴을 고르는 것 방지 |
| 소표본(수천 건) | 최소 Bin 비중 상향(예: 10%), 최소 Bad 건수 지정 | Bin별 WoE 안정성 확보 |
| 대표본(수십만 건 이상) | Fine Bin 수 확대(예: 50), 최소 비중 하향 | 경계 해상도 향상. p-value 제약은 사실상 비활성 |
| 0값 등 Mass Point 존재 | 해당 값을 Special로 분리할지 검토 | 0의 의미(무실적 vs 정상값)에 따라 결정. Special은 단조 제약 밖 |
| 범주형, 희귀 범주 다수 | 희귀 범주 기준(예: 5%)으로 others 그룹화 | 희귀 범주의 불안정한 정렬 방지 |
| 한 Bin에 표본이 과도하게 몰림 | 최대 Bin 비중 지정 또는 정규화(gamma) | IV 희생폭과 함께 판단 |
| Special/Missing 위험이 높음 | 변환값을 empirical 또는 보수적 값으로 | 기본값 0은 평균 위험 처리 |
| 빠른 처리 필요 | Quantile Pre-binning, Fine Bin 수 축소, 휴리스틱 단조 옵션 | 정밀도와 속도의 교환 |
