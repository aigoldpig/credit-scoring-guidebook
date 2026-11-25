---
title: "독립 가정의 한계"
---

# 4. 독립 가정의 한계 — 왜 외삽이 발생하는가

> Product measure(독립 가정)는 이론적으로 깔끔하지만, 실제 상관된 변수에서는 학습 범위 밖의 비현실적 조합을 평가하게 된다. 반대로 상관을 반영하는 방식은 모형이 쓰지 않는 변수에 효과를 배분한다. 이 페이지는 두 방향의 한계와, 그 사이에 있는 ALE를 다룬다.

---

## 4.1 Extrapolation(외삽) 문제

product measure와 interventional 기대값은 변수 사이의 상관을 끊는다. 즉 \(P(X_1, X_2)\) 대신 \(P(X_1)\,P(X_2)\)로 평균한다. 실제 데이터에서는 \(P(X_1, X_2) \neq P(X_1)\,P(X_2)\)이므로, 학습 데이터에 **거의 또는 전혀 없는 조합**이 평균에 들어간다. 예를 들어 연소득은 최하위 구간인데 대출 잔액은 최상위 구간인 차주는 대출 심사 구조상 거의 존재하지 않지만, 독립 가정에서는 다른 조합과 같은 방식으로 평가된다.

이런 영역에서 모형의 예측은 **학습된 값이 아니다.** 트리 모형은 데이터가 없는 영역에도 가장 가까운 분할 경계를 연장한 leaf 값을 그대로 내놓는다. 그 값은 분할 순서와 경계 위치가 우연히 정한 것이며, 바뀌어도 학습 데이터에 대한 예측 성능에는 아무 영향이 없다. 그런데 PDP·interventional SHAP·product measure fANOVA는 이 값을 정상적인 예측과 똑같이 평균에 넣는다.

### 수치 예시 — 데이터가 없는 칸이 해석을 바꾼다

소득과 부채를 각각 3구간(저·중·고)으로 나눈 가상 예시다. 두 변수는 양의 상관이 있어서, "소득 저 + 부채 고"와 "소득 고 + 부채 저" 조합에는 데이터가 없다.

| 데이터 비중 | 부채 저 | 부채 중 | 부채 고 |
|---|---:|---:|---:|
| **소득 저** | 20% | 10% | **0%** |
| **소득 중** | 10% | 20% | 10% |
| **소득 고** | **0%** | 10% | 20% |

모형 출력(불량 로짓 기여, 클수록 위험)은 데이터가 있는 칸에서 다음과 같이 학습되었다. 부채가 많을수록 위험하고, 소득이 높을수록 안전하다. 데이터가 없는 두 칸의 값 \(c\), \(d\)는 트리 경계를 연장해 정해진 임의의 값이다.

| 모형 출력 | 부채 저 | 부채 중 | 부채 고 |
|---|---:|---:|---:|
| **소득 저** | 0.0 | 0.6 | \(c\) |
| **소득 중** | −0.4 | 0.0 | 0.6 |
| **소득 고** | \(d\) | −0.4 | 0.0 |

\(c, d\)를 세 가지로 바꿔가며 **부채의 효과 곡선**(부채 저·중·고에서의 값, 평균 0으로 중심화)을 세 방법으로 계산했다.

| 빈 칸의 값 \((c, d)\) | PDP (interventional) | M-plot (observational) | ALE |
|---|---|---|---|
| (1.2, −0.8) 규칙적으로 연장 | −0.48, −0.02, +0.52 | −0.17, +0.01, +0.16 | −0.46, −0.01, +0.46 |
| (0.0, 0.0) 평평하게 연장 | −0.21, +0.01, +0.19 | −0.17, +0.01, +0.16 | −0.32, −0.07, +0.40 |
| (3.0, −3.0) 극단적으로 연장 | −1.11, +0.01, +1.09 | −0.17, +0.01, +0.16 | −0.84, +0.16, +0.63 |
| **부채 저→고 변화폭** | 1.00 / 0.40 / 2.20 | 0.33 / 0.33 / 0.33 | 0.92 / 0.72 / 1.47 |

- **PDP**는 소득을 주변분포(저·중·고 각 30%·40%·30%)로 평균하므로, 부채 저·고에서 빈 칸을 30%의 가중으로 참조한다. 학습 데이터에 대한 예측은 하나도 바뀌지 않았는데, 부채 효과의 크기가 0.40에서 2.20까지 5배 넘게 달라진다.
- **M-plot**(부채 구간별 실제 고객의 평균 예측)은 빈 칸을 전혀 쓰지 않으므로 세 경우가 모두 같다. 그러나 변화폭이 0.33으로, 데이터가 있는 소득=중 행에서 모형이 보여주는 부채 효과(−0.4 → 0.6, 폭 1.0)보다 훨씬 작다. 부채가 많은 사람은 소득도 높은 경향이 있어서, 소득의 위험 감소 효과가 부채 효과를 상쇄한 채로 섞여 들어갔기 때문이다([1절](observational-vs-interventional.md)의 observational 문제).
- **ALE**는 그 사이에 있다. 아래 4.3에서 설명한다.

## 4.2 Hooker, Mentch & Zhou (2021)의 논지

Hooker, Mentch & Zhou (2021)는 논문 제목 그대로 "제한 없는 순열(unrestricted permutation)은 외삽을 강제한다"고 주장했다. 변수 하나의 값을 다른 관측치의 값과 무작위로 바꾸는 permutation importance는, 그 변수를 다른 변수와 독립으로 뽑는 것과 같다. 따라서 상관이 강한 데이터에서는 학습 범위 밖의 조합에서 모형을 평가하게 되고, 그 결과는 모형이 데이터 밖에서 어떻게 외삽하는지에 크게 좌우된다. 논문은 같은 문제를 PDP·ICE와 functional ANOVA에서도 짚는다.

대안으로 논문이 정리한 방향은 다음과 같다.

- **조건부 순열(conditional permutation)**: 다른 변수의 값에 조건부로 순열해 데이터 분포를 유지한다.
- **재학습 기반 중요도**: 변수를 빼고 모형을 다시 학습하거나(drop-and-relearn), 순열·조건부 순열한 데이터로 다시 학습한 뒤(permute-and-relearn, condition-and-relearn) 성능 차이를 본다. 논문은 이 가운데 condition-and-relearn을 권한다. 부제("variable importance requires at least one more model")가 말하듯, 외삽을 피하려면 **모형을 최소 하나 더** 만들어야 한다는 것이다.
- **효과 곡선**: ICE를 데이터 밀도가 높은 영역으로 제한하거나, 결합분포를 쓰는 일반화 fANOVA(Hooker, 2007)를 쓴다.

## 4.3 ALE — 국소 차분으로 외삽을 줄이는 방법

Apley & Zhu (2020)의 **ALE(Accumulated Local Effects)**는 PDP의 외삽 문제와 M-plot의 혼입 문제를 함께 줄이려는 방법이다. 아이디어는 두 가지다.

1. **국소 구간의 데이터만 쓴다.** 변수 \(x_j\)의 구간 \(k\)에서는 실제로 그 구간에 속한 관측치만 사용한다. 멀리 떨어진 조합은 평가하지 않는다.
2. **수준이 아니라 차분을 평균한다.** 그 관측치들에 대해 "\(x_j\)를 구간의 아래쪽 경계에서 위쪽 경계로 옮겼을 때 예측이 얼마나 변하는가"를 계산해 평균한다. 다른 변수는 그대로 두고 \(x_j\)만 조금 옮기므로, 상관된 변수의 효과가 섞이지 않는다.

이렇게 얻은 구간별 평균 차분을 아래 구간부터 **누적**한 뒤 전체 평균이 0이 되도록 중심화한 것이 ALE 곡선이다.

$$
\text{ALE}_j(x) = \sum_{k=1}^{k(x)} \frac{1}{n_k} \sum_{i:\, x_{ij} \in N_k} \Big[f(z_k, \mathbf{x}_{i,-j}) - f(z_{k-1}, \mathbf{x}_{i,-j})\Big] - \text{상수}
\tag{1}
$$

여기서 \(N_k = (z_{k-1}, z_k]\)는 \(x_j\)의 \(k\)번째 구간, \(n_k\)는 그 구간의 관측치 수다.

위 3×3 예시에서는 "부채 저→중" 차분을 부채=중인 고객의 소득 분포로, "부채 중→고" 차분을 부채=고인 고객의 소득 분포로 평균했다. 결과(4.1의 표)를 보면 ALE는 PDP보다 빈 칸에 훨씬 덜 민감하고(변화폭 0.72~1.47 대 0.40~2.20), M-plot처럼 소득 효과에 묻히지도 않는다(0.33보다 훨씬 크고, 소득=중 행의 부채 효과 1.0에 가깝다).

다만 이 예시처럼 구간이 3개뿐인 **거친 격자**에서는, 부채=중인 고객 중 소득=고인 사람(25%)에 대해 "부채=저"로 옮긴 값 \(d\)를 참조하게 되어 한 칸짜리 외삽이 남는다. ALE는 본래 연속 변수를 잘게 나눈 구간에서 인접 구간 사이의 작은 이동만 평가하도록 설계된 방법이므로, 구간이 촘촘할수록 이런 외삽이 줄어든다. 또 ALE는 변수 하나(또는 둘)의 **global 효과 곡선**이며, 개별 고객 설명을 대신하지는 않는다.

## 4.4 Dummy 공리 위반 — observational 쪽의 문제

외삽을 피하려고 상관을 반영하면 반대쪽 문제가 생긴다. Shapley 값의 **Dummy 공리**는 "예측에 전혀 영향을 주지 않는 변수의 기여는 0이어야 한다"는 것이다. Sundararajan & Najmi (2020)는 조건부 기대값을 가치함수로 쓰는 observational SHAP(논문의 Conditional Expectation Shapley)이 이 공리를 위반한다는 것을 예시로 보였다. 모형이 쓰지 않는 변수라도 쓰이는 변수와 상관되어 있으면, 그 변수를 "알게 되는 것"만으로 다른 변수에 대한 기대가 바뀌므로 기여가 생기는 것이다. 반면 interventional(논문의 Baseline·Random Baseline 계열) Shapley는 Dummy를 만족한다.

[3절](shap-two-modes.md)의 예시가 정확히 이 경우다. 현금서비스만 쓰는 모형 \(g = 2.0\,x_1\)에서, 카드론(\(X_2\))의 observational SHAP은 0.30이고 interventional SHAP은 0이었다. 신용평가에서 이것은 실질적인 문제가 된다. 모형에 들어가지도 않은 변수가 거절 사유 상위에 올라오면, 설명 자체의 신뢰가 무너진다.

## 4.5 양쪽 모두 완벽하지 않다

| 문제 | Observational (조건부) | Interventional (독립) | ALE |
|---|---|---|---|
| 데이터 없는 조합 평가 | 없음 | **있음** (외삽) | 국소적으로만 (구간이 촘촘하면 작음) |
| 모형이 쓰지 않는 변수에 기여 | **있음** (Dummy 위반) | 없음 | 없음 (그 변수의 차분은 0) |
| 상관된 변수 효과의 분리 | 불가 (뒤섞임) | 가능 | 가능 |
| 대응하는 fANOVA와 직교성 | joint measure(Hooker 2007): 부분집합 관계인 성분끼리만 직교 | product measure: 모든 성분이 측도 위에서 직교 | — |
| 개별 고객 설명 | 가능 (SHAP) | 가능 (SHAP) | 불가 (global 곡선) |

어느 방법도 모든 문제를 해결하지 못한다. 상관이 강한 데이터에서 "변수 하나의 효과"라는 개념 자체가 하나로 정해지지 않기 때문이다. 따라서 **해석의 목적에 따라 방법을 고르고, 그 선택을 명시**하는 것이 현실적인 대응이다. 구체적인 절차는 [5절](practical-guide.md)에서 정리한다.

---

<div class="source-ref" markdown>
**참고**

- Hooker, Mentch, Zhou (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing* 31, 82
- Apley, Zhu (2020). [Visualizing the effects of predictor variables in black box supervised learning models](https://doi.org/10.1111/rssb.12377). *JRSS-B* 82(4):1059–1086
- Sundararajan, Najmi (2020). [The Many Shapley Values for Model Explanation](https://proceedings.mlr.press/v119/sundararajan20b.html). ICML, PMLR 119
- Janzing, Minorics, Blöbaum (2020). [Feature relevance quantification in explainable AI: A causal problem](https://proceedings.mlr.press/v108/janzing20a.html). AISTATS, PMLR 108
- Hooker (2007). [Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables](https://doi.org/10.1198/106186007X237892). *JCGS* 16(3)
</div>
