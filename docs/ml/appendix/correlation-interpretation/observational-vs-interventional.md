---
title: "두 가지 기대값"
---

# 1. 두 가지 기대값 — Observational vs Interventional

> 같은 질문 "변수 \(x_j\)가 \(a\)일 때 예측값은?"에 대해, 두 가지 서로 다른 답이 존재한다.

SHAP, PDP(Partial Dependence Plot), fANOVA는 모두 같은 계산을 반복한다. **일부 변수 \(S\)만 특정 값으로 고정하고, 나머지 변수 \(-S\)에 대해서는 평균을 낸다.** 문제는 "나머지 변수를 어떤 분포로 평균하느냐"다. 변수가 서로 독립이면 고민할 것이 없지만, 신용평가 변수처럼 서로 강하게 상관되어 있으면 두 가지 선택지가 갈라지고, 그 선택에 따라 해석 결과가 달라진다. 이 페이지는 이후 부록 B 전체의 출발점이 되는 두 가지 기대값을 정의하고, 하나의 작은 예시로 차이를 확인한다.

---

## 1.1 Observational (관측적) 조건부 기대값

첫 번째 답은 "실제로 \(X_S = \mathbf{x}_S\)인 사람들의 평균 예측"이다.

$$
v_{\text{obs}}(S) = \mathbb{E}\big[f(\mathbf{X}) \mid \mathbf{X}_S = \mathbf{x}_S\big] = \int f(\mathbf{x}_S, \mathbf{x}_{-S})\, d\mathbb{P}(\mathbf{x}_{-S} \mid \mathbf{x}_S)
\tag{1}
$$

나머지 변수를 **조건부 분포** \(\mathbb{P}(\mathbf{x}_{-S} \mid \mathbf{x}_S)\)로 평균한다. 변수 간 상관이 그대로 반영된다 — 현금서비스를 이용한 사람은 카드론도 이용했을 가능성이 높다는 사실이 평균 안에 들어간다.

- **장점**: 실제 데이터에 존재하는 조합 위에서만 평가하므로, 비현실적인 변수 조합을 만들지 않는다. "데이터에 충실(true to the data)"하다.
- **단점**: \(x_j\) 자체의 효과와, \(x_j\)와 상관된 다른 변수의 효과가 **뒤섞인다.** 극단적으로는 모형이 전혀 쓰지 않는 변수도 효과가 있는 것처럼 보인다(아래 예시).

## 1.2 Interventional (개입적) 기대값

두 번째 답은 "\(X_S\)만 \(\mathbf{x}_S\)로 바꿔 넣고, 나머지는 원래 분포 그대로 두었을 때의 평균 예측"이다.

$$
v_{\text{int}}(S) = \mathbb{E}_{\mathbf{X}_{-S}}\big[f(\mathbf{x}_S, \mathbf{X}_{-S})\big] = \int f(\mathbf{x}_S, \mathbf{x}_{-S})\, d\mathbb{P}(\mathbf{x}_{-S})
\tag{2}
$$

나머지 변수를 **주변 분포** \(\mathbb{P}(\mathbf{x}_{-S})\)로 평균한다. Janzing, Minorics & Blöbaum (2020)은 이것을 인과추론의 do 연산자로 \(\mathbb{E}[f(\mathbf{X}) \mid do(\mathbf{X}_S = \mathbf{x}_S)]\)라고 쓰고, 모형의 입력을 바꿔 넣는 "개입"으로 해석해야 한다고 주장했다. 여기서 개입의 대상은 **모형의 입력**이지 현실의 고객이 아니다. 즉, 식 (2)는 "현금서비스를 끊으면 이 고객의 실제 부도위험이 얼마나 줄어드는가"(현실의 인과효과)가 아니라, "모형에 현금서비스 이용 여부만 바꿔 넣으면 점수가 얼마나 달라지는가"(모형의 반응)에 답한다.

식 (2)가 끊는 것은 \(\mathbf{X}_S\)와 \(\mathbf{X}_{-S}\) **사이의** 상관뿐이다. \(\mathbf{X}_{-S}\) 내부의 상관은 결합 주변분포에 그대로 남는다. 변수를 하나씩 따로 고정하는 PDP나 product measure fANOVA처럼 모든 변수를 서로 독립으로 다루는 것은 이보다 한 단계 더 강한 가정이다.

- **장점**: 모형이 \(x_j\)에 어떻게 반응하는지를 다른 변수의 효과와 분리해서 보여준다. "모형에 충실(true to the model)"하다. 모형이 쓰지 않는 변수에는 효과가 생기지 않는다.
- **단점**: 실제로는 거의 존재하지 않는 변수 조합(예: 연소득은 매우 낮은데 대출 잔액은 매우 큰 고객)에서도 모형을 평가한다. 그런 영역의 예측은 학습 데이터가 없어 신뢰하기 어렵다([4절](independence-extrapolation.md)).

---

## 1.3 수치 예시 — 부록 B의 공통 예시

부록 B 전체에서 쓰는 작은 가상 예시를 정의한다. 두 이진 변수와 하나의 depth-2 트리 모형이다.

- \(X_1\): 최근 현금서비스 이용 여부 (1 = 있음)
- \(X_2\): 최근 카드론 이용 여부 (1 = 있음)

두 변수의 결합 분포와 모형 출력(불량 로짓에 더해지는 값, 클수록 위험)은 다음과 같다.

| | \(X_2 = 0\) | \(X_2 = 1\) | \(X_1\) 주변 |
|---|---|---|---|
| **\(X_1 = 0\)** | 40% / \(f = 0.0\) | 10% / \(f = 0.5\) | 50% |
| **\(X_1 = 1\)** | 10% / \(f = 1.0\) | 40% / \(f = 2.5\) | 50% |
| **\(X_2\) 주변** | 50% | 50% | |

모형은 \(f(x_1, x_2) = 1.0\,x_1 + 0.5\,x_2 + 1.0\,x_1 x_2\)와 같다. 현금서비스 이용이 카드론 이용보다 위험 신호가 강하고, 둘 다 이용하면 위험이 추가로 커진다. 두 변수는 같은 값끼리 몰려 있어서 상관계수가 \(\rho = (0.4 - 0.5 \times 0.5)/(0.5 \times 0.5) = 0.6\)이다. 전체 평균 예측은 \(\mathbb{E}[f] = 0.1 \times 0.5 + 0.1 \times 1.0 + 0.4 \times 2.5 = 1.15\)다.

### 한 변수만 고정했을 때의 평균 예측

\(X_1 = 1\)로 고정한 경우를 계산해보자.

- **Observational**: 실제로 \(X_1 = 1\)인 사람(50%) 중 80%가 \(X_2 = 1\)이다.
  \(\mathbb{E}[f \mid X_1 = 1] = (0.1 \times 1.0 + 0.4 \times 2.5)/0.5 = 2.20\)
- **Interventional**: \(X_2\)는 주변 분포(50:50) 그대로 둔다.
  \(\mathbb{E}_{X_2}[f(1, X_2)] = 0.5 \times 1.0 + 0.5 \times 2.5 = 1.75\)

같은 방식으로 네 가지 값을 모두 구하면:

| 고정한 값 | Observational | Interventional | 비고 |
|---|---:|---:|---|
| \(X_1 = 0\) | 0.10 | 0.25 | |
| \(X_1 = 1\) | 2.20 | 1.75 | |
| **\(X_1\)의 효과 (1 − 0)** | **2.10** | **1.50** | |
| \(X_2 = 0\) | 0.20 | 0.50 | |
| \(X_2 = 1\) | 2.10 | 1.50 | |
| **\(X_2\)의 효과 (1 − 0)** | **1.90** | **1.00** | \(0.5 + 1.0 \times P(X_1 = 1) = 1.00\) |

카드론(\(X_2\))의 효과를 보자. observational로는 1.90으로, interventional(1.00)의 거의 두 배다. 카드론을 이용한 사람은 대부분 현금서비스도 이용했기 때문에, 현금서비스의 위험이 카드론의 효과처럼 함께 잡힌 것이다. observational 값이 틀린 것은 아니다 — "카드론 이용자는 평균적으로 이 정도 점수를 받는다"는 **데이터에 대한** 사실이다. 다만 "모형이 카드론에 얼마나 반응하는가"에 대한 답은 아니다.

### 모형이 쓰지 않는 변수

상관의 효과를 더 극적으로 보려면, 현금서비스만 쓰는 모형 \(g(x_1, x_2) = 2.0\,x_1\)을 생각하면 된다. 이 모형은 카드론 여부를 전혀 보지 않는다.

| 고정한 값 | Observational | Interventional |
|---|---:|---:|
| \(X_2 = 0\) | \(2.0 \times P(X_1 = 1 \mid X_2 = 0) = 2.0 \times 0.2 = 0.40\) | 1.00 |
| \(X_2 = 1\) | \(2.0 \times P(X_1 = 1 \mid X_2 = 1) = 2.0 \times 0.8 = 1.60\) | 1.00 |

observational로 보면 카드론 이용 여부에 따라 평균 예측이 0.40에서 1.60으로 크게 움직인다. 모형이 카드론을 전혀 쓰지 않는데도 말이다. interventional로 보면 카드론 여부와 무관하게 1.00으로 평평하다. 이 차이가 [4절](independence-extrapolation.md)에서 다룰 Dummy 공리 문제의 핵심이다.

---

## 1.4 "True to the Model" vs "True to the Data"

Chen, Janizek, Lundberg & Lee (2020)는 이 두 기대값의 선택을 하나의 질문으로 정리했다. 논문 초록의 표현대로, 결국 선택은 설명이 **모형에 충실(true to the model)**하기를 원하는가, **데이터에 충실(true to the data)**하기를 원하는가에 달려 있다.

- **모형에 충실** (interventional): "이 모형은 어떤 변수에 어떻게 반응하는가?" — 모형 검증, 변수의 효과 방향 점검, 모형이 실제로 쓰는 변수의 확인에 적합하다.
- **데이터에 충실** (observational): "이 값을 가진 고객군은 실제로 어떤 점수를 받는가?" — 데이터와 고객군을 설명하는 데 적합하다.

SHAP 개발자인 Lundberg는 공개 토론에서 이 상황을 "변수들이 임의로 상관되어 있으면 모형과 데이터 모두에 충실하기는 불가능하다"는 취지로 정리한 바 있다(아래 출처). 상관이 존재하는 한 하나의 숫자로 두 질문에 동시에 답할 수는 없으므로, **어떤 질문에 답하려는지 먼저 정하는 것**이 출발점이다. 신용평가 실무에서의 선택 기준은 [5절](practical-guide.md)에서 정리한다.

## 정리

| | Observational (조건부) | Interventional (개입적) |
|---|---|---|
| 나머지 변수의 분포 | 조건부 분포 \(\mathbb{P}(\mathbf{x}_{-S} \mid \mathbf{x}_S)\) | 주변 분포 \(\mathbb{P}(\mathbf{x}_{-S})\) |
| 상관 구조 | 그대로 반영 | \(S\)와 \(-S\) 사이의 상관을 끊음 |
| 평가 영역 | 데이터가 있는 조합만 | 데이터에 없는 조합도 포함 (외삽) |
| 모형이 쓰지 않는 변수 | 효과가 생길 수 있음 | 효과 없음 |
| 답하는 질문 | 데이터에 충실 | 모형에 충실 |

예시의 수치는 모두 1.3의 분포표에서 손으로 다시 계산할 수 있다. 다음 페이지에서는 같은 선택이 fANOVA에서는 "측도(measure)"의 선택으로 나타난다는 것을 본다. → [2절. fANOVA의 두 Measure](joint-vs-product.md)

---

<div class="source-ref" markdown>
**참고**

- Chen, Janizek, Lundberg, Lee (2020). [True to the Model or True to the Data?](https://arxiv.org/abs/2006.16234) arXiv:2006.16234
- Janzing, Minorics, Blöbaum (2020). [Feature relevance quantification in explainable AI: A causal problem](https://proceedings.mlr.press/v108/janzing20a.html). AISTATS, PMLR 108
- Lundberg의 공개 코멘트 (2019): [interpretable-ml-book issue #142](https://github.com/christophM/interpretable-ml-book/issues/142#issuecomment-565145918) — "it is impossible to be both true to the model and true to the data when features are arbitrarily correlated"
</div>
