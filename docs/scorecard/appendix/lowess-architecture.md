# 3단계 모델 아키텍처: Transform → Model → Map

미니모델링은 RiskCalc 전체 아키텍처의 **첫 번째 단계**에 해당한다. 전체 구조는 변환(Transformation), 모델링(Modeling), 매핑(Mapping)의 3단계로 구성된다.

---

## 4.1 전체 흐름

![3단계 모델 아키텍처](images/architecture_3stage.png)

> **STEP 1 · Transform (미니모델링)** — 각 재무비율 \(x_i\)를 단변량 부도확률 \(T(x_i)\)로 변환. LOWESS 등 비모수 기법 사용.
>
> → **STEP 2 · Model (프로빗 추정)** — 변환된 \(T(x_i)\)들을 프로빗(Probit) 또는 로짓 모형에 투입. 다변량 가중치 추정.
>
> → **STEP 3 · Map (최종 매핑)** — 프로빗 출력을 실제 부도확률(EDF)로 비모수적 매핑. 중심경향(central tendency) 조정.

---

## 4.2 Step 1: Transform (미니모델링)

[정의와 LOWESS 기초](lowess-intro.md) 및 [RiskCalc 프로세스와 사상](lowess-riskcalc.md)에서 상세히 다뤘다. 각 재무비율을 개별적으로 단변량 부도확률로 변환하는 단계다.

---

## 4.3 Step 2: 프로빗 모형

변환된 \(T(x_i)\)들을 프로빗 모형에 투입한다:

$$
y = \Phi\!\left(f(\mathbf{x}, \boldsymbol{\beta})\right) \tag{B.5}
$$

선형 부분:

$$
f(\mathbf{x}, \boldsymbol{\beta}) = \beta_0 + \beta_1 \cdot T(x_1) + \beta_2 \cdot T(x_2) + \cdots + \beta_k \cdot T(x_k) \tag{B.6}
$$

여기서:

- \(\Phi\) = 정규 누적분포함수 (프로빗 모형)
- \(T(x_i)\) = 미니모델링으로 변환된 각 재무비율
- \(\beta_i\) = 각 변환비율의 가중치 (양수이며 유의해야 포함)

이 구조는 **일반화 가법 모형(Generalized Additive Model)**과 밀접하게 관련되어 있으며, Moody's는 이를 비선형 문제를 포착하되 투명성을 유지하는 강건한 모형 형태로 평가한다.

!!! note "프로빗 vs 로짓"
    본 가이드북에서 다룬 소매 CSS는 **로짓(logit)** 모형을 사용한다. RiskCalc는 **프로빗(probit)** 모형을 채택한다. 두 모형의 차이는 링크 함수(\(\Phi^{-1}\) vs \(\text{logit}\))뿐이며, 실무적 성능 차이는 미미하다. 프로빗은 정규분포 가정과의 정합성, 로짓은 Odds Ratio 해석의 편의성이 장점이다.

---

## 4.4 Step 3: 최종 매핑

프로빗 모형의 출력은 참 부도확률을 **과대 또는 과소 추정**하는 경향이 있다. 이를 교정하기 위해 모형 출력과 실제 표본 부도확률 간의 관계를 **비모수적으로 추정**하는 최종 매핑 단계가 적용된다.

!!! info "동일 알고리즘의 재활용"
    Moody's에 따르면, 이 매핑은 미니모델링의 입력 변환과 **동일한 평활 알고리즘(LOWESS)**을 사용하여 수행된다. 즉 LOWESS가 입력 단계(Step 1)와 출력 단계(Step 3) 모두에서 활용되는 셈이다.

### 소매 CSS와의 구조 비교

| 단계 | RiskCalc (기업) | 소매 CSS (본 가이드북) |
|------|----------------|-------------------|
| **Step 1: 변환** | LOWESS → 단변량 EDF | WoE 변환 (구간화 + 로그 Odds) |
| **Step 2: 모형** | 프로빗 회귀 | 로지스틱 회귀 |
| **Step 3: 교정** | LOWESS 매핑 → 최종 EDF | 스코어 변환 → 등급 매핑 |
| **산출물** | EDF (Expected Default Frequency) | 스코어 → 등급 → PD |

두 접근법은 **"변환 → 결합 → 교정"이라는 동일한 3단계 구조**를 공유하며, 기법의 선택만 다르다.
