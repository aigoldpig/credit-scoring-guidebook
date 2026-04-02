# Surrogate Model (대리 모형)

규제 기관에 트리 앙상블을 직접 제출하기 어려운 경우, **해석 가능한 대리 모형(Surrogate Model)**을 별도로 구축하여 보고하는 전략이 사용된다. ML의 성능은 유지하면서, 설명은 전통적 스코어카드 형태로 제공하는 것이다.

---

## 3.1 파이프라인

대리 모형의 구축 과정은 다음과 같다.

> **ML 모형 학습** → **\(\hat{p}\) 산출** → **스코어 변환** → **스코어를 종속변수로 LR 적합** → **스코어카드 생성**

구체적으로:

1. **ML 모형 학습**: XGBoost 등으로 부도 확률 \(\hat{p}\)를 예측한다
2. **스코어 변환**: \(\hat{p}\)를 PDO·Anchor Score 기반으로 신용 점수로 변환한다 ([Part 5 참조](../../scorecard/part5_scorecard/scorecard-and-rating.md))
3. **타겟 교체**: 원래의 이진 타겟(Good/Bad)을 **ML이 산출한 스코어**로 대체한다
4. **해석 모형 적합**: 동일한 입력 변수를 구간화(binning)하고, 이 스코어를 종속변수로 **일반 회귀(OLS) 또는 로지스틱 회귀**를 적합한다
5. **스코어카드 제시**: 회귀 계수를 스코어카드 포인트로 변환하여, 변수별 기여도를 사람이 읽을 수 있는 형태로 제공한다

!!! tip "핵심 아이디어"
    실제 심사는 ML 스코어로 하되, "이 고객은 왜 이 점수인가"에 대한 설명은 스코어카드 형식으로 제공한다. SHAP이 기술자용 해석 도구라면, surrogate 스코어카드는 **현업·규제기관용 설명 도구**다.

---

## 3.2 R²가 낮을 수밖에 없는 이유

대리 모형의 \(R^2\)는 일반적으로 높지 않다. 이는 결함이 아니라 **구조적 한계**다.

XGBoost가 잡아내는 것들:

- 변수 간 **비선형 교호작용** (예: 소득이 높을 때만 DTI가 중요)
- **고차 분기 효과** (3~4개 변수가 동시에 조건을 만족할 때의 패턴)
- **데이터 내 미세한 비선형성**

이 모든 것을 **가법 구조(additive)**인 선형 회귀로 재현하는 것은 원리적으로 불가능하다. 복잡한 결정 경계를 단순한 1차 함수로 근사하는 것이므로, 설명되지 않는 분산이 반드시 남는다.

$$
\underbrace{\text{Score}_{\text{ML}}}_{\text{비선형 + 교호작용}} \approx \underbrace{\beta_0 + \sum_j \beta_j \cdot \text{WoE}_j}_{\text{가법 선형 구조}} + \underbrace{\epsilon}_{\text{설명 불가 잔차}}
$$

---

## 3.3 업계 구현 사례

이 파이프라인은 이미 **상용 소프트웨어 레벨**로 구현되어 있다.

**SAS Viya Scorecard Node**에는 "Use a black-box model" 옵션이 내장되어 있으며, 위에서 설명한 파이프라인을 자동으로 수행한다. Black-box 모형(GBM, RF, NN 등)을 학습한 뒤 원래 타겟을 예측값으로 교체하고, stepwise 로지스틱 회귀를 적합하여 전통적 스코어카드를 생성한다.

학술적으로는 Dumitrescu et al. (2022, EJOR)의 **PLTR (Penalised Logistic Tree Regression)**이 유사한 철학의 모형이다. 짧은 깊이의 결정 트리에서 규칙을 추출한 뒤, 이를 이진 변수로 만들어 페널티 로지스틱 회귀(Adaptive LASSO)에 넣는 하이브리드 접근이다.

---

## 3.4 찬반 논쟁

Surrogate 접근은 업계에서 활발히 논쟁 중인 주제다.

=== "Pro: 대리 모형 옹호"

    - 기존 **규제 프레임워크에 부합**하는 익숙한 포맷 (SR 11-7, EBA 가이드라인)
    - 리스크 위원회, 감사팀이 **읽을 수 있는** 스코어카드 산출물
    - SAS 등 상용 소프트웨어에 이미 구현 — **실무 적용 검증 완료**

=== "Con: 대리 모형 비판"

    Sudjianto & Zhang (2021, Wells Fargo)는 고위험 의사결정에는 **내재적으로 해석 가능한 모형(Inherently Interpretable ML)**을 사용해야 한다고 주장한다. 대리 모형에는 본질적 딜레마가 존재한다:

    - 대리 모형이 원래 모형을 **잘 근사하면** → "그냥 그 해석 모형을 쓰면 되지?"
    - **못 근사하면** → 설명이 실제 모형의 행동과 괴리 — **misleading**

=== "제3의 길: SHAP 직접 활용"

    SHAP 값 자체를 스코어카드 포맷으로 변환하는 접근도 있다. Denis Burakov의 **xbooster** 라이브러리는 XGBoost의 리프 노드를 스코어카드 포인트로 직접 변환한다. 별도의 surrogate 단계 없이 ML 모형 자체에서 해석 가능한 산출물을 추출하는 방식이다.

---

## 3.5 참고 자료

| 자료 | 유형 | 내용 |
|------|------|------|
| **[Dumitrescu et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377221721005695)** | 논문 (EJOR) | "Machine Learning for Credit Scoring" — PLTR, 트리 규칙 + 페널티 LR 하이브리드 |
| **[Sudjianto & Zhang (2021)](https://www.semanticscholar.org/paper/Designing-Inherently-Interpretable-Machine-Learning-Sudjianto-Zhang/90409ae91767248e9ea88b7d6ab44e18f0e1a9be)** | 논문 | "Designing Inherently Interpretable ML Models" — Wells Fargo, 내재적 해석 가능 ML 옹호 |
| **[xbooster](https://github.com/deburky/boosting-scorecards)** | Python 라이브러리 | XGBoost 리프 노드 → 스코어카드 포인트 직접 변환 |

!!! tip "다음 페이지"
    [해석 가능한 ML: 1-Depth GBM과 EBM](depth1_gbm.md) --- GAM 구조의 해석 가능한 ML 모형과 성능 비교를 다룬다.
