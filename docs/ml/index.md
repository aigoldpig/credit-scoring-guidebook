---
title: 머신러닝
---

# 머신러닝 기반 신용평가

> 전통적인 로지스틱 회귀 스코어카드를 넘어, **트리 기반 머신러닝**을 신용평가모형에 적용하는 방법론을 다룬다.

!!! quote "저자의 말"
    Bias-Variance Tradeoff는 통계학이나 머신러닝을 공부하면 가장 처음 접하게 되는 개념 중 하나다. 그런데 실무를 거듭할수록, 이 개념이 단순한 입문 지식이 아니라 **모든 모형 설계를 관통하는 근본 원리**라는 확신이 강해진다.

    모형의 복잡도를 높이면 Overfitting 위험이 커지고, 낮추면 Underfitting에 빠진다. 이 원리를 모르고 알고리즘을 돌리면, 하이퍼파라미터가 왜 거기 있는지, 어떤 방향으로 조정해야 하는지 감이 오지 않는다. 결국 무작위 시행착오에 의존하게 된다.

    그래서 머신러닝 모형 적합에는 각 알고리즘의 이론적 배경과 설계 사상에 대한 이해가 필수적이다. 저자가 데이터 분석 조직에서 신입 직원을 채용할 때 가장 중요하게 보는 것도 통계학적 기초 지식이다. 라이브러리 사용법은 실무에서 금방 익히지만, 기초가 없으면 모형이 왜 이렇게 동작하는지를 설명할 수 없다.

---

## 이 섹션의 구성

| 순서 | 주제 | 핵심 내용 |
|:---:|------|----------|
| 1 | [개요](part1_overview/index.md) | 왜 ML인가, Bias-Variance Tradeoff, 정규화, 데이터 분리, 피처 엔지니어링 |
| 2 | [뉴럴넷](part2_neural_net/index.md) | 신경망 기초, LR = 단일 뉴런, TabNet, CNN·RNN의 한계 |
| 3 | [트리 앙상블](part3_tree_ensemble/index.md) | CART, RF, Boosting, XGBoost/LightGBM, 하이퍼파라미터 튜닝 |
| 4 | [해석과 설명](part4_interpretation/index.md) | XAI 개론, SHAP 이론, 1-Depth GBM, EBM(GA²M), fANOVA·Purification, 도구·구현, 저자의 해석 고민 기록 |
| 5 | [모델 검증](part5_validation/index.md) | 성능 지표(AUC, KS, Gini), OOT 검증, PSI 모니터링, 규제 프레임워크 |
| 부록 | [보충 자료](appendix/index.md) | A. SHAP과 fANOVA 심화, B. 변수 상관과 모형 해석, C. 해석 가능한 ML 실험 설계 |

---

## 성능과 해석, 두 마리 토끼

전통 스코어카드는 로지스틱 회귀의 계수가 곧 설명이다. WoE로 변환된 변수에 계수를 곱하면 점수가 나오고, 그 점수표가 곧 모형의 해석이 된다. **모형 자체가 해석 가능**한 구조다.

ML은 다르다. 수백 개의 트리가 투표하고, 수천 개의 리프 노드가 점수를 합산한다. 성능은 올라가지만, "왜 이 고객이 이 점수인가"에 대한 답이 모형 구조에서 바로 보이지 않는다. 그래서 ML 모형에는 **사후 해석(post-hoc explanation)** 도구가 필수다.

신용평가에서 해석 가능성은 선택이 아니라 **규제 요건**이다. 금융 당국은 모형이 왜 특정 고객을 거절했는지, 어떤 변수가 얼마나 기여했는지 설명할 수 있어야 한다고 요구한다. 성능이 아무리 좋아도 설명할 수 없으면 실전에 투입할 수 없다.

이 섹션에서는 SHAP을 중심으로 한 해석 기법, 1-Depth GBM/EBM처럼 **해석 가능성과 성능을 동시에 추구하는 모형**, 그리고 fANOVA와 Purification을 통한 **효과 분리**까지 다룬다. 모형을 만드는 것과 **쓸 수 있게 만드는 것**은 다른 문제이며, 후자가 더 어렵다.

---

## 전통 스코어카드와의 관계

이 섹션은 앞선 Part 1~5([개요](../scorecard/part1_overview/index.md) ~ [스코어카드](../scorecard/part5_scorecard/index.md))의 **연장선**이다. 전통 스코어카드를 대체하는 것이 아니라, 그 위에 쌓는 것이다.

```
전통 스코어카드 (Part 1~5)          머신러닝 (이 섹션)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
로지스틱 회귀                        트리 앙상블 (RF, GBM, XGB, LGBM)
WoE + IV로 변수 선정                 Feature Importance + SHAP
수작업 Classing                      트리 split이 자동 수행
점수표 = 해석                        SHAP = 사후 해석
규제모형 표준                        챌린저 모형 / 하이브리드
```

전통 스코어카드의 개념 — Odds, Good/Bad 정의, 성과 기간, 모집단 설계 — 은 ML에서도 **그대로** 적용된다. 달라지는 것은 변수 처리 방식과 모형 구조일 뿐, 신용평가의 근본 프레임은 동일하다.

---

## 추천 학습 자료

### 서적

| 자료 | 설명 |
|------|------|
| **Hastie, Tibshirani, Friedman — [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)** | Bias-Variance 분해, Boosting, Additive Model 등의 이론적 기반. 석사 과정에서 머신러닝 텍스트북으로 사용했으며, 이 가이드북의 세부 이론도 많은 부분을 여기서 참고했다. (무료 PDF 공개) |
| **Christoph Molnar — [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)** | ML 해석 가능성의 바이블. PDP, SHAP, LIME 등을 체계적으로 다룬다. CC 라이선스로 무료 공개. 저자 본인도 이 책으로 많은 공부를 했으며, ML 해석 가능성에 관한 가장 접근하기 쉬운 자료라고 생각한다. |

### 온라인 강의

| 자료 | 설명 |
|------|------|
| **Coursera — Andrew Ng의 [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)** | 머신러닝 입문의 정석. 2022년 리뉴얼 버전에는 트리 앙상블(RF, XGBoost), 추천시스템, 강화학습까지 포함. |
| **[모두를 위한 머신러닝/딥러닝 강의](https://hunkim.github.io/ml/)** | 한국어 ML/DL 입문. 이론 설명이 직관적이어서 첫 진입에 좋다. |

### 심화

| 자료 | 분야 |
|------|------|
| **Stanford CS231n** | 컴퓨터 비전, CNN |
| **Stanford CS224n** | 자연어처리, Transformer |
| **David Silver — DeepMind RL Course** | 강화학습 입문 |

!!! tip "다음 섹션"
    [개요](part1_overview/index.md)에서, 전통 스코어카드의 한계와 ML이 가져온 변화를 먼저 정리한다.
