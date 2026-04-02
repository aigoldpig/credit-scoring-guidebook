---
title: 이론 기반
---

# 이론 기반

> 신용평가모형의 통계적 기초: 이진분류, Logit 변환, 최대우도추정(MLE)

신용평가모형(CSS)은 차주의 불량 여부를 예측하는 **이진 분류** 문제다. 이를 풀기 위해 로지스틱 회귀를 사용하며, 그 이론적 기초는 세 단계로 구성된다.

1. **문제 정의** — 확률 \(p \in [0,1]\)를 직접 선형회귀로 모형화할 수 없는 이유를 확인한다.
2. **Logit 변환** — 확률을 Odds → Log-Odds로 변환하여 실수 전체 범위 \((-\infty, +\infty)\)로 매핑한다.
3. **MLE** — 변환된 공간에서 관측 데이터의 우도를 최대화하는 계수 \(\boldsymbol{\beta}\)를 추정한다.

> **확률** \(\xrightarrow{\text{Odds}}\) **\((0,\infty)\)** \(\xrightarrow{\ln}\) **\((-\infty,\infty)\)** \(\xrightarrow{\text{선형 모형}}\) **\(\beta_0 + \boldsymbol{\beta}^\top\mathbf{x}\)**

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [문제 정의](binary-classification.md) | 왜 선형회귀를 직접 쓸 수 없는가? |
| 2 | [Logit 변환](logit-transform.md) | Odds · Log-Odds · Sigmoid는 어떻게 연결되는가? |
| 3 | [MLE](mle.md) | 로지스틱 회귀의 계수는 어떻게 추정하는가? |

!!! tip "다음 섹션"
    이론을 마쳤다면, [변수 선정](../part3_variable_selection/classing/index.md)에서 연속형 변수를 구간화(Classing)하고 WoE/IV로 변별력을 측정하는 실무 절차를 학습한다.
