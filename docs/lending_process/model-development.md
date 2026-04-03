---
title: "① 모형 개발"
---

# ① 모형 개발 — 가이드북의 핵심 영역

본 가이드북의 전통 스코어카드(Part 1~5)와 머신러닝(Part 1~5)에서
모형 개발의 전 과정을 다루고 있다.

## 가이드북 매핑

| 단계 | 내용 | 가이드북 위치 |
|---|---|---|
| 데이터 준비 | 성능기간 설정, Good/Bad 정의, 관측/표본기간 | [스코어카드 Part 1](../scorecard/part1_overview/) |
| 변수 처리 | Classing, WoE/IV 변환, 단변량 LR | [스코어카드 Part 3](../scorecard/part3_variable_selection/) |
| 모형 학습 | Logistic Regression, GBM/RF 등 | [스코어카드 Part 4](../scorecard/part4_modeling/), [ML Part 1](../ml/part1_overview/)~[3](../ml/part3_tree_ensemble/) |
| 해석·설명 | SHAP, Surrogate Model, EBM | [ML Part 4](../ml/part4_evaluation/) |
| 성능 평가 | KS, AR, PSI | [스코어카드 Part 5](../scorecard/part5_scorecard/), [ML Part 5](../ml/part5_validation/) |
| 산출물 변환 | p̂ → Score(PDO 변환) → 등급 매핑 | [스코어카드 Part 5](../scorecard/part5_scorecard/) |

## 이 단계의 핵심

모형의 목적은 **서열화(ordering)** — 누가 더 위험한지를 올바르게 줄 세우는 것이다.
p̂의 절대값 정확도보다 **변별력(KS, AR)**이 핵심 지표이며,
이 산출물(Score, 등급)이 이후 단계의 입력값이 된다.
